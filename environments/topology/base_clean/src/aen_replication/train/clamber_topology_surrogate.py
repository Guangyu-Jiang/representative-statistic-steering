"""Train surrogates that predict CLAMBER topology features from token clouds."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from aen_replication.train.clamber_subclass_classification import _build_clamber_token_cloud_features
from aen_replication.train.token_cloud_topology_classifier import _topology_target_columns
from aen_replication.utils.io_utils import ensure_dir, write_json, write_markdown, write_parquet
from aen_replication.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

_MERGE_KEYS = ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "layer"]
_META_COLUMNS = ["example_id", "pair_id", "dataset", "split", "subclass", "label_ambiguous", "layer", "token_count"]


@dataclass
class _TopologyRegressionMetrics:
    mae_mean: float
    rmse_mean: float
    r2_mean: float
    explained_variance_mean: float


class _CloudTargetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, target_columns: list[str]) -> None:
        self.meta = frame.loc[:, [column for column in _META_COLUMNS if column in frame.columns]].reset_index(drop=True)
        self.clouds = [np.asarray(value, dtype=np.float32) for value in frame["cloud"].tolist()]
        self.targets = frame.loc[:, target_columns].to_numpy(dtype=np.float32)
        self.token_counts = frame["token_count"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.clouds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "cloud": self.clouds[index],
            "target": self.targets[index],
            "token_count": float(self.token_counts[index]),
            "meta_index": int(index),
        }


def _collate_cloud_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_tokens = max(int(item["cloud"].shape[0]) for item in batch)
    input_dim = int(batch[0]["cloud"].shape[1])
    target_dim = int(batch[0]["target"].shape[0])
    clouds = torch.zeros((batch_size, max_tokens, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    targets = torch.zeros((batch_size, target_dim), dtype=torch.float32)
    token_counts = torch.zeros(batch_size, dtype=torch.float32)
    meta_indices = torch.zeros(batch_size, dtype=torch.long)
    for row_index, item in enumerate(batch):
        cloud = torch.from_numpy(item["cloud"])
        token_count = int(cloud.shape[0])
        clouds[row_index, :token_count] = cloud
        mask[row_index, :token_count] = True
        targets[row_index] = torch.from_numpy(item["target"])
        token_counts[row_index] = float(item["token_count"])
        meta_indices[row_index] = int(item["meta_index"])
    return {
        "clouds": clouds,
        "mask": mask,
        "targets": targets,
        "token_counts": token_counts,
        "meta_indices": meta_indices,
    }


class DeepSetTopologySurrogate(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        token_hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
        include_token_count_input: bool,
    ) -> None:
        super().__init__()
        self.include_token_count_input = include_token_count_input
        self.token_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, token_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(token_hidden_dim, token_hidden_dim),
            torch.nn.GELU(),
        )
        head_input_dim = token_hidden_dim * 3 + (1 if include_token_count_input else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_input_dim, head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden_dim, head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, clouds: torch.Tensor, mask: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
        encoded = self.token_encoder(clouds)
        mask_f = mask.unsqueeze(-1).float()
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        mean = (encoded * mask_f).sum(dim=1) / counts
        centered = (encoded - mean.unsqueeze(1)) * mask_f
        std = torch.sqrt((centered.pow(2).sum(dim=1) / counts).clamp(min=0.0) + 1e-6)
        encoded_masked = encoded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_values = encoded_masked.max(dim=1).values
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
        pooled = [mean, std, max_values]
        if self.include_token_count_input:
            pooled.append(torch.log1p(token_counts).unsqueeze(-1))
        return self.head(torch.cat(pooled, dim=1))


def _load_cached_cloud_df(path: Path) -> pd.DataFrame:
    payload = joblib.load(path)
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict) and "cloud_df" in payload:
        return pd.DataFrame(payload["cloud_df"])
    raise ValueError(f"Unsupported token-cloud forward cache payload: {path}")


def _surrogate_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _target_stats(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y_train.mean(axis=0, dtype=np.float64)
    std = y_train.std(axis=0, dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> _TopologyRegressionMetrics:
    error = y_pred - y_true
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    true_mean = y_true.mean(axis=0)
    sse = np.sum(np.square(error), axis=0)
    sst = np.sum(np.square(y_true - true_mean), axis=0)
    r2 = np.where(sst > 1e-8, 1.0 - (sse / sst), 0.0)
    var_resid = np.var(error, axis=0)
    var_true = np.var(y_true, axis=0)
    explained = np.where(var_true > 1e-8, 1.0 - (var_resid / var_true), 0.0)
    return _TopologyRegressionMetrics(
        mae_mean=float(np.mean(mae)),
        rmse_mean=float(np.mean(rmse)),
        r2_mean=float(np.mean(r2)),
        explained_variance_mean=float(np.mean(explained)),
    )


def _per_feature_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, columns: list[str], split: str, layer: int) -> pd.DataFrame:
    error = y_pred - y_true
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    true_mean = y_true.mean(axis=0)
    sse = np.sum(np.square(error), axis=0)
    sst = np.sum(np.square(y_true - true_mean), axis=0)
    r2 = np.where(sst > 1e-8, 1.0 - (sse / sst), 0.0)
    var_resid = np.var(error, axis=0)
    var_true = np.var(y_true, axis=0)
    explained = np.where(var_true > 1e-8, 1.0 - (var_resid / var_true), 0.0)
    return pd.DataFrame(
        {
            "layer": int(layer),
            "split": split,
            "feature": columns,
            "mae": mae.astype(float),
            "rmse": rmse.astype(float),
            "r2": r2.astype(float),
            "explained_variance": explained.astype(float),
        }
    ).sort_values(["split", "r2", "feature"], ascending=[True, False, True]).reset_index(drop=True)


def _stratified_train_val_mask(frame: pd.DataFrame, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if "subclass" in frame.columns and frame["subclass"].notna().all():
        stratify_labels = frame["subclass"].astype(str)
    else:
        stratify_labels = frame["label_ambiguous"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(frame), dtype=int)
    train_index, val_index = next(splitter.split(dummy, stratify_labels))
    return np.asarray(train_index, dtype=int), np.asarray(val_index, dtype=int)


def _build_surrogate_views(
    frame: pd.DataFrame,
    *,
    target_columns: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[_CloudTargetDataset, _CloudTargetDataset, _CloudTargetDataset]:
    train_base = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    test_base = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    if train_base.empty or test_base.empty:
        raise ValueError("Topology surrogate requires non-empty train and test splits.")
    train_index, val_index = _stratified_train_val_mask(train_base, val_fraction=val_fraction, seed=seed)
    train_df = train_base.iloc[train_index].reset_index(drop=True)
    val_df = train_base.iloc[val_index].reset_index(drop=True)
    return (
        _CloudTargetDataset(train_df, target_columns=target_columns),
        _CloudTargetDataset(val_df, target_columns=target_columns),
        _CloudTargetDataset(test_base, target_columns=target_columns),
    )


def _run_loader(
    model: DeepSetTopologySurrogate,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    grad_clip_norm: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta_indices: list[np.ndarray] = []
    for batch in loader:
        clouds = batch["clouds"].to(device)
        mask = batch["mask"].to(device)
        token_counts = batch["token_counts"].to(device)
        target_raw = batch["targets"].to(device)
        target_norm = (target_raw - target_mean) / target_std
        prediction_norm = model(clouds, mask, token_counts)
        loss = criterion(prediction_norm, target_norm)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        losses.append(float(loss.item()))
        prediction_raw = prediction_norm.detach() * target_std + target_mean
        preds.append(prediction_raw.cpu().numpy())
        targets.append(target_raw.detach().cpu().numpy())
        meta_indices.append(batch["meta_indices"].cpu().numpy())
    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(meta_indices, axis=0),
    )


def _train_single_layer_surrogate(
    frame: pd.DataFrame,
    *,
    layer: int,
    target_columns: list[str],
    surrogate_cfg: dict[str, Any],
    seed: int,
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    layer_df = frame.loc[frame["layer"].eq(layer)].reset_index(drop=True)
    if layer_df.empty:
        raise ValueError(f"No token-cloud rows available for surrogate layer {layer}.")
    train_set, val_set, test_set = _build_surrogate_views(
        layer_df,
        target_columns=target_columns,
        val_fraction=float(surrogate_cfg.get("val_fraction", 0.2)),
        seed=seed + int(layer),
    )
    input_dim = int(train_set.clouds[0].shape[1])
    output_dim = int(len(target_columns))
    batch_size = int(surrogate_cfg.get("train_batch_size", 64))
    eval_batch_size = int(surrogate_cfg.get("eval_batch_size", max(batch_size, 128)))
    include_token_count_input = bool(surrogate_cfg.get("include_token_count_input", True))
    device = _surrogate_device(str(surrogate_cfg.get("device", "auto")))
    model = DeepSetTopologySurrogate(
        input_dim=input_dim,
        output_dim=output_dim,
        token_hidden_dim=int(surrogate_cfg.get("token_hidden_dim", 128)),
        head_hidden_dim=int(surrogate_cfg.get("head_hidden_dim", 256)),
        dropout=float(surrogate_cfg.get("dropout", 0.1)),
        include_token_count_input=include_token_count_input,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(surrogate_cfg.get("lr", 1e-3)),
        weight_decay=float(surrogate_cfg.get("weight_decay", 1e-4)),
    )
    loss_name = str(surrogate_cfg.get("loss", "huber")).strip().lower()
    if loss_name == "mse":
        criterion: torch.nn.Module = torch.nn.MSELoss()
    else:
        criterion = torch.nn.SmoothL1Loss(beta=float(surrogate_cfg.get("huber_beta", 1.0)))
    target_mean_np, target_std_np = _target_stats(train_set.targets)
    target_mean = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=_collate_cloud_batch)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, collate_fn=_collate_cloud_batch)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, collate_fn=_collate_cloud_batch)
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience = int(surrogate_cfg.get("patience", 8))
    epochs_without_improvement = 0
    max_epochs = int(surrogate_cfg.get("max_epochs", 50))
    grad_clip_norm = float(surrogate_cfg.get("grad_clip_norm", 1.0))
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss, train_pred, train_true, _ = _run_loader(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            target_mean=target_mean,
            target_std=target_std,
            grad_clip_norm=grad_clip_norm,
        )
        with torch.no_grad():
            val_loss, val_pred, val_true, _ = _run_loader(
                model,
                val_loader,
                device=device,
                optimizer=None,
                criterion=criterion,
                target_mean=target_mean,
                target_std=target_std,
                grad_clip_norm=grad_clip_norm,
            )
        train_metrics = _regression_summary(train_true, train_pred)
        val_metrics = _regression_summary(val_true, val_pred)
        history_rows.append(
            {
                "layer": int(layer),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mae_mean": train_metrics.mae_mean,
                "train_rmse_mean": train_metrics.rmse_mean,
                "train_r2_mean": train_metrics.r2_mean,
                "val_mae_mean": val_metrics.mae_mean,
                "val_rmse_mean": val_metrics.rmse_mean,
                "val_r2_mean": val_metrics.r2_mean,
            }
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, val_pred, val_true, val_meta_idx = _run_loader(
            model,
            val_loader,
            device=device,
            optimizer=None,
            criterion=criterion,
            target_mean=target_mean,
            target_std=target_std,
            grad_clip_norm=grad_clip_norm,
        )
        _, test_pred, test_true, test_meta_idx = _run_loader(
            model,
            test_loader,
            device=device,
            optimizer=None,
            criterion=criterion,
            target_mean=target_mean,
            target_std=target_std,
            grad_clip_norm=grad_clip_norm,
        )
    val_metrics = _regression_summary(val_true, val_pred)
    test_metrics = _regression_summary(test_true, test_pred)
    feature_metrics = pd.concat(
        [
            _per_feature_metrics(val_true, val_pred, columns=target_columns, split="val", layer=layer),
            _per_feature_metrics(test_true, test_pred, columns=target_columns, split="test", layer=layer),
        ],
        ignore_index=True,
    )
    history_df = pd.DataFrame(history_rows)
    write_parquet(history_df, output_root / f"layer_{int(layer):02d}__training_history.parquet")
    checkpoint_path = output_root / f"layer_{int(layer):02d}__surrogate.pt"
    torch.save(
        {
            "layer": int(layer),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "target_columns": target_columns,
            "target_mean": target_mean_np.tolist(),
            "target_std": target_std_np.tolist(),
            "state_dict": model.state_dict(),
            "config": {
                "token_hidden_dim": int(surrogate_cfg.get("token_hidden_dim", 128)),
                "head_hidden_dim": int(surrogate_cfg.get("head_hidden_dim", 256)),
                "dropout": float(surrogate_cfg.get("dropout", 0.1)),
                "include_token_count_input": include_token_count_input,
            },
        },
        checkpoint_path,
    )
    val_meta = val_set.meta.iloc[val_meta_idx].reset_index(drop=True).copy()
    test_meta = test_set.meta.iloc[test_meta_idx].reset_index(drop=True).copy()
    target_label_map = {column: f"target__{column}" for column in target_columns}
    pred_label_map = {column: f"pred__{column}" for column in target_columns}
    val_predictions = pd.concat(
        [
            val_meta,
            pd.DataFrame(val_true, columns=target_columns).rename(columns=target_label_map),
            pd.DataFrame(val_pred, columns=target_columns).rename(columns=pred_label_map),
        ],
        axis=1,
    )
    test_predictions = pd.concat(
        [
            test_meta,
            pd.DataFrame(test_true, columns=target_columns).rename(columns=target_label_map),
            pd.DataFrame(test_pred, columns=target_columns).rename(columns=pred_label_map),
        ],
        axis=1,
    )
    write_parquet(val_predictions, output_root / f"layer_{int(layer):02d}__val_predictions.parquet")
    write_parquet(test_predictions, output_root / f"layer_{int(layer):02d}__test_predictions.parquet")
    result = {
        "layer": int(layer),
        "best_epoch": int(best_epoch),
        "n_train": int(len(train_set)),
        "n_val": int(len(val_set)),
        "n_test": int(len(test_set)),
        "input_dim": int(input_dim),
        "target_dim": int(output_dim),
        "checkpoint_path": str(checkpoint_path),
        "val_mae_mean": val_metrics.mae_mean,
        "val_rmse_mean": val_metrics.rmse_mean,
        "val_r2_mean": val_metrics.r2_mean,
        "val_explained_variance_mean": val_metrics.explained_variance_mean,
        "test_mae_mean": test_metrics.mae_mean,
        "test_rmse_mean": test_metrics.rmse_mean,
        "test_r2_mean": test_metrics.r2_mean,
        "test_explained_variance_mean": test_metrics.explained_variance_mean,
    }
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, feature_metrics


def run_clamber_topology_surrogate(*, config: dict[str, Any], seed: int) -> dict[str, str]:
    set_global_seed(seed)
    surrogate_cfg = dict(config["clamber_topology_surrogate"])
    classifier_cfg = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    output_root = ensure_dir(Path(surrogate_cfg.get("output_dir", "artifacts/reports/clamber_topology_surrogate")) / config["model"]["name"].replace("/", "_").replace("-", "_"))
    subclass_cfg["token_cloud_distance_feature_mode"] = "none"
    subclass_cfg["force_rebuild_token_cloud_features"] = bool(surrogate_cfg.get("force_rebuild_token_cloud_features", False))
    subclass_cfg["force_rebuild_token_cloud_forward_cache"] = bool(surrogate_cfg.get("force_rebuild_token_cloud_forward_cache", False))
    raw_forward_cache_path = str(surrogate_cfg.get("token_cloud_forward_cache_path", "")).strip()
    raw_feature_cache_path = str(surrogate_cfg.get("token_cloud_feature_cache_path", "")).strip()
    forward_cache_path = Path(raw_forward_cache_path) if raw_forward_cache_path else output_root / "clamber_token_cloud_reduced_clouds.joblib"
    feature_cache_path = Path(raw_feature_cache_path) if raw_feature_cache_path else output_root / "clamber_token_cloud_all_layer_features.parquet"
    subclass_cfg["token_cloud_forward_cache_path"] = str(forward_cache_path)
    subclass_cfg["token_cloud_feature_cache_path"] = str(feature_cache_path)

    model_name, feature_df, reused_feature_cache = _build_clamber_token_cloud_features(
        config=config,
        classifier_config=classifier_cfg,
        subclass_cfg=subclass_cfg,
        seed=seed,
    )
    if not forward_cache_path.exists():
        raise FileNotFoundError(f"Expected token-cloud forward cache at {forward_cache_path}")
    cloud_df = _load_cached_cloud_df(forward_cache_path)
    target_columns = _topology_target_columns(feature_df, include_token_count=False, include_distance_features=False)
    requested_target_columns = surrogate_cfg.get("target_columns")
    if requested_target_columns:
        requested = [str(column) for column in requested_target_columns]
        missing = [column for column in requested if column not in target_columns]
        if missing:
            raise ValueError(f"Requested topology surrogate targets are unavailable: {missing}")
        target_columns = requested
    merge_columns = _MERGE_KEYS + ["subclass"] + target_columns
    training_df = cloud_df.merge(
        feature_df.loc[:, [column for column in merge_columns if column in feature_df.columns]].drop_duplicates(),
        on=_MERGE_KEYS,
        how="inner",
    )
    if training_df.empty:
        raise ValueError("Topology surrogate training frame is empty after joining token clouds and topology targets.")
    requested_layers = surrogate_cfg.get("layers")
    if requested_layers:
        layers = [int(layer) for layer in requested_layers]
    else:
        layers = sorted(training_df["layer"].unique().tolist())

    layer_rows: list[dict[str, Any]] = []
    feature_metric_frames: list[pd.DataFrame] = []
    for layer in layers:
        LOGGER.info("Training topology surrogate: model=%s layer=%s target_dim=%s", model_name, layer, len(target_columns))
        row, feature_metrics = _train_single_layer_surrogate(
            training_df,
            layer=int(layer),
            target_columns=target_columns,
            surrogate_cfg=surrogate_cfg,
            seed=seed,
            output_root=output_root,
        )
        layer_rows.append(row)
        feature_metric_frames.append(feature_metrics)

    layer_df = pd.DataFrame(layer_rows).sort_values(["test_r2_mean", "layer"], ascending=[False, True]).reset_index(drop=True)
    feature_metrics_df = pd.concat(feature_metric_frames, ignore_index=True) if feature_metric_frames else pd.DataFrame()
    layer_metrics_filename = str(surrogate_cfg.get("layer_metrics_filename", "clamber_topology_surrogate_layer_metrics.parquet"))
    feature_metrics_filename = str(surrogate_cfg.get("feature_metrics_filename", "clamber_topology_surrogate_feature_metrics.parquet"))
    summary_filename = str(surrogate_cfg.get("summary_filename", "clamber_topology_surrogate_summary.md"))
    metadata_filename = str(surrogate_cfg.get("metadata_filename", "clamber_topology_surrogate_metadata.json"))
    write_parquet(layer_df, output_root / layer_metrics_filename)
    write_parquet(feature_metrics_df, output_root / feature_metrics_filename)
    best_row = layer_df.iloc[0].to_dict() if not layer_df.empty else {}
    write_json(
        output_root / metadata_filename,
        {
            "model_name": model_name,
            "reused_feature_cache": bool(reused_feature_cache),
            "forward_cache_path": str(forward_cache_path),
            "feature_cache_path": str(feature_cache_path),
            "layers": layers,
            "target_columns": target_columns,
            "best_layer": int(best_row["layer"]) if best_row else None,
            "best_test_r2_mean": float(best_row["test_r2_mean"]) if best_row else None,
        },
    )
    report_lines = [
        f"# CLAMBER Topology Surrogate ({model_name})",
        "",
        f"- Forward cache: `{forward_cache_path}`",
        f"- Feature cache: `{feature_cache_path}`",
        f"- Reused feature cache: `{bool(reused_feature_cache)}`",
        f"- Target features: `{len(target_columns)}`",
        "",
        "## Layer Metrics",
        "",
    ]
    for row in layer_df.to_dict(orient="records"):
        report_lines.extend(
            [
                f"### Layer {int(row['layer'])}",
                "",
                f"- Best epoch: `{int(row['best_epoch'])}`",
                f"- Train/val/test: `{int(row['n_train'])}/{int(row['n_val'])}/{int(row['n_test'])}`",
                f"- Validation mean R^2: `{float(row['val_r2_mean']):.4f}`",
                f"- Test mean R^2: `{float(row['test_r2_mean']):.4f}`",
                f"- Test mean MAE: `{float(row['test_mae_mean']):.4f}`",
                "",
            ]
        )
    write_markdown(output_root / summary_filename, "\n".join(report_lines) + "\n")
    return {
        "output_root": str(output_root),
        "layer_metrics_path": str(output_root / layer_metrics_filename),
        "feature_metrics_path": str(output_root / feature_metrics_filename),
        "summary_path": str(output_root / summary_filename),
        "metadata_path": str(output_root / metadata_filename),
    }
