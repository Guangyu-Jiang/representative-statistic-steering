"""Topology-3 surrogate steering for ambiguity abstention.

This experiment uses only three H0 token-cloud features:

- h0_mean_persistence
- h0_persistence_entropy
- h0_top5_persistence_fraction

The all-layer feature stack is used to choose a nearest abstention target. A
last-layer DeepSets surrogate then maps PCA-reduced prompt-token clouds to the
three last-layer topology features. Steering optimizes a small PCA-space token
cloud perturbation toward the target topology and maps that perturbation back to
the last-layer hidden state for generation.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.models.generation import _build_generate_kwargs, render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import _compute_diagrams
from aen_replication.train.steering import _decoder_layers
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet
from aen_replication.utils.seed import set_global_seed


FEATURES = ("h0_mean_persistence", "h0_persistence_entropy", "h0_top5_persistence_fraction")
MODEL_SUBDIR = "meta_llama_llama_3_1_8b_instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runs/llama_steering_paper_style_openai_ibm.yaml")
    parser.add_argument("--datasets", nargs="+", default=["ambigqa", "situatedqa"])
    parser.add_argument("--source-root", default="artifacts/steering_paper_style_openai_ibm")
    parser.add_argument("--artifact-root", default="artifacts/steering_topology3_surrogate")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Layer whose token cloud is optimized and whose hidden state is steered. Defaults to the final layer.",
    )
    parser.add_argument("--pca-fit-token-cap", type=int, default=32000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=8)
    parser.add_argument("--surrogate-epochs", type=int, default=80)
    parser.add_argument("--surrogate-patience", type=int, default=12)
    parser.add_argument("--surrogate-batch-size", type=int, default=32)
    parser.add_argument("--surrogate-lr", type=float, default=1e-3)
    parser.add_argument("--opt-steps", type=int, default=80)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-batch-size", type=int, default=16)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[100.0])
    parser.add_argument("--eval-n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit after sampling D- eval rows.")
    parser.add_argument(
        "--behavior-limit-per-label",
        type=int,
        default=None,
        help="Optional smoke-test cap per D+/D- behavior label before feature extraction.",
    )
    parser.add_argument(
        "--apply-on",
        choices=["prefill_only", "prompt_and_decode_mean"],
        default="prompt_and_decode_mean",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["token_matrix", "shared_vector", "local_mean_vector"],
        default="token_matrix",
        help=(
            "token_matrix optimizes one perturbation per prompt token; "
            "shared_vector optimizes one PCA-space vector broadcast to all prompt and decode tokens; "
            "local_mean_vector optimizes only a scalar on the local target-minus-current mean-cloud direction."
        ),
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-surrogate", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def _slug_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _run_slug(alpha: float, lambda_value: float) -> str:
    return f"alpha_{_slug_float(alpha)}__lambda_{_slug_float(lambda_value)}"


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_behavior(config: dict[str, Any], dataset: str, source_root: Path) -> pd.DataFrame:
    model_slug = slugify(config["model"]["name"])
    path = source_root / dataset / "_base" / model_slug / f"{dataset}__base_behavior.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing base behavior table: {path}")
    behavior = pd.read_parquet(path)
    behavior = behavior.loc[behavior["judge_label"].isin(["ACCEPTABLE", "UNACCEPTABLE"])].copy()
    behavior["behavior_label"] = behavior["judge_label"].eq("ACCEPTABLE").astype(int)
    return behavior.reset_index(drop=True)


def _prompt_texts(config: dict[str, Any], texts: list[str]) -> list[str]:
    suffix = str(config["steering"].get("prompt_suffix", ""))
    return [f"{text}{suffix}" if suffix else str(text) for text in texts]


def _rendered_prompts(bundle: HFModelBundle, config: dict[str, Any], texts: list[str]) -> list[str]:
    return render_prompts(
        bundle=bundle,
        prompt_texts=_prompt_texts(config, texts),
        use_chat_template=bool(config["generation"].get("use_chat_template", True)),
        system_prompt=config["generation"].get("system_prompt"),
        add_generation_prompt=True,
    )


def _valid_token_mask(
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    *,
    special_ids: set[int],
) -> torch.Tensor:
    valid = attention_mask_row.bool().clone()
    for special_id in special_ids:
        valid &= ~input_ids_row.eq(int(special_id))
    if int(valid.sum().item()) == 0:
        valid = attention_mask_row.bool()
    return valid


def _collect_pca_fit_tokens(
    *,
    bundle: HFModelBundle,
    frame: pd.DataFrame,
    rendered_prompts: list[str],
    layers: list[int],
    batch_size: int,
    max_length: int,
    token_cap: int,
    seed: int,
) -> dict[int, np.ndarray]:
    tokenizer = bundle.tokenizer
    device = bundle.device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    rng = np.random.default_rng(seed)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in tqdm(range(0, len(frame), batch_size), desc="topology3_pca_tokens", leave=False):
            prompts = rendered_prompts[start : start + batch_size]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids_cpu = encoded["input_ids"].detach().cpu()
            attention_mask_cpu = encoded["attention_mask"].detach().cpu()
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = bundle.model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            valid_masks = [
                _valid_token_mask(input_ids_cpu[row], attention_mask_cpu[row], special_ids=special_ids)
                for row in range(input_ids_cpu.shape[0])
            ]
            for layer in layers:
                layer_output = hidden_states[layer + 1].detach().float().cpu()
                chunks = [layer_output[row][valid_masks[row]].numpy() for row in range(layer_output.shape[0])]
                chunks = [chunk for chunk in chunks if len(chunk)]
                if chunks:
                    per_layer[layer].append(np.vstack(chunks).astype(np.float32, copy=False))
            del outputs, hidden_states, model_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side

    result: dict[int, np.ndarray] = {}
    for layer, chunks in per_layer.items():
        matrix = np.vstack(chunks).astype(np.float32, copy=False)
        if len(matrix) > token_cap:
            selected = np.sort(rng.choice(len(matrix), size=token_cap, replace=False))
            matrix = matrix[selected]
        result[layer] = matrix
    return result


def _fit_reducers(token_matrices: dict[int, np.ndarray], *, n_components: int, seed: int) -> dict[int, PCA]:
    reducers: dict[int, PCA] = {}
    for layer, matrix in token_matrices.items():
        components = max(1, min(int(n_components), matrix.shape[1], matrix.shape[0] - 1))
        reducer = PCA(n_components=components, svd_solver="randomized", random_state=seed + int(layer))
        reducer.fit(matrix)
        reducers[int(layer)] = reducer
    return reducers


def _extract_reduced_clouds(
    *,
    bundle: HFModelBundle,
    frame: pd.DataFrame,
    rendered_prompts: list[str],
    layers: list[int],
    reducers: dict[int, PCA],
    batch_size: int,
    max_length: int,
    topology_dim: int,
) -> pd.DataFrame:
    tokenizer = bundle.tokenizer
    device = bundle.device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    rows: list[dict[str, Any]] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in tqdm(range(0, len(frame), batch_size), desc="topology3_clouds", leave=False):
            prompts = rendered_prompts[start : start + batch_size]
            batch_frame = frame.iloc[start : start + batch_size].reset_index(drop=True)
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids_cpu = encoded["input_ids"].detach().cpu()
            attention_mask_cpu = encoded["attention_mask"].detach().cpu()
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = bundle.model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            valid_masks = [
                _valid_token_mask(input_ids_cpu[row], attention_mask_cpu[row], special_ids=special_ids)
                for row in range(input_ids_cpu.shape[0])
            ]
            batch_records = batch_frame.to_dict(orient="records")
            for layer in layers:
                reducer = reducers[int(layer)]
                layer_output = hidden_states[layer + 1].detach().float().cpu()
                token_chunks: list[np.ndarray] = []
                token_counts: list[int] = []
                hidden_norms: list[float] = []
                for row_index in range(layer_output.shape[0]):
                    tokens = layer_output[row_index][valid_masks[row_index]].numpy().astype(np.float32, copy=False)
                    token_chunks.append(tokens)
                    token_counts.append(int(len(tokens)))
                    hidden_norms.append(float(np.linalg.norm(tokens)))
                token_matrix = np.vstack(token_chunks)
                reduced_dim = min(int(topology_dim), int(reducer.n_components_))
                reduced_matrix = reducer.transform(token_matrix)[:, :reduced_dim].astype(np.float32, copy=False)
                offset = 0
                for row, token_count, hidden_norm in zip(batch_records, token_counts, hidden_norms, strict=True):
                    cloud = reduced_matrix[offset : offset + token_count]
                    offset += token_count
                    rows.append(
                        {
                            "example_id": str(row["example_id"]),
                            "pair_id": str(row.get("pair_id", "")),
                            "dataset": str(row["dataset"]),
                            "split": str(row.get("split", "test")),
                            "label_ambiguous": int(row.get("label_ambiguous", 1)),
                            "judge_label": str(row["judge_label"]),
                            "behavior_label": int(row["behavior_label"]),
                            "layer": int(layer),
                            "token_count": int(token_count),
                            "hidden_fro_norm": hidden_norm,
                            "reduced_fro_norm": float(np.linalg.norm(cloud)),
                            "cloud": cloud,
                        }
                    )
            del outputs, hidden_states, model_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return pd.DataFrame(rows)


def _h0_three_features(row: dict[str, Any], *, grid_size: int = 24) -> dict[str, Any]:
    cloud = np.asarray(row["cloud"], dtype=float)
    diagrams = _compute_diagrams(cloud, maxdim=0, coeff=2, distance_metric="euclidean")
    finite = np.asarray(diagrams[0], dtype=float)
    if finite.size == 0:
        lifetimes = np.zeros(0, dtype=float)
    else:
        finite = finite[np.isfinite(finite[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0] if finite.size else np.zeros(0, dtype=float)
        lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if lifetimes.size == 0:
        mean_persistence = 0.0
        entropy = 0.0
        top5_fraction = 0.0
    else:
        mean_persistence = float(np.mean(lifetimes))
        if lifetimes.size <= 1:
            entropy = 0.0
        else:
            weights = lifetimes / max(float(lifetimes.sum()), 1e-12)
            entropy = float(-(weights * np.log(weights + 1e-12)).sum() / np.log(len(weights)))
        sorted_desc = np.sort(lifetimes)[::-1]
        top5_fraction = float(sorted_desc[:5].sum() / max(float(sorted_desc.sum()), 1e-12))
    return {
        "example_id": str(row["example_id"]),
        "pair_id": str(row.get("pair_id", "")),
        "dataset": str(row["dataset"]),
        "split": str(row.get("split", "test")),
        "label_ambiguous": int(row.get("label_ambiguous", 1)),
        "judge_label": str(row["judge_label"]),
        "behavior_label": int(row["behavior_label"]),
        "layer": int(row["layer"]),
        "token_count": int(row["token_count"]),
        "hidden_fro_norm": float(row["hidden_fro_norm"]),
        "reduced_fro_norm": float(row["reduced_fro_norm"]),
        "h0_mean_persistence": mean_persistence,
        "h0_persistence_entropy": entropy,
        "h0_top5_persistence_fraction": top5_fraction,
    }


def _compute_feature_frame(cloud_df: pd.DataFrame, *, parallel_jobs: int) -> pd.DataFrame:
    rows = cloud_df.to_dict(orient="records")
    feature_rows = Parallel(n_jobs=max(1, int(parallel_jobs)), backend="loky")(
        delayed(_h0_three_features)(row) for row in tqdm(rows, desc="topology3_ph", leave=False)
    )
    return pd.DataFrame(feature_rows)


def _build_or_load_features(
    *,
    config: dict[str, Any],
    dataset: str,
    behavior: pd.DataFrame,
    artifact_root: Path,
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, PCA], list[int]]:
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(artifact_root / dataset / model_slug)
    cloud_path = output_root / f"{dataset}__topology3_clouds.joblib"
    feature_path = output_root / f"{dataset}__topology3_features.parquet"
    reducer_path = output_root / f"{dataset}__topology3_reducers.joblib"

    if cloud_path.exists() and feature_path.exists() and reducer_path.exists() and not args.force_features:
        payload = joblib.load(cloud_path)
        cloud_df = pd.DataFrame(payload["cloud_df"])
        feature_df = pd.read_parquet(feature_path)
        reducers = joblib.load(reducer_path)
        layers = sorted(int(layer) for layer in reducers.keys())
        return cloud_df, feature_df, reducers, layers

    bundle = load_hf_model(config["model"], config["generation"])
    try:
        layers = list(range(len(_decoder_layers(bundle.model))))
        rendered = _rendered_prompts(bundle, config, behavior["text"].astype(str).tolist())
        token_matrices = _collect_pca_fit_tokens(
            bundle=bundle,
            frame=behavior,
            rendered_prompts=rendered,
            layers=layers,
            batch_size=int(args.feature_batch_size),
            max_length=int(args.max_length),
            token_cap=int(args.pca_fit_token_cap),
            seed=seed,
        )
        reducers = _fit_reducers(token_matrices, n_components=int(args.pca_components), seed=seed)
        cloud_df = _extract_reduced_clouds(
            bundle=bundle,
            frame=behavior,
            rendered_prompts=rendered,
            layers=layers,
            reducers=reducers,
            batch_size=int(args.feature_batch_size),
            max_length=int(args.max_length),
            topology_dim=int(args.pca_components),
        )
    finally:
        del bundle
        _release()

    feature_df = _compute_feature_frame(cloud_df, parallel_jobs=int(args.parallel_jobs))
    joblib.dump({"cloud_df": cloud_df}, cloud_path)
    joblib.dump(reducers, reducer_path)
    write_parquet(feature_df, feature_path)
    feature_df.to_csv(feature_path.with_suffix(".csv"), index=False)
    return cloud_df, feature_df, reducers, layers


def _wide_features(feature_df: pd.DataFrame, *, layers: list[int]) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = feature_df.loc[
        feature_df["layer"].eq(layers[0]),
        ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "judge_label", "behavior_label"],
    ].drop_duplicates("example_id")
    wide = base.copy()
    all_columns: list[str] = []
    last_layer_columns: list[str] = []
    for layer in layers:
        layer_df = feature_df.loc[feature_df["layer"].eq(layer), ["example_id", *FEATURES]].drop_duplicates("example_id")
        rename = {feature: f"{feature}__l{layer:02d}" for feature in FEATURES}
        layer_df = layer_df.rename(columns=rename)
        wide = wide.merge(layer_df, on="example_id", how="inner")
        columns = [rename[feature] for feature in FEATURES]
        all_columns.extend(columns)
        if layer == layers[-1]:
            last_layer_columns = columns
    return wide, all_columns, last_layer_columns


def _classifier_metrics(y_true: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def _train_classifier_diagnostics(
    *,
    wide_df: pd.DataFrame,
    all_columns: list[str],
    last_layer_columns: list[str],
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    y = wide_df["behavior_label"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(wide_df)), y))
    rows: list[dict[str, Any]] = []
    for feature_set, columns in [("all_layers", all_columns), ("last_layer", last_layer_columns)]:
        x = wide_df.loc[:, columns].to_numpy(dtype=float)
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=4000,
                random_state=seed,
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        for split_name, indices in [("train", train_idx), ("val", val_idx)]:
            scores = clf.decision_function(x[indices])
            predictions = clf.predict(x[indices])
            metrics = _classifier_metrics(y[indices], scores, predictions)
            rows.append(
                {
                    "feature_set": feature_set,
                    "split": split_name,
                    "n": int(len(indices)),
                    "positives": int(y[indices].sum()),
                    "feature_count": int(len(columns)),
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(rows)
    write_parquet(metrics_df, output_root / "classifier_diagnostics.parquet")
    metrics_df.to_csv(output_root / "classifier_diagnostics.csv", index=False)
    return {"rows": rows}


class _CloudDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_columns: list[str]) -> None:
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
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_tokens = max(int(item["cloud"].shape[0]) for item in batch)
    input_dim = int(batch[0]["cloud"].shape[1])
    output_dim = int(batch[0]["target"].shape[0])
    clouds = torch.zeros((batch_size, max_tokens, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    targets = torch.zeros((batch_size, output_dim), dtype=torch.float32)
    token_counts = torch.zeros(batch_size, dtype=torch.float32)
    for row_index, item in enumerate(batch):
        cloud = torch.from_numpy(item["cloud"])
        token_count = int(cloud.shape[0])
        clouds[row_index, :token_count] = cloud
        mask[row_index, :token_count] = True
        targets[row_index] = torch.from_numpy(item["target"])
        token_counts[row_index] = float(item["token_count"])
    return {"clouds": clouds, "mask": mask, "targets": targets, "token_counts": token_counts}


class _DeepSetSurrogate(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, token_hidden_dim: int = 128, head_hidden_dim: int = 192) -> None:
        super().__init__()
        self.token_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, token_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(token_hidden_dim, token_hidden_dim),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(token_hidden_dim * 3 + 1, head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(head_hidden_dim, head_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
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
        pooled = torch.cat([mean, std, max_values, torch.log1p(token_counts).unsqueeze(-1)], dim=1)
        return self.head(pooled)


def _run_surrogate_loader(
    model: _DeepSetSurrogate,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        clouds = batch["clouds"].to(device)
        mask = batch["mask"].to(device)
        token_counts = batch["token_counts"].to(device)
        target = batch["targets"].to(device)
        target_norm = (target - target_mean) / target_std
        pred_norm = model(clouds, mask, token_counts)
        loss = torch.nn.functional.mse_loss(pred_norm, target_norm)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        preds.append((pred_norm.detach() * target_std + target_mean).cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    return float(np.mean(losses)), np.vstack(preds), np.vstack(targets)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    error = y_pred - y_true
    mse = np.mean(np.square(error), axis=0)
    mae = np.mean(np.abs(error), axis=0)
    sse = np.sum(np.square(error), axis=0)
    sst = np.sum(np.square(y_true - y_true.mean(axis=0)), axis=0)
    r2 = np.where(sst > 1e-8, 1.0 - sse / sst, 0.0)
    return {
        "mse_mean": float(np.mean(mse)),
        "mae_mean": float(np.mean(mae)),
        "r2_mean": float(np.mean(r2)),
        "per_feature": [
            {"feature": feature, "mse": float(mse[i]), "mae": float(mae[i]), "r2": float(r2[i])}
            for i, feature in enumerate(FEATURES)
        ],
    }


def _train_or_load_surrogate(
    *,
    cloud_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    layer: int,
    output_root: Path,
    args: argparse.Namespace,
    seed: int,
) -> tuple[_DeepSetSurrogate, dict[str, Any]]:
    checkpoint_path = output_root / f"layer_{layer:02d}__topology3_surrogate.pt"
    metrics_path = output_root / f"layer_{layer:02d}__topology3_surrogate_metrics.json"
    layer_cloud = cloud_df.loc[cloud_df["layer"].eq(layer)].drop(columns=[column for column in FEATURES if column in cloud_df.columns])
    layer_features = feature_df.loc[feature_df["layer"].eq(layer), ["example_id", *FEATURES]].drop_duplicates("example_id")
    frame = layer_cloud.merge(layer_features, on="example_id", how="inner").reset_index(drop=True)
    input_dim = int(np.asarray(frame["cloud"].iloc[0]).shape[1])
    output_dim = len(FEATURES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path.exists() and metrics_path.exists() and not args.force_surrogate:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = _DeepSetSurrogate(input_dim=int(checkpoint["input_dim"]), output_dim=int(checkpoint["output_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        return model, {**checkpoint, "metrics": joblib.load(metrics_path) if metrics_path.suffix == ".joblib" else None}

    y = frame["behavior_label"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(frame)), y))
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    target_mean_np = train_df.loc[:, FEATURES].to_numpy(dtype=np.float32).mean(axis=0)
    target_std_np = train_df.loc[:, FEATURES].to_numpy(dtype=np.float32).std(axis=0)
    target_std_np = np.where(target_std_np > 1e-6, target_std_np, 1.0).astype(np.float32)
    target_mean = torch.from_numpy(target_mean_np).to(device)
    target_std = torch.from_numpy(target_std_np).to(device)
    model = _DeepSetSurrogate(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.surrogate_lr), weight_decay=1e-4)
    train_loader = DataLoader(
        _CloudDataset(train_df, list(FEATURES)),
        batch_size=int(args.surrogate_batch_size),
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        _CloudDataset(val_df, list(FEATURES)),
        batch_size=int(args.surrogate_batch_size),
        shuffle=False,
        collate_fn=_collate,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.surrogate_epochs) + 1):
        train_loss, train_pred, train_true = _run_surrogate_loader(
            model, train_loader, device=device, optimizer=optimizer, target_mean=target_mean, target_std=target_std
        )
        with torch.no_grad():
            val_loss, val_pred, val_true = _run_surrogate_loader(
                model, val_loader, device=device, optimizer=None, target_mean=target_mean, target_std=target_std
            )
        train_metrics = _regression_metrics(train_true, train_pred)
        val_metrics = _regression_metrics(val_true, val_pred)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_r2_mean": train_metrics["r2_mean"],
                "val_r2_mean": val_metrics["r2_mean"],
            }
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(args.surrogate_patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, val_pred, val_true = _run_surrogate_loader(
            model, val_loader, device=device, optimizer=None, target_mean=target_mean, target_std=target_std
        )
    metrics = {
        "layer": int(layer),
        "target_columns": list(FEATURES),
        "target_mean": target_mean_np.tolist(),
        "target_std": target_std_np.tolist(),
        "val": _regression_metrics(val_true, val_pred),
        "history": history,
    }
    torch.save(
        {
            "layer": int(layer),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "target_columns": list(FEATURES),
            "target_mean": target_mean_np.tolist(),
            "target_std": target_std_np.tolist(),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    write_json(metrics_path, metrics)
    return model, metrics


def _nearest_targets(
    *,
    wide_df: pd.DataFrame,
    behavior: pd.DataFrame,
    all_columns: list[str],
    feature_df: pd.DataFrame,
    layer: int,
    eval_n: int,
    seed: int,
    limit: int | None,
) -> pd.DataFrame:
    plus = wide_df.loc[wide_df["behavior_label"].eq(1)].copy().reset_index(drop=True)
    minus = wide_df.loc[wide_df["behavior_label"].eq(0)].copy().reset_index(drop=True)
    eval_n = min(int(eval_n), len(minus))
    eval_df = minus.sample(n=eval_n, random_state=seed).reset_index(drop=True)
    if limit is not None:
        eval_df = eval_df.head(int(limit)).reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit = imputer.fit_transform(pd.concat([plus.loc[:, all_columns], eval_df.loc[:, all_columns]], axis=0))
    fit = scaler.fit_transform(fit)
    plus_matrix = fit[: len(plus)]
    eval_matrix = fit[len(plus) :]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(plus_matrix)
    distances, indices = nn.kneighbors(eval_matrix)
    plus_reset = plus.reset_index(drop=True)
    behavior_by_id = behavior.set_index("example_id", drop=False)
    layer_features = feature_df.loc[feature_df["layer"].eq(layer), ["example_id", *FEATURES]].set_index("example_id")
    rows: list[dict[str, Any]] = []
    for row_index, eval_row in eval_df.reset_index(drop=True).iterrows():
        target_row = plus_reset.iloc[int(indices[row_index, 0])]
        eval_id = str(eval_row["example_id"])
        target_id = str(target_row["example_id"])
        source_behavior = behavior_by_id.loc[eval_id]
        target_behavior = behavior_by_id.loc[target_id]
        source_z = layer_features.loc[eval_id, list(FEATURES)].to_numpy(dtype=float)
        target_z = layer_features.loc[target_id, list(FEATURES)].to_numpy(dtype=float)
        rows.append(
            {
                "row_index": int(row_index),
                "example_id": eval_id,
                "pair_id": str(eval_row["pair_id"]),
                "dataset": str(eval_row["dataset"]),
                "split": str(eval_row["split"]),
                "label_ambiguous": int(eval_row["label_ambiguous"]),
                "text": str(source_behavior["text"]),
                "base_response_text": str(source_behavior["response_text"]),
                "base_judge_label": str(source_behavior["judge_label"]),
                "target_example_id": target_id,
                "target_pair_id": str(target_row["pair_id"]),
                "target_text": str(target_behavior["text"]),
                "target_base_response_text": str(target_behavior["response_text"]),
                "topology3_all_layer_distance": float(distances[row_index, 0]),
                **{f"current__{feature}": float(value) for feature, value in zip(FEATURES, source_z, strict=True)},
                **{f"target__{feature}": float(value) for feature, value in zip(FEATURES, target_z, strict=True)},
            }
        )
    return pd.DataFrame(rows)


def _pad_clouds(clouds: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(clouds)
    max_tokens = max(int(cloud.shape[0]) for cloud in clouds)
    input_dim = int(clouds[0].shape[1])
    tensor = torch.zeros((batch_size, max_tokens, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    counts = torch.zeros(batch_size, dtype=torch.float32)
    for index, cloud in enumerate(clouds):
        cloud_tensor = torch.from_numpy(np.asarray(cloud, dtype=np.float32))
        token_count = int(cloud_tensor.shape[0])
        tensor[index, :token_count] = cloud_tensor
        mask[index, :token_count] = True
        counts[index] = float(token_count)
    return tensor, mask, counts


def _exact_features_for_clouds(clouds: list[np.ndarray], *, parallel_jobs: int) -> np.ndarray:
    rows = [
        {
            "example_id": str(index),
            "pair_id": str(index),
            "dataset": "optimization",
            "split": "eval",
            "label_ambiguous": 1,
            "judge_label": "UNACCEPTABLE",
            "behavior_label": 0,
            "layer": 0,
            "token_count": len(cloud),
            "hidden_fro_norm": 0.0,
            "reduced_fro_norm": float(np.linalg.norm(cloud)),
            "cloud": cloud,
        }
        for index, cloud in enumerate(clouds)
    ]
    feature_df = _compute_feature_frame(pd.DataFrame(rows), parallel_jobs=parallel_jobs)
    return feature_df.loc[:, FEATURES].to_numpy(dtype=np.float32)


def _optimize_topology_deltas(
    *,
    model: _DeepSetSurrogate,
    current_clouds: list[np.ndarray],
    target_clouds: list[np.ndarray] | None,
    target_features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    lambda_value: float,
    steps: int,
    lr: float,
    batch_size: int,
    parallel_jobs: int,
    delta_mode: str,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    device = next(model.parameters()).device
    model.eval()
    target_mean_t = torch.as_tensor(target_mean, device=device, dtype=torch.float32)
    target_std_t = torch.as_tensor(target_std, device=device, dtype=torch.float32)
    deltas: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(current_clouds), batch_size), desc="topology3_opt", leave=False):
        batch_clouds = current_clouds[start : start + batch_size]
        batch_target_clouds = target_clouds[start : start + batch_size] if target_clouds is not None else None
        batch_targets = target_features[start : start + batch_size]
        clouds, mask, counts = _pad_clouds(batch_clouds)
        clouds = clouds.to(device)
        mask = mask.to(device)
        counts = counts.to(device)
        target = torch.as_tensor(batch_targets, device=device, dtype=torch.float32)
        target_norm = (target - target_mean_t) / target_std_t
        if delta_mode == "token_matrix":
            raw_delta = torch.zeros_like(clouds, requires_grad=True)
        elif delta_mode == "shared_vector":
            raw_delta = torch.zeros((clouds.shape[0], clouds.shape[2]), device=device, dtype=clouds.dtype, requires_grad=True)
        elif delta_mode == "local_mean_vector":
            if batch_target_clouds is None:
                raise ValueError("local_mean_vector requires target_clouds.")
            local_directions = np.vstack(
                [
                    np.asarray(target_cloud, dtype=np.float32).mean(axis=0)
                    - np.asarray(current_cloud, dtype=np.float32).mean(axis=0)
                    for current_cloud, target_cloud in zip(batch_clouds, batch_target_clouds, strict=True)
                ]
            ).astype(np.float32, copy=False)
            direction_t = torch.as_tensor(local_directions, device=device, dtype=clouds.dtype)
            raw_delta = torch.zeros((clouds.shape[0], 1), device=device, dtype=clouds.dtype, requires_grad=True)
        else:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")
        optimizer = torch.optim.Adam([raw_delta], lr=float(lr))
        mask_f = mask.unsqueeze(-1).float()
        denom = clouds.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)
        last_topo_loss = 0.0
        last_reg = 0.0
        token_counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        def build_delta() -> torch.Tensor:
            if delta_mode == "token_matrix":
                delta_value = raw_delta * mask_f
                delta_mean = delta_value.sum(dim=1, keepdim=True) / token_counts
                return (delta_value - delta_mean) * mask_f
            if delta_mode == "local_mean_vector":
                return raw_delta.view(-1, 1, 1) * direction_t[:, None, :] * mask_f
            return raw_delta[:, None, :].expand(-1, clouds.shape[1], -1) * mask_f

        for _step in range(int(steps)):
            delta = build_delta()
            pred_norm = model(clouds + delta, mask, counts)
            topo_loss_per = torch.mean((pred_norm - target_norm).pow(2), dim=1)
            reg_per = delta.pow(2).sum(dim=(1, 2)) / denom
            loss = torch.mean(topo_loss_per + float(lambda_value) * reg_per)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_topo_loss = float(topo_loss_per.mean().detach().cpu())
            last_reg = float(reg_per.mean().detach().cpu())
        with torch.no_grad():
            delta = build_delta()
            pred_norm = model(clouds + delta, mask, counts)
            pred = pred_norm * target_std_t + target_mean_t
            delta_cpu = delta.detach().cpu().numpy()
            pred_norm_cpu = pred_norm.detach().cpu().numpy()
            pred_cpu = pred.detach().cpu().numpy()
            clouds_cpu = clouds.detach().cpu().numpy()
            mask_cpu = mask.detach().cpu().numpy()
            target_norm_cpu = target_norm.detach().cpu().numpy()
            if delta_mode == "local_mean_vector":
                direction_cpu = direction_t.detach().cpu().numpy()
                coeff_cpu = raw_delta.detach().cpu().numpy().reshape(-1)
            else:
                direction_cpu = None
                coeff_cpu = None
        target_std_np = np.asarray(target_std, dtype=np.float32)
        steered_clouds: list[np.ndarray] = []
        for local_index in range(len(batch_clouds)):
            token_count = int(mask_cpu[local_index].sum())
            delta_valid = delta_cpu[local_index, :token_count].astype(np.float32, copy=False)
            current_valid = clouds_cpu[local_index, :token_count].astype(np.float32, copy=False)
            deltas.append(delta_valid)
            steered_clouds.append(current_valid + delta_valid)
            delta_norm = float(np.linalg.norm(delta_valid))
            cloud_norm = float(np.linalg.norm(current_valid))
            target_error = float(np.linalg.norm(pred_cpu[local_index] - batch_targets[local_index]))
            target_error_norm = float(np.linalg.norm(pred_norm_cpu[local_index] - target_norm_cpu[local_index]))
            row_payload = {
                "row_index": int(start + local_index),
                "surrogate_target_l2_error": target_error,
                "surrogate_target_l2_error_normalized": target_error_norm,
                "surrogate_topology_loss": last_topo_loss,
                "pca_delta_norm": delta_norm,
                "pca_cloud_norm": cloud_norm,
                "relative_pca_delta_norm": float(delta_norm / max(cloud_norm, 1e-12)),
                "regularization_value": last_reg,
                **{
                    f"surrogate_pred__{feature}": float(pred_cpu[local_index, feature_index])
                    for feature_index, feature in enumerate(FEATURES)
                },
            }
            if delta_mode == "local_mean_vector" and direction_cpu is not None and coeff_cpu is not None:
                row_payload["local_direction_norm"] = float(np.linalg.norm(direction_cpu[local_index]))
                row_payload["optimized_delta_coeff"] = float(coeff_cpu[local_index])
            rows.append(row_payload)
        exact = _exact_features_for_clouds(steered_clouds, parallel_jobs=parallel_jobs)
        for local_index in range(len(steered_clouds)):
            row = rows[start + local_index]
            row["exact_target_l2_error"] = float(np.linalg.norm(exact[local_index] - batch_targets[local_index]))
            row["exact_target_l2_error_normalized"] = float(
                np.linalg.norm((exact[local_index] - batch_targets[local_index]) / target_std_np)
            )
            for feature_index, feature in enumerate(FEATURES):
                row[f"exact_steered__{feature}"] = float(exact[local_index, feature_index])
    return deltas, pd.DataFrame(rows)


def _generate_with_token_deltas(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    rows_df: pd.DataFrame,
    token_deltas_h: list[np.ndarray],
    layer: int,
    max_length: int,
    apply_on: str,
) -> list[str]:
    tokenizer = bundle.tokenizer
    device = bundle.device
    target_layer = _decoder_layers(bundle.model)[layer]
    generation_cfg = dict(config["generation"])
    batch_size = int(generation_cfg.get("batch_size", 8))
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    responses: list[str] = []

    for start in tqdm(range(0, len(rows_df), batch_size), desc="topology3_generate", leave=False):
        batch_rows = rows_df.iloc[start : start + batch_size].reset_index(drop=True)
        batch_texts = batch_rows["text"].astype(str).tolist()
        batch_deltas = token_deltas_h[start : start + len(batch_rows)]
        rendered = _rendered_prompts(bundle, config, batch_texts)
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenizer.padding_side = original_padding_side
        input_ids_cpu = encoded["input_ids"].detach().cpu()
        attention_mask_cpu = encoded["attention_mask"].detach().cpu()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_delta = torch.zeros(
            (*encoded["input_ids"].shape, int(batch_deltas[0].shape[1])),
            device=device,
            dtype=next(bundle.model.parameters()).dtype,
        )
        decode_delta_rows: list[np.ndarray] = []
        for row_index, delta_h in enumerate(batch_deltas):
            valid = _valid_token_mask(input_ids_cpu[row_index], attention_mask_cpu[row_index], special_ids=special_ids)
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
            if len(valid_indices) != len(delta_h):
                if len(valid_indices) < len(delta_h):
                    delta_h = delta_h[-len(valid_indices) :]
                else:
                    valid_indices = valid_indices[-len(delta_h) :]
            delta_t = torch.as_tensor(delta_h, device=device, dtype=prompt_delta.dtype)
            prompt_delta[row_index, valid_indices, :] = delta_t
            decode_delta_rows.append(np.asarray(delta_h, dtype=np.float32).mean(axis=0))
        decode_delta = torch.as_tensor(
            np.vstack(decode_delta_rows),
            device=device,
            dtype=prompt_delta.dtype,
        )[:, None, :]
        prefill_applied = False

        def apply_delta(hidden_states: torch.Tensor) -> torch.Tensor:
            nonlocal prefill_applied
            if hidden_states.shape[:2] == prompt_delta.shape[:2] and not prefill_applied:
                prefill_applied = True
                return hidden_states + prompt_delta.to(dtype=hidden_states.dtype)
            if apply_on == "prompt_and_decode_mean" and prefill_applied:
                return hidden_states + decode_delta.to(dtype=hidden_states.dtype).expand(
                    hidden_states.shape[0], hidden_states.shape[1], -1
                )
            return hidden_states

        def output_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden_states = apply_delta(output[0].clone())
                return (hidden_states,) + output[1:]
            return apply_delta(output.clone())

        handle = target_layer.register_forward_hook(output_hook)
        try:
            generate_kwargs = _build_generate_kwargs(generation_cfg, return_entropy=False)
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
            with torch.no_grad():
                generation_output = bundle.model.generate(**encoded, **generate_kwargs)
        finally:
            handle.remove()

        prompt_length = encoded["input_ids"].shape[1]
        for row_index in range(len(rendered)):
            generated_ids = generation_output.sequences[row_index, prompt_length:]
            responses.append(tokenizer.decode(generated_ids.detach().cpu(), skip_special_tokens=True).strip())
        del encoded, generation_output, prompt_delta, decode_delta
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return responses


def _run_dataset(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> None:
    seed = int(config["seed"])
    source_root = Path(args.source_root).resolve()
    artifact_root = ensure_dir(Path(args.artifact_root).resolve())
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(artifact_root / dataset / model_slug)
    behavior = _load_behavior(config, dataset, source_root)
    if args.behavior_limit_per_label is not None:
        capped_parts: list[pd.DataFrame] = []
        for label_value, label_df in behavior.groupby("behavior_label", sort=True):
            n_rows = min(int(args.behavior_limit_per_label), len(label_df))
            capped_parts.append(label_df.sample(n=n_rows, random_state=seed + int(label_value)).reset_index(drop=True))
        behavior = pd.concat(capped_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[{dataset}] behavior rows={len(behavior)} D+={int(behavior['behavior_label'].sum())} "
        f"D-={int((1 - behavior['behavior_label']).sum())}",
        flush=True,
    )
    cloud_df, feature_df, reducers, layers = _build_or_load_features(
        config=config,
        dataset=dataset,
        behavior=behavior,
        artifact_root=artifact_root,
        args=args,
        seed=seed,
    )
    last_layer = int(layers[-1])
    steering_layer = int(last_layer if args.steering_layer is None else args.steering_layer)
    if steering_layer not in set(int(layer) for layer in layers):
        raise ValueError(f"Requested --steering-layer {steering_layer} is unavailable; layers={layers}")
    wide_df, all_columns, last_layer_columns = _wide_features(feature_df, layers=layers)
    diagnostics = _train_classifier_diagnostics(
        wide_df=wide_df,
        all_columns=all_columns,
        last_layer_columns=last_layer_columns,
        output_root=output_root,
        seed=seed,
    )
    surrogate, surrogate_metrics = _train_or_load_surrogate(
        cloud_df=cloud_df,
        feature_df=feature_df,
        layer=steering_layer,
        output_root=output_root,
        args=args,
        seed=seed,
    )
    eval_n = int(args.eval_n or config["steering"].get("eval_direct_answer_n", 500))
    neighbor_df = _nearest_targets(
        wide_df=wide_df,
        behavior=behavior,
        all_columns=all_columns,
        feature_df=feature_df,
        layer=steering_layer,
        eval_n=eval_n,
        seed=seed,
        limit=args.limit,
    )
    write_parquet(neighbor_df, output_root / f"{dataset}__topology3_neighbors.parquet")
    neighbor_df.to_csv(output_root / f"{dataset}__topology3_neighbors.csv", index=False)

    cloud_by_id = (
        cloud_df.loc[cloud_df["layer"].eq(steering_layer), ["example_id", "cloud", "hidden_fro_norm"]]
        .drop_duplicates("example_id")
        .set_index("example_id")
    )
    current_clouds = [np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32) for example_id in neighbor_df["example_id"].astype(str)]
    target_clouds = [
        np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32)
        for example_id in neighbor_df["target_example_id"].astype(str)
    ]
    current_features = neighbor_df.loc[:, [f"current__{feature}" for feature in FEATURES]].to_numpy(dtype=np.float32)
    target_features = neighbor_df.loc[:, [f"target__{feature}" for feature in FEATURES]].to_numpy(dtype=np.float32)
    reducer = reducers[steering_layer]
    components = reducer.components_[: int(args.pca_components)].astype(np.float32, copy=False)
    target_mean = np.asarray(surrogate_metrics.get("target_mean", surrogate_metrics.get("metrics", {}) or []), dtype=np.float32)
    if target_mean.size == 0:
        checkpoint = torch.load(output_root / f"layer_{steering_layer:02d}__topology3_surrogate.pt", map_location="cpu")
        target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    else:
        target_std = np.asarray(surrogate_metrics["target_std"], dtype=np.float32)

    bundle = load_hf_model(config["model"], config["generation"])
    try:
        for alpha in args.alphas:
            for lambda_value in args.lambdas:
                run = _run_slug(float(alpha), float(lambda_value))
                run_root = ensure_dir(output_root / run)
                raw_path = run_root / f"{dataset}__topology3_surrogate__raw.parquet"
                if raw_path.exists() and not args.force_generate:
                    print(f"[{dataset} {run}] raw exists: {raw_path}", flush=True)
                    continue
                z_target = current_features + float(alpha) * (target_features - current_features)
                delta_y, opt_df = _optimize_topology_deltas(
                    model=surrogate,
                    current_clouds=current_clouds,
                    target_clouds=target_clouds,
                    target_features=z_target,
                    target_mean=target_mean,
                    target_std=target_std,
                    lambda_value=float(lambda_value),
                    steps=int(args.opt_steps),
                    lr=float(args.opt_lr),
                    batch_size=int(args.opt_batch_size),
                    parallel_jobs=int(args.parallel_jobs),
                    delta_mode=str(args.delta_mode),
                )
                delta_h = [(delta @ components).astype(np.float32, copy=False) for delta in delta_y]
                delta_h_norm = np.asarray([np.linalg.norm(delta) for delta in delta_h], dtype=float)
                hidden_norm = np.asarray(
                    [float(cloud_by_id.loc[example_id, "hidden_fro_norm"]) for example_id in neighbor_df["example_id"].astype(str)],
                    dtype=float,
                )
                responses = _generate_with_token_deltas(
                    bundle=bundle,
                    config=config,
                    rows_df=neighbor_df,
                    token_deltas_h=delta_h,
                    layer=steering_layer,
                    max_length=int(args.max_length),
                    apply_on=str(args.apply_on),
                )
                raw_df = neighbor_df.copy()
                raw_df["prompt_text"] = _prompt_texts(config, raw_df["text"].astype(str).tolist())
                raw_df["response_text"] = responses
                raw_df["strategy"] = "topology3_surrogate"
                raw_df["alpha"] = float(alpha)
                raw_df["lambda"] = float(lambda_value)
                raw_df["layer"] = int(steering_layer)
                raw_df["last_layer"] = int(last_layer)
                raw_df["apply_on"] = str(args.apply_on)
                raw_df["delta_mode"] = str(args.delta_mode)
                raw_df["hidden_delta_norm"] = delta_h_norm
                raw_df["hidden_state_norm"] = hidden_norm
                raw_df["relative_hidden_delta_norm"] = delta_h_norm / np.maximum(hidden_norm, 1e-12)
                raw_df = pd.concat([raw_df.reset_index(drop=True), opt_df.drop(columns=["row_index"]).reset_index(drop=True)], axis=1)
                for feature_index, feature in enumerate(FEATURES):
                    raw_df[f"z_target__{feature}"] = z_target[:, feature_index].astype(float)
                write_parquet(raw_df, raw_path)
                raw_df.to_csv(raw_path.with_suffix(".csv"), index=False)
                summary = {
                    "dataset": dataset,
                    "run": run,
                    "alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "n_eval": int(len(raw_df)),
                    "layer": int(steering_layer),
                    "last_layer": int(last_layer),
                    "apply_on": str(args.apply_on),
                    "delta_mode": str(args.delta_mode),
                    "relative_hidden_delta_norm_mean": float(raw_df["relative_hidden_delta_norm"].mean()),
                    "relative_hidden_delta_norm_median": float(raw_df["relative_hidden_delta_norm"].median()),
                    "surrogate_target_l2_error_mean": float(raw_df["surrogate_target_l2_error"].mean()),
                    "surrogate_target_l2_error_normalized_mean": float(
                        raw_df["surrogate_target_l2_error_normalized"].mean()
                    ),
                    "exact_target_l2_error_mean": float(raw_df["exact_target_l2_error"].mean()),
                    "exact_target_l2_error_normalized_mean": float(raw_df["exact_target_l2_error_normalized"].mean()),
                    "topology_loss_space": "standardized_feature_space",
                }
                write_json(run_root / f"{dataset}__topology3_surrogate_summary.json", summary)
    finally:
        del bundle
        _release()
    metadata = {
        "dataset": dataset,
        "model": config["model"]["name"],
        "features": list(FEATURES),
        "layers": [int(layer) for layer in layers],
        "last_layer": int(last_layer),
        "steering_layer": int(steering_layer),
        "pca_components": int(args.pca_components),
        "apply_on": str(args.apply_on),
        "delta_mode": str(args.delta_mode),
        "classifier_diagnostics": diagnostics,
        # Loaded checkpoints include tensor-valued state_dict entries; metadata
        # only needs scalar/list diagnostics.
        "surrogate_metrics": {key: value for key, value in surrogate_metrics.items() if key != "state_dict"},
    }
    write_json(output_root / f"{dataset}__topology3_metadata.json", metadata)
    print(f"[{dataset}] topology3 surrogate steering complete", flush=True)


def main() -> None:
    args = _parse_args()
    set_global_seed(0)
    config = load_config(args.config)
    for dataset in args.datasets:
        _run_dataset(args, config, dataset)


if __name__ == "__main__":
    main()
