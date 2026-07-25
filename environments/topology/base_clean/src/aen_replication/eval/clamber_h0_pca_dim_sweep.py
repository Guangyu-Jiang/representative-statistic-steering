"""Sweep PCA dimensions for all-layer CLAMBER H0 mean persistence classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import warnings

from aen_replication.config import load_config
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.clamber_subclass_classification import _evaluate_multiclass
from aen_replication.train.token_cloud_topology_classifier import (
    _extract_train_token_matrices,
    _prepare_prompt_frame,
    _valid_token_mask,
)
from aen_replication.utils.io_utils import ensure_dir, read_json, slugify, write_json


DEFAULT_CONFIGS = [
    "configs/runs/gemma_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
    "configs/runs/llama_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
    "configs/runs/mistral_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
]
DEFAULT_DIMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--output-dir", default="artifacts/reports/clamber_h0_pca_dim_sweep")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pca-fit-token-cap", type=int, default=None)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size].copy()


def _h0_mean(cloud: np.ndarray) -> float:
    points = np.asarray(cloud, dtype=np.float32)
    if points.ndim != 2 or len(points) <= 1:
        return 0.0
    distances = squareform(pdist(points, metric="euclidean"))
    weights = np.asarray(minimum_spanning_tree(distances).data, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.size == 0:
        return 0.0
    return float(weights.mean())


def _feature_path(output_dir: Path, model_name: str, dim: int) -> Path:
    return output_dir / slugify(model_name) / f"clamber_h0_mean_persistence_pca{int(dim):03d}_all_layers.parquet"


def _reducers_path(output_dir: Path, model_name: str, max_dim: int) -> Path:
    return output_dir / slugify(model_name) / f"pca_reducers_max{int(max_dim):03d}.joblib"


def _cached_forward_path(config: dict[str, Any], model_name: str) -> Path:
    output_root = Path(config["clamber_subclass_classification"]["output_dir"]) / slugify(model_name)
    return output_root / "token_cloud_forward_cache" / slugify(model_name) / "clamber_token_cloud_forward_cache.joblib"


def _load_cached_cloud_frame(path: Path) -> tuple[pd.DataFrame, int]:
    payload = joblib.load(path)
    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
    elif isinstance(payload, dict) and "cloud_df" in payload:
        df = pd.DataFrame(payload["cloud_df"])
    else:
        raise ValueError(f"Unsupported cached cloud payload: {path}")
    if df.empty:
        raise ValueError(f"Empty cached cloud frame: {path}")
    first_cloud = np.asarray(df.iloc[0]["cloud"])
    if first_cloud.ndim != 2:
        raise ValueError(f"Unexpected cached cloud shape in {path}: {first_cloud.shape}")
    return df, int(first_cloud.shape[1])


def _compute_feature_from_cached_clouds(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dims: list[int],
    force_recompute: bool,
) -> tuple[set[int], list[dict[str, Any]]]:
    model_name = str(config["model"]["name"])
    cache_path = _cached_forward_path(config, model_name)
    if not cache_path.exists():
        return set(), []
    cloud_df, cached_dim = _load_cached_cloud_frame(cache_path)
    available_dims = [dim for dim in dims if dim <= cached_dim]
    missing = [dim for dim in available_dims if force_recompute or not _feature_path(output_dir, model_name, dim).exists()]
    cache_rows: list[dict[str, Any]] = []
    if not missing:
        for dim in available_dims:
            cache_rows.append(
                {
                    "model": slugify(model_name),
                    "pca_dim": int(dim),
                    "feature_cache_path": str(_feature_path(output_dir, model_name, dim)),
                    "feature_cache_reused": True,
                    "source": str(cache_path),
                    "source_cached_cloud_dim": int(cached_dim),
                }
            )
        return set(available_dims), cache_rows

    records = cloud_df.loc[
        :,
        ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "layer", "token_count", "cloud"],
    ].to_dict(orient="records")
    by_dim: dict[int, list[dict[str, Any]]] = {dim: [] for dim in missing}
    for record in tqdm(records, desc=f"{slugify(model_name)}_cached_pca_h0", unit="cloud"):
        cloud = np.asarray(record["cloud"], dtype=np.float32)
        base = {
            "example_id": str(record["example_id"]),
            "pair_id": str(record["pair_id"]),
            "dataset": str(record["dataset"]),
            "split": str(record["split"]),
            "label_ambiguous": int(record["label_ambiguous"]),
            "layer": int(record["layer"]),
            "token_count": int(record["token_count"]),
        }
        for dim in missing:
            row = dict(base)
            row["h0_mean_persistence"] = np.float32(_h0_mean(cloud[:, :dim]))
            by_dim[dim].append(row)

    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    subclass_lookup = pd.read_parquet(dataset_path).loc[:, ["example_id", "subclass"]].drop_duplicates()
    for dim, rows in by_dim.items():
        feature_df = pd.DataFrame(rows).merge(subclass_lookup, on="example_id", how="left")
        path = _feature_path(output_dir, model_name, dim)
        ensure_dir(path.parent)
        feature_df.to_parquet(path, index=False)
        write_json(
            path.with_suffix(".metadata.json"),
            {
                "model_name": model_name,
                "pca_dim": int(dim),
                "source_cached_cloud_path": str(cache_path),
                "source_cached_cloud_dim": int(cached_dim),
                "rows": int(len(feature_df)),
                "pca_reused_from_cached_cloud": True,
            },
        )

    for dim in available_dims:
        cache_rows.append(
            {
                "model": slugify(model_name),
                "pca_dim": int(dim),
                "feature_cache_path": str(_feature_path(output_dir, model_name, dim)),
                "feature_cache_reused": dim not in missing,
                "source": str(cache_path),
                "source_cached_cloud_dim": int(cached_dim),
            }
        )
    return set(available_dims), cache_rows


def _fit_or_load_reducers(
    *,
    config: dict[str, Any],
    output_dir: Path,
    model_name: str,
    max_dim: int,
    force_recompute: bool,
) -> tuple[dict[int, PCA], bool, Any, pd.DataFrame, str, list[int]]:
    path = _reducers_path(output_dir, model_name, max_dim)
    classifier_config = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    token_cfg = {
        **classifier_config,
        "batch_size": int(subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8))),
        "max_length": int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64))),
        "pca_components": int(max_dim),
        "topology_components": int(max_dim),
        "use_pca": True,
        "_seed": int(config.get("seed", 13)),
    }
    reducers_reused = path.exists() and not force_recompute
    if reducers_reused:
        reducers = joblib.load(path)
        bundle = load_hf_model(config["model"], classifier_config)
    else:
        bundle = load_hf_model(config["model"], classifier_config)
        total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
        layers = list(range(total_layers))
        dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
        dataset_df = pd.read_parquet(dataset_path).copy()
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=str(classifier_config.get("text_column", "text")),
            use_chat_template=bool(classifier_config.get("use_chat_template", False)),
            system_prompt=classifier_config.get("system_prompt"),
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column].astype(str)
        train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
        if token_cfg.get("pca_fit_token_cap") is None:
            token_cfg["pca_fit_token_cap"] = int(classifier_config.get("pca_fit_token_cap", 16000))
        token_matrices = _extract_train_token_matrices(
            bundle=bundle,
            train_df=train_df,
            text_column="_token_cloud_text",
            layers=layers,
            config=token_cfg,
        )
        reducers = {}
        for layer, matrix in tqdm(token_matrices.items(), desc=f"{slugify(model_name)}_fit_pca{max_dim}"):
            n_components = min(int(max_dim), int(matrix.shape[1]), max(1, int(matrix.shape[0]) - 1))
            svd_solver = "covariance_eigh" if n_components >= int(matrix.shape[1]) else "randomized"
            reducer = PCA(
                n_components=n_components,
                svd_solver=svd_solver,
                random_state=int(config.get("seed", 13)) + int(layer),
                whiten=bool(classifier_config.get("pca_whiten", False)),
            )
            reducer.fit(matrix)
            reducers[int(layer)] = reducer
        ensure_dir(path.parent)
        joblib.dump(reducers, path)
        write_json(
            path.with_suffix(".metadata.json"),
            {
                "model_name": model_name,
                "max_dim": int(max_dim),
                "layers": sorted(int(layer) for layer in reducers),
                "pca_fit_token_cap": int(token_cfg.get("pca_fit_token_cap", 16000)),
                "svd_solver": "covariance_eigh" if int(max_dim) >= int(token_matrices[layers[0]].shape[1]) else "randomized",
            },
        )

    total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = list(range(total_layers))
    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    dataset_df = pd.read_parquet(dataset_path).copy()
    prepared_df, prepared_text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle,
        text_column=str(classifier_config.get("text_column", "text")),
        use_chat_template=bool(classifier_config.get("use_chat_template", False)),
        system_prompt=classifier_config.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column].astype(str)
    return reducers, reducers_reused, bundle, prepared_df, "_token_cloud_text", layers


def _compute_features_from_model(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dims: list[int],
    batch_size_override: int | None,
    pca_fit_token_cap: int | None,
    force_recompute: bool,
) -> list[dict[str, Any]]:
    model_name = str(config["model"]["name"])
    pending_dims = [dim for dim in dims if force_recompute or not _feature_path(output_dir, model_name, dim).exists()]
    if not pending_dims:
        return [
            {
                "model": slugify(model_name),
                "pca_dim": int(dim),
                "feature_cache_path": str(_feature_path(output_dir, model_name, dim)),
                "feature_cache_reused": True,
                "source": "pca_sweep_feature_cache",
                "source_cached_cloud_dim": None,
            }
            for dim in dims
        ]

    max_dim = max(pending_dims)
    if pca_fit_token_cap is not None:
        config = dict(config)
        config["token_cloud_topology_classifier"] = dict(config["token_cloud_topology_classifier"])
        config["token_cloud_topology_classifier"]["pca_fit_token_cap"] = int(pca_fit_token_cap)
    reducers, reducers_reused, bundle, prepared_df, text_column, layers = _fit_or_load_reducers(
        config=config,
        output_dir=output_dir,
        model_name=model_name,
        max_dim=max_dim,
        force_recompute=force_recompute,
    )
    classifier_config = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    batch_size = int(batch_size_override or subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8)))
    max_length = int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64)))
    drop_special_tokens = bool(classifier_config.get("drop_special_tokens", True))
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    by_dim: dict[int, list[dict[str, Any]]] = {dim: [] for dim in pending_dims}

    progress = tqdm(total=len(prepared_df), desc=f"{slugify(model_name)}_pca_sweep_h0", unit="example")
    try:
        for batch_df in _iter_batches(prepared_df.reset_index(drop=True), batch_size):
            encoded = tokenizer(
                batch_df[text_column].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states for PCA sweep.")
            input_ids_cpu = input_ids.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu()
            records = batch_df.reset_index(drop=True).to_dict(orient="records")
            valid_masks = [
                _valid_token_mask(
                    input_ids_cpu[row_index],
                    attention_mask_cpu[row_index],
                    special_ids=special_ids,
                    drop_special_tokens=drop_special_tokens,
                )
                for row_index in range(len(records))
            ]
            for layer in layers:
                reducer = reducers[int(layer)]
                layer_output = hidden_states[layer + 1].detach().float().cpu()
                token_chunks: list[np.ndarray] = []
                chunk_meta: list[tuple[dict[str, Any], int]] = []
                for row_index, row in enumerate(records):
                    token_vectors = layer_output[row_index][valid_masks[row_index]].numpy()
                    if token_vectors.size == 0:
                        continue
                    token_chunks.append(token_vectors)
                    chunk_meta.append((row, len(token_vectors)))
                if not token_chunks:
                    continue
                token_matrix = np.vstack(token_chunks)
                reduced_matrix = reducer.transform(token_matrix).astype(np.float32, copy=False)
                offset = 0
                for row, token_count in chunk_meta:
                    reduced = reduced_matrix[offset : offset + token_count]
                    offset += token_count
                    base = {
                        "example_id": str(row["example_id"]),
                        "pair_id": str(row["pair_id"]),
                        "dataset": str(row["dataset"]),
                        "split": str(row["split"]),
                        "label_ambiguous": int(row["label_ambiguous"]),
                        "subclass": str(row["subclass"]),
                        "layer": int(layer),
                        "token_count": int(token_count),
                    }
                    for dim in pending_dims:
                        row_out = dict(base)
                        row_out["h0_mean_persistence"] = np.float32(_h0_mean(reduced[:, :dim]))
                        by_dim[dim].append(row_out)
            del outputs, hidden_states, model_inputs, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update(len(records))
    finally:
        progress.close()

    for dim, rows in by_dim.items():
        path = _feature_path(output_dir, model_name, dim)
        ensure_dir(path.parent)
        feature_df = pd.DataFrame(rows).sort_values(["split", "example_id", "layer"]).reset_index(drop=True)
        feature_df.to_parquet(path, index=False)
        write_json(
            path.with_suffix(".metadata.json"),
            {
                "model_name": model_name,
                "pca_dim": int(dim),
                "max_fit_dim": int(max_dim),
                "pca_reducer_cache": str(_reducers_path(output_dir, model_name, max_dim)),
                "pca_reducer_cache_reused": bool(reducers_reused),
                "rows": int(len(feature_df)),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
            },
        )

    return [
        {
            "model": slugify(model_name),
            "pca_dim": int(dim),
            "feature_cache_path": str(_feature_path(output_dir, model_name, dim)),
            "feature_cache_reused": dim not in pending_dims,
            "source": str(_reducers_path(output_dir, model_name, max_dim)),
            "source_cached_cloud_dim": None,
        }
        for dim in dims
    ]


def _wide_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    index_columns = ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "subclass"]
    wide = (
        feature_df.pivot_table(
            index=index_columns,
            columns="layer",
            values="h0_mean_persistence",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    layer_columns = sorted(column for column in wide.columns if isinstance(column, (int, np.integer)))
    rename_map = {layer: f"h0_mean_persistence__l{int(layer):02d}" for layer in layer_columns}
    wide = wide.rename(columns=rename_map)
    return wide, [rename_map[layer] for layer in layer_columns]


def _evaluate_dim(
    *,
    feature_df: pd.DataFrame,
    model_name: str,
    dim: int,
    seed: int,
    subclass_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    wide_df, feature_columns = _wide_frame(feature_df)
    train_df = wide_df.loc[wide_df["split"].eq("train")].copy()
    test_df = wide_df.loc[wide_df["split"].eq("test")].copy()
    payload = _evaluate_multiclass(
        x_train=train_df.loc[:, feature_columns].to_numpy(dtype=float),
        y_train=train_df["subclass"].astype(str).to_numpy(),
        x_eval=test_df.loc[:, feature_columns].to_numpy(dtype=float),
        y_eval=test_df["subclass"].astype(str).to_numpy(),
        max_iter=int(subclass_cfg.get("max_iter", 4000)),
        c_value=float(subclass_cfg.get("token_cloud_classifier_C", 1.0)),
        seed=seed + int(dim),
    )
    metrics = payload["metrics"]
    coefficients = np.asarray(payload["classifier"].coef_, dtype=float)
    importance = pd.DataFrame(
        {
            "model": slugify(model_name),
            "model_name": model_name,
            "pca_dim": int(dim),
            "feature": feature_columns,
            "layer": [int(column.rsplit("l", 1)[1]) for column in feature_columns],
            "mean_abs_coefficient": np.mean(np.abs(coefficients), axis=0),
            "max_abs_coefficient": np.max(np.abs(coefficients), axis=0),
        }
    )
    return (
        {
            "model": slugify(model_name),
            "model_name": model_name,
            "dataset": "clamber",
            "label_space": "9_subclasses",
            "method": "h0_mean_persistence_pca_dim_sweep_all_layers",
            "pca_dim": int(dim),
            "layer": "all",
            "selection_signature": " | ".join(str(index) for index in range(len(feature_columns))),
            "feature_count": int(len(feature_columns)),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "test_confusion_matrix": metrics["confusion_matrix"],
            "test_labels": metrics["labels"],
        },
        importance.sort_values("mean_abs_coefficient", ascending=False).reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = ensure_dir(Path(args.output_dir))
    dims = sorted(set(int(dim) for dim in args.dims))
    metric_rows: list[dict[str, Any]] = []
    importance_parts: list[pd.DataFrame] = []
    cache_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        cached_dims, cached_rows = _compute_feature_from_cached_clouds(
            config=config,
            output_dir=output_dir,
            dims=dims,
            force_recompute=bool(args.force_recompute),
        )
        cache_rows.extend(cached_rows)
        remaining_dims = [dim for dim in dims if dim not in cached_dims]
        if remaining_dims and args.cached_only:
            for dim in remaining_dims:
                feature_path = _feature_path(output_dir, model_name, dim)
                feature_exists = feature_path.exists()
                cache_rows.append(
                    {
                        "model": slugify(model_name),
                        "pca_dim": int(dim),
                        "feature_cache_path": str(feature_path),
                        "feature_cache_reused": bool(feature_exists),
                        "source": "pca_sweep_feature_cache" if feature_exists else "not_computed_cached_only",
                        "source_cached_cloud_dim": None,
                    }
                )
        elif remaining_dims:
            cache_rows.extend(
                _compute_features_from_model(
                    config=config,
                    output_dir=output_dir,
                    dims=remaining_dims,
                    batch_size_override=args.batch_size,
                    pca_fit_token_cap=args.pca_fit_token_cap,
                    force_recompute=bool(args.force_recompute),
                )
            )

        for dim in dims:
            path = _feature_path(output_dir, model_name, dim)
            if not path.exists():
                continue
            feature_df = pd.read_parquet(path)
            row, importance = _evaluate_dim(
                feature_df=feature_df,
                model_name=model_name,
                dim=dim,
                seed=int(args.seed),
                subclass_cfg=dict(config["clamber_subclass_classification"]),
            )
            metric_rows.append(row)
            importance_parts.append(importance)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "pca_dim"]).reset_index(drop=True)
    importance_df = (
        pd.concat(importance_parts, ignore_index=True)
        .sort_values(["model", "pca_dim", "mean_abs_coefficient"], ascending=[True, True, False])
        .reset_index(drop=True)
        if importance_parts
        else pd.DataFrame()
    )
    cache_df = pd.DataFrame(cache_rows).sort_values(["model", "pca_dim"]).reset_index(drop=True)

    metrics_path = output_dir / "clamber_h0_pca_dim_sweep_metrics.csv"
    importance_path = output_dir / "clamber_h0_pca_dim_sweep_importance.csv"
    cache_path = output_dir / "clamber_h0_pca_dim_sweep_cache_status.csv"
    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    cache_df.to_csv(cache_path, index=False)
    write_json(
        output_dir / "clamber_h0_pca_dim_sweep_outputs.json",
        {
            "metrics_csv": str(metrics_path),
            "importance_csv": str(importance_path),
            "cache_status_csv": str(cache_path),
        },
    )
    print(metrics_df.to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {importance_path}")
    print(f"Wrote {cache_path}")


if __name__ == "__main__":
    main()
