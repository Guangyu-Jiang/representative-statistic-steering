"""CLAMBER H0 sweep using per-layer PCA dimensions from retained variance.

This reuses the dense PCA-prefix H0 tensors produced by
``clamber_h0_pca_dense_prefix_sweep``. For each retained-variance threshold,
each layer gets its own PCA prefix length, then the classifier uses one
``h0_mean_persistence`` feature per layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import warnings

from aen_replication.config import load_config
from aen_replication.eval.clamber_h0_pca_dense_prefix_sweep import DEFAULT_OUTPUT_DIR
from aen_replication.eval.clamber_h0_pca_dim_sweep import DEFAULT_CONFIGS
from aen_replication.train.clamber_subclass_classification import _evaluate_multiclass
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json


DEFAULT_OUTPUT_SUBDIR = "variance_threshold_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--dense-prefix-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=str(Path(DEFAULT_OUTPUT_DIR) / DEFAULT_OUTPUT_SUBDIR))
    parser.add_argument("--thresholds", nargs="+", type=int, default=list(range(1, 100)))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_model_root(dense_prefix_dir: Path, model_name: str) -> tuple[Path, int]:
    model_dir = dense_prefix_dir / slugify(model_name)
    candidates = sorted(model_dir.glob("pca_prefix_*"))
    candidates = [path for path in candidates if (path / "h0_tensor_float32.dat").exists()]
    if not candidates:
        raise FileNotFoundError(f"No dense-prefix H0 tensor found under {model_dir}")
    root = candidates[-1]
    max_dim = int(root.name.rsplit("_", 1)[-1])
    return root, max_dim


def _reducers_path(dense_prefix_dir: Path, model_name: str, max_dim: int) -> Path:
    path = dense_prefix_dir / slugify(model_name) / f"pca_reducers_max{int(max_dim):03d}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA reducers: {path}")
    return path


def _dimension_for_threshold(reducer: Any, threshold: float) -> int:
    cumulative = np.cumsum(np.asarray(reducer.explained_variance_ratio_, dtype=np.float64))
    if cumulative.size == 0:
        raise ValueError("Reducer has no explained_variance_ratio_.")
    dim = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return max(1, min(dim, int(cumulative.size)))


def _select_layer_dimensions(reducers: dict[int, Any], layers: list[int], threshold_percent: int) -> dict[int, int]:
    threshold = float(threshold_percent) / 100.0
    return {int(layer): _dimension_for_threshold(reducers[int(layer)], threshold) for layer in layers}


def _feature_matrix_for_dims(tensor: np.memmap, layers: list[int], dims_by_layer: dict[int, int]) -> np.ndarray:
    columns = [np.asarray(tensor[:, int(layer), int(dims_by_layer[int(layer)]) - 1], dtype=np.float64) for layer in layers]
    x = np.column_stack(columns)
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    return x


def evaluate_model(
    *,
    config: dict[str, Any],
    dense_prefix_dir: Path,
    output_dir: Path,
    thresholds: list[int],
    seed: int,
    force: bool,
) -> tuple[Path, Path]:
    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    metrics_path = output_dir / model_slug / "variance_threshold_metrics.csv"
    dims_path = output_dir / model_slug / "variance_threshold_layer_dims.csv"
    if metrics_path.exists() and dims_path.exists() and not force:
        return metrics_path, dims_path

    model_root, max_dim = _find_model_root(dense_prefix_dir, model_name)
    projection_meta = _read_json(model_root / "projection_layout.metadata.json")
    layers = [int(layer) for layer in projection_meta["layers"]]
    examples = pd.read_parquet(model_root / "examples.parquet")
    reducers = joblib.load(_reducers_path(dense_prefix_dir, model_name, max_dim))
    tensor = np.memmap(
        model_root / "h0_tensor_float32.dat",
        mode="r",
        dtype=np.float32,
        shape=(int(len(examples)), max(layers) + 1, int(max_dim)),
    )

    subclass_cfg = dict(config["clamber_subclass_classification"])
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    train_mask = examples["split"].astype(str).eq("train").to_numpy()
    test_mask = examples["split"].astype(str).eq("test").to_numpy()
    y_train = examples.loc[train_mask, "subclass"].astype(str).to_numpy()
    y_test = examples.loc[test_mask, "subclass"].astype(str).to_numpy()

    metric_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []
    for threshold_percent in tqdm(thresholds, desc=f"{model_slug}_variance_thresholds", unit="threshold"):
        dims_by_layer = _select_layer_dimensions(reducers, layers, int(threshold_percent))
        x = _feature_matrix_for_dims(tensor, layers, dims_by_layer)
        payload = _evaluate_multiclass(
            x_train=x[train_mask],
            y_train=y_train,
            x_eval=x[test_mask],
            y_eval=y_test,
            max_iter=max_iter,
            c_value=c_value,
            seed=int(seed) + int(threshold_percent),
        )
        dims = np.asarray([dims_by_layer[layer] for layer in layers], dtype=int)
        metric_rows.append(
            {
                "model": model_slug,
                "model_name": model_name,
                "dataset": "clamber",
                "label_space": "9_subclasses",
                "method": "h0_mean_persistence_retained_variance_all_layers",
                "retained_variance_percent": int(threshold_percent),
                "layer": "all",
                "feature_count": int(len(layers)),
                "min_pca_dim": int(dims.min()),
                "median_pca_dim": float(np.median(dims)),
                "mean_pca_dim": float(np.mean(dims)),
                "max_pca_dim": int(dims.max()),
                "accuracy": float(payload["metrics"]["accuracy"]),
                "macro_f1": float(payload["metrics"]["macro_f1"]),
                "source_dense_prefix_dir": str(model_root),
            }
        )
        for layer in layers:
            dim_rows.append(
                {
                    "model": model_slug,
                    "model_name": model_name,
                    "retained_variance_percent": int(threshold_percent),
                    "layer": int(layer),
                    "pca_dim": int(dims_by_layer[layer]),
                }
            )

    ensure_dir(metrics_path.parent)
    metrics_df = pd.DataFrame(metric_rows).sort_values("retained_variance_percent").reset_index(drop=True)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(dim_rows).sort_values(["retained_variance_percent", "layer"]).to_csv(dims_path, index=False)
    best_rows = []
    for key in ["accuracy", "macro_f1"]:
        best = metrics_df.sort_values([key, "macro_f1" if key == "accuracy" else "accuracy"], ascending=False).iloc[0].to_dict()
        best["best_by"] = key
        best_rows.append(best)
    best_path = output_dir / model_slug / "variance_threshold_best_metrics.csv"
    pd.DataFrame(best_rows).to_csv(best_path, index=False)
    write_json(
        output_dir / model_slug / "variance_threshold_metadata.json",
        {
            "model_name": model_name,
            "dense_prefix_root": str(model_root),
            "reducers_path": str(_reducers_path(dense_prefix_dir, model_name, max_dim)),
            "thresholds": [int(value) for value in thresholds],
            "max_dim": int(max_dim),
            "layers": layers,
            "metrics_path": str(metrics_path),
            "dims_path": str(dims_path),
            "best_metrics_path": str(best_path),
        },
    )
    return metrics_path, dims_path


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = ensure_dir(Path(args.output_dir))
    dense_prefix_dir = Path(args.dense_prefix_dir)
    thresholds = sorted({int(value) for value in args.thresholds})
    if not thresholds or thresholds[0] < 1 or thresholds[-1] > 99:
        raise ValueError("This sweep expects retained-variance thresholds from 1 to 99 percent.")

    metrics_paths = []
    dims_paths = []
    for config_path in args.configs:
        metrics_path, dims_path = evaluate_model(
            config=load_config(config_path),
            dense_prefix_dir=dense_prefix_dir,
            output_dir=output_dir,
            thresholds=thresholds,
            seed=int(args.seed),
            force=bool(args.force),
        )
        metrics_paths.append(metrics_path)
        dims_paths.append(dims_path)

    combined = pd.concat([pd.read_csv(path) for path in metrics_paths], ignore_index=True)
    combined_path = output_dir / "variance_threshold_metrics_combined.csv"
    combined.to_csv(combined_path, index=False)
    combined_best = pd.concat(
        [pd.read_csv(path.parent / "variance_threshold_best_metrics.csv") for path in metrics_paths],
        ignore_index=True,
    )
    combined_best_path = output_dir / "variance_threshold_best_metrics_combined.csv"
    combined_best.to_csv(combined_best_path, index=False)
    combined_dims = pd.concat([pd.read_csv(path) for path in dims_paths], ignore_index=True)
    combined_dims_path = output_dir / "variance_threshold_layer_dims_combined.csv"
    combined_dims.to_csv(combined_dims_path, index=False)
    write_json(
        output_dir / "variance_threshold_outputs.json",
        {
            "metrics": str(combined_path),
            "best_metrics": str(combined_best_path),
            "layer_dims": str(combined_dims_path),
        },
    )
    print(f"Wrote {combined_path}")
    print(f"Wrote {combined_best_path}")
    print(f"Wrote {combined_dims_path}")


if __name__ == "__main__":
    main()
