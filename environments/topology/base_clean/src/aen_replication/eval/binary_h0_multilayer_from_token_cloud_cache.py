"""Evaluate binary all-layer H0 mean persistence from token-cloud forward caches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from aen_replication.config import load_config
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _evaluate_feature_set,
)
from aen_replication.utils.io_utils import ensure_dir, read_json, slugify, write_json


DEFAULT_CONFIGS = [
    "configs/runs/gemma_binary_h0_mean_persistence_alllayers.yaml",
    "configs/runs/mistral_binary_h0_mean_persistence_alllayers.yaml",
    "configs/runs/llama_binary_h0_mean_persistence_alllayers.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="artifacts/reports/binary_h0_mean_persistence")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _token_cloud_model_dir(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _load_forward_cache(path: Path) -> pd.DataFrame:
    payload = joblib.load(path)
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, dict) and "cloud_df" in payload:
        return pd.DataFrame(payload["cloud_df"])
    raise ValueError(f"Unsupported token-cloud forward cache payload: {path}")


def _h0_mean_persistence(cloud: Any, *, metric: str) -> float:
    points = np.asarray(cloud, dtype=np.float32)
    if points.ndim != 2 or len(points) <= 1:
        return 0.0
    distances = squareform(pdist(points, metric=metric))
    weights = np.asarray(minimum_spanning_tree(distances).data, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.size == 0:
        return 0.0
    return float(weights.mean())


def _compute_h0_table(
    *,
    cloud_df: pd.DataFrame,
    metric: str,
    parallel_jobs: int,
) -> pd.DataFrame:
    records = cloud_df.loc[
        :,
        ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "layer", "cloud"],
    ].to_dict(orient="records")

    values = joblib.Parallel(n_jobs=max(1, int(parallel_jobs)), backend="loky")(
        joblib.delayed(_h0_mean_persistence)(record["cloud"], metric=metric) for record in records
    )

    feature_df = pd.DataFrame(
        {
            "example_id": [str(record["example_id"]) for record in records],
            "pair_id": [str(record["pair_id"]) for record in records],
            "dataset": [str(record["dataset"]) for record in records],
            "split": [str(record["split"]) for record in records],
            "label_ambiguous": [int(record["label_ambiguous"]) for record in records],
            "layer": [int(record["layer"]) for record in records],
            "h0_mean_persistence": np.asarray(values, dtype=np.float32),
        }
    )
    return feature_df.sort_values(["dataset", "split", "example_id", "layer"]).reset_index(drop=True)


def _evaluate_all_layer_h0(
    *,
    model_slug: str,
    model_name: str,
    feature_df: pd.DataFrame,
    datasets: list[str],
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    all_layers = sorted(int(layer) for layer in feature_df["layer"].unique().tolist())
    selections = [{"layer": layer, "val_auroc": 0.0} for layer in all_layers]

    for dataset in datasets:
        dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
        if dataset_df.empty:
            continue
        train_df, test_df, meta = _build_multilayer_feature_frames(
            feature_df=dataset_df,
            dataset=dataset,
            selections=selections,
        )
        variants = {
            "h0_mean_persistence_all_layers": list(meta["topology_columns"]),
            "h0_mean_persistence_all_layers_plus_summary": list(meta["topology_columns"])
            + list(meta["topology_summary_columns"]),
        }
        for method, feature_columns in variants.items():
            _, payload = _evaluate_feature_set(
                train_features=train_df,
                eval_features=test_df,
                feature_columns=feature_columns,
                classifier_config=classifier_config,
                seed=seed,
            )
            metrics = payload["eval_metrics"]
            metric_rows.append(
                {
                    "model": model_slug,
                    "model_name": model_name,
                    "dataset": dataset,
                    "method": method,
                    "layer": "all",
                    "selection_signature": " | ".join(str(layer) for layer in all_layers),
                    "feature_count": int(len(feature_columns)),
                    "auroc": float(metrics["auroc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                }
            )
            coefficients = np.asarray(payload["coefficients"], dtype=float).ravel()
            for column, coefficient in zip(feature_columns, coefficients, strict=False):
                coefficient_rows.append(
                    {
                        "model": model_slug,
                        "dataset": dataset,
                        "method": method,
                        "feature": column,
                        "coefficient": float(coefficient),
                        "abs_coefficient": float(abs(coefficient)),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def _baseline_rows(config: dict[str, Any], *, model_slug: str) -> list[dict[str, Any]]:
    model_name = str(config["model"]["name"])
    report_root = Path(config["reports"]["output_dir"]) / slugify(model_name)
    summary = read_json(report_root / "detection_summary.json")
    default_layer = int(summary.get("cross_dataset_overlap", {}).get("default_layer", 14))
    rows: list[dict[str, Any]] = []
    for dataset in ["ambigqa", "situatedqa"]:
        dataset_summary = summary["datasets"][dataset]
        for source_key, method in [
            ("full_probe_test", f"full_probe_l{default_layer}"),
            ("aen_probe_test", f"aen_k{int(dataset_summary['aen_selection']['aen_k'])}_l{default_layer}"),
        ]:
            metrics = dataset_summary[source_key]
            rows.append(
                {
                    "model": model_slug,
                    "model_name": model_name,
                    "dataset": dataset,
                    "method": method,
                    "layer": str(default_layer),
                    "selection_signature": str(default_layer),
                    "feature_count": None if source_key == "full_probe_test" else int(dataset_summary["aen_selection"]["aen_k"]),
                    "auroc": float(metrics["auroc"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    metric_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    cache_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        model_slug = slugify(model_name)
        classifier_config = dict(config["token_cloud_topology_classifier"])
        token_model_dir = _token_cloud_model_dir(model_name)
        token_output_root = Path(classifier_config["output_dir"]) / token_model_dir
        forward_cache_path = token_output_root / "token_cloud_forward_cache.joblib"
        h0_feature_path = token_output_root / "h0_mean_persistence_all_layers.parquet"
        metadata_path = h0_feature_path.with_suffix(".metadata.json")

        if not forward_cache_path.exists():
            raise FileNotFoundError(f"Missing token-cloud forward cache: {forward_cache_path}")

        feature_reused = h0_feature_path.exists() and not args.force_recompute
        if feature_reused:
            feature_df = pd.read_parquet(h0_feature_path).copy()
        else:
            cloud_df = _load_forward_cache(forward_cache_path)
            feature_df = _compute_h0_table(
                cloud_df=cloud_df,
                metric=str(classifier_config.get("distance_metric", "euclidean")),
                parallel_jobs=int(classifier_config.get("parallel_jobs", 1)),
            )
            feature_df.to_parquet(h0_feature_path, index=False)
            write_json(
                metadata_path,
                {
                    "model_name": model_name,
                    "source_forward_cache": str(forward_cache_path),
                    "rows": int(len(feature_df)),
                    "layers": sorted(int(layer) for layer in feature_df["layer"].unique().tolist()),
                    "feature": "h0_mean_persistence",
                    "distance_metric": str(classifier_config.get("distance_metric", "euclidean")),
                },
            )

        datasets = list(classifier_config.get("datasets", ["ambigqa", "situatedqa"]))
        metric_df, coefficient_df = _evaluate_all_layer_h0(
            model_slug=model_slug,
            model_name=model_name,
            feature_df=feature_df,
            datasets=datasets,
            classifier_config=dict(classifier_config.get("classifier", {})),
            seed=int(args.seed),
        )
        metric_parts.append(metric_df)
        coefficient_parts.append(coefficient_df)
        metric_parts.append(pd.DataFrame(_baseline_rows(config, model_slug=model_slug)))
        cache_rows.append(
            {
                "model": model_slug,
                "forward_cache_path": str(forward_cache_path),
                "forward_cache_reused_for_h0_eval": True,
                "h0_feature_cache_path": str(h0_feature_path),
                "h0_feature_cache_reused": bool(feature_reused),
            }
        )

    comparison_df = pd.concat(metric_parts, ignore_index=True)
    comparison_df = comparison_df.sort_values(["model", "dataset", "method"]).reset_index(drop=True)
    coefficients_df = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else pd.DataFrame()
    if not coefficients_df.empty:
        coefficients_df = coefficients_df.sort_values(
            ["model", "dataset", "method", "abs_coefficient"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    cache_df = pd.DataFrame(cache_rows)

    comparison_path = output_dir / "binary_h0_mean_persistence_multilayer_comparison.csv"
    coefficients_path = output_dir / "binary_h0_mean_persistence_multilayer_coefficients.csv"
    cache_path = output_dir / "binary_h0_mean_persistence_multilayer_cache_status.csv"
    comparison_df.to_csv(comparison_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)
    cache_df.to_csv(cache_path, index=False)
    write_json(
        output_dir / "binary_h0_mean_persistence_multilayer_outputs.json",
        {
            "comparison_csv": str(comparison_path),
            "coefficients_csv": str(coefficients_path),
            "cache_status_csv": str(cache_path),
        },
    )

    print(comparison_df.to_string(index=False))
    print(f"\nWrote {comparison_path}")
    print(f"Wrote {coefficients_path}")
    print(f"Wrote {cache_path}")


if __name__ == "__main__":
    main()
