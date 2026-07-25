"""Evaluate binary all-topology token-cloud features from forward caches."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aen_replication.config import load_config
from aen_replication.train.independent_topology_classifier import _group_train_val_split, _selection_order
from aen_replication.train.token_cloud_topology_classifier import (
    BASE_KEY_COLUMNS,
    _build_multilayer_feature_frames,
    _evaluate_feature_set,
    _topology_feature_columns,
    build_token_cloud_feature_frame,
)
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json


DEFAULT_CONFIGS = [
    "configs/runs/gemma_binary_h0_mean_persistence_alllayers.yaml",
    "configs/runs/mistral_binary_h0_mean_persistence_alllayers.yaml",
    "configs/runs/llama_binary_h0_mean_persistence_alllayers.yaml",
]

LAYER_FEATURE_RE = re.compile(r"^(?P<base>.+)__l(?P<layer>[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="artifacts/reports/binary_all_topology_multilayer")
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


def _compute_topology_table(
    *,
    cloud_df: pd.DataFrame,
    classifier_config: dict[str, Any],
) -> pd.DataFrame:
    topology_config = {
        **classifier_config,
        "distance_feature_mode": "none",
    }
    feature_df = build_token_cloud_feature_frame(cloud_df, prototype_map=None, config=topology_config)
    return feature_df.sort_values(["dataset", "split", "example_id", "layer"]).reset_index(drop=True)


def _topology_only_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-topological controls before selection/evaluation."""

    return feature_df.drop(columns=["token_count"], errors="ignore").copy()


def _split_layer_feature_name(feature: str) -> tuple[str, int | None]:
    match = LAYER_FEATURE_RE.match(feature)
    if match is None:
        return feature, None
    return match.group("base"), int(match.group("layer"))


def _coefficient_rows(
    *,
    model_slug: str,
    dataset: str,
    method: str,
    layer: int | str,
    feature_columns: list[str],
    coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature, coefficient in zip(feature_columns, np.asarray(coefficients, dtype=float).ravel(), strict=False):
        base_feature, feature_layer = _split_layer_feature_name(feature)
        rows.append(
            {
                "model": model_slug,
                "dataset": dataset,
                "method": method,
                "model_layer": layer,
                "feature": feature,
                "base_feature": base_feature,
                "feature_layer": feature_layer,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
            }
        )
    return rows


def _evaluate_best_single_layer(
    *,
    model_slug: str,
    model_name: str,
    feature_df: pd.DataFrame,
    dataset: str,
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
    train_df = dataset_df.loc[dataset_df["split"].eq("train")].copy()
    test_df = dataset_df.loc[dataset_df["split"].eq("test")].copy()
    inner_train_ids, val_ids = _group_train_val_split(train_df, val_fraction=0.2, seed=seed)

    candidate_rows: list[dict[str, Any]] = []
    for layer in sorted(int(layer) for layer in dataset_df["layer"].unique().tolist()):
        layer_train = train_df.loc[train_df["layer"].eq(layer)].copy()
        inner_train = layer_train.loc[layer_train["example_id"].astype(str).isin(inner_train_ids)].copy()
        val_df = layer_train.loc[layer_train["example_id"].astype(str).isin(val_ids)].copy()
        if inner_train.empty or val_df.empty:
            continue
        feature_columns = _topology_feature_columns(layer_train)
        _, payload = _evaluate_feature_set(
            train_features=inner_train,
            eval_features=val_df,
            feature_columns=feature_columns,
            classifier_config=classifier_config,
            seed=seed,
        )
        metrics = payload["eval_metrics"]
        candidate_rows.append(
            {
                "model": model_slug,
                "dataset": dataset,
                "method": "all_topology_best_single_layer",
                "layer": int(layer),
                "val_auroc": float(metrics["auroc"]),
                "val_accuracy": float(metrics["accuracy"]),
                "val_f1": float(metrics["f1"]),
                "feature_count": int(len(feature_columns)),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        raise ValueError(f"No valid single-layer candidates for {model_slug} {dataset}")
    best = _selection_order(candidate_df).iloc[0]
    best_layer = int(best["layer"])
    final_train = train_df.loc[train_df["layer"].eq(best_layer)].copy()
    final_test = test_df.loc[test_df["layer"].eq(best_layer)].copy()
    feature_columns = _topology_feature_columns(final_train)
    _, payload = _evaluate_feature_set(
        train_features=final_train,
        eval_features=final_test,
        feature_columns=feature_columns,
        classifier_config=classifier_config,
        seed=seed,
    )
    metrics = payload["eval_metrics"]
    metric_row = {
        "model": model_slug,
        "model_name": model_name,
        "dataset": dataset,
        "method": "all_topology_best_single_layer",
        "layer": str(best_layer),
        "selection_signature": str(best_layer),
        "feature_count": int(len(feature_columns)),
        "auroc": float(metrics["auroc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
    }
    coefficient_rows = _coefficient_rows(
        model_slug=model_slug,
        dataset=dataset,
        method="all_topology_best_single_layer",
        layer=best_layer,
        feature_columns=feature_columns,
        coefficients=payload["coefficients"],
    )
    return metric_row, coefficient_rows, candidate_df


def _evaluate_all_layers(
    *,
    model_slug: str,
    model_name: str,
    feature_df: pd.DataFrame,
    dataset: str,
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
    all_layers = sorted(int(layer) for layer in dataset_df["layer"].unique().tolist())
    selections = [{"layer": layer, "val_auroc": 0.0} for layer in all_layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=dataset_df,
        dataset=dataset,
        selections=selections,
    )
    feature_columns = list(meta["topology_columns"])
    _, payload = _evaluate_feature_set(
        train_features=train_df,
        eval_features=test_df,
        feature_columns=feature_columns,
        classifier_config=classifier_config,
        seed=seed,
    )
    metrics = payload["eval_metrics"]
    metric_row = {
        "model": model_slug,
        "model_name": model_name,
        "dataset": dataset,
        "method": "all_topology_all_layers",
        "layer": "all",
        "selection_signature": " | ".join(str(layer) for layer in all_layers),
        "feature_count": int(len(feature_columns)),
        "auroc": float(metrics["auroc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
    }
    coefficient_rows = _coefficient_rows(
        model_slug=model_slug,
        dataset=dataset,
        method="all_topology_all_layers",
        layer="all",
        feature_columns=feature_columns,
        coefficients=payload["coefficients"],
    )
    return metric_row, coefficient_rows


def _aggregate_base_importance(coefficient_df: pd.DataFrame) -> pd.DataFrame:
    if coefficient_df.empty:
        return pd.DataFrame()
    grouped = (
        coefficient_df.groupby(["model", "dataset", "method", "base_feature"], as_index=False)
        .agg(
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
            sum_abs_coefficient=("abs_coefficient", "sum"),
            signed_sum_coefficient=("coefficient", "sum"),
            feature_layer_count=("feature", "count"),
        )
        .sort_values(["model", "dataset", "method", "sum_abs_coefficient"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.groupby(["model", "dataset", "method"])["sum_abs_coefficient"].rank(
        method="first",
        ascending=False,
    )
    return grouped


def _overall_importance(base_importance_df: pd.DataFrame) -> pd.DataFrame:
    subset = base_importance_df.loc[base_importance_df["method"].eq("all_topology_all_layers")].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby("base_feature", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_sum_abs_coefficient=("sum_abs_coefficient", "mean"),
            appearances=("rank", "count"),
        )
        .sort_values(["mean_rank", "median_rank", "mean_sum_abs_coefficient"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    candidate_parts: list[pd.DataFrame] = []
    cache_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        model_slug = slugify(model_name)
        classifier_config = dict(config["token_cloud_topology_classifier"])
        token_output_root = Path(classifier_config["output_dir"]) / _token_cloud_model_dir(model_name)
        forward_cache_path = token_output_root / "token_cloud_forward_cache.joblib"
        feature_cache_path = token_output_root / "all_topological_features_no_distance.parquet"
        metadata_path = feature_cache_path.with_suffix(".metadata.json")
        if not forward_cache_path.exists():
            raise FileNotFoundError(f"Missing token-cloud forward cache: {forward_cache_path}")

        feature_reused = feature_cache_path.exists() and not args.force_recompute
        if feature_reused:
            feature_df = pd.read_parquet(feature_cache_path).copy()
        else:
            cloud_df = _load_forward_cache(forward_cache_path)
            feature_df = _compute_topology_table(
                cloud_df=cloud_df,
                classifier_config=classifier_config,
            )
            feature_df.to_parquet(feature_cache_path, index=False)
            write_json(
                metadata_path,
                {
                    "model_name": model_name,
                    "source_forward_cache": str(forward_cache_path),
                    "rows": int(len(feature_df)),
                    "layers": sorted(int(layer) for layer in feature_df["layer"].unique().tolist()),
                    "distance_feature_mode": "none",
                    "contains_token_count": "token_count" in feature_df.columns,
                    "topological_feature_columns": [
                        column for column in _topology_feature_columns(feature_df) if column != "token_count"
                    ],
                },
            )

        topology_df = _topology_only_frame(feature_df)
        datasets = list(classifier_config.get("datasets", ["ambigqa", "situatedqa"]))
        eval_classifier_config = dict(classifier_config.get("classifier", {}))
        for dataset in datasets:
            single_metric, single_coefficients, candidates = _evaluate_best_single_layer(
                model_slug=model_slug,
                model_name=model_name,
                feature_df=topology_df,
                dataset=dataset,
                classifier_config=eval_classifier_config,
                seed=int(args.seed),
            )
            all_metric, all_coefficients = _evaluate_all_layers(
                model_slug=model_slug,
                model_name=model_name,
                feature_df=topology_df,
                dataset=dataset,
                classifier_config=eval_classifier_config,
                seed=int(args.seed),
            )
            metric_rows.extend([single_metric, all_metric])
            coefficient_rows.extend(single_coefficients)
            coefficient_rows.extend(all_coefficients)
            candidate_parts.append(candidates)

        cache_rows.append(
            {
                "model": model_slug,
                "forward_cache_path": str(forward_cache_path),
                "forward_cache_reused": True,
                "topology_feature_cache_path": str(feature_cache_path),
                "topology_feature_cache_reused": bool(feature_reused),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "dataset", "method"]).reset_index(drop=True)
    coefficients_df = pd.DataFrame(coefficient_rows).sort_values(
        ["model", "dataset", "method", "abs_coefficient"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    base_importance_df = _aggregate_base_importance(coefficients_df)
    overall_importance_df = _overall_importance(base_importance_df)
    candidates_df = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    cache_df = pd.DataFrame(cache_rows)

    metrics_path = output_dir / "binary_all_topology_multilayer_metrics.csv"
    coefficients_path = output_dir / "binary_all_topology_multilayer_coefficients.csv"
    base_importance_path = output_dir / "binary_all_topology_multilayer_base_importance.csv"
    overall_importance_path = output_dir / "binary_all_topology_multilayer_overall_importance.csv"
    candidates_path = output_dir / "binary_all_topology_single_layer_candidates.csv"
    cache_path = output_dir / "binary_all_topology_multilayer_cache_status.csv"

    metrics_df.to_csv(metrics_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)
    base_importance_df.to_csv(base_importance_path, index=False)
    overall_importance_df.to_csv(overall_importance_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    cache_df.to_csv(cache_path, index=False)
    write_json(
        output_dir / "binary_all_topology_multilayer_outputs.json",
        {
            "metrics_csv": str(metrics_path),
            "coefficients_csv": str(coefficients_path),
            "base_importance_csv": str(base_importance_path),
            "overall_importance_csv": str(overall_importance_path),
            "single_layer_candidates_csv": str(candidates_path),
            "cache_status_csv": str(cache_path),
        },
    )

    print(metrics_df.to_string(index=False))
    print("\nTop all-layer base features by average rank:")
    print(overall_importance_df.head(20).to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {coefficients_path}")
    print(f"Wrote {base_importance_path}")
    print(f"Wrote {overall_importance_path}")
    print(f"Wrote {cache_path}")


if __name__ == "__main__":
    main()
