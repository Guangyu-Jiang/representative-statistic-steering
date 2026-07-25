"""Tune token-cloud classifiers from saved feature tables.

This script avoids re-extracting token clouds. It loads a saved token-cloud
feature table, sweeps classifier families and feature subsets using an inner
train/validation split on the train partition, then refits the best config on
the full train split and reports test metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.train.independent_topology_classifier import (
    _fit_classifier,
    _group_train_val_split,
    _predict_scores,
    _transform_with_scaler,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

KEY_COLUMNS = {"example_id", "pair_id", "dataset", "split", "label_ambiguous", "layer", "token_count", "feature_variant"}


FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic": {
        "family": "logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
    "logistic_c0_25": {
        "family": "logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 0.25,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
    "logistic_c4": {
        "family": "logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 4.0,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
    "random_forest": {
        "family": "random_forest",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "standardize": False,
    },
    "random_forest_leaf4": {
        "family": "random_forest",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "standardize": False,
    },
    "extra_trees": {
        "family": "extra_trees",
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "n_jobs": -1,
        "standardize": False,
    },
    "extra_trees_leaf4": {
        "family": "extra_trees",
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "standardize": False,
    },
    "hist_gradient_boosting": {
        "family": "hist_gradient_boosting",
        "learning_rate": 0.05,
        "max_depth": None,
        "max_iter": 400,
        "min_samples_leaf": 20,
        "standardize": False,
    },
    "hist_gradient_boosting_leaf10": {
        "family": "hist_gradient_boosting",
        "learning_rate": 0.05,
        "max_depth": None,
        "max_iter": 400,
        "min_samples_leaf": 10,
        "standardize": False,
    },
}


def _available_topology_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in KEY_COLUMNS:
            continue
        if not (column.startswith("h0_") or column.startswith("h1_")):
            continue
        if frame[column].notna().any():
            columns.append(column)
    return columns


def _feature_subset_columns(frame: pd.DataFrame, subset_name: str) -> list[str]:
    columns = _available_topology_columns(frame)
    if subset_name == "all_topology":
        return columns
    if subset_name == "h0_all":
        return [col for col in columns if col.startswith("h0_")]
    if subset_name == "h1_all":
        return [col for col in columns if col.startswith("h1_")]
    if subset_name == "no_pimg":
        return [col for col in columns if "_pimg_" not in col]
    if subset_name == "no_bottleneck":
        return [col for col in columns if "_bottleneck_" not in col]
    if subset_name == "no_distance":
        return [col for col in columns if "_wasserstein_" not in col and "_bottleneck_" not in col]
    if subset_name == "descriptors_only":
        return [col for col in columns if "_pimg_" not in col and "_wasserstein_" not in col and "_bottleneck_" not in col]
    if subset_name == "distance_only":
        return [col for col in columns if "_wasserstein_" in col or "_bottleneck_" in col]
    if subset_name == "h0_core":
        return [
            col
            for col in columns
            if col.startswith("h0_") and "_pimg_" not in col and "_bottleneck_" not in col
        ]
    if subset_name == "h1_core":
        return [
            col
            for col in columns
            if col.startswith("h1_") and "_pimg_" not in col and "_bottleneck_" not in col
        ]
    if subset_name == "core_no_pimg":
        return [col for col in columns if "_pimg_" not in col and "_bottleneck_" not in col]
    raise ValueError(f"Unknown feature subset: {subset_name}")


def _evaluate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_df["label_ambiguous"].to_numpy(dtype=int)
    x_eval = eval_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_eval = eval_df["label_ambiguous"].to_numpy(dtype=int)
    clf, scaler = _fit_classifier(x_train, y_train, config=classifier_config, seed=seed)
    train_scores = _predict_scores(clf, _transform_with_scaler(x_train, scaler))
    eval_scores = _predict_scores(clf, _transform_with_scaler(x_eval, scaler))
    return {
        "classifier": clf,
        "scaler": scaler,
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "eval_metrics": binary_classification_metrics(y_eval, eval_scores),
    }


def _selection_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["val_auroc", "val_accuracy", "test_auroc", "test_accuracy", "family", "feature_subset"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def tune_feature_table(
    *,
    feature_table_path: Path,
    output_root: Path,
    seed: int,
    val_fraction: float,
    dataset_filter: list[str] | None = None,
    feature_variants: list[str] | None = None,
    feature_subsets: list[str] | None = None,
    family_names: list[str] | None = None,
) -> dict[str, str]:
    feature_df = pd.read_parquet(feature_table_path)
    output_root = ensure_dir(output_root)

    subset_names = feature_subsets or [
        "all_topology",
        "core_no_pimg",
        "descriptors_only",
        "no_pimg",
        "no_bottleneck",
        "h0_all",
        "h0_core",
        "h1_all",
        "h1_core",
        "distance_only",
    ]
    family_items = [
        (name, FAMILY_CONFIGS[name])
        for name in (family_names or list(FAMILY_CONFIGS.keys()))
    ]

    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    available_datasets = sorted(feature_df["dataset"].dropna().unique())
    for dataset in available_datasets:
        if dataset_filter and dataset not in dataset_filter:
            continue
        dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
        available_variants = sorted(dataset_df["feature_variant"].dropna().unique())
        for feature_variant in available_variants:
            if feature_variants and feature_variant not in feature_variants:
                continue
            variant_df = dataset_df.loc[dataset_df["feature_variant"].eq(feature_variant)].copy()
            train_df = variant_df.loc[variant_df["split"].eq("train")].copy()
            test_df = variant_df.loc[variant_df["split"].eq("test")].copy()
            if train_df.empty or test_df.empty:
                continue
            inner_train_ids, val_ids = _group_train_val_split(train_df, val_fraction=val_fraction, seed=seed)
            inner_train = train_df.loc[train_df["example_id"].astype(str).isin(inner_train_ids)].copy()
            val_df = train_df.loc[train_df["example_id"].astype(str).isin(val_ids)].copy()
            if inner_train.empty or val_df.empty:
                continue

            for subset_name in subset_names:
                feature_columns = _feature_subset_columns(variant_df, subset_name)
                if not feature_columns:
                    continue
                for family_name, family_config in family_items:
                    val_payload = _evaluate(
                        inner_train,
                        val_df,
                        feature_columns=feature_columns,
                        classifier_config=family_config,
                        seed=seed,
                    )
                    final_payload = _evaluate(
                        train_df,
                        test_df,
                        feature_columns=feature_columns,
                        classifier_config=family_config,
                        seed=seed,
                    )
                    candidate_rows.append(
                        {
                            "dataset": dataset,
                            "feature_variant": feature_variant,
                            "feature_subset": subset_name,
                            "family": family_name,
                            "feature_count": len(feature_columns),
                            "val_auroc": float(val_payload["eval_metrics"]["auroc"]),
                            "val_accuracy": float(val_payload["eval_metrics"]["accuracy"]),
                            "test_auroc": float(final_payload["eval_metrics"]["auroc"]),
                            "test_accuracy": float(final_payload["eval_metrics"]["accuracy"]),
                        }
                    )

            candidate_df = pd.DataFrame(candidate_rows)
            scoped = candidate_df.loc[
                candidate_df["dataset"].eq(dataset) & candidate_df["feature_variant"].eq(feature_variant)
            ].copy()
            if scoped.empty:
                continue
            best = _selection_order(scoped).iloc[0]
            final_rows.append(best.to_dict())

    candidate_df = pd.DataFrame(candidate_rows)
    final_df = pd.DataFrame(final_rows)
    candidate_path = output_root / "token_cloud_tuning_candidates.parquet"
    final_path = output_root / "token_cloud_tuning_best.parquet"
    report_path = output_root / "token_cloud_tuning_summary.md"
    metadata_path = output_root / "token_cloud_tuning_metadata.json"
    write_parquet(candidate_df, candidate_path)
    write_parquet(final_df, final_path)

    lines = [
        "# Token-Cloud Feature-Table Tuning",
        "",
        f"- Source feature table: `{feature_table_path}`",
        f"- Created at: `{utc_now_iso()}`",
        "",
    ]
    if not final_df.empty:
        for row in final_df.to_dict(orient="records"):
            lines.extend(
                [
                    f"## {row['dataset']} / {row['feature_variant']}",
                    "",
                    f"- Best family: `{row['family']}`",
                    f"- Best subset: `{row['feature_subset']}`",
                    f"- Validation AUROC `{row['val_auroc']:.4f}`, accuracy `{row['val_accuracy']:.4f}`",
                    f"- Test AUROC `{row['test_auroc']:.4f}`, accuracy `{row['test_accuracy']:.4f}`",
                    "",
                ]
            )
    write_markdown(report_path, "\n".join(lines) + "\n")
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "source_feature_table": str(feature_table_path),
            "output_artifacts": {
                "candidate_metrics": str(candidate_path),
                "best_metrics": str(final_path),
                "report": str(report_path),
            },
        },
    )
    return {
        "candidate_metrics": str(candidate_path),
        "best_metrics": str(final_path),
        "report": str(report_path),
        "metadata": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--feature-variants", nargs="*", default=None)
    parser.add_argument("--feature-subsets", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_feature_table(
        feature_table_path=Path(args.feature_table_path),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        dataset_filter=args.datasets,
        feature_variants=args.feature_variants,
        feature_subsets=args.feature_subsets,
        family_names=args.families,
    )


if __name__ == "__main__":
    main()
