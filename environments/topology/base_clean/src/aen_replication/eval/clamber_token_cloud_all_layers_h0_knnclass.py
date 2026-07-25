"""All-layer CLAMBER token-cloud rerun with H0 KNN-class Wasserstein features.

This reuses the previously computed all-layer token-cloud PH summary features,
drops the old pooled prototype-distance features, and adds per-layer H0
KNN-class Wasserstein features across all 32 layers.

The comparison is restricted to multilayer stacks:

- current compact stack: 0 | 14
- all-layer stack: 0 | 1 | ... | 31

For each task/stack pair, k is selected on a train-set validation split from a
fixed candidate list, then the model is refit on the full train split and
evaluated on the test split.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from aen_replication.config import load_config
from aen_replication.eval.clamber_token_cloud_classavg_wasserstein import (
    BASE_FEATURE_PATH,
    GROUP4_MAP,
    GROUP4_ORDER,
    MODEL_LABEL,
    MODEL_SLUG,
    PREVIOUS_METRIC_PATHS,
    SUBCLASS_ORDER,
    _compute_h0_diagrams,
    _compute_metrics,
    _drop_prototype_distance_features,
    _evaluate_multilayer,
    _extract_clamber_clouds,
    _fit_multiclass_logistic,
)
from aen_replication.eval.clamber_token_cloud_knnclass_wasserstein import (
    _knn_group4_column,
    _knn_subclass_column,
    _layer_knn_features,
)
from aen_replication.train.token_cloud_topology_classifier import _build_multilayer_feature_frames
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet


ALL_LAYER_FEATURE_PATH = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
    "clamber_token_cloud_all_layers_stack/llama_clamber_token_cloud_all_layers_features.parquet"
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_token_cloud_all_layers_h0_knnclass",
    )
    parser.add_argument("--k-candidates", default="1,2,4,8,16,32")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--parallel-jobs", type=int, default=192)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _parse_k_candidates(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid k candidate list: {raw}")
    return values


def _configure_logging(output_root: Path) -> Path:
    log_path = output_root / "run.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    return log_path


def _layer_feature_path(output_path: Path, layer: int) -> Path:
    return output_path.with_name(f"{output_path.stem}__layer_{int(layer):02d}{output_path.suffix}")


def _load_existing_knn_parts(output_path: Path, layers: list[int]) -> tuple[list[pd.DataFrame], list[int]]:
    parts: list[pd.DataFrame] = []
    completed_layers: list[int] = []
    for layer in layers:
        layer_path = _layer_feature_path(output_path, int(layer))
        if layer_path.exists():
            parts.append(pd.read_parquet(layer_path).copy())
            completed_layers.append(int(layer))
    return parts, completed_layers


def _load_reference_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for task, path in PREVIOUS_METRIC_PATHS.items():
        df = pd.read_parquet(path).copy()
        df = df.loc[(df["model"].eq(MODEL_SLUG)) & (df["method"].eq("token_cloud_multilayer"))].copy()
        if df.empty:
            continue
        row = df.iloc[0]
        rows[task] = {
            "previous_accuracy": float(row["accuracy"]),
            "previous_macro_f1": float(row["macro_f1"]),
            "previous_selection_signature": str(row["selection_signature"]),
        }
    return rows


def _prepare_task_frame(
    *,
    base_df: pd.DataFrame,
    knn_df: pd.DataFrame,
    task: str,
    k: int,
) -> tuple[pd.DataFrame, str]:
    merged = base_df.merge(knn_df, on=["example_id", "layer"], how="left")
    merged = _drop_prototype_distance_features(merged)
    if task == "4way":
        merged = merged.loc[merged["subclass"].isin(GROUP4_MAP)].copy()
        merged["group4"] = merged["subclass"].map(GROUP4_MAP)
        keep_cols = [_knn_group4_column(group, k) for group in GROUP4_ORDER if _knn_group4_column(group, k) in merged.columns]
        drop_cols = [
            column
            for column in merged.columns
            if column.startswith("h0_knnclass_") and column not in keep_cols
        ]
        merged = merged.drop(columns=drop_cols, errors="ignore")
        return merged.reset_index(drop=True), "group4"
    if task == "9way":
        keep_cols = [
            _knn_subclass_column(subclass, k)
            for subclass in SUBCLASS_ORDER
            if _knn_subclass_column(subclass, k) in merged.columns
        ]
        drop_cols = [
            column
            for column in merged.columns
            if column.startswith("h0_knnclass_") and column not in keep_cols
        ]
        merged = merged.drop(columns=drop_cols, errors="ignore")
        return merged.reset_index(drop=True), "subclass"
    raise ValueError(f"Unsupported task: {task}")


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = [0] * len(labels)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return train_idx.tolist(), val_idx.tolist()


def _validate_k(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    layers: list[int],
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, _, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    label_lookup = feature_df.loc[:, ["example_id", label_col]].drop_duplicates()
    train_df = train_df.merge(label_lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    tr_idx, val_idx = _split_indices(train_df[label_col], val_fraction=val_fraction, seed=seed)
    x_train = train_df.iloc[tr_idx].loc[:, feature_cols].to_numpy(dtype=float)
    y_train = train_df.iloc[tr_idx][label_col].astype(str).to_numpy()
    x_val = train_df.iloc[val_idx].loc[:, feature_cols].to_numpy(dtype=float)
    y_val = train_df.iloc[val_idx][label_col].astype(str).to_numpy()
    labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed)
    y_pred = clf.predict(scaler.transform(x_val))
    metrics = _compute_metrics(y_val, y_pred, labels)
    return {
        "val_accuracy": float(metrics["accuracy"]),
        "val_macro_f1": float(metrics["macro_f1"]),
        "feature_count": int(len(feature_cols)),
    }


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    log_path = _configure_logging(output_root)
    ks = _parse_k_candidates(args.k_candidates)
    config = load_config(args.config)
    LOGGER.info("Starting run with k candidates=%s", ks)
    LOGGER.info("Logging to %s", log_path)

    base_df = pd.read_parquet(ALL_LAYER_FEATURE_PATH if ALL_LAYER_FEATURE_PATH.exists() else BASE_FEATURE_PATH).copy()
    all_layers = sorted(base_df["layer"].unique().tolist())
    output_path = output_root / "llama_clamber_all_layers_h0_knnclass_features.parquet"

    if output_path.exists() and not args.force_recompute:
        LOGGER.info("Using existing merged KNN feature frame: %s", output_path)
        knn_df = pd.read_parquet(output_path).copy()
    else:
        existing_parts, completed_layers = _load_existing_knn_parts(output_path, all_layers)
        if args.force_recompute:
            LOGGER.info("Force recompute enabled; ignoring %d completed layer files", len(completed_layers))
            existing_parts = []
            completed_layers = []
        missing_layers = [int(layer) for layer in all_layers if int(layer) not in set(completed_layers)]
        LOGGER.info("Completed layers: %s", completed_layers if completed_layers else "none")
        LOGGER.info("Missing layers: %s", missing_layers if missing_layers else "none")

        if missing_layers:
            cloud_df = _extract_clamber_clouds(config=config, layers=missing_layers, seed=args.seed)
            classifier_config = dict(config["token_cloud_topology_classifier"])
            diagram_df = _compute_h0_diagrams(
                cloud_df,
                maxdim=int(classifier_config.get("maxdim", 1)),
                coeff=int(classifier_config.get("coeff", 2)),
                distance_metric=str(classifier_config.get("distance_metric", "euclidean")),
                parallel_jobs=int(args.parallel_jobs),
            )
            fresh_parts: list[pd.DataFrame] = []
            for layer in missing_layers:
                LOGGER.info("Computing KNN-class features for layer %02d", int(layer))
                layer_df = diagram_df.loc[diagram_df["layer"].eq(layer)].copy()
                layer_features = _layer_knn_features(
                    layer_df,
                    ks=ks,
                    workers=int(args.workers),
                    chunk_size=int(args.chunk_size),
                )
                layer_path = _layer_feature_path(output_path, int(layer))
                write_parquet(layer_features, layer_path)
                fresh_parts.append(layer_features)
                LOGGER.info("Wrote %s", layer_path)
            existing_parts.extend(fresh_parts)

        if not existing_parts:
            raise RuntimeError("No per-layer KNN feature parts available to assemble.")

        knn_df = pd.concat(existing_parts, ignore_index=True)
        knn_df = knn_df.sort_values(["layer", "example_id"]).reset_index(drop=True)
        write_parquet(knn_df, output_path)
        LOGGER.info("Wrote merged KNN feature frame: %s", output_path)

    references = _load_reference_rows()
    selections_map = {
        "current_0_14": [0, 14],
        "all_layers": all_layers,
    }

    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for task in ["4way", "9way"]:
        for selection_name, layers in selections_map.items():
            rows: list[dict[str, Any]] = []
            for k in ks:
                feature_df, label_col = _prepare_task_frame(base_df=base_df, knn_df=knn_df, task=task, k=k)
                metrics = _validate_k(
                    feature_df,
                    label_col=label_col,
                    layers=layers,
                    seed=args.seed + (1000 if task == "9way" else 0) + int(k),
                    val_fraction=float(args.val_fraction),
                )
                rows.append(
                    {
                        "task": task,
                        "selection_name": selection_name,
                        "k": int(k),
                        **metrics,
                    }
                )
            candidate_df = pd.DataFrame(rows).sort_values(
                ["val_macro_f1", "val_accuracy", "k"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            candidate_rows.extend(candidate_df.to_dict(orient="records"))
            best_k = int(candidate_df.iloc[0]["k"])

            feature_df, label_col = _prepare_task_frame(base_df=base_df, knn_df=knn_df, task=task, k=best_k)
            payload = _evaluate_multilayer(
                feature_df,
                label_col=label_col,
                layers=layers,
                seed=args.seed + (2000 if task == "9way" else 1000) + best_k,
            )
            reference = references[task]
            final_rows.append(
                {
                    "task": task,
                    "selection_name": selection_name,
                    "selection_signature": payload["selection_signature"],
                    "selection_size": int(len(layers)),
                    "best_k": best_k,
                    "feature_count": int(payload["feature_count"]),
                    "accuracy": float(payload["accuracy"]),
                    "macro_f1": float(payload["macro_f1"]),
                    "delta_accuracy_vs_stored": float(payload["accuracy"] - reference["previous_accuracy"]),
                    "delta_macro_f1_vs_stored": float(payload["macro_f1"] - reference["previous_macro_f1"]),
                    "stored_reference_signature": str(reference["previous_selection_signature"]),
                    "confusion_matrix": payload["confusion_matrix"],
                    "labels": payload["labels"],
                }
            )

    candidate_out = pd.DataFrame(candidate_rows)
    final_out = pd.DataFrame(final_rows)
    write_parquet(candidate_out, output_root / "clamber_token_cloud_all_layers_h0_knnclass_candidates.parquet")
    write_parquet(final_out, output_root / "clamber_token_cloud_all_layers_h0_knnclass_results.parquet")
    LOGGER.info("Wrote candidate and final result parquets")

    lines = [
        "# CLAMBER All-Layer H0 KNN-Class Token-Cloud Rerun",
        "",
        f"- Model: `{MODEL_LABEL}`",
        "- Base features: existing all-layer token-cloud PH summaries",
        "- Distance replacement: drop prototype distances and add H0 KNN-class Wasserstein features",
        "",
    ]
    for task in ["4way", "9way"]:
        lines.append(f"## {task.upper()}")
        lines.append("")
        lines.append("| Stack | Layers | Best k | Features | Acc | Macro-F1 | Delta Acc vs stored | Delta Macro-F1 vs stored |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        subset = final_out.loc[final_out["task"].eq(task)].copy()
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['selection_name']} | `{row['selection_signature']}` | {int(row['best_k'])} | "
                f"{int(row['feature_count'])} | {float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
                f"{float(row['delta_accuracy_vs_stored']):+.4f} | {float(row['delta_macro_f1_vs_stored']):+.4f} |"
            )
        lines.append("")
    write_markdown(output_root / "clamber_token_cloud_all_layers_h0_knnclass_report.md", "\n".join(lines) + "\n")
    LOGGER.info("Wrote markdown report")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Run failed with an uncaught exception")
        traceback.print_exc()
        raise
