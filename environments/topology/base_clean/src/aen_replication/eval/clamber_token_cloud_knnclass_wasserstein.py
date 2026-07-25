"""CLAMBER token-cloud follow-up with k-nearest-in-class diagram distances.

This uses the current token-cloud H0 persistence diagram per question and adds
class-local distance features:

    mean of the k nearest train-set diagrams in class c

The fixed evaluation views match the current LLaMA CLAMBER token-cloud setup:

- 4-way: single layer `0`, multilayer `0 | 14`
- 9-way: single layer `0`, multilayer `0 | 14`

For each task / method / feature variant, `k` is selected on a validation split
inside the training set and then the model is refit on the full train split and
evaluated on the test split.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

from aen_replication.eval.clamber_token_cloud_classavg_wasserstein import (
    BASE_FEATURE_PATH,
    GROUP4_MAP,
    GROUP4_ORDER,
    MODEL_LABEL,
    MODEL_SLUG,
    SUBCLASS_ORDER,
    _compute_metrics,
    _compute_h0_diagrams,
    _drop_prototype_distance_features,
    _evaluate_multilayer as _evaluate_fixed_multilayer,
    _evaluate_single as _evaluate_fixed_single,
    _extract_clamber_clouds,
    _fit_multiclass_logistic,
    _load_selected_views,
)
from aen_replication.config import load_config
from aen_replication.train.token_cloud_topology_classifier import _build_multilayer_feature_frames, _topology_feature_columns
from aen_replication.train.independent_topology_classifier import _safe_wasserstein
from aen_replication.utils.io_utils import ensure_dir, slugify, write_markdown, write_parquet

_KNN_QUERY_IDS: list[str] = []
_KNN_QUERY_DIAGRAMS: list[np.ndarray] = []
_KNN_TRAIN_IDS: list[str] = []
_KNN_TRAIN_SUBCLASSES: list[str] = []
_KNN_TRAIN_GROUP4: list[str] = []
_KNN_TRAIN_DIAGRAMS: list[np.ndarray] = []
_KNN_KS: list[int] = []
_KNN_MAX_K = 0
_KNN_SUBCLASS_LABELS: list[str] = []
_KNN_GROUP4_LABELS: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_token_cloud_knnclass_wasserstein",
    )
    parser.add_argument("--k-candidates", default="1,2,4,8,16,32")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=min(24, max(1, (os.cpu_count() or 8) - 1)))
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--parallel-jobs", type=int, default=12)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _parse_k_candidates(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid k candidate list: {raw}")
    return values


def _knn_subclass_column(subclass: str, k: int) -> str:
    return f"h0_knnclass_subclass_k{k:02d}__{slugify(subclass)}"


def _knn_group4_column(group: str, k: int) -> str:
    return f"h0_knnclass_group4_k{k:02d}__{slugify(group)}"


def _knn_all_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("h0_knnclass_")]


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _smallest_k_means(values: list[float], ks: list[int]) -> dict[int, float]:
    if not values:
        return {k: 0.0 for k in ks}
    arr = np.asarray(values, dtype=np.float64)
    max_k = min(max(ks), arr.size)
    if arr.size > max_k:
        arr = np.partition(arr, max_k - 1)[:max_k]
    arr = np.sort(arr)
    cumulative = np.cumsum(arr)
    output: dict[int, float] = {}
    for k in ks:
        use = min(int(k), arr.size)
        output[int(k)] = float(cumulative[use - 1] / use)
    return output


def _init_knn_worker(
    query_ids: list[str],
    query_diagrams: list[np.ndarray],
    train_ids: list[str],
    train_subclasses: list[str],
    train_group4: list[str],
    train_diagrams: list[np.ndarray],
    ks: list[int],
    subclass_labels: list[str],
    group4_labels: list[str],
) -> None:
    global _KNN_QUERY_IDS, _KNN_QUERY_DIAGRAMS, _KNN_TRAIN_IDS, _KNN_TRAIN_SUBCLASSES, _KNN_TRAIN_GROUP4
    global _KNN_TRAIN_DIAGRAMS, _KNN_KS, _KNN_MAX_K, _KNN_SUBCLASS_LABELS, _KNN_GROUP4_LABELS
    _KNN_QUERY_IDS = query_ids
    _KNN_QUERY_DIAGRAMS = query_diagrams
    _KNN_TRAIN_IDS = train_ids
    _KNN_TRAIN_SUBCLASSES = train_subclasses
    _KNN_TRAIN_GROUP4 = train_group4
    _KNN_TRAIN_DIAGRAMS = train_diagrams
    _KNN_KS = ks
    _KNN_MAX_K = max(ks)
    _KNN_SUBCLASS_LABELS = subclass_labels
    _KNN_GROUP4_LABELS = group4_labels


def _knn_chunk(
    start: int,
    stop: int,
) -> tuple[int, dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    subclass_outputs = {
        k: {label: np.zeros(stop - start, dtype=np.float32) for label in _KNN_SUBCLASS_LABELS}
        for k in _KNN_KS
    }
    group4_outputs = {
        k: {label: np.zeros(stop - start, dtype=np.float32) for label in _KNN_GROUP4_LABELS}
        for k in _KNN_KS
    }

    for local_index, query_index in enumerate(range(start, stop)):
        query_id = _KNN_QUERY_IDS[query_index]
        query_diagram = _KNN_QUERY_DIAGRAMS[query_index]
        subclass_values = {label: [] for label in _KNN_SUBCLASS_LABELS}
        group4_values = {label: [] for label in _KNN_GROUP4_LABELS}
        for train_id, train_subclass, train_group4, train_diagram in zip(
            _KNN_TRAIN_IDS,
            _KNN_TRAIN_SUBCLASSES,
            _KNN_TRAIN_GROUP4,
            _KNN_TRAIN_DIAGRAMS,
        ):
            if train_id == query_id:
                continue
            distance = _safe_wasserstein(query_diagram, train_diagram)
            subclass_values[train_subclass].append(distance)
            if train_group4:
                group4_values[train_group4].append(distance)
        for label in _KNN_SUBCLASS_LABELS:
            means = _smallest_k_means(subclass_values[label], _KNN_KS)
            for k in _KNN_KS:
                subclass_outputs[k][label][local_index] = np.float32(means[k])
        for label in _KNN_GROUP4_LABELS:
            means = _smallest_k_means(group4_values[label], _KNN_KS)
            for k in _KNN_KS:
                group4_outputs[k][label][local_index] = np.float32(means[k])
    return start, subclass_outputs, group4_outputs


def _layer_knn_features(
    layer_df: pd.DataFrame,
    *,
    ks: list[int],
    workers: int,
    chunk_size: int,
) -> pd.DataFrame:
    layer_df = layer_df.sort_values("example_id").reset_index(drop=True)
    train_df = layer_df.loc[layer_df["split"].eq("train")].copy().sort_values("example_id").reset_index(drop=True)

    subclass_labels = [label for label in SUBCLASS_ORDER if label in set(train_df["subclass"].astype(str))]
    group4_series = train_df["subclass"].map(GROUP4_MAP)
    group4_labels = [label for label in GROUP4_ORDER if label in set(group4_series.dropna().astype(str))]

    query_ids = layer_df["example_id"].astype(str).tolist()
    query_diagrams = layer_df["h0_diagram"].tolist()
    train_ids = train_df["example_id"].astype(str).tolist()
    train_subclasses = train_df["subclass"].astype(str).tolist()
    train_group4 = [str(GROUP4_MAP.get(subclass, "")) for subclass in train_subclasses]
    train_diagrams = train_df["h0_diagram"].tolist()

    chunk_ranges = [
        (start, min(start + int(chunk_size), len(layer_df)))
        for start in range(0, len(layer_df), int(chunk_size))
    ]

    subclass_storage = {
        k: {label: np.zeros(len(layer_df), dtype=np.float32) for label in subclass_labels}
        for k in ks
    }
    group4_storage = {
        k: {label: np.zeros(len(layer_df), dtype=np.float32) for label in group4_labels}
        for k in ks
    }

    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_init_knn_worker,
        initargs=(
            query_ids,
            query_diagrams,
            train_ids,
            train_subclasses,
            train_group4,
            train_diagrams,
            ks,
            subclass_labels,
            group4_labels,
        ),
    ) as executor:
        futures = {
            executor.submit(_knn_chunk, start, stop): (start, stop)
            for start, stop in chunk_ranges
        }
        progress = tqdm(total=len(chunk_ranges), desc=f"layer_{int(layer_df.iloc[0]['layer']):02d}_knnclass", leave=False)
        for future in as_completed(futures):
            start, subclass_chunk, group4_chunk = future.result()
            stop = futures[future][1]
            for k in ks:
                for label in subclass_labels:
                    subclass_storage[k][label][start:stop] = subclass_chunk[k][label]
                for label in group4_labels:
                    group4_storage[k][label][start:stop] = group4_chunk[k][label]
            progress.update(1)
        progress.close()

    result = layer_df.loc[:, ["example_id", "layer"]].copy()
    for k in ks:
        for label in subclass_labels:
            result[_knn_subclass_column(label, k)] = subclass_storage[k][label].astype(float)
        for label in group4_labels:
            result[_knn_group4_column(label, k)] = group4_storage[k][label].astype(float)
    return result


def _compute_knn_feature_frame(
    diagram_df: pd.DataFrame,
    *,
    output_path: Path,
    ks: list[int],
    workers: int,
    chunk_size: int,
    force_recompute: bool,
) -> pd.DataFrame:
    if output_path.exists() and not force_recompute:
        return pd.read_parquet(output_path).copy()
    parts: list[pd.DataFrame] = []
    for layer in sorted(diagram_df["layer"].unique().tolist()):
        layer_path = output_path.with_name(f"{output_path.stem}__layer_{int(layer):02d}{output_path.suffix}")
        if layer_path.exists() and not force_recompute:
            layer_features = pd.read_parquet(layer_path).copy()
        else:
            layer_df = diagram_df.loc[diagram_df["layer"].eq(layer)].copy()
            layer_features = _layer_knn_features(layer_df, ks=ks, workers=workers, chunk_size=chunk_size)
            write_parquet(layer_features, layer_path)
        parts.append(layer_features)
    knn_df = pd.concat(parts, ignore_index=True)
    write_parquet(knn_df, output_path)
    return knn_df


def _prepare_task_variant_frame(
    *,
    base_df: pd.DataFrame,
    knn_df: pd.DataFrame | None,
    task: str,
    variant: str,
    k: int | None,
) -> tuple[pd.DataFrame, str]:
    if knn_df is not None:
        df = base_df.merge(knn_df, on=["example_id", "layer"], how="left")
    else:
        df = base_df.copy()

    if variant == "no_proto_plus_knnclass":
        df = _drop_prototype_distance_features(df)

    if task == "4way":
        df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
        df["group4"] = df["subclass"].map(GROUP4_MAP)
        label_col = "group4"
        keep_knn = [] if k is None else [_knn_group4_column(group, k) for group in GROUP4_ORDER if _knn_group4_column(group, k) in df.columns]
    elif task == "9way":
        df = df.copy()
        label_col = "subclass"
        keep_knn = [] if k is None else [_knn_subclass_column(subclass, k) for subclass in SUBCLASS_ORDER if _knn_subclass_column(subclass, k) in df.columns]
    else:
        raise ValueError(f"Unsupported task: {task}")

    all_knn = _knn_all_columns(df)
    drop_knn = [column for column in all_knn if column not in keep_knn]
    if drop_knn:
        df = df.drop(columns=drop_knn, errors="ignore")
    return df.reset_index(drop=True), label_col


def _validate_single_k(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    layer: int,
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    train_df = feature_df.loc[feature_df["split"].eq("train") & feature_df["layer"].eq(layer)].copy()
    tr_idx, val_idx = _split_indices(train_df[label_col], val_fraction=val_fraction, seed=seed)
    feature_cols = _topology_feature_columns(train_df)
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


def _validate_multilayer_k(
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


def _select_best_k(
    *,
    base_df: pd.DataFrame,
    knn_df: pd.DataFrame,
    task: str,
    method: str,
    variant: str,
    layers: list[int],
    ks: list[int],
    seed: int,
    val_fraction: float,
) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for k in ks:
        feature_df, label_col = _prepare_task_variant_frame(
            base_df=base_df,
            knn_df=knn_df,
            task=task,
            variant=variant,
            k=k,
        )
        if method == "token_cloud_single":
            metrics = _validate_single_k(
                feature_df,
                label_col=label_col,
                layer=int(layers[0]),
                seed=seed + 100 + int(k),
                val_fraction=val_fraction,
            )
        else:
            metrics = _validate_multilayer_k(
                feature_df,
                label_col=label_col,
                layers=layers,
                seed=seed + 200 + int(k),
                val_fraction=val_fraction,
            )
        rows.append(
            {
                "task": task,
                "method": method,
                "variant": variant,
                "k": int(k),
                **metrics,
            }
        )
    candidate_df = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return int(candidate_df.iloc[0]["k"]), candidate_df


def main() -> None:
    args = parse_args()
    ks = _parse_k_candidates(args.k_candidates)
    output_root = ensure_dir(Path(args.output_root))

    config = load_config(args.config)
    selected_views = _load_selected_views()
    all_layers = sorted(
        {
            layer
            for task_views in selected_views.values()
            for method_payload in task_views.values()
            for layer in method_payload["layers"]
        }
    )

    base_df = pd.read_parquet(BASE_FEATURE_PATH).copy()
    cloud_df = _extract_clamber_clouds(config=config, layers=all_layers, seed=args.seed)
    classifier_config = dict(config["token_cloud_topology_classifier"])
    diagram_df = _compute_h0_diagrams(
        cloud_df,
        maxdim=int(classifier_config.get("maxdim", 1)),
        coeff=int(classifier_config.get("coeff", 2)),
        distance_metric=str(classifier_config.get("distance_metric", "euclidean")),
        parallel_jobs=int(args.parallel_jobs),
    )
    knn_df = _compute_knn_feature_frame(
        diagram_df,
        output_path=output_root / "llama_clamber_h0_knnclass_features.parquet",
        ks=ks,
        workers=args.workers,
        chunk_size=args.chunk_size,
        force_recompute=bool(args.force_recompute),
    )

    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for task in ["4way", "9way"]:
        for method in ["token_cloud_single", "token_cloud_multilayer"]:
            layers = [int(layer) for layer in selected_views[task][method]["layers"]]

            baseline_df, label_col = _prepare_task_variant_frame(
                base_df=base_df,
                knn_df=None,
                task=task,
                variant="baseline",
                k=None,
            )
            if method == "token_cloud_single":
                baseline_payload = _evaluate_fixed_single(
                    baseline_df,
                    label_col=label_col,
                    layer=int(layers[0]),
                    seed=args.seed + (0 if task == "4way" else 100),
                )
            else:
                baseline_payload = _evaluate_fixed_multilayer(
                    baseline_df,
                    label_col=label_col,
                    layers=layers,
                    seed=args.seed + (10 if task == "4way" else 110),
                )
            previous = selected_views[task][method]
            final_rows.append(
                {
                    "task": task,
                    "method": method,
                    "variant": "baseline",
                    "best_k": np.nan,
                    "selection_signature": baseline_payload["selection_signature"],
                    "feature_count": int(baseline_payload["feature_count"]),
                    "accuracy": float(baseline_payload["accuracy"]),
                    "macro_f1": float(baseline_payload["macro_f1"]),
                    "previous_accuracy": float(previous["previous_accuracy"]),
                    "previous_macro_f1": float(previous["previous_macro_f1"]),
                    "delta_accuracy_vs_previous": float(baseline_payload["accuracy"] - previous["previous_accuracy"]),
                    "delta_macro_f1_vs_previous": float(baseline_payload["macro_f1"] - previous["previous_macro_f1"]),
                    "confusion_matrix": baseline_payload["confusion_matrix"],
                    "labels": baseline_payload["labels"],
                }
            )

            for variant in ["baseline_plus_knnclass", "no_proto_plus_knnclass"]:
                best_k, candidate_df = _select_best_k(
                    base_df=base_df,
                    knn_df=knn_df,
                    task=task,
                    method=method,
                    variant=variant,
                    layers=layers,
                    ks=ks,
                    seed=args.seed + (0 if task == "4way" else 1000) + (0 if method == "token_cloud_single" else 100),
                    val_fraction=args.val_fraction,
                )
                candidate_rows.extend(candidate_df.to_dict(orient="records"))

                feature_df, label_col = _prepare_task_variant_frame(
                    base_df=base_df,
                    knn_df=knn_df,
                    task=task,
                    variant=variant,
                    k=best_k,
                )
                if method == "token_cloud_single":
                    payload = _evaluate_fixed_single(
                        feature_df,
                        label_col=label_col,
                        layer=int(layers[0]),
                        seed=args.seed + (0 if task == "4way" else 100) + best_k,
                    )
                else:
                    payload = _evaluate_fixed_multilayer(
                        feature_df,
                        label_col=label_col,
                        layers=layers,
                        seed=args.seed + (10 if task == "4way" else 110) + best_k,
                    )

                final_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "variant": variant,
                        "best_k": int(best_k),
                        "selection_signature": payload["selection_signature"],
                        "feature_count": int(payload["feature_count"]),
                        "accuracy": float(payload["accuracy"]),
                        "macro_f1": float(payload["macro_f1"]),
                        "previous_accuracy": float(previous["previous_accuracy"]),
                        "previous_macro_f1": float(previous["previous_macro_f1"]),
                        "delta_accuracy_vs_previous": float(payload["accuracy"] - previous["previous_accuracy"]),
                        "delta_macro_f1_vs_previous": float(payload["macro_f1"] - previous["previous_macro_f1"]),
                        "confusion_matrix": payload["confusion_matrix"],
                        "labels": payload["labels"],
                    }
                )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["task", "method", "variant", "k"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["task", "method", "variant"]).reset_index(drop=True)
    write_parquet(candidate_df, output_root / "clamber_token_cloud_knnclass_candidates.parquet")
    write_parquet(final_df, output_root / "clamber_token_cloud_knnclass_results.parquet")

    report_lines = [
        "# CLAMBER Token-Cloud k-Nearest-In-Class Wasserstein Features",
        "",
        f"- Model: `{MODEL_LABEL}`",
        f"- Fixed token-cloud views: `4way single=0`, `4way multi=0 | 14`, `9way single=0`, `9way multi=0 | 14`",
        f"- Candidate k values: `{', '.join(str(k) for k in ks)}`",
        f"- Added feature family: mean Wasserstein distance to the `k` nearest train diagrams in each class",
        f"- `k` selected on a validation split inside the training set for each task / method / variant",
        "",
        "Variants:",
        "- `baseline`: current token-cloud feature set",
        "- `baseline_plus_knnclass`: current token-cloud feature set plus k-nearest-in-class H0 Wasserstein features",
        "- `no_proto_plus_knnclass`: remove pooled prototype-distance features and use k-nearest-in-class H0 Wasserstein features instead",
        "",
    ]

    for task in ["4way", "9way"]:
        task_df = final_df.loc[final_df["task"].eq(task)].copy()
        report_lines.extend(
            [
                f"## {task.upper()}",
                "",
                "| Method | Variant | Best k | Macro-F1 | Delta vs current | Acc | Delta vs current | Features | View |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in task_df.iterrows():
            best_k = "-" if pd.isna(row["best_k"]) else str(int(row["best_k"]))
            report_lines.append(
                f"| {row['method']} | {row['variant']} | {best_k} | {float(row['macro_f1']):.4f} | "
                f"{float(row['delta_macro_f1_vs_previous']):+.4f} | {float(row['accuracy']):.4f} | "
                f"{float(row['delta_accuracy_vs_previous']):+.4f} | {int(row['feature_count'])} | "
                f"`{row['selection_signature']}` |"
            )
        report_lines.append("")

    write_markdown(output_root / "clamber_token_cloud_knnclass_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
