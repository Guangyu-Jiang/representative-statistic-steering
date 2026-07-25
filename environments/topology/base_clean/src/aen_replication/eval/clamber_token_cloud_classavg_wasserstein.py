"""CLAMBER token-cloud follow-up with class-average diagram distances.

This evaluates a variant of the current token-cloud topology classifier that
adds exact per-class H0 Wasserstein features:

    mean_{train i in class c} W(D_q, D_i)

where D_q is the query question's H0 persistence diagram and D_i are the
training-set diagrams from class c. For train examples, self-distance is
excluded.

The experiment is intentionally scoped to the currently selected LLaMA CLAMBER
views:

- 4-way regrouped CLAMBER: single-layer `0`, multilayer `0 | 14`
- 9-way CLAMBER: single-layer `0`, multilayer `0 | 14`

It reports three variants on the same fixed views:

- `baseline`: current token-cloud feature set
- `baseline_plus_classavg`: current token-cloud features plus class-average
  H0 Wasserstein features
- `no_proto_plus_classavg`: drop current pooled-prototype distance features and
  replace them with class-average H0 Wasserstein features
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.independent_topology_classifier import _compute_diagrams, _safe_wasserstein
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, slugify, write_markdown, write_parquet

GROUP4_MAP = {
    "polysemy": "ambiguity",
    "co-reference": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "ICL": "conflicting_condition",
    "none": "clear",
}

GROUP4_ORDER = ["ambiguity", "missing_condition", "conflicting_condition", "clear"]

PROTOTYPE_MARKERS = ("_wasserstein_to_", "_bottleneck_to_")

MODEL_SLUG = "meta_llama_llama_3_1_8b_instruct"
MODEL_LABEL = "LLaMA 3.1 8B"
BASE_FEATURE_PATH = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/"
    "meta_llama_llama_3_1_8b_instruct/clamber_token_cloud_all_layer_features.parquet"
)
PREVIOUS_METRIC_PATHS = {
    "4way": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
        "clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet"
    ),
    "9way": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
        "clamber_9way_comparison/clamber_9way_comparison_final_metrics.parquet"
    ),
}

SUBCLASS_ORDER = ["ICL", "NK", "co-reference", "none", "polysemy", "what", "when", "where", "whom"]

_DIST_QUERY_IDS: list[str] = []
_DIST_QUERY_DIAGRAMS: list[np.ndarray] = []
_DIST_TRAIN_IDS: list[str] = []
_DIST_TRAIN_LABELS: list[str] = []
_DIST_TRAIN_DIAGRAMS: list[np.ndarray] = []
_DIST_LABELS: list[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_token_cloud_classavg_wasserstein",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=min(24, max(1, (os.cpu_count() or 8) - 1)))
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--parallel-jobs", type=int, default=12)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _fit_multiclass_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _is_prototype_distance_column(column: str) -> bool:
    return any(marker in column for marker in PROTOTYPE_MARKERS)


def _drop_prototype_distance_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in _topology_feature_columns(feature_df) if _is_prototype_distance_column(column)]
    return feature_df.drop(columns=drop_columns, errors="ignore").copy()


def _classavg_subclass_column(subclass: str) -> str:
    return f"h0_classavg_subclass__{slugify(subclass)}"


def _classavg_count_column(subclass: str) -> str:
    return f"meta_classavg_count_subclass__{slugify(subclass)}"


def _classavg_group4_column(group: str) -> str:
    return f"h0_classavg_group4__{slugify(group)}"


def _subclasses_for_group(group: str) -> list[str]:
    return [subclass for subclass, mapped in GROUP4_MAP.items() if mapped == group]


def _load_selected_views() -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for task, path in PREVIOUS_METRIC_PATHS.items():
        df = pd.read_parquet(path).copy()
        df = df.loc[df["model"].eq(MODEL_SLUG) & df["method"].isin(["token_cloud_single", "token_cloud_multilayer"])].copy()
        if df.empty:
            raise FileNotFoundError(f"Could not find stored LLaMA token-cloud selections for task={task}: {path}")
        selected[task] = {}
        for method in ["token_cloud_single", "token_cloud_multilayer"]:
            row = df.loc[df["method"].eq(method)].iloc[0]
            signature_raw = row["selection_signature"]
            signature = str(int(row["layer"])) if pd.isna(signature_raw) else str(signature_raw)
            if method == "token_cloud_single":
                layers = [int(signature)]
            else:
                layers = [int(item.strip()) for item in signature.split("|")]
            selected[task][method] = {
                "selection_signature": signature,
                "layers": layers,
                "previous_accuracy": float(row["accuracy"]),
                "previous_macro_f1": float(row["macro_f1"]),
            }
    return selected


def _extract_clamber_clouds(
    *,
    config: dict[str, Any],
    layers: list[int],
    seed: int,
) -> pd.DataFrame:
    classifier_config = dict(config["token_cloud_topology_classifier"])
    classifier_config["_seed"] = int(seed)
    classifier_config["parallel_jobs"] = int(classifier_config.get("parallel_jobs", 12))
    classifier_config["candidate_layers"] = list(layers)

    bundle = load_hf_model(config["model"], classifier_config)
    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    dataset_df = pd.read_parquet(dataset_path).copy()
    prepared_df, prepared_text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle,
        text_column=str(classifier_config.get("text_column", "text")),
        use_chat_template=bool(classifier_config.get("use_chat_template", False)),
        system_prompt=classifier_config.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]

    train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=layers,
        config=classifier_config,
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
    cloud_df = _extract_reduced_clouds(
        bundle=bundle,
        df=prepared_df.reset_index(drop=True),
        text_column="_token_cloud_text",
        layers=layers,
        reducers=reducers,
        config=classifier_config,
    )
    meta = prepared_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
    return cloud_df.merge(meta, on="example_id", how="left")


def _h0_diagram_row(row: dict[str, Any], *, maxdim: int, coeff: int, distance_metric: str) -> dict[str, Any]:
    diagrams = _compute_diagrams(
        np.asarray(row["cloud"], dtype=float),
        maxdim=maxdim,
        coeff=coeff,
        distance_metric=distance_metric,
    )
    return {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "split": str(row["split"]),
        "layer": int(row["layer"]),
        "subclass": str(row["subclass"]),
        "h0_diagram": diagrams[0],
    }


def _compute_h0_diagrams(
    cloud_df: pd.DataFrame,
    *,
    maxdim: int,
    coeff: int,
    distance_metric: str,
    parallel_jobs: int,
) -> pd.DataFrame:
    rows = cloud_df.to_dict(orient="records")
    diagram_rows = joblib.Parallel(n_jobs=max(1, int(parallel_jobs)), backend="loky")(
        joblib.delayed(_h0_diagram_row)(row, maxdim=maxdim, coeff=coeff, distance_metric=distance_metric) for row in rows
    )
    return pd.DataFrame(diagram_rows)


def _init_distance_worker(
    query_ids: list[str],
    query_diagrams: list[np.ndarray],
    train_ids: list[str],
    train_labels: list[str],
    train_diagrams: list[np.ndarray],
    labels: list[str],
) -> None:
    global _DIST_QUERY_IDS, _DIST_QUERY_DIAGRAMS, _DIST_TRAIN_IDS, _DIST_TRAIN_LABELS, _DIST_TRAIN_DIAGRAMS, _DIST_LABELS
    _DIST_QUERY_IDS = query_ids
    _DIST_QUERY_DIAGRAMS = query_diagrams
    _DIST_TRAIN_IDS = train_ids
    _DIST_TRAIN_LABELS = train_labels
    _DIST_TRAIN_DIAGRAMS = train_diagrams
    _DIST_LABELS = labels


def _distance_chunk(start: int, stop: int) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
    label_to_index = {label: index for index, label in enumerate(_DIST_LABELS)}
    means_by_label = {label: np.zeros(stop - start, dtype=np.float32) for label in _DIST_LABELS}
    counts_by_label = {label: np.zeros(stop - start, dtype=np.int32) for label in _DIST_LABELS}
    for local_index, query_index in enumerate(range(start, stop)):
        query_id = _DIST_QUERY_IDS[query_index]
        query_diagram = _DIST_QUERY_DIAGRAMS[query_index]
        sums = np.zeros(len(_DIST_LABELS), dtype=np.float64)
        counts = np.zeros(len(_DIST_LABELS), dtype=np.int32)
        for train_id, train_label, train_diagram in zip(_DIST_TRAIN_IDS, _DIST_TRAIN_LABELS, _DIST_TRAIN_DIAGRAMS):
            if train_id == query_id:
                continue
            label_index = label_to_index[train_label]
            sums[label_index] += _safe_wasserstein(query_diagram, train_diagram)
            counts[label_index] += 1
        for label, label_index in label_to_index.items():
            counts_by_label[label][local_index] = counts[label_index]
            means_by_label[label][local_index] = (
                np.float32(sums[label_index] / counts[label_index]) if counts[label_index] > 0 else np.float32(0.0)
            )
    return start, means_by_label, counts_by_label


def _layer_classavg_features(
    layer_df: pd.DataFrame,
    *,
    workers: int,
    chunk_size: int,
) -> pd.DataFrame:
    layer_df = layer_df.sort_values("example_id").reset_index(drop=True)
    train_df = layer_df.loc[layer_df["split"].eq("train")].copy().sort_values("example_id").reset_index(drop=True)
    labels = [label for label in SUBCLASS_ORDER if label in set(train_df["subclass"].astype(str))]
    if not labels:
        raise ValueError(f"No training subclasses found for layer {int(layer_df.iloc[0]['layer'])}")

    query_ids = layer_df["example_id"].astype(str).tolist()
    query_diagrams = layer_df["h0_diagram"].tolist()
    train_ids = train_df["example_id"].astype(str).tolist()
    train_labels = train_df["subclass"].astype(str).tolist()
    train_diagrams = train_df["h0_diagram"].tolist()

    chunk_ranges = [
        (start, min(start + int(chunk_size), len(layer_df)))
        for start in range(0, len(layer_df), int(chunk_size))
    ]

    mean_storage = {label: np.zeros(len(layer_df), dtype=np.float32) for label in labels}
    count_storage = {label: np.zeros(len(layer_df), dtype=np.int32) for label in labels}

    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_init_distance_worker,
        initargs=(query_ids, query_diagrams, train_ids, train_labels, train_diagrams, labels),
    ) as executor:
        futures = {
            executor.submit(_distance_chunk, start, stop): (start, stop)
            for start, stop in chunk_ranges
        }
        progress = tqdm(total=len(chunk_ranges), desc=f"layer_{int(layer_df.iloc[0]['layer']):02d}_classavg", leave=False)
        for future in as_completed(futures):
            start, means_by_label, counts_by_label = future.result()
            stop = futures[future][1]
            for label in labels:
                mean_storage[label][start:stop] = means_by_label[label]
                count_storage[label][start:stop] = counts_by_label[label]
            progress.update(1)
        progress.close()

    result = layer_df.loc[:, ["example_id", "layer", "subclass", "split"]].copy()
    for label in labels:
        result[_classavg_subclass_column(label)] = mean_storage[label].astype(float)
        result[_classavg_count_column(label)] = count_storage[label].astype(int)

    for group in GROUP4_ORDER:
        subclasses = _subclasses_for_group(group)
        sum_values = np.zeros(len(result), dtype=np.float64)
        count_values = np.zeros(len(result), dtype=np.int32)
        for subclass in subclasses:
            mean_column = _classavg_subclass_column(subclass)
            count_column = _classavg_count_column(subclass)
            if mean_column not in result.columns or count_column not in result.columns:
                continue
            counts = result[count_column].to_numpy(dtype=np.int32)
            means = result[mean_column].to_numpy(dtype=float)
            sum_values += means * counts
            count_values += counts
        result[_classavg_group4_column(group)] = np.divide(
            sum_values,
            np.maximum(count_values, 1),
            out=np.zeros(len(result), dtype=np.float64),
            where=count_values > 0,
        )
    return result.drop(columns=["subclass", "split"])


def _compute_classavg_feature_frame(
    diagram_df: pd.DataFrame,
    *,
    output_path: Path,
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
            layer_features = _layer_classavg_features(layer_df, workers=workers, chunk_size=chunk_size)
            write_parquet(layer_features, layer_path)
        parts.append(layer_features)
    classavg_df = pd.concat(parts, ignore_index=True)
    write_parquet(classavg_df, output_path)
    return classavg_df


def _task_frame(
    merged_df: pd.DataFrame,
    *,
    task: str,
    variant: str,
) -> tuple[pd.DataFrame, str]:
    if variant == "no_proto_plus_classavg":
        df = _drop_prototype_distance_features(merged_df)
    else:
        df = merged_df.copy()

    subclass_classavg_columns = [column for column in df.columns if column.startswith("h0_classavg_subclass__")]
    group4_classavg_columns = [column for column in df.columns if column.startswith("h0_classavg_group4__")]
    count_columns = [column for column in df.columns if column.startswith("meta_classavg_count_subclass__")]

    if task == "4way":
        df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
        df["group4"] = df["subclass"].map(GROUP4_MAP)
        df = df.drop(columns=subclass_classavg_columns + count_columns, errors="ignore")
        return df.reset_index(drop=True), "group4"
    if task == "9way":
        df = df.drop(columns=group4_classavg_columns + count_columns, errors="ignore")
        return df.reset_index(drop=True), "subclass"
    raise ValueError(f"Unsupported task: {task}")


def _evaluate_single(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    layer: int,
    seed: int,
) -> dict[str, Any]:
    train_df = feature_df.loc[feature_df["split"].eq("train") & feature_df["layer"].eq(layer)].copy()
    test_df = feature_df.loc[feature_df["split"].eq("test") & feature_df["layer"].eq(layer)].copy()
    feature_cols = _topology_feature_columns(train_df)
    labels = sorted(set(train_df[label_col].astype(str).tolist()) | set(test_df[label_col].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df[label_col].astype(str).to_numpy(),
        seed=seed,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    metrics = _compute_metrics(test_df[label_col].astype(str).to_numpy(), y_pred, labels)
    return {
        "layer": int(layer),
        "selection_signature": str(int(layer)),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
    }


def _evaluate_multilayer(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    layers: list[int],
    seed: int,
) -> dict[str, Any]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    label_lookup = feature_df.loc[:, ["example_id", label_col]].drop_duplicates()
    train_df = train_df.merge(label_lookup, on="example_id", how="left")
    test_df = test_df.merge(label_lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    labels = sorted(set(train_df[label_col].astype(str).tolist()) | set(test_df[label_col].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df[label_col].astype(str).to_numpy(),
        seed=seed,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    metrics = _compute_metrics(test_df[label_col].astype(str).to_numpy(), y_pred, labels)
    return {
        "layer": -1,
        "selection_signature": " | ".join(str(int(layer)) for layer in layers),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
    }


def main() -> None:
    args = parse_args()
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
    classavg_df = _compute_classavg_feature_frame(
        diagram_df,
        output_path=output_root / "llama_clamber_h0_classavg_features.parquet",
        workers=args.workers,
        chunk_size=args.chunk_size,
        force_recompute=bool(args.force_recompute),
    )

    merged_df = base_df.merge(
        classavg_df.drop_duplicates(subset=["example_id", "layer"]),
        on=["example_id", "layer"],
        how="left",
    )

    results: list[dict[str, Any]] = []
    for task in ["4way", "9way"]:
        for variant in ["baseline", "baseline_plus_classavg", "no_proto_plus_classavg"]:
            if variant == "baseline":
                task_df, label_col = _task_frame(base_df, task=task, variant="baseline_plus_classavg")
            else:
                task_df, label_col = _task_frame(merged_df, task=task, variant=variant)

            single_payload = _evaluate_single(
                task_df,
                label_col=label_col,
                layer=int(selected_views[task]["token_cloud_single"]["layers"][0]),
                seed=args.seed + (0 if task == "4way" else 100),
            )
            multi_payload = _evaluate_multilayer(
                task_df,
                label_col=label_col,
                layers=[int(layer) for layer in selected_views[task]["token_cloud_multilayer"]["layers"]],
                seed=args.seed + (10 if task == "4way" else 110),
            )

            for method, payload in [
                ("token_cloud_single", single_payload),
                ("token_cloud_multilayer", multi_payload),
            ]:
                previous = selected_views[task][method]
                results.append(
                    {
                        "task": task,
                        "variant": variant,
                        "model": MODEL_SLUG,
                        "model_label": MODEL_LABEL,
                        "method": method,
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

    result_df = pd.DataFrame(results).sort_values(["task", "method", "variant"]).reset_index(drop=True)
    write_parquet(result_df, output_root / "clamber_token_cloud_classavg_wasserstein_results.parquet")

    subclass_cols = [column for column in classavg_df.columns if column.startswith("h0_classavg_subclass__")]
    group_cols = [column for column in classavg_df.columns if column.startswith("h0_classavg_group4__")]
    report_lines = [
        "# CLAMBER Token-Cloud Class-Average Wasserstein Features",
        "",
        f"- Model: `{MODEL_LABEL}`",
        f"- Fixed views from stored token-cloud baselines: `4way single=0`, `4way multi=0 | 14`, `9way single=0`, `9way multi=0 | 14`",
        f"- Added feature family: exact `H0` mean Wasserstein distance from each question diagram to all train-set diagrams in each class",
        f"- Train examples exclude self-distance when averaging",
        "",
        "Feature families by variant:",
        "- `baseline`: current token-cloud feature set",
        "- `baseline_plus_classavg`: current token-cloud features plus class-average H0 Wasserstein features",
        "- `no_proto_plus_classavg`: remove pooled prototype-distance features and use class-average H0 Wasserstein features instead",
        "",
        f"- Added 9-way class-average columns per layer: `{len(subclass_cols)}`",
        f"- Added 4-way class-average columns per layer: `{len(group_cols)}`",
        "",
    ]

    for task in ["4way", "9way"]:
        task_df = result_df.loc[result_df["task"].eq(task)].copy()
        report_lines.extend(
            [
                f"## {task.upper()}",
                "",
                "| Method | Variant | Macro-F1 | Delta vs current | Acc | Delta vs current | Features | View |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in task_df.iterrows():
            report_lines.append(
                f"| {row['method']} | {row['variant']} | {float(row['macro_f1']):.4f} | "
                f"{float(row['delta_macro_f1_vs_previous']):+.4f} | {float(row['accuracy']):.4f} | "
                f"{float(row['delta_accuracy_vs_previous']):+.4f} | {int(row['feature_count'])} | "
                f"{row['selection_signature']} |"
            )
        report_lines.append("")

    write_markdown(output_root / "clamber_token_cloud_classavg_wasserstein_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
