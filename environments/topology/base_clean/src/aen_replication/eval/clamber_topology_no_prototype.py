"""CLAMBER topology classification without prototype-distance features.

This reruns the token-cloud topology classifiers after removing the prototype
distance features:

- Wasserstein distance to class prototypes
- Bottleneck distance to class prototypes

The cached topology feature tables are reused; only the classifier feature set
changes. Two label spaces are evaluated:

- regrouped 4-way CLAMBER, excluding NK
- fine-grained 9-way CLAMBER, including none and NK
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

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

METHOD_LABELS = {
    "token_cloud_single": "Topology Single",
    "token_cloud_multilayer": "Topology Multi",
}

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/meta_llama_llama_3_1_8b_instruct",
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/mistralai_mistral_7b_instruct_v0_3",
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/google_gemma_7b_it",
    },
]

PROTOTYPE_MARKERS = ("_wasserstein_to_", "_bottleneck_to_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_topology_no_prototype",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _is_prototype_distance_column(column: str) -> bool:
    return any(marker in column for marker in PROTOTYPE_MARKERS)


def _prototype_distance_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in _topology_feature_columns(feature_df) if _is_prototype_distance_column(column)]


def _drop_prototype_distance_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = _prototype_distance_columns(feature_df)
    return feature_df.drop(columns=drop_columns, errors="ignore").copy()


def _load_base_feature_df(subclass_root: Path) -> pd.DataFrame:
    feature_path = subclass_root / "clamber_token_cloud_all_layer_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing cached CLAMBER token-cloud features: {feature_path}")
    return pd.read_parquet(feature_path).copy()


def _prepare_task_frame(feature_df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    df = _drop_prototype_distance_features(feature_df)
    if task == "4way":
        df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
        df["group4"] = df["subclass"].map(GROUP4_MAP)
        return df.reset_index(drop=True), "group4"
    if task == "9way":
        return df.reset_index(drop=True), "subclass"
    raise ValueError(f"Unsupported task: {task}")


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


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _topology_columns(feature_df: pd.DataFrame) -> list[str]:
    columns = _topology_feature_columns(feature_df)
    prototype_columns = [column for column in columns if _is_prototype_distance_column(column)]
    if prototype_columns:
        raise ValueError(f"Prototype-distance columns remain in feature frame: {prototype_columns}")
    return columns


def _evaluate_single_layers(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_all = feature_df.loc[feature_df["split"].eq("train")].copy()
    test_all = feature_df.loc[feature_df["split"].eq("test")].copy()

    for layer in sorted(feature_df["layer"].unique().tolist()):
        train_df = train_all.loc[train_all["layer"].eq(layer)].copy()
        val_train_idx, val_idx = _split_indices(train_df[label_col], val_fraction=val_fraction, seed=seed + int(layer))
        feature_cols = _topology_columns(train_df)
        x_train = train_df.iloc[val_train_idx].loc[:, feature_cols].to_numpy(dtype=float)
        y_train = train_df.iloc[val_train_idx][label_col].astype(str).to_numpy()
        x_val = train_df.iloc[val_idx].loc[:, feature_cols].to_numpy(dtype=float)
        y_val = train_df.iloc[val_idx][label_col].astype(str).to_numpy()
        labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
        clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 100 + int(layer))
        y_pred = clf.predict(scaler.transform(x_val))
        metrics = _compute_metrics(y_val, y_pred, labels)
        rows.append(
            {
                "method": "token_cloud_single",
                "layer": int(layer),
                "selection_signature": str(int(layer)),
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_accuracy": float(metrics["accuracy"]),
                "feature_count": int(len(feature_cols)),
            }
        )

    candidate_df = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()

    best_layer = int(best["layer"])
    train_df = train_all.loc[train_all["layer"].eq(best_layer)].copy()
    test_df = test_all.loc[test_all["layer"].eq(best_layer)].copy()
    feature_cols = _topology_columns(train_df)
    labels = sorted(set(train_df[label_col].astype(str).tolist()) | set(test_df[label_col].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df[label_col].astype(str).to_numpy(),
        seed=seed + 999,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    test_metrics = _compute_metrics(test_df[label_col].astype(str).to_numpy(), y_pred, labels)

    final = {
        "method": "token_cloud_single",
        "layer": best_layer,
        "selection_signature": str(best_layer),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
    }
    return candidate_df, final


def _make_multilayer_frame(
    feature_df: pd.DataFrame,
    *,
    layers: list[int],
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    prototype_cols = [column for column in feature_cols if _is_prototype_distance_column(column)]
    if prototype_cols:
        raise ValueError(f"Prototype-distance columns remain in multilayer features: {prototype_cols}")
    return train_df, test_df, feature_cols


def _evaluate_multilayer(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    layers = sorted(feature_df["layer"].unique().tolist())
    rows: list[dict[str, Any]] = []

    for width in [2, min(3, len(layers))]:
        if width > len(layers):
            continue
        for combo in itertools.combinations(layers, width):
            train_df, _, feature_cols = _make_multilayer_frame(feature_df, layers=list(combo), label_col=label_col)
            tr_idx, val_idx = _split_indices(train_df[label_col], val_fraction=val_fraction, seed=seed + sum(combo) + width)
            x_train = train_df.iloc[tr_idx].loc[:, feature_cols].to_numpy(dtype=float)
            y_train = train_df.iloc[tr_idx][label_col].astype(str).to_numpy()
            x_val = train_df.iloc[val_idx].loc[:, feature_cols].to_numpy(dtype=float)
            y_val = train_df.iloc[val_idx][label_col].astype(str).to_numpy()
            labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
            clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 200 + sum(combo))
            y_pred = clf.predict(scaler.transform(x_val))
            metrics = _compute_metrics(y_val, y_pred, labels)
            rows.append(
                {
                    "method": "token_cloud_multilayer",
                    "layer": -1,
                    "selection_signature": " | ".join(str(item) for item in combo),
                    "selection_size": int(len(combo)),
                    "val_macro_f1": float(metrics["macro_f1"]),
                    "val_accuracy": float(metrics["accuracy"]),
                    "feature_count": int(len(feature_cols)),
                }
            )

    candidate_df = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best_layers = [int(item.strip()) for item in str(best["selection_signature"]).split("|")]
    train_df, test_df, feature_cols = _make_multilayer_frame(feature_df, layers=best_layers, label_col=label_col)
    labels = sorted(set(train_df[label_col].astype(str).tolist()) | set(test_df[label_col].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df[label_col].astype(str).to_numpy(),
        seed=seed + 1999,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    test_metrics = _compute_metrics(test_df[label_col].astype(str).to_numpy(), y_pred, labels)
    final = {
        "method": "token_cloud_multilayer",
        "layer": -1,
        "selection_signature": str(best["selection_signature"]),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
    }
    return candidate_df, final


def _task_sample_counts(feature_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    single_layer = int(sorted(feature_df["layer"].unique().tolist())[0])
    df = feature_df.loc[feature_df["layer"].eq(single_layer)].copy()
    return (
        df.groupby([label_col, "split"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={label_col: "label"})
    )


def _load_previous_topology_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    previous_specs = [
        (
            "4way",
            Path(
                "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet"
            ),
        ),
        (
            "9way",
            Path(
                "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_9way_comparison/clamber_9way_comparison_final_metrics.parquet"
            ),
        ),
    ]
    for task, path in previous_specs:
        if not path.exists():
            continue
        df = pd.read_parquet(path).copy()
        df = df.loc[df["method"].isin(METHOD_LABELS)].copy()
        df["task"] = task
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    previous = pd.concat(rows, ignore_index=True)
    return previous.loc[
        :,
        [
            "task",
            "model",
            "method",
            "accuracy",
            "macro_f1",
            "feature_count",
            "layer",
            "selection_signature",
        ],
    ].rename(
        columns={
            "accuracy": "previous_accuracy",
            "macro_f1": "previous_macro_f1",
            "feature_count": "previous_feature_count",
            "layer": "previous_layer",
            "selection_signature": "previous_selection_signature",
        }
    )


def _format_view(row: pd.Series) -> str:
    if str(row["method"]) == "token_cloud_multilayer":
        layers = [item.strip() for item in str(row["selection_signature"]).split("|")]
        return f"layers {', '.join(layers)}"
    return f"layer {int(row['layer'])}"


def _append_result_table(report_lines: list[str], df: pd.DataFrame, *, title: str) -> None:
    report_lines.extend(
        [
            f"## {title}",
            "",
            "| Model | Method | Macro-F1 | Acc | Features | View |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in df.iterrows():
        report_lines.append(
            f"| {row['model_label']} | {METHOD_LABELS[str(row['method'])]} | {float(row['macro_f1']):.4f} | "
            f"{float(row['accuracy']):.4f} | {int(row['feature_count'])} | {_format_view(row)} |"
        )
    report_lines.append("")


def _append_delta_table(report_lines: list[str], df: pd.DataFrame, *, title: str) -> None:
    report_lines.extend(
        [
            f"## {title}",
            "",
            "| Model | Method | No-proto Macro-F1 | Previous Macro-F1 | Delta | No-proto Acc | Previous Acc | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in df.iterrows():
        report_lines.append(
            f"| {row['model_label']} | {METHOD_LABELS[str(row['method'])]} | {float(row['macro_f1']):.4f} | "
            f"{float(row['previous_macro_f1']):.4f} | {float(row['macro_f1'] - row['previous_macro_f1']):+.4f} | "
            f"{float(row['accuracy']):.4f} | {float(row['previous_accuracy']):.4f} | "
            f"{float(row['accuracy'] - row['previous_accuracy']):+.4f} |"
        )
    report_lines.append("")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))

    final_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    dropped_columns_by_model: dict[str, list[str]] = {}

    for model_index, spec in enumerate(MODEL_SPECS):
        base_df = _load_base_feature_df(Path(spec["subclass_root"]))
        dropped_columns_by_model[str(spec["label"])] = _prototype_distance_columns(base_df)

        for task_index, task in enumerate(["4way", "9way"]):
            task_df, label_col = _prepare_task_frame(base_df, task)
            sample_counts = _task_sample_counts(task_df, label_col)
            sample_counts["model"] = spec["slug"]
            sample_counts["model_label"] = spec["label"]
            sample_counts["task"] = task
            count_rows.extend(sample_counts.to_dict(orient="records"))

            seed = args.seed + 100 * model_index + 1000 * task_index
            single_candidates, single_final = _evaluate_single_layers(
                task_df,
                label_col=label_col,
                seed=seed,
                val_fraction=args.val_fraction,
            )
            single_candidates["model"] = spec["slug"]
            single_candidates["model_label"] = spec["label"]
            single_candidates["task"] = task
            candidate_rows.extend(single_candidates.to_dict(orient="records"))

            multi_candidates, multi_final = _evaluate_multilayer(
                task_df,
                label_col=label_col,
                seed=seed,
                val_fraction=args.val_fraction,
            )
            multi_candidates["model"] = spec["slug"]
            multi_candidates["model_label"] = spec["label"]
            multi_candidates["task"] = task
            candidate_rows.extend(multi_candidates.to_dict(orient="records"))

            for final in [single_final, multi_final]:
                final_rows.append(
                    {
                        "task": task,
                        "model": spec["slug"],
                        "model_label": spec["label"],
                        **final,
                    }
                )

    final_df = pd.DataFrame(final_rows).sort_values(["task", "model_label", "method"]).reset_index(drop=True)
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["task", "model_label", "method"]).reset_index(drop=True)
    counts_df = pd.DataFrame(count_rows).sort_values(["task", "model_label", "label"]).reset_index(drop=True)

    previous_df = _load_previous_topology_metrics()
    if previous_df.empty:
        comparison_df = pd.DataFrame()
    else:
        comparison_df = final_df.merge(previous_df, on=["task", "model", "method"], how="left")

    report_lines = [
        "# CLAMBER Topology Without Prototype Distances",
        "",
        "This rerun removes the prototype-distance topology features from classification:",
        "- `h0/h1_wasserstein_to_*`",
        "- `h0/h1_bottleneck_to_*`",
        "",
        "The cached train/test split is unchanged. Layer selection uses a stratified validation split inside the training set.",
        "",
        "Label spaces:",
        "- `4way`: `ambiguity`, `missing_condition`, `conflicting_condition`, `clear`; `NK` excluded",
        "- `9way`: original CLAMBER subclasses, including `NK` and `none`",
        "",
    ]

    first_model = next(iter(dropped_columns_by_model))
    report_lines.extend(
        [
            "Dropped columns:",
            "",
            ", ".join(f"`{column}`" for column in dropped_columns_by_model[first_model]),
            "",
        ]
    )

    _append_result_table(report_lines, final_df.loc[final_df["task"].eq("4way")], title="4-Way Results")
    _append_result_table(report_lines, final_df.loc[final_df["task"].eq("9way")], title="9-Way Results")

    if not comparison_df.empty and comparison_df["previous_macro_f1"].notna().any():
        _append_delta_table(
            report_lines,
            comparison_df.loc[comparison_df["task"].eq("4way")],
            title="4-Way Delta vs Prototype-Distance Version",
        )
        _append_delta_table(
            report_lines,
            comparison_df.loc[comparison_df["task"].eq("9way")],
            title="9-Way Delta vs Prototype-Distance Version",
        )

    report_lines.extend(
        [
            "## Sample Counts",
            "",
            "| Task | Model | Label | Train | Test |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for _, row in counts_df.iterrows():
        report_lines.append(
            f"| {row['task']} | {row['model_label']} | {row['label']} | {int(row.get('train', 0))} | {int(row.get('test', 0))} |"
        )
    report_lines.append("")

    write_parquet(final_df, output_root / "clamber_topology_no_prototype_final_metrics.parquet")
    write_parquet(candidate_df, output_root / "clamber_topology_no_prototype_candidates.parquet")
    write_parquet(counts_df, output_root / "clamber_topology_no_prototype_sample_counts.parquet")
    if not comparison_df.empty:
        write_parquet(comparison_df, output_root / "clamber_topology_no_prototype_delta_vs_previous.parquet")
    write_markdown(output_root / "clamber_topology_no_prototype_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
