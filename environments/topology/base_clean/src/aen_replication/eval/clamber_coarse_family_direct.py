"""Direct CLAMBER coarse-family classification.

This script reuses the cached feature views from the CLAMBER subclass runs, but
trains classifiers directly on a 5-way coarse label space:

- ambiguity
- missing_condition
- conflicting_condition
- epistemic_or_unknown
- clear

The goal is to test whether topology improves when the target is the coarse
family itself, rather than the original 9-way subclass problem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

TRAIN_SIZES = [20, 40, 80, 160]
METHOD_ORDER = [
    "full_probe",
    "aen_only",
    "token_cloud_single",
    "token_cloud_multilayer",
]
METHOD_LABELS = {
    "full_probe": "Full Probe",
    "aen_only": "AEN",
    "token_cloud_single": "Topology Single",
    "token_cloud_multilayer": "Topology Multi",
}
PLOT_COLORS = {
    "full_probe": "#1f77b4",
    "aen_only": "#d62728",
    "token_cloud_single": "#2ca02c",
    "token_cloud_multilayer": "#9467bd",
}
GROUP_MAP = {
    "ICL": "conflicting_condition",
    "NK": "epistemic_or_unknown",
    "co-reference": "ambiguity",
    "polysemy": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "none": "clear",
}
MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/meta_llama_llama_3_1_8b_instruct",
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/mistralai_mistral_7b_instruct_v0_3",
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/google_gemma_7b_it",
    },
]


@dataclass
class PreparedView:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    labels: list[str]
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_coarse_family_direct",
    )
    return parser.parse_args()


def _normalize_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            body = stripped[1:-1].strip()
            if not body:
                return []
            return [item.strip().strip("'").strip('"') for item in body.split(",")]
        return [stripped]
    return list(value)


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
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }


def _sample_per_class_indices(y: np.ndarray, *, train_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    y_series = pd.Series(y.astype(str))
    for label in sorted(y_series.unique()):
        label_idx = np.flatnonzero(y_series.to_numpy() == label)
        if len(label_idx) < train_per_class:
            raise ValueError(f"Label {label} has only {len(label_idx)} rows, cannot sample {train_per_class}.")
        picked = rng.choice(label_idx, size=train_per_class, replace=False)
        indices.append(np.sort(picked))
    return np.sort(np.concatenate(indices))


def _evaluate_subset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    train_per_class: int,
    seed: int,
) -> dict[str, Any]:
    subset_idx = _sample_per_class_indices(train_y, train_per_class=train_per_class, seed=seed)
    labels = sorted({str(label) for label in np.concatenate([train_y, test_y]).tolist()})
    clf, scaler = _fit_multiclass_logistic(train_x[subset_idx], train_y[subset_idx], seed=seed)
    predictions = clf.predict(scaler.transform(test_x))
    metrics = _compute_metrics(test_y, predictions, labels)
    return {
        "train_per_class": int(train_per_class),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
    }


def _remap_to_groups(values: np.ndarray) -> np.ndarray:
    return np.asarray([GROUP_MAP[str(value)] for value in values], dtype=object)


def _load_best_rows(subclass_root: Path) -> pd.DataFrame:
    return pd.read_parquet(subclass_root / "clamber_subclass_final_metrics.parquet").copy()


def _load_mean_pool_view(
    *,
    hidden_root: Path,
    layer: int,
    aen_indices: list[int] | None,
) -> PreparedView:
    meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    train_mask = meta["split"].eq("train").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    train_x = matrix[train_mask]
    test_x = matrix[test_mask]
    if aen_indices:
        train_x = train_x[:, aen_indices]
        test_x = test_x[:, aen_indices]
    train_y = _remap_to_groups(meta.loc[train_mask, "subclass"].astype(str).to_numpy())
    test_y = _remap_to_groups(meta.loc[test_mask, "subclass"].astype(str).to_numpy())
    labels = sorted(set(train_y.tolist()) | set(test_y.tolist()))
    return PreparedView(
        train_x=np.asarray(train_x, dtype=float),
        train_y=train_y,
        test_x=np.asarray(test_x, dtype=float),
        test_y=test_y,
        labels=labels,
        meta={"layer": int(layer), "feature_count": int(train_x.shape[1])},
    )


def _load_token_cloud_feature_df(subclass_root: Path) -> pd.DataFrame:
    return pd.read_parquet(subclass_root / "clamber_token_cloud_all_layer_features.parquet")


def _load_token_cloud_single_view(feature_df: pd.DataFrame, *, layer: int) -> PreparedView:
    layer_df = feature_df.loc[feature_df["layer"].eq(layer)].copy()
    columns = _topology_feature_columns(layer_df)
    train_df = layer_df.loc[layer_df["split"].eq("train")].copy()
    test_df = layer_df.loc[layer_df["split"].eq("test")].copy()
    train_y = _remap_to_groups(train_df["subclass"].astype(str).to_numpy())
    test_y = _remap_to_groups(test_df["subclass"].astype(str).to_numpy())
    labels = sorted(set(train_y.tolist()) | set(test_y.tolist()))
    return PreparedView(
        train_x=train_df.loc[:, columns].to_numpy(dtype=float),
        train_y=train_y,
        test_x=test_df.loc[:, columns].to_numpy(dtype=float),
        test_y=test_y,
        labels=labels,
        meta={"layer": int(layer), "feature_count": int(len(columns))},
    )


def _load_token_cloud_multilayer_view(feature_df: pd.DataFrame, *, selection_signature: str) -> PreparedView:
    layers = [int(item.strip()) for item in str(selection_signature).split("|")]
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    subclass_lookup = feature_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
    train_df = train_df.merge(subclass_lookup, on="example_id", how="left")
    test_df = test_df.merge(subclass_lookup, on="example_id", how="left")
    columns = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    train_y = _remap_to_groups(train_df["subclass"].astype(str).to_numpy())
    test_y = _remap_to_groups(test_df["subclass"].astype(str).to_numpy())
    labels = sorted(set(train_y.tolist()) | set(test_y.tolist()))
    return PreparedView(
        train_x=train_df.loc[:, columns].to_numpy(dtype=float),
        train_y=train_y,
        test_x=test_df.loc[:, columns].to_numpy(dtype=float),
        test_y=test_y,
        labels=labels,
        meta={
            "selection_signature": " | ".join(str(layer) for layer in layers),
            "feature_count": int(len(columns)),
        },
    )


def _prepare_views(spec: dict[str, str]) -> dict[str, PreparedView]:
    subclass_root = Path(spec["subclass_root"])
    hidden_root = Path(spec["hidden_root"])
    final_df = _load_best_rows(subclass_root)
    feature_df = _load_token_cloud_feature_df(subclass_root)

    full_row = final_df.loc[final_df["method"].eq("full_probe")].iloc[0]
    aen_row = final_df.loc[final_df["method"].eq("aen_only")].iloc[0]
    single_row = final_df.loc[final_df["method"].eq("token_cloud_single")].iloc[0]
    multi_row = final_df.loc[final_df["method"].eq("token_cloud_multilayer")].iloc[0]
    aen_indices = [int(item) for item in _normalize_list(aen_row.get("aen_indices"))]

    return {
        "full_probe": _load_mean_pool_view(hidden_root=hidden_root, layer=int(full_row["layer"]), aen_indices=None),
        "aen_only": _load_mean_pool_view(hidden_root=hidden_root, layer=int(aen_row["layer"]), aen_indices=aen_indices),
        "token_cloud_single": _load_token_cloud_single_view(feature_df, layer=int(single_row["layer"])),
        "token_cloud_multilayer": _load_token_cloud_multilayer_view(
            feature_df,
            selection_signature=str(multi_row["selection_signature"]),
        ),
    }


def _render_lowlabel_plot(result_df: pd.DataFrame, *, model_label: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for method in METHOD_ORDER:
        sub = result_df.loc[result_df["method"].eq(method)].sort_values("train_per_class")
        axes[0].plot(
            sub["train_per_class"],
            sub["macro_f1"],
            marker="o",
            linewidth=2.0,
            color=PLOT_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            sub["train_per_class"],
            sub["accuracy"],
            marker="o",
            linewidth=2.0,
            color=PLOT_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_title("Macro-F1 vs train/class")
    axes[1].set_title("Accuracy vs train/class")
    for axis in axes:
        axis.set_xlabel("Train samples per class")
        axis.grid(True, alpha=0.25)
        axis.set_xticks(TRAIN_SIZES)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{model_label}: CLAMBER coarse-family low-label scaling", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_confusion_plot(final_df: pd.DataFrame, *, model_label: str, output_path: Path) -> None:
    labels = final_df.iloc[0]["labels"]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.0))
    axes_list = list(axes.flatten())
    for axis, method in zip(axes_list, METHOD_ORDER):
        row = final_df.loc[final_df["method"].eq(method)].iloc[0]
        matrix = np.asarray(row["confusion_matrix"], dtype=float)
        row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        normalized = matrix / row_sums
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"{METHOD_LABELS[method]}\nMacro-F1 {float(row['macro_f1']):.3f}, Acc {float(row['accuracy']):.3f}"
        )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=8)
    cbar = fig.colorbar(image, ax=axes_list, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized confusion")
    fig.suptitle(f"{model_label}: CLAMBER coarse-family confusion structure", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 0.97, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    plots_root = ensure_dir(output_root / "plots")

    final_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    report_lines = [
        "# CLAMBER Direct Coarse-Family Classification",
        "",
        "This experiment trains directly on 5 coarse families:",
        "- `ambiguity`",
        "- `missing_condition`",
        "- `conflicting_condition`",
        "- `epistemic_or_unknown`",
        "- `clear`",
        "",
        "It reuses the selected feature views from the CLAMBER 9-way runs, but remaps labels before training.",
        "",
    ]

    for model_index, spec in enumerate(MODEL_SPECS):
        views = _prepare_views(spec)
        report_lines.extend([f"## {spec['label']}", ""])
        model_final_rows: list[dict[str, Any]] = []

        for method_index, method in enumerate(METHOD_ORDER):
            view = views[method]
            labels = sorted({str(label) for label in np.concatenate([view.train_y, view.test_y]).tolist()})
            clf, scaler = _fit_multiclass_logistic(view.train_x, view.train_y, seed=1000 + 50 * model_index + method_index)
            predictions = clf.predict(scaler.transform(view.test_x))
            metrics = _compute_metrics(view.test_y, predictions, labels)
            row = {
                "model": spec["slug"],
                "model_label": spec["label"],
                "method": method,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "confusion_matrix": metrics["confusion_matrix"],
                "labels": metrics["labels"],
                "feature_count": int(view.meta["feature_count"]),
                "layer": int(view.meta["layer"]) if "layer" in view.meta else -1,
                "selection_signature": view.meta.get("selection_signature"),
            }
            final_rows.append(row)
            model_final_rows.append(row)

            for size_index, train_per_class in enumerate(TRAIN_SIZES):
                subset = _evaluate_subset(
                    view.train_x,
                    view.train_y,
                    view.test_x,
                    view.test_y,
                    train_per_class=train_per_class,
                    seed=5000 + 100 * model_index + 10 * method_index + size_index,
                )
                low_rows.append(
                    {
                        "model": spec["slug"],
                        "model_label": spec["label"],
                        "method": method,
                        **subset,
                    }
                )

        model_final_df = pd.DataFrame(model_final_rows)
        low_df = pd.DataFrame([row for row in low_rows if row["model"] == spec["slug"]])
        low_plot_path = plots_root / f"{spec['slug']}__lowlabel.png"
        conf_plot_path = plots_root / f"{spec['slug']}__confusions.png"
        _render_lowlabel_plot(low_df, model_label=spec["label"], output_path=low_plot_path)
        _render_confusion_plot(model_final_df, model_label=spec["label"], output_path=conf_plot_path)

        report_lines.extend(
            [
                "| Method | Macro-F1 | Acc | Feature Count | View |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for method in METHOD_ORDER:
            row = model_final_df.loc[model_final_df["method"].eq(method)].iloc[0]
            view_label = (
                f"layers {row['selection_signature']}"
                if pd.notna(row["selection_signature"])
                else f"layer {int(row['layer'])}"
            )
            report_lines.append(
                f"| {METHOD_LABELS[method]} | {float(row['macro_f1']):.4f} | {float(row['accuracy']):.4f} | "
                f"{int(row['feature_count'])} | {view_label} |"
            )
        report_lines.extend(
            [
                "",
                f"- Low-label plot: `{low_plot_path}`",
                f"- Confusion plot: `{conf_plot_path}`",
                "",
            ]
        )

    final_df = pd.DataFrame(final_rows)
    low_df = pd.DataFrame(low_rows)
    low_mean_df = (
        low_df.groupby("method", as_index=False)[["macro_f1", "accuracy"]]
        .mean()
        .rename(columns={"macro_f1": "mean_macro_f1", "accuracy": "mean_accuracy"})
    )

    report_lines.extend(
        [
            "## Aggregate Low-Label Means",
            "",
            "| Method | Mean Macro-F1 | Mean Acc |",
            "| --- | ---: | ---: |",
        ]
    )
    for method in METHOD_ORDER:
        row = low_mean_df.loc[low_mean_df["method"].eq(method)].iloc[0]
        report_lines.append(
            f"| {METHOD_LABELS[method]} | {float(row['mean_macro_f1']):.4f} | {float(row['mean_accuracy']):.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Main Readout",
            "",
            "- This is a direct 5-way classification experiment, not a collapse of 9-way confusion matrices.",
            "- The representation views are kept fixed to the previously selected CLAMBER 9-way settings.",
            "- If topology improves here, it supports the claim that it captures coarse ill-posedness structure better than fine subclass boundaries.",
            "",
        ]
    )

    write_parquet(final_df, output_root / "clamber_coarse_family_final_metrics.parquet")
    write_parquet(low_df, output_root / "clamber_coarse_family_lowlabel_results.parquet")
    write_markdown(output_root / "clamber_coarse_family_direct.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
