"""Direct 4-way CLAMBER regrouped classification using token-cloud topology.

The regrouping follows:

- ambiguity: polysemy, co-reference
- missing_condition: what, when, where, whom
- conflicting_condition: ICL
- clear: none

The `NK` / `UNFAMILIAR` class is excluded from this experiment.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

PLOT_COLORS = {
    "token_cloud_single": "#2ca02c",
    "token_cloud_multilayer": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-features", type=int, default=8)
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


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _load_feature_df(subclass_root: Path) -> pd.DataFrame:
    df = pd.read_parquet(subclass_root / "clamber_token_cloud_all_layer_features.parquet").copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df.reset_index(drop=True)


def _evaluate_single_layers(
    feature_df: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_all = feature_df.loc[feature_df["split"].eq("train")].copy()
    test_all = feature_df.loc[feature_df["split"].eq("test")].copy()

    for layer in sorted(feature_df["layer"].unique().tolist()):
        train_df = train_all.loc[train_all["layer"].eq(layer)].copy()
        val_train_idx, val_idx = _split_indices(train_df["group4"], val_fraction=val_fraction, seed=seed + int(layer))
        feature_cols = _topology_feature_columns(train_df)
        x_train = train_df.iloc[val_train_idx].loc[:, feature_cols].to_numpy(dtype=float)
        y_train = train_df.iloc[val_train_idx]["group4"].astype(str).to_numpy()
        x_val = train_df.iloc[val_idx].loc[:, feature_cols].to_numpy(dtype=float)
        y_val = train_df.iloc[val_idx]["group4"].astype(str).to_numpy()
        labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
        clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 100 + int(layer))
        y_pred = clf.predict(scaler.transform(x_val))
        metrics = _compute_metrics(y_val, y_pred, labels)
        rows.append(
            {
                "method": "token_cloud_single",
                "layer": int(layer),
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
    feature_cols = _topology_feature_columns(train_df)
    labels = sorted(set(train_df["group4"].astype(str).tolist()) | set(test_df["group4"].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df["group4"].astype(str).to_numpy(),
        seed=seed + 999,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    test_metrics = _compute_metrics(test_df["group4"].astype(str).to_numpy(), y_pred, labels)

    final = {
        "method": "token_cloud_single",
        "layer": best_layer,
        "feature_count": int(len(feature_cols)),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
        "selection_signature": None,
        "feature_columns": feature_cols,
    }
    return candidate_df, final


def _make_multilayer_frame(feature_df: pd.DataFrame, layers: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    lookup = feature_df.loc[:, ["example_id", "group4"]].drop_duplicates()
    train_df = train_df.merge(lookup, on="example_id", how="left")
    test_df = test_df.merge(lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    return train_df, test_df, feature_cols


def _evaluate_multilayer(
    feature_df: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    layers = sorted(feature_df["layer"].unique().tolist())
    rows: list[dict[str, Any]] = []

    for width in [2, min(3, len(layers))]:
        if width > len(layers):
            continue
        for combo in itertools.combinations(layers, width):
            train_df, test_df, feature_cols = _make_multilayer_frame(feature_df, list(combo))
            tr_idx, val_idx = _split_indices(train_df["group4"], val_fraction=val_fraction, seed=seed + sum(combo) + width)
            x_train = train_df.iloc[tr_idx].loc[:, feature_cols].to_numpy(dtype=float)
            y_train = train_df.iloc[tr_idx]["group4"].astype(str).to_numpy()
            x_val = train_df.iloc[val_idx].loc[:, feature_cols].to_numpy(dtype=float)
            y_val = train_df.iloc[val_idx]["group4"].astype(str).to_numpy()
            labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
            clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 200 + sum(combo))
            y_pred = clf.predict(scaler.transform(x_val))
            metrics = _compute_metrics(y_val, y_pred, labels)
            rows.append(
                {
                    "method": "token_cloud_multilayer",
                    "selection_signature": " | ".join(str(item) for item in combo),
                    "val_macro_f1": float(metrics["macro_f1"]),
                    "val_accuracy": float(metrics["accuracy"]),
                    "feature_count": int(len(feature_cols)),
                }
            )

    candidate_df = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best_layers = [int(item.strip()) for item in str(best["selection_signature"]).split("|")]
    train_df, test_df, feature_cols = _make_multilayer_frame(feature_df, best_layers)
    labels = sorted(set(train_df["group4"].astype(str).tolist()) | set(test_df["group4"].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df["group4"].astype(str).to_numpy(),
        seed=seed + 1999,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    test_metrics = _compute_metrics(test_df["group4"].astype(str).to_numpy(), y_pred, labels)
    final = {
        "method": "token_cloud_multilayer",
        "layer": -1,
        "feature_count": int(len(feature_cols)),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
        "selection_signature": str(best["selection_signature"]),
        "feature_columns": feature_cols,
    }
    return candidate_df, final


def _eta_squared(df: pd.DataFrame, feature: str, label_col: str) -> float:
    overall_mean = float(df[feature].mean())
    ss_total = float(((df[feature] - overall_mean) ** 2).sum())
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    for _, group in df.groupby(label_col, sort=True):
        group_mean = float(group[feature].mean())
        ss_between += float(len(group) * (group_mean - overall_mean) ** 2)
    return float(ss_between / ss_total)


def _group_feature_analysis(
    feature_df: pd.DataFrame,
    *,
    layer: int,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = feature_df.loc[(feature_df["split"].eq("test")) & (feature_df["layer"].eq(layer))].copy()
    feature_cols = _topology_feature_columns(df)

    summary_rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        group_means = df.groupby("group4", sort=True)[feature].mean()
        top_group = str(group_means.idxmax())
        low_group = str(group_means.idxmin())
        summary_rows.append(
            {
                "feature": feature,
                "eta_sq": _eta_squared(df, feature, "group4"),
                "top_group": top_group,
                "low_group": low_group,
                **{f"mean__{group}": float(value) for group, value in group_means.items()},
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("eta_sq", ascending=False).reset_index(drop=True)
    top_features = summary_df.head(top_k)["feature"].tolist()

    heatmap_df = df.loc[:, ["group4"] + top_features].copy()
    # Normalize feature means so cross-feature comparisons are visible.
    z_df = heatmap_df.copy()
    for feature in top_features:
        values = z_df[feature].to_numpy(dtype=float)
        denom = float(values.std())
        if denom <= 0.0:
            z_df[feature] = 0.0
        else:
            z_df[feature] = (values - float(values.mean())) / denom
    heatmap = z_df.groupby("group4", sort=True)[top_features].mean()
    return summary_df, heatmap


def _render_feature_heatmap(heatmap_df: pd.DataFrame, *, model_label: str, output_path: Path) -> None:
    groups = heatmap_df.index.tolist()
    features = heatmap_df.columns.tolist()
    matrix = heatmap_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(features) + 2.0, 3.8))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_title(f"{model_label}: groupwise topology feature means (z-scored)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_result_plot(final_df: pd.DataFrame, *, output_path: Path) -> None:
    models = final_df["model_label"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for offset, method in [(-width / 2, "token_cloud_single"), (width / 2, "token_cloud_multilayer")]:
        subset = (
            final_df.loc[final_df["method"].eq(method)]
            .set_index("model_label")
            .loc[models]
            .reset_index()
        )
        axes[0].bar(x + offset, subset["macro_f1"], width=width, color=PLOT_COLORS[method], label=METHOD_LABELS[method])
        axes[1].bar(x + offset, subset["accuracy"], width=width, color=PLOT_COLORS[method], label=METHOD_LABELS[method])
    axes[0].set_title("Macro-F1")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("CLAMBER regrouped 4-way: topology classification", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mitigation_table() -> list[dict[str, str]]:
    return [
        {
            "label": "ambiguity",
            "failure_mode": "Multiple plausible interpretations of a term or referent.",
            "mitigation": "Ask a disambiguation question that enumerates the competing readings and defer answering until the user chooses one.",
            "system_action": "Generate 2-4 candidate senses/entities and request selection.",
        },
        {
            "label": "missing_condition",
            "failure_mode": "The query is understandable but lacks a needed personal, temporal, spatial, or task-specific slot.",
            "mitigation": "Run a targeted clarification policy that asks for the missing slot before answering.",
            "system_action": "Route to a slot-seeking prompt such as who/when/where/what based on a secondary detector or heuristic.",
        },
        {
            "label": "conflicting_condition",
            "failure_mode": "The prompt supports incompatible constraints or conflicting task interpretations.",
            "mitigation": "Surface the conflict explicitly and ask the user which constraint or interpretation should be kept.",
            "system_action": "Rewrite the detected conflict as two alternatives and require user resolution.",
        },
        {
            "label": "clear",
            "failure_mode": "No clarification is needed under this taxonomy.",
            "mitigation": "Answer directly, with ordinary uncertainty handling only.",
            "system_action": "Proceed to normal QA or generation.",
        },
    ]


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    plots_root = ensure_dir(output_root / "plots")

    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    report_lines = [
        "# CLAMBER Regrouped 4-Way Topology Study",
        "",
        "Regrouping used in this study:",
        "- `ambiguity` = `polysemy`, `co-reference`",
        "- `missing_condition` = `what`, `when`, `where`, `whom`",
        "- `conflicting_condition` = `ICL`",
        "- `clear` = `none`",
        "- `NK` is excluded",
        "",
        "This report uses token-cloud topological features only.",
        "",
    ]

    for model_index, spec in enumerate(MODEL_SPECS):
        feature_df = _load_feature_df(Path(spec["subclass_root"]))

        single_candidates, single_final = _evaluate_single_layers(
            feature_df,
            seed=args.seed + 100 * model_index,
            val_fraction=args.val_fraction,
        )
        single_candidates["model"] = spec["slug"]
        single_candidates["model_label"] = spec["label"]
        candidate_rows.extend(single_candidates.to_dict(orient="records"))

        multi_candidates, multi_final = _evaluate_multilayer(
            feature_df,
            seed=args.seed + 100 * model_index,
            val_fraction=args.val_fraction,
        )
        multi_candidates["model"] = spec["slug"]
        multi_candidates["model_label"] = spec["label"]
        candidate_rows.extend(multi_candidates.to_dict(orient="records"))

        for final in [single_final, multi_final]:
            final_rows.append(
                {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    **{key: value for key, value in final.items() if key != "feature_columns"},
                }
            )

        feature_summary, heatmap_df = _group_feature_analysis(
            feature_df,
            layer=int(single_final["layer"]),
            top_k=args.top_features,
        )
        feature_summary["model"] = spec["slug"]
        feature_summary["model_label"] = spec["label"]
        feature_summary["analysis_layer"] = int(single_final["layer"])
        feature_rows.append(feature_summary)
        heatmap_path = plots_root / f"{spec['slug']}__group_feature_heatmap.png"
        _render_feature_heatmap(heatmap_df, model_label=spec["label"], output_path=heatmap_path)

        report_lines.extend(
            [
                f"## {spec['label']}",
                "",
                "| Method | Macro-F1 | Acc | Feature Count | View |",
                "| --- | ---: | ---: | ---: | --- |",
                f"| Topology Single | {single_final['macro_f1']:.4f} | {single_final['accuracy']:.4f} | "
                f"{int(single_final['feature_count'])} | layer {int(single_final['layer'])} |",
                f"| Topology Multi | {multi_final['macro_f1']:.4f} | {multi_final['accuracy']:.4f} | "
                f"{int(multi_final['feature_count'])} | layers {multi_final['selection_signature']} |",
                "",
                "Top group-differentiating features at the best single layer:",
                "",
                "| Feature | Eta^2 | Highest group | Lowest group |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for _, row in feature_summary.head(args.top_features).iterrows():
            report_lines.append(
                f"| {row['feature']} | {float(row['eta_sq']):.4f} | {row['top_group']} | {row['low_group']} |"
            )
        report_lines.extend(
            [
                "",
                f"- Heatmap: `{heatmap_path}`",
                "",
            ]
        )

    final_df = pd.DataFrame(final_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    feature_df = pd.concat(feature_rows, ignore_index=True)
    result_plot_path = plots_root / "topology_4way_results.png"
    _render_result_plot(final_df, output_path=result_plot_path)

    mitigation_df = pd.DataFrame(_mitigation_table())
    report_lines.extend(
        [
            "## Mitigation Design",
            "",
            "| Predicted label | Failure mode | Mitigation | System action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in mitigation_df.iterrows():
        report_lines.append(
            f"| {row['label']} | {row['failure_mode']} | {row['mitigation']} | {row['system_action']} |"
        )
    report_lines.extend(
        [
            "",
            "## Main Readout",
            "",
            "- This is a direct 4-way study with `NK` removed from CLAMBER.",
            "- The classification task uses only token-cloud topological features.",
            "- Groupwise feature analysis uses the true regrouped labels on the held-out test split.",
            "- The mitigation policy is label-conditional: clarify differently for ambiguity, missing conditions, and conflicts, while answering clear queries directly.",
            "",
            f"- Result plot: `{result_plot_path}`",
            "",
        ]
    )

    write_parquet(candidate_df, output_root / "clamber_regrouped_4way_topology_candidates.parquet")
    write_parquet(final_df, output_root / "clamber_regrouped_4way_topology_final_metrics.parquet")
    write_parquet(feature_df, output_root / "clamber_regrouped_4way_topology_feature_differences.parquet")
    write_parquet(mitigation_df, output_root / "clamber_regrouped_4way_mitigation_table.parquet")
    write_markdown(output_root / "clamber_regrouped_4way_topology_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
