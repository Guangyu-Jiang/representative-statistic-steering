"""Render focused H0 groupwise plots for the regrouped 4-way CLAMBER study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aen_replication.utils.io_utils import ensure_dir, write_markdown

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

GROUP_ORDER = ["ambiguity", "missing_condition", "conflicting_condition", "clear"]
GROUP_COLORS = {
    "ambiguity": "#1f77b4",
    "missing_condition": "#d62728",
    "conflicting_condition": "#2ca02c",
    "clear": "#7f7f7f",
}

FEATURE_SPECS = [
    ("h0_mean_persistence", "H0 Mean Persistence"),
    ("h0_total_persistence_norm", "H0 Total Persistence"),
    ("h0_persistence_entropy", "H0 Persistence Entropy"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology",
    )
    return parser.parse_args()


def _load_final_metrics(report_root: Path) -> pd.DataFrame:
    return pd.read_parquet(report_root / "clamber_regrouped_4way_topology_final_metrics.parquet")


def _load_model_feature_df(model_slug: str) -> pd.DataFrame:
    base = Path("/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification")
    df = pd.read_parquet(base / model_slug / "clamber_token_cloud_all_layer_features.parquet").copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df.reset_index(drop=True)


def _render_h0_distributions(
    df: pd.DataFrame,
    *,
    model_label: str,
    layer: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(FEATURE_SPECS), figsize=(18.0, 4.8))
    for ax, (feature, title) in zip(axes, FEATURE_SPECS):
        groups = [df.loc[df["group4"].eq(group), feature].to_numpy(dtype=float) for group in GROUP_ORDER]
        parts = ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
        for body, group in zip(parts["bodies"], GROUP_ORDER):
            body.set_facecolor(GROUP_COLORS[group])
            body.set_edgecolor("none")
            body.set_alpha(0.30)
        box = ax.boxplot(groups, widths=0.18, patch_artist=True, showfliers=False)
        for patch, group in zip(box["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        for median in box["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.6)
        ax.set_title(title)
        ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels(GROUP_ORDER, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.20)
    fig.suptitle(f"{model_label}: regrouped 4-way H0 feature distributions at layer {layer}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_root = Path(args.report_root)
    plots_root = ensure_dir(report_root / "true_label_plots")
    final_df = _load_final_metrics(report_root)

    report_lines = [
        "# CLAMBER Regrouped 4-Way H0 Feature Plots",
        "",
        "These figures show true-label groupwise distributions for three H0 features:",
        "- `h0_mean_persistence`",
        "- `h0_total_persistence_norm`",
        "- `h0_persistence_entropy`",
        "",
        "For a finite H0 persistence diagram with bars `(birth_i, death_i)`, the lifetime of bar `i` is `death_i - birth_i`.",
        "`h0_mean_persistence` is the arithmetic mean of those lifetimes: `(1 / N) * sum_i(death_i - birth_i)`.",
        "For H0 Vietoris-Rips diagrams over token-cloud distances, births are usually near zero, so this is approximately the average merge distance of connected components.",
        "",
        "`h0_betti_curve_auc_norm` is omitted because it is effectively redundant with `h0_total_persistence_norm` for H0: integrating the H0 Betti curve counts each component for its lifetime, so the area is the sum of H0 lifetimes, up to grid approximation.",
        "",
    ]

    for model_slug in final_df["model"].drop_duplicates().tolist():
        model_final = final_df.loc[(final_df["model"].eq(model_slug)) & (final_df["method"].eq("token_cloud_single"))].iloc[0]
        model_label = str(model_final["model_label"])
        layer = int(model_final["layer"])
        feature_df = _load_model_feature_df(model_slug)
        layer_df = feature_df.loc[(feature_df["split"].eq("test")) & (feature_df["layer"].eq(layer))].copy()
        output_path = plots_root / f"{model_slug}__h0_group_distributions.png"
        _render_h0_distributions(layer_df, model_label=model_label, layer=layer, output_path=output_path)
        report_lines.extend(
            [
                f"## {model_label}",
                "",
                f"- Layer: `{layer}`",
                f"- Plot: `{output_path}`",
                "",
            ]
        )

    write_markdown(plots_root / "clamber_regrouped_4way_h0_feature_plots.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
