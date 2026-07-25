"""Render true-label visualization figures for the regrouped 4-way CLAMBER study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from aen_replication.train.token_cloud_topology_classifier import _topology_feature_columns
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


def _display_feature_name(feature: str) -> str:
    return (
        str(feature)
        .replace("ambiguous", "ill-posed")
        .replace("_", " ")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology",
    )
    parser.add_argument("--top-features", type=int, default=4)
    return parser.parse_args()


def _load_final_metrics(report_root: Path) -> pd.DataFrame:
    return pd.read_parquet(report_root / "clamber_regrouped_4way_topology_final_metrics.parquet")


def _load_feature_differences(report_root: Path) -> pd.DataFrame:
    return pd.read_parquet(report_root / "clamber_regrouped_4way_topology_feature_differences.parquet")


def _load_model_feature_df(model_slug: str) -> pd.DataFrame:
    base = Path("/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification")
    df = pd.read_parquet(base / model_slug / "clamber_token_cloud_all_layer_features.parquet").copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df.reset_index(drop=True)


def _render_projection_plots(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_label: str,
    layer: int,
    output_path: Path,
) -> None:
    x = df.loc[:, feature_cols].to_numpy(dtype=float)
    y = df["group4"].astype(str).to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=0)
    pca_xy = pca.fit_transform(x_scaled)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_xy = lda.fit_transform(x_scaled, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, coords, title in [
        (axes[0], pca_xy, "PCA of topology features"),
        (axes[1], lda_xy, "LDA of topology features"),
    ]:
        for group in GROUP_ORDER:
            mask = y == group
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.7,
                label=group,
                color=GROUP_COLORS[group],
                edgecolors="none",
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("LD1")
    axes[1].set_ylabel("LD2")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(f"{model_label}: regrouped 4-way true-label structure at layer {layer}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_feature_distributions(
    df: pd.DataFrame,
    *,
    top_features: list[str],
    model_label: str,
    layer: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes_list = list(axes.flatten())
    for ax, feature in zip(axes_list, top_features):
        groups = [df.loc[df["group4"].eq(group), feature].to_numpy(dtype=float) for group in GROUP_ORDER]
        parts = ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
        for body, group in zip(parts["bodies"], GROUP_ORDER):
            body.set_facecolor(GROUP_COLORS[group])
            body.set_edgecolor("none")
            body.set_alpha(0.35)
        box = ax.boxplot(groups, widths=0.18, patch_artist=True, showfliers=False)
        for patch, group in zip(box["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        for median in box["medians"]:
            median.set_color("#111111")
        ax.set_title(_display_feature_name(feature))
        ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels(GROUP_ORDER, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.2)
    for ax in axes_list[len(top_features) :]:
        ax.axis("off")
    fig.suptitle(f"{model_label}: top true-label feature distributions at layer {layer}", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_root = Path(args.report_root)
    plots_root = ensure_dir(report_root / "true_label_plots")
    final_df = _load_final_metrics(report_root)
    diff_df = _load_feature_differences(report_root)

    report_lines = [
        "# CLAMBER Regrouped 4-Way True-Label Figures",
        "",
        "These figures visualize group differences using the true regrouped labels.",
        "",
    ]

    for model_slug in final_df["model"].drop_duplicates().tolist():
        model_final = final_df.loc[(final_df["model"].eq(model_slug)) & (final_df["method"].eq("token_cloud_single"))].iloc[0]
        model_label = str(model_final["model_label"])
        layer = int(model_final["layer"])
        feature_df = _load_model_feature_df(model_slug)
        layer_df = feature_df.loc[(feature_df["split"].eq("test")) & (feature_df["layer"].eq(layer))].copy()
        feature_cols = _topology_feature_columns(layer_df)
        top_features = (
            diff_df.loc[diff_df["model"].eq(model_slug)]
            .sort_values("eta_sq", ascending=False)
            .head(args.top_features)["feature"]
            .tolist()
        )

        projection_path = plots_root / f"{model_slug}__projection.png"
        distribution_path = plots_root / f"{model_slug}__feature_distributions.png"
        _render_projection_plots(layer_df, feature_cols=feature_cols, model_label=model_label, layer=layer, output_path=projection_path)
        _render_feature_distributions(
            layer_df,
            top_features=top_features,
            model_label=model_label,
            layer=layer,
            output_path=distribution_path,
        )

        report_lines.extend(
            [
                f"## {model_label}",
                "",
                f"- Projection plot: `{projection_path}`",
                f"- Feature distributions: `{distribution_path}`",
                "",
            ]
        )

    write_markdown(report_root / "true_label_plots" / "clamber_regrouped_4way_true_label_plots.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
