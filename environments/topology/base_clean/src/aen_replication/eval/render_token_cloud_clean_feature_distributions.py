"""Render cleaner class-level token-cloud feature distribution plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aen_replication.utils.io_utils import ensure_dir, write_markdown


LABEL_NAME_MAP = {0: "Clear", 1: "Ambiguous"}
LABEL_COLOR_MAP = {0: "#4c72b0", 1: "#c44e52"}

FEATURE_SPECS = [
    ("h0_mean_persistence", "H0 Mean Persistence"),
    ("h0_total_persistence_norm", "H0 Total Persistence"),
    ("h0_betti_curve_auc_norm", "H0 Betti AUC"),
    ("h0_wasserstein_margin", "H0 Wasserstein Margin"),
]


def _load_best_layers(model_root: Path) -> dict[str, int]:
    final_df = pd.read_parquet(model_root / "token_cloud_topology_final_metrics.parquet")
    layers: dict[str, int] = {}
    for dataset in sorted(final_df["dataset"].unique()):
        subset = final_df.loc[
            final_df["dataset"].eq(dataset) & final_df["feature_set"].eq("topology_only")
        ].copy()
        if subset.empty:
            continue
        layers[dataset] = int(subset.iloc[0]["layer"])
    return layers


def _prepare_subset(feature_df: pd.DataFrame, *, dataset: str, layer: int) -> pd.DataFrame:
    subset = feature_df.loc[
        feature_df["dataset"].eq(dataset)
        & feature_df["split"].eq("test")
        & feature_df["feature_variant"].eq("single_layer")
        & feature_df["layer"].eq(layer)
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dataset={dataset}, layer={layer}.")
    subset["h0_wasserstein_margin"] = (
        subset["h0_wasserstein_to_clear"].astype(float) - subset["h0_wasserstein_to_ambiguous"].astype(float)
    )
    return subset


def _render_dataset_plot(
    subset: pd.DataFrame,
    *,
    dataset: str,
    layer: int,
    model_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    axes_list = list(axes.flatten())
    for axis, (feature_name, title) in zip(axes_list, FEATURE_SPECS):
        clear = subset.loc[subset["label_ambiguous"].eq(0), feature_name].to_numpy(dtype=float)
        ambiguous = subset.loc[subset["label_ambiguous"].eq(1), feature_name].to_numpy(dtype=float)
        box = axis.boxplot(
            [clear, ambiguous],
            labels=[LABEL_NAME_MAP[0], LABEL_NAME_MAP[1]],
            patch_artist=True,
            widths=0.56,
            showfliers=True,
        )
        for patch, label_value in zip(box["boxes"], [0, 1]):
            patch.set_facecolor(LABEL_COLOR_MAP[label_value])
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.8)
        axis.set_title(title)
        axis.set_ylabel("Value")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        f"{model_name}: {dataset} class-level token-cloud feature distributions (layer {layer})",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_root = Path(args.model_root)
    model_name = str(args.model_name)
    output_dir = Path(args.output_dir) if args.output_dir is not None else model_root / "plots_clean_feature_distributions"
    output_dir = ensure_dir(output_dir)

    feature_df = pd.read_parquet(model_root / "token_cloud_topology_features.parquet")
    best_layers = _load_best_layers(model_root)
    datasets = list(args.datasets) if args.datasets else sorted(best_layers.keys())

    lines = [
        "# Token-Cloud Clean Feature Distributions",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset in datasets:
        layer = best_layers[dataset] if str(args.layer).lower() == "auto" else int(args.layer)
        subset = _prepare_subset(feature_df, dataset=dataset, layer=layer)
        output_path = output_dir / f"{dataset}__layer_{layer:02d}__clean_feature_distributions.png"
        _render_dataset_plot(
            subset,
            dataset=dataset,
            layer=layer,
            model_name=model_name,
            output_path=output_path,
        )
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Layer: `{layer}`",
                f"- Questions per class: clear `{int(subset['label_ambiguous'].eq(0).sum())}`, ambiguous `{int(subset['label_ambiguous'].eq(1).sum())}`",
                f"- Plot: `{output_path}`",
                "",
            ]
        )

    write_markdown(output_dir / "token_cloud_clean_feature_distributions_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
