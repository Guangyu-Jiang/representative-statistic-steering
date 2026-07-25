"""Render class-averaged persistence-image comparisons for token-cloud features."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aen_replication.utils.io_utils import ensure_dir, write_markdown


LABEL_NAME = {1: "Ambiguous", 0: "Clear"}
TITLE_NAME = {"h0": "H0", "h1": "H1"}


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _pimg_columns(feature_df: pd.DataFrame, prefix: str) -> list[str]:
    pattern = re.compile(rf"^{prefix}_pimg_(\d+?)_(\d+?)$")
    matches: list[tuple[int, int, str]] = []
    for column in feature_df.columns:
        match = pattern.match(column)
        if match:
            matches.append((int(match.group(1)), int(match.group(2)), column))
    matches.sort()
    return [column for _, _, column in matches]


def _columns_to_image(mean_series: pd.Series, columns: list[str]) -> np.ndarray:
    coords: list[tuple[int, int]] = []
    for column in columns:
        _, _, row_str, col_str = column.split("_")
        coords.append((int(row_str), int(col_str)))
    side = max(max(row, col) for row, col in coords) + 1 if coords else 1
    image = np.zeros((side, side), dtype=float)
    for column, (row, col) in zip(columns, coords):
        image[row, col] = float(mean_series[column])
    return image


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#111111",
                fontsize=8,
            )


def _plot_panel_row(
    axes_row: list[plt.Axes],
    ambig_img: np.ndarray,
    clear_img: np.ndarray,
    diff_img: np.ndarray,
    *,
    prefix: str,
) -> None:
    vmax = float(max(np.max(ambig_img), np.max(clear_img), 1e-6))
    diff_abs = float(max(np.max(np.abs(diff_img)), 1e-6))
    panels = [
        (axes_row[0], ambig_img, "Reds", 0.0, vmax, f"{TITLE_NAME[prefix]} ambiguous mean"),
        (axes_row[1], clear_img, "Blues", 0.0, vmax, f"{TITLE_NAME[prefix]} clear mean"),
        (axes_row[2], diff_img, "RdBu_r", -diff_abs, diff_abs, f"{TITLE_NAME[prefix]} difference"),
    ]
    for ax, matrix, cmap, vmin, vmax_local, title in panels:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax_local, origin="lower", interpolation="nearest")
        _annotate_heatmap(ax, matrix)
        ax.set_title(title)
        ax.set_xlabel("Birth bin")
        ax.set_ylabel("Persistence bin")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _render_dataset_plot(
    feature_df: pd.DataFrame,
    *,
    dataset: str,
    layer: int,
    model_name: str,
    output_path: Path,
) -> None:
    subset = feature_df.loc[
        feature_df["dataset"].eq(dataset)
        & feature_df["split"].eq("test")
        & feature_df["feature_variant"].eq("single_layer")
        & feature_df["layer"].eq(layer)
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dataset={dataset}, layer={layer}.")
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    for row_index, prefix in enumerate(["h0", "h1"]):
        columns = _pimg_columns(subset, prefix)
        if not columns:
            raise ValueError(f"No persistence-image columns found for {prefix}.")
        ambig_mean = subset.loc[subset["label_ambiguous"].eq(1), columns].mean(axis=0)
        clear_mean = subset.loc[subset["label_ambiguous"].eq(0), columns].mean(axis=0)
        ambig_img = _columns_to_image(ambig_mean, columns)
        clear_img = _columns_to_image(clear_mean, columns)
        diff_img = ambig_img - clear_img
        _plot_panel_row(list(axes[row_index]), ambig_img, clear_img, diff_img, prefix=prefix)
    fig.suptitle(f"{model_name}: {dataset} class-averaged persistence images at layer {layer}", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else model_root / "plots_persistence_images"
    )
    output_dir = ensure_dir(output_dir)

    feature_df = pd.read_parquet(model_root / "token_cloud_topology_features.parquet")
    best_layers = _load_best_layers(model_root)
    datasets = list(args.datasets) if args.datasets else sorted(best_layers.keys())

    lines = [
        "# Token-Cloud Persistence Image Comparisons",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset in datasets:
        layer = best_layers[dataset] if str(args.layer).lower() == "auto" else int(args.layer)
        output_path = output_dir / f"{dataset}__layer_{layer:02d}__persistence_image_comparison.png"
        _render_dataset_plot(
            feature_df,
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
                f"- Plot: `{output_path}`",
                "",
            ]
        )

    write_markdown(output_dir / "token_cloud_persistence_image_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
