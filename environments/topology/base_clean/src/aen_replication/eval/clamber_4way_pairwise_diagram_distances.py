"""Pairwise persistence-diagram distances for regrouped 4-way CLAMBER.

This is a descriptive analysis, not a classifier. It avoids the pooled-token
prototype and instead compares individual question-level diagrams directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aen_replication.config import load_config
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import _compute_diagrams, _safe_wasserstein
from aen_replication.train.token_cloud_topology_classifier import (
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
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

GROUP_ORDER = ["ambiguity", "missing_condition", "conflicting_condition", "clear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_pairwise_diagram_distances",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sample-per-group", type=int, default=100)
    parser.add_argument("--pca-fit-scope", choices=["all", "train"], default="all")
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def _classifier_config(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    cfg = dict(config["token_cloud_topology_classifier"])
    cfg["_seed"] = int(seed)
    return cfg


def _load_regrouped_dataset(config: dict[str, Any]) -> pd.DataFrame:
    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    df = pd.read_parquet(dataset_path).copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df.reset_index(drop=True)


def _sample_questions(df: pd.DataFrame, *, sample_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for group in GROUP_ORDER:
        group_df = df.loc[df["group4"].eq(group)].copy()
        if group_df.empty:
            raise ValueError(f"No rows for group {group}")
        sample_n = min(int(sample_per_group), len(group_df))
        chosen = rng.choice(group_df.index.to_numpy(), size=sample_n, replace=False)
        parts.append(group_df.loc[np.sort(chosen)].copy())
    return pd.concat(parts, ignore_index=True)


def _compute_question_diagrams(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    classifier_config: dict[str, Any],
    dataset_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    layer: int,
    seed: int,
    pca_fit_scope: str,
) -> pd.DataFrame:
    prepared_all, text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle,
        text_column=str(classifier_config.get("text_column", "text")),
        use_chat_template=bool(classifier_config.get("use_chat_template", False)),
        system_prompt=classifier_config.get("system_prompt"),
    )
    prepared_all["_token_cloud_text"] = prepared_all[text_column]
    sampled_ids = set(sampled_df["example_id"].astype(str))
    prepared_sample = prepared_all.loc[prepared_all["example_id"].astype(str).isin(sampled_ids)].copy()
    prepared_sample = prepared_sample.merge(
        sampled_df.loc[:, ["example_id", "group4", "subclass"]].copy(),
        on="example_id",
        how="inner",
        suffixes=("", "_sampled"),
    )

    if pca_fit_scope == "train":
        pca_fit_df = prepared_all.loc[prepared_all["split"].eq("train")].copy().reset_index(drop=True)
    else:
        pca_fit_df = prepared_all.copy().reset_index(drop=True)
    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=pca_fit_df,
        text_column="_token_cloud_text",
        layers=[int(layer)],
        config=classifier_config,
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
    cloud_df = _extract_reduced_clouds(
        bundle=bundle,
        df=prepared_sample.reset_index(drop=True),
        text_column="_token_cloud_text",
        layers=[int(layer)],
        reducers=reducers,
        config=classifier_config,
    )
    cloud_df = cloud_df.merge(
        sampled_df.loc[:, ["example_id", "group4", "subclass"]].copy(),
        on="example_id",
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for row in cloud_df.to_dict(orient="records"):
        diagrams = _compute_diagrams(
            np.asarray(row["cloud"], dtype=float),
            maxdim=int(classifier_config.get("maxdim", 1)),
            coeff=int(classifier_config.get("coeff", 2)),
            distance_metric=str(classifier_config.get("distance_metric", "euclidean")),
        )
        rows.append(
            {
                "example_id": str(row["example_id"]),
                "group4": str(row["group4"]),
                "subclass": str(row["subclass"]),
                "split": str(row["split"]),
                "token_count": int(row["token_count"]),
                "h0_diagram": diagrams[0],
                "h1_diagram": diagrams[1] if len(diagrams) > 1 else np.zeros((0, 2), dtype=float),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_group_distances(diagram_df: pd.DataFrame, *, diagram_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records_by_group = {
        group: diagram_df.loc[diagram_df["group4"].eq(group), ["example_id", diagram_column]].to_dict(orient="records")
        for group in GROUP_ORDER
    }
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    homology = "H0" if diagram_column == "h0_diagram" else "H1"
    for left_group in GROUP_ORDER:
        left_records = records_by_group[left_group]
        for right_group in GROUP_ORDER:
            right_records = records_by_group[right_group]
            distances: list[float] = []
            if left_group == right_group:
                for i in range(len(left_records)):
                    for j in range(i + 1, len(right_records)):
                        distance = _safe_wasserstein(left_records[i][diagram_column], right_records[j][diagram_column])
                        distances.append(distance)
                        raw_rows.append(
                            {
                                "homology": homology,
                                "group_i": left_group,
                                "group_j": right_group,
                                "example_i": str(left_records[i]["example_id"]),
                                "example_j": str(right_records[j]["example_id"]),
                                "distance": float(distance),
                            }
                        )
            else:
                for left in left_records:
                    for right in right_records:
                        distance = _safe_wasserstein(left[diagram_column], right[diagram_column])
                        distances.append(distance)
                        raw_rows.append(
                            {
                                "homology": homology,
                                "group_i": left_group,
                                "group_j": right_group,
                                "example_i": str(left["example_id"]),
                                "example_j": str(right["example_id"]),
                                "distance": float(distance),
                            }
                        )
            values = np.asarray(distances, dtype=float)
            rows.append(
                {
                    "homology": homology,
                    "group_i": left_group,
                    "group_j": right_group,
                    "n_i": int(len(left_records)),
                    "n_j": int(len(right_records)),
                    "pair_count": int(len(values)),
                    "mean_distance": float(np.mean(values)) if len(values) else 0.0,
                    "median_distance": float(np.median(values)) if len(values) else 0.0,
                    "std_distance": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min_distance": float(np.min(values)) if len(values) else 0.0,
                    "max_distance": float(np.max(values)) if len(values) else 0.0,
                }
            )
    distance_df = pd.DataFrame(rows)
    within = {
        row["group_i"]: float(row["mean_distance"])
        for row in distance_df.loc[distance_df["group_i"].eq(distance_df["group_j"])].to_dict(orient="records")
    }
    sep_rows: list[dict[str, Any]] = []
    for row in distance_df.to_dict(orient="records"):
        group_i = str(row["group_i"])
        group_j = str(row["group_j"])
        if group_i == group_j:
            separation = 0.0
        else:
            separation = float(row["mean_distance"]) - 0.5 * (within[group_i] + within[group_j])
        sep_rows.append({**row, "normalized_separation": float(separation)})
    return distance_df, pd.DataFrame(sep_rows), pd.DataFrame(raw_rows)


def _matrix_from_rows(df: pd.DataFrame, value_column: str) -> np.ndarray:
    matrix = np.zeros((len(GROUP_ORDER), len(GROUP_ORDER)), dtype=float)
    for i, group_i in enumerate(GROUP_ORDER):
        for j, group_j in enumerate(GROUP_ORDER):
            row = df.loc[df["group_i"].eq(group_i) & df["group_j"].eq(group_j)].iloc[0]
            matrix[i, j] = float(row[value_column])
    return matrix


def _render_heatmaps(summary_df: pd.DataFrame, separation_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6))
    specs = [
        ("H0", summary_df, "mean_distance", "Mean H0 Wasserstein"),
        ("H1", summary_df, "mean_distance", "Mean H1 Wasserstein"),
        ("H0", separation_df, "normalized_separation", "H0 normalized separation"),
        ("H1", separation_df, "normalized_separation", "H1 normalized separation"),
    ]
    for ax, (homology, source_df, column, title) in zip(axes.flatten(), specs):
        matrix = _matrix_from_rows(source_df.loc[source_df["homology"].eq(homology)], column)
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=30, ha="right")
        ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("CLAMBER 4-way pairwise question-diagram distances", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pair_key(group_i: str, group_j: str) -> tuple[str, str]:
    order = {group: index for index, group in enumerate(GROUP_ORDER)}
    return tuple(sorted((group_i, group_j), key=lambda group: order[group]))


def _render_h0_distribution_figure(summary_df: pd.DataFrame, raw_df: pd.DataFrame, output_path: Path) -> None:
    h0_summary = summary_df.loc[summary_df["homology"].eq("H0")].copy()
    del raw_df

    matrix = _matrix_from_rows(h0_summary, "mean_distance")
    fig, heat_ax = plt.subplots(figsize=(7.2, 6.2))
    im = heat_ax.imshow(matrix, cmap="magma")
    heat_ax.set_title("Mean H0 pairwise distance")
    heat_ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=30, ha="right")
    heat_ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "black" if matrix[i, j] > matrix.max() * 0.72 else "white"
            heat_ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    config = load_config(args.config)
    classifier_config = _classifier_config(config, seed=args.seed)
    classifier_config["candidate_layers"] = [int(args.layer)]

    dataset_df = _load_regrouped_dataset(config)
    sampled_df = _sample_questions(dataset_df, sample_per_group=args.sample_per_group, seed=args.seed)

    bundle = load_hf_model(config["model"], classifier_config)
    diagram_df = _compute_question_diagrams(
        bundle=bundle,
        config=config,
        classifier_config=classifier_config,
        dataset_df=dataset_df,
        sampled_df=sampled_df,
        layer=int(args.layer),
        seed=int(args.seed),
        pca_fit_scope=str(args.pca_fit_scope),
    )

    h0_dist, h0_sep, h0_raw = _pairwise_group_distances(diagram_df, diagram_column="h0_diagram")
    h1_dist, h1_sep, h1_raw = _pairwise_group_distances(diagram_df, diagram_column="h1_diagram")
    distance_df = pd.concat([h0_dist, h1_dist], ignore_index=True)
    separation_df = pd.concat([h0_sep, h1_sep], ignore_index=True)
    raw_distance_df = pd.concat([h0_raw, h1_raw], ignore_index=True)

    write_parquet(diagram_df.drop(columns=["h0_diagram", "h1_diagram"]), output_root / "sampled_questions.parquet")
    write_parquet(distance_df, output_root / "pairwise_group_distances.parquet")
    write_parquet(separation_df, output_root / "pairwise_group_normalized_separations.parquet")
    write_parquet(raw_distance_df, output_root / "pairwise_distance_samples.parquet")
    heatmap_path = output_root / "pairwise_group_distance_heatmaps.png"
    _render_heatmaps(distance_df, separation_df, heatmap_path)
    h0_distribution_path = output_root / "h0_pairwise_distance_distributions.png"
    _render_h0_distribution_figure(distance_df, raw_distance_df, h0_distribution_path)

    counts = diagram_df.groupby("group4").size().reindex(GROUP_ORDER)
    lines = [
        "# CLAMBER 4-Way Pairwise Persistence-Diagram Distances",
        "",
        f"- Config: `{args.config}`",
        f"- Layer: `{int(args.layer)}`",
        f"- Sample per group: `{int(args.sample_per_group)}`",
        f"- Distance: Wasserstein between individual question persistence diagrams",
        f"- PCA: refit on `{args.pca_fit_scope}` regrouped CLAMBER rows using the token-cloud config",
        f"- Heatmap: `{heatmap_path}`",
        f"- H0 distribution figure: `{h0_distribution_path}`",
        f"- Raw distance samples: `{output_root / 'pairwise_distance_samples.parquet'}`",
        "",
        "![H0 pairwise distance distributions](h0_pairwise_distance_distributions.png)",
        "",
        "## Sample Counts",
        "",
        "| Group | n |",
        "| --- | ---: |",
    ]
    for group, count in counts.items():
        lines.append(f"| {group} | {int(count)} |")
    lines.extend(["", "## Mean Pairwise Distances", ""])
    for homology in ["H0", "H1"]:
        matrix = _matrix_from_rows(distance_df.loc[distance_df["homology"].eq(homology)], "mean_distance")
        lines.extend([f"### {homology}", "", "| group | " + " | ".join(GROUP_ORDER) + " |", "| --- | " + " | ".join(["---:"] * len(GROUP_ORDER)) + " |"])
        for group, row in zip(GROUP_ORDER, matrix):
            lines.append("| " + group + " | " + " | ".join(f"{value:.4f}" for value in row) + " |")
        lines.append("")
    lines.extend(["## Normalized Separation", "", "Cross-group mean minus the average of the two within-group means.", ""])
    for homology in ["H0", "H1"]:
        matrix = _matrix_from_rows(separation_df.loc[separation_df["homology"].eq(homology)], "normalized_separation")
        lines.extend([f"### {homology}", "", "| group | " + " | ".join(GROUP_ORDER) + " |", "| --- | " + " | ".join(["---:"] * len(GROUP_ORDER)) + " |"])
        for group, row in zip(GROUP_ORDER, matrix):
            lines.append("| " + group + " | " + " | ".join(f"{value:.4f}" for value in row) + " |")
        lines.append("")

    write_markdown(output_root / "pairwise_group_distance_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
