"""Run a targeted sweep for the independent topology classifier."""

from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

import pandas as pd

from aen_replication.config import load_config
from aen_replication.train.independent_topology_classifier import run_independent_topology_classifier_analysis
from aen_replication.utils.io_utils import append_command_history, ensure_dir, slugify, write_markdown, write_parquet
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Run config(s) to sweep.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/independent_topology_sweep",
        help="Root directory for sweep artifacts.",
    )
    parser.add_argument(
        "--parallel-jobs-per-run",
        type=int,
        default=8,
        help="Internal joblib workers used by each classifier run.",
    )
    return parser.parse_args()


def _variant_specs(parallel_jobs_per_run: int) -> list[dict[str, object]]:
    return [
        {
            "name": "logistic_euclidean_base",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "euclidean",
            "neighborhood_k": 24,
            "pca_components": 8,
            "geometry_components": 8,
            "topology_components": 6,
            "multilayer_top_k": 3,
            "persistence_image_grid_side": 4,
            "classifier": {"family": "logistic", "standardize": True},
        },
        {
            "name": "logistic_cosine_rich",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "cosine",
            "neighborhood_k": 24,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {"family": "logistic", "standardize": True},
        },
        {
            "name": "logistic_cosine_rich_k48",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "cosine",
            "neighborhood_k": 48,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {"family": "logistic", "standardize": True},
        },
        {
            "name": "extra_trees_euclidean_rich",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "euclidean",
            "neighborhood_k": 24,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {
                "family": "extra_trees",
                "n_estimators": 500,
                "min_samples_leaf": 2,
                "n_jobs": 1,
                "standardize": False,
            },
        },
        {
            "name": "extra_trees_cosine_rich",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "cosine",
            "neighborhood_k": 24,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {
                "family": "extra_trees",
                "n_estimators": 500,
                "min_samples_leaf": 2,
                "n_jobs": 1,
                "standardize": False,
            },
        },
        {
            "name": "extra_trees_cosine_rich_k48",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "cosine",
            "neighborhood_k": 48,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {
                "family": "extra_trees",
                "n_estimators": 500,
                "min_samples_leaf": 2,
                "n_jobs": 1,
                "standardize": False,
            },
        },
        {
            "name": "hgb_euclidean_rich",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "euclidean",
            "neighborhood_k": 24,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {
                "family": "hist_gradient_boosting",
                "learning_rate": 0.05,
                "max_iter": 400,
                "min_samples_leaf": 20,
                "standardize": False,
            },
        },
        {
            "name": "hgb_cosine_rich_k48",
            "parallel_jobs": parallel_jobs_per_run,
            "topology_metric": "cosine",
            "neighborhood_k": 48,
            "pca_components": 16,
            "geometry_components": 16,
            "topology_components": 12,
            "multilayer_top_k": 5,
            "persistence_image_grid_side": 6,
            "classifier": {
                "family": "hist_gradient_boosting",
                "learning_rate": 0.05,
                "max_iter": 400,
                "min_samples_leaf": 20,
                "standardize": False,
            },
        },
    ]


def _render_summary(model_name: str, rows: pd.DataFrame, output_path: Path) -> None:
    if rows.empty:
        write_markdown(output_path, f"# Independent Topology Sweep\n\nNo results for `{model_name}`.\n")
        return
    lines = [
        "# Independent Topology Sweep",
        "",
        f"- Model: `{model_name}`",
        f"- Variants: `{rows['variant'].nunique()}`",
        "",
        "## Best Results By Dataset/Feature Set",
        "",
    ]
    best = (
        rows.sort_values(["dataset", "feature_set", "test_auroc"], ascending=[True, True, False])
        .groupby(["dataset", "feature_set"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    for dataset in sorted(best["dataset"].unique()):
        lines.append(f"### {dataset}")
        lines.append("")
        subset = best.loc[best["dataset"].eq(dataset)].sort_values("feature_set")
        for row in subset.to_dict(orient="records"):
            lines.append(
                f"- `{row['feature_set']}`: AUROC `{row['test_auroc']:.4f}`, "
                f"accuracy `{row['test_accuracy']:.4f}`, variant `{row['variant']}`"
            )
        lines.append("")
    write_markdown(output_path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    first_config = load_config(args.config[0])
    setup_logging(
        first_config["runtime"]["log_level"],
        Path(first_config["runtime"]["log_dir"]) / "run_independent_topology_sweep.log",
    )
    set_global_seed(int(first_config["seed"]))
    append_command_history(first_config["runtime"]["command_history_path"], sys.argv)

    sweep_root = ensure_dir(args.output_dir)
    variants = _variant_specs(parallel_jobs_per_run=max(1, int(args.parallel_jobs_per_run)))

    all_rows: list[pd.DataFrame] = []
    for config_path in args.config:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        model_slug = slugify(model_name)
        hidden_state_root = Path(config["extraction"]["cache_dir"]) / model_slug
        if not hidden_state_root.exists():
            raise FileNotFoundError(f"Hidden-state cache directory not found: {hidden_state_root}")
        model_root = ensure_dir(Path(sweep_root) / model_slug)
        model_rows: list[pd.DataFrame] = []
        for variant in variants:
            classifier_config = deepcopy(config["independent_topology_classifier"])
            classifier_config.update(
                {
                    "output_dir": str(model_root / str(variant["name"])),
                    "parallel_jobs": int(variant["parallel_jobs"]),
                    "topology_metric": str(variant["topology_metric"]),
                    "neighborhood_k": int(variant["neighborhood_k"]),
                    "pca_components": int(variant["pca_components"]),
                    "geometry_components": int(variant["geometry_components"]),
                    "topology_components": int(variant["topology_components"]),
                    "multilayer_top_k": int(variant["multilayer_top_k"]),
                    "persistence_image_grid_side": int(variant["persistence_image_grid_side"]),
                }
            )
            classifier_section = deepcopy(classifier_config["classifier"])
            classifier_section.update(dict(variant["classifier"]))
            classifier_config["classifier"] = classifier_section
            outputs = run_independent_topology_classifier_analysis(
                model_name=model_name,
                hidden_state_root=hidden_state_root,
                classifier_config=classifier_config,
                seed=int(config["seed"]),
            )
            final_df = pd.read_parquet(outputs["final_metrics_path"]).copy()
            final_df["variant"] = str(variant["name"])
            final_df["model_name"] = model_name
            final_df["model_slug"] = model_slug
            model_rows.append(final_df)
        model_results = pd.concat(model_rows, ignore_index=True)
        write_parquet(model_results, model_root / "independent_topology_sweep_results.parquet")
        _render_summary(model_name, model_results, model_root / "independent_topology_sweep_summary.md")
        all_rows.append(model_results)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        write_parquet(combined, Path(sweep_root) / "independent_topology_sweep_results.parquet")


if __name__ == "__main__":
    main()
