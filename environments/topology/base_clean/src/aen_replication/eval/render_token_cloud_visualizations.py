"""Render token-cloud topology visualizations from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aen_replication.train.token_cloud_topology_classifier import _render_token_cloud_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--feature-table-path", required=True)
    parser.add_argument("--final-metrics-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plots-dirname", default="plots")
    parser.add_argument("--report-filename", default="token_cloud_topology_visualizations.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_df = pd.read_parquet(Path(args.feature_table_path))
    final_df = pd.read_parquet(Path(args.final_metrics_path))
    _render_token_cloud_visualizations(
        model_name=str(args.model_name),
        feature_table=feature_df,
        final_df=final_df,
        output_root=Path(args.output_root),
        classifier_config={
            "plots_dirname": str(args.plots_dirname),
            "visualization_report_filename": str(args.report_filename),
        },
    )


if __name__ == "__main__":
    main()
