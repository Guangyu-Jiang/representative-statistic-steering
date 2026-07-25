from __future__ import annotations

from pathlib import Path

import pandas as pd

from aen_replication.features.ph_visualization import run_ph_visualization_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_ph_visualization_analysis_writes_plots_and_aggregate(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    summary_rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in range(4):
            for subspace_name in ["aed_final", "top_20", "top_50"]:
                for label_group, offset in [("ambiguous", 0.6), ("clear", 0.1)]:
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "dataset": dataset,
                            "layer": layer,
                            "readout": "mean_pool",
                            "subspace_name": subspace_name,
                            "label_group": label_group,
                            "point_count": 16,
                            "selected_count": 5 if subspace_name == "aed_final" else 20,
                            "input_dim": 5 if subspace_name == "aed_final" else 20,
                            "h0_total_persistence_norm": 12.0 + layer + offset,
                            "h1_total_persistence_norm": 3.0 + 0.3 * layer + offset,
                            "h0_feature_count": 15.0 + offset,
                            "h1_feature_count": 6.0 + offset,
                        }
                    )

    distance_rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in range(1, 4):
            for subspace_name in ["aed_final", "top_20", "top_50"]:
                for homology_dim in [0, 1]:
                    distance_rows.append(
                        {
                            "model_name": model_name,
                            "dataset": dataset,
                            "layer": layer,
                            "subspace_name": subspace_name,
                            "homology_dim": homology_dim,
                            "bottleneck_distance": 0.2 + 0.05 * homology_dim,
                            "wasserstein_distance": 2.0 + layer + homology_dim,
                            "ambiguous_total_persistence_norm": 4.0,
                            "clear_total_persistence_norm": 3.0,
                            "delta_total_persistence_norm": 1.0,
                            "ambiguous_feature_count": 5.0,
                            "clear_feature_count": 4.0,
                            "ambiguous_selected_count": 5,
                            "clear_selected_count": 5,
                        }
                    )

    summary_path = tmp_path / "summary.parquet"
    focused_distance_path = tmp_path / "focused_distances.parquet"
    write_parquet(pd.DataFrame(summary_rows), summary_path)
    write_parquet(pd.DataFrame(distance_rows), focused_distance_path)

    outputs = run_ph_visualization_analysis(
        model_name=model_name,
        summary_path=summary_path,
        output_dir=tmp_path / "visuals",
        visualization_config={
            "datasets": ["ambigqa", "situatedqa"],
            "subspaces": ["aed_final", "top_20", "top_50"],
            "aggregate_filename": "aggregate.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
        focused_distance_path=focused_distance_path,
    )

    aggregate_df = pd.read_parquet(outputs["aggregate_path"])
    assert not aggregate_df.empty
    assert {
        "delta_h0_total_persistence_norm_mean",
        "delta_h1_total_persistence_norm_mean",
        "max_abs_h0_delta_layer",
        "h1_wasserstein_mean",
    } <= set(aggregate_df.columns)
    plots_dir = Path(outputs["report_path"]).parent / "plots"
    assert (plots_dir / "ambigqa__h0_total_persistence_norm_by_layer.png").exists()
    assert (plots_dir / "ambigqa__h1_feature_count_by_layer.png").exists()
    assert (plots_dir / "ambigqa__delta_total_persistence_heatmaps.png").exists()
    assert (plots_dir / "ambigqa__focused_wasserstein_heatmaps.png").exists()
