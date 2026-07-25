from __future__ import annotations

from pathlib import Path

import pandas as pd

from aen_replication.features.focused_persistent_homology import run_focused_persistent_homology_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_focused_persistent_homology_analysis_writes_reports_and_plots(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    summary_rows = []
    diagram_rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in range(5):
            for subspace_name, base_h1 in [("aed_final", 2.0), ("top_20", 4.0), ("top_50", 3.0), ("top_100", 2.5)]:
                for label_group, delta in [("ambiguous", 0.4), ("clear", 0.0)]:
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "dataset": dataset,
                            "layer": layer,
                            "readout": "mean_pool",
                            "subspace_name": subspace_name,
                            "label_group": label_group,
                            "point_count": 8,
                            "selected_count": 20 if subspace_name == "top_20" else 5,
                            "input_dim": 20 if subspace_name == "top_20" else 5,
                            "h0_total_persistence_norm": 10.0 + layer + delta,
                            "h1_total_persistence_norm": base_h1 + 0.2 * layer + delta,
                            "h0_feature_count": 7.0,
                            "h1_feature_count": 3.0,
                        }
                    )
                    for homology_dim in [0, 1]:
                        for birth, death in [(0.1 + 0.05 * layer, 0.6 + 0.05 * layer + delta), (0.2, 0.5 + delta)]:
                            diagram_rows.append(
                                {
                                    "model_name": model_name,
                                    "dataset": dataset,
                                    "layer": layer,
                                    "readout": "mean_pool",
                                    "subspace_name": subspace_name,
                                    "label_group": label_group,
                                    "point_count": 8,
                                    "selected_count": 20 if subspace_name == "top_20" else 5,
                                    "homology_dim": homology_dim,
                                    "birth": birth,
                                    "death": death,
                                    "persistence": death - birth,
                                }
                            )

    summary_path = tmp_path / "summary.parquet"
    diagrams_path = tmp_path / "diagrams.parquet"
    write_parquet(pd.DataFrame(summary_rows), summary_path)
    write_parquet(pd.DataFrame(diagram_rows), diagrams_path)

    outputs = run_focused_persistent_homology_analysis(
        model_name=model_name,
        summary_path=summary_path,
        diagrams_path=diagrams_path,
        focus_config={
            "output_dir": str(tmp_path / "focused"),
            "focus_subspaces": ["top_20", "top_50", "top_100"],
            "compare_subspaces": ["aed_final", "top_20", "top_50", "top_100"],
            "plot_subspaces": ["top_20", "top_50"],
            "window_size": 3,
            "barcode_top_n": 10,
            "distance_summary_filename": "distances.parquet",
            "aggregate_filename": "aggregate.parquet",
            "window_filename": "window.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
    )

    distance_df = pd.read_parquet(outputs["distance_summary_path"])
    aggregate_df = pd.read_parquet(outputs["aggregate_path"])
    window_df = pd.read_parquet(outputs["window_path"])

    assert not distance_df.empty
    assert {"wasserstein_distance", "bottleneck_distance", "homology_dim", "delta_total_persistence_norm"} <= set(distance_df.columns)
    assert set(distance_df["subspace_name"]) == {"aed_final", "top_20", "top_50", "top_100"}
    assert set(distance_df["homology_dim"]) == {0, 1}
    assert not aggregate_df.empty
    assert {"h1_wasserstein_mean", "h1_wasserstein_max", "h1_bottleneck_mean"} <= set(aggregate_df.columns)
    assert int(window_df.iloc[0]["window_end"] - window_df.iloc[0]["window_start"] + 1) == 3
    plots_dir = Path(outputs["report_path"]).parent / "plots"
    assert (plots_dir / "h1_wasserstein_by_layer.png").exists()
    assert (plots_dir / "h0_wasserstein_by_layer.png").exists()
