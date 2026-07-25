from __future__ import annotations

from pathlib import Path

import pandas as pd

from aen_replication.features.ph_descriptors import run_ph_descriptor_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_ph_descriptor_analysis_writes_curves_and_report(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    summary_rows = []
    diagram_rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in [0, 1]:
            for subspace_name in ["top_20", "top_50"]:
                for label_group, delta in [("ambiguous", 0.2), ("clear", 0.0)]:
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "dataset": dataset,
                            "layer": layer,
                            "readout": "mean_pool",
                            "subspace_name": subspace_name,
                            "label_group": label_group,
                            "h0_total_persistence_norm": 5.0 + layer + delta,
                            "h0_feature_count": 8.0,
                            "h0_max_persistence_norm": 1.5,
                            "h1_total_persistence_norm": 2.0 + 0.5 * layer + delta,
                            "h1_feature_count": 3.0,
                            "h1_max_persistence_norm": 0.8,
                        }
                    )
                    for homology_dim in [0, 1]:
                        for birth, death in [(0.1, 0.5 + delta), (0.2, 0.7 + delta)]:
                            diagram_rows.append(
                                {
                                    "model_name": model_name,
                                    "dataset": dataset,
                                    "layer": layer,
                                    "readout": "mean_pool",
                                    "subspace_name": subspace_name,
                                    "label_group": label_group,
                                    "point_count": 8,
                                    "selected_count": 20,
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

    outputs = run_ph_descriptor_analysis(
        model_name=model_name,
        summary_path=summary_path,
        diagrams_path=diagrams_path,
        descriptor_config={
            "output_dir": str(tmp_path / "descriptors"),
            "datasets": ["ambigqa", "situatedqa"],
            "subspaces": ["top_20", "top_50"],
            "grid_size": 16,
            "summary_filename": "descriptor_summary.parquet",
            "curves_filename": "betti_curves.parquet",
            "comparison_filename": "comparison.parquet",
            "aggregate_filename": "aggregate.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
    )

    descriptor_df = pd.read_parquet(outputs["summary_path"])
    curves_df = pd.read_parquet(outputs["curves_path"])
    comparison_df = pd.read_parquet(outputs["comparison_path"])
    assert not descriptor_df.empty
    assert {"persistence_entropy", "betti_curve_auc_norm", "mean_lifetime"} <= set(descriptor_df.columns)
    assert not curves_df.empty
    assert {"filtration_value_norm", "betti_number"} <= set(curves_df.columns)
    assert not comparison_df.empty
    assert {"delta_persistence_entropy", "delta_betti_curve_auc_norm"} <= set(comparison_df.columns)
    plots_dir = Path(outputs["report_path"]).parent / "plots"
    assert (plots_dir / "ambigqa__h0__persistence_entropy_by_layer.png").exists()
    assert (plots_dir / "ambigqa__descriptor_deltas.png").exists()
    assert (plots_dir / "ambigqa__representative_betti_curves.png").exists()
