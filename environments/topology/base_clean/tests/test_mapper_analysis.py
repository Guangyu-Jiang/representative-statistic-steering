from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.features.mapper_analysis import run_mapper_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_mapper_analysis_writes_graph_stats_and_plots(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    rng = np.random.default_rng(13)
    rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in [0, 1]:
            for subspace_name in ["top_20", "top_50"]:
                for label_ambiguous, center in [(1, -1.0), (0, 1.0)]:
                    for idx in range(40):
                        rows.append(
                            {
                                "example_id": f"{dataset}_{layer}_{subspace_name}_{label_ambiguous}_{idx}",
                                "pair_id": f"{dataset}_{idx}",
                                "dataset": dataset,
                                "split": "test",
                                "text": f"question {idx}",
                                "label_ambiguous": label_ambiguous,
                                "source_id": f"{dataset}_{idx}",
                                "context_type": "synthetic",
                                "model_name": model_name,
                                "layer": layer,
                                "readout": "mean_pool",
                                "decision_value": float(center + 0.2 * rng.normal()),
                                "signed_distance": float(center + 0.3 * rng.normal()),
                                "aen_k": 5,
                                "aen_count": 5,
                                "subspace_name": subspace_name,
                                "selected_count": 20,
                                "effective_reduction_dim": 2,
                                "z_0": float(center + 0.4 * rng.normal()),
                                "z_1": float(0.5 * layer + 0.4 * rng.normal()),
                            }
                        )
    features_path = tmp_path / "layerwise_features.parquet"
    write_parquet(pd.DataFrame(rows), features_path)

    outputs = run_mapper_analysis(
        model_name=model_name,
        layerwise_features_path=features_path,
        mapper_config={
            "output_dir": str(tmp_path / "mapper"),
            "datasets": ["ambigqa", "situatedqa"],
            "split": "test",
            "subspaces": ["top_20", "top_50"],
            "lenses": ["signed_distance", "z_0"],
            "n_intervals": [4],
            "overlaps": [0.3],
            "dbscan_eps": 0.9,
            "min_samples": 3,
            "plot_lens": "signed_distance",
            "plot_subspaces": ["top_20", "top_50"],
            "stats_filename": "stats.parquet",
            "nodes_filename": "nodes.parquet",
            "edges_filename": "edges.parquet",
            "comparison_filename": "comparison.parquet",
            "aggregate_filename": "aggregate.parquet",
            "sensitivity_filename": "sensitivity.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
    )

    stats_df = pd.read_parquet(outputs["stats_path"])
    comparison_df = pd.read_parquet(outputs["comparison_path"])
    sensitivity_df = pd.read_parquet(outputs["sensitivity_path"])
    assert not stats_df.empty
    assert {"node_count", "branch_node_count", "connected_components"} <= set(stats_df.columns)
    assert not comparison_df.empty
    assert {"delta_node_count", "delta_branch_node_count"} <= set(comparison_df.columns)
    assert not sensitivity_df.empty
    plots_dir = Path(outputs["report_path"]).parent / "plots"
    assert (plots_dir / "ambigqa__signed_distance__node_count_by_layer.png").exists()
    assert (plots_dir / "ambigqa__signed_distance__branch_delta_heatmap.png").exists()
    assert (plots_dir / "ambigqa__signed_distance__representative_mapper.png").exists()
