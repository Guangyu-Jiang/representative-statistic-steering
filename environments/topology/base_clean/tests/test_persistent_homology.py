from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.features.persistent_homology import run_persistent_homology_analysis
from aen_replication.utils.io_utils import slugify, write_parquet


def _hidden_state_table(dataset: str, matrix: np.ndarray) -> pd.DataFrame:
    labels = np.array([0] * (len(matrix) // 2) + [1] * (len(matrix) - len(matrix) // 2), dtype=int)
    return pd.DataFrame(
        {
            "example_id": [f"{dataset}_ex_{idx}" for idx in range(len(matrix))],
            "pair_id": [f"{dataset}_pair_{idx}" for idx in range(len(matrix))],
            "dataset": dataset,
            "split": ["test"] * len(matrix),
            "text": [f"{dataset} question {idx}" for idx in range(len(matrix))],
            "label_ambiguous": labels,
            "source_id": [f"{dataset}_source_{idx}" for idx in range(len(matrix))],
            "context_type": ["mixed"] * len(matrix),
            "vector": [row.tolist() for row in matrix],
        }
    )


def test_run_persistent_homology_analysis_writes_summary_and_diagrams(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    model_slug = slugify(model_name)
    cache_root = tmp_path / "hidden_states" / model_slug
    cache_root.mkdir(parents=True, exist_ok=True)

    ambig_layer0 = np.array(
        [
            [-2.0, -1.0, 0.0, 0.1],
            [-1.5, -0.8, 0.1, 0.0],
            [-1.8, -1.2, -0.1, 0.0],
            [-1.6, -1.1, 0.0, -0.1],
            [2.0, 1.0, 0.0, -0.1],
            [1.8, 1.2, 0.1, 0.0],
            [1.7, 0.9, -0.1, 0.0],
            [1.9, 1.1, 0.0, 0.1],
        ]
    )
    ambig_layer1 = ambig_layer0[:, [1, 0, 2, 3]]
    situated_layer0 = ambig_layer0 + np.array([0.2, -0.1, 0.0, 0.0])
    situated_layer1 = ambig_layer1 + np.array([-0.2, 0.1, 0.0, 0.0])

    for dataset, layer_tables in {
        "ambigqa": {0: ambig_layer0, 1: ambig_layer1},
        "situatedqa": {0: situated_layer0, 1: situated_layer1},
    }.items():
        for layer, matrix in layer_tables.items():
            write_parquet(
                _hidden_state_table(dataset, matrix),
                cache_root / f"{dataset}__layer_{layer:02d}__mean_pool.parquet",
            )

    feature_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model_name": model_name,
                "layer": layer,
                "readout": "mean_pool",
                "subspace_name": subspace_name,
                "selected_indices": selected_indices,
                "selected_count": len(selected_indices),
            }
            for dataset in ["ambigqa", "situatedqa"]
            for layer in [0, 1]
            for subspace_name, selected_indices in {
                "top_20": [0, 1],
                "top_100": [0, 1, 2],
            }.items()
        ]
    )
    feature_summary_path = tmp_path / "layerwise_aen_summary.parquet"
    write_parquet(feature_summary, feature_summary_path)

    outputs = run_persistent_homology_analysis(
        project_root=tmp_path,
        model_name=model_name,
        feature_summary_path=feature_summary_path,
        cache_dir=tmp_path / "hidden_states",
        ph_config={
            "output_dir": str(tmp_path / "topology"),
            "datasets": ["ambigqa", "situatedqa"],
            "readout": "mean_pool",
            "subspaces": ["top_20", "top_100"],
            "split": "test",
            "sample_n_per_group": 4,
            "min_points": 4,
            "maxdim": 1,
            "coeff": 2,
            "standardize": True,
            "summary_filename": "summary.parquet",
            "diagrams_filename": "diagrams.parquet",
            "subspace_comparison_filename": "comparison.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
        seed=13,
    )

    summary = pd.read_parquet(outputs["summary_path"])
    diagrams = pd.read_parquet(outputs["diagrams_path"])
    comparison = pd.read_parquet(outputs["comparison_path"])

    assert len(summary) == 16
    assert {"h0_total_persistence_norm", "h1_total_persistence_norm", "point_count", "input_dim"} <= set(summary.columns)
    assert set(summary["subspace_name"]) == {"top_20", "top_100"}
    assert set(summary["label_group"]) == {"clear", "ambiguous"}
    assert summary["point_count"].eq(4).all()
    assert not diagrams.empty
    assert {"homology_dim", "birth", "death", "persistence"} <= set(diagrams.columns)
    assert set(comparison["metric_name"]) == {
        "h0_total_persistence_norm",
        "h1_total_persistence_norm",
        "h1_max_persistence_norm",
    }
