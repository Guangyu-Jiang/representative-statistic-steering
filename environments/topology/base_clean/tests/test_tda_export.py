from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.features.layerwise import export_layerwise_tda_features
from aen_replication.utils.io_utils import write_parquet


def _make_hidden_state_table(dataset: str, matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [f"{dataset}_ex_{idx}" for idx in range(len(matrix))],
            "pair_id": [f"{dataset}_pair_{idx}" for idx in range(len(matrix))],
            "dataset": dataset,
            "split": ["train", "train", "test", "test"],
            "text": [f"{dataset} question {idx}" for idx in range(len(matrix))],
            "label_ambiguous": [0, 1, 0, 1],
            "source_id": [f"{dataset}_source_{idx}" for idx in range(len(matrix))],
            "context_type": ["mixed"] * len(matrix),
            "vector": [row.tolist() for row in matrix],
        }
    )


def _write_manifest(
    root: Path,
    model_name: str,
    dataset: str,
    layer_tables: dict[int, pd.DataFrame],
) -> Path:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for layer, table in layer_tables.items():
        parquet_path = dataset_dir / f"layer_{layer:02d}.parquet"
        write_parquet(table, parquet_path)
        files.append(
            {
                "dataset": dataset,
                "layer": layer,
                "readout": "mean_pool",
                "parquet_path": str(parquet_path),
                "metadata_path": str(dataset_dir / f"layer_{layer:02d}.json"),
            }
        )
    manifest_path = root / f"{dataset}_manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset": dataset, "model_name": model_name, "files": files}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def test_export_layerwise_tda_features_builds_long_and_trajectory_tables(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    ambig_layer_0 = np.array(
        [
            [-3.0, 0.1, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [-2.5, -0.1, 0.0, 0.0],
            [2.5, 0.0, 0.0, 0.0],
        ]
    )
    ambig_layer_1 = np.array(
        [
            [0.0, -2.0, 0.2, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, -1.5, -0.2, 0.0],
            [0.0, 1.5, 0.0, 0.0],
        ]
    )
    situated_layer_0 = np.array(
        [
            [-4.0, 0.0, 0.0, 0.1],
            [4.0, 0.0, 0.0, 0.0],
            [-3.5, 0.0, 0.0, -0.1],
            [3.5, 0.0, 0.0, 0.0],
        ]
    )
    situated_layer_1 = np.array(
        [
            [0.0, -3.0, 0.0, 0.1],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, -2.5, 0.0, -0.1],
            [0.0, 2.5, 0.0, 0.0],
        ]
    )

    manifest_paths = [
        _write_manifest(
            root=tmp_path,
            model_name=model_name,
            dataset="ambigqa",
            layer_tables={
                0: _make_hidden_state_table("ambigqa", ambig_layer_0),
                1: _make_hidden_state_table("ambigqa", ambig_layer_1),
            },
        ),
        _write_manifest(
            root=tmp_path,
            model_name=model_name,
            dataset="situatedqa",
            layer_tables={
                0: _make_hidden_state_table("situatedqa", situated_layer_0),
                1: _make_hidden_state_table("situatedqa", situated_layer_1),
            },
        ),
    ]

    export_config = {
        "output_dir": str(tmp_path / "features"),
        "datasets": ["ambigqa", "situatedqa"],
        "readout": "mean_pool",
        "subspaces": ["aed_final", "top_20", "top_50", "top_100", "nonzero_support"],
        "support_probe": {
            "penalty": "l1",
            "solver": "liblinear",
            "class_weight": None,
            "C": 1.0,
            "max_iter": 2000,
            "standardize": True,
        },
        "reduction": "pca",
        "pca_components": 2,
        "layer_features_filename": "layerwise.parquet",
        "trajectories_filename": "trajectories.parquet",
        "trajectory_summary_filename": "trajectory_summary.parquet",
        "layer_summary_filename": "summary.parquet",
        "cross_dataset_overlap_filename": "overlap.parquet",
        "metadata_filename": "metadata.json",
    }
    probe_config = {
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": None,
        "C": 1.0,
        "max_iter": 2000,
        "standardize": False,
        "perturb_top_k": [1],
        "perturb_sigma": 0.25,
        "perturb_trials": 1,
    }

    outputs = export_layerwise_tda_features(
        manifest_paths=manifest_paths,
        probe_config=probe_config,
        export_config=export_config,
        seed=13,
    )

    layer_features = pd.read_parquet(outputs["layer_features_path"])
    trajectories = pd.read_parquet(outputs["trajectories_path"])
    trajectory_summary = pd.read_parquet(outputs["trajectory_summary_path"])
    summary = pd.read_parquet(outputs["layer_summary_path"])
    overlap = pd.read_parquet(outputs["cross_dataset_overlap_path"])

    assert len(layer_features) == 80
    assert {
        "decision_value",
        "signed_distance",
        "z_0",
        "z_1",
        "layer",
        "dataset",
        "subspace_name",
        "selected_count",
    } <= set(layer_features.columns)
    assert set(layer_features["subspace_name"]) == {"aed_final", "top_20", "top_50", "top_100", "nonzero_support"}
    assert len(summary) == 20
    assert set(summary["dataset"]) == {"ambigqa", "situatedqa"}
    assert summary["selected_count"].ge(1).all()
    assert len(trajectories) == 40
    assert trajectories["layer_count"].eq(2).all()
    assert trajectories["layer"].apply(len).eq(2).all()
    assert set(trajectories["subspace_name"]) == {"aed_final", "top_20", "top_50", "top_100", "nonzero_support"}
    assert len(trajectory_summary) == 40
    assert {
        "boundary_crossing_count",
        "boundary_crossing_rate",
        "z_path_length",
        "z_displacement",
        "subspace_name",
        "selected_count_mean",
    } <= set(trajectory_summary.columns)
    assert {"top_5_overlap", "top_10_overlap", "selected_overlap", "subspace_name"} <= set(overlap.columns)
    assert len(overlap) == 10
    assert set(overlap["subspace_name"]) == {"aed_final", "top_20", "top_50", "top_100", "nonzero_support"}
