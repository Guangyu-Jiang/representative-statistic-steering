from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.train.topology_classifier import run_topology_classifier_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_topology_classifier_analysis_writes_feature_and_metric_artifacts(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    rng = np.random.default_rng(13)
    rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in [0, 1]:
            for subspace_name, spread in [("top_20", 0.35), ("top_100", 0.5)]:
                for split, n_examples in [("train", 60), ("test", 30)]:
                    for label_ambiguous, center in [(1, -1.0 - 0.1 * layer), (0, 1.0 + 0.1 * layer)]:
                        for idx in range(n_examples):
                            global_idx = f"{dataset}_{split}_{label_ambiguous}_{idx}"
                            pair_idx = idx if split == "train" else idx + 100
                            rows.append(
                                {
                                    "example_id": global_idx,
                                    "pair_id": f"{dataset}_{split}_{pair_idx}",
                                    "dataset": dataset,
                                    "split": split,
                                    "text": f"{dataset} question {idx}",
                                    "label_ambiguous": label_ambiguous,
                                    "source_id": f"{dataset}_{pair_idx}",
                                    "context_type": "synthetic",
                                    "model_name": model_name,
                                    "layer": layer,
                                    "readout": "mean_pool",
                                    "decision_value": float(center + 0.3 * rng.normal()),
                                    "signed_distance": float(center + 0.25 * rng.normal()),
                                    "aen_k": 5,
                                    "aen_count": 5,
                                    "subspace_name": subspace_name,
                                    "selected_count": 20 if subspace_name == "top_20" else 100,
                                    "effective_reduction_dim": 2,
                                    "z_0": float(center + spread * rng.normal()),
                                    "z_1": float(0.6 * label_ambiguous + 0.4 * layer + spread * rng.normal()),
                                }
                            )

    features_path = tmp_path / "layerwise_features.parquet"
    write_parquet(pd.DataFrame(rows), features_path)

    outputs = run_topology_classifier_analysis(
        model_name=model_name,
        layerwise_features_path=features_path,
        classifier_config={
            "output_dir": str(tmp_path / "topology_classifier"),
            "datasets": ["ambigqa", "situatedqa"],
            "readout": "mean_pool",
            "candidate_subspaces": ["top_20", "top_100"],
            "candidate_layers": [0, 1],
            "layer_selection_strategy": "manual",
            "multilayer_enabled": True,
            "multilayer_top_k": 2,
            "val_fraction": 0.25,
            "neighborhood_k": 8,
            "coordinate_columns": ["z_0", "z_1", "signed_distance"],
            "standardize_coordinates": True,
            "prototype_sample_n": 24,
            "betti_grid_size": 16,
            "maxdim": 1,
            "coeff": 2,
            "classifier": {
                "penalty": "l2",
                "solver": "liblinear",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 4000,
                "standardize": True,
            },
            "feature_table_filename": "features.parquet",
            "candidate_metrics_filename": "candidate_metrics.parquet",
            "final_metrics_filename": "final_metrics.parquet",
            "selected_candidates_filename": "selected_candidates.parquet",
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
        seed=13,
    )

    candidate_df = pd.read_parquet(outputs["candidate_metrics_path"])
    final_df = pd.read_parquet(outputs["final_metrics_path"])
    selected_df = pd.read_parquet(outputs["selected_candidates_path"])
    feature_df = pd.read_parquet(outputs["feature_table_path"])

    assert not candidate_df.empty
    assert {"dataset", "layer", "subspace_name", "feature_set", "val_auroc"} <= set(candidate_df.columns)
    assert not final_df.empty
    assert {"test_auroc", "test_accuracy", "test_f1"} <= set(final_df.columns)
    assert set(final_df["feature_set"]) == {
        "topology_only",
        "geometry_only",
        "hybrid",
        "topology_multilayer",
        "geometry_multilayer",
        "hybrid_multilayer",
    }
    assert not selected_df.empty
    assert {"layer", "subspace_name", "selection_mode", "component_rank"} <= set(selected_df.columns)
    assert {"single_layer", "multilayer_component"} <= set(selected_df["selection_mode"])
    assert not feature_df.empty
    assert {"h0_total_persistence_norm", "h1_wasserstein_to_clear", "knn_distance_mean"} <= set(feature_df.columns)
    assert {"feature_variant", "selected_for_dataset"} <= set(feature_df.columns)
    assert {"single_layer", "multilayer"} <= set(feature_df["feature_variant"].dropna())
    assert any(column.startswith("topology__stack_") for column in feature_df.columns)

    report_dir = Path(outputs["report_path"]).parent
    assert (report_dir / "plots" / "ambigqa__candidate_heatmap.png").exists()
    assert (report_dir / "plots" / "ambigqa__final_classifier_metrics.png").exists()
