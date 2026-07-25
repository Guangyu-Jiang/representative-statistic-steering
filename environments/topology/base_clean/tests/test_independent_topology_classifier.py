from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.train.independent_topology_classifier import run_independent_topology_classifier_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_independent_topology_classifier_analysis_writes_expected_artifacts(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    model_slug = "unit_test_model"
    hidden_root = tmp_path / "hidden_states" / model_slug
    rng = np.random.default_rng(17)

    for dataset in ["ambigqa", "situatedqa"]:
        for layer in [0, 1, 2]:
            rows = []
            for split, n_examples in [("train", 30), ("test", 16)]:
                for label_ambiguous, center in [(1, -0.8 - 0.2 * layer), (0, 0.8 + 0.2 * layer)]:
                    for idx in range(n_examples):
                        pair_idx = idx if split == "train" else idx + 100
                        vector = center + 0.35 * rng.normal(size=12)
                        rows.append(
                            {
                                "example_id": f"{dataset}_{split}_{label_ambiguous}_{idx}",
                                "pair_id": f"{dataset}_{split}_{pair_idx}",
                                "dataset": dataset,
                                "split": split,
                                "text": f"{dataset} question {idx}",
                                "label_ambiguous": label_ambiguous,
                                "source_id": f"{dataset}_{pair_idx}",
                                "context_type": "synthetic",
                                "vector": vector.tolist(),
                            }
                        )
            write_parquet(pd.DataFrame(rows), hidden_root / f"{dataset}__layer_{layer:02d}__mean_pool.parquet")

    outputs = run_independent_topology_classifier_analysis(
        model_name=model_name,
        hidden_state_root=hidden_root,
        classifier_config={
            "output_dir": str(tmp_path / "independent_topology_classifier"),
            "datasets": ["ambigqa", "situatedqa"],
            "readout": "mean_pool",
            "candidate_layers": [0, 1, 2],
            "layer_selection_strategy": "all",
            "max_candidate_layers": 3,
            "val_fraction": 0.25,
            "pca_components": 4,
            "geometry_components": 4,
            "topology_components": 3,
            "standardize_hidden_states": True,
            "standardize_reduced_coordinates": True,
            "pca_whiten": False,
            "neighborhood_k": 8,
            "betti_grid_size": 16,
            "maxdim": 1,
            "coeff": 2,
            "multilayer_enabled": True,
            "multilayer_top_k": 2,
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
        seed=17,
    )

    candidate_df = pd.read_parquet(outputs["candidate_metrics_path"])
    final_df = pd.read_parquet(outputs["final_metrics_path"])
    selected_df = pd.read_parquet(outputs["selected_candidates_path"])
    feature_df = pd.read_parquet(outputs["feature_table_path"])

    assert not candidate_df.empty
    assert {"dataset", "layer", "feature_set", "val_auroc"} <= set(candidate_df.columns)
    assert set(candidate_df["feature_set"]) == {"topology_only", "geometry_only", "hybrid"}

    assert not final_df.empty
    assert {"dataset", "feature_set", "test_auroc", "test_accuracy", "test_f1"} <= set(final_df.columns)
    assert set(final_df["feature_set"]) == {
        "topology_only",
        "geometry_only",
        "hybrid",
        "topology_multilayer",
        "geometry_multilayer",
        "hybrid_multilayer",
    }

    assert not selected_df.empty
    assert {"dataset", "feature_set", "selection_mode", "component_rank"} <= set(selected_df.columns)
    assert {"single_layer", "multilayer_component"} <= set(selected_df["selection_mode"])

    assert not feature_df.empty
    assert {"h0_total_persistence_norm", "h1_feature_count", "pc_00", "knn_distance_mean"} <= set(feature_df.columns)
    assert {"feature_variant", "selected_for_dataset", "selected_for_feature_set"} <= set(feature_df.columns)
    assert {"single_layer", "multilayer"} <= set(feature_df["feature_variant"].dropna())
    assert any(column.startswith("topology__stack_") for column in feature_df.columns)

    report_dir = Path(outputs["report_path"]).parent
    assert (report_dir / "plots" / "ambigqa__topology_only__candidate_heatmap.png").exists()
    assert (report_dir / "plots" / "ambigqa__final_classifier_metrics.png").exists()
