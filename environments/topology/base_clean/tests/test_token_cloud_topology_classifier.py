from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.train.token_cloud_topology_classifier import (
    _prototype_diagrams_from_clouds,
    build_token_cloud_feature_frame,
    run_token_cloud_topology_classifier_from_features,
)


def test_run_token_cloud_topology_classifier_from_synthetic_clouds(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    cloud_rows = []
    for dataset in ["ambigqa", "situatedqa"]:
        for layer in [0, 2]:
            for split, n_pairs in [("train", 18), ("test", 12)]:
                for label_ambiguous, center in [(1, -0.8 - 0.1 * layer), (0, 0.8 + 0.1 * layer)]:
                    for idx in range(n_pairs):
                        pair_id = idx if split == "train" else idx + 100
                        token_count = 9 + (idx % 3)
                        cloud = center + 0.20 * rng.normal(size=(token_count, 4))
                        cloud_rows.append(
                            {
                                "example_id": f"{dataset}_{split}_{label_ambiguous}_{idx}",
                                "pair_id": f"{dataset}_{split}_{pair_id}",
                                "dataset": dataset,
                                "split": split,
                                "label_ambiguous": label_ambiguous,
                                "layer": layer,
                                "token_count": token_count,
                                "cloud": cloud.astype(np.float32),
                            }
                        )

    cloud_df = pd.DataFrame(cloud_rows)
    config = {
        "parallel_jobs": 2,
        "prototype_token_cap": 96,
        "distance_metric": "euclidean",
        "betti_grid_size": 16,
        "persistence_image_grid_side": 3,
        "maxdim": 1,
        "coeff": 2,
        "val_fraction": 0.25,
        "multilayer_enabled": True,
        "multilayer_top_k": 2,
        "output_dir": str(tmp_path / "token_cloud"),
        "datasets": ["ambigqa", "situatedqa"],
        "feature_table_filename": "features.parquet",
        "candidate_metrics_filename": "candidate_metrics.parquet",
        "final_metrics_filename": "final_metrics.parquet",
        "selected_candidates_filename": "selected_candidates.parquet",
        "report_filename": "report.md",
        "metadata_filename": "metadata.json",
        "classifier": {
            "penalty": "l2",
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 4000,
            "standardize": True,
        },
    }
    prototype_map = _prototype_diagrams_from_clouds(cloud_df, layers=[0, 2], config=config, seed=19)
    feature_df = build_token_cloud_feature_frame(cloud_df, prototype_map=prototype_map, config=config)

    assert not feature_df.empty
    assert {"layer", "token_count", "h0_total_persistence_norm", "h1_feature_count"} <= set(feature_df.columns)
    assert any(column.startswith("h0_pimg_") for column in feature_df.columns)
    assert any(column.startswith("h1_pimg_") for column in feature_df.columns)

    outputs = run_token_cloud_topology_classifier_from_features(
        model_name="unit/test-token-cloud",
        feature_df=feature_df,
        classifier_config=config,
        seed=19,
    )

    candidate_df = pd.read_parquet(outputs["candidate_metrics_path"])
    final_df = pd.read_parquet(outputs["final_metrics_path"])
    selected_df = pd.read_parquet(outputs["selected_candidates_path"])
    saved_feature_df = pd.read_parquet(outputs["feature_table_path"])

    assert not candidate_df.empty
    assert set(candidate_df["feature_set"]) == {"topology_only"}
    assert {"dataset", "layer", "val_auroc"} <= set(candidate_df.columns)

    assert not final_df.empty
    assert {"topology_only", "topology_multilayer"} <= set(final_df["feature_set"])
    assert {"dataset", "test_auroc", "test_accuracy", "test_f1"} <= set(final_df.columns)

    assert not selected_df.empty
    assert {"single_layer", "multilayer_component"} <= set(selected_df["selection_mode"])

    assert not saved_feature_df.empty
    assert {"feature_variant", "token_count"} <= set(saved_feature_df.columns)
    assert {"single_layer", "multilayer"} <= set(saved_feature_df["feature_variant"].dropna())

    assert Path(outputs["plots_dir"]).exists()
    assert Path(outputs["visualization_report_path"]).exists()
    assert (Path(outputs["plots_dir"]) / "ambigqa__descriptor_trajectories.png").exists()
    assert (Path(outputs["plots_dir"]) / "ambigqa__best_layer_feature_distributions.png").exists()
