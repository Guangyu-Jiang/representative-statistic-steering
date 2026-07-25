"""CLAMBER topology extension experiments.

This runner adds four extension ideas on top of the existing topology setup:

1. Neighborhood persistent homology on question embeddings.
2. Persistence-image features with an RBF-kernel SVM.
3. Multi-layer stacked neighborhood-topology features.
4. Attention-graph topology features (pilot on one model).

The default run targets LLaMA 3.1 8B on CLAMBER 4-way and 9-way label spaces.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aen_replication.config import load_config
from aen_replication.features.ph_descriptors import _normalized_entropy
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import (
    _betti_curve,
    _finite_diagram,
    _persistence_image_features,
    _stacked_summary_features,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)

GROUP4_MAP = {
    "polysemy": "ambiguity",
    "co-reference": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "ICL": "conflicting_condition",
    "none": "clear",
}
GROUP4_ORDER = ["ambiguity", "clear", "conflicting_condition", "missing_condition"]

MODEL_SPECS = {
    "meta_llama_llama_3_1_8b_instruct": {
        "label": "LLaMA 3.1 8B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_clamber.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
    },
    "mistralai_mistral_7b_instruct_v0_3": {
        "label": "Mistral 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/mistral_clamber_pca16.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
    },
    "google_gemma_7b_it": {
        "label": "Gemma 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/gemma_clamber_pca16.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
    },
}

BASE_METHODS = {
    "neighborhood_ph": "Neighborhood PH",
    "pimg_rbf_svm": "Persistence Images + RBF SVM",
    "neighborhood_ph_multilayer": "Neighborhood PH Multi-layer",
    "attention_graph_topology": "Attention-graph Topology",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-slugs",
        nargs="+",
        default=["meta_llama_llama_3_1_8b_instruct"],
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument("--label-spaces", nargs="+", default=["4way", "9way"], choices=["4way", "9way"])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 14, 31])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--topology-components", type=int, default=6)
    parser.add_argument("--neighborhood-k", type=int, default=24)
    parser.add_argument("--betti-grid-size", type=int, default=32)
    parser.add_argument("--persistence-image-grid-side", type=int, default=4)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--coeff", type=int, default=2)
    parser.add_argument("--attention-max-length", type=int, default=96)
    parser.add_argument("--attention-batch-size", type=int, default=4)
    parser.add_argument(
        "--attention-model-slugs",
        nargs="+",
        default=["meta_llama_llama_3_1_8b_instruct"],
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_topology_extensions",
    )
    return parser.parse_args()


def _logistic_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _rbf_svm_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[SVC, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _available_layers(hidden_root: Path) -> list[int]:
    layers = []
    for path in sorted(hidden_root.glob("clamber__layer_*__mean_pool.parquet")):
        layer = int(path.stem.split("__")[1].replace("layer_", ""))
        layers.append(layer)
    if not layers:
        raise FileNotFoundError(f"No CLAMBER mean-pool caches found under {hidden_root}")
    return sorted(set(layers))


def _resolve_layers(hidden_root: Path, requested: list[int]) -> list[int]:
    available = _available_layers(hidden_root)
    resolved: list[int] = []
    for layer in requested:
        if layer in available:
            resolved.append(int(layer))
    if not resolved:
        if len(available) >= 3:
            indices = np.linspace(0, len(available) - 1, num=3, dtype=int)
            resolved = [available[index] for index in indices.tolist()]
        else:
            resolved = available
    return sorted(set(resolved))


def _prepare_label_space(
    meta: pd.DataFrame,
    matrix: np.ndarray,
    *,
    label_space: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = meta.copy()
    vectors = np.asarray(matrix, dtype=np.float32)
    if label_space == "4way":
        mask = frame["subclass"].isin(GROUP4_MAP).to_numpy()
        frame = frame.loc[mask].reset_index(drop=True)
        vectors = vectors[mask]
        frame["target_label"] = frame["subclass"].map(GROUP4_MAP)
        return frame, vectors, list(GROUP4_ORDER)
    if label_space == "9way":
        frame = frame.reset_index(drop=True)
        frame["target_label"] = frame["subclass"].astype(str)
        labels = sorted(frame["target_label"].astype(str).unique().tolist())
        return frame, vectors, labels
    raise ValueError(f"Unsupported label space: {label_space}")


def _split_train_val(
    labels: pd.Series,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _fit_reducer(
    train_matrix: np.ndarray,
    *,
    pca_components: int,
    seed: int,
) -> tuple[StandardScaler, PCA, StandardScaler, int]:
    input_scaler = StandardScaler()
    centered = input_scaler.fit_transform(train_matrix)
    max_components = min(centered.shape[1], max(1, centered.shape[0] - 1))
    n_components = max(1, min(int(pca_components), max_components))
    reducer = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
        whiten=False,
    )
    reduced = reducer.fit_transform(centered)
    reduced_scaler = StandardScaler()
    reduced_scaler.fit(reduced)
    return input_scaler, reducer, reduced_scaler, n_components


def _transform_reducer(
    matrix: np.ndarray,
    *,
    input_scaler: StandardScaler,
    reducer: PCA,
    reduced_scaler: StandardScaler,
) -> np.ndarray:
    transformed = input_scaler.transform(matrix)
    transformed = reducer.transform(transformed)
    transformed = reduced_scaler.transform(transformed)
    return np.asarray(transformed, dtype=np.float32)


def _compute_diagrams(points: np.ndarray, *, maxdim: int, coeff: int) -> list[np.ndarray]:
    if len(points) <= 1:
        return [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
    distance_matrix = squareform(pdist(points, metric="euclidean"))
    diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim, coeff=coeff).get("dgms", [])
    if len(diagrams) < maxdim + 1:
        diagrams = diagrams + [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1 - len(diagrams))]
    return [np.asarray(diagram, dtype=float) for diagram in diagrams[: maxdim + 1]]


def _diagram_descriptors(diagram: np.ndarray, *, prefix: str, grid_size: int) -> dict[str, float]:
    finite = _finite_diagram(diagram)
    betti_bins = max(0, int(grid_size // 4))
    if finite.size == 0:
        result = {
            f"{prefix}_feature_count": 0.0,
            f"{prefix}_total_persistence_norm": 0.0,
            f"{prefix}_max_persistence_norm": 0.0,
            f"{prefix}_mean_persistence": 0.0,
            f"{prefix}_persistence_entropy": 0.0,
            f"{prefix}_betti_curve_auc_norm": 0.0,
        }
        for index in range(betti_bins):
            result[f"{prefix}_betti_bin_{index:02d}"] = 0.0
        return result
    lifetimes = finite[:, 1] - finite[:, 0]
    max_death = float(max(np.max(finite[:, 1]), 1e-6))
    grid = np.linspace(0.0, max_death, grid_size)
    betti = _betti_curve(finite, grid)
    betti_auc = float(getattr(np, "trapezoid", np.trapz)(betti, grid))
    result = {
        f"{prefix}_feature_count": float(len(lifetimes)),
        f"{prefix}_total_persistence_norm": float(lifetimes.sum() / max_death),
        f"{prefix}_max_persistence_norm": float(lifetimes.max() / max_death),
        f"{prefix}_mean_persistence": float(lifetimes.mean()),
        f"{prefix}_persistence_entropy": _normalized_entropy(lifetimes),
        f"{prefix}_betti_curve_auc_norm": float(betti_auc / max_death),
    }
    if betti_bins:
        partitions = np.array_split(betti, betti_bins)
        scale = float(max(np.max(betti), 1.0))
        for index, partition in enumerate(partitions):
            result[f"{prefix}_betti_bin_{index:02d}"] = float(np.mean(partition) / scale)
    return result


def _feature_columns(feature_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    key_cols = {"example_id", "split", "target_label", "layer"}
    pimg_columns = [column for column in feature_df.columns if "_pimg_" in column]
    summary_columns = [
        column
        for column in feature_df.columns
        if column not in key_cols and column not in pimg_columns
    ]
    return summary_columns, pimg_columns, summary_columns + pimg_columns


def _build_neighborhood_feature_frame(
    *,
    query_meta: pd.DataFrame,
    query_coords: np.ndarray,
    reference_meta: pd.DataFrame,
    reference_coords: np.ndarray,
    neighborhood_k: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> pd.DataFrame:
    model = NearestNeighbors(n_neighbors=min(len(reference_coords), neighborhood_k + 1), metric="euclidean")
    model.fit(reference_coords)
    distances, indices = model.kneighbors(query_coords, return_distance=True)
    reference_ids = reference_meta["example_id"].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    iterator = zip(query_meta.to_dict(orient="records"), query_coords, indices, distances, strict=False)
    for row_meta, query_point, candidate_indices, candidate_distances in iterator:
        neighbor_indices = np.asarray(candidate_indices, dtype=int)
        neighbor_distances = np.asarray(candidate_distances, dtype=float)
        if row_meta["split"] == "train":
            keep_mask = reference_ids[neighbor_indices] != str(row_meta["example_id"])
            neighbor_indices = neighbor_indices[keep_mask]
            neighbor_distances = neighbor_distances[keep_mask]
        neighbor_indices = neighbor_indices[:neighborhood_k]
        neighbor_distances = neighbor_distances[:neighborhood_k]
        neighbor_points = reference_coords[neighbor_indices] if len(neighbor_indices) else np.zeros((0, query_coords.shape[1]), dtype=float)
        centered_cloud = np.vstack([np.zeros((1, query_point.shape[0]), dtype=float), neighbor_points - query_point[None, :]])
        diagrams = _compute_diagrams(centered_cloud, maxdim=maxdim, coeff=coeff)
        feature_row = {
            "example_id": str(row_meta["example_id"]),
            "split": str(row_meta["split"]),
            "target_label": str(row_meta["target_label"]),
            "layer": int(row_meta["layer"]),
            "knn_distance_mean": float(np.mean(neighbor_distances)) if len(neighbor_distances) else 0.0,
            "knn_distance_std": float(np.std(neighbor_distances)) if len(neighbor_distances) else 0.0,
            "knn_distance_max": float(np.max(neighbor_distances)) if len(neighbor_distances) else 0.0,
        }
        for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
            diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
            feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=betti_grid_size))
            feature_row.update(
                _persistence_image_features(
                    diagram,
                    prefix=prefix,
                    grid_side=persistence_image_grid_side,
                )
            )
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _fit_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    labels: list[str],
    family: str,
    seed: int,
) -> dict[str, Any]:
    x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_df["target_label"].astype(str).to_numpy()
    x_eval = eval_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_eval = eval_df["target_label"].astype(str).to_numpy()
    if family == "logistic":
        clf, scaler = _logistic_fit(x_train, y_train, seed=seed)
    elif family == "rbf_svm":
        clf, scaler = _rbf_svm_fit(x_train, y_train)
    else:
        raise ValueError(f"Unsupported classifier family: {family}")
    predictions = clf.predict(scaler.transform(x_eval))
    metrics = _multiclass_metrics(y_eval, predictions, labels=labels)
    return {
        "classifier": clf,
        "scaler": scaler,
        "metrics": metrics,
    }


def _candidate_selection_rows(
    *,
    hidden_root: Path,
    layers: list[int],
    label_space: str,
    labels: list[str],
    seed: int,
    val_fraction: float,
    pca_components: int,
    topology_components: int,
    neighborhood_k: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, pd.DataFrame]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[int, dict[str, pd.DataFrame]] = {}

    for layer in layers:
        meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
        meta, matrix, _ = _prepare_label_space(meta, matrix, label_space=label_space)
        meta["layer"] = int(layer)
        train_mask = meta["split"].eq("train").to_numpy()
        train_meta = meta.loc[train_mask].reset_index(drop=True)
        train_matrix = matrix[train_mask]
        inner_idx, val_idx = _split_train_val(train_meta["target_label"], val_fraction=val_fraction, seed=seed + int(layer))
        inner_meta = train_meta.iloc[inner_idx].reset_index(drop=True)
        inner_matrix = train_matrix[inner_idx]
        val_meta = train_meta.iloc[val_idx].reset_index(drop=True)
        val_matrix = train_matrix[val_idx]

        input_scaler, reducer, reduced_scaler, n_components = _fit_reducer(
            inner_matrix,
            pca_components=pca_components,
            seed=seed + 100 + int(layer),
        )
        topology_dim = min(int(topology_components), int(n_components))
        inner_coords = _transform_reducer(
            inner_matrix,
            input_scaler=input_scaler,
            reducer=reducer,
            reduced_scaler=reduced_scaler,
        )[:, :topology_dim]
        val_coords = _transform_reducer(
            val_matrix,
            input_scaler=input_scaler,
            reducer=reducer,
            reduced_scaler=reduced_scaler,
        )[:, :topology_dim]

        inner_features = _build_neighborhood_feature_frame(
            query_meta=inner_meta,
            query_coords=inner_coords,
            reference_meta=inner_meta,
            reference_coords=inner_coords,
            neighborhood_k=neighborhood_k,
            betti_grid_size=betti_grid_size,
            persistence_image_grid_side=persistence_image_grid_side,
            maxdim=maxdim,
            coeff=coeff,
        )
        val_features = _build_neighborhood_feature_frame(
            query_meta=val_meta,
            query_coords=val_coords,
            reference_meta=inner_meta,
            reference_coords=inner_coords,
            neighborhood_k=neighborhood_k,
            betti_grid_size=betti_grid_size,
            persistence_image_grid_side=persistence_image_grid_side,
            maxdim=maxdim,
            coeff=coeff,
        )
        cache[int(layer)] = {
            "train": inner_features,
            "val": val_features,
        }

        summary_columns, pimg_columns, _ = _feature_columns(inner_features)
        neighborhood_eval = _fit_predict(
            inner_features,
            val_features,
            feature_columns=summary_columns,
            labels=labels,
            family="logistic",
            seed=seed + 200 + int(layer),
        )
        pimg_eval = _fit_predict(
            inner_features,
            val_features,
            feature_columns=pimg_columns,
            labels=labels,
            family="rbf_svm",
            seed=seed + 300 + int(layer),
        )
        rows.extend(
            [
                {
                    "method": "neighborhood_ph",
                    "layer": int(layer),
                    "val_accuracy": float(neighborhood_eval["metrics"]["accuracy"]),
                    "val_macro_f1": float(neighborhood_eval["metrics"]["macro_f1"]),
                    "feature_count": int(len(summary_columns)),
                    "selection_signature": str(layer),
                },
                {
                    "method": "pimg_rbf_svm",
                    "layer": int(layer),
                    "val_accuracy": float(pimg_eval["metrics"]["accuracy"]),
                    "val_macro_f1": float(pimg_eval["metrics"]["macro_f1"]),
                    "feature_count": int(len(pimg_columns)),
                    "selection_signature": str(layer),
                },
            ]
        )
    candidate_df = pd.DataFrame(rows).sort_values(
        ["method", "val_macro_f1", "val_accuracy", "layer"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return candidate_df, cache


def _multilayer_selection_rows(
    candidate_cache: dict[int, dict[str, pd.DataFrame]],
    *,
    labels: list[str],
    seed: int,
) -> pd.DataFrame:
    layers = sorted(candidate_cache.keys())
    rows: list[dict[str, Any]] = []
    if len(layers) < 2:
        return pd.DataFrame(rows)

    for width in [2, 3]:
        if width > len(layers):
            continue
        for combo in itertools.combinations(layers, width):
            merged_train: pd.DataFrame | None = None
            merged_val: pd.DataFrame | None = None
            summary_groups: dict[str, list[str]] = {}
            for layer in combo:
                train_frame = candidate_cache[layer]["train"].copy()
                val_frame = candidate_cache[layer]["val"].copy()
                summary_columns, _, _ = _feature_columns(train_frame)
                layer_suffix = f"l{int(layer):02d}"
                layer_train = train_frame.loc[:, ["example_id", "split", "target_label"] + summary_columns].rename(
                    columns={column: f"{column}__{layer_suffix}" for column in summary_columns}
                )
                layer_val = val_frame.loc[:, ["example_id", "split", "target_label"] + summary_columns].rename(
                    columns={column: f"{column}__{layer_suffix}" for column in summary_columns}
                )
                if merged_train is None:
                    merged_train = layer_train
                    merged_val = layer_val
                else:
                    merged_train = merged_train.merge(layer_train, on=["example_id", "split", "target_label"], how="inner")
                    merged_val = merged_val.merge(layer_val, on=["example_id", "split", "target_label"], how="inner")
                for column in summary_columns:
                    summary_groups.setdefault(column, []).append(f"{column}__{layer_suffix}")
            if merged_train is None or merged_val is None:
                continue
            train_summary_df, train_cols_by_group = _stacked_summary_features(merged_train, metric_groups=summary_groups)
            val_summary_df, _ = _stacked_summary_features(merged_val, metric_groups=summary_groups)
            train_final = pd.concat([merged_train.loc[:, ["example_id", "split", "target_label"]].reset_index(drop=True), train_summary_df], axis=1)
            val_final = pd.concat([merged_val.loc[:, ["example_id", "split", "target_label"]].reset_index(drop=True), val_summary_df], axis=1)
            feature_columns = [column for columns in train_cols_by_group.values() for column in columns]
            eval_result = _fit_predict(
                train_final,
                val_final,
                feature_columns=feature_columns,
                labels=labels,
                family="logistic",
                seed=seed + 500 + int(sum(combo)),
            )
            rows.append(
                {
                    "method": "neighborhood_ph_multilayer",
                    "layer": -1,
                    "val_accuracy": float(eval_result["metrics"]["accuracy"]),
                    "val_macro_f1": float(eval_result["metrics"]["macro_f1"]),
                    "feature_count": int(len(feature_columns)),
                    "selection_signature": " | ".join(str(layer) for layer in combo),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy", "selection_signature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_full_layer_features(
    *,
    hidden_root: Path,
    layer: int,
    label_space: str,
    seed: int,
    pca_components: int,
    topology_components: int,
    neighborhood_k: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    meta, matrix, _ = _prepare_label_space(meta, matrix, label_space=label_space)
    meta["layer"] = int(layer)
    train_mask = meta["split"].eq("train").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    train_matrix = matrix[train_mask]
    test_matrix = matrix[test_mask]

    input_scaler, reducer, reduced_scaler, n_components = _fit_reducer(
        train_matrix,
        pca_components=pca_components,
        seed=seed + 900 + int(layer),
    )
    topology_dim = min(int(topology_components), int(n_components))
    train_coords = _transform_reducer(
        train_matrix,
        input_scaler=input_scaler,
        reducer=reducer,
        reduced_scaler=reduced_scaler,
    )[:, :topology_dim]
    test_coords = _transform_reducer(
        test_matrix,
        input_scaler=input_scaler,
        reducer=reducer,
        reduced_scaler=reduced_scaler,
    )[:, :topology_dim]
    train_features = _build_neighborhood_feature_frame(
        query_meta=train_meta,
        query_coords=train_coords,
        reference_meta=train_meta,
        reference_coords=train_coords,
        neighborhood_k=neighborhood_k,
        betti_grid_size=betti_grid_size,
        persistence_image_grid_side=persistence_image_grid_side,
        maxdim=maxdim,
        coeff=coeff,
    )
    test_features = _build_neighborhood_feature_frame(
        query_meta=test_meta,
        query_coords=test_coords,
        reference_meta=train_meta,
        reference_coords=train_coords,
        neighborhood_k=neighborhood_k,
        betti_grid_size=betti_grid_size,
        persistence_image_grid_side=persistence_image_grid_side,
        maxdim=maxdim,
        coeff=coeff,
    )
    return train_features, test_features


def _evaluate_final_layer_method(
    *,
    hidden_root: Path,
    layer: int,
    label_space: str,
    labels: list[str],
    method: str,
    seed: int,
    pca_components: int,
    topology_components: int,
    neighborhood_k: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> dict[str, Any]:
    train_features, test_features = _build_full_layer_features(
        hidden_root=hidden_root,
        layer=layer,
        label_space=label_space,
        seed=seed,
        pca_components=pca_components,
        topology_components=topology_components,
        neighborhood_k=neighborhood_k,
        betti_grid_size=betti_grid_size,
        persistence_image_grid_side=persistence_image_grid_side,
        maxdim=maxdim,
        coeff=coeff,
    )
    summary_columns, pimg_columns, _ = _feature_columns(train_features)
    if method == "neighborhood_ph":
        feature_columns = summary_columns
        family = "logistic"
    elif method == "pimg_rbf_svm":
        feature_columns = pimg_columns
        family = "rbf_svm"
    else:
        raise ValueError(f"Unsupported single-layer final method: {method}")
    result = _fit_predict(
        train_features,
        test_features,
        feature_columns=feature_columns,
        labels=labels,
        family=family,
        seed=seed + 1200 + int(layer),
    )
    return {
        "method": method,
        "layer": int(layer),
        "selection_signature": str(layer),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(result["metrics"]["accuracy"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "confusion_matrix": result["metrics"]["confusion_matrix"],
        "labels": result["metrics"]["labels"],
    }


def _evaluate_final_multilayer(
    *,
    hidden_root: Path,
    layers: list[int],
    label_space: str,
    labels: list[str],
    seed: int,
    pca_components: int,
    topology_components: int,
    neighborhood_k: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> dict[str, Any]:
    merged_train: pd.DataFrame | None = None
    merged_test: pd.DataFrame | None = None
    summary_groups: dict[str, list[str]] = {}
    for layer in layers:
        train_features, test_features = _build_full_layer_features(
            hidden_root=hidden_root,
            layer=layer,
            label_space=label_space,
            seed=seed,
            pca_components=pca_components,
            topology_components=topology_components,
            neighborhood_k=neighborhood_k,
            betti_grid_size=betti_grid_size,
            persistence_image_grid_side=persistence_image_grid_side,
            maxdim=maxdim,
            coeff=coeff,
        )
        summary_columns, _, _ = _feature_columns(train_features)
        suffix = f"l{int(layer):02d}"
        layer_train = train_features.loc[:, ["example_id", "split", "target_label"] + summary_columns].rename(
            columns={column: f"{column}__{suffix}" for column in summary_columns}
        )
        layer_test = test_features.loc[:, ["example_id", "split", "target_label"] + summary_columns].rename(
            columns={column: f"{column}__{suffix}" for column in summary_columns}
        )
        if merged_train is None:
            merged_train = layer_train
            merged_test = layer_test
        else:
            merged_train = merged_train.merge(layer_train, on=["example_id", "split", "target_label"], how="inner")
            merged_test = merged_test.merge(layer_test, on=["example_id", "split", "target_label"], how="inner")
        for column in summary_columns:
            summary_groups.setdefault(column, []).append(f"{column}__{suffix}")
    if merged_train is None or merged_test is None:
        raise ValueError("No multilayer features built.")
    train_summary_df, train_cols_by_group = _stacked_summary_features(merged_train, metric_groups=summary_groups)
    test_summary_df, _ = _stacked_summary_features(merged_test, metric_groups=summary_groups)
    train_final = pd.concat([merged_train.loc[:, ["example_id", "split", "target_label"]].reset_index(drop=True), train_summary_df], axis=1)
    test_final = pd.concat([merged_test.loc[:, ["example_id", "split", "target_label"]].reset_index(drop=True), test_summary_df], axis=1)
    feature_columns = [column for columns in train_cols_by_group.values() for column in columns]
    result = _fit_predict(
        train_final,
        test_final,
        feature_columns=feature_columns,
        labels=labels,
        family="logistic",
        seed=seed + 1600 + int(sum(layers)),
    )
    return {
        "method": "neighborhood_ph_multilayer",
        "layer": -1,
        "selection_signature": " | ".join(str(layer) for layer in layers),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(result["metrics"]["accuracy"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "confusion_matrix": result["metrics"]["confusion_matrix"],
        "labels": result["metrics"]["labels"],
    }


def _attention_graph_summaries(attn: np.ndarray) -> dict[str, float]:
    token_count = int(attn.shape[0])
    diagonal = np.diag(attn)
    offdiag_mask = ~np.eye(token_count, dtype=bool)
    offdiag = attn[offdiag_mask] if token_count > 1 else np.zeros(0, dtype=float)
    row_probs = np.clip(attn, 1e-12, 1.0)
    entropy = -(row_probs * np.log(row_probs)).sum(axis=1)
    norm = float(np.log(max(token_count, 2)))
    entropy = entropy / norm
    return {
        "token_count": float(token_count),
        "attn_diag_mean": float(np.mean(diagonal)) if diagonal.size else 0.0,
        "attn_diag_std": float(np.std(diagonal)) if diagonal.size else 0.0,
        "attn_offdiag_mean": float(np.mean(offdiag)) if offdiag.size else 0.0,
        "attn_offdiag_std": float(np.std(offdiag)) if offdiag.size else 0.0,
        "attn_entropy_mean": float(np.mean(entropy)) if entropy.size else 0.0,
        "attn_entropy_std": float(np.std(entropy)) if entropy.size else 0.0,
        "attn_symmetry_gap": float(np.mean(np.abs(attn - attn.T))) if token_count > 1 else 0.0,
    }


def _compute_attention_graph_diagrams(attn: np.ndarray, *, maxdim: int, coeff: int) -> list[np.ndarray]:
    affinity = 0.5 * (attn + attn.T)
    affinity = np.clip(affinity, 1e-6, 1.0)
    np.fill_diagonal(affinity, 1.0)
    edge_lengths = -np.log(affinity)
    np.fill_diagonal(edge_lengths, 0.0)
    graph_distances = shortest_path(edge_lengths, directed=False, unweighted=False)
    diagrams = ripser(graph_distances, distance_matrix=True, maxdim=maxdim, coeff=coeff).get("dgms", [])
    if len(diagrams) < maxdim + 1:
        diagrams = diagrams + [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1 - len(diagrams))]
    return [np.asarray(diagram, dtype=float) for diagram in diagrams[: maxdim + 1]]


def _extract_attention_feature_table(
    *,
    bundle: HFModelBundle,
    base_df: pd.DataFrame,
    layers: list[int],
    max_length: int,
    batch_size: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> pd.DataFrame:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    rows: list[dict[str, Any]] = []

    batches = [base_df.iloc[start : start + batch_size] for start in range(0, len(base_df), batch_size)]
    for batch_df in tqdm(batches, desc="attention_topology_extract", leave=False):
        encoded = tokenizer(
            batch_df["text"].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"]
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**model_inputs, output_attentions=True, use_cache=False)
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attention tensors.")

        for batch_index, row_meta in enumerate(batch_df.to_dict(orient="records")):
            valid_tokens = int(attention_mask[batch_index].sum().item())
            for layer in layers:
                layer_attn = attentions[layer][batch_index, :, :valid_tokens, :valid_tokens].detach().float().cpu().numpy()
                pooled_attn = np.asarray(layer_attn.mean(axis=0), dtype=float)
                diagrams = _compute_attention_graph_diagrams(pooled_attn, maxdim=maxdim, coeff=coeff)
                feature_row = {
                    "example_id": str(row_meta["example_id"]),
                    "split": str(row_meta["split"]),
                    "subclass": str(row_meta["subclass"]),
                    "layer": int(layer),
                }
                feature_row.update(_attention_graph_summaries(pooled_attn))
                for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
                    diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
                    feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=betti_grid_size))
                    feature_row.update(
                        _persistence_image_features(
                            diagram,
                            prefix=prefix,
                            grid_side=persistence_image_grid_side,
                        )
                    )
                rows.append(feature_row)
        del outputs, attentions, model_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _load_or_build_attention_features(
    *,
    model_slug: str,
    model_config_path: str,
    layers: list[int],
    output_root: Path,
    max_length: int,
    batch_size: int,
    betti_grid_size: int,
    persistence_image_grid_side: int,
    maxdim: int,
    coeff: int,
) -> Path:
    feature_path = output_root / f"{model_slug}__attention_graph_features.parquet"
    if feature_path.exists():
        return feature_path
    config = load_config(model_config_path)
    data_path = Path("/home/ubuntu/sparse_neurons_ambiguity_replication/data/processed/clamber_pairs.parquet")
    base_df = pd.read_parquet(data_path).loc[:, ["example_id", "text", "split", "subclass"]].copy()
    bundle = load_hf_model(config["model"], config["extraction"])
    if hasattr(bundle.model, "set_attn_implementation"):
        bundle.model.set_attn_implementation("eager")
    feature_df = _extract_attention_feature_table(
        bundle=bundle,
        base_df=base_df,
        layers=layers,
        max_length=max_length,
        batch_size=batch_size,
        betti_grid_size=betti_grid_size,
        persistence_image_grid_side=persistence_image_grid_side,
        maxdim=maxdim,
        coeff=coeff,
    )
    write_parquet(feature_df, feature_path)
    return feature_path


def _evaluate_attention_topology(
    *,
    feature_df: pd.DataFrame,
    label_space: str,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = feature_df.copy()
    if label_space == "4way":
        df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
        df["target_label"] = df["subclass"].map(GROUP4_MAP)
        labels = list(GROUP4_ORDER)
    else:
        df["target_label"] = df["subclass"].astype(str)
        labels = sorted(df["target_label"].unique().tolist())

    rows: list[dict[str, Any]] = []
    feature_columns = [
        column
        for column in df.columns
        if column not in {"example_id", "split", "subclass", "target_label", "layer"}
    ]
    train_all = df.loc[df["split"].eq("train")].copy()
    test_all = df.loc[df["split"].eq("test")].copy()

    for layer in sorted(df["layer"].unique().tolist()):
        layer_train = train_all.loc[train_all["layer"].eq(layer)].reset_index(drop=True)
        train_idx, val_idx = _split_train_val(layer_train["target_label"], val_fraction=val_fraction, seed=seed + int(layer))
        inner_train = layer_train.iloc[train_idx].reset_index(drop=True)
        val_df = layer_train.iloc[val_idx].reset_index(drop=True)
        eval_result = _fit_predict(
            inner_train,
            val_df,
            feature_columns=feature_columns,
            labels=labels,
            family="logistic",
            seed=seed + 2000 + int(layer),
        )
        rows.append(
            {
                "method": "attention_graph_topology",
                "layer": int(layer),
                "val_accuracy": float(eval_result["metrics"]["accuracy"]),
                "val_macro_f1": float(eval_result["metrics"]["macro_f1"]),
                "feature_count": int(len(feature_columns)),
                "selection_signature": str(layer),
            }
        )
    candidate_df = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy", "layer"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_layer = int(candidate_df.iloc[0]["layer"])
    train_df = train_all.loc[train_all["layer"].eq(best_layer)].reset_index(drop=True)
    test_df = test_all.loc[test_all["layer"].eq(best_layer)].reset_index(drop=True)
    final_result = _fit_predict(
        train_df,
        test_df,
        feature_columns=feature_columns,
        labels=labels,
        family="logistic",
        seed=seed + 2100 + best_layer,
    )
    final = {
        "method": "attention_graph_topology",
        "layer": int(best_layer),
        "selection_signature": str(best_layer),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(final_result["metrics"]["accuracy"]),
        "macro_f1": float(final_result["metrics"]["macro_f1"]),
        "confusion_matrix": final_result["metrics"]["confusion_matrix"],
        "labels": final_result["metrics"]["labels"],
    }
    return candidate_df, final


def _render_report(results_df: pd.DataFrame, candidate_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# CLAMBER Topology Extensions",
        "",
        "This report evaluates four topology extensions on CLAMBER:",
        "",
        "1. Neighborhood PH on reduced question embeddings.",
        "2. Persistence-image features with an RBF-kernel SVM.",
        "3. Multi-layer stacked neighborhood-topology features.",
        "4. Attention-graph topology features.",
        "",
    ]
    for (model_slug, label_space), group_df in results_df.groupby(["model_slug", "label_space"], dropna=False):
        lines.extend([f"## {model_slug} / {label_space}", ""])
        ordered = group_df.sort_values(["macro_f1", "accuracy"], ascending=False)
        for row in ordered.to_dict(orient="records"):
            method_label = BASE_METHODS.get(str(row["method"]), str(row["method"]))
            lines.append(
                f"- `{method_label}`: accuracy `{row['accuracy']:.4f}`, macro-F1 `{row['macro_f1']:.4f}`, "
                f"selection `{row['selection_signature']}`, features `{int(row['feature_count'])}`."
            )
        lines.append("")
        candidate_subset = candidate_df.loc[
            candidate_df["model_slug"].eq(model_slug) & candidate_df["label_space"].eq(label_space)
        ].sort_values(["method", "val_macro_f1", "val_accuracy"], ascending=[True, False, False])
        if not candidate_subset.empty:
            lines.extend(["### Validation Selections", ""])
            for row in candidate_subset.head(12).to_dict(orient="records"):
                method_label = BASE_METHODS.get(str(row["method"]), str(row["method"]))
                lines.append(
                    f"- `{method_label}` / `{row['selection_signature']}`: val acc `{row['val_accuracy']:.4f}`, "
                    f"val macro-F1 `{row['val_macro_f1']:.4f}`."
                )
            lines.append("")
    write_markdown(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for model_slug in args.model_slugs:
        spec = MODEL_SPECS[model_slug]
        hidden_root = Path(spec["hidden_root"])
        layers = _resolve_layers(hidden_root, list(args.layers))
        LOGGER.info("Running neighborhood extensions for %s on layers %s", model_slug, layers)
        for label_space in args.label_spaces:
            meta_ref, matrix_ref = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layers[0]):02d}__mean_pool.parquet")
            _, _, labels = _prepare_label_space(meta_ref, matrix_ref, label_space=label_space)
            candidate_df, candidate_cache = _candidate_selection_rows(
                hidden_root=hidden_root,
                layers=layers,
                label_space=label_space,
                labels=labels,
                seed=args.seed,
                val_fraction=args.val_fraction,
                pca_components=args.pca_components,
                topology_components=args.topology_components,
                neighborhood_k=args.neighborhood_k,
                betti_grid_size=args.betti_grid_size,
                persistence_image_grid_side=args.persistence_image_grid_side,
                maxdim=args.maxdim,
                coeff=args.coeff,
            )
            multilayer_candidates = _multilayer_selection_rows(
                candidate_cache,
                labels=labels,
                seed=args.seed,
            )
            if not multilayer_candidates.empty:
                candidate_df = pd.concat([candidate_df, multilayer_candidates], ignore_index=True)
            candidate_df["model_slug"] = model_slug
            candidate_df["label_space"] = label_space
            candidate_rows.extend(candidate_df.to_dict(orient="records"))

            for method in ["neighborhood_ph", "pimg_rbf_svm"]:
                best_row = candidate_df.loc[candidate_df["method"].eq(method)].sort_values(
                    ["val_macro_f1", "val_accuracy", "layer"],
                    ascending=[False, False, True],
                ).iloc[0]
                final = _evaluate_final_layer_method(
                    hidden_root=hidden_root,
                    layer=int(best_row["layer"]),
                    label_space=label_space,
                    labels=labels,
                    method=method,
                    seed=args.seed,
                    pca_components=args.pca_components,
                    topology_components=args.topology_components,
                    neighborhood_k=args.neighborhood_k,
                    betti_grid_size=args.betti_grid_size,
                    persistence_image_grid_side=args.persistence_image_grid_side,
                    maxdim=args.maxdim,
                    coeff=args.coeff,
                )
                final["model_slug"] = model_slug
                final["label_space"] = label_space
                final_rows.append(final)

            if not multilayer_candidates.empty:
                best_multi = multilayer_candidates.iloc[0]
                best_layers = [int(item.strip()) for item in str(best_multi["selection_signature"]).split("|")]
                final_multi = _evaluate_final_multilayer(
                    hidden_root=hidden_root,
                    layers=best_layers,
                    label_space=label_space,
                    labels=labels,
                    seed=args.seed,
                    pca_components=args.pca_components,
                    topology_components=args.topology_components,
                    neighborhood_k=args.neighborhood_k,
                    betti_grid_size=args.betti_grid_size,
                    persistence_image_grid_side=args.persistence_image_grid_side,
                    maxdim=args.maxdim,
                    coeff=args.coeff,
                )
                final_multi["model_slug"] = model_slug
                final_multi["label_space"] = label_space
                final_rows.append(final_multi)

    attention_dir = ensure_dir(output_root / "attention_features")
    for model_slug in args.attention_model_slugs:
        if model_slug not in args.model_slugs:
            continue
        spec = MODEL_SPECS[model_slug]
        layers = _resolve_layers(Path(spec["hidden_root"]), list(args.layers))
        LOGGER.info("Running attention-topology pilot for %s on layers %s", model_slug, layers)
        attention_path = _load_or_build_attention_features(
            model_slug=model_slug,
            model_config_path=spec["config_path"],
            layers=layers,
            output_root=attention_dir,
            max_length=args.attention_max_length,
            batch_size=args.attention_batch_size,
            betti_grid_size=args.betti_grid_size,
            persistence_image_grid_side=args.persistence_image_grid_side,
            maxdim=args.maxdim,
            coeff=args.coeff,
        )
        attention_df = pd.read_parquet(attention_path)
        for label_space in args.label_spaces:
            candidate_df, final = _evaluate_attention_topology(
                feature_df=attention_df,
                label_space=label_space,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )
            candidate_df["model_slug"] = model_slug
            candidate_df["label_space"] = label_space
            candidate_rows.extend(candidate_df.to_dict(orient="records"))
            final["model_slug"] = model_slug
            final["label_space"] = label_space
            final_rows.append(final)

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["model_slug", "label_space", "method", "val_macro_f1", "val_accuracy"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    results_df = pd.DataFrame(final_rows).sort_values(
        ["model_slug", "label_space", "macro_f1", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    candidate_path = output_root / "clamber_topology_extensions_candidates.parquet"
    results_path = output_root / "clamber_topology_extensions_results.parquet"
    report_path = output_root / "clamber_topology_extensions_report.md"
    metadata_path = output_root / "clamber_topology_extensions_metadata.json"
    write_parquet(candidate_df, candidate_path)
    write_parquet(results_df, results_path)
    _render_report(results_df, candidate_df, report_path)
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "model_slugs": list(args.model_slugs),
            "attention_model_slugs": list(args.attention_model_slugs),
            "label_spaces": list(args.label_spaces),
            "layers": list(args.layers),
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "pca_components": int(args.pca_components),
            "topology_components": int(args.topology_components),
            "neighborhood_k": int(args.neighborhood_k),
            "betti_grid_size": int(args.betti_grid_size),
            "persistence_image_grid_side": int(args.persistence_image_grid_side),
            "maxdim": int(args.maxdim),
            "coeff": int(args.coeff),
            "candidate_path": str(candidate_path),
            "results_path": str(results_path),
            "report_path": str(report_path),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
