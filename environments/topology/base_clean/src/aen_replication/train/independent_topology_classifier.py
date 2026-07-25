"""Independent ambiguity classifiers from raw hidden states.

This stage avoids the AEN/probe-derived layerwise ambiguity exports and instead
builds unsupervised per-layer geometry/topology features directly from the
cached mean-pooled hidden states.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import bottleneck, wasserstein
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, slugify, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)

TOPOLOGY_PREFIXES = ("h0_", "h1_")
GEOMETRY_PREFIXES = ("pc_",)
GEOMETRY_EXTRA_COLUMNS = [
    "reduced_norm",
    "knn_distance_mean",
    "knn_distance_std",
    "knn_distance_max",
    "local_centroid_distance",
    "local_spread_mean",
    "local_spread_max",
]
LABEL_COLORS = {
    "topology_only": "#4C78A8",
    "geometry_only": "#72B7B2",
    "hybrid": "#E45756",
    "topology_multilayer": "#F58518",
    "geometry_multilayer": "#54A24B",
    "hybrid_multilayer": "#B279A2",
}
BASE_KEY_COLUMNS = ["example_id", "pair_id", "dataset", "split", "label_ambiguous"]
SINGLE_FEATURE_SETS = ["topology_only", "geometry_only", "hybrid"]
MULTILAYER_FEATURE_SETS = ["topology_multilayer", "geometry_multilayer", "hybrid_multilayer"]
LABEL_NAMES = {0: "clear", 1: "ambiguous"}


def _normalized_entropy(lifetimes: np.ndarray) -> float:
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if lifetimes.size <= 1:
        return 0.0
    weights = lifetimes / lifetimes.sum()
    entropy = float(-(weights * np.log(weights + 1e-12)).sum())
    return float(entropy / np.log(len(weights)))


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size <= 1:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n)
    return float(max(gini, 0.0))


def _persistence_concentration_features(lifetimes: np.ndarray, *, prefix: str) -> dict[str, float]:
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if lifetimes.size == 0:
        return {
            f"{prefix}_top1_persistence": 0.0,
            f"{prefix}_top3_persistence_sum": 0.0,
            f"{prefix}_top5_persistence_fraction": 0.0,
            f"{prefix}_persistence_gini": 0.0,
        }
    sorted_desc = np.sort(lifetimes)[::-1]
    total = float(sorted_desc.sum())
    top5 = float(sorted_desc[:5].sum()) if total > 0.0 else 0.0
    return {
        f"{prefix}_top1_persistence": float(sorted_desc[0]),
        f"{prefix}_top3_persistence_sum": float(sorted_desc[:3].sum()),
        f"{prefix}_top5_persistence_fraction": float(top5 / total) if total > 0.0 else 0.0,
        f"{prefix}_persistence_gini": _gini(sorted_desc),
    }


def _betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros_like(grid, dtype=float)
    births = diagram[:, 0][:, None]
    deaths = diagram[:, 1][:, None]
    active = (births <= grid[None, :]) & (grid[None, :] < deaths)
    return active.sum(axis=0).astype(float)


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros((0, 2), dtype=float)
    finite = np.asarray(diagram, dtype=float)
    finite = finite[np.isfinite(finite[:, 1])]
    if finite.size == 0:
        return np.zeros((0, 2), dtype=float)
    lifetimes = finite[:, 1] - finite[:, 0]
    finite = finite[lifetimes > 0]
    if finite.size == 0:
        return np.zeros((0, 2), dtype=float)
    return finite


def _diagram_descriptors(diagram: np.ndarray, *, prefix: str, grid_size: int) -> dict[str, float]:
    finite = _finite_diagram(diagram)
    betti_bins = max(0, int(grid_size // 4))
    if finite.size == 0:
        empty = {
            f"{prefix}_feature_count": 0.0,
            f"{prefix}_total_persistence_norm": 0.0,
            f"{prefix}_max_persistence_norm": 0.0,
            f"{prefix}_mean_persistence": 0.0,
            f"{prefix}_persistence_entropy": 0.0,
            f"{prefix}_betti_curve_auc_norm": 0.0,
            f"{prefix}_mean_birth": 0.0,
            f"{prefix}_std_birth": 0.0,
            f"{prefix}_mean_death": 0.0,
            f"{prefix}_std_death": 0.0,
            f"{prefix}_top1_persistence": 0.0,
            f"{prefix}_top3_persistence_sum": 0.0,
            f"{prefix}_top5_persistence_fraction": 0.0,
            f"{prefix}_persistence_gini": 0.0,
        }
        for index in range(betti_bins):
            empty[f"{prefix}_betti_bin_{index:02d}"] = 0.0
        return empty
    births = finite[:, 0]
    deaths = finite[:, 1]
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
        f"{prefix}_mean_birth": float(np.mean(births)),
        f"{prefix}_std_birth": float(np.std(births)),
        f"{prefix}_mean_death": float(np.mean(deaths)),
        f"{prefix}_std_death": float(np.std(deaths)),
    }
    result.update(_persistence_concentration_features(lifetimes, prefix=prefix))
    if betti_bins:
        partitions = np.array_split(betti, betti_bins)
        scale = float(max(np.max(betti), 1.0))
        for index, partition in enumerate(partitions):
            result[f"{prefix}_betti_bin_{index:02d}"] = float(np.mean(partition) / scale)
    return result


def _persistence_image_features(diagram: np.ndarray, *, prefix: str, grid_side: int) -> dict[str, float]:
    finite = _finite_diagram(diagram)
    if grid_side <= 0:
        return {}
    result = {f"{prefix}_pimg_{row:02d}_{col:02d}": 0.0 for row in range(grid_side) for col in range(grid_side)}
    if finite.size == 0:
        return result
    births = finite[:, 0]
    persistences = finite[:, 1] - finite[:, 0]
    max_scale = float(max(np.max(finite[:, 1]), 1e-6))
    birth_coords = np.clip(births / max_scale, 0.0, 1.0 - 1e-8)
    persistence_coords = np.clip(persistences / max_scale, 0.0, 1.0 - 1e-8)
    hist, _, _ = np.histogram2d(
        persistence_coords,
        birth_coords,
        bins=grid_side,
        range=[[0.0, 1.0], [0.0, 1.0]],
        weights=persistences / max_scale,
    )
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    for row in range(grid_side):
        for col in range(grid_side):
            result[f"{prefix}_pimg_{row:02d}_{col:02d}"] = float(hist[row, col])
    return result


def _safe_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_diagram(left)
    right = _finite_diagram(right)
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(wasserstein(left, right, matching=False))


def _safe_bottleneck(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_diagram(left)
    right = _finite_diagram(right)
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(bottleneck(left, right, matching=False))


def _extract_model_signal(model: Any) -> tuple[list[float], float]:
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float).ravel().tolist()
        intercept_attr = getattr(model, "intercept_", np.zeros(1, dtype=float))
        intercept = float(np.asarray(intercept_attr, dtype=float).ravel()[0])
        return coefficients, intercept
    if hasattr(model, "feature_importances_"):
        coefficients = np.asarray(model.feature_importances_, dtype=float).ravel().tolist()
        return coefficients, 0.0
    return [], 0.0


def _prototype_diagrams(
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    sample_n: int,
    maxdim: int,
    coeff: int,
    seed: int,
    distance_metric: str,
) -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    prototypes: dict[str, list[np.ndarray]] = {}
    for label_value, label_name in LABEL_NAMES.items():
        indices = np.flatnonzero(reference_labels == label_value)
        if len(indices) == 0:
            prototypes[label_name] = [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
            continue
        if len(indices) > sample_n:
            indices = np.sort(rng.choice(indices, size=sample_n, replace=False))
        prototypes[label_name] = _compute_diagrams(
            reference_points[indices],
            maxdim=maxdim,
            coeff=coeff,
            distance_metric=distance_metric,
        )
    return prototypes


def _compute_diagrams(points: np.ndarray, *, maxdim: int, coeff: int, distance_metric: str = "euclidean") -> list[np.ndarray]:
    if len(points) <= 1:
        return [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
    distance_matrix = squareform(pdist(points, metric=distance_metric))
    result = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim, coeff=coeff)
    diagrams = result.get("dgms", [])
    if len(diagrams) < maxdim + 1:
        diagrams = diagrams + [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1 - len(diagrams))]
    return [np.asarray(diagram, dtype=float) for diagram in diagrams[: maxdim + 1]]


def _fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, StandardScaler | None]:
    family = str(config.get("family", "logistic"))
    scaler = None
    x_fit = x_train
    standardize = bool(config.get("standardize", True))
    if standardize and family == "logistic":
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)
    if family == "logistic":
        clf = LogisticRegression(
            penalty=str(config.get("penalty", "l2")),
            solver=str(config.get("solver", "liblinear")),
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 4000)),
            class_weight=config.get("class_weight"),
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*penalty=l1 with l1_ratio=0.0.*", category=UserWarning)
            clf.fit(x_fit, y_train)
        return clf, scaler

    if family == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 400)),
            max_depth=config.get("max_depth"),
            min_samples_leaf=int(config.get("min_samples_leaf", 1)),
            class_weight=config.get("class_weight"),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
        )
        clf.fit(x_fit, y_train)
        return clf, scaler

    if family == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=int(config.get("n_estimators", 500)),
            max_depth=config.get("max_depth"),
            min_samples_leaf=int(config.get("min_samples_leaf", 1)),
            class_weight=config.get("class_weight"),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
        )
        clf.fit(x_fit, y_train)
        return clf, scaler

    if family == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=config.get("max_depth"),
            max_iter=int(config.get("max_iter", 300)),
            min_samples_leaf=int(config.get("min_samples_leaf", 20)),
            random_state=seed,
        )
        clf.fit(x_fit, y_train)
        return clf, scaler

    raise ValueError(f"Unsupported classifier family: {family}")


def _transform_with_scaler(matrix: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return matrix
    return scaler.transform(matrix)


def _predict_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix), dtype=float)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1] - 0.5
    predictions = np.asarray(model.predict(matrix), dtype=float)
    return predictions - 0.5


def _group_train_val_split(train_frame: pd.DataFrame, *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    unique_examples = train_frame.loc[:, ["example_id", "pair_id", "label_ambiguous"]].drop_duplicates().reset_index(drop=True)
    if unique_examples["pair_id"].nunique() < 2:
        example_ids = set(unique_examples["example_id"].astype(str))
        return example_ids, set()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(
            unique_examples["example_id"],
            unique_examples["label_ambiguous"],
            groups=unique_examples["pair_id"],
        )
    )
    train_ids = set(unique_examples.iloc[train_idx]["example_id"].astype(str))
    val_ids = set(unique_examples.iloc[val_idx]["example_id"].astype(str))
    return train_ids, val_ids


def _selection_order(candidate_df: pd.DataFrame) -> pd.DataFrame:
    return candidate_df.sort_values(
        ["val_auroc", "val_f1", "val_accuracy", "layer"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _select_multilayer_candidates(dataset_candidates: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    ranked = _selection_order(dataset_candidates)
    selected: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for row in ranked.to_dict(orient="records"):
        layer = int(row["layer"])
        if layer in seen_layers:
            continue
        selected.append(row)
        seen_layers.add(layer)
        if len(selected) >= top_k:
            break
    return selected


def _layer_suffix(layer: int) -> str:
    return f"l{int(layer):02d}"


def _summary_ops() -> list[str]:
    return ["mean", "std", "min", "max", "delta", "slope"]


def _stacked_summary_features(
    merged_frame: pd.DataFrame,
    *,
    metric_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    summary_columns_by_group = {group: [] for group in metric_groups}
    summary_data: dict[str, np.ndarray] = {}
    for group_name, columns in metric_groups.items():
        if not columns:
            continue
        values = merged_frame.loc[:, columns].to_numpy(dtype=float)
        if values.ndim != 2:
            continue
        ops = {
            "mean": np.mean(values, axis=1),
            "std": np.std(values, axis=1),
            "min": np.min(values, axis=1),
            "max": np.max(values, axis=1),
            "delta": values[:, -1] - values[:, 0],
        }
        if values.shape[1] == 1:
            ops["slope"] = np.zeros(len(values), dtype=float)
        else:
            x_axis = np.arange(values.shape[1], dtype=float)
            x_centered = x_axis - x_axis.mean()
            denom = float(np.sum(x_centered**2))
            centered = values - values.mean(axis=1, keepdims=True)
            ops["slope"] = centered.dot(x_centered) / max(denom, 1e-12)
        for op_name in _summary_ops():
            column_name = f"{group_name}__stack_{op_name}"
            summary_data[column_name] = np.asarray(ops[op_name], dtype=float)
            summary_columns_by_group[group_name].append(column_name)
    return pd.DataFrame(summary_data), summary_columns_by_group


def _available_layers(hidden_root: Path, dataset: str, readout: str) -> list[int]:
    pattern = f"{dataset}__layer_*__{readout}.parquet"
    layers: list[int] = []
    for path in sorted(hidden_root.glob(pattern)):
        stem = path.stem
        try:
            layer = int(stem.split("__")[1].replace("layer_", ""))
        except Exception:
            continue
        layers.append(layer)
    if not layers:
        raise FileNotFoundError(f"No hidden-state caches found for dataset={dataset}, readout={readout} in {hidden_root}")
    return sorted(set(layers))


def _resolve_candidate_layers(available_layers: list[int], config: dict[str, Any]) -> list[int]:
    configured = config.get("candidate_layers", "auto")
    if isinstance(configured, list):
        selected = [int(layer) for layer in configured if int(layer) in available_layers]
        if selected:
            return sorted(set(selected))
    strategy = str(config.get("layer_selection_strategy", "evenly_spaced"))
    if configured == "all" or strategy == "all":
        return available_layers
    max_layers = int(config.get("max_candidate_layers", len(available_layers)))
    if max_layers >= len(available_layers):
        return available_layers
    if strategy == "evenly_spaced":
        indices = np.linspace(0, len(available_layers) - 1, num=max_layers, dtype=int)
        return [available_layers[index] for index in sorted(set(indices.tolist()))]
    raise ValueError(f"Unsupported independent layer selection strategy: {strategy}")


def _load_layer_cache(hidden_root: Path, dataset: str, layer: int, readout: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = hidden_root / f"{dataset}__layer_{layer:02d}__{readout}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Hidden-state cache missing: {path}")
    meta, matrix = load_hidden_state_table(path)
    return meta.copy(), np.asarray(matrix, dtype=np.float32)


def _fit_reducer(
    train_matrix: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[StandardScaler | None, PCA, StandardScaler | None, int]:
    input_scaler = None
    matrix = train_matrix
    if bool(config.get("standardize_hidden_states", True)):
        input_scaler = StandardScaler()
        matrix = input_scaler.fit_transform(train_matrix)
    max_components = min(train_matrix.shape[0] - 1, train_matrix.shape[1])
    requested = int(config.get("pca_components", 8))
    n_components = max(1, min(requested, max_components))
    reducer = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
        whiten=bool(config.get("pca_whiten", False)),
    )
    reduced = reducer.fit_transform(matrix)
    reduced_scaler = None
    if bool(config.get("standardize_reduced_coordinates", True)):
        reduced_scaler = StandardScaler()
        reduced_scaler.fit(reduced)
    return input_scaler, reducer, reduced_scaler, n_components


def _transform_reducer(
    matrix: np.ndarray,
    *,
    input_scaler: StandardScaler | None,
    reducer: PCA,
    reduced_scaler: StandardScaler | None,
) -> np.ndarray:
    transformed = matrix
    if input_scaler is not None:
        transformed = input_scaler.transform(transformed)
    transformed = reducer.transform(transformed)
    if reduced_scaler is not None:
        transformed = reduced_scaler.transform(transformed)
    return np.asarray(transformed, dtype=np.float32)


def _pc_column_names(count: int) -> list[str]:
    return [f"pc_{index:02d}" for index in range(count)]


def _build_unsupervised_frame(
    meta: pd.DataFrame,
    coords: np.ndarray,
    *,
    layer: int,
    readout: str,
    geometry_dim: int,
) -> pd.DataFrame:
    geometry_dim = min(int(geometry_dim), coords.shape[1])
    frame = meta.loc[:, [column for column in meta.columns if column != "vector"]].copy().reset_index(drop=True)
    frame["layer"] = int(layer)
    frame["readout"] = str(readout)
    for index, column_name in enumerate(_pc_column_names(geometry_dim)):
        frame[column_name] = coords[:, index].astype(float)
    frame["reduced_norm"] = np.linalg.norm(coords[:, :geometry_dim], axis=1).astype(float)
    return frame


def _local_feature_row(
    *,
    row: pd.Series,
    query_topology: np.ndarray,
    query_geometry: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    reference_topology: np.ndarray,
    reference_geometry: np.ndarray,
    prototypes: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    local_points = np.vstack([query_topology, reference_topology[neighbor_indices]]) if len(neighbor_indices) else query_topology[None, :]
    topology_metric = str(config.get("topology_metric", "euclidean"))
    diagrams = _compute_diagrams(
        local_points,
        maxdim=int(config.get("maxdim", 1)),
        coeff=int(config.get("coeff", 2)),
        distance_metric=topology_metric,
    )
    feature_row: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "layer": int(row["layer"]),
        "readout": str(row["readout"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "knn_distance_mean": float(np.mean(neighbor_distances)) if len(neighbor_distances) else 0.0,
        "knn_distance_std": float(np.std(neighbor_distances)) if len(neighbor_distances) else 0.0,
        "knn_distance_max": float(np.max(neighbor_distances)) if len(neighbor_distances) else 0.0,
    }
    geometry_columns = [column for column in row.index if column.startswith(GEOMETRY_PREFIXES)]
    for column in geometry_columns:
        feature_row[column] = float(row[column])
    feature_row["reduced_norm"] = float(row.get("reduced_norm", np.linalg.norm(query_geometry)))
    if len(neighbor_indices):
        neighbor_geometry = reference_geometry[neighbor_indices]
        centroid = neighbor_geometry.mean(axis=0)
        feature_row["local_centroid_distance"] = float(np.linalg.norm(query_geometry - centroid))
        spreads = np.linalg.norm(neighbor_geometry - centroid, axis=1)
        feature_row["local_spread_mean"] = float(np.mean(spreads))
        feature_row["local_spread_max"] = float(np.max(spreads))
    else:
        feature_row["local_centroid_distance"] = 0.0
        feature_row["local_spread_mean"] = 0.0
        feature_row["local_spread_max"] = 0.0
    grid_size = int(config.get("betti_grid_size", 32))
    for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
        diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
        feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=grid_size))
        feature_row.update(
            _persistence_image_features(
                diagram,
                prefix=prefix,
                grid_side=int(config.get("persistence_image_grid_side", 4)),
            )
        )
        feature_row[f"{prefix}_wasserstein_to_clear"] = _safe_wasserstein(diagram, prototypes["clear"][homology_dim])
        feature_row[f"{prefix}_wasserstein_to_ambiguous"] = _safe_wasserstein(
            diagram,
            prototypes["ambiguous"][homology_dim],
        )
        feature_row[f"{prefix}_bottleneck_to_clear"] = _safe_bottleneck(diagram, prototypes["clear"][homology_dim])
        feature_row[f"{prefix}_bottleneck_to_ambiguous"] = _safe_bottleneck(
            diagram,
            prototypes["ambiguous"][homology_dim],
        )
    return feature_row


def build_local_independent_features(
    *,
    query_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    topology_columns: list[str],
    geometry_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    if query_frame.empty:
        return pd.DataFrame()
    if reference_frame.empty:
        raise ValueError("Reference frame for independent local features is empty.")

    reference_topology = reference_frame.loc[:, topology_columns].to_numpy(dtype=float)
    query_topology = query_frame.loc[:, topology_columns].to_numpy(dtype=float)
    reference_geometry = reference_frame.loc[:, geometry_columns].to_numpy(dtype=float)
    query_geometry = query_frame.loc[:, geometry_columns].to_numpy(dtype=float)
    reference_labels = reference_frame["label_ambiguous"].to_numpy(dtype=int)
    topology_metric = str(config.get("topology_metric", "euclidean"))
    prototypes = _prototype_diagrams(
        reference_topology,
        reference_labels,
        sample_n=int(config.get("prototype_sample_n", 96)),
        maxdim=int(config.get("maxdim", 1)),
        coeff=int(config.get("coeff", 2)),
        seed=int(config.get("_prototype_seed", 0)),
        distance_metric=topology_metric,
    )

    neighborhood_k = int(config.get("neighborhood_k", 24))
    search_k = min(len(reference_frame), neighborhood_k + 1)
    model = NearestNeighbors(n_neighbors=max(search_k, 1), metric=topology_metric)
    model.fit(reference_topology)
    distances, indices = model.kneighbors(query_topology, return_distance=True)
    reference_example_ids = reference_frame["example_id"].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for query_idx, row in enumerate(query_frame.to_dict(orient="records")):
        neighbor_indices = indices[query_idx]
        neighbor_distances = distances[query_idx]
        if str(row["split"]) == "train":
            mask = reference_example_ids[neighbor_indices] != str(row["example_id"])
            neighbor_indices = neighbor_indices[mask]
            neighbor_distances = neighbor_distances[mask]
        neighbor_indices = neighbor_indices[:neighborhood_k]
        neighbor_distances = neighbor_distances[:neighborhood_k]
        rows.append(
            _local_feature_row(
                row=pd.Series(row),
                query_topology=query_topology[query_idx],
                query_geometry=query_geometry[query_idx],
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                reference_topology=reference_topology,
                reference_geometry=reference_geometry,
                prototypes=prototypes,
                config=config,
            )
        )
    return pd.DataFrame(rows)


def _feature_columns(feature_frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    topology_columns = [column for column in feature_frame.columns if column.startswith(TOPOLOGY_PREFIXES)]
    geometry_columns = [
        column
        for column in feature_frame.columns
        if column.startswith(GEOMETRY_PREFIXES) or column in GEOMETRY_EXTRA_COLUMNS
    ]
    hybrid_columns = topology_columns + geometry_columns
    return topology_columns, geometry_columns, hybrid_columns


def _prepare_layer_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    layer: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    topology_columns, geometry_columns, _ = _feature_columns(feature_frame)
    suffix = _layer_suffix(layer)
    renamed = feature_frame.loc[:, BASE_KEY_COLUMNS + topology_columns + geometry_columns].copy()
    rename_map = {column: f"{column}__{suffix}" for column in topology_columns + geometry_columns}
    renamed = renamed.rename(columns=rename_map)
    return renamed, [rename_map[column] for column in topology_columns], [rename_map[column] for column in geometry_columns]


def _evaluate_feature_set(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    *,
    feature_columns: list[str],
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train = train_features.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_features["label_ambiguous"].to_numpy(dtype=int)
    x_eval = eval_features.loc[:, feature_columns].to_numpy(dtype=float)
    y_eval = eval_features["label_ambiguous"].to_numpy(dtype=int)
    clf, scaler = _fit_classifier(x_train, y_train, config=classifier_config, seed=seed)
    train_scores = _predict_scores(clf, _transform_with_scaler(x_train, scaler))
    eval_scores = _predict_scores(clf, _transform_with_scaler(x_eval, scaler))
    coefficients, intercept = _extract_model_signal(clf)
    payload = {
        "classifier": clf,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "eval_metrics": binary_classification_metrics(y_eval, eval_scores),
        "coefficients": coefficients,
        "intercept": intercept,
    }
    return payload["train_metrics"], payload


def _build_layer_feature_tables(
    *,
    meta: pd.DataFrame,
    matrix: np.ndarray,
    train_ids: set[str],
    eval_ids: set[str],
    layer: int,
    readout: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = meta["example_id"].astype(str).isin(train_ids)
    eval_mask = meta["example_id"].astype(str).isin(eval_ids)
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    eval_meta = meta.loc[eval_mask].reset_index(drop=True)
    train_matrix = matrix[train_mask.to_numpy()]
    eval_matrix = matrix[eval_mask.to_numpy()]
    if train_meta.empty or eval_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    input_scaler, reducer, reduced_scaler, n_components = _fit_reducer(train_matrix, config=config, seed=seed + layer)
    train_coords = _transform_reducer(
        train_matrix,
        input_scaler=input_scaler,
        reducer=reducer,
        reduced_scaler=reduced_scaler,
    )
    eval_coords = _transform_reducer(
        eval_matrix,
        input_scaler=input_scaler,
        reducer=reducer,
        reduced_scaler=reduced_scaler,
    )
    geometry_dim = min(int(config.get("geometry_components", n_components)), n_components)
    topology_dim = min(int(config.get("topology_components", geometry_dim)), geometry_dim)
    train_frame = _build_unsupervised_frame(train_meta, train_coords, layer=layer, readout=readout, geometry_dim=geometry_dim)
    eval_frame = _build_unsupervised_frame(eval_meta, eval_coords, layer=layer, readout=readout, geometry_dim=geometry_dim)
    geometry_columns = _pc_column_names(geometry_dim)
    topology_columns = geometry_columns[:topology_dim]
    train_features = build_local_independent_features(
        query_frame=train_frame,
        reference_frame=train_frame,
        topology_columns=topology_columns,
        geometry_columns=geometry_columns,
        config={**config, "_prototype_seed": seed + layer + 17},
    )
    eval_features = build_local_independent_features(
        query_frame=eval_frame,
        reference_frame=train_frame,
        topology_columns=topology_columns,
        geometry_columns=geometry_columns,
        config={**config, "_prototype_seed": seed + layer + 17},
    )
    return train_features, eval_features


def _candidate_rows_for_layer(
    *,
    hidden_root: Path,
    dataset: str,
    layer: int,
    readout: str,
    inner_train_ids: set[str],
    val_ids: set[str],
    classifier_config: dict[str, Any],
    classifier_section: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    meta, matrix = _load_layer_cache(hidden_root, dataset=dataset, layer=layer, readout=readout)
    local_train, local_val = _build_layer_feature_tables(
        meta=meta,
        matrix=matrix,
        train_ids=inner_train_ids,
        eval_ids=val_ids,
        layer=layer,
        readout=readout,
        config=classifier_config,
        seed=seed,
    )
    if local_train.empty or local_val.empty:
        return []
    topology_columns, geometry_columns, hybrid_columns = _feature_columns(local_train)
    feature_sets = {
        "topology_only": topology_columns,
        "geometry_only": geometry_columns,
        "hybrid": hybrid_columns,
    }
    rows: list[dict[str, Any]] = []
    for feature_set, columns in feature_sets.items():
        if not columns:
            continue
        _, payload = _evaluate_feature_set(
            train_features=local_train,
            eval_features=local_val,
            feature_columns=columns,
            classifier_config=classifier_section,
            seed=seed,
        )
        metrics = payload["eval_metrics"]
        rows.append(
            {
                "dataset": dataset,
                "layer": int(layer),
                "selection_mode": "single_layer",
                "feature_set": feature_set,
                "val_auroc": float(metrics["auroc"]),
                "val_accuracy": float(metrics["accuracy"]),
                "val_f1": float(metrics["f1"]),
                "train_n": int(len(local_train)),
                "val_n": int(len(local_val)),
                "feature_count": int(len(columns)),
            }
        )
    return rows


def _build_multilayer_feature_frames(
    *,
    hidden_root: Path,
    dataset: str,
    readout: str,
    train_ids: set[str],
    test_ids: set[str],
    selections: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not selections:
        raise ValueError("Multilayer feature construction requires at least one selected layer.")
    train_merged: pd.DataFrame | None = None
    test_merged: pd.DataFrame | None = None
    topology_columns: list[str] = []
    geometry_columns: list[str] = []
    selection_specs: list[dict[str, Any]] = []

    for rank, selection in enumerate(selections, start=1):
        layer = int(selection["layer"])
        selection_specs.append({"rank": rank, "layer": layer, "val_auroc": float(selection["val_auroc"])})
        meta, matrix = _load_layer_cache(hidden_root, dataset=dataset, layer=layer, readout=readout)
        train_features, test_features = _build_layer_feature_tables(
            meta=meta,
            matrix=matrix,
            train_ids=train_ids,
            eval_ids=test_ids,
            layer=layer,
            readout=readout,
            config=config,
            seed=seed + 401 + rank,
        )
        if train_features.empty or test_features.empty:
            continue
        train_prepared, train_topology, train_geometry = _prepare_layer_feature_frame(train_features, layer=layer)
        test_prepared, test_topology, test_geometry = _prepare_layer_feature_frame(test_features, layer=layer)
        topology_columns.extend(train_topology)
        geometry_columns.extend(train_geometry)
        if train_merged is None:
            train_merged = train_prepared
            test_merged = test_prepared
        else:
            train_merged = train_merged.merge(train_prepared, on=BASE_KEY_COLUMNS, how="inner")
            test_merged = test_merged.merge(test_prepared, on=BASE_KEY_COLUMNS, how="inner")

    if train_merged is None or test_merged is None or train_merged.empty or test_merged.empty:
        raise ValueError("Failed to build non-empty multilayer independent features.")

    topology_columns = [column for column in topology_columns if column in train_merged.columns]
    geometry_columns = [column for column in geometry_columns if column in train_merged.columns]
    train_summary, train_groups = _stacked_summary_features(
        train_merged,
        metric_groups={"topology": topology_columns, "geometry": geometry_columns},
    )
    test_summary, _ = _stacked_summary_features(
        test_merged,
        metric_groups={"topology": topology_columns, "geometry": geometry_columns},
    )
    train_multilayer = pd.concat([train_merged.reset_index(drop=True), train_summary], axis=1)
    test_multilayer = pd.concat([test_merged.reset_index(drop=True), test_summary], axis=1)
    return train_multilayer, test_multilayer, {
        "selections": selection_specs,
        "topology_columns": topology_columns,
        "geometry_columns": geometry_columns,
        "topology_summary_columns": train_groups["topology"],
        "geometry_summary_columns": train_groups["geometry"],
    }


def _plot_candidate_heatmap(candidate_df: pd.DataFrame, *, dataset: str, feature_set: str, output_path: Path) -> None:
    subset = candidate_df.loc[
        candidate_df["dataset"].eq(dataset) & candidate_df["feature_set"].eq(feature_set)
    ]
    if subset.empty:
        return
    layers = sorted(subset["layer"].unique())
    values = np.asarray([subset.loc[subset["layer"].eq(layer), "val_auroc"].max() for layer in layers], dtype=float)[None, :]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(layers)), 2.6))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_title(f"{dataset}: {feature_set} validation AUROC")
    ax.set_xlabel("Layer")
    ax.set_yticks([0])
    ax.set_yticklabels([feature_set])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(int(layer)) for layer in layers], rotation=90, fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Validation AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_metric_bars(final_df: pd.DataFrame, *, dataset: str, output_path: Path) -> None:
    subset = final_df.loc[final_df["dataset"].eq(dataset)].copy()
    if subset.empty:
        return
    feature_sets = [feature_set for feature_set in SINGLE_FEATURE_SETS + MULTILAYER_FEATURE_SETS if feature_set in set(subset["feature_set"])]
    metrics = ["test_auroc", "test_accuracy", "test_f1"]
    x = np.arange(len(metrics))
    width = min(0.13, 0.75 / max(len(feature_sets), 1))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    center = (len(feature_sets) - 1) / 2.0
    for offset, feature_set in enumerate(feature_sets):
        row = subset.loc[subset["feature_set"].eq(feature_set)].iloc[0]
        values = [float(row[metric]) for metric in metrics]
        ax.bar(
            x + (offset - center) * width,
            values,
            width=width,
            label=feature_set,
            color=LABEL_COLORS.get(feature_set, "#777777"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("test_", "").upper() for metric in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset}: independent classifier comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=min(3, len(feature_sets)), frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(top=0.72)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_report(*, model_name: str, selected_df: pd.DataFrame, final_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Independent Topology Classifier Summary",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset in sorted(final_df["dataset"].unique()):
        lines.extend([f"## {dataset}", ""])
        for feature_set in SINGLE_FEATURE_SETS:
            rows = selected_df.loc[
                selected_df["dataset"].eq(dataset)
                & selected_df["selection_mode"].eq("single_layer")
                & selected_df["feature_set"].eq(feature_set)
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            lines.append(
                f"- `{feature_set}` single-layer selection: layer `{int(row['layer'])}` "
                f"(val AUROC `{row['val_auroc']:.4f}`)."
            )
        lines.extend(["", "### Multi-layer Selection", ""])
        for feature_set in MULTILAYER_FEATURE_SETS:
            rows = selected_df.loc[
                selected_df["dataset"].eq(dataset)
                & selected_df["selection_mode"].eq("multilayer_component")
                & selected_df["feature_set"].eq(feature_set)
            ].sort_values("component_rank")
            if rows.empty:
                continue
            lines.append(f"- `{feature_set}` components:")
            for row in rows.to_dict(orient="records"):
                lines.append(
                    f"  - rank `{int(row['component_rank'])}`: layer `{int(row['layer'])}` "
                    f"(val AUROC `{row['val_auroc']:.4f}`)."
                )
        lines.extend(["", "### Final Test Metrics", ""])
        dataset_final = final_df.loc[final_df["dataset"].eq(dataset)]
        for row in dataset_final.to_dict(orient="records"):
            lines.append(
                f"- `{row['feature_set']}`: AUROC `{row['test_auroc']:.4f}`, accuracy `{row['test_accuracy']:.4f}`, "
                f"F1 `{row['test_f1']:.4f}`."
            )
        lines.append("")
    write_markdown(output_path, "\n".join(lines) + "\n")


def run_independent_topology_classifier_analysis(
    *,
    model_name: str,
    hidden_state_root: str | Path,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    """Train independent ambiguity classifiers from raw cached hidden states."""

    hidden_root = Path(hidden_state_root)
    readout = str(classifier_config.get("readout", "mean_pool"))
    datasets = list(classifier_config.get("datasets", ["ambigqa", "situatedqa"]))
    output_root = ensure_dir(Path(classifier_config["output_dir"]) / slugify(model_name))
    plots_root = ensure_dir(output_root / "plots")
    models_root = ensure_dir(output_root / "models")
    parallel_jobs = max(1, int(classifier_config.get("parallel_jobs", 1)))

    classifier_section = dict(classifier_config.get("classifier", {}))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_feature_tables: list[pd.DataFrame] = []

    for dataset in datasets:
        available_layers = _available_layers(hidden_root, dataset=dataset, readout=readout)
        candidate_layers = _resolve_candidate_layers(available_layers, classifier_config)
        split_meta, _ = _load_layer_cache(hidden_root, dataset=dataset, layer=candidate_layers[0], readout=readout)
        train_meta = split_meta.loc[split_meta["split"].eq("train")].copy()
        test_ids = set(split_meta.loc[split_meta["split"].eq("test"), "example_id"].astype(str))
        if train_meta.empty or not test_ids:
            LOGGER.warning("Skipping dataset %s because train/test rows are missing.", dataset)
            continue
        inner_train_ids, val_ids = _group_train_val_split(
            train_meta,
            val_fraction=float(classifier_config.get("val_fraction", 0.2)),
            seed=seed,
        )
        if not val_ids:
            raise ValueError(f"Failed to allocate an inner validation set for dataset {dataset}.")

        candidate_batches = joblib.Parallel(n_jobs=parallel_jobs, backend="loky")(
            joblib.delayed(_candidate_rows_for_layer)(
                hidden_root=hidden_root,
                dataset=dataset,
                layer=int(layer),
                readout=readout,
                inner_train_ids=inner_train_ids,
                val_ids=val_ids,
                classifier_config=classifier_config,
                classifier_section=classifier_section,
                seed=seed,
            )
            for layer in candidate_layers
        )
        for batch in candidate_batches:
            candidate_rows.extend(batch)

        candidate_df = pd.DataFrame(candidate_rows)
        for feature_set in SINGLE_FEATURE_SETS:
            dataset_candidates = candidate_df.loc[
                candidate_df["dataset"].eq(dataset) & candidate_df["feature_set"].eq(feature_set)
            ].copy()
            if dataset_candidates.empty:
                continue
            best_row = _selection_order(dataset_candidates).iloc[0]
            selected_rows.append({**best_row.to_dict(), "selection_mode": "single_layer", "component_rank": 1})

            meta, matrix = _load_layer_cache(hidden_root, dataset=dataset, layer=int(best_row["layer"]), readout=readout)
            train_ids = set(meta.loc[meta["split"].eq("train"), "example_id"].astype(str))
            local_train, local_test = _build_layer_feature_tables(
                meta=meta,
                matrix=matrix,
                train_ids=train_ids,
                eval_ids=test_ids,
                layer=int(best_row["layer"]),
                readout=readout,
                config=classifier_config,
                seed=seed + 101,
            )
            local_train["feature_role"] = "train"
            local_test["feature_role"] = "test"
            local_train["selected_for_dataset"] = dataset
            local_test["selected_for_dataset"] = dataset
            local_train["feature_variant"] = "single_layer"
            local_test["feature_variant"] = "single_layer"
            local_train["selected_for_feature_set"] = feature_set
            local_test["selected_for_feature_set"] = feature_set
            selected_feature_tables.extend([local_train, local_test])

            topology_columns, geometry_columns, hybrid_columns = _feature_columns(local_train)
            feature_map = {
                "topology_only": topology_columns,
                "geometry_only": geometry_columns,
                "hybrid": hybrid_columns,
            }
            columns = feature_map[feature_set]
            _, payload = _evaluate_feature_set(
                train_features=local_train,
                eval_features=local_test,
                feature_columns=columns,
                classifier_config=classifier_section,
                seed=seed,
            )
            metrics = payload["eval_metrics"]
            final_rows.append(
                {
                    "dataset": dataset,
                    "selection_mode": "single_layer",
                    "layer": int(best_row["layer"]),
                    "selection_signature": str(int(best_row["layer"])),
                    "selection_size": 1,
                    "feature_set": feature_set,
                    "test_auroc": float(metrics["auroc"]),
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_f1": float(metrics["f1"]),
                    "feature_count": int(len(columns)),
                }
            )
            model_path = models_root / f"{dataset}__{feature_set}.joblib"
            joblib.dump(
                {
                    "classifier": payload["classifier"],
                    "scaler": payload["scaler"],
                    "feature_columns": columns,
                    "dataset": dataset,
                    "selection_mode": "single_layer",
                    "layer": int(best_row["layer"]),
                    "train_metrics": payload["train_metrics"],
                    "test_metrics": payload["eval_metrics"],
                },
                model_path,
            )

        multilayer_enabled = bool(classifier_config.get("multilayer_enabled", True))
        if multilayer_enabled:
            multilayer_map = {
                "topology_multilayer": "topology_only",
                "geometry_multilayer": "geometry_only",
                "hybrid_multilayer": "hybrid",
            }
            train_ids = set(split_meta.loc[split_meta["split"].eq("train"), "example_id"].astype(str))
            for multi_feature_set, source_feature_set in multilayer_map.items():
                dataset_candidates = candidate_df.loc[
                    candidate_df["dataset"].eq(dataset) & candidate_df["feature_set"].eq(source_feature_set)
                ].copy()
                if dataset_candidates.empty:
                    continue
                selections = _select_multilayer_candidates(
                    dataset_candidates,
                    top_k=int(classifier_config.get("multilayer_top_k", 3)),
                )
                for rank, row in enumerate(selections, start=1):
                    selected_rows.append(
                        {
                            **row,
                            "feature_set": multi_feature_set,
                            "selection_mode": "multilayer_component",
                            "component_rank": rank,
                        }
                    )
                multi_train, multi_test, multi_meta = _build_multilayer_feature_frames(
                    hidden_root=hidden_root,
                    dataset=dataset,
                    readout=readout,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    selections=selections,
                    config=classifier_config,
                    seed=seed,
                )
                multi_train["feature_role"] = "train"
                multi_test["feature_role"] = "test"
                multi_train["selected_for_dataset"] = dataset
                multi_test["selected_for_dataset"] = dataset
                multi_train["feature_variant"] = "multilayer"
                multi_test["feature_variant"] = "multilayer"
                multi_train["selected_for_feature_set"] = multi_feature_set
                multi_test["selected_for_feature_set"] = multi_feature_set
                selected_feature_tables.extend([multi_train, multi_test])

                multi_topology_columns = multi_meta["topology_columns"] + multi_meta["topology_summary_columns"]
                multi_geometry_columns = multi_meta["geometry_columns"] + multi_meta["geometry_summary_columns"]
                feature_map = {
                    "topology_multilayer": multi_topology_columns,
                    "geometry_multilayer": multi_geometry_columns,
                    "hybrid_multilayer": multi_topology_columns + multi_geometry_columns,
                }
                columns = feature_map[multi_feature_set]
                _, payload = _evaluate_feature_set(
                    train_features=multi_train,
                    eval_features=multi_test,
                    feature_columns=columns,
                    classifier_config=classifier_section,
                    seed=seed,
                )
                metrics = payload["eval_metrics"]
                selection_signature = " | ".join(str(int(item["layer"])) for item in multi_meta["selections"])
                final_rows.append(
                    {
                        "dataset": dataset,
                        "selection_mode": "multilayer",
                        "layer": -1,
                        "selection_signature": selection_signature,
                        "selection_size": int(len(multi_meta["selections"])),
                        "feature_set": multi_feature_set,
                        "test_auroc": float(metrics["auroc"]),
                        "test_accuracy": float(metrics["accuracy"]),
                        "test_f1": float(metrics["f1"]),
                        "feature_count": int(len(columns)),
                    }
                )
                model_path = models_root / f"{dataset}__{multi_feature_set}.joblib"
                joblib.dump(
                    {
                        "classifier": payload["classifier"],
                        "scaler": payload["scaler"],
                        "feature_columns": columns,
                        "dataset": dataset,
                        "selection_mode": "multilayer",
                        "selections": multi_meta["selections"],
                        "train_metrics": payload["train_metrics"],
                        "test_metrics": payload["eval_metrics"],
                    },
                    model_path,
                )

        for feature_set in SINGLE_FEATURE_SETS:
            _plot_candidate_heatmap(
                candidate_df,
                dataset=dataset,
                feature_set=feature_set,
                output_path=plots_root / f"{dataset}__{feature_set}__candidate_heatmap.png",
            )
        _plot_final_metric_bars(pd.DataFrame(final_rows), dataset=dataset, output_path=plots_root / f"{dataset}__final_classifier_metrics.png")

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["dataset", "feature_set", "layer"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["dataset", "feature_set"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["dataset", "feature_set", "selection_mode", "component_rank"]).reset_index(drop=True)
    feature_table = pd.concat(selected_feature_tables, ignore_index=True, sort=False) if selected_feature_tables else pd.DataFrame()

    candidate_path = output_root / str(classifier_config["candidate_metrics_filename"])
    final_path = output_root / str(classifier_config["final_metrics_filename"])
    selected_path = output_root / str(classifier_config["selected_candidates_filename"])
    feature_table_path = output_root / str(classifier_config["feature_table_filename"])
    report_path = output_root / str(classifier_config["report_filename"])
    metadata_path = output_root / str(classifier_config["metadata_filename"])

    write_parquet(candidate_df, candidate_path)
    write_parquet(final_df, final_path)
    write_parquet(selected_df, selected_path)
    if not feature_table.empty:
        write_parquet(feature_table, feature_table_path)
    _render_report(model_name=model_name, selected_df=selected_df, final_df=final_df, output_path=report_path)
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "hidden_state_root": str(hidden_root),
            "created_at": utc_now_iso(),
            "datasets": datasets,
            "readout": readout,
            "candidate_layers_setting": classifier_config.get("candidate_layers", "auto"),
            "layer_selection_strategy": classifier_config.get("layer_selection_strategy", "evenly_spaced"),
            "output_artifacts": {
                "candidate_metrics": str(candidate_path),
                "final_metrics": str(final_path),
                "selected_candidates": str(selected_path),
                "feature_table": str(feature_table_path),
                "report": str(report_path),
            },
        },
    )
    return {
        "candidate_metrics_path": str(candidate_path),
        "final_metrics_path": str(final_path),
        "selected_candidates_path": str(selected_path),
        "feature_table_path": str(feature_table_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }
