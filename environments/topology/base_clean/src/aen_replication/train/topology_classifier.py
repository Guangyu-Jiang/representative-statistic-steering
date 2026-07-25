"""Per-question topology-based ambiguity classifier."""

from __future__ import annotations

import logging
from math import ceil
from pathlib import Path
from typing import Any
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import wasserstein
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {0: "clear", 1: "ambiguous"}
TOPOLOGY_PREFIXES = ("h0_", "h1_")
DEFAULT_GEOMETRY_COLUMNS = ["signed_distance", "decision_value", "z_0", "z_1"]
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


def _normalized_entropy(lifetimes: np.ndarray) -> float:
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if lifetimes.size <= 1:
        return 0.0
    weights = lifetimes / lifetimes.sum()
    entropy = float(-(weights * np.log(weights + 1e-12)).sum())
    return float(entropy / np.log(len(weights)))


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


def _safe_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_diagram(left)
    right = _finite_diagram(right)
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(wasserstein(left, right, matching=False))


def _diagram_descriptors(diagram: np.ndarray, *, prefix: str, grid_size: int) -> dict[str, float]:
    finite = _finite_diagram(diagram)
    if finite.size == 0:
        return {
            f"{prefix}_feature_count": 0.0,
            f"{prefix}_total_persistence_norm": 0.0,
            f"{prefix}_max_persistence_norm": 0.0,
            f"{prefix}_mean_persistence": 0.0,
            f"{prefix}_persistence_entropy": 0.0,
            f"{prefix}_betti_curve_auc_norm": 0.0,
        }
    lifetimes = finite[:, 1] - finite[:, 0]
    max_death = float(max(np.max(finite[:, 1]), 1e-6))
    grid = np.linspace(0.0, max_death, grid_size)
    betti = _betti_curve(finite, grid)
    betti_auc = float(getattr(np, "trapezoid", np.trapz)(betti, grid))
    return {
        f"{prefix}_feature_count": float(len(lifetimes)),
        f"{prefix}_total_persistence_norm": float(lifetimes.sum() / max_death),
        f"{prefix}_max_persistence_norm": float(lifetimes.max() / max_death),
        f"{prefix}_mean_persistence": float(lifetimes.mean()),
        f"{prefix}_persistence_entropy": _normalized_entropy(lifetimes),
        f"{prefix}_betti_curve_auc_norm": float(betti_auc / max_death),
    }


def _compute_diagrams(points: np.ndarray, *, maxdim: int, coeff: int) -> list[np.ndarray]:
    if len(points) <= 1:
        return [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
    distance_matrix = squareform(pdist(points, metric="euclidean"))
    result = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim, coeff=coeff)
    diagrams = result.get("dgms", [])
    if len(diagrams) < maxdim + 1:
        diagrams = diagrams + [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1 - len(diagrams))]
    return [np.asarray(diagram, dtype=float) for diagram in diagrams[: maxdim + 1]]


def _coordinate_columns(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    available = [column for column in requested if column in frame.columns]
    if not available:
        raise ValueError(f"None of the requested coordinate columns are present: {requested}")
    return available


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


def _resolve_candidate_layers(
    *,
    dataset_frame: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
) -> list[int]:
    configured = config.get("candidate_layers", "auto")
    available_layers = sorted(int(layer) for layer in dataset_frame["layer"].unique())
    if isinstance(configured, list):
        selected = [int(layer) for layer in configured if int(layer) in available_layers]
        if selected:
            return selected
    strategy = str(config.get("layer_selection_strategy", "ph_distance_peak"))
    if strategy != "ph_distance_peak":
        return available_layers

    ph_dir = Path(config["ph_distance_dir"])
    ph_path = ph_dir / _slugify(model_name) / str(config["ph_distance_filename"])
    if not ph_path.exists():
        LOGGER.warning("PH distance artifact not found at %s; falling back to all layers.", ph_path)
        return available_layers

    distance_df = pd.read_parquet(ph_path)
    compare_subspaces = set(config.get("ph_distance_subspaces", [])) or set(dataset_frame["subspace_name"].unique())
    subset = distance_df.loc[
        distance_df["dataset"].eq(dataset_name)
        & distance_df["homology_dim"].eq(int(config.get("ph_distance_homology_dim", 1)))
        & distance_df["subspace_name"].isin(compare_subspaces)
    ]
    if subset.empty:
        return available_layers
    score_column = str(config.get("ph_distance_metric", "wasserstein_distance"))
    ranked = (
        subset.groupby("layer", dropna=False)[score_column]
        .max()
        .sort_values(ascending=False)
        .head(int(config.get("max_candidate_layers", 6)))
    )
    selected_layers = sorted(int(layer) for layer in ranked.index if int(layer) in available_layers)
    return selected_layers or available_layers


def _slugify(text: str) -> str:
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char.lower())
        else:
            allowed.append("_")
    slug = "".join(allowed)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _prototype_diagrams(
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    sample_n: int,
    maxdim: int,
    coeff: int,
    seed: int,
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
        prototypes[label_name] = _compute_diagrams(reference_points[indices], maxdim=maxdim, coeff=coeff)
    return prototypes


def _local_feature_row(
    *,
    row: pd.Series,
    query_point: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    reference_points: np.ndarray,
    prototypes: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    local_points = np.vstack([query_point, reference_points[neighbor_indices]]) if len(neighbor_indices) else query_point[None, :]
    diagrams = _compute_diagrams(
        local_points,
        maxdim=int(config.get("maxdim", 1)),
        coeff=int(config.get("coeff", 2)),
    )
    feature_row: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "layer": int(row["layer"]),
        "subspace_name": str(row["subspace_name"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "knn_distance_mean": float(np.mean(neighbor_distances)) if len(neighbor_distances) else 0.0,
        "knn_distance_std": float(np.std(neighbor_distances)) if len(neighbor_distances) else 0.0,
        "knn_distance_max": float(np.max(neighbor_distances)) if len(neighbor_distances) else 0.0,
        "neighborhood_size": int(len(neighbor_indices) + 1),
    }
    for column in DEFAULT_GEOMETRY_COLUMNS:
        if column in row.index:
            feature_row[column] = float(row[column])
    grid_size = int(config.get("betti_grid_size", 32))
    for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
        diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
        feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=grid_size))
        feature_row[f"{prefix}_wasserstein_to_clear"] = _safe_wasserstein(diagram, prototypes["clear"][homology_dim])
        feature_row[f"{prefix}_wasserstein_to_ambiguous"] = _safe_wasserstein(
            diagram,
            prototypes["ambiguous"][homology_dim],
        )
    return feature_row


def build_local_topology_features(
    *,
    query_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Build per-example local topology features from exported ambiguity coordinates."""

    if query_frame.empty:
        return pd.DataFrame()
    if reference_frame.empty:
        raise ValueError("Reference frame for local topology features is empty.")

    coordinate_columns = _coordinate_columns(reference_frame, list(config.get("coordinate_columns", ["z_0", "z_1", "signed_distance"])))
    reference_points = reference_frame.loc[:, coordinate_columns].to_numpy(dtype=float)
    query_points = query_frame.loc[:, coordinate_columns].to_numpy(dtype=float)

    if bool(config.get("standardize_coordinates", True)):
        scaler = StandardScaler()
        reference_points = scaler.fit_transform(reference_points)
        query_points = scaler.transform(query_points)

    reference_labels = reference_frame["label_ambiguous"].to_numpy(dtype=int)
    prototypes = _prototype_diagrams(
        reference_points,
        reference_labels,
        sample_n=int(config.get("prototype_sample_n", 96)),
        maxdim=int(config.get("maxdim", 1)),
        coeff=int(config.get("coeff", 2)),
        seed=seed,
    )

    neighborhood_k = int(config.get("neighborhood_k", 24))
    search_k = min(len(reference_frame), neighborhood_k + 1)
    model = NearestNeighbors(n_neighbors=max(search_k, 1), metric="euclidean")
    model.fit(reference_points)
    distances, indices = model.kneighbors(query_points, return_distance=True)
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
                query_point=query_points[query_idx],
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                reference_points=reference_points,
                prototypes=prototypes,
                config=config,
            )
        )
    return pd.DataFrame(rows)


def _fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[LogisticRegression, StandardScaler | None]:
    scaler = None
    x_fit = x_train
    if bool(config.get("standardize", True)):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)
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


def _transform_with_scaler(matrix: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return matrix
    return scaler.transform(matrix)


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
    train_scores = clf.decision_function(_transform_with_scaler(x_train, scaler))
    eval_scores = clf.decision_function(_transform_with_scaler(x_eval, scaler))
    payload = {
        "classifier": clf,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "eval_metrics": binary_classification_metrics(y_eval, eval_scores),
        "coefficients": clf.coef_.ravel().astype(float).tolist(),
        "intercept": float(clf.intercept_.ravel()[0]),
    }
    return payload["train_metrics"], payload


def _feature_columns(feature_frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    topology_columns = [
        column
        for column in feature_frame.columns
        if column.startswith(TOPOLOGY_PREFIXES)
    ]
    geometry_columns = [column for column in DEFAULT_GEOMETRY_COLUMNS if column in feature_frame.columns]
    hybrid_columns = topology_columns + geometry_columns
    return topology_columns, geometry_columns, hybrid_columns


def _selection_order(candidate_df: pd.DataFrame) -> pd.DataFrame:
    return candidate_df.sort_values(
        ["val_auroc", "val_f1", "val_accuracy", "layer", "subspace_name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def _select_multilayer_candidates(
    dataset_candidates: pd.DataFrame,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
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


def _layer_suffix(layer: int, subspace_name: str) -> str:
    return f"l{int(layer):02d}__{_slugify(str(subspace_name))}"


def _mergeable_feature_columns(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    topology_columns, geometry_columns, _ = _feature_columns(feature_frame)
    return topology_columns, geometry_columns


def _prepare_layer_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    layer: int,
    subspace_name: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    topology_columns, geometry_columns = _mergeable_feature_columns(feature_frame)
    suffix = _layer_suffix(layer, subspace_name)
    renamed = feature_frame.loc[:, BASE_KEY_COLUMNS + topology_columns + geometry_columns].copy()
    rename_map = {
        column: f"{column}__{suffix}"
        for column in topology_columns + geometry_columns
    }
    renamed = renamed.rename(columns=rename_map)
    return renamed, [rename_map[column] for column in topology_columns], [rename_map[column] for column in geometry_columns]


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
    summary_df = pd.DataFrame(summary_data)
    return summary_df, summary_columns_by_group


def _build_multilayer_feature_frames(
    *,
    train_all: pd.DataFrame,
    test_all: pd.DataFrame,
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
        subspace_name = str(selection["subspace_name"])
        selection_specs.append(
            {
                "rank": rank,
                "layer": layer,
                "subspace_name": subspace_name,
                "val_auroc": float(selection["val_auroc"]),
            }
        )
        selected_train = train_all.loc[
            train_all["layer"].eq(layer) & train_all["subspace_name"].eq(subspace_name)
        ].copy()
        selected_test = test_all.loc[
            test_all["layer"].eq(layer) & test_all["subspace_name"].eq(subspace_name)
        ].copy()
        if selected_train.empty or selected_test.empty:
            continue
        train_features = build_local_topology_features(
            query_frame=selected_train,
            reference_frame=selected_train,
            config=config,
            seed=seed + 401 + rank,
        )
        test_features = build_local_topology_features(
            query_frame=selected_test,
            reference_frame=selected_train,
            config=config,
            seed=seed + 401 + rank,
        )
        train_prepared, train_topology, train_geometry = _prepare_layer_feature_frame(
            train_features,
            layer=layer,
            subspace_name=subspace_name,
        )
        test_prepared, test_topology, test_geometry = _prepare_layer_feature_frame(
            test_features,
            layer=layer,
            subspace_name=subspace_name,
        )
        topology_columns.extend(train_topology)
        geometry_columns.extend(train_geometry)
        if train_merged is None:
            train_merged = train_prepared
            test_merged = test_prepared
        else:
            train_merged = train_merged.merge(train_prepared, on=BASE_KEY_COLUMNS, how="inner")
            test_merged = test_merged.merge(test_prepared, on=BASE_KEY_COLUMNS, how="inner")

    if train_merged is None or test_merged is None or train_merged.empty or test_merged.empty:
        raise ValueError("Failed to build non-empty multilayer topology features.")

    topology_columns = [column for column in topology_columns if column in train_merged.columns]
    geometry_columns = [column for column in geometry_columns if column in train_merged.columns]
    train_summary, train_summary_groups = _stacked_summary_features(
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
        "topology_summary_columns": train_summary_groups["topology"],
        "geometry_summary_columns": train_summary_groups["geometry"],
    }


def _plot_candidate_heatmap(candidate_df: pd.DataFrame, *, dataset: str, output_path: Path) -> None:
    subset = candidate_df.loc[
        candidate_df["dataset"].eq(dataset) & candidate_df["feature_set"].eq("topology_only")
    ]
    if subset.empty:
        return
    subspaces = list(dict.fromkeys(subset["subspace_name"].tolist()))
    layers = sorted(subset["layer"].unique())
    matrix = subset.pivot(index="subspace_name", columns="layer", values="val_auroc").reindex(index=subspaces, columns=layers)
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(layers)), 3.8))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_title(f"{dataset}: topology-only validation AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Subspace")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(int(layer)) for layer in layers], rotation=90, fontsize=8)
    ax.set_yticks(range(len(subspaces)))
    ax.set_yticklabels(subspaces)
    fig.colorbar(image, ax=ax, shrink=0.9, label="Validation AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_metric_bars(final_df: pd.DataFrame, *, dataset: str, output_path: Path) -> None:
    subset = final_df.loc[final_df["dataset"].eq(dataset)].copy()
    if subset.empty:
        return
    feature_sets = [
        feature_set
        for feature_set in SINGLE_FEATURE_SETS + MULTILAYER_FEATURE_SETS
        if feature_set in set(subset["feature_set"])
    ]
    metrics = ["test_auroc", "test_accuracy", "test_f1"]
    x = np.arange(len(metrics))
    width = min(0.13, 0.75 / max(len(feature_sets), 1))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    center = (len(feature_sets) - 1) / 2.0
    for offset, feature_set in enumerate(feature_sets):
        rows = subset.loc[subset["feature_set"].eq(feature_set)]
        if rows.empty:
            continue
        row = rows.iloc[0]
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
    ax.set_title(f"{dataset}: final classifier comparison")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=min(3, len(feature_sets)),
        frameon=False,
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(top=0.72)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_report(
    *,
    model_name: str,
    selected_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Topology Classifier Summary",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset in sorted(selected_df["dataset"].unique()):
        selected = selected_df.loc[
            selected_df["dataset"].eq(dataset) & selected_df["selection_mode"].eq("single_layer")
        ].iloc[0]
        multilayer_rows = selected_df.loc[
            selected_df["dataset"].eq(dataset) & selected_df["selection_mode"].eq("multilayer_component")
        ].sort_values("component_rank")
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Selected single layer: `{int(selected['layer'])}` / `{selected['subspace_name']}`",
                f"- Single-layer topology validation AUROC: `{selected['val_auroc']:.4f}`",
                f"- Coordinate columns: `{selected['coordinate_columns']}`",
                "",
                "### Multi-layer Selection",
                "",
            ]
        )
        if multilayer_rows.empty:
            lines.append("- No multi-layer stack selected.")
        else:
            for row in multilayer_rows.to_dict(orient="records"):
                lines.append(
                    f"- Rank `{int(row['component_rank'])}`: layer `{int(row['layer'])}` / `{row['subspace_name']}` "
                    f"(val AUROC `{row['val_auroc']:.4f}`)."
                )
        lines.extend(
            [
                "",
                "### Final Test Metrics",
                "",
            ]
        )
        dataset_final = final_df.loc[final_df["dataset"].eq(dataset)]
        for row in dataset_final.to_dict(orient="records"):
            lines.append(
                f"- `{row['feature_set']}`: AUROC `{row['test_auroc']:.4f}`, accuracy `{row['test_accuracy']:.4f}`, "
                f"F1 `{row['test_f1']:.4f}`."
            )
        lines.append("")
    write_markdown(output_path, "\n".join(lines) + "\n")


def run_topology_classifier_analysis(
    *,
    model_name: str,
    layerwise_features_path: str | Path,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    """Train ambiguity classifiers from local topological descriptors."""

    features_df = pd.read_parquet(layerwise_features_path)
    if features_df.empty:
        raise ValueError("Topology classifier requires non-empty layerwise features.")

    readout = str(classifier_config.get("readout", "mean_pool"))
    datasets = set(classifier_config.get("datasets", sorted(features_df["dataset"].unique())))
    subspaces = set(classifier_config.get("candidate_subspaces", sorted(features_df["subspace_name"].unique())))
    filtered = features_df.loc[
        features_df["dataset"].isin(datasets)
        & features_df["readout"].eq(readout)
        & features_df["subspace_name"].isin(subspaces)
    ].copy()
    if filtered.empty:
        raise ValueError("Topology classifier found no matching rows after dataset/readout/subspace filtering.")

    model_slug = _slugify(model_name)
    output_root = ensure_dir(Path(classifier_config["output_dir"]) / model_slug)
    plots_root = ensure_dir(output_root / "plots")
    models_root = ensure_dir(output_root / "models")

    classifier_section = dict(classifier_config.get("classifier", {}))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_feature_tables: list[pd.DataFrame] = []

    for dataset in sorted(filtered["dataset"].unique()):
        dataset_df = filtered.loc[filtered["dataset"].eq(dataset)].copy()
        train_all = dataset_df.loc[dataset_df["split"].eq("train")].copy()
        test_all = dataset_df.loc[dataset_df["split"].eq("test")].copy()
        if train_all.empty or test_all.empty:
            LOGGER.warning("Skipping dataset %s because train/test split rows are missing.", dataset)
            continue

        candidate_layers = _resolve_candidate_layers(
            dataset_frame=dataset_df,
            model_name=model_name,
            dataset_name=dataset,
            config=classifier_config,
        )
        inner_train_ids, val_ids = _group_train_val_split(
            train_all,
            val_fraction=float(classifier_config.get("val_fraction", 0.2)),
            seed=seed,
        )
        if not val_ids:
            raise ValueError(f"Failed to allocate an inner validation set for dataset {dataset}.")

        for layer in candidate_layers:
            for subspace_name in sorted(subspaces):
                combo_train = train_all.loc[
                    train_all["layer"].eq(layer) & train_all["subspace_name"].eq(subspace_name)
                ].copy()
                combo_test = test_all.loc[
                    test_all["layer"].eq(layer) & test_all["subspace_name"].eq(subspace_name)
                ].copy()
                if combo_train.empty or combo_test.empty:
                    continue
                inner_train = combo_train.loc[combo_train["example_id"].astype(str).isin(inner_train_ids)].copy()
                inner_val = combo_train.loc[combo_train["example_id"].astype(str).isin(val_ids)].copy()
                if inner_train.empty or inner_val.empty:
                    continue
                if inner_train["label_ambiguous"].nunique() < 2 or inner_val["label_ambiguous"].nunique() < 2:
                    continue

                local_train = build_local_topology_features(
                    query_frame=inner_train,
                    reference_frame=inner_train,
                    config=classifier_config,
                    seed=seed + int(layer),
                )
                local_val = build_local_topology_features(
                    query_frame=inner_val,
                    reference_frame=inner_train,
                    config=classifier_config,
                    seed=seed + int(layer),
                )
                topology_columns, geometry_columns, hybrid_columns = _feature_columns(local_train)
                feature_sets = {
                    "topology_only": topology_columns,
                    "geometry_only": geometry_columns,
                    "hybrid": hybrid_columns,
                }
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
                    candidate_rows.append(
                        {
                            "dataset": dataset,
                            "layer": int(layer),
                            "subspace_name": subspace_name,
                            "selection_mode": "single_layer",
                            "feature_set": feature_set,
                            "val_auroc": float(metrics["auroc"]),
                            "val_accuracy": float(metrics["accuracy"]),
                            "val_f1": float(metrics["f1"]),
                            "train_n": int(len(local_train)),
                            "val_n": int(len(local_val)),
                            "feature_count": int(len(columns)),
                            "coordinate_columns": _coordinate_columns(combo_train, list(classifier_config.get("coordinate_columns", ["z_0", "z_1", "signed_distance"]))),
                        }
                    )

        candidate_df = pd.DataFrame(candidate_rows)
        dataset_candidates = candidate_df.loc[
            candidate_df["dataset"].eq(dataset) & candidate_df["feature_set"].eq("topology_only")
        ].copy()
        if dataset_candidates.empty:
            raise ValueError(f"No topology-only candidates were produced for dataset {dataset}.")
        dataset_candidates = _selection_order(dataset_candidates)
        best_row = dataset_candidates.iloc[0]
        selected_rows.append(
            {
                **best_row.to_dict(),
                "selection_mode": "single_layer",
                "component_rank": 1,
            }
        )

        multilayer_enabled = bool(classifier_config.get("multilayer_enabled", True))
        multilayer_candidates: list[dict[str, Any]] = []
        if multilayer_enabled:
            multilayer_candidates = _select_multilayer_candidates(
                dataset_candidates,
                top_k=int(classifier_config.get("multilayer_top_k", 3)),
            )
            for rank, row in enumerate(multilayer_candidates, start=1):
                selected_rows.append(
                    {
                        **row,
                        "selection_mode": "multilayer_component",
                        "component_rank": rank,
                    }
                )

        selected_train = train_all.loc[
            train_all["layer"].eq(int(best_row["layer"]))
            & train_all["subspace_name"].eq(str(best_row["subspace_name"]))
        ].copy()
        selected_test = test_all.loc[
            test_all["layer"].eq(int(best_row["layer"]))
            & test_all["subspace_name"].eq(str(best_row["subspace_name"]))
        ].copy()
        final_train_features = build_local_topology_features(
            query_frame=selected_train,
            reference_frame=selected_train,
            config=classifier_config,
            seed=seed + 101,
        )
        final_test_features = build_local_topology_features(
            query_frame=selected_test,
            reference_frame=selected_train,
            config=classifier_config,
            seed=seed + 101,
        )
        final_train_features["feature_role"] = "train"
        final_test_features["feature_role"] = "test"
        final_train_features["selected_for_dataset"] = dataset
        final_test_features["selected_for_dataset"] = dataset
        final_train_features["feature_variant"] = "single_layer"
        final_test_features["feature_variant"] = "single_layer"
        selected_feature_tables.extend([final_train_features, final_test_features])

        topology_columns, geometry_columns, hybrid_columns = _feature_columns(final_train_features)
        feature_sets = {
            "topology_only": topology_columns,
            "geometry_only": geometry_columns,
            "hybrid": hybrid_columns,
        }
        for feature_set, columns in feature_sets.items():
            if not columns:
                continue
            _, payload = _evaluate_feature_set(
                train_features=final_train_features,
                eval_features=final_test_features,
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
                    "subspace_name": str(best_row["subspace_name"]),
                    "selection_signature": f"{int(best_row['layer'])}:{str(best_row['subspace_name'])}",
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
                    "subspace_name": str(best_row["subspace_name"]),
                    "train_metrics": payload["train_metrics"],
                    "test_metrics": payload["eval_metrics"],
                },
                model_path,
            )

        if multilayer_candidates:
            multi_train_features, multi_test_features, multi_meta = _build_multilayer_feature_frames(
                train_all=train_all,
                test_all=test_all,
                selections=multilayer_candidates,
                config=classifier_config,
                seed=seed,
            )
            multi_train_features["feature_role"] = "train"
            multi_test_features["feature_role"] = "test"
            multi_train_features["selected_for_dataset"] = dataset
            multi_test_features["selected_for_dataset"] = dataset
            multi_train_features["feature_variant"] = "multilayer"
            multi_test_features["feature_variant"] = "multilayer"
            selected_feature_tables.extend([multi_train_features, multi_test_features])

            multi_topology_columns = multi_meta["topology_columns"] + multi_meta["topology_summary_columns"]
            multi_geometry_columns = multi_meta["geometry_columns"] + multi_meta["geometry_summary_columns"]
            multi_feature_sets = {
                "topology_multilayer": multi_topology_columns,
                "geometry_multilayer": multi_geometry_columns,
                "hybrid_multilayer": multi_topology_columns + multi_geometry_columns,
            }
            selection_signature = " | ".join(
                f"{item['layer']}:{item['subspace_name']}" for item in multi_meta["selections"]
            )
            for feature_set, columns in multi_feature_sets.items():
                if not columns:
                    continue
                _, payload = _evaluate_feature_set(
                    train_features=multi_train_features,
                    eval_features=multi_test_features,
                    feature_columns=columns,
                    classifier_config=classifier_section,
                    seed=seed,
                )
                metrics = payload["eval_metrics"]
                final_rows.append(
                    {
                        "dataset": dataset,
                        "selection_mode": "multilayer",
                        "layer": -1,
                        "subspace_name": "stacked",
                        "selection_signature": selection_signature,
                        "selection_size": int(len(multi_meta["selections"])),
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
                        "selection_mode": "multilayer",
                        "selection_signature": selection_signature,
                        "selection_components": multi_meta["selections"],
                        "train_metrics": payload["train_metrics"],
                        "test_metrics": payload["eval_metrics"],
                    },
                    model_path,
                )

        _plot_candidate_heatmap(candidate_df, dataset=dataset, output_path=plots_root / f"{dataset}__candidate_heatmap.png")

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["dataset", "feature_set", "selection_mode", "val_auroc", "layer", "subspace_name"],
        ascending=[True, True, True, False, True, True],
    ).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["dataset", "selection_mode", "feature_set"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["dataset", "selection_mode", "component_rank"]).reset_index(drop=True)
    selected_features_df = pd.concat(selected_feature_tables, ignore_index=True) if selected_feature_tables else pd.DataFrame()

    for dataset in sorted(final_df["dataset"].unique()):
        _plot_final_metric_bars(final_df, dataset=dataset, output_path=plots_root / f"{dataset}__final_classifier_metrics.png")

    candidate_path = output_root / str(classifier_config["candidate_metrics_filename"])
    final_path = output_root / str(classifier_config["final_metrics_filename"])
    selected_path = output_root / str(classifier_config["selected_candidates_filename"])
    feature_path = output_root / str(classifier_config["feature_table_filename"])
    report_path = output_root / str(classifier_config["report_filename"])
    metadata_path = output_root / str(classifier_config["metadata_filename"])

    write_parquet(candidate_df, candidate_path)
    write_parquet(final_df, final_path)
    write_parquet(selected_df, selected_path)
    if not selected_features_df.empty:
        write_parquet(selected_features_df, feature_path)
    _render_report(model_name=model_name, selected_df=selected_df, final_df=final_df, output_path=report_path)
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "generated_at": utc_now_iso(),
            "config": classifier_config,
            "outputs": {
                "candidate_metrics": str(candidate_path),
                "final_metrics": str(final_path),
                "selected_candidates": str(selected_path),
                "feature_table": str(feature_path) if feature_path.exists() else None,
                "report": str(report_path),
            },
        },
    )
    LOGGER.info("Saved topology classifier artifacts to %s", output_root)
    return {
        "candidate_metrics_path": str(candidate_path),
        "final_metrics_path": str(final_path),
        "selected_candidates_path": str(selected_path),
        "feature_table_path": str(feature_path) if feature_path.exists() else "",
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }
