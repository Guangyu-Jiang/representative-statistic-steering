"""CLAMBER subclass classification experiments.

This module evaluates whether different representation views can separate the
fine-grained CLAMBER subclasses. The default setting is 9-way classification
that includes the `none` class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.aen import _fit_probe, _transform
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _distance_feature_mode,
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
    _prototype_diagrams_from_clouds,
    _selection_order,
    _token_cloud_forward_cache_signature,
    _token_cloud_feature_cache_signature,
    _topology_feature_columns,
    build_token_cloud_feature_frame,
    load_cached_token_cloud_forward_frame,
    load_cached_token_cloud_feature_frame,
    run_token_cloud_topology_classifier_from_features,
    save_cached_token_cloud_forward_frame,
    save_cached_token_cloud_feature_frame,
)
from aen_replication.utils.io_utils import ensure_dir, slugify, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)


def _write_progress(path: Path, *, stage: str, model_name: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "model_name": model_name,
        "updated_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }


def _normalize_object_column(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return None
    return value


def _fit_multiclass_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int,
    c_value: float,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(c_value),
        max_iter=int(max_iter),
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _evaluate_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    max_iter: int,
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    labels = sorted({str(label) for label in np.concatenate([y_train, y_eval]).tolist()})
    clf, scaler = _fit_multiclass_logistic(
        x_train=x_train,
        y_train=y_train,
        max_iter=max_iter,
        c_value=c_value,
        seed=seed,
    )
    predictions = clf.predict(scaler.transform(x_eval))
    return {
        "classifier": clf,
        "scaler": scaler,
        "metrics": _multiclass_metrics(y_eval, predictions, labels),
    }


def _stratified_split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _subclass_train_val_ids(dataset_df: pd.DataFrame, *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    unique_train = dataset_df.loc[dataset_df["split"].eq("train"), ["example_id", "subclass"]].drop_duplicates().reset_index(drop=True)
    train_idx, val_idx = _stratified_split_indices(unique_train["subclass"], val_fraction=val_fraction, seed=seed)
    return (
        set(unique_train.iloc[train_idx]["example_id"].astype(str)),
        set(unique_train.iloc[val_idx]["example_id"].astype(str)),
    )


def _select_train_only_aens(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    probe_cfg: dict[str, Any],
    val_fraction: float,
    perturb_top_k: list[int],
    sigma: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    train_idx, val_idx = _stratified_split_indices(pd.Series(y_train), val_fraction=val_fraction, seed=seed)
    clf, scaler = _fit_probe(
        x_train=x_train[train_idx],
        y_train=y_train[train_idx],
        probe_cfg=probe_cfg,
        seed=seed,
    )
    x_val = _transform(x_train[val_idx], scaler)
    y_val = y_train[val_idx]
    weights = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    ranked = np.argsort(-weights)
    baseline_pred = clf.decision_function(x_val)
    baseline_accuracy = float(np.mean((baseline_pred >= 0.0).astype(int) == y_val))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for k in perturb_top_k:
        k = min(int(k), x_val.shape[1])
        if k <= 0:
            continue
        indices = ranked[:k]
        trial_accs: list[float] = []
        for _ in range(max(1, int(trials))):
            perturbed = x_val.copy()
            perturbed[:, indices] += rng.normal(0.0, float(sigma), size=(perturbed.shape[0], len(indices)))
            scores = clf.decision_function(perturbed)
            trial_accs.append(float(np.mean((scores >= 0.0).astype(int) == y_val)))
        mean_acc = float(np.mean(trial_accs))
        results.append(
            {
                "k": k,
                "indices": indices.tolist(),
                "accuracy_after_perturb": mean_acc,
                "accuracy_drop": baseline_accuracy - mean_acc,
            }
        )
    best = max(results, key=lambda row: float(row["accuracy_drop"]))
    return {
        "baseline_accuracy": baseline_accuracy,
        "aen_indices": list(best["indices"]),
        "aen_k": int(best["k"]),
        "results": results,
    }


def _filter_subclass_rows(
    meta: pd.DataFrame,
    matrix: np.ndarray,
    *,
    split: str,
    ambiguous_only: bool,
    include_none_class: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.asarray(meta["split"].eq(split).to_numpy(), dtype=bool).copy()
    if ambiguous_only:
        mask &= np.asarray(meta["label_ambiguous"].eq(1).to_numpy(), dtype=bool)
    if not include_none_class and "subclass" in meta.columns:
        mask &= np.asarray(meta["subclass"].astype(str).ne("none").to_numpy(), dtype=bool)
    subset_meta = meta.loc[mask].reset_index(drop=True)
    subset_matrix = matrix[mask]
    return subset_meta, subset_matrix


def _evaluate_full_and_aen_layers(
    *,
    hidden_root: Path,
    candidate_layers: list[int],
    probe_cfg: dict[str, Any],
    subclass_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    ambiguous_only = bool(subclass_cfg.get("ambiguous_only", False))
    include_none_class = bool(subclass_cfg.get("include_none_class", True))
    val_fraction = float(subclass_cfg.get("val_fraction", 0.2))

    for layer in candidate_layers:
        meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
        train_meta, train_matrix = _filter_subclass_rows(
            meta,
            matrix,
            split="train",
            ambiguous_only=ambiguous_only,
            include_none_class=include_none_class,
        )
        if train_meta.empty:
            continue
        tr_idx, val_idx = _stratified_split_indices(train_meta["subclass"], val_fraction=val_fraction, seed=seed + layer)
        x_train = train_matrix[tr_idx]
        y_train = train_meta.iloc[tr_idx]["subclass"].astype(str).to_numpy()
        x_val = train_matrix[val_idx]
        y_val = train_meta.iloc[val_idx]["subclass"].astype(str).to_numpy()

        full_eval = _evaluate_multiclass(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_val,
            y_eval=y_val,
            max_iter=max_iter,
            c_value=c_value,
            seed=seed + layer,
        )

        binary_train_meta = meta.loc[meta["split"].eq("train")].reset_index(drop=True)
        binary_train_matrix = matrix[meta["split"].eq("train").to_numpy()]
        aen_info = _select_train_only_aens(
            x_train=binary_train_matrix,
            y_train=binary_train_meta["label_ambiguous"].to_numpy(dtype=int),
            probe_cfg=probe_cfg,
            val_fraction=val_fraction,
            perturb_top_k=list(subclass_cfg.get("perturb_top_k", [1, 2, 3, 5, 10, 20])),
            sigma=float(subclass_cfg.get("perturb_sigma", 0.15)),
            trials=int(subclass_cfg.get("perturb_trials", 8)),
            seed=seed + 100 + layer,
        )
        aen_indices = list(aen_info["aen_indices"])
        aen_eval = _evaluate_multiclass(
            x_train=x_train[:, aen_indices],
            y_train=y_train,
            x_eval=x_val[:, aen_indices],
            y_eval=y_val,
            max_iter=max_iter,
            c_value=c_value,
            seed=seed + 200 + layer,
        )
        rows.extend(
            [
                {
                    "method": "full_probe",
                    "layer": int(layer),
                    "val_accuracy": float(full_eval["metrics"]["accuracy"]),
                    "val_macro_f1": float(full_eval["metrics"]["macro_f1"]),
                    "feature_count": int(x_train.shape[1]),
                    "aen_k": np.nan,
                },
                {
                    "method": "aen_only",
                    "layer": int(layer),
                    "val_accuracy": float(aen_eval["metrics"]["accuracy"]),
                    "val_macro_f1": float(aen_eval["metrics"]["macro_f1"]),
                    "feature_count": int(len(aen_indices)),
                    "aen_k": int(aen_info["aen_k"]),
                },
            ]
        )

    candidate_df = pd.DataFrame(rows).sort_values(["method", "val_macro_f1", "val_accuracy", "layer"], ascending=[True, False, False, True]).reset_index(drop=True)
    if candidate_df.empty:
        raise ValueError("No subclass candidates were evaluated for CLAMBER.")

    best_full = candidate_df.loc[candidate_df["method"].eq("full_probe")].iloc[0].to_dict()
    best_aen = candidate_df.loc[candidate_df["method"].eq("aen_only")].iloc[0].to_dict()
    return candidate_df, best_full, best_aen


def _finalize_mean_pool_result(
    *,
    hidden_root: Path,
    layer: int,
    probe_cfg: dict[str, Any],
    subclass_cfg: dict[str, Any],
    seed: int,
    method: str,
) -> dict[str, Any]:
    meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    ambiguous_only = bool(subclass_cfg.get("ambiguous_only", False))
    include_none_class = bool(subclass_cfg.get("include_none_class", True))
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    val_fraction = float(subclass_cfg.get("val_fraction", 0.2))

    train_meta, train_matrix = _filter_subclass_rows(
        meta,
        matrix,
        split="train",
        ambiguous_only=ambiguous_only,
        include_none_class=include_none_class,
    )
    test_meta, test_matrix = _filter_subclass_rows(
        meta,
        matrix,
        split="test",
        ambiguous_only=ambiguous_only,
        include_none_class=include_none_class,
    )
    y_train = train_meta["subclass"].astype(str).to_numpy()
    y_test = test_meta["subclass"].astype(str).to_numpy()
    x_train = train_matrix
    x_test = test_matrix
    aen_info = None
    if method == "aen_only":
        binary_train_meta = meta.loc[meta["split"].eq("train")].reset_index(drop=True)
        binary_train_matrix = matrix[meta["split"].eq("train").to_numpy()]
        aen_info = _select_train_only_aens(
            x_train=binary_train_matrix,
            y_train=binary_train_meta["label_ambiguous"].to_numpy(dtype=int),
            probe_cfg=probe_cfg,
            val_fraction=val_fraction,
            perturb_top_k=list(subclass_cfg.get("perturb_top_k", [1, 2, 3, 5, 10, 20])),
            sigma=float(subclass_cfg.get("perturb_sigma", 0.15)),
            trials=int(subclass_cfg.get("perturb_trials", 8)),
            seed=seed + 100 + int(layer),
        )
        indices = list(aen_info["aen_indices"])
        x_train = x_train[:, indices]
        x_test = x_test[:, indices]
    payload = _evaluate_multiclass(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_test,
        y_eval=y_test,
        max_iter=max_iter,
        c_value=c_value,
        seed=seed + 1000 + int(layer),
    )
    result = {
        "method": method,
        "layer": int(layer),
        "test_accuracy": float(payload["metrics"]["accuracy"]),
        "test_macro_f1": float(payload["metrics"]["macro_f1"]),
        "feature_count": int(x_train.shape[1]),
        "test_confusion_matrix": payload["metrics"]["confusion_matrix"],
        "test_labels": payload["metrics"]["labels"],
    }
    if aen_info is not None:
        result["aen_k"] = int(aen_info["aen_k"])
        result["aen_indices"] = list(aen_info["aen_indices"])
    return result


def _build_clamber_token_cloud_features(
    *,
    config: dict[str, Any],
    classifier_config: dict[str, Any],
    subclass_cfg: dict[str, Any],
    seed: int,
) -> tuple[str, pd.DataFrame, bool]:
    model_name = str(config["model"]["name"])
    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    layers = [int(layer) for layer in subclass_cfg.get("token_cloud_candidate_layers", classifier_config.get("candidate_layers", [0, 14, 31]))]
    token_cfg = {
        **classifier_config,
        "batch_size": int(subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8))),
        "max_length": int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64))),
        "parallel_jobs": int(subclass_cfg.get("token_cloud_parallel_jobs", classifier_config.get("parallel_jobs", 12))),
        "pca_components": int(subclass_cfg.get("token_cloud_pca_components", classifier_config.get("pca_components", 16))),
        "topology_components": int(subclass_cfg.get("token_cloud_topology_components", classifier_config.get("topology_components", 16))),
        "prototype_token_cap": int(subclass_cfg.get("token_cloud_prototype_token_cap", classifier_config.get("prototype_token_cap", 192))),
        "distance_feature_mode": str(
            subclass_cfg.get("token_cloud_distance_feature_mode", classifier_config.get("distance_feature_mode", "knn_class"))
        ),
        "distance_feature_k": int(
            subclass_cfg.get("token_cloud_distance_feature_k", classifier_config.get("distance_feature_k", 8))
        ),
        "distance_feature_chunk_size": int(
            subclass_cfg.get("token_cloud_distance_feature_chunk_size", classifier_config.get("distance_feature_chunk_size", 24))
        ),
        "subclass_distance_max_workers": int(
            subclass_cfg.get(
                "token_cloud_subclass_distance_max_workers",
                classifier_config.get("subclass_distance_max_workers", 2),
            )
        ),
        "subclass_distance_executor": str(
            subclass_cfg.get(
                "token_cloud_subclass_distance_executor",
                classifier_config.get("subclass_distance_executor", "process"),
            )
        ),
        "betti_grid_size": int(subclass_cfg.get("token_cloud_betti_grid_size", classifier_config.get("betti_grid_size", 24))),
        "persistence_image_grid_side": int(
            subclass_cfg.get("token_cloud_persistence_image_grid_side", classifier_config.get("persistence_image_grid_side", 3))
        ),
        "_seed": seed,
    }
    raw_forward_cache_path = str(
        subclass_cfg.get(
            "token_cloud_forward_cache_path",
            classifier_config.get("forward_cache_path", ""),
        )
    ).strip()
    forward_cache_path = Path(raw_forward_cache_path) if raw_forward_cache_path else None
    raw_feature_cache_path = str(
        subclass_cfg.get(
            "token_cloud_feature_cache_path",
            classifier_config.get("feature_cache_path", ""),
        )
    ).strip()
    feature_cache_path = Path(raw_feature_cache_path) if raw_feature_cache_path else None
    feature_signature = _token_cloud_feature_cache_signature(
        model_name=model_name,
        dataset_paths=[dataset_path],
        layers=layers,
        classifier_config=token_cfg,
        seed=seed,
    )
    forward_signature = _token_cloud_forward_cache_signature(
        model_name=model_name,
        dataset_paths=[dataset_path],
        layers=layers,
        config=token_cfg,
        seed=seed,
    )
    if feature_cache_path is not None and not bool(subclass_cfg.get("force_rebuild_token_cloud_features", False)):
        cached_feature_df = load_cached_token_cloud_feature_frame(
            feature_path=feature_cache_path,
            signature=feature_signature,
        )
        if cached_feature_df is not None:
            return model_name, cached_feature_df, True

    dataset_df = pd.read_parquet(dataset_path)
    cached_cloud = None
    if forward_cache_path is not None and not bool(subclass_cfg.get("force_rebuild_token_cloud_forward_cache", False)):
        cached_cloud = load_cached_token_cloud_forward_frame(
            cache_path=forward_cache_path,
            signature=forward_signature,
        )
    if cached_cloud is not None:
        cloud_df, _ = cached_cloud
        prepared_df = dataset_df.copy()
    else:
        bundle: HFModelBundle = load_hf_model(config["model"], classifier_config)
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=str(classifier_config.get("text_column", "text")),
            use_chat_template=bool(classifier_config.get("use_chat_template", False)),
            system_prompt=classifier_config.get("system_prompt"),
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
        train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
        token_matrices = _extract_train_token_matrices(
            bundle=bundle,
            train_df=train_df,
            text_column="_token_cloud_text",
            layers=layers,
            config=token_cfg,
        )
        reducers = _fit_layer_reducers(token_matrices, config=token_cfg, seed=seed)
        cloud_df = _extract_reduced_clouds(
            bundle=bundle,
            df=prepared_df,
            text_column="_token_cloud_text",
            layers=layers,
            reducers=reducers,
            config=token_cfg,
        )
        if forward_cache_path is not None:
            save_cached_token_cloud_forward_frame(
                cloud_df=cloud_df,
                cache_path=forward_cache_path,
                signature=forward_signature,
            )
    join_meta = prepared_df.loc[:, ["example_id", "subclass"]].drop_duplicates().copy()
    cloud_df = cloud_df.merge(join_meta, on="example_id", how="left")
    prototype_map = None
    if _distance_feature_mode(token_cfg) == "prototype":
        prototype_map = _prototype_diagrams_from_clouds(cloud_df, layers=layers, config=token_cfg, seed=seed)
    feature_df = build_token_cloud_feature_frame(cloud_df, prototype_map=prototype_map, config=token_cfg)
    if "subclass" not in feature_df.columns:
        feature_df = feature_df.merge(join_meta, on="example_id", how="left")
    if feature_cache_path is not None:
        save_cached_token_cloud_feature_frame(
            feature_df=feature_df,
            feature_path=feature_cache_path,
            signature=feature_signature,
        )
    return model_name, feature_df, False


def _token_cloud_dataset(feature_df: pd.DataFrame, *, subclass_cfg: dict[str, Any]) -> pd.DataFrame:
    dataset_df = feature_df.copy()
    if bool(subclass_cfg.get("ambiguous_only", False)):
        dataset_df = dataset_df.loc[dataset_df["label_ambiguous"].eq(1)].copy()
    if not bool(subclass_cfg.get("include_none_class", True)):
        dataset_df = dataset_df.loc[dataset_df["subclass"].astype(str).ne("none")].copy()
    return dataset_df


def _token_cloud_selection_records(
    candidate_df: pd.DataFrame,
    *,
    top_k: int | None = None,
    use_all_layers: bool = False,
) -> list[dict[str, Any]]:
    ranked = candidate_df.sort_values(["val_macro_f1", "val_accuracy", "layer"], ascending=[False, False, True]).reset_index(drop=True)
    if use_all_layers:
        selected_df = candidate_df.sort_values(["layer"], ascending=[True]).reset_index(drop=True)
    else:
        selected_df = ranked.head(min(max(1, int(top_k or 1)), len(ranked))).reset_index(drop=True)
    return [
        {
            "layer": int(row["layer"]),
            "val_auroc": float(row["val_macro_f1"]),
            "val_accuracy": float(row["val_accuracy"]),
            "val_f1": float(row["val_macro_f1"]),
        }
        for row in selected_df.to_dict(orient="records")
    ]


def _build_token_cloud_single_feature_frames(
    dataset_df: pd.DataFrame,
    *,
    layer: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    layer_df = dataset_df.loc[dataset_df["layer"].eq(layer)].copy()
    columns = _topology_feature_columns(layer_df)
    base_columns = ["example_id", "subclass"] + columns
    train_frame = layer_df.loc[layer_df["split"].eq("train"), base_columns].drop_duplicates("example_id").reset_index(drop=True)
    test_frame = layer_df.loc[layer_df["split"].eq("test"), base_columns].drop_duplicates("example_id").reset_index(drop=True)
    return train_frame, test_frame, columns


def _build_token_cloud_multilayer_result(
    dataset_df: pd.DataFrame,
    *,
    selections: list[dict[str, Any]],
    max_iter: int,
    c_value: float,
    seed: int,
    method: str,
) -> dict[str, Any]:
    subclass_lookup = dataset_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
    multi_train, multi_test, multi_meta = _build_multilayer_feature_frames(
        feature_df=dataset_df,
        dataset="clamber",
        selections=selections,
    )
    multi_train = multi_train.merge(subclass_lookup, on="example_id", how="left")
    multi_test = multi_test.merge(subclass_lookup, on="example_id", how="left")
    multi_columns = multi_meta["topology_columns"] + multi_meta["topology_summary_columns"]
    multi_payload = _evaluate_multiclass(
        x_train=multi_train.loc[:, multi_columns].to_numpy(dtype=float),
        y_train=multi_train["subclass"].astype(str).to_numpy(),
        x_eval=multi_test.loc[:, multi_columns].to_numpy(dtype=float),
        y_eval=multi_test["subclass"].astype(str).to_numpy(),
        max_iter=max_iter,
        c_value=c_value,
        seed=seed,
    )
    return {
        "method": method,
        "layer": -1,
        "selection_signature": " | ".join(str(int(item["layer"])) for item in multi_meta["selections"]),
        "test_accuracy": float(multi_payload["metrics"]["accuracy"]),
        "test_macro_f1": float(multi_payload["metrics"]["macro_f1"]),
        "feature_count": int(len(multi_columns)),
        "test_confusion_matrix": multi_payload["metrics"]["confusion_matrix"],
        "test_labels": multi_payload["metrics"]["labels"],
    }


def _load_mean_pool_subclass_feature_frames(
    *,
    hidden_root: Path,
    layer: int,
    probe_cfg: dict[str, Any],
    subclass_cfg: dict[str, Any],
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    train_meta, train_matrix = _filter_subclass_rows(
        meta,
        matrix,
        split="train",
        ambiguous_only=bool(subclass_cfg.get("ambiguous_only", False)),
        include_none_class=bool(subclass_cfg.get("include_none_class", True)),
    )
    test_meta, test_matrix = _filter_subclass_rows(
        meta,
        matrix,
        split="test",
        ambiguous_only=bool(subclass_cfg.get("ambiguous_only", False)),
        include_none_class=bool(subclass_cfg.get("include_none_class", True)),
    )
    extra: dict[str, Any] = {}
    x_train = train_matrix
    x_test = test_matrix
    if method == "aen_only":
        aen_info = _select_train_only_aens(
            x_train=matrix[meta["split"].eq("train").to_numpy()],
            y_train=meta.loc[meta["split"].eq("train"), "label_ambiguous"].to_numpy(dtype=int),
            probe_cfg=probe_cfg,
            val_fraction=float(subclass_cfg.get("val_fraction", 0.2)),
            perturb_top_k=list(subclass_cfg.get("perturb_top_k", [1, 2, 3, 5, 10, 20])),
            sigma=float(subclass_cfg.get("perturb_sigma", 0.15)),
            trials=int(subclass_cfg.get("perturb_trials", 8)),
            seed=seed + 100 + int(layer),
        )
        indices = list(aen_info["aen_indices"])
        x_train = x_train[:, indices]
        x_test = x_test[:, indices]
        extra["aen_k"] = int(aen_info["aen_k"])
        extra["aen_indices"] = list(indices)
    columns = [f"{method}_feature_{index:04d}" for index in range(x_train.shape[1])]
    train_frame = pd.DataFrame(x_train, columns=columns)
    train_frame.insert(0, "example_id", train_meta["example_id"].astype(str).to_numpy())
    train_frame.insert(1, "subclass", train_meta["subclass"].astype(str).to_numpy())
    test_frame = pd.DataFrame(x_test, columns=columns)
    test_frame.insert(0, "example_id", test_meta["example_id"].astype(str).to_numpy())
    test_frame.insert(1, "subclass", test_meta["subclass"].astype(str).to_numpy())
    return train_frame, test_frame, columns, extra


def _build_token_cloud_hybrid_feature_sets(
    dataset_df: pd.DataFrame,
    *,
    candidate_df: pd.DataFrame,
    multilayer_top_k: int,
) -> dict[str, dict[str, Any]]:
    feature_sets: dict[str, dict[str, Any]] = {}
    for layer in sorted(dataset_df["layer"].unique()):
        train_frame, test_frame, columns = _build_token_cloud_single_feature_frames(dataset_df, layer=int(layer))
        feature_sets[f"token_cloud_single::{int(layer)}"] = {
            "hybrid_label": "token_cloud_single",
            "topology_key": f"token_cloud_single::{int(layer)}",
            "selection_signature": str(int(layer)),
            "train_frame": train_frame,
            "test_frame": test_frame,
            "feature_columns": columns,
        }
    top_k_selections = _token_cloud_selection_records(candidate_df, top_k=multilayer_top_k)
    if top_k_selections:
        train_frame, test_frame, meta = _build_multilayer_feature_frames(
            feature_df=dataset_df,
            dataset="clamber",
            selections=top_k_selections,
        )
        subclass_lookup = dataset_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
        train_frame = train_frame.merge(subclass_lookup, on="example_id", how="left")
        test_frame = test_frame.merge(subclass_lookup, on="example_id", how="left")
        columns = meta["topology_columns"] + meta["topology_summary_columns"]
        feature_sets["token_cloud_multilayer"] = {
            "hybrid_label": "token_cloud_multilayer",
            "topology_key": "token_cloud_multilayer",
            "selection_signature": " | ".join(str(int(item["layer"])) for item in meta["selections"]),
            "train_frame": train_frame.loc[:, ["example_id", "subclass"] + columns].copy(),
            "test_frame": test_frame.loc[:, ["example_id", "subclass"] + columns].copy(),
            "feature_columns": columns,
        }
    all_selections = _token_cloud_selection_records(candidate_df, use_all_layers=True)
    if all_selections:
        train_frame, test_frame, meta = _build_multilayer_feature_frames(
            feature_df=dataset_df,
            dataset="clamber",
            selections=all_selections,
        )
        subclass_lookup = dataset_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
        train_frame = train_frame.merge(subclass_lookup, on="example_id", how="left")
        test_frame = test_frame.merge(subclass_lookup, on="example_id", how="left")
        columns = meta["topology_columns"] + meta["topology_summary_columns"]
        feature_sets["token_cloud_multilayer_all"] = {
            "hybrid_label": "token_cloud_multilayer_all",
            "topology_key": "token_cloud_multilayer_all",
            "selection_signature": " | ".join(str(int(item["layer"])) for item in meta["selections"]),
            "train_frame": train_frame.loc[:, ["example_id", "subclass"] + columns].copy(),
            "test_frame": test_frame.loc[:, ["example_id", "subclass"] + columns].copy(),
            "feature_columns": columns,
        }
    return feature_sets


def _evaluate_hybrid_subclasses(
    *,
    hidden_root: Path,
    feature_df: pd.DataFrame,
    candidate_layers: list[int],
    token_cloud_candidates: pd.DataFrame,
    probe_cfg: dict[str, Any],
    subclass_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    val_fraction = float(subclass_cfg.get("val_fraction", 0.2))
    multilayer_top_k = max(1, int(subclass_cfg.get("token_cloud_multilayer_top_k", 2)))
    dataset_df = _token_cloud_dataset(feature_df, subclass_cfg=subclass_cfg)
    inner_train_ids, val_ids = _subclass_train_val_ids(dataset_df, val_fraction=val_fraction, seed=seed)
    topology_feature_sets = _build_token_cloud_hybrid_feature_sets(
        dataset_df,
        candidate_df=token_cloud_candidates,
        multilayer_top_k=multilayer_top_k,
    )
    mean_pool_cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]] = {}

    def _cached_mean_pool(method_name: str, layer: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
        key = (method_name, int(layer))
        if key not in mean_pool_cache:
            mean_pool_cache[key] = _load_mean_pool_subclass_feature_frames(
                hidden_root=hidden_root,
                layer=int(layer),
                probe_cfg=probe_cfg,
                subclass_cfg=subclass_cfg,
                seed=seed,
                method=method_name,
            )
        return mean_pool_cache[key]

    rows: list[dict[str, Any]] = []
    for mean_method in ("full_probe", "aen_only"):
        for layer in candidate_layers:
            mean_train, _, mean_columns, mean_extra = _cached_mean_pool(mean_method, int(layer))
            inner_mean = mean_train.loc[mean_train["example_id"].astype(str).isin(inner_train_ids)].copy()
            val_mean = mean_train.loc[mean_train["example_id"].astype(str).isin(val_ids)].copy()
            if inner_mean.empty or val_mean.empty:
                continue
            for topo_key, topo_spec in topology_feature_sets.items():
                topo_train = topo_spec["train_frame"]
                inner_topo = topo_train.loc[topo_train["example_id"].astype(str).isin(inner_train_ids)].copy()
                val_topo = topo_train.loc[topo_train["example_id"].astype(str).isin(val_ids)].copy()
                inner_merged = inner_mean.merge(inner_topo, on=["example_id", "subclass"], how="inner")
                val_merged = val_mean.merge(val_topo, on=["example_id", "subclass"], how="inner")
                if inner_merged.empty or val_merged.empty:
                    continue
                feature_columns = mean_columns + topo_spec["feature_columns"]
                payload = _evaluate_multiclass(
                    x_train=inner_merged.loc[:, feature_columns].to_numpy(dtype=float),
                    y_train=inner_merged["subclass"].astype(str).to_numpy(),
                    x_eval=val_merged.loc[:, feature_columns].to_numpy(dtype=float),
                    y_eval=val_merged["subclass"].astype(str).to_numpy(),
                    max_iter=max_iter,
                    c_value=c_value,
                    seed=seed + 4000 + int(layer),
                )
                row = {
                    "method": f"{mean_method}_plus_{topo_spec['hybrid_label']}",
                    "mean_pool_method": mean_method,
                    "topology_key": topo_key,
                    "layer": int(layer),
                    "selection_signature": topo_spec["selection_signature"],
                    "val_accuracy": float(payload["metrics"]["accuracy"]),
                    "val_macro_f1": float(payload["metrics"]["macro_f1"]),
                    "feature_count": int(len(feature_columns)),
                    "mean_pool_feature_count": int(len(mean_columns)),
                    "topology_feature_count": int(len(topo_spec["feature_columns"])),
                }
                if "aen_k" in mean_extra:
                    row["aen_k"] = int(mean_extra["aen_k"])
                rows.append(row)

    candidate_df = pd.DataFrame(rows).sort_values(
        ["method", "val_macro_f1", "val_accuracy", "layer"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    if candidate_df.empty:
        raise ValueError("No hybrid subclass candidates were produced.")

    final_rows: list[dict[str, Any]] = []
    for method_name in sorted(candidate_df["method"].unique()):
        best = candidate_df.loc[candidate_df["method"].eq(method_name)].iloc[0].to_dict()
        mean_train, mean_test, mean_columns, mean_extra = _cached_mean_pool(str(best["mean_pool_method"]), int(best["layer"]))
        topo_spec = topology_feature_sets[str(best["topology_key"])]
        train_merged = mean_train.merge(topo_spec["train_frame"], on=["example_id", "subclass"], how="inner")
        test_merged = mean_test.merge(topo_spec["test_frame"], on=["example_id", "subclass"], how="inner")
        feature_columns = mean_columns + topo_spec["feature_columns"]
        payload = _evaluate_multiclass(
            x_train=train_merged.loc[:, feature_columns].to_numpy(dtype=float),
            y_train=train_merged["subclass"].astype(str).to_numpy(),
            x_eval=test_merged.loc[:, feature_columns].to_numpy(dtype=float),
            y_eval=test_merged["subclass"].astype(str).to_numpy(),
            max_iter=max_iter,
            c_value=c_value,
            seed=seed + 5000 + int(best["layer"]),
        )
        result = {
            "method": method_name,
            "layer": int(best["layer"]),
            "selection_signature": str(best["selection_signature"]),
            "test_accuracy": float(payload["metrics"]["accuracy"]),
            "test_macro_f1": float(payload["metrics"]["macro_f1"]),
            "feature_count": int(len(feature_columns)),
            "mean_pool_feature_count": int(len(mean_columns)),
            "topology_feature_count": int(len(topo_spec["feature_columns"])),
            "test_confusion_matrix": payload["metrics"]["confusion_matrix"],
            "test_labels": payload["metrics"]["labels"],
        }
        if "aen_k" in mean_extra:
            result["aen_k"] = int(mean_extra["aen_k"])
            result["aen_indices"] = list(mean_extra["aen_indices"])
        final_rows.append(result)
    return candidate_df, pd.DataFrame(final_rows)


def _evaluate_token_cloud_subclasses(
    feature_df: pd.DataFrame,
    *,
    subclass_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    val_fraction = float(subclass_cfg.get("val_fraction", 0.2))
    dataset_df = _token_cloud_dataset(feature_df, subclass_cfg=subclass_cfg)
    train_df = dataset_df.loc[dataset_df["split"].eq("train")].copy()
    test_df = dataset_df.loc[dataset_df["split"].eq("test")].copy()

    inner_train_ids, val_ids = _subclass_train_val_ids(dataset_df, val_fraction=val_fraction, seed=seed)

    rows: list[dict[str, Any]] = []
    for layer in sorted(train_df["layer"].unique()):
        layer_train = train_df.loc[train_df["layer"].eq(layer)].copy()
        inner_train = layer_train.loc[layer_train["example_id"].astype(str).isin(inner_train_ids)].copy()
        val_layer = layer_train.loc[layer_train["example_id"].astype(str).isin(val_ids)].copy()
        if inner_train.empty or val_layer.empty:
            continue
        columns = _topology_feature_columns(layer_train)
        payload = _evaluate_multiclass(
            x_train=inner_train.loc[:, columns].to_numpy(dtype=float),
            y_train=inner_train["subclass"].astype(str).to_numpy(),
            x_eval=val_layer.loc[:, columns].to_numpy(dtype=float),
            y_eval=val_layer["subclass"].astype(str).to_numpy(),
            max_iter=max_iter,
            c_value=c_value,
            seed=seed + int(layer),
        )
        rows.append(
            {
                "method": "token_cloud_single",
                "layer": int(layer),
                "val_accuracy": float(payload["metrics"]["accuracy"]),
                "val_macro_f1": float(payload["metrics"]["macro_f1"]),
                "feature_count": int(len(columns)),
            }
        )
    candidate_df = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy", "layer"], ascending=[False, False, True]).reset_index(drop=True)
    if candidate_df.empty:
        raise ValueError("No token-cloud subclass candidates were produced.")
    best_single = candidate_df.iloc[0].to_dict()

    best_layer = int(best_single["layer"])
    final_train = train_df.loc[train_df["layer"].eq(best_layer)].copy()
    final_test = test_df.loc[test_df["layer"].eq(best_layer)].copy()
    columns = _topology_feature_columns(final_train)
    single_payload = _evaluate_multiclass(
        x_train=final_train.loc[:, columns].to_numpy(dtype=float),
        y_train=final_train["subclass"].astype(str).to_numpy(),
        x_eval=final_test.loc[:, columns].to_numpy(dtype=float),
        y_eval=final_test["subclass"].astype(str).to_numpy(),
        max_iter=max_iter,
        c_value=c_value,
        seed=seed + 1000 + best_layer,
    )
    single_result = {
        "method": "token_cloud_single",
        "layer": best_layer,
        "test_accuracy": float(single_payload["metrics"]["accuracy"]),
        "test_macro_f1": float(single_payload["metrics"]["macro_f1"]),
        "feature_count": int(len(columns)),
        "test_confusion_matrix": single_payload["metrics"]["confusion_matrix"],
        "test_labels": single_payload["metrics"]["labels"],
    }

    top_k = max(1, int(subclass_cfg.get("token_cloud_multilayer_top_k", 2)))
    multi_result = _build_token_cloud_multilayer_result(
        dataset_df,
        selections=_token_cloud_selection_records(candidate_df, top_k=top_k),
        max_iter=max_iter,
        c_value=c_value,
        seed=seed + 2000,
        method="token_cloud_multilayer",
    )
    multi_all_result = _build_token_cloud_multilayer_result(
        dataset_df,
        selections=_token_cloud_selection_records(candidate_df, use_all_layers=True),
        max_iter=max_iter,
        c_value=c_value,
        seed=seed + 3000,
        method="token_cloud_multilayer_all",
    )
    return candidate_df, single_result, multi_result, multi_all_result


def run_clamber_subclass_classification(
    *,
    config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    subclass_cfg = dict(config["clamber_subclass_classification"])
    probe_cfg = dict(config["probe"])
    model_name = str(config["model"]["name"])
    output_root = ensure_dir(Path(subclass_cfg["output_dir"]) / slugify(model_name))
    hidden_root = Path(config["extraction"]["cache_dir"]) / slugify(model_name)
    progress_path = output_root / "run_progress.json"
    mean_pool_enabled = bool(subclass_cfg.get("mean_pool_enabled", True))
    candidate_layers = [int(layer) for layer in subclass_cfg.get("candidate_layers", [0, 4, 8, 12, 14, 16, 20, 24, 28, 31])]
    mean_pool_candidates = pd.DataFrame()
    full_result: dict[str, Any] | None = None
    aen_result: dict[str, Any] | None = None
    best_full: dict[str, Any] | None = None
    best_aen: dict[str, Any] | None = None
    if mean_pool_enabled:
        LOGGER.info("Stage=mean_pool_search model=%s hidden_root=%s", model_name, hidden_root)
        _write_progress(progress_path, stage="mean_pool_search", model_name=model_name, extra={"hidden_root": str(hidden_root)})
        mean_pool_candidates, best_full, best_aen = _evaluate_full_and_aen_layers(
            hidden_root=hidden_root,
            candidate_layers=candidate_layers,
            probe_cfg=probe_cfg,
            subclass_cfg=subclass_cfg,
            seed=seed,
        )
        write_parquet(mean_pool_candidates, output_root / "clamber_subclass_mean_pool_candidates.parquet")
        LOGGER.info(
            "Stage=mean_pool_search_complete model=%s best_full_layer=%s best_aen_layer=%s",
            model_name,
            int(best_full["layer"]),
            int(best_aen["layer"]),
        )
        full_result = _finalize_mean_pool_result(
            hidden_root=hidden_root,
            layer=int(best_full["layer"]),
            probe_cfg=probe_cfg,
            subclass_cfg=subclass_cfg,
            seed=seed,
            method="full_probe",
        )
        aen_result = _finalize_mean_pool_result(
            hidden_root=hidden_root,
            layer=int(best_aen["layer"]),
            probe_cfg=probe_cfg,
            subclass_cfg=subclass_cfg,
            seed=seed,
            method="aen_only",
        )
        write_parquet(pd.DataFrame([full_result, aen_result]), output_root / "clamber_subclass_mean_pool_final_metrics.parquet")
    else:
        LOGGER.info("Stage=mean_pool_search_skipped model=%s reason=config_disabled", model_name)
        _write_progress(progress_path, stage="mean_pool_search_skipped", model_name=model_name)

    token_cloud_cfg = dict(config["token_cloud_topology_classifier"])
    token_cloud_cfg["datasets"] = ["clamber"]
    token_cloud_cfg["output_dir"] = str(output_root / "token_cloud_binary")
    subclass_cfg["token_cloud_multilayer_top_k"] = int(
        subclass_cfg.get("token_cloud_multilayer_top_k", token_cloud_cfg.get("multilayer_top_k", 2))
    )
    token_feature_path = output_root / "clamber_token_cloud_all_layer_features.parquet"
    shared_forward_cache_path = (
        Path(token_cloud_cfg["output_dir"]).resolve().parent
        / "token_cloud_forward_cache"
        / slugify(model_name)
        / "clamber_token_cloud_forward_cache.joblib"
    )
    subclass_cfg["token_cloud_forward_cache_path"] = str(shared_forward_cache_path)
    subclass_cfg["token_cloud_feature_cache_path"] = str(token_feature_path)
    LOGGER.info("Stage=token_cloud_feature_build model=%s layers=%s", model_name, subclass_cfg.get("token_cloud_candidate_layers", token_cloud_cfg.get("candidate_layers")))
    _write_progress(
        progress_path,
        stage="token_cloud_feature_build",
        model_name=model_name,
        extra=(
            {
                "best_full_layer": int(best_full["layer"]),
                "best_aen_layer": int(best_aen["layer"]),
            }
            if best_full is not None and best_aen is not None
            else None
        ),
    )
    model_name_tc, token_cloud_feature_df, reused_token_cloud_features = _build_clamber_token_cloud_features(
        config=config,
        classifier_config=token_cloud_cfg,
        subclass_cfg=subclass_cfg,
        seed=seed,
    )
    if reused_token_cloud_features:
        LOGGER.info(
            "Stage=token_cloud_feature_reuse model=%s path=%s rows=%s columns=%s",
            model_name,
            token_feature_path,
            len(token_cloud_feature_df),
            len(token_cloud_feature_df.columns),
        )
    else:
        LOGGER.info(
            "Stage=token_cloud_feature_build_complete model=%s rows=%s columns=%s",
            model_name,
            len(token_cloud_feature_df),
            len(token_cloud_feature_df.columns),
        )
    _write_progress(
        progress_path,
        stage="token_cloud_binary",
        model_name=model_name,
        extra={
            "token_cloud_feature_path": str(token_feature_path),
            "token_cloud_rows": int(len(token_cloud_feature_df)),
            "token_cloud_feature_cached": bool(reused_token_cloud_features),
        },
    )
    binary_outputs = run_token_cloud_topology_classifier_from_features(
        model_name=model_name_tc,
        feature_df=token_cloud_feature_df,
        classifier_config=token_cloud_cfg,
        seed=seed,
    )
    LOGGER.info("Stage=token_cloud_binary_complete model=%s report=%s", model_name, binary_outputs["report_path"])
    _write_progress(
        progress_path,
        stage="token_cloud_subclass_eval",
        model_name=model_name,
        extra={"binary_report_path": str(binary_outputs["report_path"])},
    )
    token_cloud_candidates, token_cloud_single, token_cloud_multi, token_cloud_multi_all = _evaluate_token_cloud_subclasses(
        token_cloud_feature_df,
        subclass_cfg=subclass_cfg,
        seed=seed,
    )
    write_parquet(token_cloud_candidates, output_root / "clamber_subclass_token_cloud_candidates.parquet")
    write_parquet(
        pd.DataFrame([token_cloud_single, token_cloud_multi, token_cloud_multi_all]),
        output_root / "clamber_subclass_token_cloud_final_metrics.parquet",
    )
    LOGGER.info(
        "Stage=token_cloud_subclass_eval_complete model=%s single_f1=%.4f multilayer_f1=%.4f multilayer_all_f1=%.4f",
        model_name,
        float(token_cloud_single["test_macro_f1"]),
        float(token_cloud_multi["test_macro_f1"]),
        float(token_cloud_multi_all["test_macro_f1"]),
    )

    hybrid_enabled = bool(subclass_cfg.get("hybrid_enabled", True)) and mean_pool_enabled
    hybrid_candidates = pd.DataFrame()
    hybrid_final = pd.DataFrame()
    if hybrid_enabled:
        _write_progress(
            progress_path,
            stage="hybrid_subclass_eval",
            model_name=model_name,
            extra={
                "token_cloud_single_f1": float(token_cloud_single["test_macro_f1"]),
                "token_cloud_multilayer_f1": float(token_cloud_multi["test_macro_f1"]),
                "token_cloud_multilayer_all_f1": float(token_cloud_multi_all["test_macro_f1"]),
            },
        )
        hybrid_candidates, hybrid_final = _evaluate_hybrid_subclasses(
            hidden_root=hidden_root,
            feature_df=token_cloud_feature_df,
            candidate_layers=candidate_layers,
            token_cloud_candidates=token_cloud_candidates,
            probe_cfg=probe_cfg,
            subclass_cfg=subclass_cfg,
            seed=seed,
        )
        write_parquet(hybrid_candidates, output_root / "clamber_subclass_hybrid_candidates.parquet")
        write_parquet(hybrid_final, output_root / "clamber_subclass_hybrid_final_metrics.parquet")
        LOGGER.info(
            "Stage=hybrid_subclass_eval_complete model=%s best_method=%s best_f1=%.4f",
            model_name,
            str(hybrid_final.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False]).iloc[0]["method"]),
            float(hybrid_final.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False]).iloc[0]["test_macro_f1"]),
        )

    candidate_path = output_root / "clamber_subclass_candidate_metrics.parquet"
    final_path = output_root / "clamber_subclass_final_metrics.parquet"
    report_path = output_root / "clamber_subclass_summary.md"
    metadata_path = output_root / "clamber_subclass_metadata.json"

    candidate_parts = [token_cloud_candidates]
    if not mean_pool_candidates.empty:
        candidate_parts.insert(0, mean_pool_candidates)
    if not hybrid_candidates.empty:
        candidate_parts.append(hybrid_candidates)
    candidate_out = pd.concat(
        candidate_parts,
        ignore_index=True,
        sort=False,
    ).sort_values(["method", "val_macro_f1", "val_accuracy", "layer"], ascending=[True, False, False, True]).reset_index(drop=True)
    final_rows: list[dict[str, Any]] = [token_cloud_single, token_cloud_multi, token_cloud_multi_all]
    if full_result is not None:
        final_rows.insert(0, full_result)
    if aen_result is not None:
        insert_at = 1 if full_result is not None else 0
        final_rows.insert(insert_at, aen_result)
    final_out = pd.DataFrame(final_rows)
    if not hybrid_final.empty:
        final_out = pd.concat([final_out, hybrid_final], ignore_index=True, sort=False)
    final_out = final_out.sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False]).reset_index(drop=True)
    for column in ["test_confusion_matrix", "test_labels", "aen_indices"]:
        if column in final_out.columns:
            final_out[column] = final_out[column].apply(_normalize_object_column)

    write_parquet(candidate_out, candidate_path)
    write_parquet(final_out, final_path)

    lines = [
        "# CLAMBER Subclass Classification",
        "",
        f"- Model: `{model_name}`",
        f"- Created at: `{utc_now_iso()}`",
        f"- Setting: ambiguous-only `{bool(subclass_cfg.get('ambiguous_only', False))}`, include `none` `{bool(subclass_cfg.get('include_none_class', True))}`",
        "",
        "## Final Test Results",
        "",
    ]
    for row in final_out.to_dict(orient="records"):
        line = (
            f"- `{row['method']}`: macro-F1 `{float(row['test_macro_f1']):.4f}`, "
            f"accuracy `{float(row['test_accuracy']):.4f}`"
        )
        if int(row.get("layer", -1)) >= 0:
            line += f", layer `{int(row['layer'])}`"
        if "selection_signature" in row and not pd.isna(row["selection_signature"]):
            line += f", layers `{row['selection_signature']}`"
        if not pd.isna(row.get("feature_count", np.nan)):
            line += f", features `{int(row['feature_count'])}`"
        if not pd.isna(row.get("aen_k", np.nan)):
            line += f", AEN k `{int(row['aen_k'])}`"
        lines.append(line)
    lines.extend(
        [
            "",
            "## Binary Token-Cloud Output",
            "",
            f"- Binary token-cloud artifacts: `{binary_outputs['report_path']}`",
            "",
        ]
    )
    write_markdown(report_path, "\n".join(lines) + "\n")
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "created_at": utc_now_iso(),
            "candidate_metrics_path": str(candidate_path),
            "final_metrics_path": str(final_path),
            "hybrid_candidate_metrics_path": str(output_root / "clamber_subclass_hybrid_candidates.parquet") if hybrid_enabled else None,
            "hybrid_final_metrics_path": str(output_root / "clamber_subclass_hybrid_final_metrics.parquet") if hybrid_enabled else None,
            "token_cloud_feature_path": str(token_feature_path),
            "binary_token_cloud_outputs": binary_outputs,
        },
    )
    _write_progress(
        progress_path,
        stage="complete",
        model_name=model_name,
        extra={
            "candidate_metrics_path": str(candidate_path),
            "final_metrics_path": str(final_path),
            "report_path": str(report_path),
        },
    )
    return {
        "candidate_metrics_path": str(candidate_path),
        "final_metrics_path": str(final_path),
        "token_cloud_feature_path": str(token_feature_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }
