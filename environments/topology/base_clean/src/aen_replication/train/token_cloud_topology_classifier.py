"""Token-cloud topology classifier from per-token hidden states.

This stage treats the token embeddings for each question at a given layer as a
point cloud, computes persistent-homology descriptors for that question-level
token cloud, and trains ambiguity classifiers from those topology features.
"""

from __future__ import annotations

import heapq
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.generation import render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import (
    _compute_diagrams,
    _diagram_descriptors,
    _extract_model_signal,
    _fit_classifier,
    _group_train_val_split,
    _persistence_image_features,
    _predict_scores,
    _safe_bottleneck,
    _safe_wasserstein,
    _select_multilayer_candidates,
    _selection_order,
    _stacked_summary_features,
    _transform_with_scaler,
)
from aen_replication.utils.io_utils import ensure_dir, read_json, slugify, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)

BASE_KEY_COLUMNS = ["example_id", "pair_id", "dataset", "split", "label_ambiguous"]
TOPOLOGY_PREFIXES = ("h0_", "h1_")
TOPOLOGY_EXTRA_COLUMNS = ("token_count",)
TOKEN_CLOUD_FEATURE_SCHEMA_VERSION = 2
LABEL_NAME_MAP = {1: "Ambiguous", 0: "Clear"}
LABEL_COLOR_MAP = {1: "#c44e52", 0: "#4c72b0"}
PLOT_FEATURE_CANDIDATES = [
    "h0_feature_count",
    "h0_total_persistence_norm",
    "h0_mean_persistence",
    "h0_betti_curve_auc_norm",
    "h0_wasserstein_to_clear",
    "h0_wasserstein_to_ambiguous",
    "h1_feature_count",
    "h1_total_persistence_norm",
    "h1_mean_persistence",
    "h1_betti_curve_auc_norm",
    "h1_wasserstein_to_clear",
    "h1_wasserstein_to_ambiguous",
]
TRAJECTORY_FEATURES = [
    ("h0_feature_count", "H0 feature count"),
    ("h0_total_persistence_norm", "H0 total persistence"),
    ("h1_feature_count", "H1 feature count"),
    ("h1_total_persistence_norm", "H1 total persistence"),
]

_BINARY_DISTANCE_LABELS = ((0, "clear"), (1, "ambiguous"))
_KNN_QUERY_IDS: list[str] = []
_KNN_QUERY_H0: list[np.ndarray] = []
_KNN_QUERY_H1: list[np.ndarray] = []
_KNN_TRAIN_BY_LABEL: dict[str, dict[str, list[Any]]] = {}
_KNN_LABEL_NAMES: list[str] = []
_KNN_DISTANCE_K = 0


def _token_cloud_feature_cache_signature(
    *,
    model_name: str,
    dataset_paths: list[Path],
    layers: list[int],
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": TOKEN_CLOUD_FEATURE_SCHEMA_VERSION,
        "model_name": model_name,
        "dataset_inputs": [
            {
                "path": str(path.resolve()),
                "mtime_ns": int(path.stat().st_mtime_ns) if path.exists() else None,
            }
            for path in dataset_paths
        ],
        "layers": [int(layer) for layer in layers],
        "seed": int(seed),
        "settings": {
            "text_column": str(classifier_config.get("text_column", "text")),
            "use_chat_template": bool(classifier_config.get("use_chat_template", False)),
            "system_prompt": classifier_config.get("system_prompt"),
            "drop_special_tokens": bool(classifier_config.get("drop_special_tokens", True)),
            "use_pca": bool(classifier_config.get("use_pca", True)),
            "pca_components": classifier_config.get("pca_components"),
            "pca_whiten": bool(classifier_config.get("pca_whiten", False)),
            "topology_components": classifier_config.get("topology_components"),
            "prototype_token_cap": classifier_config.get("prototype_token_cap"),
            "distance_metric": str(classifier_config.get("distance_metric", "euclidean")),
            "distance_feature_mode": str(classifier_config.get("distance_feature_mode", "knn_class")),
            "distance_feature_k": int(classifier_config.get("distance_feature_k", 8)),
            "betti_grid_size": int(classifier_config.get("betti_grid_size", 24)),
            "persistence_image_grid_side": int(classifier_config.get("persistence_image_grid_side", 3)),
            "maxdim": int(classifier_config.get("maxdim", 1)),
            "coeff": int(classifier_config.get("coeff", 2)),
            "max_length": int(classifier_config.get("max_length", 64)),
        },
    }


def _token_cloud_feature_cache_metadata_path(feature_path: Path) -> Path:
    return feature_path.with_name(f"{feature_path.stem}.metadata.json")


def _token_cloud_forward_cache_signature(
    *,
    model_name: str,
    dataset_paths: list[Path],
    layers: list[int],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "dataset_inputs": [
            {
                "path": str(path.resolve()),
                "mtime_ns": int(path.stat().st_mtime_ns) if path.exists() else None,
            }
            for path in dataset_paths
        ],
        "layers": [int(layer) for layer in layers],
        "seed": int(seed),
        "settings": {
            "text_column": str(config.get("text_column", "text")),
            "use_chat_template": bool(config.get("use_chat_template", False)),
            "system_prompt": config.get("system_prompt"),
            "drop_special_tokens": bool(config.get("drop_special_tokens", True)),
            "max_length": int(config.get("max_length", 96)),
            "use_pca": bool(config.get("use_pca", True)),
            "pca_components": config.get("pca_components"),
            "pca_whiten": bool(config.get("pca_whiten", False)),
            "pca_fit_token_cap": int(config.get("pca_fit_token_cap", 24000)),
            "topology_components": config.get("topology_components"),
        },
    }


def _token_cloud_forward_cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}.metadata.json")


def load_cached_token_cloud_forward_frame(
    *,
    cache_path: Path,
    signature: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None] | None:
    metadata_path = _token_cloud_forward_cache_metadata_path(cache_path)
    if not cache_path.exists():
        return None
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
        except Exception:
            LOGGER.exception("Failed to read token-cloud forward cache metadata: %s", metadata_path)
            return None
        if metadata.get("signature") != signature:
            LOGGER.info("Token-cloud forward cache signature mismatch, rebuilding: %s", cache_path)
            return None
    else:
        LOGGER.warning("Reusing unversioned token-cloud forward cache without signature check: %s", cache_path)

    payload = joblib.load(cache_path)
    if isinstance(payload, pd.DataFrame):
        return payload, None
    if isinstance(payload, dict) and "cloud_df" in payload:
        return pd.DataFrame(payload["cloud_df"]), payload.get("pca_variance")
    raise ValueError(f"Unsupported token-cloud forward cache payload: {cache_path}")


def save_cached_token_cloud_forward_frame(
    *,
    cloud_df: pd.DataFrame,
    cache_path: Path,
    signature: dict[str, Any],
    pca_variance: dict[str, Any] | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "cloud_df": cloud_df,
            "pca_variance": pca_variance,
        },
        cache_path,
    )
    write_json(
        _token_cloud_forward_cache_metadata_path(cache_path),
        {
            "created_at": utc_now_iso(),
            "rows": int(len(cloud_df)),
            "signature": signature,
        },
    )


def load_cached_token_cloud_feature_frame(
    *,
    feature_path: Path,
    signature: dict[str, Any],
) -> pd.DataFrame | None:
    metadata_path = _token_cloud_feature_cache_metadata_path(feature_path)
    if not feature_path.exists():
        return None
    if not metadata_path.exists():
        LOGGER.warning("Found unversioned token-cloud feature cache, validating required columns: %s", feature_path)
        cached = pd.read_parquet(feature_path)
        required_columns = {
            "token_count",
            "h0_mean_birth",
            "h0_std_birth",
            "h0_mean_death",
            "h0_std_death",
            "h0_top1_persistence",
            "h0_top3_persistence_sum",
            "h0_top5_persistence_fraction",
            "h0_persistence_gini",
            "h1_mean_birth",
            "h1_std_birth",
            "h1_mean_death",
            "h1_std_death",
            "h1_top1_persistence",
            "h1_top3_persistence_sum",
            "h1_top5_persistence_fraction",
            "h1_persistence_gini",
        }
        if required_columns.issubset(set(cached.columns)):
            return cached
        LOGGER.info("Unversioned token-cloud feature cache is missing new columns, rebuilding: %s", feature_path)
        return None
    try:
        metadata = read_json(metadata_path)
    except Exception:
        LOGGER.exception("Failed to read token-cloud feature cache metadata: %s", metadata_path)
        return None
    if metadata.get("signature") != signature:
        LOGGER.info("Token-cloud feature cache signature mismatch, rebuilding: %s", feature_path)
        return None
    LOGGER.info("Reusing cached token-cloud features: %s", feature_path)
    return pd.read_parquet(feature_path)


def save_cached_token_cloud_feature_frame(
    *,
    feature_df: pd.DataFrame,
    feature_path: Path,
    signature: dict[str, Any],
) -> None:
    write_parquet(feature_df, feature_path)
    write_json(
        _token_cloud_feature_cache_metadata_path(feature_path),
        {
            "created_at": utc_now_iso(),
            "rows": int(len(feature_df)),
            "columns": list(feature_df.columns),
            "signature": signature,
        },
    )


def _iter_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def _distance_feature_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("distance_feature_mode", "knn_class")).strip().lower()
    if mode not in {"knn_class", "knn_subclass", "prototype", "none"}:
        raise ValueError(f"Unsupported token-cloud distance feature mode: {mode}")
    return mode


def _distance_feature_k(config: dict[str, Any]) -> int:
    return max(1, int(config.get("distance_feature_k", 8)))


def _distance_feature_chunk_size(config: dict[str, Any]) -> int:
    return max(1, int(config.get("distance_feature_chunk_size", 24)))


def _resolve_candidate_layers(total_layers: int, config: dict[str, Any]) -> list[int]:
    configured = config.get("candidate_layers", "auto")
    available_layers = list(range(total_layers))
    if isinstance(configured, list):
        selected = [int(layer) for layer in configured if 0 <= int(layer) < total_layers]
        if selected:
            return sorted(set(selected))
    strategy = str(config.get("layer_selection_strategy", "evenly_spaced"))
    if configured == "all" or strategy == "all":
        return available_layers
    max_layers = int(config.get("max_candidate_layers", len(available_layers)))
    max_layers = max(1, min(max_layers, len(available_layers)))
    if max_layers >= len(available_layers):
        return available_layers
    indices = np.linspace(0, len(available_layers) - 1, num=max_layers, dtype=int)
    return [available_layers[index] for index in sorted(set(indices.tolist()))]


def _prepare_prompt_frame(
    df: pd.DataFrame,
    *,
    bundle: HFModelBundle,
    text_column: str,
    use_chat_template: bool,
    system_prompt: str | None,
) -> tuple[pd.DataFrame, str]:
    if not use_chat_template and not system_prompt:
        return df.copy(), text_column
    prompt_df = df.copy()
    prompt_df["_rendered_text"] = render_prompts(
        bundle=bundle,
        prompt_texts=df[text_column].astype(str).tolist(),
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        add_generation_prompt=False,
    )
    return prompt_df, "_rendered_text"


def _valid_token_mask(
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    *,
    special_ids: set[int],
    drop_special_tokens: bool,
) -> torch.Tensor:
    valid = attention_mask_row.bool().clone()
    if drop_special_tokens and special_ids:
        special_mask = torch.zeros_like(valid)
        for special_id in special_ids:
            special_mask |= input_ids_row.eq(int(special_id))
        valid &= ~special_mask
    if int(valid.sum().item()) == 0:
        valid = attention_mask_row.bool()
    return valid


def _extract_train_token_matrices(
    *,
    bundle: HFModelBundle,
    train_df: pd.DataFrame,
    text_column: str,
    layers: list[int],
    config: dict[str, Any],
) -> dict[int, np.ndarray]:
    encoder = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    batch_size = int(config.get("batch_size", 4))
    max_length = int(config.get("max_length", 96))
    drop_special_tokens = bool(config.get("drop_special_tokens", True))
    special_ids = set(int(token_id) for token_id in getattr(encoder, "all_special_ids", []) if token_id is not None)
    token_cap = int(config.get("pca_fit_token_cap", 24000))

    per_layer_tokens: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for batch_df in tqdm(_iter_batches(train_df, batch_size), desc="token_pca_fit", leave=False):
        encoded = encoder(
            batch_df[text_column].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"].to(device)
        input_ids = encoded["input_ids"].to(device)
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states for token-cloud extraction.")
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        for layer in layers:
            layer_output = hidden_states[layer + 1].detach().float().cpu()
            batch_tokens: list[np.ndarray] = []
            for row_index in range(layer_output.shape[0]):
                valid = _valid_token_mask(
                    input_ids_cpu[row_index],
                    attention_mask_cpu[row_index],
                    special_ids=special_ids,
                    drop_special_tokens=drop_special_tokens,
                )
                if int(valid.sum().item()) == 0:
                    continue
                batch_tokens.append(layer_output[row_index][valid].numpy())
            if batch_tokens:
                per_layer_tokens[layer].append(np.vstack(batch_tokens))
        del outputs, hidden_states, model_inputs, input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fitted: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(int(config.get("_seed", 0)))
    for layer, chunks in per_layer_tokens.items():
        if not chunks:
            raise ValueError(f"No valid tokens collected for PCA fitting at layer {layer}.")
        matrix = np.vstack(chunks).astype(np.float32, copy=False)
        if len(matrix) > token_cap:
            selected = np.sort(rng.choice(len(matrix), size=token_cap, replace=False))
            matrix = matrix[selected]
        fitted[layer] = matrix
    return fitted


def _fit_layer_reducers(
    token_matrices: dict[int, np.ndarray],
    *,
    config: dict[str, Any],
    seed: int,
) -> dict[int, PCA | None]:
    reducers: dict[int, PCA | None] = {}
    use_pca = bool(config.get("use_pca", True))
    if not use_pca:
        for layer in token_matrices:
            reducers[layer] = None
        return reducers
    for layer, matrix in token_matrices.items():
        n_components = min(int(config.get("pca_components", 8)), matrix.shape[1], max(1, matrix.shape[0] - 1))
        reducer = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed + layer,
            whiten=bool(config.get("pca_whiten", False)),
        )
        reducer.fit(matrix)
        reducers[layer] = reducer
    return reducers


def _extract_reduced_clouds(
    *,
    bundle: HFModelBundle,
    df: pd.DataFrame,
    text_column: str,
    layers: list[int],
    reducers: dict[int, PCA | None],
    config: dict[str, Any],
) -> pd.DataFrame:
    encoder = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    batch_size = int(config.get("batch_size", 4))
    max_length = int(config.get("max_length", 96))
    use_pca = bool(config.get("use_pca", True))
    topology_dim_value = config.get("topology_components", config.get("pca_components", 8))
    topology_dim = int(topology_dim_value) if topology_dim_value is not None else None
    drop_special_tokens = bool(config.get("drop_special_tokens", True))
    special_ids = set(int(token_id) for token_id in getattr(encoder, "all_special_ids", []) if token_id is not None)

    rows: list[dict[str, Any]] = []
    for batch_df in tqdm(_iter_batches(df, batch_size), desc="token_cloud_extract", leave=False):
        encoded = encoder(
            batch_df[text_column].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"].to(device)
        input_ids = encoded["input_ids"].to(device)
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states for token-cloud extraction.")
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        batch_rows = batch_df.reset_index(drop=True).to_dict(orient="records")
        valid_masks = [
            _valid_token_mask(
                input_ids_cpu[row_index],
                attention_mask_cpu[row_index],
                special_ids=special_ids,
                drop_special_tokens=drop_special_tokens,
            )
            for row_index in range(len(batch_rows))
        ]
        for layer in layers:
            reducer = reducers.get(layer)
            layer_output = hidden_states[layer + 1].detach().float().cpu()
            token_chunks: list[np.ndarray] = []
            chunk_rows: list[tuple[dict[str, Any], int]] = []
            for row_index, row in enumerate(batch_rows):
                token_vectors = layer_output[row_index][valid_masks[row_index]].numpy()
                if token_vectors.size == 0:
                    continue
                token_chunks.append(token_vectors)
                chunk_rows.append((row, len(token_vectors)))
            if not token_chunks:
                continue
            if use_pca:
                if reducer is None:
                    raise RuntimeError(f"Missing PCA reducer for layer {layer}.")
                token_matrix = np.vstack(token_chunks)
                reduced_dim = min(int(topology_dim or token_matrix.shape[1]), int(reducer.n_components_))
                reduced_matrix = reducer.transform(token_matrix)[:, :reduced_dim].astype(np.float32, copy=False)
                offset = 0
                for row, token_count in chunk_rows:
                    reduced = reduced_matrix[offset : offset + token_count]
                    offset += token_count
                    rows.append(
                        {
                            "example_id": str(row["example_id"]),
                            "pair_id": str(row["pair_id"]),
                            "dataset": str(row["dataset"]),
                            "split": str(row["split"]),
                            "label_ambiguous": int(row["label_ambiguous"]),
                            "layer": int(layer),
                            "token_count": int(len(reduced)),
                            "cloud": reduced,
                        }
                    )
            else:
                for token_vectors, (row, _token_count) in zip(token_chunks, chunk_rows, strict=True):
                    reduced = token_vectors.astype(np.float32, copy=False)
                    rows.append(
                        {
                            "example_id": str(row["example_id"]),
                            "pair_id": str(row["pair_id"]),
                            "dataset": str(row["dataset"]),
                            "split": str(row["split"]),
                            "label_ambiguous": int(row["label_ambiguous"]),
                            "layer": int(layer),
                            "token_count": int(len(reduced)),
                            "cloud": reduced,
                        }
                    )
        del outputs, hidden_states, model_inputs, input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _prototype_diagrams_from_clouds(
    cloud_df: pd.DataFrame,
    *,
    layers: list[int],
    config: dict[str, Any],
    seed: int,
) -> dict[tuple[int, str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    prototype_cap = int(config.get("prototype_token_cap", 128))
    distance_metric = str(config.get("distance_metric", "euclidean"))
    maxdim = int(config.get("maxdim", 1))
    coeff = int(config.get("coeff", 2))
    prototypes: dict[tuple[int, str], list[np.ndarray]] = {}
    train_df = cloud_df.loc[cloud_df["split"].eq("train")].copy()
    for layer in layers:
        layer_df = train_df.loc[train_df["layer"].eq(layer)]
        for label_value, label_name in [(0, "clear"), (1, "ambiguous")]:
            label_df = layer_df.loc[layer_df["label_ambiguous"].eq(label_value)]
            if label_df.empty:
                prototypes[(layer, label_name)] = [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
                continue
            token_matrix = np.vstack(label_df["cloud"].to_list()).astype(np.float32, copy=False)
            if len(token_matrix) > prototype_cap:
                selected = np.sort(rng.choice(len(token_matrix), size=prototype_cap, replace=False))
                token_matrix = token_matrix[selected]
            prototypes[(layer, label_name)] = _compute_diagrams(
                token_matrix,
                maxdim=maxdim,
                coeff=coeff,
                distance_metric=distance_metric,
            )
    return prototypes


def _cloud_base_feature_row(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    cloud = np.asarray(row["cloud"], dtype=float)
    distance_metric = str(config.get("distance_metric", "euclidean"))
    diagrams = _compute_diagrams(
        cloud,
        maxdim=int(config.get("maxdim", 1)),
        coeff=int(config.get("coeff", 2)),
        distance_metric=distance_metric,
    )
    feature_row: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "layer": int(row["layer"]),
        "token_count": int(row["token_count"]),
    }
    if "subclass" in row and pd.notna(row["subclass"]):
        feature_row["subclass"] = str(row["subclass"])
    grid_size = int(config.get("betti_grid_size", 32))
    pimg_side = int(config.get("persistence_image_grid_side", 4))
    for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
        diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
        feature_row[f"{prefix}_diagram"] = diagram
        feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=grid_size))
        feature_row.update(_persistence_image_features(diagram, prefix=prefix, grid_side=pimg_side))
    return feature_row


def _add_prototype_distance_features(
    feature_df: pd.DataFrame,
    *,
    prototype_map: dict[tuple[int, str], list[np.ndarray]],
) -> pd.DataFrame:
    output = feature_df.copy()
    for row_index, row in output.iterrows():
        layer = int(row["layer"])
        for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
            diagram = row[f"{prefix}_diagram"]
            clear_proto = prototype_map[(layer, "clear")][homology_dim]
            ambig_proto = prototype_map[(layer, "ambiguous")][homology_dim]
            output.at[row_index, f"{prefix}_wasserstein_to_clear"] = _safe_wasserstein(diagram, clear_proto)
            output.at[row_index, f"{prefix}_wasserstein_to_ambiguous"] = _safe_wasserstein(diagram, ambig_proto)
            output.at[row_index, f"{prefix}_bottleneck_to_clear"] = _safe_bottleneck(diagram, clear_proto)
            output.at[row_index, f"{prefix}_bottleneck_to_ambiguous"] = _safe_bottleneck(diagram, ambig_proto)
    return output


def _push_smallest_distance(heap: list[float], value: float, *, k: int) -> None:
    if len(heap) < k:
        heapq.heappush(heap, -float(value))
        return
    if value < -heap[0]:
        heapq.heapreplace(heap, -float(value))


def _heap_mean(heap: list[float]) -> float:
    if not heap:
        return 0.0
    return float(-sum(heap) / len(heap))


def _init_binary_knn_distance_worker(
    query_ids: list[str],
    query_h0: list[np.ndarray],
    query_h1: list[np.ndarray],
    train_by_label: dict[str, dict[str, list[Any]]],
    distance_k: int,
) -> None:
    global _KNN_QUERY_IDS, _KNN_QUERY_H0, _KNN_QUERY_H1, _KNN_TRAIN_BY_LABEL, _KNN_LABEL_NAMES, _KNN_DISTANCE_K
    _KNN_QUERY_IDS = query_ids
    _KNN_QUERY_H0 = query_h0
    _KNN_QUERY_H1 = query_h1
    _KNN_TRAIN_BY_LABEL = train_by_label
    _KNN_LABEL_NAMES = [label_name for _, label_name in _BINARY_DISTANCE_LABELS]
    _KNN_DISTANCE_K = distance_k


def _binary_knn_distance_chunk(
    start: int,
    stop: int,
) -> tuple[int, dict[str, np.ndarray]]:
    columns = [
        "h0_wasserstein_to_clear",
        "h0_wasserstein_to_ambiguous",
        "h0_bottleneck_to_clear",
        "h0_bottleneck_to_ambiguous",
        "h1_wasserstein_to_clear",
        "h1_wasserstein_to_ambiguous",
        "h1_bottleneck_to_clear",
        "h1_bottleneck_to_ambiguous",
    ]
    outputs = {column: np.zeros(stop - start, dtype=np.float32) for column in columns}
    for local_index, query_index in enumerate(range(start, stop)):
        query_id = _KNN_QUERY_IDS[query_index]
        query_h0 = _KNN_QUERY_H0[query_index]
        query_h1 = _KNN_QUERY_H1[query_index]
        for prefix, query_diagram in [("h0", query_h0), ("h1", query_h1)]:
            for _, label_name in _BINARY_DISTANCE_LABELS:
                train_ids = _KNN_TRAIN_BY_LABEL[label_name]["ids"]
                train_diagrams = _KNN_TRAIN_BY_LABEL[label_name][prefix]
                wasserstein_heap: list[float] = []
                bottleneck_heap: list[float] = []
                for train_id, train_diagram in zip(train_ids, train_diagrams):
                    if train_id == query_id:
                        continue
                    _push_smallest_distance(
                        wasserstein_heap,
                        _safe_wasserstein(query_diagram, train_diagram),
                        k=_KNN_DISTANCE_K,
                    )
                    _push_smallest_distance(
                        bottleneck_heap,
                        _safe_bottleneck(query_diagram, train_diagram),
                        k=_KNN_DISTANCE_K,
                    )
                outputs[f"{prefix}_wasserstein_to_{label_name}"][local_index] = np.float32(_heap_mean(wasserstein_heap))
                outputs[f"{prefix}_bottleneck_to_{label_name}"][local_index] = np.float32(_heap_mean(bottleneck_heap))
    return start, outputs


def _distance_feature_column(prefix: str, metric: str, label_name: str) -> str:
    return f"{prefix}_{metric}_to_{slugify(label_name)}"


def _init_labeled_knn_distance_worker(
    query_ids: list[str],
    query_h0: list[np.ndarray],
    query_h1: list[np.ndarray],
    train_by_label: dict[str, dict[str, list[Any]]],
    label_names: list[str],
    distance_k: int,
) -> None:
    global _KNN_QUERY_IDS, _KNN_QUERY_H0, _KNN_QUERY_H1, _KNN_TRAIN_BY_LABEL, _KNN_LABEL_NAMES, _KNN_DISTANCE_K
    _KNN_QUERY_IDS = query_ids
    _KNN_QUERY_H0 = query_h0
    _KNN_QUERY_H1 = query_h1
    _KNN_TRAIN_BY_LABEL = train_by_label
    _KNN_LABEL_NAMES = label_names
    _KNN_DISTANCE_K = distance_k


def _labeled_knn_distance_chunk(
    start: int,
    stop: int,
) -> tuple[int, dict[str, np.ndarray]]:
    columns = [
        _distance_feature_column(prefix, metric, label_name)
        for prefix in ("h0", "h1")
        for metric in ("wasserstein", "bottleneck")
        for label_name in _KNN_LABEL_NAMES
    ]
    outputs = {column: np.zeros(stop - start, dtype=np.float32) for column in columns}
    for local_index, query_index in enumerate(range(start, stop)):
        query_id = _KNN_QUERY_IDS[query_index]
        query_h0 = _KNN_QUERY_H0[query_index]
        query_h1 = _KNN_QUERY_H1[query_index]
        for prefix, query_diagram in [("h0", query_h0), ("h1", query_h1)]:
            for label_name in _KNN_LABEL_NAMES:
                train_ids = _KNN_TRAIN_BY_LABEL[label_name]["ids"]
                train_diagrams = _KNN_TRAIN_BY_LABEL[label_name][prefix]
                wasserstein_heap: list[float] = []
                bottleneck_heap: list[float] = []
                for train_id, train_diagram in zip(train_ids, train_diagrams):
                    if train_id == query_id:
                        continue
                    _push_smallest_distance(
                        wasserstein_heap,
                        _safe_wasserstein(query_diagram, train_diagram),
                        k=_KNN_DISTANCE_K,
                    )
                    _push_smallest_distance(
                        bottleneck_heap,
                        _safe_bottleneck(query_diagram, train_diagram),
                        k=_KNN_DISTANCE_K,
                    )
                outputs[_distance_feature_column(prefix, "wasserstein", label_name)][local_index] = np.float32(
                    _heap_mean(wasserstein_heap)
                )
                outputs[_distance_feature_column(prefix, "bottleneck", label_name)][local_index] = np.float32(
                    _heap_mean(bottleneck_heap)
                )
    return start, outputs


def _add_knn_class_distance_features(
    feature_df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df.copy()

    parallel_jobs = max(1, int(config.get("parallel_jobs", 1)))
    distance_k = _distance_feature_k(config)
    chunk_size = _distance_feature_chunk_size(config)
    output_parts: list[pd.DataFrame] = []
    distance_columns = [
        "h0_wasserstein_to_clear",
        "h0_wasserstein_to_ambiguous",
        "h0_bottleneck_to_clear",
        "h0_bottleneck_to_ambiguous",
        "h1_wasserstein_to_clear",
        "h1_wasserstein_to_ambiguous",
        "h1_bottleneck_to_clear",
        "h1_bottleneck_to_ambiguous",
    ]

    for layer in sorted(feature_df["layer"].unique().tolist()):
        layer_df = feature_df.loc[feature_df["layer"].eq(layer)].copy().reset_index(drop=False)
        train_df = layer_df.loc[layer_df["split"].eq("train")].copy().reset_index(drop=True)
        if train_df.empty:
            for column in distance_columns:
                layer_df[column] = 0.0
            output_parts.append(layer_df)
            continue

        train_by_label: dict[str, dict[str, list[Any]]] = {}
        for label_value, label_name in _BINARY_DISTANCE_LABELS:
            label_df = train_df.loc[train_df["label_ambiguous"].eq(label_value)].copy()
            train_by_label[label_name] = {
                "ids": label_df["example_id"].astype(str).tolist(),
                "h0": label_df["h0_diagram"].tolist(),
                "h1": label_df["h1_diagram"].tolist(),
            }

        query_ids = layer_df["example_id"].astype(str).tolist()
        query_h0 = layer_df["h0_diagram"].tolist()
        query_h1 = layer_df["h1_diagram"].tolist()
        chunk_ranges = [
            (start, min(start + chunk_size, len(layer_df)))
            for start in range(0, len(layer_df), chunk_size)
        ]
        storage = {column: np.zeros(len(layer_df), dtype=np.float32) for column in distance_columns}

        with ProcessPoolExecutor(
            max_workers=max(1, min(parallel_jobs, len(chunk_ranges))),
            initializer=_init_binary_knn_distance_worker,
            initargs=(query_ids, query_h0, query_h1, train_by_label, distance_k),
        ) as executor:
            futures = {
                executor.submit(_binary_knn_distance_chunk, start, stop): (start, stop)
                for start, stop in chunk_ranges
            }
            progress = tqdm(
                total=len(chunk_ranges),
                desc=f"layer_{int(layer):02d}_knn_distance",
                leave=False,
            )
            for future in as_completed(futures):
                start, outputs = future.result()
                stop = futures[future][1]
                for column in distance_columns:
                    storage[column][start:stop] = outputs[column]
                progress.update(1)
            progress.close()

        for column in distance_columns:
            layer_df[column] = storage[column].astype(float)
        output_parts.append(layer_df)

    output = pd.concat(output_parts, ignore_index=True)
    output = output.sort_values(["layer", "index"]).drop(columns=["index"], errors="ignore")
    return output.reset_index(drop=True)


def _add_knn_subclass_distance_features(
    feature_df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df.copy()
    if "subclass" not in feature_df.columns:
        raise ValueError("Subclass-conditioned KNN distance mode requires a 'subclass' column in the feature frame.")

    parallel_jobs = max(1, int(config.get("parallel_jobs", 1)))
    max_subclass_workers = max(1, int(config.get("subclass_distance_max_workers", 2)))
    executor_mode = str(config.get("subclass_distance_executor", "process")).strip().lower()
    if executor_mode not in {"process", "thread"}:
        raise ValueError(f"Unsupported subclass distance executor mode: {executor_mode}")
    distance_k = _distance_feature_k(config)
    chunk_size = _distance_feature_chunk_size(config)
    output_parts: list[pd.DataFrame] = []

    for layer in sorted(feature_df["layer"].unique().tolist()):
        layer_df = feature_df.loc[feature_df["layer"].eq(layer)].copy().reset_index(drop=False)
        train_df = layer_df.loc[layer_df["split"].eq("train")].copy().reset_index(drop=True)
        label_names = sorted(train_df["subclass"].dropna().astype(str).unique().tolist())
        distance_columns = [
            _distance_feature_column(prefix, metric, label_name)
            for prefix in ("h0", "h1")
            for metric in ("wasserstein", "bottleneck")
            for label_name in label_names
        ]
        if train_df.empty or not label_names:
            for column in distance_columns:
                layer_df[column] = 0.0
            output_parts.append(layer_df)
            continue

        train_by_label: dict[str, dict[str, list[Any]]] = {}
        for label_name in label_names:
            label_df = train_df.loc[train_df["subclass"].astype(str).eq(label_name)].copy()
            train_by_label[label_name] = {
                "ids": label_df["example_id"].astype(str).tolist(),
                "h0": label_df["h0_diagram"].tolist(),
                "h1": label_df["h1_diagram"].tolist(),
            }

        query_ids = layer_df["example_id"].astype(str).tolist()
        query_h0 = layer_df["h0_diagram"].tolist()
        query_h1 = layer_df["h1_diagram"].tolist()
        chunk_ranges = [
            (start, min(start + chunk_size, len(layer_df)))
            for start in range(0, len(layer_df), chunk_size)
        ]
        storage = {column: np.zeros(len(layer_df), dtype=np.float32) for column in distance_columns}
        max_workers = max(1, min(parallel_jobs, len(chunk_ranges), max_subclass_workers))

        def _run_serial() -> None:
            LOGGER.warning(
                "Falling back to serial subclass KNN distance computation: layer=%s rows=%s labels=%s",
                int(layer),
                int(len(layer_df)),
                len(label_names),
            )
            _init_labeled_knn_distance_worker(query_ids, query_h0, query_h1, train_by_label, label_names, distance_k)
            progress = tqdm(
                total=len(chunk_ranges),
                desc=f"layer_{int(layer):02d}_knn_subclass_serial",
                leave=False,
            )
            try:
                for start, stop in chunk_ranges:
                    _, outputs = _labeled_knn_distance_chunk(start, stop)
                    for column in distance_columns:
                        storage[column][start:stop] = outputs[column]
                    progress.update(1)
            finally:
                progress.close()

        if executor_mode == "thread":
            LOGGER.info(
                "Running subclass KNN distance with threads: layer=%s rows=%s labels=%s max_workers=%s",
                int(layer),
                int(len(layer_df)),
                len(label_names),
                max_workers,
            )
            _init_labeled_knn_distance_worker(query_ids, query_h0, query_h1, train_by_label, label_names, distance_k)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_labeled_knn_distance_chunk, start, stop): (start, stop)
                    for start, stop in chunk_ranges
                }
                progress = tqdm(
                    total=len(chunk_ranges),
                    desc=f"layer_{int(layer):02d}_knn_subclass_thread",
                    leave=False,
                )
                try:
                    for future in as_completed(futures):
                        start, outputs = future.result()
                        stop = futures[future][1]
                        for column in distance_columns:
                            storage[column][start:stop] = outputs[column]
                        progress.update(1)
                finally:
                    progress.close()
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_labeled_knn_distance_worker,
                    initargs=(query_ids, query_h0, query_h1, train_by_label, label_names, distance_k),
                ) as executor:
                    futures = {
                        executor.submit(_labeled_knn_distance_chunk, start, stop): (start, stop)
                        for start, stop in chunk_ranges
                    }
                    progress = tqdm(
                        total=len(chunk_ranges),
                        desc=f"layer_{int(layer):02d}_knn_subclass",
                        leave=False,
                    )
                    try:
                        for future in as_completed(futures):
                            start, outputs = future.result()
                            stop = futures[future][1]
                            for column in distance_columns:
                                storage[column][start:stop] = outputs[column]
                            progress.update(1)
                    finally:
                        progress.close()
            except BrokenProcessPool:
                LOGGER.exception(
                    "Subclass KNN distance worker pool broke: layer=%s rows=%s labels=%s max_workers=%s",
                    int(layer),
                    int(len(layer_df)),
                    len(label_names),
                    max_workers,
                )
                _run_serial()

        for column in distance_columns:
            layer_df[column] = storage[column].astype(float)
        output_parts.append(layer_df)

    output = pd.concat(output_parts, ignore_index=True)
    output = output.sort_values(["layer", "index"]).drop(columns=["index"], errors="ignore")
    return output.reset_index(drop=True)


def build_token_cloud_feature_frame(
    cloud_df: pd.DataFrame,
    *,
    prototype_map: dict[tuple[int, str], list[np.ndarray]] | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    if cloud_df.empty:
        return pd.DataFrame()
    rows = cloud_df.to_dict(orient="records")
    parallel_jobs = max(1, int(config.get("parallel_jobs", 1)))
    base_rows = joblib.Parallel(n_jobs=parallel_jobs, backend="loky")(
        joblib.delayed(_cloud_base_feature_row)(row, config=config) for row in rows
    )
    feature_df = pd.DataFrame(base_rows)
    distance_mode = _distance_feature_mode(config)
    if distance_mode == "none":
        return feature_df.drop(columns=["h0_diagram", "h1_diagram"], errors="ignore")
    if distance_mode == "prototype":
        if prototype_map is None:
            raise ValueError("Prototype distance mode requires a non-empty prototype_map.")
        feature_df = _add_prototype_distance_features(feature_df, prototype_map=prototype_map)
    elif distance_mode == "knn_subclass":
        feature_df = _add_knn_subclass_distance_features(feature_df, config=config)
    else:
        feature_df = _add_knn_class_distance_features(feature_df, config=config)
    return feature_df.drop(columns=["h0_diagram", "h1_diagram"], errors="ignore")


def _topology_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_df.columns
        if column.startswith(TOPOLOGY_PREFIXES) or column in TOPOLOGY_EXTRA_COLUMNS
    ]


def _is_distance_feature_column(column: str) -> bool:
    return "_wasserstein_to_" in column or "_bottleneck_to_" in column


def _topology_target_columns(
    feature_df: pd.DataFrame,
    *,
    include_token_count: bool = False,
    include_distance_features: bool = False,
) -> list[str]:
    columns = _topology_feature_columns(feature_df)
    selected: list[str] = []
    for column in columns:
        if column == "token_count" and not include_token_count:
            continue
        if _is_distance_feature_column(column) and not include_distance_features:
            continue
        selected.append(column)
    return selected


def _clean_feature_label(name: str) -> str:
    label = name.replace("_norm", "").replace("_", " ")
    label = label.replace("h0 ", "H0 ").replace("h1 ", "H1 ")
    return label


def _cohens_d(ambiguous_values: np.ndarray, clear_values: np.ndarray) -> float:
    ambiguous = np.asarray(ambiguous_values, dtype=float)
    clear = np.asarray(clear_values, dtype=float)
    ambiguous = ambiguous[np.isfinite(ambiguous)]
    clear = clear[np.isfinite(clear)]
    if len(ambiguous) < 2 or len(clear) < 2:
        return 0.0
    ambiguous_var = float(np.var(ambiguous, ddof=1))
    clear_var = float(np.var(clear, ddof=1))
    pooled_num = (len(ambiguous) - 1) * ambiguous_var + (len(clear) - 1) * clear_var
    pooled_den = max(len(ambiguous) + len(clear) - 2, 1)
    pooled_std = float(np.sqrt(max(pooled_num / pooled_den, 1e-12)))
    if pooled_std <= 0.0:
        return 0.0
    return float((float(np.mean(ambiguous)) - float(np.mean(clear))) / pooled_std)


def _best_single_layer_for_dataset(final_df: pd.DataFrame, dataset: str) -> int | None:
    subset = final_df.loc[
        final_df["dataset"].eq(dataset) & final_df["feature_set"].eq("topology_only")
    ]
    if subset.empty:
        return None
    return int(subset.iloc[0]["layer"])


def _plot_token_cloud_trajectories(
    feature_table: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    output_path: Path,
) -> None:
    subset = feature_table.loc[
        feature_table["dataset"].eq(dataset) & feature_table["feature_variant"].eq("single_layer")
    ].copy()
    if subset.empty:
        return
    test_subset = subset.loc[subset["split"].eq("test")].copy()
    if not test_subset.empty:
        subset = test_subset
    available = [(column, label) for column, label in TRAJECTORY_FEATURES if column in subset.columns]
    if not available:
        return
    layers = sorted(subset["layer"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    axes_list = list(axes.flatten())
    for axis, (column, title) in zip(axes_list, available):
        summary = (
            subset.groupby(["layer", "label_ambiguous"])[column]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        for label_value in [1, 0]:
            label_rows = summary.loc[summary["label_ambiguous"].eq(label_value)].copy()
            if label_rows.empty:
                continue
            label_rows = label_rows.sort_values("layer")
            means = label_rows["mean"].to_numpy(dtype=float)
            counts = np.maximum(label_rows["count"].to_numpy(dtype=float), 1.0)
            sem = np.nan_to_num(label_rows["std"].to_numpy(dtype=float) / np.sqrt(counts), nan=0.0)
            x = label_rows["layer"].to_numpy(dtype=int)
            color = LABEL_COLOR_MAP[label_value]
            axis.plot(x, means, marker="o", linewidth=2.0, color=color, label=LABEL_NAME_MAP[label_value])
            axis.fill_between(x, means - sem, means + sem, color=color, alpha=0.16)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.set_xticks(layers)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Value")
    for axis in axes_list[len(available) :]:
        axis.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.subplots_adjust(top=0.88)
    fig.suptitle(f"{model_name}: {dataset} token-cloud topology trajectories", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_token_cloud_distributions(
    feature_table: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    best_layer: int,
    output_path: Path,
) -> list[dict[str, float]]:
    subset = feature_table.loc[
        feature_table["dataset"].eq(dataset)
        & feature_table["feature_variant"].eq("single_layer")
        & feature_table["layer"].eq(best_layer)
    ].copy()
    if subset.empty:
        return []
    test_subset = subset.loc[subset["split"].eq("test")].copy()
    if not test_subset.empty:
        subset = test_subset

    effect_rows: list[dict[str, float]] = []
    available_columns = [column for column in PLOT_FEATURE_CANDIDATES if column in subset.columns]
    for column in available_columns:
        ambiguous = subset.loc[subset["label_ambiguous"].eq(1), column].to_numpy(dtype=float)
        clear = subset.loc[subset["label_ambiguous"].eq(0), column].to_numpy(dtype=float)
        effect_rows.append(
            {
                "feature": column,
                "cohens_d": _cohens_d(ambiguous, clear),
                "ambiguous_mean": float(np.nanmean(ambiguous)) if len(ambiguous) else 0.0,
                "clear_mean": float(np.nanmean(clear)) if len(clear) else 0.0,
            }
        )
    effect_rows = sorted(effect_rows, key=lambda row: abs(float(row["cohens_d"])), reverse=True)
    top_rows = effect_rows[:4]
    if not top_rows:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axes_list = list(axes.flatten())
    for axis, row in zip(axes_list, top_rows):
        column = str(row["feature"])
        ambiguous = subset.loc[subset["label_ambiguous"].eq(1), column].to_numpy(dtype=float)
        clear = subset.loc[subset["label_ambiguous"].eq(0), column].to_numpy(dtype=float)
        box = axis.boxplot(
            [clear, ambiguous],
            labels=[LABEL_NAME_MAP[0], LABEL_NAME_MAP[1]],
            patch_artist=True,
            widths=0.55,
        )
        for patch, label_value in zip(box["boxes"], [0, 1]):
            patch.set_facecolor(LABEL_COLOR_MAP[label_value])
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.8)
        axis.set_title(_clean_feature_label(column))
        axis.set_ylabel("Value")
        axis.grid(True, axis="y", alpha=0.25)
        axis.text(
            0.98,
            0.98,
            f"d = {float(row['cohens_d']):.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis in axes_list[len(top_rows) :]:
        axis.axis("off")
    fig.suptitle(f"{model_name}: {dataset} best-layer token-cloud feature distributions (layer {best_layer})", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return top_rows


def _render_token_cloud_visualizations(
    *,
    model_name: str,
    feature_table: pd.DataFrame,
    final_df: pd.DataFrame,
    output_root: Path,
    classifier_config: dict[str, Any],
) -> dict[str, str]:
    if feature_table.empty or final_df.empty:
        return {}
    plots_root = ensure_dir(output_root / str(classifier_config.get("plots_dirname", "plots")))
    report_path = output_root / str(
        classifier_config.get("visualization_report_filename", "token_cloud_topology_visualizations.md")
    )

    lines = [
        "# Token-Cloud Topology Visualizations",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset in sorted(final_df["dataset"].unique()):
        best_layer = _best_single_layer_for_dataset(final_df, dataset)
        if best_layer is None:
            continue
        trajectory_path = plots_root / f"{dataset}__descriptor_trajectories.png"
        distribution_path = plots_root / f"{dataset}__best_layer_feature_distributions.png"
        _plot_token_cloud_trajectories(
            feature_table,
            dataset=dataset,
            model_name=model_name,
            output_path=trajectory_path,
        )
        top_rows = _plot_token_cloud_distributions(
            feature_table,
            dataset=dataset,
            model_name=model_name,
            best_layer=best_layer,
            output_path=distribution_path,
        )
        lines.extend([f"## {dataset}", "", f"- Best single layer: `{best_layer}`", ""])
        if trajectory_path.exists():
            lines.append(f"- Layerwise descriptor trajectories: `{trajectory_path}`")
        if distribution_path.exists():
            lines.append(f"- Best-layer feature distributions: `{distribution_path}`")
        if top_rows:
            lines.append("- Top best-layer effect-size features:")
            for row in top_rows:
                lines.append(
                    f"  - `{row['feature']}`: Cohen's d `{float(row['cohens_d']):.2f}`, "
                    f"ambiguous mean `{float(row['ambiguous_mean']):.3f}`, clear mean `{float(row['clear_mean']):.3f}`"
                )
        lines.append("")
    write_markdown(report_path, "\n".join(lines) + "\n")
    return {
        "plots_dir": str(plots_root),
        "visualization_report_path": str(report_path),
    }


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


def _prepare_layer_feature_frame(feature_df: pd.DataFrame, *, layer: int) -> tuple[pd.DataFrame, list[str]]:
    topology_columns = _topology_feature_columns(feature_df)
    suffix = f"l{int(layer):02d}"
    renamed = feature_df.loc[:, BASE_KEY_COLUMNS + topology_columns].copy()
    rename_map = {column: f"{column}__{suffix}" for column in topology_columns}
    renamed = renamed.rename(columns=rename_map)
    return renamed, [rename_map[column] for column in topology_columns]


def _build_multilayer_feature_frames(
    feature_df: pd.DataFrame,
    *,
    dataset: str,
    selections: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_merged: pd.DataFrame | None = None
    test_merged: pd.DataFrame | None = None
    topology_columns: list[str] = []
    selection_specs: list[dict[str, Any]] = []
    dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
    for rank, selection in enumerate(selections, start=1):
        layer = int(selection["layer"])
        selection_specs.append({"rank": rank, "layer": layer, "val_auroc": float(selection["val_auroc"])})
        layer_df = dataset_df.loc[dataset_df["layer"].eq(layer)].copy()
        train_layer = layer_df.loc[layer_df["split"].eq("train")].copy()
        test_layer = layer_df.loc[layer_df["split"].eq("test")].copy()
        train_prepared, train_topology = _prepare_layer_feature_frame(train_layer, layer=layer)
        test_prepared, _ = _prepare_layer_feature_frame(test_layer, layer=layer)
        topology_columns.extend(train_topology)
        if train_merged is None:
            train_merged = train_prepared
            test_merged = test_prepared
        else:
            train_merged = train_merged.merge(train_prepared, on=BASE_KEY_COLUMNS, how="inner")
            test_merged = test_merged.merge(test_prepared, on=BASE_KEY_COLUMNS, how="inner")
    if train_merged is None or test_merged is None or train_merged.empty or test_merged.empty:
        raise ValueError("Failed to build non-empty token-cloud multilayer features.")
    topology_columns = [column for column in topology_columns if column in train_merged.columns]
    train_summary, train_groups = _stacked_summary_features(train_merged, metric_groups={"topology": topology_columns})
    test_summary, _ = _stacked_summary_features(test_merged, metric_groups={"topology": topology_columns})
    train_multilayer = pd.concat([train_merged.reset_index(drop=True), train_summary], axis=1)
    test_multilayer = pd.concat([test_merged.reset_index(drop=True), test_summary], axis=1)
    return train_multilayer, test_multilayer, {
        "selections": selection_specs,
        "topology_columns": topology_columns,
        "topology_summary_columns": train_groups["topology"],
    }


def run_token_cloud_topology_classifier_from_features(
    *,
    model_name: str,
    feature_df: pd.DataFrame,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    if feature_df.empty:
        raise ValueError("Token-cloud topology classifier received an empty feature table.")

    output_root = ensure_dir(Path(classifier_config["output_dir"]) / model_name.replace("/", "_").replace("-", "_"))
    models_root = ensure_dir(output_root / "models")

    datasets = list(classifier_config.get("datasets", sorted(feature_df["dataset"].unique())))
    classifier_section = dict(classifier_config.get("classifier", {}))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_tables: list[pd.DataFrame] = []

    for dataset in datasets:
        dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
        train_df = dataset_df.loc[dataset_df["split"].eq("train")].copy()
        test_df = dataset_df.loc[dataset_df["split"].eq("test")].copy()
        if train_df.empty or test_df.empty:
            LOGGER.warning("Skipping token-cloud dataset %s because train/test rows are missing.", dataset)
            continue
        inner_train_ids, val_ids = _group_train_val_split(train_df, val_fraction=float(classifier_config.get("val_fraction", 0.2)), seed=seed)
        for layer in sorted(dataset_df["layer"].unique()):
            layer_train = train_df.loc[train_df["layer"].eq(layer)].copy()
            inner_train = layer_train.loc[layer_train["example_id"].astype(str).isin(inner_train_ids)].copy()
            val_df = layer_train.loc[layer_train["example_id"].astype(str).isin(val_ids)].copy()
            if inner_train.empty or val_df.empty:
                continue
            columns = _topology_feature_columns(layer_train)
            _, payload = _evaluate_feature_set(
                train_features=inner_train,
                eval_features=val_df,
                feature_columns=columns,
                classifier_config=classifier_section,
                seed=seed,
            )
            metrics = payload["eval_metrics"]
            candidate_rows.append(
                {
                    "dataset": dataset,
                    "layer": int(layer),
                    "selection_mode": "single_layer",
                    "feature_set": "topology_only",
                    "val_auroc": float(metrics["auroc"]),
                    "val_accuracy": float(metrics["accuracy"]),
                    "val_f1": float(metrics["f1"]),
                    "feature_count": int(len(columns)),
                }
            )

        candidate_df = pd.DataFrame(candidate_rows)
        dataset_candidates = candidate_df.loc[candidate_df["dataset"].eq(dataset)].copy()
        if dataset_candidates.empty:
            continue
        best_row = _selection_order(dataset_candidates).iloc[0]
        selected_rows.append({**best_row.to_dict(), "component_rank": 1})

        best_layer = int(best_row["layer"])
        final_train = train_df.loc[train_df["layer"].eq(best_layer)].copy()
        final_test = test_df.loc[test_df["layer"].eq(best_layer)].copy()
        feature_columns = _topology_feature_columns(final_train)
        feature_tables.extend([final_train.assign(feature_variant="single_layer"), final_test.assign(feature_variant="single_layer")])
        _, payload = _evaluate_feature_set(
            train_features=final_train,
            eval_features=final_test,
            feature_columns=feature_columns,
            classifier_config=classifier_section,
            seed=seed,
        )
        metrics = payload["eval_metrics"]
        final_rows.append(
            {
                "dataset": dataset,
                "selection_mode": "single_layer",
                "layer": best_layer,
                "selection_signature": str(best_layer),
                "selection_size": 1,
                "feature_set": "topology_only",
                "test_auroc": float(metrics["auroc"]),
                "test_accuracy": float(metrics["accuracy"]),
                "test_f1": float(metrics["f1"]),
                "feature_count": int(len(feature_columns)),
            }
        )
        joblib.dump(
            {
                "classifier": payload["classifier"],
                "scaler": payload["scaler"],
                "feature_columns": feature_columns,
                "dataset": dataset,
                "selection_mode": "single_layer",
                "layer": best_layer,
                "train_metrics": payload["train_metrics"],
                "test_metrics": payload["eval_metrics"],
            },
            models_root / f"{dataset}__topology_only.joblib",
        )

        if bool(classifier_config.get("multilayer_enabled", True)):
            selections = _select_multilayer_candidates(
                dataset_candidates,
                top_k=int(classifier_config.get("multilayer_top_k", 3)),
            )
            for rank, row in enumerate(selections, start=1):
                selected_rows.append({**row, "feature_set": "topology_multilayer", "selection_mode": "multilayer_component", "component_rank": rank})
            multi_train, multi_test, multi_meta = _build_multilayer_feature_frames(
                feature_df,
                dataset=dataset,
                selections=selections,
            )
            multi_columns = multi_meta["topology_columns"] + multi_meta["topology_summary_columns"]
            feature_tables.extend([multi_train.assign(feature_variant="multilayer"), multi_test.assign(feature_variant="multilayer")])
            _, payload = _evaluate_feature_set(
                train_features=multi_train,
                eval_features=multi_test,
                feature_columns=multi_columns,
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
                    "feature_set": "topology_multilayer",
                    "test_auroc": float(metrics["auroc"]),
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_f1": float(metrics["f1"]),
                    "feature_count": int(len(multi_columns)),
                }
            )
            joblib.dump(
                {
                    "classifier": payload["classifier"],
                    "scaler": payload["scaler"],
                    "feature_columns": multi_columns,
                    "dataset": dataset,
                    "selection_mode": "multilayer",
                    "selections": multi_meta["selections"],
                    "train_metrics": payload["train_metrics"],
                    "test_metrics": payload["eval_metrics"],
                },
                models_root / f"{dataset}__topology_multilayer.joblib",
            )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["dataset", "layer"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["dataset", "feature_set"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["dataset", "feature_set", "selection_mode", "component_rank"]).reset_index(drop=True)
    feature_table = pd.concat(feature_tables, ignore_index=True, sort=False) if feature_tables else pd.DataFrame()

    candidate_path = output_root / str(classifier_config["candidate_metrics_filename"])
    final_path = output_root / str(classifier_config["final_metrics_filename"])
    selected_path = output_root / str(classifier_config["selected_candidates_filename"])
    feature_path = output_root / str(classifier_config["feature_table_filename"])
    report_path = output_root / str(classifier_config["report_filename"])
    metadata_path = output_root / str(classifier_config["metadata_filename"])

    write_parquet(candidate_df, candidate_path)
    write_parquet(final_df, final_path)
    write_parquet(selected_df, selected_path)
    if not feature_table.empty:
        write_parquet(feature_table, feature_path)

    lines = [
        "# Token-Cloud Topology Classifier",
        "",
        f"- Model: `{model_name}`",
        "",
        "## Final Results",
        "",
    ]
    if not final_df.empty:
        for dataset in sorted(final_df["dataset"].unique()):
            lines.append(f"### {dataset}")
            lines.append("")
            dataset_df = final_df.loc[final_df["dataset"].eq(dataset)]
            for row in dataset_df.to_dict(orient="records"):
                lines.append(
                    f"- `{row['feature_set']}`: AUROC `{row['test_auroc']:.4f}`, "
                    f"accuracy `{row['test_accuracy']:.4f}`, selection `{row['selection_signature']}`"
                )
            lines.append("")
    write_markdown(report_path, "\n".join(lines) + "\n")
    visualization_outputs = {}
    if bool(classifier_config.get("visualization_enabled", True)) and not feature_table.empty:
        visualization_outputs = _render_token_cloud_visualizations(
            model_name=model_name,
            feature_table=feature_table,
            final_df=final_df,
            output_root=output_root,
            classifier_config=classifier_config,
        )
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "created_at": utc_now_iso(),
            "datasets": datasets,
            "output_artifacts": {
                "candidate_metrics": str(candidate_path),
                "final_metrics": str(final_path),
                "selected_candidates": str(selected_path),
                "feature_table": str(feature_path),
                "report": str(report_path),
                **visualization_outputs,
            },
        },
    )
    return {
        "candidate_metrics_path": str(candidate_path),
        "final_metrics_path": str(final_path),
        "selected_candidates_path": str(selected_path),
        "feature_table_path": str(feature_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
        **{f"{key}": value for key, value in visualization_outputs.items()},
    }


def run_token_cloud_topology_classifier_analysis(
    *,
    config: dict[str, Any],
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    model_name = str(config["model"]["name"])
    output_root = ensure_dir(Path(classifier_config["output_dir"]) / model_name.replace("/", "_").replace("-", "_"))
    feature_path = output_root / str(classifier_config["feature_table_filename"])
    cloud_cache_path = output_root / "token_cloud_forward_cache.joblib"
    bundle = load_hf_model(config["model"], classifier_config)
    total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = _resolve_candidate_layers(total_layers, classifier_config)
    LOGGER.info("Token-cloud candidate layers: %s", layers)

    dataset_frames: list[pd.DataFrame] = []
    pair_output_dir = Path(config["data"]["pair_output_dir"])
    dataset_paths: list[Path] = []
    text_column = str(classifier_config.get("text_column", "text"))
    use_chat_template = bool(classifier_config.get("use_chat_template", False))
    system_prompt = classifier_config.get("system_prompt")
    datasets = list(classifier_config.get("datasets", ["ambigqa"]))
    for dataset in datasets:
        path = pair_output_dir / f"{dataset}_pairs.parquet"
        dataset_paths.append(path)
        dataset_df = pd.read_parquet(path)
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=text_column,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
        dataset_frames.append(prepared_df)
    full_df = pd.concat(dataset_frames, ignore_index=True)
    feature_signature = _token_cloud_feature_cache_signature(
        model_name=model_name,
        dataset_paths=dataset_paths,
        layers=layers,
        classifier_config=classifier_config,
        seed=seed,
    )
    forward_signature = _token_cloud_forward_cache_signature(
        model_name=model_name,
        dataset_paths=dataset_paths,
        layers=layers,
        config=classifier_config,
        seed=seed,
    )
    if not bool(classifier_config.get("force_rebuild_feature_table", False)):
        cached_feature_df = load_cached_token_cloud_feature_frame(
            feature_path=feature_path,
            signature=feature_signature,
        )
        if cached_feature_df is not None:
            return run_token_cloud_topology_classifier_from_features(
                model_name=model_name,
                feature_df=cached_feature_df,
                classifier_config=classifier_config,
                seed=seed,
            )
    train_df = full_df.loc[full_df["split"].eq("train")].copy().reset_index(drop=True)

    use_pca = bool(classifier_config.get("use_pca", True))
    pca_components = int(classifier_config.get("pca_components", 8)) if use_pca else None
    topology_value = classifier_config.get("topology_components", pca_components)
    topology_components = int(topology_value) if topology_value is not None else None
    if not use_pca:
        topology_components = None
    cached_cloud = None
    if not bool(classifier_config.get("force_rebuild_forward_cache", False)):
        cached_cloud = load_cached_token_cloud_forward_frame(
            cache_path=cloud_cache_path,
            signature=forward_signature,
        )
    if cached_cloud is not None:
        cloud_df, cached_pca_variance = cached_cloud
        if cached_pca_variance is not None:
            write_json(output_root / "token_cloud_pca_variance.json", cached_pca_variance)
    else:
        if use_pca:
            token_matrices = _extract_train_token_matrices(
                bundle=bundle,
                train_df=train_df,
                text_column="_token_cloud_text",
                layers=layers,
                config={**classifier_config, "_seed": seed},
            )
            reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
        else:
            token_matrices = {}
            reducers = {layer: None for layer in layers}
        raw_hidden_size = int(
            getattr(bundle.model.config, "hidden_size", getattr(bundle.model.config, "d_model", 0)) or 0
        )
        pca_variance = {
            "use_pca": use_pca,
            "pca_components": pca_components,
            "topology_components": topology_components,
            "candidate_layers": layers,
            "raw_hidden_size": raw_hidden_size,
            "effective_cloud_dim": raw_hidden_size if not use_pca else topology_components,
            "explained_variance_ratio": {},
            "explained_variance_cumulative": {},
        }
        for layer, reducer in reducers.items():
            if use_pca and reducer is not None:
                ratio = np.asarray(getattr(reducer, "explained_variance_ratio_", []), dtype=float)
                pca_variance["explained_variance_ratio"][str(layer)] = ratio.tolist()
                if ratio.size:
                    cumulative = np.cumsum(ratio)
                    cap = min(int(topology_components or cumulative.size), cumulative.size)
                    pca_variance["explained_variance_cumulative"][str(layer)] = {
                        "all_components": cumulative.tolist(),
                        "topology_components": float(cumulative[cap - 1]) if cap > 0 else 0.0,
                    }
                else:
                    pca_variance["explained_variance_cumulative"][str(layer)] = {
                        "all_components": [],
                        "topology_components": 0.0,
                    }
            else:
                pca_variance["explained_variance_ratio"][str(layer)] = []
                pca_variance["explained_variance_cumulative"][str(layer)] = {
                    "all_components": [],
                    "topology_components": None,
                }
        write_json(output_root / "token_cloud_pca_variance.json", pca_variance)
        cloud_df = _extract_reduced_clouds(
            bundle=bundle,
            df=full_df,
            text_column="_token_cloud_text",
            layers=layers,
            reducers=reducers,
            config=classifier_config,
        )
        save_cached_token_cloud_forward_frame(
            cloud_df=cloud_df,
            cache_path=cloud_cache_path,
            signature=forward_signature,
            pca_variance=pca_variance,
        )
    prototype_map = None
    if _distance_feature_mode(classifier_config) == "prototype":
        prototype_map = _prototype_diagrams_from_clouds(cloud_df, layers=layers, config=classifier_config, seed=seed)
    feature_df = build_token_cloud_feature_frame(cloud_df, prototype_map=prototype_map, config=classifier_config)
    save_cached_token_cloud_feature_frame(
        feature_df=feature_df,
        feature_path=feature_path,
        signature=feature_signature,
    )
    return run_token_cloud_topology_classifier_from_features(
        model_name=model_name,
        feature_df=feature_df,
        classifier_config=classifier_config,
        seed=seed,
    )
