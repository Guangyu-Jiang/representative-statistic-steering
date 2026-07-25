"""Trajectory/process experiments for intrinsic token-cloud topology.

Implements four ideas on author-style prompts with large splits:
1) Layer-to-layer drift features on intrinsic PH descriptors
2) Sequence model (GRU) over intrinsic descriptor sequences
3) Token trajectory statistics across layers
4) Dynamic topology distances (Wasserstein between per-layer diagrams)

Matched neuron baselines:
- Drift features on full neurons and official super neurons
- Sequence GRU on full neurons (PCA-reduced) and super neurons
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.author_repo_eval import (
    DATASET_SPECS,
    MODEL_SPECS,
    _format_prompt,
    _load_author_prompts,
    _shuffle_and_slice_author_style,
)
from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.eval.tune_token_cloud_feature_tables import _feature_subset_columns
from aen_replication.train.independent_topology_classifier import (
    _diagram_descriptors,
    _fit_classifier,
    _persistence_image_features,
    _safe_wasserstein,
    _transform_with_scaler,
)
from aen_replication.train.token_cloud_topology_classifier import (
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _valid_token_mask,
)
from aen_replication.models.hf_model import load_hf_model
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic": {
        "family": "logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
    "rf200_leaf4": {
        "family": "random_forest",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "standardize": False,
    },
    "et200_leaf4": {
        "family": "extra_trees",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "standardize": False,
    },
}

TOKEN_CLOUD_CONFIG: dict[str, Any] = {
    "batch_size": 6,
    "max_length": 64,
    "candidate_layers": [0, 14, 31],
    "pca_components": 8,
    "topology_components": 6,
    "pca_whiten": False,
    "drop_special_tokens": True,
    "pca_fit_token_cap": 16000,
    "betti_grid_size": 24,
    "persistence_image_grid_side": 3,
    "maxdim": 1,
    "coeff": 2,
}

INTRINSIC_SUBSET = "no_distance"
INTRINSIC_FEATURE_VARIANT = "single_layer"

NEURON_PCA_DIM = 32

HMM_SEQUENCE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "gaussian_hmm_1state_diag",
        "n_components": 1,
        "covariance_type": "diag",
    },
    {
        "name": "gaussian_hmm_2state_diag",
        "n_components": 2,
        "covariance_type": "diag",
    },
    {
        "name": "gaussian_hmm_3state_diag",
        "n_components": 3,
        "covariance_type": "diag",
    },
]


def _predict_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(matrix), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    predictions = np.asarray(model.predict(matrix), dtype=float)
    return np.clip(predictions, 0.0, 1.0)


def _threshold_grid(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    quantiles = np.linspace(0.02, 0.98, 97)
    grid = np.unique(np.clip(np.quantile(probs, quantiles), 0.0, 1.0))
    if grid.size == 0:
        return np.array([0.5], dtype=float)
    return np.unique(np.concatenate([grid, np.array([0.5], dtype=float)]))


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = binary_classification_metrics(y_true, probabilities, threshold=best_threshold)
    best_score = (
        float(best_metrics["accuracy"]),
        float(best_metrics["f1"]),
        -abs(best_threshold - 0.5),
    )
    for threshold in _threshold_grid(probabilities):
        metrics = binary_classification_metrics(y_true, probabilities, threshold=float(threshold))
        score = (
            float(metrics["accuracy"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def _evaluate_matrix(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    *,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    clf, scaler = _fit_classifier(train_X, train_y, config=classifier_config, seed=seed)
    val_matrix = _transform_with_scaler(val_X, scaler)
    test_matrix = _transform_with_scaler(test_X, scaler)
    val_probs = _predict_probabilities(clf, val_matrix)
    test_probs = _predict_probabilities(clf, test_matrix)
    threshold, val_metrics = _select_threshold(val_y, val_probs)
    test_metrics = binary_classification_metrics(test_y, test_probs, threshold=threshold)
    return {
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_auroc": float(roc_auc_score(test_y, test_probs)),
    }


def _build_author_prompt_frame(
    *,
    author_repo_root: Path,
    dataset_name: str,
    prompt_model_name: str,
    train_per_class: int,
    test_per_class: int,
) -> pd.DataFrame:
    ambig_prompts, clear_prompts = _load_author_prompts(
        author_repo_root=author_repo_root,
        dataset_name=dataset_name,
        prompt_model_name=prompt_model_name,
    )
    split_payload = _shuffle_and_slice_author_style(
        ambig_prompts,
        clear_prompts,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
    )
    rows: list[dict[str, Any]] = []
    for split_name, prompts, label_ambiguous in [
        ("train", split_payload["train_ambig"], 1),
        ("train", split_payload["train_clear"], 0),
        ("test", split_payload["test_ambig"], 1),
        ("test", split_payload["test_clear"], 0),
    ]:
        label_name = "ambiguous" if label_ambiguous else "clear"
        for index, prompt in enumerate(prompts):
            example_id = f"{dataset_name}__{split_name}__{label_name}__{index:04d}"
            rows.append(
                {
                    "example_id": example_id,
                    "pair_id": example_id,
                    "dataset": dataset_name,
                    "split": split_name,
                    "label_ambiguous": label_ambiguous,
                    "text": prompt,
                }
            )
    return pd.DataFrame(rows)


def _extract_author_hidden_states_multi(
    *,
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_indices: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    vectors: dict[int, list[np.ndarray]] = {layer: [] for layer in layer_indices}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        model_inputs = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states for author prompt extraction.")
        expected_layers = int(getattr(model.config, "num_hidden_layers", len(hidden_states)))
        offset = 1 if len(hidden_states) == expected_layers + 1 else 0
        for layer in layer_indices:
            index = layer + offset
            if index >= len(hidden_states):
                raise IndexError(
                    f"Requested layer {layer} (index {index}) but hidden_states has {len(hidden_states)} entries."
                )
            layer_hidden = hidden_states[index]
            vectors[layer].append(layer_hidden.mean(dim=1).float().cpu().numpy())
        del outputs
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in vectors.items()}


def _resolve_model_layers(model: AutoModelForCausalLM, candidate_layers: list[int]) -> list[int]:
    total_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if total_layers <= 0:
        return sorted(set(int(layer) for layer in candidate_layers))
    valid = [int(layer) for layer in candidate_layers if 0 <= int(layer) < total_layers]
    if valid:
        return sorted(set(valid))
    return [max(total_layers - 1, 0)]


def _split_classwise(
    matrices: dict[int, np.ndarray],
    train_per_class: int,
    test_per_class: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    train_end = train_per_class * 2
    test_ambig_start = train_end
    test_clear_start = train_end + test_per_class
    train_y = np.array([0] * train_per_class + [1] * train_per_class, dtype=int)
    test_y = np.array([0] * test_per_class + [1] * test_per_class, dtype=int)
    train_X = {}
    test_X = {}
    for layer, matrix in matrices.items():
        train_X[layer] = np.vstack([matrix[:train_per_class], matrix[train_per_class:train_end]])
        test_X[layer] = np.vstack(
            [
                matrix[test_ambig_start:test_clear_start],
                matrix[test_clear_start:test_clear_start + test_per_class],
            ]
        )
    return train_X, test_X, train_y, test_y


def _random_train_val_split(n_train: int, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_train)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_train * val_fraction)))
    val_idx = np.sort(indices[:val_size])
    inner_train_idx = np.sort(indices[val_size:])
    return inner_train_idx, val_idx


def _build_drift_features(per_layer: dict[int, np.ndarray]) -> np.ndarray:
    layers = sorted(per_layer.keys())
    matrices = [per_layer[layer] for layer in layers]
    n = matrices[0].shape[0]
    features: list[np.ndarray] = []
    for idx, matrix in enumerate(matrices):
        if matrix.shape[0] != n:
            raise ValueError("Mismatched row counts for drift feature inputs.")
        features.append(matrix)
    for a, b in zip(matrices, matrices[1:]):
        delta = b - a
        features.append(delta)
        features.append(np.abs(delta))
    if len(matrices) >= 2:
        total = matrices[-1] - matrices[0]
        features.append(total)
        features.append(np.abs(total))
    x = np.stack([np.arange(len(matrices))] * n, axis=0).astype(float)
    stacked = np.stack(matrices, axis=1)
    x_center = x - x.mean(axis=1, keepdims=True)
    x_center = x_center[:, :, None]
    slope = (x_center * (stacked - stacked.mean(axis=1, keepdims=True))).sum(axis=1) / (
        (x_center**2).sum(axis=1) + 1e-12
    )
    features.append(slope)
    return np.concatenate(features, axis=1)


def _compute_intrinsic_descriptors(
    reduced_cloud: np.ndarray,
    *,
    grid_size: int,
    pimg_side: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for dim, prefix in [(0, "h0"), (1, "h1")]:
        diagram = _compute_diagrams_from_cloud(reduced_cloud, maxdim=1)[dim]
        result.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=grid_size))
        result.update(_persistence_image_features(diagram, prefix=prefix, grid_side=pimg_side))
    return result


def _compute_diagrams_from_cloud(cloud: np.ndarray, *, maxdim: int = 1) -> list[np.ndarray]:
    from aen_replication.train.independent_topology_classifier import _compute_diagrams

    return _compute_diagrams(cloud, maxdim=maxdim, coeff=2, distance_metric="euclidean")


@dataclass
class SequenceDataset:
    sequences: np.ndarray
    labels: np.ndarray


class GRUClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def _train_gru(
    train: SequenceDataset,
    val: SequenceDataset,
    test: SequenceDataset,
    *,
    seed: int,
    device: torch.device,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = GRUClassifier(train.sequences.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state = None
    best_val = -1.0
    epochs_no_improve = 0

    def _run_epoch(dataset: SequenceDataset, train_mode: bool) -> tuple[float, np.ndarray]:
        model.train(train_mode)
        logits_list = []
        losses = []
        batch_size = 64
        for start in range(0, len(dataset.labels), batch_size):
            x = torch.tensor(dataset.sequences[start : start + batch_size], dtype=torch.float32, device=device)
            y = torch.tensor(dataset.labels[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logits_list.append(logits.detach().cpu().numpy())
            losses.append(loss.item())
        return float(np.mean(losses)), np.concatenate(logits_list)

    for _ in range(max_epochs):
        _run_epoch(train, train_mode=True)
        _, val_logits = _run_epoch(val, train_mode=False)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50.0, 50.0)))
        val_auroc = float(roc_auc_score(val.labels, val_probs))
        if val_auroc > best_val + 1e-4:
            best_val = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    _, test_logits = _run_epoch(test, train_mode=False)
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -50.0, 50.0)))
    threshold, _ = _select_threshold(val.labels, val_probs)
    test_metrics = binary_classification_metrics(test.labels, test_probs, threshold=threshold)
    return {
        "test_auroc": float(roc_auc_score(test.labels, test_probs)),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "threshold": float(threshold),
    }


def _prepare_sequences(per_layer_features: dict[int, np.ndarray]) -> np.ndarray:
    layers = sorted(per_layer_features.keys())
    sequence = np.stack([per_layer_features[layer] for layer in layers], axis=1)
    return sequence


def _standardize_sequence(train_seq: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_seq.mean(axis=(0, 1), keepdims=True)
    std = train_seq.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    train_norm = (train_seq - mean) / std
    rest = [(seq - mean) / std for seq in others]
    return train_norm, rest, mean.squeeze(), std.squeeze()


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_gaussian_hmm(
    sequences: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    seed: int,
) -> GaussianHMM:
    lengths = [int(sequences.shape[1])] * int(sequences.shape[0])
    flat = np.asarray(sequences, dtype=np.float64).reshape(-1, sequences.shape[-1])
    model = GaussianHMM(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        n_iter=200,
        tol=1e-3,
        min_covar=1e-3,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(flat, lengths)
    return model


def _score_hmm_likelihood_ratio(
    negative_model: GaussianHMM,
    positive_model: GaussianHMM,
    sequences: np.ndarray,
) -> np.ndarray:
    scores: list[float] = []
    for sequence in np.asarray(sequences, dtype=np.float64):
        ll_negative = float(negative_model.score(sequence))
        ll_positive = float(positive_model.score(sequence))
        length = max(int(sequence.shape[0]), 1)
        scores.append((ll_positive - ll_negative) / float(length))
    return np.asarray(scores, dtype=float)


def _train_class_conditional_hmm(
    train: SequenceDataset,
    val: SequenceDataset,
    test: SequenceDataset,
    *,
    seed: int,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for config in HMM_SEQUENCE_CONFIGS:
        n_components = int(config["n_components"])
        negative_train = train.sequences[train.labels == 0]
        positive_train = train.sequences[train.labels == 1]
        if len(negative_train) < n_components or len(positive_train) < n_components:
            continue
        try:
            negative_model = _fit_gaussian_hmm(
                negative_train,
                n_components=n_components,
                covariance_type=str(config["covariance_type"]),
                seed=seed,
            )
            positive_model = _fit_gaussian_hmm(
                positive_train,
                n_components=n_components,
                covariance_type=str(config["covariance_type"]),
                seed=seed + 1,
            )
            val_scores = _score_hmm_likelihood_ratio(negative_model, positive_model, val.sequences)
            val_probs = _sigmoid(val_scores)
            threshold, val_metrics = _select_threshold(val.labels, val_probs)
            val_auroc = float(roc_auc_score(val.labels, val_probs))
            test_scores = _score_hmm_likelihood_ratio(negative_model, positive_model, test.sequences)
            test_probs = _sigmoid(test_scores)
            test_metrics = binary_classification_metrics(test.labels, test_probs, threshold=threshold)
            candidate = {
                "config_name": str(config["name"]),
                "threshold": float(threshold),
                "val_auroc": val_auroc,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_f1": float(val_metrics["f1"]),
                "test_auroc": float(roc_auc_score(test.labels, test_probs)),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1": float(test_metrics["f1"]),
            }
        except Exception:
            continue
        candidate_key = (
            float(candidate["val_auroc"]),
            float(candidate["val_accuracy"]),
            float(candidate["val_f1"]),
            -float(n_components),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_result = candidate

    if best_result is None:
        raise RuntimeError("Failed to fit any HMM sequence configuration.")
    return best_result


def _extract_token_trajectory_features(
    *,
    bundle: Any,
    df: pd.DataFrame,
    layers: list[int],
    reducers: dict[int, PCA],
    config: dict[str, Any],
) -> pd.DataFrame:
    encoder = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    batch_size = int(config.get("batch_size", 6))
    max_length = int(config.get("max_length", 64))
    drop_special_tokens = bool(config.get("drop_special_tokens", True))
    special_ids = set(int(token_id) for token_id in getattr(encoder, "all_special_ids", []) if token_id is not None)

    rows: list[dict[str, Any]] = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        encoded = encoder(
            batch_df["text"].tolist(),
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
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        batch_rows = batch_df.reset_index(drop=True).to_dict(orient="records")
        for row_index, row in enumerate(batch_rows):
            valid = _valid_token_mask(
                input_ids_cpu[row_index],
                attention_mask_cpu[row_index],
                special_ids=special_ids,
                drop_special_tokens=drop_special_tokens,
            )
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            reduced_layers: list[np.ndarray] = []
            for layer in layers:
                reducer = reducers[layer]
                layer_output = hidden_states[layer + 1][row_index].detach().float().cpu()
                token_vectors = layer_output[valid_idx].numpy()
                reduced = reducer.transform(token_vectors)[:, : int(config.get("topology_components", 6))]
                reduced_layers.append(reduced.astype(np.float32, copy=False))
            if len(reduced_layers) < 2:
                continue
            step01 = np.linalg.norm(reduced_layers[1] - reduced_layers[0], axis=1)
            if len(reduced_layers) >= 3:
                step12 = np.linalg.norm(reduced_layers[2] - reduced_layers[1], axis=1)
            else:
                step12 = np.zeros_like(step01)
            total = step01 + step12
            def _stats(arr: np.ndarray) -> tuple[float, float, float]:
                return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))
            row_out = {
                "example_id": row["example_id"],
                "pair_id": row["pair_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "label_ambiguous": row["label_ambiguous"],
            }
            row_out.update({f"token_step01_{k}": v for k, v in zip(["mean", "std", "max"], _stats(step01))})
            row_out.update({f"token_step12_{k}": v for k, v in zip(["mean", "std", "max"], _stats(step12))})
            row_out.update({f"token_total_{k}": v for k, v in zip(["mean", "std", "max"], _stats(total))})
            rows.append(row_out)
        del outputs, hidden_states, model_inputs, input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _compute_dynamic_topology_features(
    feature_df: pd.DataFrame,
    *,
    layers: list[int],
    subset_name: str,
) -> pd.DataFrame:
    subset = feature_df.loc[feature_df["feature_variant"].eq("single_layer")].copy()
    feature_columns = _feature_subset_columns(subset, subset_name)
    rows: list[dict[str, Any]] = []
    for dataset in sorted(subset["dataset"].unique()):
        dataset_df = subset.loc[subset["dataset"].eq(dataset)]
        for split in sorted(dataset_df["split"].unique()):
            split_df = dataset_df.loc[dataset_df["split"].eq(split)]
            for example_id, group in split_df.groupby("example_id"):
                group = group.set_index("layer")
                if any(layer not in group.index for layer in layers):
                    continue
                layer_features = {layer: group.loc[layer, feature_columns].to_numpy(dtype=float) for layer in layers}
                row = {
                    "example_id": example_id,
                    "pair_id": group["pair_id"].iloc[0],
                    "dataset": dataset,
                    "split": split,
                    "label_ambiguous": int(group["label_ambiguous"].iloc[0]),
                }
                row.update({f"drift_{col}": np.abs(layer_features[layers[1]] - layer_features[layers[0]])[i] for i, col in enumerate(feature_columns)})
                rows.append(row)
    return pd.DataFrame(rows)


def run_suite(
    *,
    author_repo_root: Path,
    output_root: Path,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int,
    test_per_class: int,
    seed: int,
    val_fraction: float,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    candidate_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        model_path = snapshot_download(repo_id=spec.load_repo, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True,
        )
        model.eval()

        model_output_dir = ensure_dir(output_root / spec.output_dir_name)
        for dataset_name in dataset_names:
            frame = _build_author_prompt_frame(
                author_repo_root=author_repo_root,
                dataset_name=dataset_name,
                prompt_model_name=spec.prompt_model_name,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
            )
            texts = frame["text"].tolist()
            layers = _resolve_model_layers(model, list(TOKEN_CLOUD_CONFIG["candidate_layers"]))
            # Neuron baselines (per-layer mean pooled).
            layer_matrices = _extract_author_hidden_states_multi(
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                layer_indices=layers,
                batch_size=int(TOKEN_CLOUD_CONFIG["batch_size"]),
            )
            train_X, test_X, train_y, test_y = _split_classwise(layer_matrices, train_per_class, test_per_class)
            inner_train_idx, val_idx = _random_train_val_split(len(train_y), seed=seed, val_fraction=val_fraction)
            # Drift baseline for neurons.
            neuron_drift_train = _build_drift_features({layer: train_X[layer][inner_train_idx] for layer in layers})
            neuron_drift_val = _build_drift_features({layer: train_X[layer][val_idx] for layer in layers})
            neuron_drift_test = _build_drift_features({layer: test_X[layer] for layer in layers})
            for name, config in FAMILY_CONFIGS.items():
                payload = _evaluate_matrix(
                    neuron_drift_train,
                    train_y[inner_train_idx],
                    neuron_drift_val,
                    train_y[val_idx],
                    neuron_drift_test,
                    test_y,
                    classifier_config=config,
                    seed=seed,
                )
                candidate_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "feature_source": "full_neurons_drift",
                        "feature_variant": "trajectory",
                        "feature_subset": "drift",
                        "config_name": name,
                        "test_auroc": float(payload["test_auroc"]),
                        "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                        "test_f1": float(payload["test_metrics"]["f1"]),
                    }
                )
            # Super neurons drift baseline.
            super_layers = {layer: train_X[layer][:, spec.super_neurons] for layer in layers}
            super_train = {layer: train_X[layer][:, spec.super_neurons] for layer in layers}
            super_test = {layer: test_X[layer][:, spec.super_neurons] for layer in layers}
            super_drift_train = _build_drift_features({layer: super_train[layer][inner_train_idx] for layer in layers})
            super_drift_val = _build_drift_features({layer: super_train[layer][val_idx] for layer in layers})
            super_drift_test = _build_drift_features(super_test)
            for name, config in FAMILY_CONFIGS.items():
                payload = _evaluate_matrix(
                    super_drift_train,
                    train_y[inner_train_idx],
                    super_drift_val,
                    train_y[val_idx],
                    super_drift_test,
                    test_y,
                    classifier_config=config,
                    seed=seed,
                )
                candidate_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "feature_source": "super_neurons_drift",
                        "feature_variant": "trajectory",
                        "feature_subset": "drift",
                        "config_name": name,
                        "test_auroc": float(payload["test_auroc"]),
                        "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                        "test_f1": float(payload["test_metrics"]["f1"]),
                    }
                )

            # Sequence model baselines for neurons with PCA reduction (fit on train only).
            pca_models: dict[int, PCA] = {}
            for layer in layers:
                n_components = min(NEURON_PCA_DIM, train_X[layer].shape[1])
                pca = PCA(n_components=n_components, random_state=seed)
                pca.fit(train_X[layer][inner_train_idx])
                pca_models[layer] = pca
            full_reduced = {layer: pca_models[layer].transform(train_X[layer]) for layer in layers}
            full_reduced_test = {layer: pca_models[layer].transform(test_X[layer]) for layer in layers}
            full_train_seq = _prepare_sequences({layer: full_reduced[layer] for layer in layers})
            full_test_seq = _prepare_sequences({layer: full_reduced_test[layer] for layer in layers})
            full_train_seq, [full_test_seq_norm], _, _ = _standardize_sequence(full_train_seq, full_test_seq)
            val_seq = full_train_seq[val_idx]
            train_seq = full_train_seq[inner_train_idx]
            seq_result = _train_gru(
                SequenceDataset(train_seq, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(full_test_seq_norm, test_y),
                seed=seed,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "full_neurons_sequence_gru",
                    "config_name": "gru_hidden64",
                    "test_auroc": seq_result["test_auroc"],
                    "test_accuracy": seq_result["test_accuracy"],
                    "test_f1": seq_result["test_f1"],
                }
            )
            hmm_result = _train_class_conditional_hmm(
                SequenceDataset(train_seq, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(full_test_seq_norm, test_y),
                seed=seed,
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "full_neurons_sequence_hmm",
                    "config_name": hmm_result["config_name"],
                    "test_auroc": hmm_result["test_auroc"],
                    "test_accuracy": hmm_result["test_accuracy"],
                    "test_f1": hmm_result["test_f1"],
                }
            )

            super_train_seq = _prepare_sequences({layer: super_train[layer] for layer in layers})
            super_test_seq = _prepare_sequences(super_test)
            super_train_seq, [super_test_seq_norm], _, _ = _standardize_sequence(super_train_seq, super_test_seq)
            val_seq = super_train_seq[val_idx]
            train_seq = super_train_seq[inner_train_idx]
            seq_result = _train_gru(
                SequenceDataset(train_seq, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(super_test_seq_norm, test_y),
                seed=seed,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "super_neurons_sequence_gru",
                    "config_name": "gru_hidden64",
                    "test_auroc": seq_result["test_auroc"],
                    "test_accuracy": seq_result["test_accuracy"],
                    "test_f1": seq_result["test_f1"],
                }
            )
            hmm_result = _train_class_conditional_hmm(
                SequenceDataset(train_seq, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(super_test_seq_norm, test_y),
                seed=seed + 100,
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "super_neurons_sequence_hmm",
                    "config_name": hmm_result["config_name"],
                    "test_auroc": hmm_result["test_auroc"],
                    "test_accuracy": hmm_result["test_accuracy"],
                    "test_f1": hmm_result["test_f1"],
                }
            )

            # Token-cloud intrinsic descriptors.
            bundle = load_hf_model(
                {
                    "name": spec.load_repo,
                    "tokenizer_name": spec.load_repo,
                    "trust_remote_code": False,
                    "torch_dtype": "bfloat16",
                    "device": "auto",
                    "local_files_only": True,
                    "use_fast": True,
                    "model_class": "causal_lm",
                },
                TOKEN_CLOUD_CONFIG,
            )
            train_df = frame.loc[frame["split"].eq("train")].copy().reset_index(drop=True)
            token_matrices = _extract_train_token_matrices(
                bundle=bundle,
                train_df=train_df,
                text_column="text",
                layers=layers,
                config={**TOKEN_CLOUD_CONFIG, "_seed": seed},
            )
            reducers = _fit_layer_reducers(token_matrices, config=TOKEN_CLOUD_CONFIG, seed=seed)
            cloud_df = _extract_reduced_clouds(
                bundle=bundle,
                df=frame,
                text_column="text",
                layers=layers,
                reducers=reducers,
                config=TOKEN_CLOUD_CONFIG,
            )
            # Build intrinsic descriptor rows per example/layer.
            per_layer_rows: list[dict[str, Any]] = []
            grid_size = int(TOKEN_CLOUD_CONFIG["betti_grid_size"])
            pimg_side = int(TOKEN_CLOUD_CONFIG["persistence_image_grid_side"])
            for row in cloud_df.to_dict(orient="records"):
                descriptors = _compute_intrinsic_descriptors(row["cloud"], grid_size=grid_size, pimg_side=pimg_side)
                per_layer_rows.append(
                    {
                        "example_id": row["example_id"],
                        "pair_id": row["pair_id"],
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "label_ambiguous": row["label_ambiguous"],
                        "layer": int(row["layer"]),
                        **descriptors,
                    }
                )
            feature_df = pd.DataFrame(per_layer_rows)

            # Drift features for intrinsic descriptors.
            subset = feature_df.loc[feature_df["split"].eq("train")]
            feature_columns = _feature_subset_columns(feature_df, INTRINSIC_SUBSET)
            for split_name, split_df in feature_df.groupby("split"):
                pass
            def _pivot_features(split: str) -> dict[int, np.ndarray]:
                split_df = feature_df.loc[feature_df["split"].eq(split)]
                per_layer: dict[int, np.ndarray] = {}
                for layer in layers:
                    layer_df = split_df.loc[split_df["layer"].eq(layer)]
                    per_layer[layer] = layer_df.loc[:, feature_columns].to_numpy(dtype=float)
                return per_layer
            train_per_layer = _pivot_features("train")
            test_per_layer = _pivot_features("test")
            intrinsic_train = _build_drift_features({layer: train_per_layer[layer][inner_train_idx] for layer in layers})
            intrinsic_val = _build_drift_features({layer: train_per_layer[layer][val_idx] for layer in layers})
            intrinsic_test = _build_drift_features(test_per_layer)
            for name, config in FAMILY_CONFIGS.items():
                payload = _evaluate_matrix(
                    intrinsic_train,
                    train_y[inner_train_idx],
                    intrinsic_val,
                    train_y[val_idx],
                    intrinsic_test,
                    test_y,
                    classifier_config=config,
                    seed=seed,
                )
                candidate_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "feature_source": "token_cloud_intrinsic_drift",
                        "feature_variant": "trajectory",
                        "feature_subset": INTRINSIC_SUBSET,
                        "config_name": name,
                        "test_auroc": float(payload["test_auroc"]),
                        "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                        "test_f1": float(payload["test_metrics"]["f1"]),
                    }
                )

            # Dynamic topology distances between layers.
            per_layer_diagrams = {}
            for row in cloud_df.to_dict(orient="records"):
                diag = _compute_diagrams_from_cloud(row["cloud"])
                per_layer_diagrams.setdefault(row["example_id"], {})[int(row["layer"])] = diag
            dyn_rows: list[dict[str, Any]] = []
            for example_id, diag_map in per_layer_diagrams.items():
                if any(layer not in diag_map for layer in layers):
                    continue
                row = feature_df.loc[feature_df["example_id"].eq(example_id)].iloc[0]
                metrics = {
                    "example_id": example_id,
                    "pair_id": row["pair_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label_ambiguous": row["label_ambiguous"],
                }
                if len(layers) < 2:
                    continue
                for dim, prefix in [(0, "h0"), (1, "h1")]:
                    d01 = _safe_wasserstein(diag_map[layers[0]][dim], diag_map[layers[1]][dim])
                    metrics[f"{prefix}_dyn_l{layers[0]}_{layers[1]}"] = d01
                    if len(layers) >= 3:
                        d12 = _safe_wasserstein(diag_map[layers[1]][dim], diag_map[layers[2]][dim])
                        metrics[f"{prefix}_dyn_l{layers[1]}_{layers[2]}"] = d12
                    else:
                        d12 = 0.0
                    metrics[f"{prefix}_dyn_path"] = d01 + d12
                dyn_rows.append(metrics)
            dyn_df = pd.DataFrame(dyn_rows)
            train_dyn = dyn_df.loc[dyn_df["split"].eq("train")]
            test_dyn = dyn_df.loc[dyn_df["split"].eq("test")]
            dyn_cols = [c for c in dyn_df.columns if c.startswith("h0_") or c.startswith("h1_")]
            dyn_train = train_dyn.loc[:, dyn_cols].to_numpy(dtype=float)
            dyn_test = test_dyn.loc[:, dyn_cols].to_numpy(dtype=float)
            dyn_train_inner = dyn_train[inner_train_idx]
            dyn_train_val = dyn_train[val_idx]
            for name, config in FAMILY_CONFIGS.items():
                payload = _evaluate_matrix(
                    dyn_train_inner,
                    train_y[inner_train_idx],
                    dyn_train_val,
                    train_y[val_idx],
                    dyn_test,
                    test_y,
                    classifier_config=config,
                    seed=seed,
                )
                candidate_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "feature_source": "token_cloud_dynamic_topology",
                        "feature_variant": "trajectory",
                        "feature_subset": "dynamic_wasserstein",
                        "config_name": name,
                        "test_auroc": float(payload["test_auroc"]),
                        "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                        "test_f1": float(payload["test_metrics"]["f1"]),
                    }
                )

            # Token trajectory stats.
            token_stats_df = _extract_token_trajectory_features(
                bundle=bundle,
                df=frame,
                layers=layers,
                reducers=reducers,
                config=TOKEN_CLOUD_CONFIG,
            )
            train_tok = token_stats_df.loc[token_stats_df["split"].eq("train")]
            test_tok = token_stats_df.loc[token_stats_df["split"].eq("test")]
            tok_cols = [c for c in token_stats_df.columns if c.startswith("token_")]
            tok_train = train_tok.loc[:, tok_cols].to_numpy(dtype=float)
            tok_test = test_tok.loc[:, tok_cols].to_numpy(dtype=float)
            tok_train_inner = tok_train[inner_train_idx]
            tok_train_val = tok_train[val_idx]
            for name, config in FAMILY_CONFIGS.items():
                payload = _evaluate_matrix(
                    tok_train_inner,
                    train_y[inner_train_idx],
                    tok_train_val,
                    train_y[val_idx],
                    tok_test,
                    test_y,
                    classifier_config=config,
                    seed=seed,
                )
                candidate_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "feature_source": "token_cloud_token_path",
                        "feature_variant": "trajectory",
                        "feature_subset": "token_path_stats",
                        "config_name": name,
                        "test_auroc": float(payload["test_auroc"]),
                        "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                        "test_f1": float(payload["test_metrics"]["f1"]),
                    }
                )

            # Sequence model on intrinsic descriptors.
            train_seq = _prepare_sequences({layer: train_per_layer[layer] for layer in layers})
            test_seq = _prepare_sequences({layer: test_per_layer[layer] for layer in layers})
            train_seq, [test_seq], _, _ = _standardize_sequence(train_seq, test_seq)
            val_seq = train_seq[val_idx]
            train_seq_inner = train_seq[inner_train_idx]
            seq_result = _train_gru(
                SequenceDataset(train_seq_inner, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(test_seq, test_y),
                seed=seed,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "token_cloud_sequence_gru",
                    "config_name": "gru_hidden64",
                    "test_auroc": seq_result["test_auroc"],
                    "test_accuracy": seq_result["test_accuracy"],
                    "test_f1": seq_result["test_f1"],
                }
            )
            hmm_result = _train_class_conditional_hmm(
                SequenceDataset(train_seq_inner, train_y[inner_train_idx]),
                SequenceDataset(val_seq, train_y[val_idx]),
                SequenceDataset(test_seq, test_y),
                seed=seed + 200,
            )
            sequence_rows.append(
                {
                    "model": spec.label,
                    "dataset": dataset_name,
                    "feature_source": "token_cloud_sequence_hmm",
                    "config_name": hmm_result["config_name"],
                    "test_auroc": hmm_result["test_auroc"],
                    "test_accuracy": hmm_result["test_accuracy"],
                    "test_f1": hmm_result["test_f1"],
                }
            )

            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["model", "dataset", "feature_source", "test_auroc"],
        ascending=[True, True, True, False],
    )
    sequence_df = pd.DataFrame(sequence_rows).sort_values(
        ["model", "dataset", "feature_source"],
        ascending=[True, True, True],
    )

    candidate_path = output_root / "trajectory_candidates.parquet"
    sequence_path = output_root / "trajectory_sequence_results.parquet"
    write_parquet(candidate_df, candidate_path)
    write_parquet(sequence_df, sequence_path)

    lines = [
        "# Trajectory/Process Suite Results",
        "",
        f"- Created at: `{utc_now_iso()}`",
        f"- Train per class: `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        f"- Layers: `{TOKEN_CLOUD_CONFIG['candidate_layers']}`",
        "",
        "## Drift/Dynamic/Token-Path Features",
        "",
        "| Model | Dataset | Feature Source | Best Config | AUROC | Acc |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for (model, dataset, feature_source), group in candidate_df.groupby(["model", "dataset", "feature_source"]):
        best = group.sort_values(["test_auroc", "test_accuracy"], ascending=[False, False]).iloc[0]
        lines.append(
            f"| {model} | {dataset} | {feature_source} | {best['config_name']} | {best['test_auroc']:.4f} | {best['test_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Sequence Model Results",
            "",
            "| Model | Dataset | Feature Source | Config | AUROC | Acc |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in sequence_df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['feature_source']} | {row.get('config_name', '-') } | {row['test_auroc']:.4f} | {row['test_accuracy']:.4f} |"
        )
    report_path = output_root / "trajectory_suite_report.md"
    write_markdown(report_path, "\n".join(lines) + "\n")

    metadata_path = output_root / "trajectory_suite_metadata.json"
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "candidate_layers": list(TOKEN_CLOUD_CONFIG["candidate_layers"]),
            "candidate_path": str(candidate_path),
            "sequence_path": str(sequence_path),
            "report_path": str(report_path),
        },
    )
    return {
        "candidate_path": str(candidate_path),
        "sequence_path": str(sequence_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/trajectory_suite_large_1000_2000",
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()))
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_suite(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )
