"""Pure utilities for topology-local FalseQA steering."""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from aen_replication.eval.falseqa_topology import paired_index_plan


H0_THREE = (
    "h0_mean_persistence",
    "h0_persistence_entropy",
    "h0_top5_persistence_fraction",
)

FALSEQA_JUDGE_LABELS = (
    "GROUNDED_REBUTTAL",
    "GENERIC_REJECTION",
    "PREMISE_ACCEPTANCE",
    "NEITHER",
)


def parse_falseqa_judge_label(text: str) -> str:
    """Parse local-judge labels even when the model omits the requested XML tags."""

    upper = str(text).upper()
    tagged = re.search(r"<label>\s*([A-Z_]+)\s*</label>", upper)
    if tagged and tagged.group(1) in FALSEQA_JUDGE_LABELS:
        return tagged.group(1)
    matches = [label for label in FALSEQA_JUDGE_LABELS if label in upper]
    return matches[0] if len(matches) == 1 else "NEITHER"


def falseqa_reference_variants(value: Any) -> list[str]:
    """Normalize FalseQA's scalar or stringified-list reference rebuttals."""

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = text
        else:
            parsed = text
    if isinstance(parsed, (list, tuple, set)):
        variants = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        variants = [str(parsed).strip()] if str(parsed).strip() else []
    return variants or [""]


def nli_gated_falseqa_label(
    llm_label: str,
    max_entailment: float,
    *,
    threshold: float,
) -> str:
    """Require reference entailment for grounded labels while preserving other behavior classes."""

    if float(max_entailment) >= float(threshold):
        return "GROUNDED_REBUTTAL"
    if llm_label == "GENERIC_REJECTION":
        return "GENERIC_REJECTION"
    if llm_label == "NEITHER":
        return "NEITHER"
    return "PREMISE_ACCEPTANCE"


def build_topology_tensor(
    metadata: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return examples x layers x H0-statistics in metadata row order."""

    layers = sorted(int(layer) for layer in feature_frame["layer"].unique())
    example_ids = metadata["example_id"].astype(str).tolist()
    matrices: list[np.ndarray] = []
    for statistic in H0_THREE:
        pivot = feature_frame.pivot(index="example_id", columns="layer", values=statistic)
        pivot = pivot.reindex(index=example_ids, columns=layers)
        if pivot.isna().any().any():
            raise ValueError(f"Missing values while constructing {statistic} topology tensor")
        matrices.append(pivot.to_numpy(dtype=np.float32))
    return np.stack(matrices, axis=-1), layers, example_ids


def select_layer_with_train_validation(
    metadata: pd.DataFrame,
    topology_tensor: np.ndarray,
    layers: list[int],
    *,
    split_column: str,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[int, pd.DataFrame]:
    """Select an intervention layer without consulting held-out test pairs."""

    train_mask = metadata[split_column].eq("train").to_numpy()
    train_metadata = metadata.loc[train_mask].reset_index(drop=True)
    train_tensor = np.asarray(topology_tensor[train_mask], dtype=np.float32)
    pair_metadata, first_indices, second_indices = paired_index_plan(train_metadata, seed=seed)
    pair_ids = pair_metadata["pair_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed + 173)
    shuffled = pair_ids[rng.permutation(len(pair_ids))]
    validation_count = max(1, int(round(validation_fraction * len(shuffled))))
    validation_ids = set(shuffled[:validation_count])
    fit_mask = ~pair_metadata["pair_id"].isin(validation_ids).to_numpy()
    validation_mask = ~fit_mask
    labels = pair_metadata["label_false_first"].to_numpy(dtype=int)

    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        per_example = train_tensor[:, layer_index, :]
        differences = per_example[first_indices] - per_example[second_indices]
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(differences[fit_mask])
        x_validation = scaler.transform(differences[validation_mask])
        classifier = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        )
        classifier.fit(x_fit, labels[fit_mask])
        scores = classifier.decision_function(x_validation)
        predictions = (scores >= 0.0).astype(int)
        rows.append(
            {
                "layer": int(layer),
                "validation_n": int(validation_mask.sum()),
                "validation_accuracy": float(accuracy_score(labels[validation_mask], predictions)),
                "validation_auroc": float(roc_auc_score(labels[validation_mask], scores)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["validation_auroc", "validation_accuracy", "layer"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return int(metrics.iloc[0]["layer"]), metrics


def topology_neighbor_matrices(
    metadata: pd.DataFrame,
    topology_tensor: np.ndarray,
    *,
    split_column: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, StandardScaler]:
    """Build standardized all-layer H0-three train/query matrices."""

    flattened = np.asarray(topology_tensor, dtype=np.float32).reshape(len(metadata), -1)
    train_mask = metadata[split_column].eq("train").to_numpy()
    scaler = StandardScaler()
    scaler.fit(flattened[train_mask])
    standardized = scaler.transform(flattened).astype(np.float32, copy=False)
    train_false_mask = train_mask & metadata["label_false_premise"].eq(1).to_numpy()
    test_false_mask = metadata[split_column].eq("test").to_numpy() & metadata[
        "label_false_premise"
    ].eq(1).to_numpy()
    return (
        metadata.loc[train_false_mask].reset_index(drop=True),
        standardized[train_false_mask],
        metadata.loc[test_false_mask].reset_index(drop=True),
        standardized[test_false_mask],
        scaler,
    )


def paired_hidden_differences(
    train_false_metadata: pd.DataFrame,
    hidden_by_example: dict[str, np.ndarray],
    full_metadata: pd.DataFrame,
) -> np.ndarray:
    """Return false-minus-corrected activation contrasts in train-false order."""

    corrected_by_pair = (
        full_metadata.loc[full_metadata["label_false_premise"].eq(0), ["pair_id", "example_id"]]
        .drop_duplicates("pair_id")
        .set_index("pair_id")["example_id"]
        .astype(str)
        .to_dict()
    )
    differences: list[np.ndarray] = []
    for row in train_false_metadata.to_dict(orient="records"):
        false_id = str(row["example_id"])
        corrected_id = corrected_by_pair[str(row["pair_id"])]
        differences.append(
            np.asarray(hidden_by_example[false_id], dtype=np.float32)
            - np.asarray(hidden_by_example[corrected_id], dtype=np.float32)
        )
    return np.vstack(differences)


def build_paired_local_directions(
    train_topology: np.ndarray,
    query_topology: np.ndarray,
    train_hidden_differences: np.ndarray,
    *,
    neighbor_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average paired activation contrasts over topology-nearest neighbors."""

    k = min(max(1, int(neighbor_k)), len(train_topology))
    nearest = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nearest.fit(train_topology)
    distances, indices = nearest.kneighbors(query_topology)
    directions = np.stack(
        [np.asarray(train_hidden_differences[row_indices]).mean(axis=0) for row_indices in indices],
        axis=0,
    ).astype(np.float32, copy=False)
    return directions, indices.astype(int), distances.astype(np.float32)


def response_quality(response: str) -> dict[str, Any]:
    """Conservative non-semantic checks for empty or degenerate generations."""

    normalized = " ".join(str(response).strip().split())
    words = normalized.split()
    if not words:
        return {
            "response_word_count": 0,
            "response_empty": True,
            "response_repetition_ratio": 1.0,
            "response_valid": False,
        }
    counts: dict[str, int] = {}
    for word in words:
        key = word.lower()
        counts[key] = counts.get(key, 0) + 1
    repetition_ratio = max(counts.values()) / len(words)
    valid = len(words) >= 3 and repetition_ratio < 0.7
    return {
        "response_word_count": int(len(words)),
        "response_empty": False,
        "response_repetition_ratio": float(repetition_ratio),
        "response_valid": bool(valid),
    }
