"""Leakage-safe utilities for FalseQA topology classification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


FALSEQA_SPLITS = ("train", "valid", "test")


def load_falseqa_pairs(dataset_root: str | Path) -> pd.DataFrame:
    """Load FalseQA and recover its positional false/corrected pairs.

    In each official CSV, the first half contains false-premise questions and
    the second half contains their corrected counterparts in the same order.
    """

    dataset_root = Path(dataset_root)
    rows: list[dict[str, Any]] = []
    for source_split in FALSEQA_SPLITS:
        path = dataset_root / f"{source_split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing FalseQA split: {path}")
        frame = pd.read_csv(path)
        required = {"question", "answer", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if len(frame) % 2:
            raise ValueError(f"FalseQA split must have an even row count: {path}")

        pair_count = len(frame) // 2
        false_half = frame.iloc[:pair_count].reset_index(drop=True)
        corrected_half = frame.iloc[pair_count:].reset_index(drop=True)
        if not false_half["label"].eq(1).all() or not corrected_half["label"].eq(0).all():
            raise ValueError(
                f"Unexpected FalseQA ordering in {path}; expected false rows then corrected rows."
            )

        for pair_index, (false_row, corrected_row) in enumerate(
            zip(false_half.to_dict(orient="records"), corrected_half.to_dict(orient="records"), strict=True)
        ):
            pair_id = f"{source_split}_{pair_index:04d}"
            for variant, row in (("false", false_row), ("corrected", corrected_row)):
                rows.append(
                    {
                        "example_id": f"{pair_id}__{variant}",
                        "pair_id": pair_id,
                        "source_split": source_split,
                        "pair_index": int(pair_index),
                        "variant": variant,
                        "question": str(row["question"]),
                        "reference_answer": str(row["answer"]),
                        "label_false_premise": int(row["label"]),
                    }
                )

    result = pd.DataFrame(rows)
    counts = result.groupby("pair_id")["label_false_premise"].agg(["count", "sum"])
    if not counts["count"].eq(2).all() or not counts["sum"].eq(1).all():
        raise ValueError("Every FalseQA pair must contain one false and one corrected question.")
    return result


def assign_evaluation_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    seed: int,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Add pair-grouped random and official split assignments."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    pair_table = frame.loc[:, ["pair_id", "source_split"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if max_pairs > 0 and max_pairs < len(pair_table):
        selected = np.sort(rng.choice(len(pair_table), size=int(max_pairs), replace=False))
        pair_table = pair_table.iloc[selected].reset_index(drop=True)
        frame = frame.loc[frame["pair_id"].isin(set(pair_table["pair_id"]))].copy()

    pair_ids = pair_table["pair_id"].astype(str).to_numpy()
    shuffled = pair_ids[rng.permutation(len(pair_ids))]
    train_count = int(round(float(train_fraction) * len(shuffled)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    random_train = set(shuffled[:train_count])

    result = frame.copy()
    result["split_random80"] = np.where(result["pair_id"].isin(random_train), "train", "test")
    result["split_official"] = np.where(result["source_split"].eq("test"), "test", "train")
    result["pca_fit"] = result["split_random80"].eq("train") & result["split_official"].eq("train")
    return result.sort_values(["source_split", "pair_index", "label_false_premise"], ascending=[True, True, False]).reset_index(drop=True)


def h0_features_from_cloud(cloud: np.ndarray) -> dict[str, Any]:
    """Compute exact Euclidean H0 persistence from MST edge lengths."""

    cloud = np.asarray(cloud, dtype=np.float64)
    if cloud.ndim != 2:
        raise ValueError(f"Expected a 2D token cloud, got shape={cloud.shape}")
    if len(cloud) <= 1:
        lifetimes = np.zeros(0, dtype=np.float64)
    else:
        distances = squareform(pdist(cloud, metric="euclidean"))
        tree = minimum_spanning_tree(distances)
        lifetimes = np.asarray(tree.data, dtype=np.float64)
        lifetimes = np.sort(lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)])[::-1]

    if lifetimes.size == 0:
        mean_persistence = 0.0
        entropy = 0.0
        top5_fraction = 0.0
    else:
        mean_persistence = float(lifetimes.mean())
        total = float(lifetimes.sum())
        weights = lifetimes / max(total, 1e-12)
        entropy = 0.0 if len(lifetimes) <= 1 else float(
            -(weights * np.log(weights + 1e-12)).sum() / np.log(len(lifetimes))
        )
        top5_fraction = float(lifetimes[:5].sum() / max(total, 1e-12))
    return {
        "h0_mean_persistence": mean_persistence,
        "h0_persistence_entropy": entropy,
        "h0_top5_persistence_fraction": top5_fraction,
        "h0_lifetimes": lifetimes.astype(np.float32).tolist(),
    }


def stable_pair_orientation(pair_id: str, seed: int) -> bool:
    """Return whether the false question is first in a deterministic random order."""

    digest = hashlib.blake2b(f"{seed}:{pair_id}".encode("utf-8"), digest_size=8).digest()
    return bool(int.from_bytes(digest, byteorder="little") & 1)


def paired_difference_dataset(
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct an orientation-balanced paired-difference classification set."""

    matrix = np.asarray(matrix, dtype=float)
    if len(metadata) != len(matrix):
        raise ValueError("metadata and matrix row counts differ")
    pair_rows, first_indices, second_indices = paired_index_plan(metadata, seed=seed)
    return pair_rows, matrix[first_indices] - matrix[second_indices]


def paired_index_plan(
    metadata: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build reusable row indices for deterministic paired differences."""

    index_by_example = {str(example_id): index for index, example_id in enumerate(metadata["example_id"])}
    pair_rows: list[dict[str, Any]] = []
    first_indices: list[int] = []
    second_indices: list[int] = []
    for pair_id, group in metadata.groupby("pair_id", sort=True):
        if len(group) != 2 or set(group["label_false_premise"].astype(int)) != {0, 1}:
            raise ValueError(f"Malformed FalseQA pair: {pair_id}")
        false_row = group.loc[group["label_false_premise"].eq(1)].iloc[0]
        corrected_row = group.loc[group["label_false_premise"].eq(0)].iloc[0]
        false_index = index_by_example[str(false_row["example_id"])]
        corrected_index = index_by_example[str(corrected_row["example_id"])]
        false_first = stable_pair_orientation(str(pair_id), seed)
        if false_first:
            first_row, second_row = false_row, corrected_row
            first_index, second_index = false_index, corrected_index
            label = 1
        else:
            first_row, second_row = corrected_row, false_row
            first_index, second_index = corrected_index, false_index
            label = 0
        first_indices.append(first_index)
        second_indices.append(second_index)
        pair_rows.append(
            {
                "example_id": f"{pair_id}__paired",
                "pair_id": str(pair_id),
                "source_split": str(false_row["source_split"]),
                "split_random80": str(false_row["split_random80"]),
                "split_official": str(false_row["split_official"]),
                "label_false_first": int(label),
                "first_example_id": str(first_row["example_id"]),
                "second_example_id": str(second_row["example_id"]),
                "first_question": str(first_row["question"]),
                "second_question": str(second_row["question"]),
            }
        )
    return (
        pd.DataFrame(pair_rows),
        np.asarray(first_indices, dtype=int),
        np.asarray(second_indices, dtype=int),
    )


def assert_group_disjoint(frame: pd.DataFrame, split_column: str) -> None:
    """Raise if a pair appears on both sides of an evaluation split."""

    counts = frame.groupby("pair_id")[split_column].nunique()
    if not counts.eq(1).all():
        bad = counts.loc[counts.gt(1)].index.astype(str).tolist()[:5]
        raise AssertionError(f"Pairs cross {split_column}: {bad}")
