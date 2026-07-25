"""Schema helpers for contrastive ambiguity pairs."""

from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[list[str]] = [
    "example_id",
    "pair_id",
    "dataset",
    "split",
    "text",
    "label_ambiguous",
    "source_id",
    "context_type",
]


def validate_pairs(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Contrastive pair DataFrame is empty.")
    if not df["label_ambiguous"].isin([0, 1]).all():
        raise ValueError("label_ambiguous must be binary.")
    pair_counts = df.groupby("pair_id")["split"].nunique()
    if (pair_counts > 1).any():
        raise ValueError("A pair_id spans multiple splits.")


def coerce_pairs(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["label_ambiguous"] = normalized["label_ambiguous"].astype(int)
    validate_pairs(normalized)
    return normalized
