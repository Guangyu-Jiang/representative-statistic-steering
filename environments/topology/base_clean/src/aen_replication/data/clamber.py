"""CLAMBER benchmark preparation for binary ambiguity detection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_record(line: str) -> dict[str, Any]:
    payload = json.loads(line)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("CLAMBER record must decode to a mapping.")
    return payload


def _iter_clamber_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(_parse_record(line))
    return records


def _format_text(question: str, context: str) -> str:
    question = question.strip()
    context = context.strip()
    if not context:
        return question
    return f"Question: {question}\nContext: {context}"


def _assign_stratified_split(df: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    split_by_id: dict[str, str] = {}
    for _, group in df.groupby(["label_ambiguous", "subclass"], dropna=False, sort=True):
        pair_ids = group["pair_id"].astype(str).tolist()
        rng.shuffle(pair_ids)
        if len(pair_ids) == 1:
            train_count = 1
        else:
            train_count = int(round(len(pair_ids) * train_fraction))
            train_count = max(1, min(len(pair_ids) - 1, train_count))
        train_ids = set(pair_ids[:train_count])
        for pair_id in pair_ids:
            split_by_id[pair_id] = "train" if pair_id in train_ids else "test"
    out = df.copy()
    out["split"] = out["pair_id"].map(split_by_id)
    return out.reset_index(drop=True)


def build_clamber_pairs(
    source_path: str | Path,
    seed: int,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    """Build a binary ambiguity dataset from the released CLAMBER benchmark."""

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(_iter_clamber_records(source_path)):
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        context = record.get("context", "")
        if context is None:
            context = ""
        if not isinstance(context, str):
            context = str(context)
        label = int(record.get("require_clarification", 0))
        category = str(record.get("category", "unknown"))
        subclass = str(record.get("subclass", "unknown"))
        pair_id = f"clamber__{idx:05d}"
        rows.append(
            {
                "example_id": pair_id,
                "pair_id": pair_id,
                "dataset": "clamber",
                "text": _format_text(question, context),
                "label_ambiguous": label,
                "source_id": str(idx),
                "context_type": subclass,
                "source_question": question.strip(),
                "context": context.strip(),
                "clarifying_question": str(record.get("clarifying_question", "") or "").strip(),
                "category": category,
                "subclass": subclass,
                "require_clarification": label,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable CLAMBER rows were found in {source_path}")
    return _assign_stratified_split(df, train_fraction=train_fraction, seed=seed)
