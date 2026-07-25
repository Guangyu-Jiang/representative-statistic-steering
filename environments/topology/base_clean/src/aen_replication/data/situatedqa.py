"""SituatedQA contrastive pair construction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


def _iter_jsonl(paths: Iterable[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _select_candidate(candidates: list[str], selection_strategy: str, rng: random.Random) -> str | None:
    if not candidates:
        return None
    if selection_strategy == "first":
        return candidates[0]
    if selection_strategy == "random_seeded":
        return rng.choice(candidates)
    raise ValueError(f"Unsupported clear selection strategy: {selection_strategy}")


def _choose_edited_question(record: dict, rng: random.Random, selection_strategy: str) -> str | None:
    pairs = record.get("context_answer_pairs")
    if not isinstance(pairs, list) or not pairs:
        return None
    candidates = [
        item.get("edited_question", "").strip()
        for item in pairs
        if isinstance(item, dict) and isinstance(item.get("edited_question"), str) and item.get("edited_question").strip()
    ]
    return _select_candidate(candidates, selection_strategy=selection_strategy, rng=rng)


def _is_context_dependent(record: dict) -> bool:
    """Determine whether a raw SituatedQA record should enter the contrastive set.

    Temporal raw data includes an explicit `is_dependent` flag, while the
    geographical raw files in the official release omit that field and contain
    only context-dependent examples. In the latter case, the presence of at
    least one edited question is the best available indicator.
    """

    marker = record.get("is_dependent")
    if marker is not None:
        return bool(marker)
    pairs = record.get("context_answer_pairs")
    return isinstance(pairs, list) and len(pairs) > 0


def build_situatedqa_pairs(
    temp_paths: list[str],
    geo_paths: list[str],
    seed: int,
    selection_strategy: str = "random_seeded",
) -> pd.DataFrame:
    """Build SituatedQA ambiguous/clear pairs from official raw data."""

    rng = random.Random(seed)
    rows: list[dict] = []
    for context_type, paths in (("temp", temp_paths), ("geo", geo_paths)):
        for record in _iter_jsonl(paths):
            if not _is_context_dependent(record):
                continue
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            edited_question = _choose_edited_question(record, rng, selection_strategy=selection_strategy)
            if edited_question is None:
                continue
            source_id = str(record["id"])
            pair_id = f"situatedqa__{context_type}__{source_id}"
            rows.extend(
                [
                    {
                        "example_id": f"{pair_id}__ambiguous",
                        "pair_id": pair_id,
                        "dataset": "situatedqa",
                        "text": question.strip(),
                        "label_ambiguous": 1,
                        "source_id": source_id,
                        "context_type": context_type,
                    },
                    {
                        "example_id": f"{pair_id}__clear",
                        "pair_id": pair_id,
                        "dataset": "situatedqa",
                        "text": edited_question,
                        "label_ambiguous": 0,
                        "source_id": source_id,
                        "context_type": context_type,
                    },
                ]
            )
    return pd.DataFrame(rows)
