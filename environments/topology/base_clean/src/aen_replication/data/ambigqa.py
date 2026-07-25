"""AmbigQA contrastive pair construction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def _select_candidate(candidates: list[str], selection_strategy: str, rng: random.Random) -> str | None:
    if not candidates:
        return None
    if selection_strategy == "first":
        return candidates[0]
    if selection_strategy == "random_seeded":
        return rng.choice(candidates)
    raise ValueError(f"Unsupported clear selection strategy: {selection_strategy}")


def _select_rewrite(record: dict[str, Any], rng: random.Random, selection_strategy: str) -> str | None:
    annotations = record.get("annotations", [])
    candidates: list[str] = []
    for annotation in annotations:
        if annotation.get("type") != "multipleQAs":
            continue
        for qa_pair in annotation.get("qaPairs", []):
            question = qa_pair.get("question")
            if isinstance(question, str) and question.strip():
                candidates.append(question.strip())
    candidates = [candidate for candidate in dict.fromkeys(candidates) if candidate != record["question"]]
    return _select_candidate(candidates, selection_strategy=selection_strategy, rng=rng)


def build_ambigqa_pairs(
    train_path: str | Path,
    dev_path: str | Path,
    seed: int,
    selection_strategy: str = "random_seeded",
) -> pd.DataFrame:
    """Build AmbigQA ambiguous/clear pairs from official public data."""

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for path in (train_path, dev_path):
        for record in _load_json(path):
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            rewrite = _select_rewrite(record, rng, selection_strategy=selection_strategy)
            if rewrite is None:
                continue
            source_id = str(record["id"])
            pair_id = f"ambigqa__{source_id}"
            rows.extend(
                [
                    {
                        "example_id": f"{pair_id}__ambiguous",
                        "pair_id": pair_id,
                        "dataset": "ambigqa",
                        "text": question.strip(),
                        "label_ambiguous": 1,
                        "source_id": source_id,
                        "context_type": "rewrite",
                    },
                    {
                        "example_id": f"{pair_id}__clear",
                        "pair_id": pair_id,
                        "dataset": "ambigqa",
                        "text": rewrite,
                        "label_ambiguous": 0,
                        "source_id": source_id,
                        "context_type": "rewrite",
                    },
                ]
            )
    return pd.DataFrame(rows)
