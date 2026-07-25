#!/usr/bin/env python3
"""Audit generated-answer diversity, validity, and intervention diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIAGNOSTIC_SUFFIXES = (
    "relative_action_norm",
    "intervention_rate",
    "pre_target_error",
    "post_target_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--answer-columns", nargs="+", required=True)
    parser.add_argument("--baseline-column", default="baseline_answer")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_frames(
    frames: list[pd.DataFrame],
    answer_columns: list[str],
    baseline_column: str,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one generation frame is required")
    frame = pd.concat(frames, ignore_index=True)
    required = {"dataset_index", baseline_column, *answer_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Generation frames are missing columns: {missing}")
    if frame["dataset_index"].duplicated().any():
        raise ValueError("dataset_index must be unique across generation frames")

    baseline = frame[baseline_column].fillna("").astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for answer_column in answer_columns:
        answers = frame[answer_column].fillna("").astype(str).str.strip()
        nonempty = answers.ne("")
        value_counts = answers[nonempty].value_counts()
        row: dict[str, object] = {
            "answer_column": answer_column,
            "n": len(frame),
            "nonempty_answers": int(nonempty.sum()),
            "empty_answers": int((~nonempty).sum()),
            "unique_answers": int(value_counts.size),
            "unique_rate": float(value_counts.size / len(frame)),
            "max_duplicate_count": int(value_counts.max()) if not value_counts.empty else 0,
            "mean_characters": float(answers.str.len().mean()),
            "changed_from_baseline": int(answers.ne(baseline).sum()),
            "changed_rate": float(answers.ne(baseline).mean()),
        }
        prefix = answer_column.removesuffix("_answer")
        for suffix in DIAGNOSTIC_SUFFIXES:
            column = f"{prefix}_{suffix}"
            row[suffix] = (
                float(pd.to_numeric(frame[column], errors="coerce").mean())
                if column in frame
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    result = audit_frames(
        [pd.read_csv(path) for path in args.inputs],
        args.answer_columns,
        args.baseline_column,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
