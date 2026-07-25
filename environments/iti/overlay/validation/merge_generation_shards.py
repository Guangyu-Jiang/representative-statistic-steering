#!/usr/bin/env python3
"""Merge resumable generation shards while enforcing complete, unique rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--enrich-from",
        nargs="*",
        type=Path,
        help="copy additional row-aligned columns from complete artifacts",
    )
    parser.add_argument(
        "--allow-empty-answers",
        action="store_true",
        help="retain completed rows whose generated answer is empty",
    )
    return parser.parse_args()


def merge_generation_frames(
    frames: list[pd.DataFrame], *, allow_empty_answers: bool = False
) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one generation frame is required")
    answer_columns = [column for column in frames[0] if column.endswith("_answer")]
    if not answer_columns:
        raise ValueError("No generation answer columns were found")

    required = {"dataset_index", *answer_columns}
    complete_frames = []
    for index, frame in enumerate(frames):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Frame {index} is missing columns: {missing}")
        if allow_empty_answers:
            complete = frame.copy()
        else:
            complete = frame.loc[
                frame[answer_columns]
                .fillna("")
                .astype(str)
                .apply(lambda column: column.str.strip().ne(""))
                .all(axis=1)
            ].copy()
        complete_frames.append(complete)

    merged = pd.concat(complete_frames, ignore_index=True)
    duplicate_ids = merged.loc[merged["dataset_index"].duplicated(), "dataset_index"].unique()
    for dataset_index in duplicate_ids:
        rows = merged.loc[merged["dataset_index"].eq(dataset_index)]
        for column in answer_columns:
            if rows[column].astype(str).nunique() != 1:
                raise ValueError(
                    f"Conflicting answers for dataset_index={dataset_index}, column={column}"
                )
    return merged.drop_duplicates("dataset_index", keep="first").reset_index(drop=True)


def enrich_generation_frame(base: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Add columns from a complete source after strict dataset-index alignment."""

    if source["dataset_index"].duplicated().any():
        raise ValueError("Enrichment artifact has duplicate dataset_index values")
    source_by_id = source.set_index("dataset_index")
    missing_ids = sorted(set(base["dataset_index"]).difference(source_by_id.index))
    if missing_ids:
        raise ValueError(f"Enrichment artifact is missing {len(missing_ids)} requested rows")
    result = base.copy()
    for column in source:
        if column == "dataset_index":
            continue
        aligned = result["dataset_index"].map(source_by_id[column])
        if column in result:
            if column.endswith("_answer"):
                left = result[column].fillna("").astype(str)
                right = aligned.fillna("").astype(str)
                if not left.equals(right):
                    raise ValueError(f"Conflicting enrichment column: {column}")
            continue
        result[column] = aligned
    return result


def main() -> None:
    args = parse_args()
    merged = merge_generation_frames(
        [pd.read_csv(path) for path in args.inputs],
        allow_empty_answers=args.allow_empty_answers,
    )
    for path in args.enrich_from or []:
        merged = enrich_generation_frame(merged, pd.read_csv(path))
    if args.expected_rows is not None and len(merged) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} complete rows, found {len(merged)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} complete rows to {args.output}")


if __name__ == "__main__":
    main()
