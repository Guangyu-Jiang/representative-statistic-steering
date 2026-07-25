#!/usr/bin/env python3
"""Merge disjoint causal-head evaluation shards for one fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_frames(frames: list[pd.DataFrame], expected_rows: int | None) -> pd.DataFrame:
    if not frames:
        raise ValueError("No result shards were provided")
    columns = list(frames[0].columns)
    for index, frame in enumerate(frames[1:], start=1):
        if list(frame.columns) != columns:
            raise ValueError(f"Shard {index} has different columns")
    merged = pd.concat(frames, ignore_index=True)
    if merged["dataset_index"].duplicated().any():
        raise ValueError("Result shards contain duplicate dataset indices")
    merged = merged.sort_values("dataset_index").reset_index(drop=True)
    if expected_rows is not None and len(merged) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(merged)}")
    return merged


def main() -> None:
    args = parse_args()
    merged = merge_frames([pd.read_csv(path) for path in args.inputs], args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
