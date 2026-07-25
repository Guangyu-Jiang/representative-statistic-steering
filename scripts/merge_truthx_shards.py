#!/usr/bin/env python3
"""Merge disjoint TruthX JSONL shards and recompute a verified summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repstat_steering.truthx_control import (
    TruthXInterventionConfig,
    summarize_truthx_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path / "results.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_configuration(paths: list[Path]) -> TruthXInterventionConfig:
    configurations = []
    for path in paths:
        summary_path = path / "summary.json"
        if summary_path.exists():
            with open(summary_path) as handle:
                configurations.append(json.load(handle)["configuration"])
    if not configurations:
        raise RuntimeError("At least one completed shard must contain summary.json")
    if any(value != configurations[0] for value in configurations[1:]):
        raise RuntimeError("Shard configurations do not match")
    return TruthXInterventionConfig(**configurations[0])


def main() -> None:
    args = parse_args()
    shard_dirs = [Path(value) for value in args.shard_dir]
    by_index = {}
    for shard_dir in shard_dirs:
        for row in read_rows(shard_dir):
            index = int(row["dataset_index"])
            if index in by_index and row != by_index[index]:
                raise RuntimeError(f"Conflicting duplicate dataset index {index}")
            by_index[index] = row
    expected_indices = set(range(args.expected_count))
    actual_indices = set(by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise RuntimeError(
            f"Incomplete shards: missing={missing[:20]} extra={extra[:20]}"
        )

    rows = [by_index[index] for index in range(args.expected_count)]
    configuration = read_configuration(shard_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    summary = summarize_truthx_rows(rows, configuration)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
