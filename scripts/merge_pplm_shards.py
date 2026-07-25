#!/usr/bin/env python3
"""Merge completed PPLM generation shards with uniqueness validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repstat_steering.pplm_control import distinct_ngram_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(Path(path) / "generations.csv") for path in args.shard_dir]
    frame = pd.concat(frames, ignore_index=True)
    key = ["method", "target_label", "prefix", "seed"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise RuntimeError(f"Duplicate generation keys:\n{frame.loc[duplicates, key]}")
    if len(frame) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} rows, found {len(frame)}")
    configuration_counts = frame.groupby(
        ["method", "target_label"]
    )["configuration"].nunique()
    if (configuration_counts != 1).any():
        raise RuntimeError(
            "Shard generation configurations do not match within method/target: "
            f"{configuration_counts[configuration_counts != 1].to_dict()}"
        )

    frame = frame.sort_values(key).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "generations.csv", index=False)
    with open(output_dir / "generations.jsonl", "w") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row) + "\n")

    rows = []
    for (method, target), group in frame.groupby(["method", "target_label"]):
        probability_column = f"{target}_probability"
        rows.append(
            {
                "method": method,
                "target_label": target,
                "n": len(group),
                "mean_target_probability": group[probability_column].mean(),
                "target_probability_ge_0p5": (
                    group[probability_column] >= 0.5
                ).mean(),
                "mean_relative_cache_change": group[
                    "mean_relative_cache_change"
                ].mean(),
                "distinct_1": distinct_ngram_fraction(group["continuation"], 1),
                "distinct_2": distinct_ngram_fraction(group["continuation"], 2),
                "distinct_3": distinct_ngram_fraction(group["continuation"], 3),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
