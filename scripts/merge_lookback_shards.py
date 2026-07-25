#!/usr/bin/env python3
"""Merge disjoint resumable Lookback result shards with strict validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repstat_steering.lookback_control import summarize_lookback_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument(
        "--expected-methods",
        nargs="+",
        default=["baseline", "baseline_rerank", "minimum_norm_rerank"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keyed: dict[tuple[int, str], dict] = {}
    shard_configs = []
    for shard in args.shard_dir:
        config_path = shard / "config.json"
        if config_path.exists():
            shard_configs.append(json.loads(config_path.read_text()))
        result_path = shard / "results.jsonl"
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["dataset_index"]), str(row["method"]))
            if key in keyed and keyed[key] != row:
                raise ValueError(f"conflicting duplicate result: {key}")
            keyed[key] = row

    indices = sorted({key[0] for key in keyed})
    method_counts = {
        method: sum(key[1] == method for key in keyed)
        for method in args.expected_methods
    }
    unexpected = sorted({key[1] for key in keyed} - set(args.expected_methods))
    if len(indices) != args.expected_questions:
        raise ValueError(
            f"expected {args.expected_questions} questions, found {len(indices)}"
        )
    if any(count != args.expected_questions for count in method_counts.values()):
        raise ValueError(f"incomplete method counts: {method_counts}")
    if unexpected:
        raise ValueError(f"unexpected methods: {unexpected}")

    method_order = {
        method: index for index, method in enumerate(args.expected_methods)
    }
    rows = [
        keyed[key]
        for key in sorted(keyed, key=lambda key: (key[0], method_order[key[1]]))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)
    summary = summarize_lookback_rows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "output_dir": str(args.output_dir),
            "offset": min(indices),
            "limit": args.expected_questions,
            "indices": indices,
            "methods": args.expected_methods,
            "n_examples": args.expected_questions,
            "merged_shards": [str(path) for path in args.shard_dir],
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(merged_config, indent=2)
    )
    print(
        json.dumps(
            {
                "questions": len(indices),
                "rows": len(rows),
                "method_counts": method_counts,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
