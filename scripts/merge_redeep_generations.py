#!/usr/bin/env python3
"""Merge ReDeEP generation shards while validating method/source coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: dict[tuple[int, str], dict[str, object]] = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (int(row["source_id"]), str(row["method"]))
                if key in records and records[key]["response"] != row["response"]:
                    raise ValueError(f"conflicting duplicate generation for {key}")
                records[key] = row
    methods = sorted({method for _, method in records})
    source_ids = sorted({source_id for source_id, _ in records})
    missing = [
        (source_id, method)
        for source_id in source_ids
        for method in methods
        if (source_id, method) not in records
    ]
    if missing:
        raise ValueError(f"missing method/source rows: {missing[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for source_id in source_ids:
            for method in methods:
                row = records[(source_id, method)]
                row["split"] = "evaluation"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(records)} rows for {len(source_ids)} sources and "
        f"{len(methods)} methods to {args.output}"
    )


if __name__ == "__main__":
    main()
