"""Recompute protocol-aligned CAA summaries from final result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence

from caa_exact_replication.run import DEFAULT_OUTPUT_ROOT, _matching_probability


RESULT_PATTERN = re.compile(
    r"^results_layer=(?P<layer>\d+)_multiplier=(?P<multiplier>-?\d+(?:\.\d+)?)_"
    r"behavior=(?P<behavior>.+?)_type=(?P<eval_type>ab|open_ended)"
    r"(?:_system_prompt=(?P<system_prompt>pos|neg))?_use_base_model=False_"
    r"model_size=(?P<model_size>7b|13b)\.json$"
)


def _metadata(path: Path) -> dict:
    match = RESULT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected CAA result filename: {path.name}")
    result = match.groupdict()
    result["layer"] = int(result["layer"])
    result["multiplier"] = float(result["multiplier"])
    result["system_prompt"] = result["system_prompt"] or "none"
    return result


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_ab(output_root: Path) -> list[dict]:
    rows = []
    for path in sorted((output_root / "results").glob("*/*_type=ab_*.json")):
        metadata = _metadata(path)
        data = json.loads(path.read_text())
        scores = [_matching_probability(item) for item in data]
        rows.append(
            {
                **metadata,
                "n": len(data),
                "mean_matching_probability": sum(scores) / len(scores),
                "result_path": str(path.relative_to(output_root)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["model_size"],
            row["behavior"],
            row["layer"],
            row["system_prompt"],
            row["multiplier"],
        )
    )
    return rows


def summarize_open_ended(output_root: Path) -> list[dict]:
    rows = []
    score_root = output_root / "open_ended_scores"
    for path in sorted(score_root.glob("*/*_type=open_ended_*.json")):
        metadata = _metadata(path)
        data = json.loads(path.read_text())
        scores = [float(item["score"]) for item in data if "score" in item]
        rows.append(
            {
                **metadata,
                "n": len(data),
                "n_scored": len(scores),
                "mean_local_judge_score": (
                    sum(scores) / len(scores) if scores else ""
                ),
                "result_path": str(path.relative_to(output_root)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["model_size"],
            row["behavior"],
            row["layer"],
            row["multiplier"],
        )
    )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    ab_rows = summarize_ab(args.output_root)
    open_rows = summarize_open_ended(args.output_root)
    summary_root = args.output_root / "summaries"
    _write_csv(ab_rows, summary_root / "ab_recomputed.csv")
    _write_csv(open_rows, summary_root / "local_open_ended_recomputed.csv")
    print(
        json.dumps(
            {
                "ab_settings": len(ab_rows),
                "open_ended_settings": len(open_rows),
                "open_ended_fully_scored": sum(
                    row["n"] == row["n_scored"] for row in open_rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

