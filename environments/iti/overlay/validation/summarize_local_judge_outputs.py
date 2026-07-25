#!/usr/bin/env python3
"""Rebuild paper-compatible local-judge summaries from checkpointed CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judged", nargs="+", type=Path)
    parser.add_argument("--judge-model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="summarize checkpoint rows with missing judge output as negative labels",
    )
    return parser.parse_args()


def summarize_frame(
    frame: pd.DataFrame, *, allow_incomplete: bool = False
) -> dict[str, float | int]:
    required = {
        "local_judge_raw",
        "local_truthful",
        "local_informative",
        "local_truthful_acc",
        "local_informative_acc",
        "local_truth_info_acc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Judged frame is missing columns: {missing}")
    completed = frame["local_judge_raw"].fillna("").astype(str).str.strip().ne("")
    if not allow_incomplete and not completed.all():
        raise ValueError(f"Judged frame has {(~completed).sum()} incomplete rows")
    truthful = float(frame["local_truthful_acc"].mean())
    informative = float(frame["local_informative_acc"].mean())
    return {
        "n": len(frame),
        "truthful": truthful,
        "informative": informative,
        "truth_x_info": truthful * informative,
        "joint_truth_info": float(frame["local_truth_info_acc"].mean()),
        "parse_rate": float(
            (
                frame["local_truthful"].ne("UNKNOWN")
                & frame["local_informative"].ne("UNKNOWN")
            ).mean()
        ),
    }


def parse_filename(path: Path) -> tuple[str, str, str]:
    pieces = path.stem.split("__")
    if len(pieces) != 3 or not pieces[2].endswith("_judged"):
        raise ValueError(f"Cannot infer input, answer column, and judge from {path}")
    return pieces[0], pieces[1], pieces[2].removesuffix("_judged")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in args.judged:
        input_name, answer_column, judge_tag = parse_filename(path)
        rows.append(
            {
                "input": input_name,
                "judge_model": args.judge_model or judge_tag,
                "answer_column": answer_column,
                "output": str(path),
                **summarize_frame(
                    pd.read_csv(path), allow_incomplete=args.allow_incomplete
                ),
            }
        )

    summary = pd.DataFrame(rows)
    combined = []
    for (judge_model, answer_column), group in summary.groupby(
        ["judge_model", "answer_column"], sort=False
    ):
        total = int(group["n"].sum())
        # Match the official two-fold report: average each fold metric first.
        truthful = float(group["truthful"].mean())
        informative = float(group["informative"].mean())
        combined.append(
            {
                "input": "COMBINED",
                "judge_model": judge_model,
                "answer_column": answer_column,
                "output": "",
                "n": total,
                "truthful": truthful,
                "informative": informative,
                "truth_x_info": truthful * informative,
                "joint_truth_info": float(group["joint_truth_info"].mean()),
                "parse_rate": float(group["parse_rate"].mean()),
            }
        )
    result = pd.concat([summary, pd.DataFrame(combined)], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
