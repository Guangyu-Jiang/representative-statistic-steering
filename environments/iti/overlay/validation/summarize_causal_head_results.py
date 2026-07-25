#!/usr/bin/env python3
"""Aggregate causal head-steering summaries without averaging fold sizes equally."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "mc1",
    "mc2",
    "relative_action_norm",
    "intervention_rate",
    "pre_target_error",
    "post_target_error",
    "active_signed_target_error",
    "active_absolute_target_error",
    "active_target_overshoot",
    "clip_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--min-folds", type=int, default=1)
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    valid = frame[column].notna()
    if not valid.any():
        return float("nan")
    return float(np.average(frame.loc[valid, column], weights=frame.loc[valid, "n"]))


def summarize_frames(
    frames: list[pd.DataFrame], *, min_folds: int = 1, min_sources: int = 1
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for setting, group in combined.groupby("setting"):
        folds = group["fold"].nunique()
        sources = group["source"].nunique()
        if folds < min_folds or sources < min_sources:
            continue
        row = {
            "setting": setting,
            "folds": folds,
            "sources": sources,
            "n": int(group["n"].sum()),
        }
        for metric in METRICS:
            row[metric] = weighted_mean(group, metric)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=("setting", "folds", "sources", "n", *METRICS))
    return pd.DataFrame(rows).sort_values(["mc1", "mc2"], ascending=False)


def main() -> None:
    args = parse_args()
    frames = []
    for path in args.summaries:
        current = pd.read_csv(path)
        current["source"] = str(path)
        frames.append(current)
    summary = summarize_frames(
        frames,
        min_folds=args.min_folds,
        min_sources=args.min_sources,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
