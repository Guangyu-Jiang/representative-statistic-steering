#!/usr/bin/env python3
"""Compute fold-stratified paired bootstrap intervals for MC interventions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--candidate-results",
        nargs="+",
        type=Path,
        help="optional fold-aligned files containing candidate columns",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def paired_bootstrap(
    fold_differences: list[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Return mean difference, 95% interval, and P(difference <= 0)."""

    rng = np.random.default_rng(seed)
    observed = float(np.concatenate(fold_differences).mean())
    draws = np.empty(samples, dtype=np.float64)
    total = sum(len(values) for values in fold_differences)
    for draw in range(samples):
        value = 0.0
        for differences in fold_differences:
            indices = rng.integers(0, len(differences), size=len(differences))
            value += float(differences[indices].sum())
        draws[draw] = value / total
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high), float(np.mean(draws <= 0.0))


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.results]
    if args.candidate_results:
        if len(args.candidate_results) != len(frames):
            raise ValueError("Candidate and reference result file counts must match")
        candidate_frames = [pd.read_csv(path) for path in args.candidate_results]
        for index, (frame, candidates) in enumerate(zip(frames, candidate_frames)):
            if list(frame["dataset_index"]) != list(candidates["dataset_index"]):
                raise ValueError(f"Fold {index} candidate rows are not aligned")
            for column in candidates:
                if column not in frame:
                    frame[column] = candidates[column].to_numpy()
    available = [column[:-4] for column in frames[0] if column.endswith(" MC1")]
    settings = args.settings or [setting for setting in available if setting != args.reference]
    rows = []
    for setting in settings:
        for metric in ("MC1", "MC2"):
            reference_column = f"{args.reference} {metric}"
            setting_column = f"{setting} {metric}"
            differences = [
                (frame[setting_column] - frame[reference_column]).dropna().to_numpy(float)
                for frame in frames
            ]
            difference, low, high, p_nonpositive = paired_bootstrap(
                differences,
                samples=args.samples,
                seed=args.seed,
            )
            reference = np.concatenate(
                [frame[reference_column].dropna().to_numpy(float) for frame in frames]
            )
            candidate = np.concatenate(
                [frame[setting_column].dropna().to_numpy(float) for frame in frames]
            )
            rows.append(
                {
                    "reference": args.reference,
                    "setting": setting,
                    "metric": metric.lower(),
                    "n": len(candidate),
                    "reference_mean": reference.mean(),
                    "setting_mean": candidate.mean(),
                    "difference": difference,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_nonpositive": p_nonpositive,
                }
            )
    result = pd.DataFrame(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
