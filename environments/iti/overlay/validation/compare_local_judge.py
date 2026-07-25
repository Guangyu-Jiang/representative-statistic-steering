#!/usr/bin/env python3
"""Compare local generation-judge outcomes with paired bootstrap intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "truthful": "local_truthful_acc",
    "informative": "local_informative_acc",
    "joint_truth_info": "local_truth_info_acc",
}


def fold_balanced_paired_bootstrap(
    fold_differences: list[np.ndarray], *, samples: int, seed: int
) -> tuple[float, float, float, float]:
    """Bootstrap the unweighted mean of fold-level paired differences."""

    observed = float(np.mean([differences.mean() for differences in fold_differences]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        draws[draw] = np.mean(
            [
                differences[
                    rng.integers(0, len(differences), size=len(differences))
                ].mean()
                for differences in fold_differences
            ]
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high), float(np.mean(draws <= 0.0))


def product_of_means_bootstrap(
    frames: list[pd.DataFrame], *, samples: int, seed: int
) -> tuple[float, float, float, float, float, float]:
    """Compare paper-style mean(truth) * mean(info) with paired fold sampling."""

    truth = "local_truthful_acc"
    info = "local_informative_acc"

    def score(suffix: str) -> float:
        truth_mean = np.mean(
            [frame[f"{truth}_{suffix}"].mean() for frame in frames]
        )
        info_mean = np.mean([frame[f"{info}_{suffix}"].mean() for frame in frames])
        return float(truth_mean * info_mean)

    reference = score("reference")
    candidate = score("candidate")
    observed = candidate - reference
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_folds = []
        for frame in frames:
            indices = rng.integers(0, len(frame), size=len(frame))
            sampled_folds.append(frame.iloc[indices])
        truth_reference = np.mean(
            [frame[f"{truth}_reference"].mean() for frame in sampled_folds]
        )
        info_reference = np.mean(
            [frame[f"{info}_reference"].mean() for frame in sampled_folds]
        )
        truth_candidate = np.mean(
            [frame[f"{truth}_candidate"].mean() for frame in sampled_folds]
        )
        info_candidate = np.mean(
            [frame[f"{info}_candidate"].mean() for frame in sampled_folds]
        )
        reference_draw = truth_reference * info_reference
        candidate_draw = truth_candidate * info_candidate
        draws[draw] = candidate_draw - reference_draw
    low, high = np.quantile(draws, [0.025, 0.975])
    return (
        reference,
        candidate,
        observed,
        float(low),
        float(high),
        float(np.mean(draws <= 0.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_aligned(reference_paths: list[Path], candidate_paths: list[Path]) -> list[pd.DataFrame]:
    if len(reference_paths) != len(candidate_paths):
        raise ValueError("Reference and candidate file counts must match")
    aligned = []
    for fold, (reference_path, candidate_path) in enumerate(
        zip(reference_paths, candidate_paths)
    ):
        reference = pd.read_csv(reference_path)
        candidate = pd.read_csv(candidate_path)
        required = {"dataset_index", "local_judge_raw", *METRICS.values()}
        for name, frame in (("reference", reference), ("candidate", candidate)):
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"Fold {fold} {name} is missing columns: {missing}")
            completed = (
                frame["local_judge_raw"].fillna("").astype(str).str.strip().ne("")
            )
            if not completed.all():
                raise ValueError(
                    f"Fold {fold} {name} has {(~completed).sum()} incomplete judge rows"
                )
        joined = reference[["dataset_index", *METRICS.values()]].merge(
            candidate[["dataset_index", *METRICS.values()]],
            on="dataset_index",
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        if len(joined) != len(reference) or len(joined) != len(candidate):
            raise ValueError(f"Fold {fold} reference and candidate rows are not aligned")
        aligned.append(joined)
    return aligned


def compare_frames(
    frames: list[pd.DataFrame], *, samples: int, seed: int
) -> pd.DataFrame:
    rows = []
    for metric, column in METRICS.items():
        differences = [
            (frame[f"{column}_candidate"] - frame[f"{column}_reference"]).to_numpy(float)
            for frame in frames
        ]
        difference, low, high, p_nonpositive = fold_balanced_paired_bootstrap(
            differences,
            samples=samples,
            seed=seed,
        )
        reference = float(
            np.mean([frame[f"{column}_reference"].mean() for frame in frames])
        )
        candidate = float(
            np.mean([frame[f"{column}_candidate"].mean() for frame in frames])
        )
        rows.append(
            {
                "metric": metric,
                "n": sum(len(frame) for frame in frames),
                "reference_mean": reference,
                "candidate_mean": candidate,
                "difference": difference,
                "ci95_low": low,
                "ci95_high": high,
                "p_nonpositive": p_nonpositive,
            }
        )
    reference, candidate, difference, low, high, p_nonpositive = (
        product_of_means_bootstrap(frames, samples=samples, seed=seed)
    )
    rows.insert(
        2,
        {
            "metric": "truth_x_info",
            "n": sum(len(frame) for frame in frames),
            "reference_mean": reference,
            "candidate_mean": candidate,
            "difference": difference,
            "ci95_low": low,
            "ci95_high": high,
            "p_nonpositive": p_nonpositive,
        },
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    result = compare_frames(
        load_aligned(args.reference, args.candidate),
        samples=args.samples,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
