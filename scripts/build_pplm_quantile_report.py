#!/usr/bin/env python3
"""Build development and paired validation reports for PPLM quantile targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["target_label", "prefix", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-root",
        type=Path,
        default=Path("artifacts/pplm_sentiment/quantile_target_development"),
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "artifacts/pplm_sentiment/corrected_accumulated_validation_seeds22_33/"
            "external_eval/evaluated_generations.csv"
        ),
    )
    parser.add_argument(
        "--shift3",
        type=Path,
        default=Path(
            "artifacts/pplm_sentiment/"
            "corrected_accumulated_shift3_validation_seeds22_33/"
            "external_eval/evaluated_generations.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/pplm_quantile_targets"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize(frame: pd.DataFrame, method: str) -> dict[str, object]:
    success = frame["external_target_probability"] >= 0.5
    return {
        "method": method,
        "n": len(frame),
        "external_target_probability": frame["external_target_probability"].mean(),
        "external_success": success.mean(),
        "perplexity": frame["perplexity"].mean(),
        "mean_relative_cache_change": frame["mean_relative_cache_change"].mean(),
        "mean_token_kl": (
            frame["mean_token_kl"].mean()
            if "mean_token_kl" in frame
            else float("nan")
        ),
        "unique_continuations": frame["continuation"].nunique(),
    }


def bootstrap_interval(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_row(
    candidate: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_name: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    merged = candidate.merge(
        comparison,
        on=KEY,
        suffixes=("_candidate", "_comparison"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(comparison):
        raise RuntimeError(f"incomplete pairing against {comparison_name}")
    row: dict[str, object] = {
        "candidate": "quantile_q0.5",
        "comparison": comparison_name,
        "paired_n": len(merged),
    }
    metrics = {
        "target_probability": (
            merged["external_target_probability_candidate"].to_numpy()
            - merged["external_target_probability_comparison"].to_numpy()
        ),
        "success": (
            (merged["external_target_probability_candidate"] >= 0.5).astype(float).to_numpy()
            - (merged["external_target_probability_comparison"] >= 0.5).astype(float).to_numpy()
        ),
        "perplexity": (
            merged["perplexity_candidate"].to_numpy()
            - merged["perplexity_comparison"].to_numpy()
        ),
        "relative_cache_change": (
            merged["mean_relative_cache_change_candidate"].to_numpy()
            - merged["mean_relative_cache_change_comparison"].to_numpy()
        ),
    }
    for index, (name, difference) in enumerate(metrics.items()):
        low, high = bootstrap_interval(
            difference, samples=samples, seed=seed + index
        )
        row[f"{name}_difference"] = float(difference.mean())
        row[f"{name}_ci95_low"] = low
        row[f"{name}_ci95_high"] = high
    return row


def main() -> None:
    args = parse_args()
    development_rows = []
    for path in sorted(args.development_root.glob("q*/external_eval/evaluated_generations.csv")):
        frame = pd.read_csv(path)
        config = json.loads((path.parents[1] / "config.json").read_text())
        quantile = float(config["target_quantile"])
        row = summarize(frame, f"quantile_q{quantile:g}")
        row["quantile"] = quantile
        development_rows.append(row)
    development = pd.DataFrame(development_rows).sort_values("quantile")

    candidate = pd.read_csv(args.candidate)
    reference = pd.read_csv(args.reference)
    shift3 = pd.read_csv(args.shift3)
    methods = {
        "baseline": reference[reference["method"] == "baseline"].copy(),
        "original_pplm": reference[reference["method"] == "pplm"].copy(),
        "relative_shift3": shift3[shift3["method"] == "minimum_norm"].copy(),
        "quantile_q0.5": candidate.copy(),
    }
    for name, frame in methods.items():
        if frame.duplicated(KEY).any():
            raise RuntimeError(f"duplicate validation keys for {name}")
    validation = pd.DataFrame(
        [summarize(frame, name) for name, frame in methods.items()]
    )
    paired = pd.DataFrame(
        [
            paired_row(
                candidate,
                frame,
                name,
                samples=args.bootstrap_samples,
                seed=args.seed + 10 * index,
            )
            for index, (name, frame) in enumerate(methods.items())
            if name != "quantile_q0.5"
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(args.output_dir / "development_quantiles.csv", index=False)
    validation.to_csv(args.output_dir / "validation_summary.csv", index=False)
    paired.to_csv(args.output_dir / "validation_paired_differences.csv", index=False)
    print("Development")
    print(development.to_string(index=False))
    print("\nValidation")
    print(validation.to_string(index=False))
    print("\nPaired differences")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
