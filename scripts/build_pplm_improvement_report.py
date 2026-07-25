#!/usr/bin/env python3
"""Build matched PPLM and adaptive minimum-norm steering comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts/pplm_sentiment")
OUTPUT_DIR = Path("artifacts/reports")
KEY = ["target_label", "prefix", "seed"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def calibrated_candidate(positive_path: str, negative_path: str) -> pd.DataFrame:
    positive = load(positive_path)
    negative = load(negative_path)
    positive = positive.loc[positive["target_label"] == "positive"]
    negative = negative.loc[negative["target_label"] == "negative"]
    return pd.concat([positive, negative], ignore_index=True)


def trimmed_mean(values: pd.Series, fraction: float = 0.1) -> float:
    ordered = np.sort(values.to_numpy(dtype=float))
    trim = int(len(ordered) * fraction)
    if trim == 0:
        return float(ordered.mean())
    return float(ordered[trim:-trim].mean())


def summarize(split: str, method: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    groups = [(target, group) for target, group in frame.groupby("target_label")]
    groups.append(("macro", frame))
    for target, group in groups:
        rows.append(
            {
                "split": split,
                "method": method,
                "target_label": target,
                "n": len(group),
                "target_probability": group["external_target_probability"].mean(),
                "success": (group["external_target_probability"] >= 0.5).mean(),
                "mean_perplexity": group["perplexity"].mean(),
                "median_perplexity": group["perplexity"].median(),
                "trimmed_10_perplexity": trimmed_mean(group["perplexity"]),
                "p90_perplexity": group["perplexity"].quantile(0.9),
                "max_perplexity": group["perplexity"].max(),
                "relative_cache_change": group["mean_relative_cache_change"].mean(),
                "mean_token_kl": group["mean_token_kl"].mean(),
                "mean_raw_token_kl": group["mean_raw_token_kl"].mean(),
                "unique_continuations": group["continuation"].nunique(),
            }
        )
    return rows


def paired_deltas(
    split: str, reference: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, object]]:
    columns = KEY + ["external_target_probability", "perplexity"]
    merged = reference[columns].merge(
        candidate[columns], on=KEY, suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    merged["success_reference"] = (
        merged["external_target_probability_reference"] >= 0.5
    ).astype(float)
    merged["success_candidate"] = (
        merged["external_target_probability_candidate"] >= 0.5
    ).astype(float)
    rng = np.random.default_rng(1729)
    sample_indices = rng.integers(0, len(merged), size=(20_000, len(merged)))
    rows = []
    for metric in ("external_target_probability", "success", "perplexity"):
        differences = (
            merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
        ).to_numpy()
        bootstrap_means = differences[sample_indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "split": split,
                "metric": metric,
                "n": len(merged),
                "candidate_minus_reference": differences.mean(),
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
            }
        )
    return rows


def main() -> None:
    heldout_reference = load(
        "persistent_reference_heldout/merged/external_eval/evaluated_generations.csv"
    )
    heldout_global = load(
        "persistent_candidate_v2_heldout/merged/external_eval/evaluated_generations.csv"
    )
    heldout_calibrated = calibrated_candidate(
        "persistent_candidate_v2_heldout/merged/external_eval/evaluated_generations.csv",
        "persistent_negative_gm045_heldout/merged/external_eval/evaluated_generations.csv",
    )
    independent_reference = load(
        "persistent_independent_validation/reference_merged/external_eval/evaluated_generations.csv"
    )
    independent_calibrated = calibrated_candidate(
        "persistent_independent_validation/candidate_positive_merged/external_eval/evaluated_generations.csv",
        "persistent_independent_validation/candidate_negative_merged/external_eval/evaluated_generations.csv",
    )
    adaptive_development_reference = load(
        "persistent_independent_validation_v2/reference_merged/external_eval/evaluated_generations.csv"
    )
    adaptive_development_candidate = load(
        "persistent_adaptive_policy_validation_v2/merged/external_eval/evaluated_generations.csv"
    )
    adaptive_independent_reference = load(
        "persistent_adaptive_policy_validation_v3/reference_merged/external_eval/evaluated_generations.csv"
    )
    adaptive_independent_candidate = load(
        "persistent_adaptive_policy_validation_v3/candidate_merged/external_eval/evaluated_generations.csv"
    )

    report_rows = []
    for split, method, frame in (
        ("heldout", "persistent_pplm_10step", heldout_reference),
        ("heldout", "minimum_norm_global_mix", heldout_global),
        ("heldout", "minimum_norm_target_calibrated", heldout_calibrated),
        ("independent", "persistent_pplm_10step", independent_reference),
        ("independent", "minimum_norm_target_calibrated", independent_calibrated),
        (
            "adaptive_development_v2",
            "persistent_pplm_10step",
            adaptive_development_reference,
        ),
        (
            "adaptive_development_v2",
            "minimum_norm_adaptive_kl",
            adaptive_development_candidate,
        ),
        (
            "adaptive_independent_v3",
            "persistent_pplm_10step",
            adaptive_independent_reference,
        ),
        (
            "adaptive_independent_v3",
            "minimum_norm_adaptive_kl",
            adaptive_independent_candidate,
        ),
    ):
        report_rows.extend(summarize(split, method, frame))

    paired_rows = []
    paired_rows.extend(
        paired_deltas("heldout", heldout_reference, heldout_calibrated)
    )
    paired_rows.extend(
        paired_deltas(
            "independent", independent_reference, independent_calibrated
        )
    )
    paired_rows.extend(
        paired_deltas(
            "adaptive_development_v2",
            adaptive_development_reference,
            adaptive_development_candidate,
        )
    )
    paired_rows.extend(
        paired_deltas(
            "adaptive_independent_v3",
            adaptive_independent_reference,
            adaptive_independent_candidate,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(report_rows)
    paired = pd.DataFrame(paired_rows)
    report.to_csv(OUTPUT_DIR / "pplm_perturbation_comparison.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "pplm_perturbation_paired_deltas.csv", index=False)
    print(report.loc[report["target_label"] == "macro"].to_string(index=False))
    print("\nPaired deltas\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
