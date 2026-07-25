#!/usr/bin/env python3
"""Compare corrected accumulated-action PPLM development and validation runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts/pplm_sentiment")
KEY = ["target_label", "prefix", "seed"]


def summarize(run: str, method: str, frame: pd.DataFrame, config: dict) -> dict[str, object]:
    expected_n = (
        len(config.get("prefixes", []))
        * len(config.get("seeds", []))
        * int(frame["target_label"].nunique())
    )
    if expected_n == 0:
        expected_n = len(frame)
    return {
        "run": run,
        "method": method,
        "solver_version": "accumulated_v2",
        "minimum_norm_steps": config.get("minimum_norm_steps"),
        "minimum_norm_damping": config.get("minimum_norm_damping"),
        "ridge": config.get("ridge"),
        "gm_scale": config.get("gm_scale"),
        "target_margin_shift": config.get("target_margin_shift"),
        "cache_component": config.get("cache_component"),
        "cache_last_n_layers": config.get("cache_last_n_layers"),
        "gradient_block_normalization": config.get(
            "gradient_block_normalization"
        ),
        "preserve_top_log_probs": config.get("preserve_top_log_probs"),
        "log_probability_preservation_weight": config.get(
            "log_probability_preservation_weight"
        ),
        "n": len(frame),
        "expected_n": expected_n,
        "complete": len(frame) >= expected_n,
        "target_probability": frame["external_target_probability"].mean(),
        "success": (frame["external_target_probability"] >= 0.5).mean(),
        "mean_perplexity": frame["perplexity"].mean(),
        "median_perplexity": frame["perplexity"].median(),
        "relative_cache_change": frame["mean_relative_cache_change"].mean(),
        "mean_token_kl": frame["mean_token_kl"].mean(),
        "unique_continuations": frame["continuation"].nunique(),
    }


def bootstrap_interval(
    values: pd.Series, *, seed: int = 42, samples: int = 10_000
) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = array[generator.integers(0, array.size, size=(samples, array.size))].mean(1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def paired_delta_summary(
    candidate: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, object]:
    paired = candidate.merge(
        reference,
        on=KEY,
        suffixes=("_minimum_norm", "_pplm"),
        validate="one_to_one",
    )
    target_delta = (
        paired.external_target_probability_minimum_norm
        - paired.external_target_probability_pplm
    )
    perplexity_delta = paired.perplexity_minimum_norm - paired.perplexity_pplm
    target_lower, target_upper = bootstrap_interval(target_delta)
    ppl_lower, ppl_upper = bootstrap_interval(perplexity_delta)
    return {
        "paired_n": len(paired),
        "minimum_norm_minus_pplm_target_probability": target_delta.mean(),
        "target_probability_delta_ci95_lower": target_lower,
        "target_probability_delta_ci95_upper": target_upper,
        "minimum_norm_minus_pplm_perplexity": perplexity_delta.mean(),
        "perplexity_delta_ci95_lower": ppl_lower,
        "perplexity_delta_ci95_upper": ppl_upper,
    }


def main() -> None:
    reference = pd.read_csv(
        ROOT
        / "persistent_adaptive_policy_validation_v3/reference_merged/external_eval/evaluated_generations.csv"
    )
    corrected_reference_frames = [reference]
    for reference_path in ROOT.glob(
        "corrected_accumulated*/**/external_eval/evaluated_generations.csv"
    ):
        reference_frame = pd.read_csv(reference_path)
        if "method" in reference_frame and (reference_frame.method == "pplm").any():
            corrected_reference_frames.append(
                reference_frame[reference_frame.method == "pplm"]
            )
    reference = pd.concat(corrected_reference_frames, ignore_index=True).drop_duplicates(
        KEY, keep="last"
    )
    legacy = pd.read_csv(
        ROOT
        / "persistent_adaptive_policy_validation_v3/candidate_merged/external_eval/evaluated_generations.csv"
    )
    rows = []
    target_rows = []
    for path in sorted(
        ROOT.glob("corrected_accumulated*/**/external_eval/evaluated_generations.csv")
    ):
        candidate = pd.read_csv(path)
        name = str(path.parents[1].relative_to(ROOT))
        config_path = path.parents[1] / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        methods = set(candidate["method"])
        for method, group in candidate.groupby("method", sort=True):
            rows.append(summarize(name, method, group, config))
            for target_label, target_group in group.groupby("target_label", sort=True):
                target_row = summarize(name, method, target_group, config)
                target_row["target_label"] = target_label
                target_rows.append(target_row)

        if methods == {"minimum_norm"}:
            matched_reference = reference.merge(
                candidate[KEY], on=KEY, how="inner", validate="one_to_one"
            )
            matched_legacy = legacy.merge(
                candidate[KEY], on=KEY, how="inner", validate="one_to_one"
            )
            if len(matched_reference) == len(candidate):
                rows.append(
                    summarize(
                        name, "matched_pplm_reference", matched_reference, config
                    )
                )
                if len(matched_legacy) == len(candidate):
                    rows.append(
                        summarize(
                            name,
                            "matched_legacy_minimum_norm",
                            matched_legacy,
                            config,
                        )
                    )
                minimum_norm_row = next(
                    row
                    for row in reversed(rows)
                    if row["run"] == name and row["method"] == "minimum_norm"
                )
                minimum_norm_row.update(
                    paired_delta_summary(candidate, matched_reference)
                )

        if {"pplm", "minimum_norm"}.issubset(methods):
            pplm = candidate[candidate.method == "pplm"]
            minimum_norm = candidate[candidate.method == "minimum_norm"]
            minimum_norm_row = next(
                row
                for row in reversed(rows)
                if row["run"] == name and row["method"] == "minimum_norm"
            )
            minimum_norm_row.update(paired_delta_summary(minimum_norm, pplm))

    output = Path("artifacts/reports/pplm_corrected_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    pd.DataFrame(target_rows).to_csv(
        Path("artifacts/reports/pplm_corrected_comparison_by_target.csv"),
        index=False,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
