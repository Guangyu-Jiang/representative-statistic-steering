#!/usr/bin/env python3
"""Build matched corrected-TruthX comparisons from raw question-level records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts/truthx_mc")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows).drop_duplicates("dataset_index", keep="first")


def bootstrap_interval(
    values: pd.Series, *, seed: int = 42, samples: int = 10_000
) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = array[generator.integers(0, array.size, size=(samples, array.size))].mean(1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    baseline = read_jsonl(ROOT / "preproj_full_baseline/results.jsonl")
    published = read_jsonl(ROOT / "preproj_full_original_s4p5/results.jsonl")
    records: list[dict] = []
    result_paths = {
        *ROOT.glob("corrected_accumulated_*/*/results.jsonl"),
        *ROOT.glob("corrected_linesearch_*/*/results.jsonl"),
        *ROOT.glob("corrected_direction_*/*/results.jsonl"),
        *ROOT.glob("corrected_published_gate_*/*/results.jsonl"),
    }
    for path in sorted(result_paths):
        candidate = read_jsonl(path)
        if candidate.empty:
            continue
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        matched = candidate.merge(
            baseline[["dataset_index", "mc1", "mc2", "mc3"]],
            on="dataset_index",
            suffixes=("", "_baseline"),
            validate="one_to_one",
        ).merge(
            published[
                [
                    "dataset_index",
                    "mc1",
                    "mc2",
                    "mc3",
                    "mean_relative_action_norm",
                    "intervention_rate",
                ]
            ],
            on="dataset_index",
            suffixes=("", "_published"),
            validate="one_to_one",
        )
        mc1_delta = matched.mc1 - matched.mc1_baseline
        mc2_delta = matched.mc2 - matched.mc2_baseline
        mc1_published_delta = matched.mc1 - matched.mc1_published
        mc2_published_delta = matched.mc2 - matched.mc2_published
        mc1_lower, mc1_upper = bootstrap_interval(mc1_delta)
        mc2_lower, mc2_upper = bootstrap_interval(mc2_delta)
        mc1_published_lower, mc1_published_upper = bootstrap_interval(
            mc1_published_delta
        )
        mc2_published_lower, mc2_published_upper = bootstrap_interval(
            mc2_published_delta
        )
        intervention = config.get("intervention", {})
        expected_n = int(config.get("limit") or len(matched))
        candidate_relative_norm = matched.mean_relative_action_norm.mean()
        published_relative_norm = matched.mean_relative_action_norm_published.mean()
        candidate_all_position_relative_norm = (
            matched.mean_relative_action_norm * matched.intervention_rate
        ).mean()
        published_all_position_relative_norm = (
            matched.mean_relative_action_norm_published
            * matched.intervention_rate_published
        ).mean()
        candidate_mc2_efficiency = (
            mc2_delta.mean() / candidate_relative_norm
            if candidate_relative_norm > 0
            else float("nan")
        )
        published_mc2_efficiency = (
            (matched.mc2_published - matched.mc2_baseline).mean()
            / published_relative_norm
            if published_relative_norm > 0
            else float("nan")
        )
        candidate_mc2_all_position_efficiency = (
            mc2_delta.mean() / candidate_all_position_relative_norm
            if candidate_all_position_relative_norm > 0
            else float("nan")
        )
        published_mc2_all_position_efficiency = (
            (matched.mc2_published - matched.mc2_baseline).mean()
            / published_all_position_relative_norm
            if published_all_position_relative_norm > 0
            else float("nan")
        )
        records.append(
            {
                "run": str(path.parent.relative_to(ROOT)),
                "solver_version": intervention.get("solver_version", "accumulated_v2"),
                "n": len(matched),
                "expected_n": expected_n,
                "complete": len(matched) >= expected_n,
                "target_mode": intervention.get("target_mode"),
                "target_strength": intervention.get("target_strength"),
                "ridge": intervention.get("ridge"),
                "damping": intervention.get("learning_rate"),
                "maximum_relative_norm": intervention.get("maximum_relative_norm"),
                "intervention_margin_threshold": intervention.get(
                    "intervention_margin_threshold"
                ),
                "directional_backtracking_steps": intervention.get(
                    "directional_backtracking_steps", 0
                ),
                "directional_nonnegative": intervention.get(
                    "directional_nonnegative", False
                ),
                "mc1": matched.mc1.mean(),
                "mc2": matched.mc2.mean(),
                "mc3": matched.mc3.mean(),
                "baseline_mc1": matched.mc1_baseline.mean(),
                "baseline_mc2": matched.mc2_baseline.mean(),
                "published_mc1": matched.mc1_published.mean(),
                "published_mc2": matched.mc2_published.mean(),
                "mc1_minus_baseline": mc1_delta.mean(),
                "mc1_delta_ci95_lower": mc1_lower,
                "mc1_delta_ci95_upper": mc1_upper,
                "mc2_minus_baseline": mc2_delta.mean(),
                "mc2_delta_ci95_lower": mc2_lower,
                "mc2_delta_ci95_upper": mc2_upper,
                "mc1_minus_published": mc1_published_delta.mean(),
                "mc1_minus_published_ci95_lower": mc1_published_lower,
                "mc1_minus_published_ci95_upper": mc1_published_upper,
                "mc2_minus_published": mc2_published_delta.mean(),
                "mc2_minus_published_ci95_lower": mc2_published_lower,
                "mc2_minus_published_ci95_upper": mc2_published_upper,
                "mean_relative_action_norm": candidate_relative_norm,
                "published_mean_relative_action_norm": published_relative_norm,
                "all_position_relative_action_norm": (
                    candidate_all_position_relative_norm
                ),
                "published_all_position_relative_action_norm": (
                    published_all_position_relative_norm
                ),
                "mc2_gain_per_relative_action": candidate_mc2_efficiency,
                "published_mc2_gain_per_relative_action": published_mc2_efficiency,
                "mc2_gain_per_all_position_relative_action": (
                    candidate_mc2_all_position_efficiency
                ),
                "published_mc2_gain_per_all_position_relative_action": (
                    published_mc2_all_position_efficiency
                ),
                "intervention_rate": matched.intervention_rate.mean(),
                "published_intervention_rate": (
                    matched.intervention_rate_published.mean()
                ),
                "initial_target_rmse": matched.initial_target_rmse.mean(),
                "final_target_rmse": matched.final_target_rmse.mean(),
                "valid_scores": matched.valid_scores.mean(),
            }
        )
    output = Path("artifacts/reports/truthx_corrected_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
