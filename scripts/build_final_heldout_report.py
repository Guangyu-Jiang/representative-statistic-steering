#!/usr/bin/env python3
"""Build compact final held-out tables from the frozen experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ARTIFACTS = Path("artifacts")
REPORTS = ARTIFACTS / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    return pd.DataFrame(rows)


def interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = values[
        generator.integers(0, values.size, size=(samples, values.size))
    ].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def effect_row(
    study: str,
    comparison: str,
    metric: str,
    values: np.ndarray,
) -> dict[str, Any]:
    lower, upper = interval(values)
    return {
        "study": study,
        "comparison": comparison,
        "metric": metric,
        "n": values.size,
        "mean_difference": float(values.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def truthx_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    candidate = read_jsonl(
        ARTIFACTS
        / "truthx_mc/corrected_accumulated_confirmation"
        / "decoder_t0p25_r0_d1_cap6p0_offset704_n113/results.jsonl"
    ).drop_duplicates("dataset_index")
    baseline = read_jsonl(
        ARTIFACTS / "truthx_mc/preproj_full_baseline/results.jsonl"
    ).drop_duplicates("dataset_index")
    published = read_jsonl(
        ARTIFACTS / "truthx_mc/preproj_full_original_s4p5/results.jsonl"
    ).drop_duplicates("dataset_index")
    matched = candidate.merge(
        baseline[
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
    summaries = []
    for method, suffix in (
        ("minimum_norm", ""),
        ("unsteered", "_baseline"),
        ("published_truthx", "_published"),
    ):
        summaries.append(
            {
                "study": "truthx",
                "method": method,
                "n": len(matched),
                "primary_metric": "mc2",
                "primary_value": float(matched[f"mc2{suffix}"].mean()),
                "secondary_metric": "mc1",
                "secondary_value": float(matched[f"mc1{suffix}"].mean()),
                "quality_metric": "mc3",
                "quality_value": float(matched[f"mc3{suffix}"].mean()),
                "relative_change": float(
                    matched[f"mean_relative_action_norm{suffix}"].mean()
                ),
                "relative_change_definition": (
                    "mean_relative_action_norm_over_changed_positions"
                ),
                "intervention_rate": float(
                    matched[f"intervention_rate{suffix}"].mean()
                ),
                "all_position_relative_change": float(
                    (
                        matched[f"mean_relative_action_norm{suffix}"]
                        * matched[f"intervention_rate{suffix}"]
                    ).mean()
                ),
            }
        )
    effects = []
    for reference, suffix in (("unsteered", "_baseline"), ("published_truthx", "_published")):
        for metric in ("mc1", "mc2"):
            effects.append(
                effect_row(
                    "truthx",
                    f"minimum_norm-minus-{reference}",
                    metric,
                    (matched[metric] - matched[f"{metric}{suffix}"]).to_numpy(),
                )
            )
    return summaries, effects, len(matched) == 113


def pplm_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    frame = pd.read_csv(REPORTS / "pplm_corrected_comparison.csv")
    run = "corrected_accumulated_output_preservation_validation_top2_w0p05_seeds22_33"
    selected = frame[frame.run.eq(run)]
    summaries = []
    for method in ("minimum_norm", "matched_pplm_reference"):
        row = selected[selected.method.eq(method)].iloc[0]
        summaries.append(
            {
                "study": "pplm",
                "method": method,
                "n": int(row.n),
                "primary_metric": "target_probability",
                "primary_value": float(row.target_probability),
                "secondary_metric": "success",
                "secondary_value": float(row.success),
                "quality_metric": "perplexity",
                "quality_value": float(row.mean_perplexity),
                "relative_change": float(row.relative_cache_change),
                "relative_change_definition": "mean_relative_cache_change",
            }
        )
    candidate = selected[selected.method.eq("minimum_norm")].iloc[0]
    effects = [
        {
            "study": "pplm",
            "comparison": "minimum_norm-minus-pplm",
            "metric": "target_probability",
            "n": int(candidate.paired_n),
            "mean_difference": float(
                candidate.minimum_norm_minus_pplm_target_probability
            ),
            "ci95_lower": float(candidate.target_probability_delta_ci95_lower),
            "ci95_upper": float(candidate.target_probability_delta_ci95_upper),
        },
        {
            "study": "pplm",
            "comparison": "minimum_norm-minus-pplm",
            "metric": "perplexity",
            "n": int(candidate.paired_n),
            "mean_difference": float(candidate.minimum_norm_minus_pplm_perplexity),
            "ci95_lower": float(candidate.perplexity_delta_ci95_lower),
            "ci95_upper": float(candidate.perplexity_delta_ci95_upper),
        },
    ]
    return summaries, effects, bool(selected.complete.all() and len(selected) == 2)


def lookback_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rows = read_jsonl(
        ARTIFACTS
        / "lookback_nq/validation_offset160_n100_minimum_norm_rerank_replay"
        / "candidates4_sparse128_shift4_cap0.5/results.jsonl"
    )
    baseline = rows[rows.method.eq("baseline")].drop_duplicates("dataset_index")
    candidate = rows[rows.method.eq("minimum_norm_rerank")].drop_duplicates(
        "dataset_index"
    )
    baseline_rerank_path = (
        ARTIFACTS
        / "lookback_nq/validation_offset160_n100_baseline_rerank_replay"
        / "candidates4/results.jsonl"
    )
    if baseline_rerank_path.exists():
        baseline_rerank = read_jsonl(baseline_rerank_path).drop_duplicates(
            "dataset_index"
        )
    else:
        baseline_rerank = pd.DataFrame()
    matched = candidate.merge(
        baseline[["dataset_index", "exact_match"]],
        on="dataset_index",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    labels = np.asarray(
        [label for values in candidate.candidate_exact_matches for label in values],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            score
            for values in candidate.candidate_replay_factual_probabilities
            for score in values
        ],
        dtype=np.float64,
    )
    candidate_mean = float(labels.mean()) if labels.size else float("nan")
    oracle = (
        float(np.mean([max(values) for values in candidate.candidate_exact_matches]))
        if len(candidate)
        else float("nan")
    )
    auc = (
        float(roc_auc_score(labels, scores))
        if labels.size and np.unique(labels).size > 1
        else float("nan")
    )
    summaries = [
        {
            "study": "lookback",
            "method": "minimum_norm_rerank",
            "n": len(candidate),
            "primary_metric": "exact_match",
            "primary_value": float(candidate.exact_match.mean()),
            "secondary_metric": "candidate_mean_exact_match",
            "secondary_value": candidate_mean,
            "quality_metric": "candidate_replay_auc",
            "quality_value": auc,
            "relative_change": float(candidate.mean_bias_rms.mean()),
            "relative_change_definition": "mean_attention_logit_bias_rms",
        },
        {
            "study": "lookback",
            "method": "sampled_unsteered",
            "n": len(baseline),
            "primary_metric": "exact_match",
            "primary_value": float(baseline.exact_match.mean()),
            "secondary_metric": "candidate_oracle_exact_match",
            "secondary_value": oracle,
            "quality_metric": "output_kl",
            "quality_value": 0.0,
            "relative_change": 0.0,
            "relative_change_definition": "mean_attention_logit_bias_rms",
        },
    ]
    if not baseline_rerank.empty:
        baseline_rerank_labels = np.asarray(
            [
                label
                for values in baseline_rerank.candidate_exact_matches
                for label in values
            ],
            dtype=np.float64,
        )
        baseline_rerank_scores = np.asarray(
            [
                score
                for values in baseline_rerank.candidate_replay_factual_probabilities
                for score in values
            ],
            dtype=np.float64,
        )
        summaries.append(
            {
                "study": "lookback",
                "method": "unsteered_rerank",
                "n": len(baseline_rerank),
                "primary_metric": "exact_match",
                "primary_value": float(baseline_rerank.exact_match.mean()),
                "secondary_metric": "candidate_mean_exact_match",
                "secondary_value": float(baseline_rerank_labels.mean()),
                "quality_metric": "candidate_replay_auc",
                "quality_value": (
                    float(
                        roc_auc_score(
                            baseline_rerank_labels, baseline_rerank_scores
                        )
                    )
                    if np.unique(baseline_rerank_labels).size > 1
                    else float("nan")
                ),
                "relative_change": 0.0,
                "relative_change_definition": "mean_attention_logit_bias_rms",
            }
        )
    effects = [
        effect_row(
            "lookback",
            "minimum_norm_rerank-minus-sampled_unsteered",
            "exact_match",
            (matched.exact_match - matched.exact_match_baseline).to_numpy(),
        )
    ]
    candidate_expectation = pd.DataFrame(
        {
            "dataset_index": candidate.dataset_index.to_numpy(),
            "candidate_mean_exact_match": [
                float(np.mean(values)) for values in candidate.candidate_exact_matches
            ],
        }
    )
    decomposition = candidate[
        ["dataset_index", "exact_match"]
    ].merge(
        candidate_expectation,
        on="dataset_index",
        validate="one_to_one",
    ).merge(
        baseline[["dataset_index", "exact_match"]],
        on="dataset_index",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    effects.extend(
        [
            effect_row(
                "lookback",
                "mean_controlled_candidate-minus-sampled_unsteered",
                "exact_match",
                (
                    decomposition.candidate_mean_exact_match
                    - decomposition.exact_match_baseline
                ).to_numpy(),
            ),
            effect_row(
                "lookback",
                "replay_selection-minus-mean_controlled_candidate",
                "exact_match",
                (
                    decomposition.exact_match_selected
                    - decomposition.candidate_mean_exact_match
                ).to_numpy(),
            ),
        ]
    )
    if not baseline_rerank.empty:
        rerank_comparison = candidate[
            ["dataset_index", "exact_match", "candidate_exact_matches"]
        ].merge(
            baseline_rerank[
                ["dataset_index", "exact_match", "candidate_exact_matches"]
            ],
            on="dataset_index",
            suffixes=("_controlled", "_unsteered"),
            validate="one_to_one",
        )
        effects.extend(
            [
                effect_row(
                    "lookback",
                    "controlled_rerank-minus-unsteered_rerank",
                    "exact_match",
                    (
                        rerank_comparison.exact_match_controlled
                        - rerank_comparison.exact_match_unsteered
                    ).to_numpy(),
                ),
                effect_row(
                    "lookback",
                    "mean_controlled_candidate-minus-mean_unsteered_candidate",
                    "exact_match",
                    np.asarray(
                        [
                            np.mean(controlled) - np.mean(unsteered)
                            for controlled, unsteered in zip(
                                rerank_comparison.candidate_exact_matches_controlled,
                                rerank_comparison.candidate_exact_matches_unsteered,
                                strict=True,
                            )
                        ]
                    ),
                ),
            ]
        )
    complete = (
        len(candidate) == 100
        and len(baseline) == 100
        and len(baseline_rerank) == 100
    )
    return summaries, effects, complete


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    completion: dict[str, bool] = {}
    for study, builder in (
        ("lookback", lookback_tables),
        ("pplm", pplm_tables),
        ("truthx", truthx_tables),
    ):
        study_summaries, study_effects, complete = builder()
        summaries.extend(study_summaries)
        effects.extend(study_effects)
        completion[study] = complete

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    effect_frame = pd.DataFrame(effects)
    summary_frame.to_csv(REPORTS / "final_heldout_summary.csv", index=False)
    effect_frame.to_csv(REPORTS / "final_heldout_paired_effects.csv", index=False)
    (REPORTS / "final_heldout_completion.json").write_text(
        json.dumps(completion, indent=2)
    )
    print(summary_frame.to_string(index=False))
    print("\nPaired effects")
    print(effect_frame.to_string(index=False))
    print(f"\nCompletion: {json.dumps(completion, sort_keys=True)}")
    if args.require_complete and not all(completion.values()):
        raise SystemExit("one or more final comparisons are incomplete")


if __name__ == "__main__":
    main()
