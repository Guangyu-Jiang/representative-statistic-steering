#!/usr/bin/env python3
"""Aggregate Lookback Lens experiment summaries into comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repstat_steering.lookback_control import summarize_lookback_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/lookback_nq")
    parser.add_argument(
        "--output", default="artifacts/reports/lookback_nq_comparison.csv"
    )
    return parser.parse_args()


def bootstrap_mean_interval(
    values: list[float], *, seed: int = 42, samples: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = array[generator.integers(0, array.size, size=(samples, array.size))].mean(
        axis=1
    )
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def load_run_config(path: Path) -> dict:
    config_path = path.parent / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def select_paired_baseline(
    run_path: Path,
    run_rows: list[dict],
    run_config: dict,
    candidates: list[tuple[Path, list[dict], dict]],
    *,
    preferred_do_sample: bool,
) -> tuple[Path | None, dict[int, dict]]:
    """Select a baseline without leaking across dataset splits or decode lengths."""
    same_run = [
        item
        for item in candidates
        if item[0].parent == run_path.parent
        and bool(item[2].get("do_sample", False)) == preferred_do_sample
    ]
    if same_run:
        path, rows, _ = max(same_run, key=lambda item: len(item[1]))
        return path, {row["dataset_index"]: row for row in rows}

    run_indices = {row["dataset_index"] for row in run_rows}
    run_max_tokens = run_config.get("max_new_tokens")
    compatible = [
        item
        for item in candidates
        if (
            run_max_tokens is None
            or item[2].get("max_new_tokens") is None
            or item[2].get("max_new_tokens") == run_max_tokens
        )
        and bool(item[2].get("do_sample", False)) == preferred_do_sample
    ]
    if not compatible:
        compatible = candidates

    def score(item: tuple[Path, list[dict], dict]) -> tuple[int, int, float, int]:
        _, rows, _ = item
        baseline_indices = {row["dataset_index"] for row in rows}
        overlap = len(run_indices & baseline_indices)
        union = len(run_indices | baseline_indices)
        return (
            int(run_indices == baseline_indices),
            overlap,
            overlap / union if union else 0.0,
            -abs(len(run_indices) - len(baseline_indices)),
        )

    if not compatible:
        return None, {}
    path, rows, _ = max(compatible, key=score)
    if not run_indices & {row["dataset_index"] for row in rows}:
        return None, {}
    return path, {row["dataset_index"]: row for row in rows}


def main() -> None:
    args = parse_args()
    records = []
    runs: list[tuple[Path, list[dict], dict]] = []
    for path in sorted(Path(args.root).rglob("results.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:
            continue
        runs.append((path, rows, load_run_config(path)))

    baseline_candidates = [
        (path, [row for row in rows if row["method"] == "baseline"], config)
        for path, rows, config in runs
    ]
    baseline_candidates = [item for item in baseline_candidates if item[1]]

    for path, rows, config in runs:
        for summary in summarize_lookback_rows(rows):
            first = rows[0]
            method_rows = [row for row in rows if row["method"] == summary["method"]]
            exact_matches = np.asarray(
                [row["exact_match"] for row in method_rows], dtype=np.float64
            )
            factual_scores = np.asarray(
                [row["mean_factual_probability"] for row in method_rows],
                dtype=np.float64,
            )
            score_auc = (
                float(roc_auc_score(exact_matches, factual_scores))
                if np.unique(exact_matches).size > 1
                else float("nan")
            )
            score_correlation = (
                float(np.corrcoef(exact_matches, factual_scores)[0, 1])
                if exact_matches.std() > 0 and factual_scores.std() > 0
                else float("nan")
            )
            candidate_rows = [
                row
                for row in method_rows
                if row.get("candidate_exact_matches")
                and row.get("candidate_replay_factual_probabilities")
            ]
            candidate_metrics = {}
            if candidate_rows:
                candidate_labels = np.asarray(
                    [
                        label
                        for row in candidate_rows
                        for label in row["candidate_exact_matches"]
                    ],
                    dtype=np.float64,
                )
                candidate_scores = np.asarray(
                    [
                        score
                        for row in candidate_rows
                        for score in row["candidate_replay_factual_probabilities"]
                    ],
                    dtype=np.float64,
                )
                candidate_score_auc = (
                    float(roc_auc_score(candidate_labels, candidate_scores))
                    if np.unique(candidate_labels).size > 1
                    else float("nan")
                )
                candidate_metrics = {
                    "mean_candidate_count": float(
                        np.mean(
                            [len(row["candidate_exact_matches"]) for row in candidate_rows]
                        )
                    ),
                    "candidate_mean_exact_match": float(candidate_labels.mean()),
                    "candidate_oracle_exact_match": float(
                        np.mean(
                            [max(row["candidate_exact_matches"]) for row in candidate_rows]
                        )
                    ),
                    "candidate_replay_score_auc": candidate_score_auc,
                    "candidate_replay_score_range": float(
                        np.mean(
                            [
                                max(row["candidate_replay_factual_probabilities"])
                                - min(row["candidate_replay_factual_probabilities"])
                                for row in candidate_rows
                            ]
                        )
                    ),
                }
            preferred_do_sample = (
                True
                if summary["method"]
                in {"guided", "baseline_rerank", "minimum_norm_rerank"}
                else bool(config.get("do_sample", False))
            )
            baseline_path, baseline_by_index = select_paired_baseline(
                path,
                method_rows,
                config,
                baseline_candidates,
                preferred_do_sample=preferred_do_sample,
            )
            paired = [
                (row, baseline_by_index[row["dataset_index"]])
                for row in method_rows
                if row["dataset_index"] in baseline_by_index
            ]
            pair_metrics = {}
            if paired:
                exact_match_deltas = [
                    row["exact_match"] - baseline["exact_match"]
                    for row, baseline in paired
                ]
                delta_lower, delta_upper = bootstrap_mean_interval(
                    exact_match_deltas
                )
                score_deltas = np.asarray(
                    [
                        row["mean_factual_probability"]
                        - baseline["mean_factual_probability"]
                        for row, baseline in paired
                    ],
                    dtype=np.float64,
                )
                outcome_deltas = np.asarray(exact_match_deltas, dtype=np.float64)
                score_delta_outcome_delta_correlation = (
                    float(np.corrcoef(score_deltas, outcome_deltas)[0, 1])
                    if score_deltas.std() > 0 and outcome_deltas.std() > 0
                    else float("nan")
                )
                pair_metrics = {
                    "paired_n": len(paired),
                    "paired_baseline_exact_match": sum(
                        baseline["exact_match"] for _, baseline in paired
                    )
                    / len(paired),
                    "paired_exact_match_delta": sum(exact_match_deltas) / len(paired),
                    "paired_exact_match_delta_ci95_lower": delta_lower,
                    "paired_exact_match_delta_ci95_upper": delta_upper,
                    "paired_response_changed": sum(
                        row["response"] != baseline["response"]
                        for row, baseline in paired
                    )
                    / len(paired),
                    "paired_improved": sum(
                        row["exact_match"] > baseline["exact_match"]
                        for row, baseline in paired
                    )
                    / len(paired),
                    "paired_regressed": sum(
                        row["exact_match"] < baseline["exact_match"]
                        for row, baseline in paired
                    )
                    / len(paired),
                    "paired_factual_probability_delta": sum(
                        row["mean_factual_probability"]
                        - baseline["mean_factual_probability"]
                        for row, baseline in paired
                    )
                    / len(paired),
                    "score_delta_outcome_delta_correlation": (
                        score_delta_outcome_delta_correlation
                    ),
                }
            records.append(
                {
                    "run": str(path.parent.relative_to(args.root)),
                    "paired_baseline_run": (
                        str(baseline_path.parent.relative_to(args.root))
                        if baseline_path is not None
                        else ""
                    ),
                    "solver_version": first.get("solver_version", "legacy_unspecified"),
                    "target_mode": first.get("target_mode", "absolute"),
                    "target_logit_shift": first.get("target_logit_shift", 0.0),
                    "control_trigger_probability": first.get(
                        "control_trigger_probability", 1.0
                    ),
                    "high_confidence_logit_shift": first.get(
                        "high_confidence_logit_shift", 0.0
                    ),
                    "target_probability": first.get("target_probability", 0.0),
                    "ridge": first.get("ridge", 0.0),
                    "maximum_bias_rms_config": first.get("maximum_bias_rms"),
                    "context_bias_mode": first.get("context_bias_mode", "uniform"),
                    "context_top_fraction": first.get("context_top_fraction"),
                    "context_overlap_radius": first.get("context_overlap_radius"),
                    "active_control_count": first.get("active_control_count", 0),
                    "bias_constraint": first.get(
                        "bias_constraint", "unrestricted"
                    ),
                    "expected_n": int(config.get("n_examples") or len(method_rows)),
                    "complete": len(method_rows)
                    >= int(config.get("n_examples") or len(method_rows)),
                    "factual_score_auc": score_auc,
                    "factual_score_outcome_correlation": score_correlation,
                    **summary,
                    **pair_metrics,
                    **candidate_metrics,
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(["method", "exact_match"], ascending=[True, False])
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
