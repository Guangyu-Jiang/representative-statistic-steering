#!/usr/bin/env python3
"""Select a Lookback perturbation setting using development candidates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_BASELINE = Path(
    "artifacts/lookback_nq/development_n60_matched_rerank_diagnostics/"
    "candidates4/results.jsonl"
)
DEFAULT_CURRENT = DEFAULT_BASELINE
DEFAULT_REFINEMENT_ROOT = Path(
    "artifacts/lookback_nq/development_n60_rerank_refinement"
)
DEFAULT_OUTPUT = Path("artifacts/reports/lookback_rerank_refinement_selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--refinement-root", type=Path, default=DEFAULT_REFINEMENT_ROOT
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_method(path: Path, method: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") == method:
            rows[int(row["dataset_index"])] = row
    return [rows[index] for index in sorted(rows)]


def interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = values[
        generator.integers(0, values.size, size=(samples, values.size))
    ].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def configuration(path: Path) -> dict[str, Any]:
    config_path = path.parent / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def summarize(
    path: Path,
    candidate_rows: list[dict[str, Any]],
    baseline_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    paired = [
        (row, baseline_by_index[int(row["dataset_index"])])
        for row in candidate_rows
        if int(row["dataset_index"]) in baseline_by_index
    ]
    selected = np.asarray([row[0]["exact_match"] for row in paired], dtype=float)
    baseline_selected = np.asarray(
        [row[1]["exact_match"] for row in paired], dtype=float
    )
    candidate_means = np.asarray(
        [np.mean(row[0]["candidate_exact_matches"]) for row in paired],
        dtype=float,
    )
    baseline_candidate_means = np.asarray(
        [np.mean(row[1]["candidate_exact_matches"]) for row in paired],
        dtype=float,
    )
    candidate_labels = np.asarray(
        [label for row, _ in paired for label in row["candidate_exact_matches"]],
        dtype=float,
    )
    candidate_scores = np.asarray(
        [
            score
            for row, _ in paired
            for score in row["candidate_replay_factual_probabilities"]
        ],
        dtype=float,
    )
    selected_delta = selected - baseline_selected
    candidate_delta = candidate_means - baseline_candidate_means
    selected_lower, selected_upper = interval(selected_delta)
    candidate_lower, candidate_upper = interval(candidate_delta)
    config = configuration(path)
    return {
        "run": str(path.parent),
        "n": len(paired),
        "complete": len(paired) == 60,
        "target_logit_shift": config.get("target_logit_shift"),
        "maximum_bias_rms_config": config.get("maximum_bias_rms"),
        "selected_exact_match": float(selected.mean()),
        "baseline_selected_exact_match": float(baseline_selected.mean()),
        "selected_minus_baseline": float(selected_delta.mean()),
        "selected_delta_ci95_lower": selected_lower,
        "selected_delta_ci95_upper": selected_upper,
        "candidate_mean_exact_match": float(candidate_means.mean()),
        "baseline_candidate_mean_exact_match": float(
            baseline_candidate_means.mean()
        ),
        "candidate_mean_minus_baseline": float(candidate_delta.mean()),
        "candidate_mean_delta_ci95_lower": candidate_lower,
        "candidate_mean_delta_ci95_upper": candidate_upper,
        "candidate_oracle_exact_match": float(
            np.mean([max(row[0]["candidate_exact_matches"]) for row in paired])
        ),
        "candidate_replay_auc": (
            float(roc_auc_score(candidate_labels, candidate_scores))
            if np.unique(candidate_labels).size > 1
            else float("nan")
        ),
        "mean_bias_rms": float(np.mean([row[0]["mean_bias_rms"] for row in paired])),
        "mean_output_kl": float(np.mean([row[0]["mean_output_kl"] for row in paired])),
    }


def main() -> None:
    args = parse_args()
    baseline_rows = read_method(args.baseline, "baseline_rerank")
    baseline_by_index = {
        int(row["dataset_index"]): row for row in baseline_rows
    }
    paths = [args.current]
    paths.extend(sorted(args.refinement_root.glob("*/results.jsonl")))
    records = []
    for path in paths:
        rows = read_method(path, "minimum_norm_rerank")
        if rows:
            records.append(summarize(path, rows, baseline_by_index))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit("no completed perturbation candidates found")
    if args.require_complete and (
        len(baseline_rows) != 60 or not bool(frame.complete.all())
    ):
        raise SystemExit(
            f"incomplete development data: baseline={len(baseline_rows)}, "
            f"candidate_counts={frame[['run', 'n']].to_dict('records')}"
        )

    # Gold labels are used only on development. Candidate-mean EM is primary
    # because it measures generation rather than selector luck.
    ranked = frame.sort_values(
        [
            "candidate_mean_exact_match",
            "selected_exact_match",
            "mean_bias_rms",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "development_rank", np.arange(1, len(ranked) + 1))
    selected = ranked.iloc[0].to_dict()
    payload = {
        "selection_protocol": (
            "lexicographic: candidate_mean_exact_match descending, "
            "selected_exact_match descending, mean_bias_rms ascending"
        ),
        "baseline_questions": len(baseline_rows),
        "selected": selected,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2)
    )
    print(ranked.to_string(index=False))
    print("\nSelected development setting")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
