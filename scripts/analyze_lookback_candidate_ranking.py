#!/usr/bin/env python3
"""Audit label-free ranking rules over saved Lookback generation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPLAY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def standardize(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def bootstrap_interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    means = values[
        generator.integers(0, values.size, size=(samples, values.size))
    ].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    candidates = [
        row
        for row in rows
        if row.get("candidate_exact_matches")
        and row.get("candidate_replay_factual_probabilities")
        and row.get("candidate_controlled_factual_probabilities")
    ]
    if not candidates:
        raise ValueError(f"no candidate-ranking records in {args.results}")

    flat_labels = np.asarray(
        [label for row in candidates for label in row["candidate_exact_matches"]],
        dtype=np.float64,
    )
    flat_replay = np.asarray(
        [
            score
            for row in candidates
            for score in row["candidate_replay_factual_probabilities"]
        ],
        dtype=np.float64,
    )
    flat_controlled = np.asarray(
        [
            score
            for row in candidates
            for score in row["candidate_controlled_factual_probabilities"]
        ],
        dtype=np.float64,
    )
    replay_auc = (
        float(roc_auc_score(flat_labels, flat_replay))
        if np.unique(flat_labels).size > 1
        else float("nan")
    )
    controlled_auc = (
        float(roc_auc_score(flat_labels, flat_controlled))
        if np.unique(flat_labels).size > 1
        else float("nan")
    )

    records: list[dict[str, float | int]] = []
    for replay_weight in REPLAY_WEIGHTS:
        selected: list[float] = []
        random_expectation: list[float] = []
        oracle: list[float] = []
        for row in candidates:
            labels = np.asarray(row["candidate_exact_matches"], dtype=np.float64)
            replay = standardize(
                np.asarray(
                    row["candidate_replay_factual_probabilities"], dtype=np.float64
                )
            )
            controlled = standardize(
                np.asarray(
                    row["candidate_controlled_factual_probabilities"],
                    dtype=np.float64,
                )
            )
            score = replay_weight * replay + (1.0 - replay_weight) * controlled
            selected.append(float(labels[int(np.argmax(score))]))
            random_expectation.append(float(labels.mean()))
            oracle.append(float(labels.max()))

        selected_array = np.asarray(selected)
        random_array = np.asarray(random_expectation)
        delta = selected_array - random_array
        lower, upper = bootstrap_interval(delta)
        records.append(
            {
                "n_questions": len(candidates),
                "candidates_per_question": int(
                    np.mean(
                        [len(row["candidate_exact_matches"]) for row in candidates]
                    )
                ),
                "replay_weight": replay_weight,
                "selected_exact_match": float(selected_array.mean()),
                "random_candidate_exact_match": float(random_array.mean()),
                "oracle_exact_match": float(np.mean(oracle)),
                "selected_minus_random": float(delta.mean()),
                "selected_minus_random_ci95_lower": lower,
                "selected_minus_random_ci95_upper": upper,
                "candidate_replay_auc": replay_auc,
                "candidate_controlled_auc": controlled_auc,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
