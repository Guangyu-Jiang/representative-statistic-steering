#!/usr/bin/env python3
"""Report the untouched confirmation for the development-selected Lookback edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "artifacts/reports/lookback_rerank_refinement_selection.json"
        ),
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=Path("artifacts/lookback_nq/refined_confirmation_offset260_n100"),
    )
    parser.add_argument(
        "--ranker-summary",
        type=Path,
        default=Path(
            "artifacts/lookback_nq/candidate_ranker_refined_confirmation/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "artifacts/reports/lookback_refinement_confirmation"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def setting_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def read_rows(path: Path, method: str) -> pd.DataFrame:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") == method:
            rows[int(row["dataset_index"])] = row
    return pd.DataFrame([rows[index] for index in sorted(rows)])


def interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = values[
        generator.integers(0, values.size, size=(samples, values.size))
    ].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def candidate_means(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [np.mean(values) for values in frame.candidate_exact_matches], dtype=float
    )


def effect(
    comparison: str, values: np.ndarray
) -> dict[str, Any]:
    lower, upper = interval(values)
    return {
        "comparison": comparison,
        "n": int(values.size),
        "mean_difference": float(values.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    selected = selection["selected"]
    shift = float(selected["target_logit_shift"])
    cap = float(selected["maximum_bias_rms_config"])
    run = args.confirmation_root / (
        f"shift{setting_tag(shift)}_cap{setting_tag(cap)}"
    )
    result_path = run / "results.jsonl"
    baseline = read_rows(result_path, "baseline")
    unsteered = read_rows(result_path, "baseline_rerank")
    controlled = read_rows(result_path, "minimum_norm_rerank")
    counts = {
        "baseline": len(baseline),
        "baseline_rerank": len(unsteered),
        "minimum_norm_rerank": len(controlled),
    }
    complete = all(count == 100 for count in counts.values())
    if args.require_complete and not complete:
        raise SystemExit(f"incomplete confirmation: {counts}")
    paired = controlled[
        ["dataset_index", "exact_match", "candidate_exact_matches"]
    ].merge(
        unsteered[
            ["dataset_index", "exact_match", "candidate_exact_matches"]
        ],
        on="dataset_index",
        suffixes=("_controlled", "_unsteered"),
        validate="one_to_one",
    ).merge(
        baseline[["dataset_index", "exact_match"]],
        on="dataset_index",
        validate="one_to_one",
    )
    controlled_candidate_mean = np.asarray(
        [
            np.mean(values)
            for values in paired.candidate_exact_matches_controlled
        ],
        dtype=float,
    )
    unsteered_candidate_mean = np.asarray(
        [
            np.mean(values)
            for values in paired.candidate_exact_matches_unsteered
        ],
        dtype=float,
    )
    summaries = [
        {
            "method": "sampled_unsteered",
            "n": len(baseline),
            "exact_match": float(baseline.exact_match.mean()),
            "candidate_mean_exact_match": float("nan"),
            "candidate_oracle_exact_match": float("nan"),
            "mean_bias_rms": 0.0,
        },
        {
            "method": "unsteered_replay_rerank",
            "n": len(unsteered),
            "exact_match": float(unsteered.exact_match.mean()),
            "candidate_mean_exact_match": float(
                candidate_means(unsteered).mean()
            ),
            "candidate_oracle_exact_match": float(
                np.mean([max(values) for values in unsteered.candidate_exact_matches])
            ),
            "mean_bias_rms": 0.0,
        },
        {
            "method": "minimum_norm_replay_rerank",
            "n": len(controlled),
            "exact_match": float(controlled.exact_match.mean()),
            "candidate_mean_exact_match": float(
                candidate_means(controlled).mean()
            ),
            "candidate_oracle_exact_match": float(
                np.mean([max(values) for values in controlled.candidate_exact_matches])
            ),
            "mean_bias_rms": float(controlled.mean_bias_rms.mean()),
        },
    ]
    effects = [
        effect(
            "minimum_norm_replay_minus_unsteered_replay",
            (
                paired.exact_match_controlled
                - paired.exact_match_unsteered
            ).to_numpy(),
        ),
        effect(
            "mean_controlled_candidate_minus_mean_unsteered_candidate",
            controlled_candidate_mean - unsteered_candidate_mean,
        ),
        effect(
            "minimum_norm_replay_minus_sampled_unsteered",
            (
                paired.exact_match_controlled - paired.exact_match
            ).to_numpy(),
        ),
    ]
    ranker = json.loads(args.ranker_summary.read_text())
    validation = ranker["validation"]
    summaries.extend(
        [
            {
                "method": "minimum_norm_learned_ranker",
                "n": int(validation["ranker"]["n_questions"]),
                "exact_match": float(validation["ranker"]["selected_exact_match"]),
                "candidate_mean_exact_match": float(
                    validation["ranker"]["random_candidate_exact_match"]
                ),
                "candidate_oracle_exact_match": float(
                    validation["ranker"]["oracle_exact_match"]
                ),
                "mean_bias_rms": float(controlled.mean_bias_rms.mean()),
            },
            {
                "method": "unsteered_learned_ranker",
                "n": int(validation["unsteered_rerank"]["n_questions"]),
                "exact_match": float(validation["ranked_unsteered_exact_match"]),
                "candidate_mean_exact_match": float(
                    validation["unsteered_rerank"]["random_candidate_exact_match"]
                ),
                "candidate_oracle_exact_match": float(
                    validation["unsteered_rerank"]["oracle_exact_match"]
                ),
                "mean_bias_rms": 0.0,
            },
        ]
    )
    effects.append(
        {
            "comparison": "minimum_norm_learned_minus_unsteered_learned",
            "n": int(validation["ranker"]["n_questions"]),
            "mean_difference": float(
                validation[
                    "ranked_controlled_minus_ranked_unsteered_exact_match"
                ]
            ),
            "ci95_lower": float(
                validation[
                    "ranked_controlled_minus_ranked_unsteered_ci95"
                ][0]
            ),
            "ci95_upper": float(
                validation[
                    "ranked_controlled_minus_ranked_unsteered_ci95"
                ][1]
            ),
        }
    )
    payload = {
        "complete": complete,
        "selection_protocol": selection["selection_protocol"],
        "selected_development_setting": selected,
        "confirmation_run": str(run),
        "counts": counts,
        "ranker_development": ranker["development"],
        "summaries": summaries,
        "paired_effects": effects,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        index=False,
    )
    pd.DataFrame(effects).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_effects.csv"),
        index=False,
    )
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2)
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nPaired effects")
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()
