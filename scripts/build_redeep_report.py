#!/usr/bin/env python3
"""Build paired ReDeEP detector and Dolly steering reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_SCORE = {
    "SUPPORTED": 1.0,
    "PARTIALLY_SUPPORTED": 0.5,
    "UNSUPPORTED": 0.0,
    "NONANSWER": 0.0,
    "PARSE_ERROR": np.nan,
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--pairwise", type=Path)
    parser.add_argument(
        "--detector-report",
        type=Path,
        default=Path("artifacts/redeep/detector_reproduction/detector_report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/redeep/report")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_mean_interval(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def diagnostics_value(row: pd.Series, name: str) -> float:
    value = row["diagnostics"].get(name)
    return float(value) if value is not None else float("nan")


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.judged.open(encoding="utf-8")]
    frame = pd.DataFrame(rows)
    frame["judge_score"] = frame["local_judge_label"].map(LABEL_SCORE)
    for name in (
        "mean_control_rms",
        "mean_final_hidden_relative_change",
        "mean_baseline_detector_score",
        "mean_controlled_detector_score",
        "mean_achieved_target_fraction",
    ):
        frame[name] = frame.apply(lambda row: diagnostics_value(row, name), axis=1)
    frame["detector_reduction"] = (
        frame["mean_baseline_detector_score"]
        - frame["mean_controlled_detector_score"]
    )

    summary_rows: list[dict[str, object]] = []
    for method, group in frame.groupby("method"):
        labels = group["local_judge_label"].value_counts()
        summary_rows.append(
            {
                "method": method,
                "examples": len(group),
                "supported_pct": 100 * labels.get("SUPPORTED", 0) / len(group),
                "partially_supported_pct": 100
                * labels.get("PARTIALLY_SUPPORTED", 0)
                / len(group),
                "unsupported_pct": 100
                * labels.get("UNSUPPORTED", 0)
                / len(group),
                "nonanswer_pct": 100 * labels.get("NONANSWER", 0) / len(group),
                "mean_judge_score": group["judge_score"].mean(),
                "mean_reference_token_f1": group["reference_token_f1"].mean(),
                "mean_reference_token_recall": group[
                    "reference_token_recall"
                ].mean(),
                "mean_passage_grounding_ratio": group[
                    "passage_grounding_ratio"
                ].mean(),
                "mean_control_rms": group["mean_control_rms"].mean(),
                "mean_final_hidden_relative_change": group[
                    "mean_final_hidden_relative_change"
                ].mean(),
                "mean_detector_reduction": group["detector_reduction"].mean(),
                "mean_achieved_target_fraction": group[
                    "mean_achieved_target_fraction"
                ].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("method")

    baseline = frame[frame["method"] == "baseline"].set_index("source_id")
    paired_rows: list[dict[str, object]] = []
    for method in sorted(set(frame["method"]) - {"baseline"}):
        candidate = frame[frame["method"] == method].set_index("source_id")
        common = baseline.index.intersection(candidate.index)
        base = baseline.loc[common]
        controlled = candidate.loc[common]
        judge_difference = (
            controlled["judge_score"].to_numpy()
            - base["judge_score"].to_numpy()
        )
        supported_difference = (
            (controlled["local_judge_label"] == "SUPPORTED").astype(float).to_numpy()
            - (base["local_judge_label"] == "SUPPORTED").astype(float).to_numpy()
        )
        low, high = bootstrap_mean_interval(
            judge_difference, samples=args.bootstrap_samples, seed=args.seed
        )
        supported_low, supported_high = bootstrap_mean_interval(
            supported_difference,
            samples=args.bootstrap_samples,
            seed=args.seed + 1,
        )
        paired_rows.append(
            {
                "method": method,
                "paired_examples": len(common),
                "judge_win_pct": 100 * np.mean(judge_difference > 0),
                "judge_tie_pct": 100 * np.mean(judge_difference == 0),
                "judge_loss_pct": 100 * np.mean(judge_difference < 0),
                "mean_judge_score_difference": float(np.nanmean(judge_difference)),
                "judge_score_difference_ci_low": low,
                "judge_score_difference_ci_high": high,
                "supported_rate_difference": float(supported_difference.mean()),
                "supported_rate_difference_ci_low": supported_low,
                "supported_rate_difference_ci_high": supported_high,
                "response_change_pct": 100
                * np.mean(
                    controlled["response"].to_numpy()
                    != base["response"].to_numpy()
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "steering_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    detector = json.loads(args.detector_report.read_text())
    pairwise_judge = None
    if args.pairwise is not None:
        pairwise_rows = [
            json.loads(line) for line in args.pairwise.open(encoding="utf-8")
        ]
        pairwise_judge = []
        for method in sorted({str(row["method"]) for row in pairwise_rows}):
            selected = [row for row in pairwise_rows if row["method"] == method]
            winners = pd.Series([row["winner"] for row in selected]).value_counts()
            pairwise_judge.append(
                {
                    "method": method,
                    "pairs": len(selected),
                    "controlled_win_pct": 100
                    * winners.get("controlled", 0)
                    / len(selected),
                    "baseline_win_pct": 100
                    * winners.get("baseline", 0)
                    / len(selected),
                    "tie_pct": 100 * winners.get("tie", 0) / len(selected),
                    "parse_error_pct": 100
                    * winners.get("parse_error", 0)
                    / len(selected),
                }
            )
        pd.DataFrame(pairwise_judge).to_csv(
            args.output_dir / "two_order_pairwise_judge.csv", index=False
        )
    report = {
        "detector": detector,
        "steering_summary": summary.to_dict(orient="records"),
        "paired_comparison": paired.to_dict(orient="records"),
        "two_order_pairwise_judge": pairwise_judge,
        "judge": str(frame["local_judge_model"].iloc[0]),
        "external_api_calls": 0,
    }
    (args.output_dir / "redeep_report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nSteering summary")
    print(summary.to_string(index=False))
    print("\nPaired comparison")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
