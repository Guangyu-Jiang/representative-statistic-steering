from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "caa_perturbation"
MODEL_SLUG = "meta-llama__Llama-3.1-8B-Instruct"
OFFICIAL_RESULT_PATTERN = re.compile(
    r"results_layer=(?P<layer>\d+)_multiplier=(?P<multiplier>-?\d+(?:\.\d+)?)_"
    r"behavior=(?P<behavior>[^_]+)_type=ab_use_base_model=False_model_size=7b\.json"
)

SELECTED_SETTINGS = {
    "sycophancy": {
        "fixed_caa": "caa__r0__strength0.5__ridge0",
        "scalar_target": "scalar_target__r1__strength0.5__ridge0.1",
        "pca_target": "pca_target__r1__strength0.5__ridge0.01",
    },
    "refusal": {
        "fixed_caa": "caa__r0__strength1__ridge0",
        "scalar_target": "scalar_target__r1__strength1.5__ridge0.1",
        "pca_target": "pca_target__r1__strength1.5__ridge0.01",
    },
}

OPEN_ENDED_SETTINGS = {
    "sycophancy": {
        "judge_directory": "local_judged",
        "unsteered": "none__r8__strength0__ridge0",
        "fixed_caa": "caa__r8__strength0.5__ridge0",
        "scalar_target": "scalar_target__r8__strength0.5__ridge0.1",
        "pca_target": "pca_target__r1__strength0.5__ridge0.01",
    },
    "refusal": {
        "judge_directory": "local_judged_selected",
        "unsteered": "none__r8__strength0__ridge0",
        "fixed_caa": "caa__r8__strength1__ridge0",
        "scalar_target": "scalar_target__r8__strength1.5__ridge0.1",
        "pca_target": "pca_target__r1__strength1.5__ridge0.01",
    },
}


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matching_probability(row: dict) -> float:
    denominator = row["a_prob"] + row["b_prob"]
    matching = row["answer_matching_behavior"]
    return (row["a_prob"] if "A" in matching else row["b_prob"]) / denominator


def summarize_official_release() -> list[dict]:
    grouped: dict[tuple[str, int], dict[float, float]] = {}
    for path in (REPO_ROOT / "results").glob("*/*.json"):
        match = OFFICIAL_RESULT_PATTERN.fullmatch(path.name)
        if not match:
            continue
        behavior = match.group("behavior")
        layer = int(match.group("layer"))
        multiplier = float(match.group("multiplier"))
        if multiplier not in {-1.0, 0.0, 1.0}:
            continue
        rows = read_json(path)
        value = float(np.mean([matching_probability(row) for row in rows]))
        grouped.setdefault((behavior, layer), {})[multiplier] = value

    summaries = []
    for (behavior, layer), values in grouped.items():
        if set(values) != {-1.0, 0.0, 1.0}:
            continue
        summaries.append(
            {
                "behavior": behavior,
                "layer": layer,
                "minus_1": values[-1.0],
                "baseline": values[0.0],
                "plus_1": values[1.0],
                "symmetric_effect": values[1.0] - values[-1.0],
            }
        )
    return sorted(summaries, key=lambda row: (row["behavior"], row["layer"]))


def paired_bootstrap_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_portability_run(
    artifact_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for behavior, settings in SELECTED_SETTINGS.items():
        behavior_root = artifact_root / MODEL_SLUG / behavior
        predictions = read_json(behavior_root / "mc_predictions.json")
        summaries = {}
        with (behavior_root / "mc_summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                summaries[row["setting_id"]] = row
        by_setting: dict[str, list[float]] = {}
        for row in predictions:
            by_setting.setdefault(row["setting_id"], []).append(
                float(row["matching_probability"])
            )
        baseline_id = "none__r0__strength0__ridge0"
        baseline = np.asarray(by_setting[baseline_id])
        baseline_summary = summaries[baseline_id]
        output.append(
            {
                "behavior": behavior,
                "method": "unsteered",
                "setting_id": baseline_id,
                "n": len(baseline),
                "matching_probability": float(baseline.mean()),
                "matching_accuracy": float(baseline_summary["matching_accuracy"]),
                "relative_action_norm": 0.0,
                "gain_over_unsteered": 0.0,
                "gain_ci_low": 0.0,
                "gain_ci_high": 0.0,
            }
        )
        for method, setting_id in settings.items():
            values = np.asarray(by_setting[setting_id])
            differences = values - baseline
            low, high = paired_bootstrap_interval(differences, rng, bootstrap_samples)
            summary = summaries[setting_id]
            output.append(
                {
                    "behavior": behavior,
                    "method": method,
                    "setting_id": setting_id,
                    "n": len(values),
                    "matching_probability": float(values.mean()),
                    "matching_accuracy": float(summary["matching_accuracy"]),
                    "relative_action_norm": float(summary["mean_action_relative_norm"]),
                    "gain_over_unsteered": float(differences.mean()),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                }
            )
    return output


def summarize_local_open_ended(
    artifact_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    judge_suffix = "__Qwen__Qwen2.5-7B-Instruct.json"
    for behavior, settings in OPEN_ENDED_SETTINGS.items():
        judge_root = (
            artifact_root
            / MODEL_SLUG
            / behavior
            / "open_ended"
            / settings["judge_directory"]
        )
        scores = {}
        for method in ("unsteered", "fixed_caa", "scalar_target", "pca_target"):
            rows = read_json(judge_root / f"{settings[method]}{judge_suffix}")
            parsed = [row["local_judge_score"] for row in rows]
            if any(score is None for score in parsed):
                raise ValueError(f"Unparsed local score in {behavior}/{method}")
            scores[method] = np.asarray(parsed, dtype=float)
        baseline = scores["unsteered"]
        for method, values in scores.items():
            differences = values - baseline
            if method == "unsteered":
                low = high = 0.0
            else:
                low, high = paired_bootstrap_interval(
                    differences, rng, bootstrap_samples
                )
            output.append(
                {
                    "behavior": behavior,
                    "method": method,
                    "setting_id": settings[method],
                    "n": len(values),
                    "valid": int(np.isfinite(values).sum()),
                    "mean_local_score": float(values.mean()),
                    "gain_over_unsteered": float(differences.mean()),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                }
            )
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = args.artifact_root / "reports"
    official_rows = summarize_official_release()
    write_csv(report_root / "official_released_layer_scan.csv", official_rows)
    selected_official = []
    for behavior in sorted({row["behavior"] for row in official_rows}):
        candidates = [row for row in official_rows if row["behavior"] == behavior]
        selected_official.append(max(candidates, key=lambda row: row["symmetric_effect"]))
    write_csv(report_root / "official_released_best_layers.csv", selected_official)

    portability_rows = summarize_portability_run(
        args.artifact_root,
        args.bootstrap_samples,
        args.seed,
    )
    write_csv(report_root / "portability_selected_comparison.csv", portability_rows)
    for row in portability_rows:
        print(row)

    open_ended_rows = summarize_local_open_ended(
        args.artifact_root,
        args.bootstrap_samples,
        args.seed,
    )
    write_csv(report_root / "local_open_ended_selected_comparison.csv", open_ended_rows)
    for row in open_ended_rows:
        print(row)


if __name__ == "__main__":
    main()
