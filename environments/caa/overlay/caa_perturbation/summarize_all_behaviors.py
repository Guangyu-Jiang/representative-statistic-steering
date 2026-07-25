from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from caa_perturbation.run_experiment import model_slug


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "caa_perturbation_all_behaviors"
METHOD_NAMES = {
    "none": "unsteered",
    "caa": "fixed_caa",
    "scalar_target": "scalar_target",
    "pca_target": "pca_target",
    "scalar_hinge": "scalar_hinge",
    "clean_scalar_hinge": "clean_scalar_hinge",
    "fisher_hinge": "fisher_hinge",
    "clean_statistic_shift": "clean_statistic_shift",
    "fisher_statistic_shift": "fisher_statistic_shift",
    "pca_statistic_shift": "pca_statistic_shift",
    "pca_margin_hinge": "pca_margin_hinge",
}
METHOD_ORDER = tuple(METHOD_NAMES)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row["index"]))
    probabilities = np.asarray([row["matching_probability"] for row in rows], dtype=float)
    accuracy = np.asarray(
        [row["predicted_matching_behavior"] for row in rows], dtype=float
    )
    return probabilities, accuracy


def paired_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def stratified_tune_eval_indices(
    baseline_rows: list[dict],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for row in baseline_rows:
        answer = row["answer_matching_behavior"].strip()
        groups.setdefault(answer, []).append(int(row["index"]))
    tune = []
    evaluation = []
    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        split = len(shuffled) // 2
        tune.extend(shuffled[:split])
        evaluation.extend(shuffled[split:])
    return np.asarray(sorted(tune)), np.asarray(sorted(evaluation))


def candidate_rows(summaries: list[dict], method: str) -> list[dict]:
    rows = [row for row in summaries if row["method"] == method]
    if method == "caa":
        # Positive CAA strength points toward the matching-behavior examples.
        rows = [row for row in rows if float(row["strength"]) > 0]
    return rows


def metrics(values: tuple[np.ndarray, np.ndarray], indices: np.ndarray) -> tuple[float, float]:
    probabilities, accuracy = values
    return float(probabilities[indices].mean()), float(accuracy[indices].mean())


def summarize_behavior(
    behavior_root: Path,
    seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    behavior = behavior_root.name
    summaries = read_csv(behavior_root / "mc_summary.csv")
    raw_predictions = read_json(behavior_root / "mc_predictions.json")
    predictions: dict[str, list[dict]] = {}
    for row in raw_predictions:
        predictions.setdefault(row["setting_id"], []).append(row)
    arrays = {key: prediction_arrays(rows) for key, rows in predictions.items()}
    summary_by_id = {row["setting_id"]: row for row in summaries}
    baseline_id = next(row["setting_id"] for row in summaries if row["method"] == "none")
    baseline_rows = predictions[baseline_id]
    baseline = arrays[baseline_id]
    tune_indices, eval_indices = stratified_tune_eval_indices(baseline_rows, seed)
    rng = np.random.default_rng(seed)

    full_grid = []
    for row in summaries:
        full_grid.append(
            {
                **row,
                "method_name": METHOD_NAMES[row["method"]],
            }
        )

    oracle_rows = []
    split_rows = []
    selected: dict[str, dict] = {}
    available_methods = [
        method
        for method in METHOD_ORDER
        if any(row["method"] == method for row in summaries)
    ]
    for method in available_methods:
        candidates = candidate_rows(summaries, method)
        if method == "none":
            candidates = [summary_by_id[baseline_id]]

        oracle = max(candidates, key=lambda row: float(row["mean_matching_probability"]))
        oracle_values = arrays[oracle["setting_id"]]
        probability_difference = oracle_values[0] - baseline[0]
        accuracy_difference = oracle_values[1] - baseline[1]
        probability_ci = paired_interval(probability_difference, rng, bootstrap_samples)
        accuracy_ci = paired_interval(accuracy_difference, rng, bootstrap_samples)
        oracle_rows.append(
            {
                "behavior": behavior,
                "method": METHOD_NAMES[method],
                "selection": "oracle_full_heldout",
                "setting_id": oracle["setting_id"],
                "layer": oracle["layer"],
                "n": len(oracle_values[0]),
                "matching_probability": float(oracle_values[0].mean()),
                "matching_accuracy": float(oracle_values[1].mean()),
                "relative_action_norm": float(oracle["mean_action_relative_norm"]),
                "probability_gain": float(probability_difference.mean()),
                "probability_gain_ci_low": probability_ci[0],
                "probability_gain_ci_high": probability_ci[1],
                "accuracy_gain": float(accuracy_difference.mean()),
                "accuracy_gain_ci_low": accuracy_ci[0],
                "accuracy_gain_ci_high": accuracy_ci[1],
            }
        )

        selected_row = max(
            candidates,
            key=lambda row: (
                metrics(arrays[row["setting_id"]], tune_indices)[0],
                -float(row["mean_action_relative_norm"]),
            ),
        )
        selected[method] = selected_row
        selected_values = arrays[selected_row["setting_id"]]
        probability_difference = (
            selected_values[0][eval_indices] - baseline[0][eval_indices]
        )
        accuracy_difference = selected_values[1][eval_indices] - baseline[1][eval_indices]
        probability_ci = paired_interval(probability_difference, rng, bootstrap_samples)
        accuracy_ci = paired_interval(accuracy_difference, rng, bootstrap_samples)
        tune_probability, tune_accuracy = metrics(selected_values, tune_indices)
        eval_probability, eval_accuracy = metrics(selected_values, eval_indices)
        split_rows.append(
            {
                "behavior": behavior,
                "method": METHOD_NAMES[method],
                "selection": "stratified_half_tune_half_eval",
                "setting_id": selected_row["setting_id"],
                "layer": selected_row["layer"],
                "n_tune": len(tune_indices),
                "n_eval": len(eval_indices),
                "tune_matching_probability": tune_probability,
                "tune_matching_accuracy": tune_accuracy,
                "eval_matching_probability": eval_probability,
                "eval_matching_accuracy": eval_accuracy,
                "relative_action_norm": float(selected_row["mean_action_relative_norm"]),
                "eval_probability_gain": float(probability_difference.mean()),
                "eval_probability_gain_ci_low": probability_ci[0],
                "eval_probability_gain_ci_high": probability_ci[1],
                "eval_accuracy_gain": float(accuracy_difference.mean()),
                "eval_accuracy_gain_ci_low": accuracy_ci[0],
                "eval_accuracy_gain_ci_high": accuracy_ci[1],
            }
        )

    norm_rows = []
    caa_candidates = candidate_rows(summaries, "caa")
    for method in available_methods:
        if method in {"none", "caa"}:
            continue
        target_row = selected[method]
        target_norm = float(target_row["mean_action_relative_norm"])
        caa_row = min(
            caa_candidates,
            key=lambda row: abs(float(row["mean_action_relative_norm"]) - target_norm),
        )
        target_values = arrays[target_row["setting_id"]]
        caa_values = arrays[caa_row["setting_id"]]
        probability_difference = (
            target_values[0][eval_indices] - caa_values[0][eval_indices]
        )
        accuracy_difference = target_values[1][eval_indices] - caa_values[1][eval_indices]
        probability_ci = paired_interval(probability_difference, rng, bootstrap_samples)
        accuracy_ci = paired_interval(accuracy_difference, rng, bootstrap_samples)
        norm_rows.append(
            {
                "behavior": behavior,
                "target_method": METHOD_NAMES[method],
                "target_setting_id": target_row["setting_id"],
                "target_relative_action_norm": target_norm,
                "matched_caa_setting_id": caa_row["setting_id"],
                "matched_caa_relative_action_norm": float(
                    caa_row["mean_action_relative_norm"]
                ),
                "n_eval": len(eval_indices),
                "target_minus_caa_probability": float(probability_difference.mean()),
                "probability_ci_low": probability_ci[0],
                "probability_ci_high": probability_ci[1],
                "target_minus_caa_accuracy": float(accuracy_difference.mean()),
                "accuracy_ci_low": accuracy_ci[0],
                "accuracy_ci_high": accuracy_ci[1],
            }
        )
    return full_grid, oracle_rows, split_rows, norm_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.artifact_root / model_slug(args.model)
    completed = sorted(
        path.parent for path in model_root.glob("*/mc_summary.csv")
        if (path.parent / "mc_predictions.json").exists()
    )
    if not completed:
        raise FileNotFoundError(f"No completed evaluations under {model_root}")

    full_grid = []
    oracle = []
    split_selected = []
    norm_matched = []
    for behavior_root in completed:
        outputs = summarize_behavior(
            behavior_root,
            args.seed,
            args.bootstrap_samples,
        )
        for destination, rows in zip(
            (full_grid, oracle, split_selected, norm_matched), outputs
        ):
            destination.extend(rows)

    report_root = args.artifact_root / "reports"
    write_csv(report_root / "all_behaviors_full_grid.csv", full_grid)
    write_csv(report_root / "all_behaviors_oracle_best.csv", oracle)
    write_csv(report_root / "all_behaviors_split_selected.csv", split_selected)
    write_csv(report_root / "all_behaviors_norm_matched.csv", norm_matched)

    fixed_by_behavior = {
        row["behavior"]: row for row in split_selected if row["method"] == "fixed_caa"
    }
    macro_rows = []
    selected_method_names = [
        METHOD_NAMES[method]
        for method in METHOD_ORDER
        if any(row["method"] == METHOD_NAMES[method] for row in split_selected)
    ]
    for method in selected_method_names:
        rows = [row for row in split_selected if row["method"] == method]
        differences = [
            float(row["eval_matching_probability"])
            - float(fixed_by_behavior[row["behavior"]]["eval_matching_probability"])
            for row in rows
        ]
        macro_rows.append(
            {
                "method": method,
                "behaviors": len(rows),
                "macro_eval_matching_probability": sum(
                    float(row["eval_matching_probability"]) for row in rows
                )
                / len(rows),
                "macro_eval_matching_accuracy": sum(
                    float(row["eval_matching_accuracy"]) for row in rows
                )
                / len(rows),
                "macro_relative_action_norm": sum(
                    float(row["relative_action_norm"]) for row in rows
                )
                / len(rows),
                "macro_probability_gain_over_unsteered": sum(
                    float(row["eval_probability_gain"]) for row in rows
                )
                / len(rows),
                "macro_probability_delta_vs_fixed_caa": sum(differences) / len(rows),
                "behaviors_beating_fixed_caa": sum(value > 0 for value in differences),
            }
        )
    write_csv(report_root / "all_behaviors_macro.csv", macro_rows)

    print(f"Completed behaviors: {len(completed)}")
    for row in split_selected:
        print(row)


if __name__ == "__main__":
    main()
