from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from caa_perturbation.run_experiment import model_slug


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(values: np.ndarray, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/caa_perturbation_all_behaviors"),
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_root = args.artifact_root / "reports"
    selected = read_csv(report_root / "all_behaviors_split_selected.csv")
    if args.methods:
        selected = [row for row in selected if row["method"] in set(args.methods)]
    model_root = args.artifact_root / model_slug(args.model)
    judge_suffix = f"__{args.judge_model.replace('/', '__')}.json"
    rows = []
    for behavior in sorted({row["behavior"] for row in selected}):
        behavior_selected = [row for row in selected if row["behavior"] == behavior]
        judge_root = model_root / behavior / "open_ended_split_selected" / "local_judged"
        scores = {}
        for row in behavior_selected:
            setting_id = row["setting_id"]
            path = judge_root / f"{setting_id}{judge_suffix}"
            judged = sorted(read_json(path), key=lambda value: int(value["index"]))
            scores[row["method"]] = np.asarray(
                [
                    np.nan if value["local_judge_score"] is None else value["local_judge_score"]
                    for value in judged
                ],
                dtype=float,
            )

        baseline = scores["unsteered"]
        fixed_caa = scores["fixed_caa"]
        for row in behavior_selected:
            method = row["method"]
            values = scores[method]
            valid = np.isfinite(values) & np.isfinite(baseline)
            differences = values[valid] - baseline[valid]
            if method == "unsteered":
                low = high = 0.0
            else:
                low, high = paired_interval(
                    differences, args.bootstrap_samples, args.seed
                )
            fixed_valid = np.isfinite(values) & np.isfinite(fixed_caa)
            fixed_differences = values[fixed_valid] - fixed_caa[fixed_valid]
            if method == "fixed_caa":
                fixed_low = fixed_high = 0.0
            else:
                fixed_low, fixed_high = paired_interval(
                    fixed_differences, args.bootstrap_samples, args.seed
                )
            rows.append(
                {
                    "behavior": behavior,
                    "method": method,
                    "setting_id": row["setting_id"],
                    "n": len(values),
                    "valid": int(np.isfinite(values).sum()),
                    "mean_local_score": float(np.nanmean(values)),
                    "gain_over_unsteered": float(differences.mean()),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                    "delta_vs_fixed_caa": float(fixed_differences.mean()),
                    "delta_vs_fixed_caa_ci_low": fixed_low,
                    "delta_vs_fixed_caa_ci_high": fixed_high,
                }
            )

    write_csv(report_root / "all_behaviors_local_open_ended.csv", rows)
    macro_rows = []
    methods = list(dict.fromkeys(row["method"] for row in selected))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        macro_rows.append(
            {
                "method": method,
                "behaviors": len(method_rows),
                "macro_mean_local_score": sum(
                    row["mean_local_score"] for row in method_rows
                )
                / len(method_rows),
                "macro_gain_over_unsteered": sum(
                    row["gain_over_unsteered"] for row in method_rows
                )
                / len(method_rows),
                "macro_delta_vs_fixed_caa": sum(
                    row["delta_vs_fixed_caa"] for row in method_rows
                )
                / len(method_rows),
                "behaviors_beating_fixed_caa": sum(
                    row["delta_vs_fixed_caa"] > 0 for row in method_rows
                ),
            }
        )
    write_csv(report_root / "all_behaviors_local_open_ended_macro.csv", macro_rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
