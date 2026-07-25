#!/usr/bin/env python3
"""Verify that final confirmations do not overlap their tuning examples."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any


ROOT = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/final_split_audit.json"
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_split(
    name: str,
    development_results: Path,
    validation_results: Path,
    *,
    expected_development: int,
    expected_validation: int,
) -> dict[str, Any]:
    development = {
        int(row["dataset_index"]) for row in read_jsonl(development_results)
    }
    validation = {
        int(row["dataset_index"]) for row in read_jsonl(validation_results)
    }
    overlap = sorted(development & validation)
    return {
        "name": name,
        "unit": "dataset_index",
        "development_count": len(development),
        "expected_development_count": expected_development,
        "validation_count": len(validation),
        "expected_validation_count": expected_validation,
        "development_complete": len(development) == expected_development,
        "validation_complete": len(validation) == expected_validation,
        "overlap_count": len(overlap),
        "overlap": overlap,
    }


def pplm_split() -> dict[str, Any]:
    development_path = (
        ROOT
        / "pplm_sentiment/corrected_accumulated_output_preservation_development"
        / "top2_w0p05/config.json"
    )
    validation_path = (
        ROOT
        / "pplm_sentiment"
        / "corrected_accumulated_output_preservation_validation_top2_w0p05_seeds22_33"
        / "config.json"
    )
    development = read_config(development_path)
    validation = read_config(validation_path)

    def units(config: dict[str, Any]) -> set[tuple[str, str, int]]:
        return set(
            product(config["prefixes"], config["targets"], config["seeds"])
        )

    development_units = units(development)
    validation_units = units(validation)
    overlap = sorted(development_units & validation_units)
    return {
        "name": "pplm_output_preservation",
        "unit": "prefix_target_seed",
        "development_count": len(development_units),
        "expected_development_count": 10,
        "validation_count": len(validation_units),
        "expected_validation_count": 60,
        "development_complete": len(development_units) == 10,
        "validation_complete": len(validation_units) == 60,
        "overlap_count": len(overlap),
        "overlap": overlap,
    }


def setting_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def lookback_refinement_split() -> dict[str, Any] | None:
    selection_path = (
        ROOT / "reports/lookback_rerank_refinement_selection.json"
    )
    if not selection_path.exists():
        return None
    selection = read_config(selection_path)
    selected = selection["selected"]
    development_path = Path(selected["run"]) / "results.jsonl"
    shift_tag = setting_tag(float(selected["target_logit_shift"]))
    cap_tag = setting_tag(float(selected["maximum_bias_rms_config"]))
    validation_path = (
        ROOT
        / "lookback_nq/refined_confirmation_offset260_n100"
        / f"shift{shift_tag}_cap{cap_tag}/results.jsonl"
    )
    result = index_split(
        "lookback_refinement_confirmation",
        development_path,
        validation_path,
        expected_development=60,
        expected_validation=100,
    )
    validation_rows = read_jsonl(validation_path)
    validation_method_counts = {
        method: len(
            {
                int(row["dataset_index"])
                for row in validation_rows
                if row.get("method") == method
            }
        )
        for method in ("baseline", "baseline_rerank", "minimum_norm_rerank")
    }
    ranker_summary = (
        ROOT
        / "lookback_nq/candidate_ranker_refined_confirmation/summary.json"
    )
    result.update(
        {
            "selection_complete": bool(selected.get("complete", False)),
            "selected_target_logit_shift": float(
                selected["target_logit_shift"]
            ),
            "selected_maximum_bias_rms": float(
                selected["maximum_bias_rms_config"]
            ),
            "validation_method_counts": validation_method_counts,
            "expected_validation_method_count": 100,
            "validation_methods_complete": all(
                count == 100 for count in validation_method_counts.values()
            ),
            "ranker_summary_exists": ranker_summary.exists(),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    truthx = index_split(
        "truthx_cap6_target0p25",
        ROOT
        / "truthx_mc/corrected_accumulated_strength_extension"
        / "decoder_t0p25_r0_d1_cap6p0/results.jsonl",
        ROOT
        / "truthx_mc/corrected_accumulated_confirmation"
        / "decoder_t0p25_r0_d1_cap6p0_offset704_n113/results.jsonl",
        expected_development=64,
        expected_validation=113,
    )
    lookback = index_split(
        "lookback_minimum_norm_rerank",
        ROOT
        / "lookback_nq/development_n30_minimum_norm_rerank_replay"
        / "candidates4_sparse128_shift4_cap0.5/results.jsonl",
        ROOT
        / "lookback_nq/validation_offset160_n100_minimum_norm_rerank_replay"
        / "candidates4_sparse128_shift4_cap0.5/results.jsonl",
        expected_development=30,
        expected_validation=100,
    )
    lookback_baseline_path = (
        ROOT
        / "lookback_nq/validation_offset160_n100_baseline_rerank_replay"
        / "candidates4/results.jsonl"
    )
    lookback_baseline_count = len(
        {
            int(row["dataset_index"])
            for row in read_jsonl(lookback_baseline_path)
            if row.get("method") == "baseline_rerank"
        }
    )
    lookback["matched_rerank_baseline_count"] = lookback_baseline_count
    lookback["expected_matched_rerank_baseline_count"] = 100
    lookback["matched_rerank_baseline_complete"] = (
        lookback_baseline_count == 100
    )
    lookback_ranker_development_path = (
        ROOT
        / "lookback_nq/development_n60_matched_rerank_diagnostics"
        / "candidates4/results.jsonl"
    )
    lookback_ranker = index_split(
        "lookback_answer_blind_candidate_ranker",
        lookback_ranker_development_path,
        ROOT
        / "lookback_nq/validation_offset160_n100_minimum_norm_rerank_replay"
        / "candidates4_sparse128_shift4_cap0.5/results.jsonl",
        expected_development=60,
        expected_validation=100,
    )
    ranker_development_rows = read_jsonl(lookback_ranker_development_path)
    ranker_method_counts = {
        method: len(
            {
                int(row["dataset_index"])
                for row in ranker_development_rows
                if row.get("method") == method
            }
        )
        for method in ("baseline_rerank", "minimum_norm_rerank")
    }
    lookback_ranker["development_method_counts"] = ranker_method_counts
    lookback_ranker["expected_development_method_count"] = 60
    lookback_ranker["development_methods_complete"] = all(
        count == 60 for count in ranker_method_counts.values()
    )
    lookback_ranker["matched_rerank_baseline_count"] = lookback_baseline_count
    lookback_ranker["expected_matched_rerank_baseline_count"] = 100
    lookback_ranker["matched_rerank_baseline_complete"] = (
        lookback_baseline_count == 100
    )
    studies = [lookback, lookback_ranker, pplm_split(), truthx]
    refinement = lookback_refinement_split()
    if refinement is not None:
        studies.append(refinement)
    report = {
        "all_disjoint": all(row["overlap_count"] == 0 for row in studies),
        "all_complete": all(
            row["development_complete"]
            and row["validation_complete"]
            and row.get("development_methods_complete", True)
            and row.get("matched_rerank_baseline_complete", True)
            and row.get("selection_complete", True)
            and row.get("validation_methods_complete", True)
            and row.get("ranker_summary_exists", True)
            for row in studies
        ),
        "studies": studies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_disjoint"]:
        raise SystemExit("split overlap detected")
    if args.require_complete and not report["all_complete"]:
        raise SystemExit("one or more final splits are incomplete")


if __name__ == "__main__":
    main()
