#!/usr/bin/env python3
"""Aggregate locally judged token-specific exact-H0 steering campaigns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import judge_exact_h0_gn_local as local_judge  # noqa: E402


ACCEPTABLE = {"GROUNDED_ACCEPTABLE", "GENERIC_ACCEPTABLE"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-root", default="artifacts/reports/exact_h0_perturbation_campaign")
    parser.add_argument(
        "--judge-run-name",
        default=local_judge._judge_run_name(
            "Qwen/Qwen2.5-7B-Instruct", "rotating_choice", None
        ),
    )
    return parser.parse_args()


def _first_matching_control(row: pd.Series, frame: pd.DataFrame) -> pd.Series | None:
    candidates = frame.loc[frame["topology_alpha"].eq(0.0)].copy()
    keys = (
        "campaign",
        "dataset",
        "model",
        "method",
        "direction_source",
        "retrieval_feature_mode",
        "retrieval_geometry",
        "behavior_rank",
        "causal_anchor_ratio",
        "shared_intervention_site",
        "target_mode",
        "neighbor_k",
        "mean_alpha",
        "shared_target_ratio",
        "lambda",
        "damping",
        "trust_ratio",
        "topology_decode_mode",
        "topology_decode_scale",
        "topology_decode_suffix_fraction",
        "classifier_target_quantile",
        "transport_match_mode",
        "transport_prior_ratio",
    )
    for key in keys:
        if key in {"topology_decode_scale", "topology_decode_suffix_fraction"} and row.get(
            "topology_decode_mode", "none"
        ) == "none":
            continue
        if key not in frame or pd.isna(row.get(key)):
            continue
        candidates = candidates.loc[candidates[key].eq(row[key])]
    if candidates.empty and row.get("topology_decode_mode", "none") != "none":
        # A zero token deformation makes decode reduction irrelevant.
        relaxed = row.copy()
        relaxed["topology_decode_mode"] = "none"
        relaxed["topology_decode_scale"] = 0.0
        return _first_matching_control(relaxed, frame)
    return None if candidates.empty else candidates.iloc[0]


def _paired_counts(judged_path: str, control_path: str) -> dict[str, Any]:
    candidate = pd.read_parquet(judged_path, columns=["example_id", "local_judge_label"])
    control = pd.read_parquet(control_path, columns=["example_id", "local_judge_label"])
    paired = control.merge(candidate, on="example_id", suffixes=("_control", "_candidate"))
    control_total = paired["local_judge_label_control"].isin(ACCEPTABLE)
    candidate_total = paired["local_judge_label_candidate"].isin(ACCEPTABLE)
    control_grounded = paired["local_judge_label_control"].eq("GROUNDED_ACCEPTABLE")
    candidate_grounded = paired["local_judge_label_candidate"].eq("GROUNDED_ACCEPTABLE")
    total_improved = int((~control_total & candidate_total).sum())
    total_worsened = int((control_total & ~candidate_total).sum())
    grounded_improved = int((~control_grounded & candidate_grounded).sum())
    grounded_worsened = int((control_grounded & ~candidate_grounded).sum())

    def exact_pvalue(improved: int, worsened: int) -> float:
        discordant = improved + worsened
        if discordant == 0:
            return 1.0
        return float(binomtest(improved, n=discordant, p=0.5).pvalue)

    return {
        "paired_n": len(paired),
        "paired_total_improved": total_improved,
        "paired_total_worsened": total_worsened,
        "paired_total_exact_pvalue": exact_pvalue(total_improved, total_worsened),
        "paired_grounded_improved": grounded_improved,
        "paired_grounded_worsened": grounded_worsened,
        "paired_grounded_exact_pvalue": exact_pvalue(
            grounded_improved,
            grounded_worsened,
        ),
    }


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    pattern = (
        f"steering_exact_h0_*/_local_fourway_judge_{args.judge_run_name}/"
        "exact_h0_gn_local_fourway_summary.csv"
    )
    paths = sorted(artifact_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No campaign summaries matched {artifact_root / pattern}")

    tables = []
    for path in paths:
        table = pd.read_csv(path)
        table["campaign"] = path.parent.parent.name
        table["summary_path"] = str(path)
        tables.append(table)
    frame = pd.concat(tables, ignore_index=True)
    frame = frame.drop_duplicates("raw_path", keep="last").reset_index(drop=True)
    defaults: dict[str, Any] = {
        "topology_decode_mode": "none",
        "topology_decode_scale": 0.0,
        "topology_decode_suffix_fraction": float("nan"),
        "classifier_target_quantile": float("nan"),
        "transport_match_mode": "nearest",
        "transport_prior_ratio": float("nan"),
        "shared_target_ratio": float("nan"),
        "degenerate_responses": 0,
        "direction_source": "observed_groups",
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value

    effect_rows = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        control = None if float(row["topology_alpha"]) == 0.0 else _first_matching_control(row, frame)
        if control is not None:
            record["matched_control_total_pct"] = float(control["total_acceptable_pct"])
            record["matched_control_grounded_pct"] = float(control["grounded_pct"])
            record["matched_control_neither_pct"] = float(control["neither_pct"])
            record["matched_control_unique_pct"] = float(control["unique_pct"])
            record["topology_total_effect_pp"] = float(
                row["total_acceptable_pct"] - control["total_acceptable_pct"]
            )
            record["topology_grounded_effect_pp"] = float(
                row["grounded_pct"] - control["grounded_pct"]
            )
            record["topology_neither_effect_pp"] = float(
                row["neither_pct"] - control["neither_pct"]
            )
            try:
                record.update(_paired_counts(str(row["judged_path"]), str(control["judged_path"])))
            except (FileNotFoundError, KeyError):
                pass
        effect_rows.append(record)
    report = pd.DataFrame(effect_rows)
    report["degenerate_pct"] = 100.0 * report["degenerate_responses"] / report["n"].clip(lower=1)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_path = output_root / "all_settings.csv"
    report.to_csv(all_path, index=False)
    report.to_parquet(all_path.with_suffix(".parquet"), index=False)
    eligible = report.loc[
        report["unique_pct"].ge(90.0)
        & report["neither_pct"].le(20.0)
        & report["degenerate_pct"].le(5.0)
    ].copy()
    best = eligible.sort_values(
        ["dataset", "model", "grounded_pct", "total_acceptable_pct"],
        ascending=[True, True, False, False],
    ).groupby(["dataset", "model"], as_index=False).head(1)
    best.to_csv(output_root / "best_quality_constrained.csv", index=False)
    incremental = eligible.loc[
        eligible["topology_alpha"].gt(0.0)
        & eligible["topology_grounded_effect_pp"].notna()
        & eligible["topology_total_effect_pp"].notna()
    ].copy()
    best_incremental = incremental.sort_values(
        [
            "dataset",
            "model",
            "topology_grounded_effect_pp",
            "topology_total_effect_pp",
            "grounded_pct",
        ],
        ascending=[True, True, False, False, False],
    ).groupby(["dataset", "model"], as_index=False).head(1)
    best_incremental.to_csv(output_root / "best_incremental_effect.csv", index=False)
    print(
        best[
            [
                "dataset",
                "model",
                "campaign",
                "target_mode",
                "topology_alpha",
                "mean_alpha",
                "shared_target_ratio",
                "grounded_pct",
                "generic_pct",
                "total_acceptable_pct",
                "neither_pct",
                "unique_pct",
                "degenerate_pct",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Wrote {all_path}", flush=True)


if __name__ == "__main__":
    main()
