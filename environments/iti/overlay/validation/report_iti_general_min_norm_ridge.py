#!/usr/bin/env python3
"""Report ridge/action tradeoffs against matched zero-ridge ITI settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .build_iti_general_min_norm_sweep import setting_tag
except ImportError:
    from build_iti_general_min_norm_sweep import setting_tag


MATCH_KEYS = ["method", "num_heads", "alpha", "target_quantile", "relative_cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-mean-mc-loss",
        type=float,
        default=0.005,
        help="Largest allowed loss in mean(MC1, MC2) versus matched ridge zero.",
    )
    parser.add_argument(
        "--max-individual-mc-loss",
        type=float,
        default=0.01,
        help="Largest allowed loss in either MC1 or MC2 versus matched ridge zero.",
    )
    return parser.parse_args()


def build_report(
    summary: pd.DataFrame,
    settings: list[dict[str, object]],
    *,
    max_mean_mc_loss: float,
    max_individual_mc_loss: float,
) -> pd.DataFrame:
    setting_rows = []
    for setting in settings:
        if not str(setting["method"]).startswith("aggregate_"):
            continue
        row = dict(setting)
        row["setting"] = setting_tag(setting)
        setting_rows.append(row)
    metadata = pd.DataFrame(setting_rows)
    report = metadata.merge(summary, on="setting", how="left", validate="one_to_one")
    if report[["mc1", "mc2", "relative_action_norm"]].isna().any().any():
        missing = report.loc[report["mc1"].isna(), "setting"].tolist()
        raise ValueError(f"Summary is missing {len(missing)} ridge settings: {missing}")

    report["effective_alpha"] = report["alpha"] / (1.0 + report["ridge_ratio"])
    report["mean_mc"] = (report["mc1"] + report["mc2"]) / 2.0
    references = report.loc[report["ridge_ratio"].eq(0.0)].copy()
    if references.duplicated(MATCH_KEYS).any():
        raise ValueError("Zero-ridge references are not unique")
    reference_columns = MATCH_KEYS + ["mc1", "mc2", "mean_mc", "relative_action_norm"]
    references = references[reference_columns].rename(
        columns={
            "mc1": "reference_mc1",
            "mc2": "reference_mc2",
            "mean_mc": "reference_mean_mc",
            "relative_action_norm": "reference_action_norm",
        }
    )
    report = report.merge(references, on=MATCH_KEYS, how="left", validate="many_to_one")
    if report["reference_mc1"].isna().any():
        raise ValueError("At least one ridge setting lacks a matched zero-ridge reference")

    report["mc1_difference"] = report["mc1"] - report["reference_mc1"]
    report["mc2_difference"] = report["mc2"] - report["reference_mc2"]
    report["mean_mc_difference"] = report["mean_mc"] - report["reference_mean_mc"]
    report["action_reduction"] = report["reference_action_norm"] - report["relative_action_norm"]
    report["action_reduction_pct"] = (
        100.0 * report["action_reduction"] / report["reference_action_norm"].clip(lower=1e-12)
    )
    report["similar_performance"] = (
        report["mean_mc_difference"].ge(-max_mean_mc_loss)
        & report["mc1_difference"].ge(-max_individual_mc_loss)
        & report["mc2_difference"].ge(-max_individual_mc_loss)
    )
    report["ridge_reduces_action"] = report["action_reduction"].gt(0.0)
    return report.sort_values(["alpha", "ridge_ratio"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    report = build_report(
        pd.read_csv(args.summary),
        json.loads(args.settings.read_text()),
        max_mean_mc_loss=args.max_mean_mc_loss,
        max_individual_mc_loss=args.max_individual_mc_loss,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    selected = report.loc[
        report["ridge_ratio"].gt(0.0)
        & report["similar_performance"]
        & report["ridge_reduces_action"]
    ].copy()
    if not selected.empty:
        selected = selected.sort_values(
            ["alpha", "action_reduction", "mean_mc"],
            ascending=[True, False, False],
        ).groupby("alpha", as_index=False).head(1)
    selected_path = args.output.with_name(f"{args.output.stem}_selected.csv")
    selected.to_csv(selected_path, index=False)
    columns = [
        "setting",
        "effective_alpha",
        "mc1",
        "mc2",
        "relative_action_norm",
        "mc1_difference",
        "mc2_difference",
        "action_reduction_pct",
        "similar_performance",
    ]
    print(report[columns].to_string(index=False))
    print(f"Wrote {args.output} and {selected_path}")


if __name__ == "__main__":
    main()
