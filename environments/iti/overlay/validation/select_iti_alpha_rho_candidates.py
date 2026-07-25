#!/usr/bin/env python3
"""Select a broad alpha/rho candidate union for expensive local rejudging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .build_iti_alpha_rho_sweep import setting_tag
except ImportError:
    from build_iti_alpha_rho_sweep import setting_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    parser.add_argument("--columns-output", type=Path, required=True)
    parser.add_argument("--selected-settings-output", type=Path, required=True)
    parser.add_argument("--top-product", type=int, default=8)
    parser.add_argument("--top-joint", type=int, default=8)
    parser.add_argument("--best-per-alpha", action="store_true")
    parser.add_argument("--force-tag", action="append", default=[])
    parser.add_argument("--allow-subset", action="store_true")
    return parser.parse_args()


def select_candidates(
    summary: pd.DataFrame,
    settings: list[dict[str, object]],
    *,
    top_product: int,
    top_joint: int,
    best_per_alpha: bool,
    force_tags: list[str],
    allow_subset: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    setting_by_column = {
        f"{setting_tag(setting)}_answer": setting for setting in settings
    }
    combined = summary.loc[summary["input"].eq("COMBINED")].copy()
    combined = combined.loc[combined["answer_column"].isin(setting_by_column)].copy()
    if combined["answer_column"].duplicated().any():
        raise ValueError("Summary contains duplicate COMBINED answer columns")
    if not allow_subset and len(combined) != len(settings):
        missing = sorted(set(setting_by_column).difference(combined["answer_column"]))
        raise ValueError(f"Summary is missing {len(missing)} sweep settings")
    if combined.empty:
        raise ValueError("Summary contains no alpha/rho sweep settings")
    if not combined["parse_rate"].eq(1.0).all():
        raise ValueError("Candidate selection requires a 100% judge parse rate")

    combined["alpha"] = combined["answer_column"].map(
        lambda column: float(setting_by_column[column]["alpha"])
    )
    combined["rho"] = combined["answer_column"].map(
        lambda column: float(setting_by_column[column]["relative_cap"])
    )
    combined["tag"] = combined["answer_column"].str.removesuffix("_answer")
    reasons: dict[str, set[str]] = {}

    def add(rows: pd.DataFrame, reason: str) -> None:
        for tag in rows["tag"]:
            reasons.setdefault(tag, set()).add(reason)

    product_order = combined.sort_values(
        ["truth_x_info", "joint_truth_info", "truthful", "informative"],
        ascending=False,
    )
    joint_order = combined.sort_values(
        ["joint_truth_info", "truth_x_info", "truthful", "informative"],
        ascending=False,
    )
    add(product_order.head(top_product), "top_product")
    add(joint_order.head(top_joint), "top_joint")
    if best_per_alpha:
        add(product_order.groupby("alpha", sort=True).head(1), "best_for_alpha")
    for tag in force_tags:
        if tag not in set(combined["tag"]):
            raise ValueError(f"Forced tag is absent from the summary: {tag}")
        reasons.setdefault(tag, set()).add("forced")

    ranking = product_order.reset_index(drop=True)
    ranking.insert(0, "product_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking["tag"].isin(reasons)
    ranking["selection_reason"] = ranking["tag"].map(
        lambda tag: ",".join(sorted(reasons.get(tag, ())))
    )
    selected_columns = ranking.loc[ranking["selected"], "answer_column"].tolist()
    selected_settings = [setting_by_column[column] for column in selected_columns]
    return ranking, selected_settings


def main() -> None:
    args = parse_args()
    settings = json.loads(args.settings.read_text())
    ranking, selected = select_candidates(
        pd.read_csv(args.summary),
        settings,
        top_product=args.top_product,
        top_joint=args.top_joint,
        best_per_alpha=args.best_per_alpha,
        force_tags=args.force_tag,
        allow_subset=args.allow_subset,
    )
    args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.ranking_output, index=False)
    args.columns_output.write_text(
        "".join(f"{setting_tag(setting)}_answer\n" for setting in selected)
    )
    args.selected_settings_output.write_text(json.dumps(selected, indent=2) + "\n")
    print(
        ranking.loc[ranking["selected"], [
            "product_rank",
            "alpha",
            "rho",
            "truthful",
            "informative",
            "truth_x_info",
            "joint_truth_info",
            "selection_reason",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
