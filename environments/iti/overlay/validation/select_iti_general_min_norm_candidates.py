#!/usr/bin/env python3
"""Select general minimum-norm candidates for full validation confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .build_iti_general_min_norm_sweep import GENERAL_METHODS, setting_tag
except ImportError:
    from build_iti_general_min_norm_sweep import GENERAL_METHODS, setting_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    parser.add_argument("--selected-settings-output", type=Path, required=True)
    parser.add_argument("--top-mc1", type=int, default=5)
    parser.add_argument("--top-mc2", type=int, default=5)
    parser.add_argument("--top-mean", type=int, default=5)
    parser.add_argument("--top-efficiency", type=int, default=2)
    parser.add_argument(
        "--efficiency-max-mean-gap",
        type=float,
        help="only rank efficiency among settings within this gap of best mean MC",
    )
    parser.add_argument(
        "--include-setting-tags",
        nargs="*",
        default=(),
        help="general-setting tags that must be included in confirmation",
    )
    return parser.parse_args()


def select_candidates(
    summary: pd.DataFrame,
    settings: list[dict[str, object]],
    *,
    top_mc1: int,
    top_mc2: int,
    top_mean: int,
    top_efficiency: int,
    efficiency_max_mean_gap: float | None = None,
    include_setting_tags: tuple[str, ...] | list[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    by_tag = {setting_tag(setting): setting for setting in settings}
    general_tags = {
        tag
        for tag, setting in by_tag.items()
        if setting["method"] in GENERAL_METHODS
    }
    candidates = summary.loc[summary["setting"].isin(general_tags)].copy()
    if len(candidates) != len(general_tags):
        missing = sorted(general_tags.difference(candidates["setting"]))
        raise ValueError(f"Summary is missing {len(missing)} general settings")
    if candidates.empty:
        raise ValueError("Summary contains no general minimum-norm settings")

    candidates["mean_mc"] = (candidates["mc1"] + candidates["mc2"]) / 2.0
    candidates["efficiency"] = candidates["mean_mc"] / (
        1.0 + candidates["relative_action_norm"].fillna(float("inf"))
    )
    reasons: dict[str, set[str]] = {}

    def add(rows: pd.DataFrame, reason: str) -> None:
        for tag in rows["setting"]:
            reasons.setdefault(tag, set()).add(reason)

    add(candidates.nlargest(top_mc1, ["mc1", "mc2"]), "top_mc1")
    add(candidates.nlargest(top_mc2, ["mc2", "mc1"]), "top_mc2")
    add(candidates.nlargest(top_mean, ["mean_mc", "mc1"]), "top_mean")
    efficiency_candidates = candidates
    if efficiency_max_mean_gap is not None:
        threshold = candidates["mean_mc"].max() - efficiency_max_mean_gap
        efficiency_candidates = candidates.loc[candidates["mean_mc"] >= threshold]
    add(
        efficiency_candidates.nlargest(top_efficiency, ["efficiency", "mean_mc"]),
        "top_efficiency",
    )
    add(
        candidates.sort_values(["mean_mc", "mc1"], ascending=False)
        .groupby(candidates["setting"].map(lambda tag: by_tag[tag]["method"]))
        .head(1),
        "best_for_statistic",
    )
    unknown_tags = sorted(set(include_setting_tags).difference(general_tags))
    if unknown_tags:
        raise ValueError(f"Required general settings are unknown: {unknown_tags}")
    add(
        candidates.loc[candidates["setting"].isin(include_setting_tags)],
        "required_reference",
    )

    ranking = candidates.sort_values(
        ["mean_mc", "mc1", "mc2"], ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "mean_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking["setting"].isin(reasons)
    ranking["selection_reason"] = ranking["setting"].map(
        lambda tag: ",".join(sorted(reasons.get(tag, ())))
    )

    selected = [by_tag[tag] for tag in ranking.loc[ranking["selected"], "setting"]]
    selected.extend(
        setting for setting in settings if setting["method"] == "fixed_com"
    )
    return ranking, selected


def main() -> None:
    args = parse_args()
    settings = json.loads(args.settings.read_text())
    ranking, selected = select_candidates(
        pd.read_csv(args.summary),
        settings,
        top_mc1=args.top_mc1,
        top_mc2=args.top_mc2,
        top_mean=args.top_mean,
        top_efficiency=args.top_efficiency,
        efficiency_max_mean_gap=args.efficiency_max_mean_gap,
        include_setting_tags=args.include_setting_tags,
    )
    args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.ranking_output, index=False)
    args.selected_settings_output.write_text(json.dumps(selected, indent=2) + "\n")
    print(
        ranking.loc[
            ranking["selected"],
            [
                "mean_rank",
                "setting",
                "n",
                "mc1",
                "mc2",
                "mean_mc",
                "relative_action_norm",
                "selection_reason",
            ],
        ].to_string(index=False)
    )
    print(f"Selected {len(selected) - 2} general settings plus two fixed controls")


if __name__ == "__main__":
    main()
