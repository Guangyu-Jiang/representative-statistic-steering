#!/usr/bin/env python3
"""Freeze one validation-selected general minimum-norm setting."""

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
    parser.add_argument("--candidate-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    return parser.parse_args()


def select_final(
    summary: pd.DataFrame,
    settings: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    by_tag = {setting_tag(setting): setting for setting in settings}
    general_tags = {
        tag
        for tag, setting in by_tag.items()
        if setting["method"] in GENERAL_METHODS
    }
    ranking = summary.loc[summary["setting"].isin(general_tags)].copy()
    if len(ranking) != len(general_tags):
        missing = sorted(general_tags.difference(ranking["setting"]))
        raise ValueError(f"Confirmation summary is missing {len(missing)} candidates")
    ranking["mean_mc"] = (ranking["mc1"] + ranking["mc2"]) / 2.0
    ranking = ranking.sort_values(
        ["mean_mc", "mc1", "mc2", "relative_action_norm"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", range(1, len(ranking) + 1))
    winner = by_tag[str(ranking.iloc[0]["setting"])]
    controls = [setting for setting in settings if setting["method"] == "fixed_com"]
    return ranking, [winner, *controls]


def main() -> None:
    args = parse_args()
    settings = json.loads(args.candidate_settings.read_text())
    ranking, selected = select_final(pd.read_csv(args.summary), settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.ranking_output, index=False)
    print(ranking.to_string(index=False))
    print(f"Selected {setting_tag(selected[0])} with {len(selected) - 1} controls")


if __name__ == "__main__":
    main()
