#!/usr/bin/env python3
"""Select one high-alpha candidate per inverse map and normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from select_iti_group_direction_candidates import setting_tag


METHODS = (
    "group_direction_probe_iti",
    "group_direction_probe_min_norm",
)
NORMALIZATIONS = ("standardized", "raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-settings", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    settings = json.loads(args.grid.read_text())
    settings_by_tag = {setting_tag(item): item for item in settings}
    summary["joint_score"] = (summary["mc1"] + summary["mc2"]) / 2.0

    selected_settings = []
    selected_rows = []
    for normalization in NORMALIZATIONS:
        for method in METHODS:
            tags = {
                setting_tag(item)
                for item in settings
                if item["method"] == method
                and item.get("probe_score_normalization", "standardized")
                == normalization
            }
            candidates = summary[summary["setting"].isin(tags)].copy()
            if candidates.empty:
                raise ValueError(f"No rows for {normalization}/{method}")
            candidates = candidates.sort_values(
                ["joint_score", "relative_action_norm"],
                ascending=[False, True],
            )
            winner = candidates.iloc[0].copy()
            tag = str(winner["setting"])
            selected_settings.append(settings_by_tag[tag])
            winner["normalization"] = normalization
            winner["method_name"] = method
            selected_rows.append(winner)

    args.output_settings.write_text(json.dumps(selected_settings, indent=2) + "\n")
    report = pd.DataFrame(selected_rows)
    report.to_csv(args.output_report, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
