#!/usr/bin/env python3
"""Select one validation candidate per group-direction inverse map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = (
    "group_direction_probe_iti",
    "group_direction_probe_min_norm",
)


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def setting_tag(setting: dict[str, object]) -> str:
    pieces = [
        str(setting["method"]),
        f"k{int(setting['num_heads'])}",
        f"a{slug(float(setting['alpha']))}",
    ]
    if setting.get("target_quantile") is not None:
        pieces.append(f"q{slug(float(setting['target_quantile']))}")
    if setting.get("ridge_ratio") is not None:
        pieces.append(f"r{slug(float(setting['ridge_ratio']))}")
    if setting.get("relative_cap") is not None:
        pieces.append(f"c{slug(float(setting['relative_cap']))}")
    if setting.get("coefficient_cap") is not None:
        pieces.append(f"b{slug(float(setting['coefficient_cap']))}")
    if setting.get("probe_score_normalization") == "raw":
        pieces.append("nraw")
    return "_".join(pieces)


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
    for method in METHODS:
        candidates = summary[summary["setting"].str.startswith(f"{method}_")].copy()
        if candidates.empty:
            raise ValueError(f"No completed validation rows for {method}")
        candidates = candidates.sort_values(
            ["joint_score", "relative_action_norm"],
            ascending=[False, True],
        )
        winner = candidates.iloc[0]
        tag = str(winner["setting"])
        selected_settings.append(settings_by_tag[tag])
        selected_rows.append(winner)

    args.output_settings.write_text(json.dumps(selected_settings, indent=2) + "\n")
    report = pd.DataFrame(selected_rows)
    report.to_csv(args.output_report, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
