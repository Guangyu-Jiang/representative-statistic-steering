#!/usr/bin/env python3
"""Select a bounded controller from validation-only local-judge results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from validate_causal_head_perturbation import Setting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--settings-file", type=Path, required=True)
    parser.add_argument("--fixed-alpha", type=float, default=8.0)
    parser.add_argument("--output-settings", type=Path, required=True)
    parser.add_argument("--output-ranking", type=Path, required=True)
    parser.add_argument("--output-tag", type=Path, required=True)
    return parser.parse_args()


def select_bounded(
    summary: pd.DataFrame,
    settings: list[Setting],
) -> tuple[Setting, pd.DataFrame]:
    candidates = summary[
        (summary["input"] == "COMBINED")
        & summary["answer_column"].str.startswith("bounded_targeted_probe_iti_")
    ].copy()
    if candidates.empty:
        raise ValueError("No combined bounded-controller rows were found")
    candidates["tag"] = candidates["answer_column"].str.removesuffix("_answer")
    by_tag = {setting.tag: setting for setting in settings}
    unknown = sorted(set(candidates["tag"]).difference(by_tag))
    if unknown:
        raise ValueError(f"Judged settings are absent from the manifest: {unknown}")
    candidates["coefficient_cap"] = candidates["tag"].map(
        lambda tag: by_tag[tag].coefficient_cap
    )
    ranking = candidates.sort_values(
        ["truth_x_info", "joint_truth_info", "coefficient_cap"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return by_tag[ranking.loc[0, "tag"]], ranking


def main() -> None:
    args = parse_args()
    settings = [
        Setting(**item) for item in json.loads(args.settings_file.read_text())
    ]
    selected, ranking = select_bounded(pd.read_csv(args.summary), settings)
    final_settings = [
        Setting(method="fixed_com", num_heads=selected.num_heads, alpha=args.fixed_alpha),
        selected,
    ]
    for path in (args.output_settings, args.output_ranking, args.output_tag):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_settings.write_text(
        json.dumps([asdict(setting) for setting in final_settings], indent=2) + "\n"
    )
    ranking.to_csv(args.output_ranking, index=False)
    args.output_tag.write_text(selected.tag + "\n")
    print(ranking.to_string(index=False))
    print(f"selected={selected.tag}")


if __name__ == "__main__":
    main()
