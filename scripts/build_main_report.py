#!/usr/bin/env python3
"""Build compact comparison tables from completed PPLM and TruthX runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pplm-evaluated", required=True)
    parser.add_argument("--truthx-summary", nargs="+", required=True)
    parser.add_argument("--output-dir", default="artifacts/reports")
    return parser.parse_args()


def summarize_pplm(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.groupby(["method", "target_label"], as_index=False).agg(
        n=("text", "size"),
        external_target_probability=("external_target_probability", "mean"),
        external_success=(
            "external_target_probability", lambda values: (values >= 0.5).mean()
        ),
        perplexity=("perplexity", "mean"),
        relative_cache_change=("mean_relative_cache_change", "mean"),
    )


def summarize_truthx(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with open(path) as handle:
            summary = json.load(handle)
        configuration = summary["configuration"]
        method = configuration["method"]
        if method == "baseline":
            target_mode = "none"
            target_strength = None
            norm_cap = None
        elif method == "truthx_original":
            target_mode = "published_decoder_edit"
            target_strength = configuration["original_strength"]
            norm_cap = None
        else:
            target_mode = configuration["target_mode"]
            target_strength = configuration["target_strength"]
            norm_cap = configuration["maximum_relative_norm"]
        rows.append(
            {
                "method": method,
                "target_mode": target_mode,
                "target_strength": target_strength,
                "norm_cap": norm_cap,
                "n": summary["n"],
                "mc1": summary["mc1"],
                "mc2": summary["mc2"],
                "mc3": summary["mc3"],
                "relative_state_change": summary["mean_relative_action_norm"],
                "initial_target_rmse": summary["initial_target_rmse"],
                "final_target_rmse": summary["final_target_rmse"],
                "valid_scores": summary["valid_scores"],
                "source": path,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pplm = summarize_pplm(args.pplm_evaluated)
    truthx = summarize_truthx(args.truthx_summary)
    pplm.to_csv(output_dir / "pplm_main_comparison.csv", index=False)
    truthx.to_csv(output_dir / "truthx_main_comparison.csv", index=False)
    print("PPLM\n", pplm.to_string(index=False))
    print("\nTruthX\n", truthx.drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()
