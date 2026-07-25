#!/usr/bin/env python3
"""Reproduce ReDeEP's chunk detector and run a grouped held-out audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repstat_steering.redeep_control import (
    evaluate_redeep_detector,
    load_official_chunk_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "external/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/"
            "llama2_7B_response_chunk.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/redeep/detector_reproduction"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_official_chunk_features(args.features)
    rows: list[dict[str, object]] = []
    official, official_scored = evaluate_redeep_detector(
        frame, leaky_official_protocol=True
    )
    rows.append(official)
    official_scored.to_csv(args.output_dir / "official_scored_chunks.csv", index=False)
    for seed in args.seeds:
        result, scored = evaluate_redeep_detector(frame, seed=seed)
        rows.append(result)
        scored.to_csv(args.output_dir / f"heldout_seed{seed}_scored_chunks.csv", index=False)

    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(args.output_dir / "detector_summary.csv", index=False)
    heldout = summary[summary["protocol"] == "grouped_heldout"]
    report = {
        "source_features": str(args.features),
        "official": official,
        "heldout_seeds": args.seeds,
        "heldout_response_auc_mean": float(heldout["response_auc"].mean()),
        "heldout_response_auc_std": float(heldout["response_auc"].std(ddof=1)),
        "heldout_chunk_auc_mean": float(heldout["chunk_auc"].mean()),
        "heldout_chunk_auc_std": float(heldout["chunk_auc"].std(ddof=1)),
    }
    (args.output_dir / "detector_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
