#!/usr/bin/env python3
"""Build high-quantile K-dimensional direct-target settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.95, 0.975, 0.99, 1.0),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = [
        {
            "method": method,
            "num_heads": args.num_heads,
            "alpha": 1.0,
            "target_quantile": quantile,
            "ridge_ratio": 0.0,
        }
        for method in ("headwise_probe_iti", "headwise_probe_min_norm")
        for quantile in args.quantiles
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
