#!/usr/bin/env python3
"""Build stronger-target and regularized direct-target ITI settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def setting(
    quantile: float,
    *,
    ridge: float = 0.0,
    relative_cap: float | None = None,
) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "method": "targeted_probe_iti",
        "num_heads": 48,
        "alpha": 1.0,
        "target_quantile": quantile,
        "ridge_ratio": ridge,
    }
    if relative_cap is not None:
        result["relative_cap"] = relative_cap
    return result


def main() -> None:
    args = parse_args()
    settings = [setting(q) for q in (0.95, 0.975, 0.99)]
    settings.extend(
        setting(q, ridge=ridge)
        for q in (0.95, 0.99)
        for ridge in (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
    )
    settings.extend(
        setting(q, relative_cap=cap)
        for q in (0.95, 0.99)
        for cap in (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
