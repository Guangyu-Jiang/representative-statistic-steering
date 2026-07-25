#!/usr/bin/env python3
"""Build a deterministic bounded-ITI alpha/rho validation sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ALPHAS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0)
DEFAULT_RHOS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def setting_tag(setting: dict[str, float | int | str]) -> str:
    return (
        f"{setting['method']}_k{setting['num_heads']}"
        f"_a{slug(float(setting['alpha']))}"
        f"_q{slug(float(setting['target_quantile']))}"
        f"_r{slug(float(setting['ridge_ratio']))}"
        f"_c{slug(float(setting['relative_cap']))}"
        f"_b{slug(float(setting['coefficient_cap']))}"
    )


def build_settings(
    alphas: list[float],
    rhos: list[float],
    *,
    num_heads: int,
    target_quantile: float,
    ridge_ratio: float,
    coefficient_cap: float,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "method": "bounded_targeted_probe_iti",
            "num_heads": num_heads,
            "alpha": alpha,
            "target_quantile": target_quantile,
            "ridge_ratio": ridge_ratio,
            "relative_cap": rho,
            "coefficient_cap": coefficient_cap,
        }
        for alpha in alphas
        for rho in rhos
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns-output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS)
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--ridge-ratio", type=float, default=0.0)
    parser.add_argument("--coefficient-cap", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(
        args.alphas,
        args.rhos,
        num_heads=args.num_heads,
        target_quantile=args.target_quantile,
        ridge_ratio=args.ridge_ratio,
        coefficient_cap=args.coefficient_cap,
    )
    tags = [setting_tag(setting) for setting in settings]
    if len(tags) != len(set(tags)):
        raise ValueError("The alpha/rho grid produces duplicate setting tags")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")
    args.columns_output.parent.mkdir(parents=True, exist_ok=True)
    args.columns_output.write_text("".join(f"{tag}_answer\n" for tag in tags))
    print(f"Wrote {len(settings)} settings to {args.output}")


if __name__ == "__main__":
    main()
