#!/usr/bin/env python3
"""Build the expanded exact minimum-norm ITI validation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ALPHAS = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
DEFAULT_QUANTILES = (0.75, 0.90)
DEFAULT_RELATIVE_CAPS = (1.5, 2.0, 3.0)
GENERAL_METHODS = ("aggregate_com", "aggregate_probe")


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def setting_tag(setting: dict[str, object]) -> str:
    pieces = [
        str(setting["method"]),
        f"k{int(setting['num_heads'])}",
        f"a{slug(float(setting['alpha']))}",
    ]
    for key, prefix in (
        ("target_quantile", "q"),
        ("ridge_ratio", "r"),
        ("relative_cap", "c"),
        ("coefficient_cap", "b"),
    ):
        value = setting.get(key)
        if value is not None:
            pieces.append(f"{prefix}{slug(float(value))}")
    return "_".join(pieces)


def build_settings(
    alphas: list[float],
    quantiles: list[float],
    relative_caps: list[float],
    *,
    num_heads: int,
    ridge_ratio: float,
    include_legacy_best: bool,
    fixed_alphas: list[float],
    methods: list[str] | None = None,
    ridge_ratios: list[float] | None = None,
) -> list[dict[str, object]]:
    selected_methods = list(GENERAL_METHODS if methods is None else methods)
    selected_ridges = [ridge_ratio] if ridge_ratios is None else list(ridge_ratios)
    settings: list[dict[str, object]] = [
        {
            "method": method,
            "num_heads": num_heads,
            "alpha": alpha,
            "target_quantile": quantile,
            "ridge_ratio": ridge,
            "relative_cap": cap,
        }
        for method in selected_methods
        for alpha in alphas
        for quantile in quantiles
        for ridge in selected_ridges
        for cap in relative_caps
    ]
    if include_legacy_best:
        settings.append(
            {
                "method": "aggregate_com",
                "num_heads": num_heads,
                "alpha": 16.0,
                "target_quantile": 0.75,
                "ridge_ratio": 0.0,
                "relative_cap": 1.0,
            }
        )
    settings.extend(
        {
            "method": "fixed_com",
            "num_heads": num_heads,
            "alpha": alpha,
        }
        for alpha in fixed_alphas
    )

    unique: dict[str, dict[str, object]] = {}
    for setting in settings:
        unique[setting_tag(setting)] = setting
    return list(unique.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--target-quantiles", type=float, nargs="+", default=DEFAULT_QUANTILES
    )
    parser.add_argument(
        "--relative-caps", type=float, nargs="+", default=DEFAULT_RELATIVE_CAPS
    )
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument("--ridge-ratio", type=float, default=0.0)
    parser.add_argument(
        "--ridge-ratios",
        type=float,
        nargs="+",
        help="Optional ridge grid; overrides --ridge-ratio when provided.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=GENERAL_METHODS,
        default=list(GENERAL_METHODS),
    )
    parser.add_argument("--fixed-alphas", type=float, nargs="+", default=(8.0, 15.0))
    parser.add_argument(
        "--no-legacy-best", dest="include_legacy_best", action="store_false"
    )
    parser.set_defaults(include_legacy_best=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(
        args.alphas,
        args.target_quantiles,
        args.relative_caps,
        num_heads=args.num_heads,
        ridge_ratio=args.ridge_ratio,
        include_legacy_best=args.include_legacy_best,
        fixed_alphas=args.fixed_alphas,
        methods=args.methods,
        ridge_ratios=args.ridge_ratios,
    )
    tags = [setting_tag(setting) for setting in settings]
    if len(tags) != len(set(tags)):
        raise ValueError("The general minimum-norm grid contains duplicate tags")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")
    general_count = sum(setting["method"] in GENERAL_METHODS for setting in settings)
    print(
        f"Wrote {len(settings)} settings ({general_count} general, "
        f"{len(settings) - general_count} controls) to {args.output}"
    )


if __name__ == "__main__":
    main()
