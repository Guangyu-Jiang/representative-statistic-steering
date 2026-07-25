#!/usr/bin/env python3
"""Create a prefix subset of a cached, validation-ranked ITI head bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_causal_head_perturbation import (
    StatisticBank,
    load_statistics,
    save_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def subset_bank(bank: StatisticBank, num_heads: int) -> StatisticBank:
    available = len(bank.top_heads)
    if not 0 < num_heads <= available:
        raise ValueError(f"num_heads must be in [1, {available}], got {num_heads}")
    return StatisticBank(
        top_heads=bank.top_heads[:num_heads],
        com_directions=bank.com_directions[:num_heads],
        fixed_scales=bank.fixed_scales[:num_heads],
        com_means=bank.com_means[:num_heads],
        com_stds=bank.com_stds[:num_heads],
        com_truthful=bank.com_truthful[:, :num_heads],
        probe_directions=bank.probe_directions[:num_heads],
        probe_intercepts=bank.probe_intercepts[:num_heads],
        probe_means=bank.probe_means[:num_heads],
        probe_stds=bank.probe_stds[:num_heads],
        probe_truthful=bank.probe_truthful[:, :num_heads],
        probe_group_direction=bank.probe_group_direction[:0],
        probe_group_positive_projection=bank.probe_group_positive_projection[:0],
        probe_group_negative_projection=bank.probe_group_negative_projection[:0],
        validation_accuracies=bank.validation_accuracies[:num_heads],
    )


def main() -> None:
    args = parse_args()
    save_statistics(args.output, subset_bank(load_statistics(args.source), args.num_heads))


if __name__ == "__main__":
    main()
