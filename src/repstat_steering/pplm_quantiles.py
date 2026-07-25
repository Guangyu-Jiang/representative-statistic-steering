"""Calibration-file utilities for class-conditional PPLM margin targets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping


PPLM_SST5_DATASET_LABELS = {
    "positive": 4,  # very positive
    "negative": 0,  # very negative
}


def quantile_key(value: float) -> str:
    if not 0 <= value <= 1:
        raise ValueError("quantile must be in [0, 1]")
    return format(float(value), ".6g")


def load_quantile_margin_targets(
    path: str | Path,
    quantile: float,
) -> dict[str, float]:
    """Load positive/negative absolute margins for one calibrated quantile."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported PPLM quantile calibration schema")
    requested = float(quantile)
    targets: dict[str, float] = {}
    for label in ("positive", "negative"):
        available: Mapping[str, float] = payload["targets"][label]["quantiles"]
        matches = [
            float(value)
            for key, value in available.items()
            if math.isclose(float(key), requested, rel_tol=0, abs_tol=1e-9)
        ]
        if not matches:
            choices = ", ".join(sorted(available, key=float))
            raise ValueError(
                f"quantile {requested:g} is unavailable for {label}; choices: {choices}"
            )
        targets[label] = matches[0]
    return targets
