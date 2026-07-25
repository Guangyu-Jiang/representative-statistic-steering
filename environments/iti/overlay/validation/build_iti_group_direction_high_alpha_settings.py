#!/usr/bin/env python3
"""Build the standardized and raw group-direction high-alpha grid."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("validation/iti_group_direction_high_alpha_settings.json")
METHODS = (
    "group_direction_probe_iti",
    "group_direction_probe_min_norm",
)
NORMALIZATIONS = ("standardized", "raw")
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
ALPHAS = (5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)


def main() -> None:
    settings = [
        {
            "method": method,
            "num_heads": 48,
            "alpha": alpha,
            "target_quantile": quantile,
            "ridge_ratio": 0.0,
            "probe_score_normalization": normalization,
        }
        for normalization in NORMALIZATIONS
        for method in METHODS
        for quantile in QUANTILES
        for alpha in ALPHAS
    ]
    OUTPUT.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {len(settings)} settings to {OUTPUT}")


if __name__ == "__main__":
    main()
