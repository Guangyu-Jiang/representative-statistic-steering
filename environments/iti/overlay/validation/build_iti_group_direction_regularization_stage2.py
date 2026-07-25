#!/usr/bin/env python3
"""Build the focused minimum-norm ridge-cap refinement grid."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("validation/iti_group_direction_regularization_stage2_settings.json")
RIDGE_RATIOS = (0.0, 0.05, 0.1, 0.25)
RELATIVE_CAPS = (1.5, 1.75, 2.0, 2.25, 2.5, 3.0)


def main() -> None:
    settings = [
        {
            "method": "group_direction_probe_min_norm",
            "num_heads": 48,
            "alpha": 4.0,
            "target_quantile": 1.0,
            "ridge_ratio": ridge,
            "relative_cap": cap,
        }
        for ridge in RIDGE_RATIOS
        for cap in RELATIVE_CAPS
    ]
    OUTPUT.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {len(settings)} settings to {OUTPUT}")


if __name__ == "__main__":
    main()
