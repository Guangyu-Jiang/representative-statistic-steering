#!/usr/bin/env python3
"""Build ridge-only and cap-only group-direction validation settings."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("validation/iti_group_direction_regularization_stage1_settings.json")
METHODS = (
    "group_direction_probe_iti",
    "group_direction_probe_min_norm",
)
RIDGE_RATIOS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RELATIVE_CAPS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def base_setting(method: str) -> dict[str, object]:
    return {
        "method": method,
        "num_heads": 48,
        "alpha": 4.0,
        "target_quantile": 1.0,
    }


def main() -> None:
    ridge_settings = [
        base_setting(method) | {"ridge_ratio": ridge}
        for method in METHODS
        for ridge in RIDGE_RATIOS
    ]
    cap_settings = [
        base_setting(method) | {"ridge_ratio": 0.0, "relative_cap": cap}
        for method in METHODS
        for cap in RELATIVE_CAPS
    ]
    settings = [*ridge_settings, *cap_settings]
    OUTPUT.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {len(settings)} settings to {OUTPUT}")


if __name__ == "__main__":
    main()
