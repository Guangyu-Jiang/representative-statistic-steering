from __future__ import annotations

import pandas as pd

from validation.build_iti_general_min_norm_sweep import setting_tag
from validation.report_iti_general_min_norm_ridge import build_report


def test_ridge_report_matches_zero_ridge_and_flags_tradeoff() -> None:
    settings = [
        {
            "method": "aggregate_com",
            "num_heads": 48,
            "alpha": 20.0,
            "target_quantile": 0.75,
            "ridge_ratio": ridge,
            "relative_cap": 3.0,
        }
        for ridge in (0.0, 1.0, 2.0)
    ]
    summary = pd.DataFrame(
        [
            {
                "setting": setting_tag(setting),
                "mc1": mc1,
                "mc2": mc2,
                "relative_action_norm": norm,
            }
            for setting, mc1, mc2, norm in zip(
                settings,
                (0.43, 0.425, 0.39),
                (0.60, 0.598, 0.55),
                (2.0, 1.5, 1.0),
                strict=True,
            )
        ]
    )

    report = build_report(
        summary,
        settings,
        max_mean_mc_loss=0.005,
        max_individual_mc_loss=0.01,
    )

    ridge_one = report.loc[report["ridge_ratio"].eq(1.0)].iloc[0]
    ridge_two = report.loc[report["ridge_ratio"].eq(2.0)].iloc[0]
    assert ridge_one["effective_alpha"] == 10.0
    assert ridge_one["action_reduction_pct"] == 25.0
    assert bool(ridge_one["similar_performance"])
    assert not bool(ridge_two["similar_performance"])
