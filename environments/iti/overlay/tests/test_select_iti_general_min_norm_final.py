from __future__ import annotations

import pandas as pd

from validation.build_iti_general_min_norm_sweep import build_settings, setting_tag
from validation.select_iti_general_min_norm_final import select_final


def test_select_final_uses_mean_mc_and_retains_controls() -> None:
    settings = build_settings(
        [8.0, 16.0],
        [0.75],
        [2.0],
        num_heads=48,
        ridge_ratio=0.0,
        include_legacy_best=False,
        fixed_alphas=[8.0, 15.0],
    )
    general = [setting for setting in settings if setting["method"] != "fixed_com"]
    summary = pd.DataFrame(
        [
            {
                "setting": setting_tag(setting),
                "mc1": 0.4 + index * 0.01,
                "mc2": 0.5 + index * 0.02,
                "relative_action_norm": 1.0,
            }
            for index, setting in enumerate(general)
        ]
    )

    ranking, selected = select_final(summary, settings)

    assert setting_tag(selected[0]) == ranking.iloc[0]["setting"]
    assert sum(setting["method"] == "fixed_com" for setting in selected) == 2
