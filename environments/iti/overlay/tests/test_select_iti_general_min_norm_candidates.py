from __future__ import annotations

import pandas as pd

from validation.build_iti_general_min_norm_sweep import build_settings, setting_tag
from validation.select_iti_general_min_norm_candidates import select_candidates


def test_general_candidate_selection_unions_metrics_and_controls() -> None:
    settings = build_settings(
        [8.0, 16.0],
        [0.75],
        [1.5, 2.0],
        num_heads=48,
        ridge_ratio=0.0,
        include_legacy_best=False,
        fixed_alphas=[8.0, 15.0],
    )
    general = [setting for setting in settings if setting["method"] != "fixed_com"]
    rows = []
    for index, setting in enumerate(general):
        rows.append(
            {
                "setting": setting_tag(setting),
                "n": 48,
                "mc1": 0.3 + index * 0.01,
                "mc2": 0.6 - index * 0.01,
                "relative_action_norm": 1.0 + index * 0.1,
            }
        )

    ranking, selected = select_candidates(
        pd.DataFrame(rows),
        settings,
        top_mc1=1,
        top_mc2=1,
        top_mean=1,
        top_efficiency=1,
    )

    selected_tags = {setting_tag(setting) for setting in selected}
    assert setting_tag(general[0]) in selected_tags
    assert setting_tag(general[-1]) in selected_tags
    assert sum(setting["method"] == "fixed_com" for setting in selected) == 2
    assert ranking["selected"].sum() >= 2


def test_general_grid_supports_method_and_ridge_subsets() -> None:
    settings = build_settings(
        [20.0, 24.0],
        [0.75],
        [3.0],
        num_heads=48,
        ridge_ratio=0.0,
        ridge_ratios=[0.0, 0.25, 1.0],
        methods=["aggregate_com"],
        include_legacy_best=False,
        fixed_alphas=[8.0, 15.0],
    )

    general = [setting for setting in settings if setting["method"] != "fixed_com"]
    assert len(general) == 6
    assert {setting["method"] for setting in general} == {"aggregate_com"}
    assert {setting["ridge_ratio"] for setting in general} == {0.0, 0.25, 1.0}
    assert len({setting_tag(setting) for setting in settings}) == len(settings)


def test_efficiency_selection_can_require_performance_and_reference() -> None:
    settings = build_settings(
        [20.0],
        [0.5, 0.75],
        [0.5, 3.0],
        num_heads=48,
        ridge_ratio=0.1,
        methods=["aggregate_com"],
        include_legacy_best=False,
        fixed_alphas=[8.0, 15.0],
    )
    general = [setting for setting in settings if setting["method"] != "fixed_com"]
    required_tag = setting_tag(general[-1])
    rows = [
        {
            "setting": setting_tag(setting),
            "n": 48,
            "mc1": 0.30 if index == 0 else 0.50 - index * 0.01,
            "mc2": 0.50 if index == 0 else 0.65 - index * 0.01,
            "relative_action_norm": 0.1 if index == 0 else 1.0 + index * 0.1,
        }
        for index, setting in enumerate(general)
    ]

    ranking, selected = select_candidates(
        pd.DataFrame(rows),
        settings,
        top_mc1=0,
        top_mc2=0,
        top_mean=0,
        top_efficiency=1,
        efficiency_max_mean_gap=0.02,
        include_setting_tags=[required_tag],
    )

    selected_tags = {setting_tag(setting) for setting in selected}
    assert setting_tag(general[0]) not in selected_tags
    assert required_tag in selected_tags
    reason = ranking.set_index("setting").loc[required_tag, "selection_reason"]
    assert "required_reference" in reason
