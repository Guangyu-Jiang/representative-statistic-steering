from __future__ import annotations

import pandas as pd

from validation.build_iti_alpha_rho_sweep import build_settings, setting_tag
from validation.select_iti_alpha_rho_candidates import select_candidates


def test_select_candidates_unions_metrics_and_alpha_coverage() -> None:
    settings = build_settings(
        [1.0, 2.0],
        [0.5, 1.0],
        num_heads=48,
        target_quantile=0.75,
        ridge_ratio=0.0,
        coefficient_cap=10.0,
    )
    scores = [
        (0.40, 0.30),
        (0.35, 0.50),
        (0.60, 0.20),
        (0.50, 0.45),
    ]
    rows = []
    for setting, (product, joint) in zip(settings, scores, strict=True):
        rows.append(
            {
                "input": "COMBINED",
                "answer_column": f"{setting_tag(setting)}_answer",
                "truthful": 0.7,
                "informative": 0.8,
                "truth_x_info": product,
                "joint_truth_info": joint,
                "parse_rate": 1.0,
            }
        )

    ranking, selected = select_candidates(
        pd.DataFrame(rows),
        settings,
        top_product=1,
        top_joint=1,
        best_per_alpha=True,
        force_tags=[setting_tag(settings[0])],
        allow_subset=False,
    )

    selected_tags = {setting_tag(setting) for setting in selected}
    assert setting_tag(settings[0]) in selected_tags
    assert setting_tag(settings[1]) in selected_tags
    assert setting_tag(settings[2]) in selected_tags
    assert ranking["product_rank"].tolist() == [1, 2, 3, 4]
