import pandas as pd

from validation.select_bounded_generation_setting import select_bounded
from validation.validate_causal_head_perturbation import Setting


def test_select_bounded_prefers_score_then_lower_cap() -> None:
    settings = [
        Setting(
            method="bounded_targeted_probe_iti",
            num_heads=48,
            alpha=12,
            target_quantile=0.75,
            ridge_ratio=0,
            relative_cap=2,
            coefficient_cap=cap,
        )
        for cap in (6, 8, 10)
    ]
    summary = pd.DataFrame(
        {
            "input": ["COMBINED"] * 3,
            "answer_column": [f"{setting.tag}_answer" for setting in settings],
            "truth_x_info": [0.50, 0.55, 0.55],
            "joint_truth_info": [0.49, 0.52, 0.52],
        }
    )

    selected, ranking = select_bounded(summary, settings)

    assert selected.coefficient_cap == 8
    assert list(ranking["coefficient_cap"]) == [8, 10, 6]
