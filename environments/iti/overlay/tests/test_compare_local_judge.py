import pandas as pd
import pytest

from validation.compare_local_judge import (
    compare_frames,
    fold_balanced_paired_bootstrap,
    product_of_means_bootstrap,
)


def test_compare_frames_reports_paired_gain():
    frame = pd.DataFrame(
        {
            "local_truthful_acc_reference": [0, 1, 0, 1],
            "local_truthful_acc_candidate": [1, 1, 1, 1],
            "local_informative_acc_reference": [1, 1, 0, 0],
            "local_informative_acc_candidate": [1, 1, 1, 0],
            "local_truth_info_acc_reference": [0, 1, 0, 0],
            "local_truth_info_acc_candidate": [1, 1, 1, 0],
        }
    )

    result = compare_frames([frame], samples=1000, seed=1).set_index("metric")

    assert result.loc["truthful", "difference"] == 0.5
    assert result.loc["informative", "difference"] == 0.25
    assert result.loc["truth_x_info", "difference"] == 0.5
    assert result.loc["truth_x_info", "reference_mean"] == 0.25
    assert result.loc["truth_x_info", "candidate_mean"] == 0.75
    assert result.loc["joint_truth_info", "difference"] == 0.5


def test_paper_product_balances_folds_before_multiplying():
    columns = {
        "local_truthful_acc_reference": 0,
        "local_truthful_acc_candidate": 1,
        "local_informative_acc_reference": 0,
        "local_informative_acc_candidate": 1,
    }
    small_fold = pd.DataFrame([columns])
    large_fold = pd.DataFrame(
        [
            {
                "local_truthful_acc_reference": 1,
                "local_truthful_acc_candidate": 1,
                "local_informative_acc_reference": 1,
                "local_informative_acc_candidate": 1,
            }
        ]
        * 3
    )

    reference, candidate, difference, *_ = product_of_means_bootstrap(
        [small_fold, large_fold], samples=100, seed=1
    )

    assert reference == pytest.approx(0.25)
    assert candidate == pytest.approx(1.0)
    assert difference == pytest.approx(0.75)


def test_paired_bootstrap_balances_unequal_fold_sizes():
    difference, *_ = fold_balanced_paired_bootstrap(
        [pd.Series([0.0]).to_numpy(), pd.Series([1.0, 1.0, 1.0]).to_numpy()],
        samples=100,
        seed=1,
    )

    assert difference == pytest.approx(0.5)
