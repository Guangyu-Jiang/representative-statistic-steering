import pandas as pd
import pytest

from validation.summarize_local_judge_outputs import parse_filename, summarize_frame


def test_summary_distinguishes_paper_product_from_joint_rate(tmp_path):
    frame = pd.DataFrame(
        {
            "local_judge_raw": ["a", "b", "c", "d"],
            "local_truthful": ["YES", "YES", "NO", "NO"],
            "local_informative": ["NO", "NO", "YES", "YES"],
            "local_truthful_acc": [1, 1, 0, 0],
            "local_informative_acc": [0, 0, 1, 1],
            "local_truth_info_acc": [0, 0, 0, 0],
        }
    )

    summary = summarize_frame(frame)

    assert summary["truth_x_info"] == 0.25
    assert summary["joint_truth_info"] == 0.0
    assert parse_filename(
        tmp_path / "fold_0_test__candidate_answer__qwen_judged.csv"
    ) == ("fold_0_test", "candidate_answer", "qwen")


def test_summary_rejects_incomplete_checkpoint():
    frame = pd.DataFrame(
        {
            "local_judge_raw": [""],
            "local_truthful": ["UNKNOWN"],
            "local_informative": ["UNKNOWN"],
            "local_truthful_acc": [0],
            "local_informative_acc": [0],
            "local_truth_info_acc": [0],
        }
    )

    with pytest.raises(ValueError, match="incomplete"):
        summarize_frame(frame)
