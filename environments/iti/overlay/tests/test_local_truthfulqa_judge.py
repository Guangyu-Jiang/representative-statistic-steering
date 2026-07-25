import pandas as pd
import pytest

from validation.local_truthfulqa_judge import (
    enforce_empty_answer_failures,
    extract_label,
    finalize_labels,
    load_judge_checkpoint,
    prompt_value,
)


def test_extract_label_is_case_and_whitespace_tolerant():
    text = "<truthful> yes </truthful>\n<informative>NO</informative>"

    assert extract_label(text, "truthful") == "YES"
    assert extract_label(text, "informative") == "NO"
    assert extract_label(text, "missing") == "UNKNOWN"


def test_prompt_value_preserves_empty_csv_answers():
    assert prompt_value(float("nan")) == ""
    assert prompt_value(None) == ""
    assert prompt_value("answer") == "answer"


def test_empty_answers_are_deterministic_failures():
    judged = pd.DataFrame(
        {
            "candidate_answer": ["substantive", "", None, "   "],
            "local_judge_raw": [
                "<truthful>YES</truthful>\n<informative>YES</informative>"
            ] * 4,
        }
    )

    empty = enforce_empty_answer_failures(judged, "candidate_answer")
    finalize_labels(judged)

    assert empty.tolist() == [False, True, True, True]
    assert judged.loc[0, "local_truthful"] == "YES"
    assert judged.loc[empty, "local_truthful_acc"].eq(0).all()
    assert judged.loc[empty, "local_informative_acc"].eq(0).all()
    assert judged.loc[empty, "local_truth_info_acc"].eq(0).all()


def test_load_judge_checkpoint_resumes_only_matching_rows(tmp_path):
    frame = pd.DataFrame(
        {
            "Question": ["Question one?", "Question two?"],
            "Correct Answers": ["Correct one", "Correct two"],
            "Incorrect Answers": ["Wrong one", "Wrong two"],
            "candidate_answer": ["First", "Second"],
        }
    )
    output_path = tmp_path / "checkpoint.csv"
    prior = frame.copy()
    prior["local_judge_raw"] = [
        "<truthful>YES</truthful>\n<informative>YES</informative>",
        "",
    ]
    prior.to_csv(output_path, index=False)

    resumed = load_judge_checkpoint(frame, output_path, "candidate_answer")

    assert resumed.loc[0, "local_judge_raw"].startswith("<truthful>YES")
    assert resumed.loc[1, "local_judge_raw"] == ""

    changed = frame.copy()
    changed.loc[1, "candidate_answer"] = "Changed"
    with pytest.raises(ValueError, match="candidate_answer"):
        load_judge_checkpoint(changed, output_path, "candidate_answer")
