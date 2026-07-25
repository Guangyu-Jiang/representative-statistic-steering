import pandas as pd
import pytest

from validation.audit_generation_outputs import audit_frames


def test_audit_reports_diversity_changes_and_diagnostics():
    frame = pd.DataFrame(
        {
            "dataset_index": [0, 1, 2],
            "baseline_answer": ["one", "two", "three"],
            "candidate_answer": ["one", "changed", "changed"],
            "candidate_relative_action_norm": [0.5, 1.0, 1.5],
            "candidate_intervention_rate": [0.0, 0.5, 1.0],
        }
    )

    result = audit_frames([frame], ["baseline_answer", "candidate_answer"], "baseline_answer")
    candidate = result.set_index("answer_column").loc["candidate_answer"]

    assert candidate["n"] == 3
    assert candidate["unique_answers"] == 2
    assert candidate["max_duplicate_count"] == 2
    assert candidate["changed_rate"] == pytest.approx(2 / 3)
    assert candidate["relative_action_norm"] == pytest.approx(1.0)
    assert candidate["intervention_rate"] == pytest.approx(0.5)


def test_audit_rejects_duplicate_dataset_indices():
    frame = pd.DataFrame(
        {
            "dataset_index": [0],
            "baseline_answer": ["one"],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        audit_frames([frame, frame], ["baseline_answer"], "baseline_answer")
