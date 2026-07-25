import pandas as pd
import pytest

from validation.merge_generation_shards import enrich_generation_frame, merge_generation_frames


def _frame(indices, baseline, steered):
    return pd.DataFrame(
        {
            "dataset_index": indices,
            "baseline_answer": baseline,
            "steered_answer": steered,
        }
    )


def test_merge_generation_frames_drops_incomplete_preallocated_rows():
    head = _frame([10, 20, 30], ["a", "b", ""], ["x", "y", ""])
    tail = _frame([30, 40], ["c", "d"], ["z", "w"])

    merged = merge_generation_frames([head, tail])

    assert merged["dataset_index"].tolist() == [10, 20, 30, 40]


def test_merge_generation_frames_rejects_conflicting_overlap():
    first = _frame([10], ["a"], ["x"])
    second = _frame([10], ["a"], ["different"])

    with pytest.raises(ValueError, match="Conflicting answers"):
        merge_generation_frames([first, second])


def test_merge_generation_frames_can_retain_completed_empty_answers():
    frame = _frame([10, 20], ["a", "b"], ["", "x"])

    merged = merge_generation_frames([frame], allow_empty_answers=True)

    assert merged["dataset_index"].tolist() == [10, 20]
    assert merged["steered_answer"].fillna("").tolist() == ["", "x"]


def test_enrich_generation_frame_adds_aligned_method_columns():
    base = _frame([20, 10], ["b", "a"], ["y", "x"])
    source = pd.DataFrame(
        {
            "dataset_index": [10, 20],
            "baseline_answer": ["a", "b"],
            "fixed_answer": ["fixed-a", "fixed-b"],
        }
    )

    enriched = enrich_generation_frame(base, source)

    assert enriched["fixed_answer"].tolist() == ["fixed-b", "fixed-a"]
