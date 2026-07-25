from __future__ import annotations

import pandas as pd
import pytest

from validation.merge_causal_head_result_shards import merge_frames


def test_merge_frames_sorts_and_checks_row_count() -> None:
    first = pd.DataFrame({"dataset_index": [2, 0], "value": [20, 0]})
    second = pd.DataFrame({"dataset_index": [3, 1], "value": [30, 10]})

    merged = merge_frames([first, second], expected_rows=4)

    assert merged["dataset_index"].tolist() == [0, 1, 2, 3]


def test_merge_frames_rejects_duplicate_indices() -> None:
    first = pd.DataFrame({"dataset_index": [0], "value": [0]})
    second = pd.DataFrame({"dataset_index": [0], "value": [1]})

    with pytest.raises(ValueError, match="duplicate"):
        merge_frames([first, second], expected_rows=None)
