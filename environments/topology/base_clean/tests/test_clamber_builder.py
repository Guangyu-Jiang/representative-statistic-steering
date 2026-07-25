from pathlib import Path

import pandas as pd

from aen_replication.data.clamber import build_clamber_pairs


def test_build_clamber_pairs_parses_nested_json_and_splits(tmp_path: Path) -> None:
    records = [
        '{"question": "Q1", "context": "", "clarifying_question": "C1", "require_clarification": 1, "category": "LA", "subclass": "polysemy"}',
        '{"question": "Q2", "context": "", "clarifying_question": "", "require_clarification": 0, "category": "MC", "subclass": "none"}',
        '{"question": "Q3", "context": "", "clarifying_question": "C3", "require_clarification": 1, "category": "LA", "subclass": "polysemy"}',
        '{"question": "Q4", "context": "", "clarifying_question": "", "require_clarification": 0, "category": "MC", "subclass": "none"}',
    ]
    source_path = tmp_path / "clamber.jsonl"
    source_path.write_text("\n".join(f'"{line.replace(chr(34), chr(92) + chr(34))}"' for line in records), encoding="utf-8")

    df = build_clamber_pairs(source_path=source_path, seed=7, train_fraction=0.5)

    assert set(df["dataset"]) == {"clamber"}
    assert set(df["label_ambiguous"]) == {0, 1}
    assert set(df["subclass"]) == {"none", "polysemy"}
    assert set(df["split"]) == {"train", "test"}
    assert df["pair_id"].nunique() == len(df)
    assert all(df["example_id"] == df["pair_id"])
    split_counts = df.groupby(["label_ambiguous", "subclass", "split"]).size().to_dict()
    assert split_counts[(1, "polysemy", "train")] == 1
    assert split_counts[(1, "polysemy", "test")] == 1
    assert split_counts[(0, "none", "train")] == 1
    assert split_counts[(0, "none", "test")] == 1
