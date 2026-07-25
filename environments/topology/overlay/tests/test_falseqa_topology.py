from pathlib import Path

import numpy as np
import pandas as pd

from aen_replication.eval.falseqa_topology import (
    assert_group_disjoint,
    assign_evaluation_splits,
    h0_features_from_cloud,
    load_falseqa_pairs,
    paired_difference_dataset,
)


def _write_split(root: Path, name: str, false_questions: list[str], corrected_questions: list[str]) -> None:
    rows = [
        {"question": question, "answer": f"rebuttal {index}", "label": 1}
        for index, question in enumerate(false_questions)
    ]
    rows.extend(
        {"question": question, "answer": f"answer {index}", "label": 0}
        for index, question in enumerate(corrected_questions)
    )
    pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)


def test_falseqa_pair_recovery_and_grouped_splits(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        _write_split(
            tmp_path,
            split,
            [f"false {split} 0", f"false {split} 1"],
            [f"true {split} 0", f"true {split} 1"],
        )
    frame = load_falseqa_pairs(tmp_path)
    assert len(frame) == 12
    assert frame["pair_id"].nunique() == 6
    assert frame.groupby("pair_id")["label_false_premise"].sum().eq(1).all()

    split_frame = assign_evaluation_splits(frame, train_fraction=2 / 3, seed=7)
    assert_group_disjoint(split_frame, "split_random80")
    assert_group_disjoint(split_frame, "split_official")
    official_test = split_frame.loc[split_frame["split_official"].eq("test")]
    assert set(official_test["source_split"]) == {"test"}


def test_h0_features_match_mst_edge_lengths() -> None:
    cloud = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    features = h0_features_from_cloud(cloud)
    np.testing.assert_allclose(features["h0_lifetimes"], [2.0, 1.0])
    assert np.isclose(features["h0_mean_persistence"], 1.5)
    assert np.isclose(features["h0_top5_persistence_fraction"], 1.0)
    assert 0.0 < features["h0_persistence_entropy"] < 1.0


def test_paired_difference_orientation_is_consistent() -> None:
    metadata = pd.DataFrame(
        [
            {
                "example_id": "p0__false",
                "pair_id": "p0",
                "source_split": "train",
                "split_random80": "train",
                "split_official": "train",
                "label_false_premise": 1,
                "question": "false question",
            },
            {
                "example_id": "p0__corrected",
                "pair_id": "p0",
                "source_split": "train",
                "split_random80": "train",
                "split_official": "train",
                "label_false_premise": 0,
                "question": "corrected question",
            },
        ]
    )
    matrix = np.asarray([[4.0, 2.0], [1.0, 1.0]])
    paired, differences = paired_difference_dataset(metadata, matrix, seed=11)
    sign = 1.0 if int(paired.iloc[0]["label_false_first"]) == 1 else -1.0
    np.testing.assert_allclose(differences[0], sign * np.asarray([3.0, 1.0]))
