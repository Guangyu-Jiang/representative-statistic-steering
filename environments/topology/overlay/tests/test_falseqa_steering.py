import numpy as np
import pandas as pd

from aen_replication.eval.falseqa_steering import (
    build_paired_local_directions,
    falseqa_reference_variants,
    nli_gated_falseqa_label,
    parse_falseqa_judge_label,
    response_quality,
    select_layer_with_train_validation,
)


def _paired_metadata(pair_count: int) -> pd.DataFrame:
    rows = []
    for pair_index in range(pair_count):
        pair_id = f"p{pair_index:03d}"
        for label, variant in ((1, "false"), (0, "corrected")):
            rows.append(
                {
                    "example_id": f"{pair_id}__{variant}",
                    "pair_id": pair_id,
                    "question": variant,
                    "source_split": "train",
                    "split_random80": "train",
                    "split_official": "train",
                    "label_false_premise": label,
                }
            )
    return pd.DataFrame(rows)


def test_train_validation_layer_selection_uses_paired_signal() -> None:
    metadata = _paired_metadata(80)
    rng = np.random.default_rng(3)
    tensor = rng.normal(scale=0.1, size=(len(metadata), 2, 3)).astype(np.float32)
    labels = metadata["label_false_premise"].to_numpy()
    tensor[:, 1, 0] += np.where(labels == 1, 3.0, -3.0)
    layer, metrics = select_layer_with_train_validation(
        metadata,
        tensor,
        [0, 1],
        split_column="split_random80",
        seed=42,
    )
    assert layer == 1
    assert float(metrics.iloc[0]["validation_auroc"]) > 0.99


def test_local_directions_average_selected_pair_contrasts() -> None:
    train_topology = np.asarray([[0.0], [1.0], [10.0]], dtype=np.float32)
    query_topology = np.asarray([[0.2]], dtype=np.float32)
    differences = np.asarray([[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]], dtype=np.float32)
    directions, indices, _distances = build_paired_local_directions(
        train_topology,
        query_topology,
        differences,
        neighbor_k=2,
    )
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_allclose(directions[0], [2.0, 1.0])


def test_response_quality_rejects_empty_and_degenerate_text() -> None:
    assert not response_quality("")["response_valid"]
    assert not response_quality("word word word word")["response_valid"]
    assert response_quality("The premise is false because cats do not have feathers.")["response_valid"]


def test_local_judge_parser_accepts_tagged_and_untagged_labels() -> None:
    assert parse_falseqa_judge_label("<label>GROUNDED_REBUTTAL</label>") == "GROUNDED_REBUTTAL"
    assert parse_falseqa_judge_label("GROUNDED_REBUTTALLABEL") == "GROUNDED_REBUTTAL"
    assert parse_falseqa_judge_label("Answer: PREMISE_ACCEPTANCE") == "PREMISE_ACCEPTANCE"
    assert parse_falseqa_judge_label("No usable classification") == "NEITHER"


def test_falseqa_reference_variants_parse_stringified_lists() -> None:
    assert falseqa_reference_variants("['first correction', 'second correction']") == [
        "first correction",
        "second correction",
    ]
    assert falseqa_reference_variants("one correction") == ["one correction"]


def test_nli_gate_requires_entailment_for_grounded_rebuttal() -> None:
    assert (
        nli_gated_falseqa_label("PREMISE_ACCEPTANCE", 0.95, threshold=0.8)
        == "GROUNDED_REBUTTAL"
    )
    assert (
        nli_gated_falseqa_label("GROUNDED_REBUTTAL", 0.2, threshold=0.8)
        == "PREMISE_ACCEPTANCE"
    )
    assert (
        nli_gated_falseqa_label("GENERIC_REJECTION", 0.2, threshold=0.8)
        == "GENERIC_REJECTION"
    )
