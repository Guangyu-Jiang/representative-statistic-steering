import json

import pytest
import torch

from caa_exact_replication.local_judge import parse_score
from caa_exact_replication.prompts import JUDGE_SYSTEM_PROMPT, make_prompts
from caa_exact_replication.run import (
    SteeringHook,
    _matching_probability,
    _result_filename,
    _vector_directory,
)
from caa_exact_replication.summarize import _metadata


def test_hook_changes_only_positions_at_or_after_boundary():
    hook = SteeringHook()
    hook.configure(torch.tensor([1.0, -2.0]), from_position=2)
    hidden = torch.zeros(1, 4, 2)
    output = hook(
        None,
        (),
        {"position_ids": torch.tensor([[0, 1, 2, 3]])},
        (hidden,),
    )[0]
    assert torch.equal(output[0, :2], torch.zeros(2, 2))
    assert torch.equal(output[0, 2:], torch.tensor([[1.0, -2.0], [1.0, -2.0]]))


def test_hook_changes_cached_decode_token():
    hook = SteeringHook()
    hook.configure(torch.tensor([0.5, 1.0]), from_position=4)
    output = hook(
        None,
        (),
        {"cache_position": torch.tensor([9])},
        (torch.zeros(1, 1, 2),),
    )[0]
    assert torch.equal(output, torch.tensor([[[0.5, 1.0]]]))


def test_result_name_matches_official_convention():
    name = _result_filename(
        layer=13,
        multiplier=-0.5,
        behavior="sycophancy",
        eval_type="ab",
        model_size="7b",
        system_prompt="pos",
    )
    assert name == (
        "results_layer=13_multiplier=-0.5_behavior=sycophancy_type=ab_"
        "system_prompt=pos_use_base_model=False_model_size=7b.json"
    )


def test_result_name_round_trips_through_summary_parser():
    path = type("PathLike", (), {"name": (
        "results_layer=14_multiplier=-1.5_behavior=refusal_type=open_ended_"
        "use_base_model=False_model_size=13b.json"
    )})()
    assert _metadata(path) == {
        "layer": 14,
        "multiplier": -1.5,
        "behavior": "refusal",
        "eval_type": "open_ended",
        "system_prompt": "none",
        "model_size": "13b",
    }


def test_matching_probability_renormalizes_over_a_and_b():
    row = {
        "a_prob": 0.2,
        "b_prob": 0.3,
        "answer_matching_behavior": "(B)",
    }
    assert _matching_probability(row) == pytest.approx(0.6)


def test_matching_probability_mirrors_original_non_ab_label_handling():
    row = {
        "a_prob": 0.2,
        "b_prob": 0.3,
        "answer_matching_behavior": "(E)",
    }
    assert _matching_probability(row) == 0.0


def test_paper_vector_mapping_matches_published_artifact_history():
    assert _vector_directory("sycophancy", "paper") == "normalized_vectors"
    assert _vector_directory("survival-instinct", "paper") == "normalized_vectors"
    assert _vector_directory("refusal", "paper") == "vectors"
    assert _vector_directory("sycophancy", "raw") == "vectors"
    assert _vector_directory("refusal", "normalized") == "normalized_vectors"


def test_local_prompt_preserves_original_scoring_format():
    system, user = make_prompts("Question?", "Answer.", "sycophancy")
    assert system == JUDGE_SYSTEM_PROMPT
    assert user.endswith("\n\nQuestion:\nQuestion?\n\nAnswer:\nAnswer.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("7", 7.0), ("7.5", 7.5), ("Score: 10", 10.0), ("0\n", 0.0)],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


def test_parse_score_rejects_non_numeric_output():
    with pytest.raises(ValueError):
        parse_score("high")
