import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from repstat_steering.redeep_control import ReDeEPStatisticConfig
from repstat_steering.redeep_generation import (
    ReDeEPMechanismController,
    finite_difference_jacobian,
    fixed_aarf_control,
    load_dolly_examples,
    split_dolly_examples,
    split_dolly_steering_evaluation,
)


def tiny_config() -> ReDeEPStatisticConfig:
    return ReDeEPStatisticConfig(
        copy_heads=((2, 0), (2, 1), (2, 2), (1, 3)),
        knowledge_layers=(2, 0, 1),
        external_min=-1.0,
        external_max=1.0,
        parameter_min=0.0,
        parameter_max=1000.0,
        detector_min=-1.0,
        detector_max=1.0,
    )


def tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(config).eval()


def test_controller_captures_finite_statistics_and_changes_hidden_state() -> None:
    torch.manual_seed(1)
    model = tiny_model()
    controller = ReDeEPMechanismController(model, tiny_config())
    input_ids = torch.randint(0, 100, (1, 12))
    controller.reset_capture()
    controller.set_control(None)
    output = model(input_ids, use_cache=False)
    baseline_hidden = controller.final_hidden_state.detach().clone()
    controller.set_prefix_hidden_state(baseline_hidden, 12)
    statistic, record = controller.statistics(baseline_hidden)
    assert statistic.shape == (2,)
    assert torch.isfinite(statistic).all()
    assert record.parameter_score >= -1e-3

    controller.reset_capture()
    controller.set_control(torch.full((7,), 0.2))
    changed = model(input_ids, use_cache=False)
    assert not torch.equal(output.logits, changed.logits)
    controller.close()


def test_fixed_aarf_control_matches_published_scales() -> None:
    control = fixed_aarf_control(tiny_config())
    torch.testing.assert_close(control[:4], torch.full((4,), 0.2))
    torch.testing.assert_close(control[4:], torch.full((3,), -0.2))


def test_finite_difference_jacobian_has_expected_shape() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])

    def evaluate(control):
        statistic = matrix @ control
        return None, torch.empty(0), statistic, None

    base = torch.zeros(3)
    jacobian = finite_difference_jacobian(
        evaluate, base, matrix @ base, epsilon=0.01
    )
    torch.testing.assert_close(jacobian, matrix, atol=1e-5, rtol=1e-5)


def test_dolly_loader_and_stratified_split(tmp_path) -> None:
    source_path = tmp_path / "source.jsonl"
    response_path = tmp_path / "response.jsonl"
    with source_path.open("w") as source, response_path.open("w") as response:
        for index in range(10):
            source.write(
                json.dumps(
                    {
                        "source_id": index,
                        "source_info": {
                            "question": f"q{index}",
                            "passages": f"p{index}",
                        },
                        "human_response": f"a{index}",
                        "prompt": f"prompt{index}",
                    }
                )
                + "\n"
            )
            response.write(
                json.dumps(
                    {
                        "source_id": index,
                        "model": "llama-2-7b-chat",
                        "split": "test",
                        "labels": ["Yes"] if index % 2 else [],
                    }
                )
                + "\n"
            )
    examples = load_dolly_examples(source_path, response_path)
    development, heldout = split_dolly_examples(examples, test_size=0.2, seed=3)
    assert len(development) == 8
    assert len(heldout) == 2
    assert {example.hallucination_label for example in heldout} == {0, 1}
    tuning, evaluation = split_dolly_steering_evaluation(
        examples, tuning_size=2, test_size=0.2, seed=3
    )
    assert len(tuning) == 2
    assert len(evaluation) == 8
    assert not ({example.source_id for example in tuning} & {
        example.source_id for example in evaluation
    })
