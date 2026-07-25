import json

import torch

from repstat_steering.redeep_control import (
    LLAMA2_7B_DOLLY_CONFIG,
    load_official_chunk_features,
    minmax_normalize,
    official_redeep_divergence,
    project_to_redeep_target,
    redeep_detector_score,
    solve_linearized_statistic_inverse,
)


def test_official_divergence_is_zero_for_equal_logits() -> None:
    logits = torch.randn(3, 17)
    score = official_redeep_divergence(logits, logits)
    torch.testing.assert_close(score, torch.zeros_like(score), atol=2e-4, rtol=0)


def test_official_divergence_is_symmetric_and_positive() -> None:
    first = torch.tensor([[3.0, -1.0, 0.0]])
    second = torch.tensor([[-2.0, 2.0, 0.5]])
    forward = official_redeep_divergence(first, second)
    reverse = official_redeep_divergence(second, first)
    torch.testing.assert_close(forward, reverse)
    assert forward.item() > 0


def test_minmax_normalize_does_not_clip() -> None:
    values = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    normalized = minmax_normalize(values, 0.0, 1.0)
    torch.testing.assert_close(normalized, values)


def test_target_projection_reduces_detector_score_by_requested_shift() -> None:
    config = LLAMA2_7B_DOLLY_CONFIG
    statistic = torch.tensor([0.7, 0.2])
    target = project_to_redeep_target(statistic, config, score_shift=0.25)
    before = redeep_detector_score(statistic, config)
    after = redeep_detector_score(target, config)
    torch.testing.assert_close(after, before - 0.25, atol=1e-6, rtol=0)
    assert target[0] < statistic[0]
    assert target[1] > statistic[1]


def test_linearized_inverse_matches_regularized_solution() -> None:
    jacobian = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
    current_control = torch.zeros(3)
    current_statistic = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, -0.5])
    ridge = 0.2

    control, diagnostics = solve_linearized_statistic_inverse(
        jacobian,
        current_control,
        current_statistic,
        target,
        ridge=ridge,
    )

    expected = torch.linalg.solve(
        jacobian.T @ jacobian + ridge * torch.eye(3),
        jacobian.T @ target,
    )
    torch.testing.assert_close(control, expected, atol=2e-6, rtol=2e-6)
    assert diagnostics.predicted_error < diagnostics.initial_error


def test_official_chunk_loader_preserves_named_features(tmp_path) -> None:
    path = tmp_path / "features.json"
    payload = [
        {
            "source_id": "one",
            "scores": [
                {
                    "prompt_attention_score": {"(2, 3)": 0.4},
                    "parameter_knowledge_scores": {"layer_5": 1.2},
                    "hallucination_label": 1,
                }
            ],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    frame = load_official_chunk_features(path)
    assert frame.loc[0, "ecs::(2, 3)"] == 0.4
    assert frame.loc[0, "pks::layer_5"] == 1.2
    assert frame.loc[0, "hallucination_label"] == 1
