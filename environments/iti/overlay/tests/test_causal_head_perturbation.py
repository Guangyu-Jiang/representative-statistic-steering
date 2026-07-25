from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT))

from validate_causal_head_perturbation import (
    AttentionHeadController,
    Setting,
    StatisticBank,
    build_aggregate_actions,
    load_statistics,
    save_statistics,
)


def run_hook(controller: AttentionHeadController, state: torch.Tensor) -> torch.Tensor:
    result = controller(None, (state,))
    assert result is not None
    return result[0]


def test_adaptive_closed_form_reaches_linear_target() -> None:
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.adaptive(
        heads=[0],
        directions=np.array([[1.0, 0.0]], dtype=np.float32),
        means=np.array([0.0], dtype=np.float32),
        stds=np.array([1.0], dtype=np.float32),
        targets=np.array([2.0], dtype=np.float32),
        alpha=1.0,
        ridge_ratio=0.0,
        relative_cap=None,
        start=0,
        stop=1,
    )
    source = torch.tensor([[[1.0, 0.0, 3.0, 4.0]]])
    output = run_hook(controller, source)
    assert torch.allclose(output, torch.tensor([[[2.0, 0.0, 3.0, 4.0]]]))
    assert torch.isclose(controller.last_post_error, torch.tensor(0.0))


def test_relative_cap_limits_each_selected_head() -> None:
    controller = AttentionHeadController(head_dim=2, hidden_size=2)
    controller.adaptive(
        heads=[0],
        directions=np.array([[1.0, 0.0]], dtype=np.float32),
        means=np.array([0.0], dtype=np.float32),
        stds=np.array([1.0], dtype=np.float32),
        targets=np.array([10.0], dtype=np.float32),
        alpha=1.0,
        ridge_ratio=0.0,
        relative_cap=0.1,
        start=0,
        stop=1,
    )
    source = torch.tensor([[[1.0, 0.0]]])
    output = run_hook(controller, source)
    assert torch.allclose(output, torch.tensor([[[1.1, 0.0]]]), atol=1e-6)


def test_fixed_action_only_changes_requested_causal_positions() -> None:
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.fixed(torch.tensor([1.0, 2.0, 0.0, 0.0]), [0], start=1, stop=3)
    source = torch.zeros((1, 4, 4))
    output = run_hook(controller, source)
    assert torch.equal(output[0, 0], source[0, 0])
    assert torch.equal(output[0, 3], source[0, 3])
    assert torch.equal(output[0, 1], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.equal(output[0, 2], torch.tensor([1.0, 2.0, 0.0, 0.0]))


def test_negative_start_selects_last_decode_position() -> None:
    controller = AttentionHeadController(head_dim=2, hidden_size=2)
    controller.fixed(torch.tensor([1.0, 2.0]), [0], start=-1, stop=10**9)
    source = torch.zeros((1, 4, 2))
    output = run_hook(controller, source)
    assert torch.equal(output[0, :3], source[0, :3])
    assert torch.equal(output[0, 3], torch.tensor([1.0, 2.0]))


def simple_bank() -> StatisticBank:
    return StatisticBank(
        top_heads=[(0, 0)],
        com_directions=np.array([[1.0, 0.0]], dtype=np.float32),
        fixed_scales=np.array([1.0], dtype=np.float32),
        com_means=np.array([0.0], dtype=np.float32),
        com_stds=np.array([1.0], dtype=np.float32),
        com_truthful=np.array([[2.0]], dtype=np.float32),
        probe_directions=np.array([[1.0, 0.0]], dtype=np.float32),
        probe_intercepts=np.array([0.0], dtype=np.float32),
        probe_means=np.array([0.0], dtype=np.float32),
        probe_stds=np.array([1.0], dtype=np.float32),
        probe_truthful=np.array([[2.0]], dtype=np.float32),
        probe_group_direction=np.array([1.0], dtype=np.float32),
        probe_group_positive_projection=np.array([2.0], dtype=np.float32),
        probe_group_negative_projection=np.array([-2.0], dtype=np.float32),
        validation_accuracies=np.array([1.0], dtype=np.float32),
        probe_raw_group_direction=np.array([1.0], dtype=np.float32),
        probe_raw_group_positive_projection=np.array([3.0], dtype=np.float32),
        probe_raw_group_negative_projection=np.array([-3.0], dtype=np.float32),
    )


def test_aggregate_action_reaches_target() -> None:
    controller = AttentionHeadController(head_dim=2, hidden_size=2)
    controller.last_collected = torch.tensor([[[[1.0, 0.0]]]])
    actions, metrics = build_aggregate_actions(
        controllers={0: controller},
        bank=simple_bank(),
        setting=Setting(
            method="aggregate_com",
            num_heads=1,
            alpha=1.0,
            target_quantile=0.5,
            ridge_ratio=0.0,
            relative_cap=None,
        ),
        hidden_size=2,
        head_dim=2,
    )
    assert torch.allclose(actions[0], torch.tensor([[1.0, 0.0]]))
    assert metrics["post_target_error"] < 1e-7
    assert abs(metrics["active_signed_target_error"]) < 1e-7
    assert metrics["active_absolute_target_error"] < 1e-7
    assert metrics["active_target_overshoot"] < 1e-7
    assert metrics["clip_rate"] == 0.0


def test_probe_target_can_use_original_iti_basis() -> None:
    bank = simple_bank()
    bank.probe_directions = np.array([[1.0, 1.0]], dtype=np.float32)
    controller = AttentionHeadController(head_dim=2, hidden_size=2)
    controller.last_collected = torch.tensor([[[[1.0, 0.0]]]])
    actions, metrics = build_aggregate_actions(
        controllers={0: controller},
        bank=bank,
        setting=Setting(
            method="targeted_probe_iti",
            num_heads=1,
            alpha=1.0,
            target_quantile=0.5,
            ridge_ratio=0.0,
            relative_cap=None,
        ),
        hidden_size=2,
        head_dim=2,
    )
    assert torch.allclose(actions[0], torch.tensor([[1.0, 0.0]]))
    assert metrics["post_target_error"] < 1e-7
    assert abs(metrics["active_signed_target_error"]) < 1e-7
    assert metrics["active_absolute_target_error"] < 1e-7
    assert metrics["clip_rate"] == 0.0


def test_probe_target_coefficient_cap_limits_original_iti_scale() -> None:
    bank = simple_bank()
    controller = AttentionHeadController(head_dim=2, hidden_size=2)
    controller.last_collected = torch.tensor([[[[0.0, 0.0]]]])
    setting = Setting(
        method="bounded_targeted_probe_iti",
        num_heads=1,
        alpha=4.0,
        target_quantile=0.5,
        ridge_ratio=0.0,
        relative_cap=None,
        coefficient_cap=1.5,
    )
    actions, metrics = build_aggregate_actions(
        controllers={0: controller},
        bank=bank,
        setting=setting,
        hidden_size=2,
        head_dim=2,
    )
    assert torch.allclose(actions[0], torch.tensor([[1.5, 0.0]]))
    assert metrics["post_target_error"] == 0.5
    assert metrics["active_signed_target_error"] == -0.5
    assert metrics["active_absolute_target_error"] == 0.5
    assert metrics["clip_rate"] == 1.0
    assert "b1p5" in setting.tag


def test_headwise_probe_target_corrects_each_original_iti_head() -> None:
    bank = StatisticBank(
        top_heads=[(0, 0), (0, 1)],
        com_directions=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        fixed_scales=np.ones(2, dtype=np.float32),
        com_means=np.zeros(2, dtype=np.float32),
        com_stds=np.ones(2, dtype=np.float32),
        com_truthful=np.array([[2.0, 4.0]], dtype=np.float32),
        probe_directions=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        probe_intercepts=np.zeros(2, dtype=np.float32),
        probe_means=np.zeros(2, dtype=np.float32),
        probe_stds=np.ones(2, dtype=np.float32),
        probe_truthful=np.array([[2.0, 4.0]], dtype=np.float32),
        probe_group_direction=np.ones(2, dtype=np.float32),
        probe_group_positive_projection=np.array([3.0], dtype=np.float32),
        probe_group_negative_projection=np.array([-3.0], dtype=np.float32),
        validation_accuracies=np.ones(2, dtype=np.float32),
    )
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.last_collected = torch.tensor([[[[1.0, 0.0], [3.0, 0.0]]]])

    actions, metrics = build_aggregate_actions(
        controllers={0: controller},
        bank=bank,
        setting=Setting(
            method="headwise_probe_iti",
            num_heads=2,
            alpha=1.0,
            target_quantile=0.5,
            ridge_ratio=0.0,
            relative_cap=None,
        ),
        hidden_size=4,
        head_dim=2,
    )

    assert torch.allclose(actions[0], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert metrics["post_target_error"] < 1e-7


def test_headwise_probe_min_norm_corrects_each_head_along_probe_jacobian() -> None:
    bank = StatisticBank(
        top_heads=[(0, 0), (0, 1)],
        com_directions=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        fixed_scales=np.ones(2, dtype=np.float32),
        com_means=np.zeros(2, dtype=np.float32),
        com_stds=np.ones(2, dtype=np.float32),
        com_truthful=np.array([[2.0, 4.0]], dtype=np.float32),
        probe_directions=np.array([[1.0, 1.0], [0.0, 2.0]], dtype=np.float32),
        probe_intercepts=np.zeros(2, dtype=np.float32),
        probe_means=np.zeros(2, dtype=np.float32),
        probe_stds=np.ones(2, dtype=np.float32),
        probe_truthful=np.array([[2.0, 4.0]], dtype=np.float32),
        probe_group_direction=np.ones(2, dtype=np.float32),
        probe_group_positive_projection=np.array([3.0], dtype=np.float32),
        probe_group_negative_projection=np.array([-3.0], dtype=np.float32),
        validation_accuracies=np.ones(2, dtype=np.float32),
    )
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.last_collected = torch.zeros((1, 1, 2, 2))

    actions, metrics = build_aggregate_actions(
        controllers={0: controller},
        bank=bank,
        setting=Setting(
            method="headwise_probe_min_norm",
            num_heads=2,
            alpha=1.0,
            target_quantile=0.5,
            ridge_ratio=0.0,
            relative_cap=None,
        ),
        hidden_size=4,
        head_dim=2,
    )

    assert torch.allclose(actions[0], torch.tensor([[1.0, 1.0, 0.0, 2.0]]))
    assert metrics["post_target_error"] < 1e-7
    assert metrics["active_absolute_target_error"] < 1e-7


def test_group_direction_target_moves_joint_probe_statistic() -> None:
    bank = StatisticBank(
        top_heads=[(0, 0), (0, 1)],
        com_directions=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        fixed_scales=np.ones(2, dtype=np.float32),
        com_means=np.zeros(2, dtype=np.float32),
        com_stds=np.ones(2, dtype=np.float32),
        com_truthful=np.array([[1.2, 1.6]], dtype=np.float32),
        probe_directions=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        probe_intercepts=np.zeros(2, dtype=np.float32),
        probe_means=np.zeros(2, dtype=np.float32),
        probe_stds=np.ones(2, dtype=np.float32),
        probe_truthful=np.array([[1.2, 1.6]], dtype=np.float32),
        probe_group_direction=np.array([3.0, 4.0], dtype=np.float32),
        probe_group_positive_projection=np.array([2.0], dtype=np.float32),
        probe_group_negative_projection=np.array([-2.0], dtype=np.float32),
        validation_accuracies=np.ones(2, dtype=np.float32),
    )
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.last_collected = torch.zeros((1, 1, 2, 2))

    for method in ("group_direction_probe_iti", "group_direction_probe_min_norm"):
        actions, metrics = build_aggregate_actions(
            controllers={0: controller},
            bank=bank,
            setting=Setting(
                method=method,
                num_heads=2,
                alpha=1.0,
                target_quantile=0.5,
                ridge_ratio=0.0,
                relative_cap=None,
            ),
            hidden_size=4,
            head_dim=2,
        )

        assert torch.allclose(
            actions[0], torch.tensor([[1.2, 0.0, 1.6, 0.0]]), atol=1e-6
        )
        assert metrics["post_target_error"] < 1e-6
        assert metrics["active_absolute_target_error"] < 1e-6


def test_raw_group_direction_uses_unstandardized_probe_jacobians() -> None:
    bank = StatisticBank(
        top_heads=[(0, 0), (0, 1)],
        com_directions=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        fixed_scales=np.ones(2, dtype=np.float32),
        com_means=np.zeros(2, dtype=np.float32),
        com_stds=np.ones(2, dtype=np.float32),
        com_truthful=np.array([[1.0, 1.0]], dtype=np.float32),
        probe_directions=np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        probe_intercepts=np.zeros(2, dtype=np.float32),
        probe_means=np.array([100.0, -100.0], dtype=np.float32),
        probe_stds=np.array([10.0, 0.1], dtype=np.float32),
        probe_truthful=np.array([[1.0, 1.0]], dtype=np.float32),
        probe_group_direction=np.array([1.0, 1.0], dtype=np.float32),
        probe_group_positive_projection=np.array([1.0], dtype=np.float32),
        probe_group_negative_projection=np.array([-1.0], dtype=np.float32),
        validation_accuracies=np.ones(2, dtype=np.float32),
        probe_raw_group_direction=np.array([3.0, 4.0], dtype=np.float32),
        probe_raw_group_positive_projection=np.array([2.0], dtype=np.float32),
        probe_raw_group_negative_projection=np.array([-2.0], dtype=np.float32),
    )
    controller = AttentionHeadController(head_dim=2, hidden_size=4)
    controller.last_collected = torch.zeros((1, 1, 2, 2))

    for method in ("group_direction_probe_iti", "group_direction_probe_min_norm"):
        actions, metrics = build_aggregate_actions(
            controllers={0: controller},
            bank=bank,
            setting=Setting(
                method=method,
                num_heads=2,
                alpha=1.0,
                target_quantile=0.5,
                ridge_ratio=0.0,
                probe_score_normalization="raw",
            ),
            hidden_size=4,
            head_dim=2,
        )

        assert torch.allclose(
            actions[0], torch.tensor([[0.6, 0.0, 0.0, 1.6]]), atol=1e-6
        )
        assert metrics["post_target_error"] < 1e-6
        assert metrics["active_absolute_target_error"] < 1e-6


def test_statistics_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "statistics.npz"
    save_statistics(path, simple_bank())
    loaded = load_statistics(path)
    assert loaded.top_heads == [(0, 0)]
    assert np.array_equal(loaded.com_truthful, np.array([[2.0]], dtype=np.float32))
    assert np.array_equal(loaded.probe_group_direction, np.array([1.0], dtype=np.float32))
    assert np.array_equal(
        loaded.probe_raw_group_direction, np.array([1.0], dtype=np.float32)
    )
