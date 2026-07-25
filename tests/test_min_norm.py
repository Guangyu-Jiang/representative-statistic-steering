import torch

from repstat_steering.min_norm import (
    cap_relative_row_norm,
    solve_nonlinear_min_norm,
    solve_scalar_directional_min_norm,
    solve_scalar_gauss_newton_min_norm,
)


def test_nonlinear_solver_reaches_linear_vector_target() -> None:
    torch.manual_seed(3)
    source = torch.randn(4, 6)
    matrix = torch.randn(3, 6)
    target = source @ matrix.T + 0.4

    delta, diagnostics = solve_nonlinear_min_norm(
        source,
        lambda value: value @ matrix.T,
        target,
        ridge=1e-4,
        steps=250,
        learning_rate=0.05,
    )

    assert diagnostics.final_target_rmse < diagnostics.initial_target_rmse * 0.05
    assert delta.norm() > 0


def test_relative_cap_is_applied_per_row() -> None:
    source = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    delta = torch.full_like(source, 10.0)
    capped = cap_relative_row_norm(delta, source, 0.1)
    ratios = capped.norm(dim=-1) / source.norm(dim=-1)
    torch.testing.assert_close(ratios, torch.tensor([0.1, 0.1]))


def test_zero_steps_returns_zero_action() -> None:
    source = torch.randn(2, 4)
    target = torch.ones(2, 1)
    delta, diagnostics = solve_nonlinear_min_norm(
        source,
        lambda value: value[:, :1],
        target,
        ridge=0.1,
        steps=0,
        learning_rate=0.1,
    )
    torch.testing.assert_close(delta, torch.zeros_like(source))
    assert diagnostics.iterations == 0


def test_scalar_gauss_newton_recovers_linear_minimum_norm_solution() -> None:
    source = torch.zeros(2, 3)
    weights = torch.tensor([1.0, 2.0, -1.0])
    target = torch.tensor([[2.0], [-1.0]])

    delta, diagnostics = solve_scalar_gauss_newton_min_norm(
        source,
        lambda value: (value * weights).sum(dim=-1, keepdim=True),
        target,
        ridge=0.0,
        steps=1,
    )

    expected = target * weights / weights.square().sum()
    assert torch.allclose(delta, expected, atol=1e-6)
    assert diagnostics.final_target_rmse < 1e-6


def test_scalar_gauss_newton_regularizes_accumulated_action() -> None:
    source = torch.ones(1, 3)
    weights = torch.tensor([1.0, 2.0, -1.0])
    initial = (source * weights).sum(dim=-1, keepdim=True)
    target = initial + 2.0
    ridge = 1.0

    delta, _ = solve_scalar_gauss_newton_min_norm(
        source,
        lambda value: (value * weights).sum(dim=-1, keepdim=True),
        target,
        ridge=ridge,
        steps=5,
    )

    expected = 2.0 * weights / (weights.square().sum() + ridge)
    torch.testing.assert_close(delta[0], expected)


def test_scalar_directional_solver_stays_in_requested_subspace() -> None:
    source = torch.ones(2, 3)
    direction = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    target = torch.tensor([[3.0], [2.0]])

    delta, diagnostics = solve_scalar_directional_min_norm(
        source,
        direction,
        lambda value: value[:, :2].sum(dim=-1, keepdim=True),
        target,
        ridge=0.0,
        steps=1,
    )

    assert torch.allclose(delta[:, 2], torch.zeros(2))
    assert torch.allclose(delta[0, 0], delta[0, 1])
    assert torch.allclose(delta[1], torch.zeros(3))
    assert diagnostics.final_target_rmse < 1e-6


def test_directional_solver_regularizes_accumulated_magnitude() -> None:
    source = torch.ones(1, 2)
    direction = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[4.0]])
    ridge = 2.0
    unit = direction / direction.norm(dim=-1, keepdim=True)
    derivative = unit.sum()
    expected_magnitude = (target.item() - source.sum().item()) * derivative / (
        derivative.square() + ridge
    )

    delta, _ = solve_scalar_directional_min_norm(
        source,
        direction,
        lambda value: value.sum(dim=-1, keepdim=True),
        target,
        ridge=ridge,
        steps=5,
    )

    torch.testing.assert_close(delta, unit * expected_magnitude)


def test_directional_solver_backtracking_rejects_nonlinear_overshoot() -> None:
    source = torch.tensor([[0.1]])
    direction = torch.ones_like(source)
    target = torch.tensor([[1.0]])

    without_search, _ = solve_scalar_directional_min_norm(
        source,
        direction,
        lambda value: value.pow(3),
        target,
        ridge=0.0,
        steps=1,
        damping=1.0,
    )
    with_search, diagnostics = solve_scalar_directional_min_norm(
        source,
        direction,
        lambda value: value.pow(3),
        target,
        ridge=0.0,
        steps=1,
        damping=1.0,
        backtracking_steps=8,
    )

    without_error = ((source + without_search).pow(3) - target).abs()
    with_error = ((source + with_search).pow(3) - target).abs()
    assert with_error.item() < without_error.item()
    assert diagnostics.final_target_rmse < diagnostics.initial_target_rmse


def test_directional_solver_can_constrain_magnitude_to_truthful_ray() -> None:
    source = torch.zeros(1, 1)
    direction = torch.ones_like(source)
    target = torch.ones(1, 1)
    statistic = lambda value: -value

    unconstrained, _ = solve_scalar_directional_min_norm(
        source,
        direction,
        statistic,
        target,
        ridge=0.0,
        steps=1,
    )
    constrained, _ = solve_scalar_directional_min_norm(
        source,
        direction,
        statistic,
        target,
        ridge=0.0,
        steps=1,
        maximum_relative_norm=0.5,
        nonnegative_magnitude=True,
    )

    assert unconstrained.item() < 0
    assert constrained.item() == 0
