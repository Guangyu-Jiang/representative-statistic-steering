"""Generic differentiable minimum-norm perturbation solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


TensorStatistic = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PerturbationDiagnostics:
    initial_target_rmse: float
    final_target_rmse: float
    action_norm: float
    relative_action_norm: float
    iterations: int


def _row_norm(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).norm(dim=-1)


def cap_relative_row_norm(
    delta: torch.Tensor,
    source: torch.Tensor,
    maximum_relative_norm: float | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cap each row of ``delta`` relative to the corresponding source row."""

    if maximum_relative_norm is None:
        return delta
    if maximum_relative_norm <= 0:
        return torch.zeros_like(delta)
    delta_norm = _row_norm(delta)
    source_norm = _row_norm(source).clamp_min(eps)
    scale = (maximum_relative_norm * source_norm / delta_norm.clamp_min(eps)).clamp_max(1.0)
    return delta * scale.reshape((-1,) + (1,) * (delta.ndim - 1))


def solve_nonlinear_min_norm(
    source: torch.Tensor,
    statistic_fn: TensorStatistic,
    target: torch.Tensor,
    *,
    ridge: float,
    steps: int,
    learning_rate: float,
    maximum_relative_norm: float | None = None,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, PerturbationDiagnostics]:
    """Solve a regularized nonlinear statistic-targeting problem with Adam.

    The loss is the batch mean of the per-example squared statistic error plus
    ``ridge`` times the per-example squared perturbation norm. Both terms use
    sums over feature dimensions, matching the mathematical L2 objectives.
    """

    if source.ndim < 2:
        raise ValueError("source must include a batch dimension and feature dimensions")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    source_fp32 = source.detach().float()
    with torch.enable_grad():
        initial_stat = statistic_fn(source_fp32).float()
    target_fp32 = target.detach().to(device=source.device, dtype=torch.float32)
    if initial_stat.shape != target_fp32.shape:
        raise ValueError(
            f"statistic shape {tuple(initial_stat.shape)} does not match target "
            f"shape {tuple(target_fp32.shape)}"
        )

    with torch.enable_grad():
        delta = torch.zeros_like(source_fp32, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        completed_steps = 0

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            statistic = statistic_fn(source_fp32 + delta)
            statistic_error = (statistic.float() - target_fp32).reshape(
                source.shape[0], -1
            )
            action = delta.reshape(source.shape[0], -1)
            statistic_loss = statistic_error.square().sum(dim=-1).mean()
            action_loss = action.square().sum(dim=-1).mean()
            loss = statistic_loss + ridge * action_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(
                    cap_relative_row_norm(delta, source_fp32, maximum_relative_norm)
                )
            completed_steps = step + 1
            if float(statistic_loss.detach().sqrt()) <= tolerance:
                break

    with torch.no_grad():
        final_stat = statistic_fn(source_fp32 + delta).float()
        initial_rmse = (initial_stat.detach() - target_fp32).square().mean().sqrt()
        final_rmse = (final_stat - target_fp32).square().mean().sqrt()
        action_norm = _row_norm(delta).mean()
        relative_norm = (_row_norm(delta) / _row_norm(source_fp32).clamp_min(1e-12)).mean()

    diagnostics = PerturbationDiagnostics(
        initial_target_rmse=float(initial_rmse),
        final_target_rmse=float(final_rmse),
        action_norm=float(action_norm),
        relative_action_norm=float(relative_norm),
        iterations=completed_steps,
    )
    return delta.detach().to(dtype=source.dtype), diagnostics


def solve_scalar_gauss_newton_min_norm(
    source: torch.Tensor,
    statistic_fn: TensorStatistic,
    target: torch.Tensor,
    *,
    ridge: float,
    steps: int,
    damping: float = 1.0,
    maximum_relative_norm: float | None = None,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, PerturbationDiagnostics]:
    """Iteratively solve the local minimum-norm problem for a scalar statistic.

    Each source row must produce one independent scalar. At the current point,
    the regularized solution for the *accumulated* action is

    ``delta_new = (error + grad^T delta) grad / (||grad||^2 + ridge)``.

    Re-linearizing after every update handles a nonlinear statistic while
    retaining the closed-form minimum-norm update at each iteration.
    """

    if source.ndim < 2:
        raise ValueError("source must include a batch dimension and feature dimensions")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")

    source_fp32 = source.detach().float()
    target_fp32 = target.detach().to(device=source.device, dtype=torch.float32)
    if target_fp32.shape not in {(source.shape[0],), (source.shape[0], 1)}:
        raise ValueError("target must contain exactly one scalar per source row")
    target_flat = target_fp32.reshape(source.shape[0])
    delta = torch.zeros_like(source_fp32)

    with torch.enable_grad():
        initial = statistic_fn(source_fp32).float().reshape(source.shape[0])
    completed_steps = 0
    for step in range(steps):
        point = (source_fp32 + delta).detach().requires_grad_(True)
        with torch.enable_grad():
            statistic = statistic_fn(point).float().reshape(source.shape[0])
            gradient = torch.autograd.grad(statistic.sum(), point)[0]
        error = target_flat - statistic.detach()
        if float(error.square().mean().sqrt()) <= tolerance:
            break
        gradient_rows = gradient.detach().reshape(source.shape[0], -1)
        delta_rows = delta.reshape(source.shape[0], -1)
        denominator = gradient_rows.square().sum(dim=-1) + ridge
        accumulated_error = error + (gradient_rows * delta_rows).sum(dim=-1)
        coefficient = accumulated_error / denominator.clamp_min(1e-12)
        regularized_solution = gradient.detach() * coefficient.reshape(
            (-1,) + (1,) * (source.ndim - 1)
        )
        update = damping * (regularized_solution - delta)
        delta = cap_relative_row_norm(
            delta + update, source_fp32, maximum_relative_norm
        ).detach()
        completed_steps = step + 1

    with torch.no_grad():
        final = statistic_fn(source_fp32 + delta).float().reshape(source.shape[0])
        initial_rmse = (initial.detach() - target_flat).square().mean().sqrt()
        final_rmse = (final - target_flat).square().mean().sqrt()
        action_norm = _row_norm(delta).mean()
        relative_norm = (_row_norm(delta) / _row_norm(source_fp32).clamp_min(1e-12)).mean()

    diagnostics = PerturbationDiagnostics(
        initial_target_rmse=float(initial_rmse),
        final_target_rmse=float(final_rmse),
        action_norm=float(action_norm),
        relative_action_norm=float(relative_norm),
        iterations=completed_steps,
    )
    return delta.to(dtype=source.dtype), diagnostics


def solve_scalar_directional_min_norm(
    source: torch.Tensor,
    direction: torch.Tensor,
    statistic_fn: TensorStatistic,
    target: torch.Tensor,
    *,
    ridge: float,
    steps: int,
    damping: float = 1.0,
    maximum_relative_norm: float | None = None,
    backtracking_steps: int = 0,
    nonnegative_magnitude: bool = False,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, PerturbationDiagnostics]:
    """Solve a scalar target in a fixed per-row intervention subspace.

    The direction is normalized per row, so the optimized coefficient is the
    perturbation norm. This makes the ridge penalty and norm cap directly
    comparable to those of unrestricted hidden-state optimization.
    """

    if source.shape != direction.shape:
        raise ValueError("source and direction must have identical shapes")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if backtracking_steps < 0:
        raise ValueError("backtracking_steps must be non-negative")

    source_fp32 = source.detach().float()
    direction_fp32 = direction.detach().float()
    direction_norm = _row_norm(direction_fp32).clamp_min(1e-12)
    unit_direction = direction_fp32 / direction_norm.reshape(
        (-1,) + (1,) * (source.ndim - 1)
    )
    target_flat = target.detach().to(source.device, torch.float32).reshape(source.shape[0])
    magnitude = torch.zeros(source.shape[0], device=source.device, dtype=torch.float32)
    source_norm = _row_norm(source_fp32)

    with torch.enable_grad():
        initial = statistic_fn(source_fp32).float().reshape(source.shape[0])
    completed_steps = 0
    for step in range(steps):
        delta = unit_direction * magnitude.reshape(
            (-1,) + (1,) * (source.ndim - 1)
        )
        point = (source_fp32 + delta).detach().requires_grad_(True)
        with torch.enable_grad():
            statistic = statistic_fn(point).float().reshape(source.shape[0])
            gradient = torch.autograd.grad(statistic.sum(), point)[0]
        error = target_flat - statistic.detach()
        if float(error.square().mean().sqrt()) <= tolerance:
            break
        directional_derivative = (
            gradient.detach().reshape(source.shape[0], -1)
            * unit_direction.reshape(source.shape[0], -1)
        ).sum(dim=-1)
        regularized_magnitude = (
            (error + directional_derivative * magnitude)
            * directional_derivative
            / (directional_derivative.square() + ridge).clamp_min(1e-12)
        )
        proposed_magnitude = magnitude + damping * (regularized_magnitude - magnitude)
        if maximum_relative_norm is not None:
            limit = maximum_relative_norm * source_norm
            proposed_magnitude = torch.minimum(proposed_magnitude, limit)
            proposed_magnitude = (
                proposed_magnitude.clamp_min(0)
                if nonnegative_magnitude
                else torch.maximum(proposed_magnitude, -limit)
            )
        elif nonnegative_magnitude:
            proposed_magnitude = proposed_magnitude.clamp_min(0)
        if backtracking_steps:
            with torch.no_grad():
                base_objective = (statistic.detach() - target_flat).square() + ridge * magnitude.square()
                accepted = torch.zeros_like(magnitude, dtype=torch.bool)
                accepted_magnitude = magnitude.clone()
                step_delta = proposed_magnitude - magnitude
                for backtrack in range(backtracking_steps):
                    candidate_magnitude = magnitude + (0.5**backtrack) * step_delta
                    candidate_delta = unit_direction * candidate_magnitude.reshape(
                        (-1,) + (1,) * (source.ndim - 1)
                    )
                    candidate_statistic = statistic_fn(
                        source_fp32 + candidate_delta
                    ).float().reshape(source.shape[0])
                    candidate_objective = (
                        (candidate_statistic - target_flat).square()
                        + ridge * candidate_magnitude.square()
                    )
                    improve = (~accepted) & (candidate_objective <= base_objective)
                    accepted_magnitude[improve] = candidate_magnitude[improve]
                    accepted |= improve
                    if bool(accepted.all()):
                        break
                magnitude = accepted_magnitude
        else:
            magnitude = proposed_magnitude
        completed_steps = step + 1

    delta = unit_direction * magnitude.reshape((-1,) + (1,) * (source.ndim - 1))
    with torch.no_grad():
        final = statistic_fn(source_fp32 + delta).float().reshape(source.shape[0])
        initial_rmse = (initial.detach() - target_flat).square().mean().sqrt()
        final_rmse = (final - target_flat).square().mean().sqrt()
        action_norm = _row_norm(delta).mean()
        relative_norm = (_row_norm(delta) / source_norm.clamp_min(1e-12)).mean()

    diagnostics = PerturbationDiagnostics(
        initial_target_rmse=float(initial_rmse),
        final_target_rmse=float(final_rmse),
        action_norm=float(action_norm),
        relative_action_norm=float(relative_norm),
        iterations=completed_steps,
    )
    return delta.to(dtype=source.dtype), diagnostics
