from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BehaviorStatistics:
    center: torch.Tensor
    caa_vector: torch.Tensor
    caa_direction: torch.Tensor
    scalar_positive_target: torch.Tensor
    target_quantiles: torch.Tensor
    scalar_positive_targets: torch.Tensor
    clean_caa_vector: torch.Tensor
    clean_caa_direction: torch.Tensor
    clean_positive_targets: torch.Tensor
    fisher_direction: torch.Tensor
    fisher_positive_targets: torch.Tensor
    fisher_mean_pair_shift: torch.Tensor
    components: torch.Tensor
    component_scale: torch.Tensor
    positive_centroid: torch.Tensor
    negative_centroid: torch.Tensor
    pca_margin_positive_targets: torch.Tensor
    letter_direction: torch.Tensor
    explained_variance_ratio: torch.Tensor

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "BehaviorStatistics":
        values: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            target_dtype = torch.float32 if field_name == "target_quantiles" else dtype
            values[field_name] = value.to(device=device, dtype=target_dtype)
        return BehaviorStatistics(**values)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            field_name: getattr(self, field_name).detach().cpu()
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "BehaviorStatistics":
        # Older experiment artifacts predate the improved statistics. Retain
        # enough compatibility to load them for inspection and generation.
        if "target_quantiles" not in state:
            state = dict(state)
            state["target_quantiles"] = torch.tensor([0.5, 0.75, 0.9])
            state["scalar_positive_targets"] = state[
                "scalar_positive_target"
            ].repeat(3)
            state["clean_caa_vector"] = state["caa_vector"]
            state["clean_caa_direction"] = state["caa_direction"]
            state["clean_positive_targets"] = state[
                "scalar_positive_target"
            ].repeat(3)
            state["fisher_direction"] = state["caa_direction"]
            state["fisher_positive_targets"] = state[
                "scalar_positive_target"
            ].repeat(3)
            state["fisher_mean_pair_shift"] = state["caa_vector"].norm()
            pca_axis = _safe_unit(
                state["positive_centroid"] - state["negative_centroid"]
            )
            pca_target = state["positive_centroid"] @ pca_axis
            state["pca_margin_positive_targets"] = pca_target.repeat(
                state["components"].shape[0], 3
            )
        return cls(**state)


def _safe_unit(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm().clamp_min(eps)


def remove_direction(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove either one direction or the row span of several directions."""
    if direction.ndim == 1:
        direction = _safe_unit(direction)
        return values - (values @ direction).unsqueeze(-1) * direction
    if direction.ndim != 2:
        raise ValueError("direction must have shape [hidden] or [rank, hidden]")
    if direction.shape[0] == 0:
        return values
    # SVD makes this robust to linearly dependent category-mean directions.
    _, singular_values, right_vectors = torch.linalg.svd(
        direction.float(), full_matrices=False
    )
    threshold = singular_values.max().clamp_min(1e-8) * 1e-6
    basis = right_vectors[singular_values > threshold].to(values)
    return values - (values @ basis.T) @ basis


def _option_nuisance_directions(
    all_values: torch.Tensor,
    all_options: torch.Tensor,
) -> torch.Tensor:
    """Return the activation subspace explained by answer-option identity."""
    unique_options = torch.unique(all_options, sorted=True)
    if unique_options.numel() < 2:
        raise ValueError("At least two answer options are required")

    # Preserve the original A/B implementation exactly for old artifacts and
    # strictly binary datasets.
    if unique_options.tolist() == [0, 1]:
        mean_a = all_values[all_options == 0].mean(dim=0)
        mean_b = all_values[all_options == 1].mean(dim=0)
        return _safe_unit(mean_a - mean_b)

    center = all_values.mean(dim=0)
    category_directions = torch.stack(
        [all_values[all_options == option].mean(dim=0) - center for option in unique_options]
    )
    _, singular_values, right_vectors = torch.linalg.svd(
        category_directions, full_matrices=False
    )
    threshold = singular_values.max().clamp_min(1e-8) * 1e-6
    return right_vectors[singular_values > threshold].contiguous()


def _fit_low_rank_fisher_direction(
    positive: torch.Tensor,
    negative: torch.Tensor,
    mean_direction: torch.Tensor,
    nuisance_direction: torch.Tensor,
    rank: int,
    shrinkage: float,
    seed: int,
) -> torch.Tensor:
    """Fit a regularized Fisher direction in a stable low-rank subspace."""
    positive_centered = positive - positive.mean(dim=0)
    negative_centered = negative - negative.mean(dim=0)
    within = torch.cat([positive_centered, negative_centered], dim=0)
    within = remove_direction(within, nuisance_direction)

    # Always retain the behavior mean direction. PCA then models only
    # within-class variation orthogonal to that direction.
    mean_direction = _safe_unit(mean_direction)
    residual = remove_direction(within, mean_direction)
    residual_rank = min(max(rank - 1, 0), min(residual.shape) - 1)
    basis_rows = [mean_direction.unsqueeze(0)]
    if residual_rank > 0:
        torch.manual_seed(seed)
        q = min(residual_rank + 4, min(residual.shape))
        _, _, right_vectors = torch.pca_lowrank(
            residual,
            q=q,
            center=False,
            niter=6,
        )
        basis_rows.append(right_vectors[:, :residual_rank].T)
    basis = torch.cat(basis_rows, dim=0)
    basis = remove_direction(basis, nuisance_direction)
    basis = torch.linalg.qr(basis.T, mode="reduced").Q.T.contiguous()

    positive_z = positive @ basis.T
    negative_z = negative @ basis.T
    positive_residual = positive_z - positive_z.mean(dim=0)
    negative_residual = negative_z - negative_z.mean(dim=0)
    denominator = max(1, positive.shape[0] + negative.shape[0] - 2)
    covariance = (
        positive_residual.T @ positive_residual
        + negative_residual.T @ negative_residual
    ) / denominator
    mean_difference = positive_z.mean(dim=0) - negative_z.mean(dim=0)
    covariance_scale = (
        torch.trace(covariance) / covariance.shape[0]
    ).clamp_min(1e-6)
    regularized = covariance + shrinkage * covariance_scale * torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    coefficients = torch.linalg.solve(regularized, mean_difference)
    direction = _safe_unit(coefficients @ basis)
    # Resolve the arbitrary sign so larger scores always denote the positive
    # behavior represented by the matching CAA examples.
    if (positive.mean(dim=0) - negative.mean(dim=0)) @ direction < 0:
        direction = -direction
    return direction


def fit_behavior_statistics(
    positive: torch.Tensor,
    negative: torch.Tensor,
    positive_is_a: torch.Tensor | None,
    n_components: int,
    remove_letter: bool = True,
    scalar_quantile: float = 0.75,
    target_quantiles: tuple[float, ...] = (0.5, 0.75, 0.9),
    fisher_rank: int = 32,
    fisher_shrinkage: float = 0.1,
    seed: int = 0,
    positive_option_ids: torch.Tensor | None = None,
    negative_option_ids: torch.Tensor | None = None,
) -> BehaviorStatistics:
    """Fit CAA and pairwise-centered PCA statistics for one layer.

    The PCA basis is learned from +/- half pair differences. This removes each
    question's midpoint before finding behavior-varying directions. The answer
    letter direction is projected out because it is a known nuisance in CAA's
    multiple-choice activation geometry.
    """
    if positive.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("positive and negative must have shape [examples, hidden_size]")
    if not 1 <= n_components <= min(positive.shape):
        raise ValueError("invalid PCA component count")

    positive = positive.float()
    negative = negative.float()
    all_values = torch.cat([positive, negative], dim=0)
    center = all_values.mean(dim=0)

    if positive_option_ids is not None or negative_option_ids is not None:
        if positive_option_ids is None or negative_option_ids is None:
            raise ValueError("Both positive and negative option IDs must be provided")
        if (
            positive_option_ids.numel() != positive.shape[0]
            or negative_option_ids.numel() != positive.shape[0]
        ):
            raise ValueError("Option IDs must contain one value per activation pair")
        all_options = torch.cat(
            [positive_option_ids.long(), negative_option_ids.long()], dim=0
        )
    else:
        if positive_is_a is None or positive.shape[0] != positive_is_a.numel():
            raise ValueError("positive_is_a must contain one value per activation pair")
        positive_is_a = positive_is_a.bool()
        all_options = torch.cat(
            [
                torch.where(positive_is_a, 0, 1),
                torch.where(positive_is_a, 1, 0),
            ],
            dim=0,
        )
    letter_direction = _option_nuisance_directions(all_values, all_options)

    pair_difference = positive - negative
    caa_vector = pair_difference.mean(dim=0)
    caa_direction = _safe_unit(caa_vector)

    clean_pair_difference = pair_difference
    if remove_letter:
        clean_pair_difference = remove_direction(
            clean_pair_difference, letter_direction
        )
    clean_caa_vector = clean_pair_difference.mean(dim=0)
    clean_caa_direction = _safe_unit(clean_caa_vector)

    pair_centered = torch.cat([0.5 * pair_difference, -0.5 * pair_difference], dim=0)
    if remove_letter:
        pair_centered = remove_direction(pair_centered, letter_direction)

    torch.manual_seed(seed)
    q = min(max(n_components + 4, n_components), min(pair_centered.shape))
    _, singular_values, right_vectors = torch.pca_lowrank(
        pair_centered,
        q=q,
        center=False,
        niter=6,
    )
    components = right_vectors[:, :n_components].T.contiguous()

    # Numerical noise can reintroduce a tiny letter component.
    if remove_letter:
        components = remove_direction(components, letter_direction)
        components = torch.linalg.qr(components.T, mode="reduced").Q.T.contiguous()

    raw_positive = (positive - center) @ components.T
    raw_negative = (negative - center) @ components.T
    pooled = torch.cat([raw_positive, raw_negative], dim=0)
    component_scale = pooled.std(dim=0, unbiased=False).clamp_min(1e-5)
    positive_z = raw_positive / component_scale
    negative_z = raw_negative / component_scale

    # Orient each component so that the positive centroid is non-negative.
    signs = torch.where(positive_z.mean(dim=0) >= 0, 1.0, -1.0)
    components = components * signs.unsqueeze(-1)
    positive_z = positive_z * signs
    negative_z = negative_z * signs

    quantiles = torch.tensor(target_quantiles, dtype=positive.dtype)
    if quantiles.ndim != 1 or quantiles.numel() == 0:
        raise ValueError("target_quantiles must contain at least one value")
    if torch.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("target quantiles must lie strictly between zero and one")

    scalar_positive = (positive - center) @ caa_direction
    scalar_positive_target = torch.quantile(scalar_positive, scalar_quantile)
    scalar_positive_targets = torch.quantile(scalar_positive, quantiles)
    clean_positive = (positive - center) @ clean_caa_direction
    clean_positive_targets = torch.quantile(clean_positive, quantiles)

    clean_positive_values = remove_direction(positive - center, letter_direction)
    clean_negative_values = remove_direction(negative - center, letter_direction)
    fisher_direction = _fit_low_rank_fisher_direction(
        clean_positive_values,
        clean_negative_values,
        clean_caa_direction,
        letter_direction,
        rank=min(fisher_rank, min(positive.shape)),
        shrinkage=fisher_shrinkage,
        seed=seed,
    )
    fisher_positive = (positive - center) @ fisher_direction
    fisher_positive_targets = torch.quantile(fisher_positive, quantiles)
    fisher_mean_pair_shift = ((positive - negative) @ fisher_direction).mean()

    pca_margin_positive_targets = []
    for component_count in range(1, n_components + 1):
        behavior_axis = _safe_unit(
            positive_z[:, :component_count].mean(dim=0)
            - negative_z[:, :component_count].mean(dim=0)
        )
        positive_margin = positive_z[:, :component_count] @ behavior_axis
        pca_margin_positive_targets.append(
            torch.quantile(positive_margin, quantiles)
        )
    pca_margin_positive_targets = torch.stack(pca_margin_positive_targets)

    variance = singular_values[:n_components].square()
    explained_variance_ratio = variance / singular_values.square().sum().clamp_min(1e-8)
    return BehaviorStatistics(
        center=center,
        caa_vector=caa_vector,
        caa_direction=caa_direction,
        scalar_positive_target=scalar_positive_target,
        target_quantiles=quantiles,
        scalar_positive_targets=scalar_positive_targets,
        clean_caa_vector=clean_caa_vector,
        clean_caa_direction=clean_caa_direction,
        clean_positive_targets=clean_positive_targets,
        fisher_direction=fisher_direction,
        fisher_positive_targets=fisher_positive_targets,
        fisher_mean_pair_shift=fisher_mean_pair_shift,
        components=components,
        component_scale=component_scale,
        positive_centroid=positive_z.mean(dim=0),
        negative_centroid=negative_z.mean(dim=0),
        pca_margin_positive_targets=pca_margin_positive_targets,
        letter_direction=letter_direction,
        explained_variance_ratio=explained_variance_ratio,
    )


def project_pca(hidden: torch.Tensor, stats: BehaviorStatistics) -> torch.Tensor:
    return ((hidden - stats.center) @ stats.components.T) / stats.component_scale


def cap_relative_norm(
    delta: torch.Tensor,
    hidden: torch.Tensor,
    max_relative_norm: float | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if max_relative_norm is None:
        return delta
    delta_norm = delta.norm(dim=-1, keepdim=True)
    hidden_norm = hidden.norm(dim=-1, keepdim=True).clamp_min(eps)
    maximum = max_relative_norm * hidden_norm
    factor = torch.minimum(torch.ones_like(delta_norm), maximum / delta_norm.clamp_min(eps))
    return delta * factor


def fixed_caa_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    delta = strength * stats.caa_vector.expand_as(hidden)
    return cap_relative_norm(delta, hidden, max_relative_norm)


def scalar_target_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    current = (hidden - stats.center) @ stats.caa_direction
    error = strength * (stats.scalar_positive_target - current)
    delta = (error / (1.0 + ridge)).unsqueeze(-1) * stats.caa_direction
    return cap_relative_norm(delta, hidden, max_relative_norm)


def _quantile_target(
    stats: BehaviorStatistics,
    targets: torch.Tensor,
    target_quantile: float,
) -> torch.Tensor:
    distances = (stats.target_quantiles.float() - target_quantile).abs()
    index = int(distances.argmin())
    if float(distances[index]) > 5e-3:
        available = ", ".join(f"{float(value):g}" for value in stats.target_quantiles)
        raise ValueError(
            f"Target quantile {target_quantile:g} is unavailable; choose one of {available}"
        )
    return targets[index]


def _one_sided_linear_target_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    direction: torch.Tensor,
    targets: torch.Tensor,
    target_quantile: float,
    strength: float,
    ridge: float,
    max_relative_norm: float | None,
) -> torch.Tensor:
    """Minimum-norm action that only raises an under-target linear score."""
    current = (hidden - stats.center) @ direction
    target = _quantile_target(stats, targets, target_quantile)
    error = strength * (target - current).clamp_min(0)
    denominator = direction.square().sum() + ridge
    delta = (error / denominator).unsqueeze(-1) * direction
    return cap_relative_norm(delta, hidden, max_relative_norm)


def scalar_hinge_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    target_quantile: float = 0.75,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    return _one_sided_linear_target_delta(
        hidden,
        stats,
        stats.caa_direction,
        stats.scalar_positive_targets,
        target_quantile,
        strength,
        ridge,
        max_relative_norm,
    )


def clean_scalar_hinge_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    target_quantile: float = 0.75,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    return _one_sided_linear_target_delta(
        hidden,
        stats,
        stats.clean_caa_direction,
        stats.clean_positive_targets,
        target_quantile,
        strength,
        ridge,
        max_relative_norm,
    )


def fisher_hinge_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    target_quantile: float = 0.75,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    return _one_sided_linear_target_delta(
        hidden,
        stats,
        stats.fisher_direction,
        stats.fisher_positive_targets,
        target_quantile,
        strength,
        ridge,
        max_relative_norm,
    )


def _linear_statistic_shift_delta(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    score_shift: torch.Tensor,
    strength: float,
    ridge: float,
    max_relative_norm: float | None,
) -> torch.Tensor:
    """Invert a requested relative shift of a linear statistic."""
    denominator = direction.square().sum() + ridge
    magnitude = strength * score_shift / denominator
    delta = magnitude * direction.expand_as(hidden)
    return cap_relative_norm(delta, hidden, max_relative_norm)


def clean_statistic_shift_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    score_shift = stats.clean_caa_vector @ stats.clean_caa_direction
    return _linear_statistic_shift_delta(
        hidden,
        stats.clean_caa_direction,
        score_shift,
        strength,
        ridge,
        max_relative_norm,
    )


def fisher_statistic_shift_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    return _linear_statistic_shift_delta(
        hidden,
        stats.fisher_direction,
        stats.fisher_mean_pair_shift,
        strength,
        ridge,
        max_relative_norm,
    )


def pca_target_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    """Minimum-norm action toward the positive PCA-cluster centroid."""
    # Keep the model forward in BF16, but solve the small r x r system in
    # FP32 because torch.linalg does not support BF16 inversion/solves.
    hidden_float = hidden.float()
    center = stats.center.float()
    components = stats.components.float()
    component_scale = stats.component_scale.float()
    positive_centroid = stats.positive_centroid.float()
    current = ((hidden_float - center) @ components.T) / component_scale
    error = strength * (positive_centroid - current)
    operator = components / component_scale.unsqueeze(-1)
    gram = operator @ operator.T
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(regularized, error.T).T
    delta = (coefficients @ operator).to(hidden.dtype)
    return cap_relative_norm(delta, hidden, max_relative_norm)


def pca_statistic_shift_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    """Invert a relative positive-minus-negative shift in PCA space."""
    components = stats.components.float()
    component_scale = stats.component_scale.float()
    error = strength * (
        stats.positive_centroid.float() - stats.negative_centroid.float()
    )
    operator = components / component_scale.unsqueeze(-1)
    gram = operator @ operator.T
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(regularized, error)
    delta_vector = (coefficients @ operator).to(hidden.dtype)
    delta = delta_vector.expand_as(hidden)
    return cap_relative_norm(delta, hidden, max_relative_norm)


def pca_margin_hinge_delta(
    hidden: torch.Tensor,
    stats: BehaviorStatistics,
    strength: float,
    ridge: float,
    target_quantile: float = 0.75,
    max_relative_norm: float | None = None,
) -> torch.Tensor:
    """Raise only an under-target discriminative margin in PCA space."""
    hidden_float = hidden.float()
    components = stats.components.float()
    component_scale = stats.component_scale.float()
    current_z = (
        (hidden_float - stats.center.float()) @ components.T
    ) / component_scale
    behavior_axis = _safe_unit(
        stats.positive_centroid.float() - stats.negative_centroid.float()
    )
    current_margin = current_z @ behavior_axis
    component_count = components.shape[0]
    targets = stats.pca_margin_positive_targets[component_count - 1]
    target = _quantile_target(stats, targets, target_quantile).float()
    error = strength * (target - current_margin).clamp_min(0)

    operator = components / component_scale.unsqueeze(-1)
    margin_gradient = behavior_axis @ operator
    denominator = margin_gradient.square().sum() + ridge
    delta = (error / denominator).unsqueeze(-1) * margin_gradient
    delta = delta.to(hidden.dtype)
    return cap_relative_norm(delta, hidden, max_relative_norm)


def relative_norm(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return delta.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1e-8)
