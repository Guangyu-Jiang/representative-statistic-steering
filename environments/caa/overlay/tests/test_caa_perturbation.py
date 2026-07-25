import torch

from caa_perturbation.core import (
    fisher_hinge_delta,
    fit_behavior_statistics,
    pca_statistic_shift_delta,
    pca_target_delta,
    project_pca,
    relative_norm,
    scalar_hinge_delta,
    scalar_target_delta,
)


def _synthetic_stats():
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(80, 12, generator=generator)
    behavior = torch.zeros(12)
    behavior[0] = 2.0
    letter = torch.zeros(12)
    letter[1] = 6.0
    positive_is_a = torch.arange(80) % 2 == 0
    letter_sign = torch.where(positive_is_a, 1.0, -1.0).unsqueeze(-1)
    positive = base + behavior + letter_sign * letter
    negative = base - behavior - letter_sign * letter
    stats = fit_behavior_statistics(
        positive,
        negative,
        positive_is_a,
        n_components=3,
        remove_letter=True,
    )
    return stats, positive, negative


def test_pairwise_pca_removes_letter_direction():
    stats, _, _ = _synthetic_stats()
    overlap = stats.components @ stats.letter_direction
    assert torch.all(overlap.abs() < 1e-5)


def test_pca_action_reduces_target_error():
    stats, _, negative = _synthetic_stats()
    hidden = negative[:8]
    before = (project_pca(hidden, stats) - stats.positive_centroid).norm(dim=-1)
    delta = pca_target_delta(hidden, stats, strength=1.0, ridge=0.01)
    after = (project_pca(hidden + delta, stats) - stats.positive_centroid).norm(dim=-1)
    assert torch.all(after < before)


def test_scalar_action_respects_relative_norm_cap():
    stats, _, negative = _synthetic_stats()
    hidden = negative[:8]
    delta = scalar_target_delta(
        hidden,
        stats,
        strength=10.0,
        ridge=0.0,
        max_relative_norm=0.05,
    )
    assert torch.all(relative_norm(delta, hidden) <= 0.050001)


def test_one_sided_scalar_action_does_not_pull_back_above_target_state():
    stats, _, negative = _synthetic_stats()
    target = stats.scalar_positive_targets[1]
    above_target = (
        stats.center + (target + 1.0) * stats.caa_direction
    ).unsqueeze(0)
    stationary = scalar_hinge_delta(
        above_target, stats, strength=2.0, ridge=0.1, target_quantile=0.75
    )
    assert torch.allclose(stationary, torch.zeros_like(stationary))

    below_target = negative[:8]
    before = (below_target - stats.center) @ stats.caa_direction
    delta = scalar_hinge_delta(
        below_target, stats, strength=1.0, ridge=0.1, target_quantile=0.75
    )
    after = (below_target + delta - stats.center) @ stats.caa_direction
    assert torch.all(after >= before)


def test_quantile_lookup_survives_bfloat16_model_conversion():
    stats, _, negative = _synthetic_stats()
    stats = stats.to("cpu", torch.bfloat16)
    delta = scalar_hinge_delta(
        negative[:2].bfloat16(),
        stats,
        strength=1.0,
        ridge=0.1,
        target_quantile=0.9,
    )
    assert torch.isfinite(delta).all()


def test_fisher_direction_removes_option_nuisance_and_hinge_raises_margin():
    stats, _, negative = _synthetic_stats()
    overlap = stats.fisher_direction @ stats.letter_direction
    assert overlap.abs() < 1e-5
    before = (negative[:8] - stats.center) @ stats.fisher_direction
    delta = fisher_hinge_delta(
        negative[:8], stats, strength=1.0, ridge=0.1, target_quantile=0.75
    )
    after = (negative[:8] + delta - stats.center) @ stats.fisher_direction
    assert torch.all(after >= before)


def test_pca_relative_shift_moves_in_positive_cluster_direction():
    stats, _, negative = _synthetic_stats()
    hidden = negative[:8]
    direction = stats.positive_centroid - stats.negative_centroid
    before = project_pca(hidden, stats) @ direction
    delta = pca_statistic_shift_delta(hidden, stats, strength=0.5, ridge=0.1)
    after = project_pca(hidden + delta, stats) @ direction
    assert torch.all(after > before)


def test_multichoice_pca_removes_option_subspace():
    generator = torch.Generator().manual_seed(11)
    examples = 90
    hidden_size = 14
    base = torch.randn(examples, hidden_size, generator=generator)
    behavior = torch.zeros(hidden_size)
    behavior[0] = 1.5
    option_vectors = torch.zeros(3, hidden_size)
    option_vectors[0, 1] = 5.0
    option_vectors[1, 2] = 5.0
    option_vectors[2, 3] = 5.0
    positive_options = torch.arange(examples) % 3
    negative_options = (positive_options + 1) % 3
    positive = base + behavior + option_vectors[positive_options]
    negative = base - behavior + option_vectors[negative_options]

    stats = fit_behavior_statistics(
        positive,
        negative,
        positive_is_a=None,
        n_components=3,
        remove_letter=True,
        positive_option_ids=positive_options,
        negative_option_ids=negative_options,
    )

    assert stats.letter_direction.ndim == 2
    overlap = stats.components @ stats.letter_direction.T
    assert torch.all(overlap.abs() < 1e-5)
