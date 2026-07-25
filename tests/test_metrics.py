import torch

from repstat_steering.pplm_control import (
    _categorical_kl,
    _geometric_probability_mix,
    _mask_cache_gradients,
    _select_token_kl_budget,
    distinct_ngram_fraction,
)
from repstat_steering.truthx_control import (
    TruthXInterventionConfig,
    calculate_mc_metrics,
    format_best,
    one_sided_margin_threshold,
    split_answers,
)


def test_truthx_intervention_threshold_can_be_decoupled_from_target() -> None:
    default = TruthXInterventionConfig(
        method="minimum_norm",
        target_mode="cosine_margin_decoder",
        target_strength=0.25,
    )
    matched_gate = TruthXInterventionConfig(
        method="minimum_norm",
        target_mode="cosine_margin_decoder",
        target_strength=0.25,
        intervention_margin_threshold=0.0,
    )
    assert one_sided_margin_threshold(default) == 0.25
    assert one_sided_margin_threshold(matched_gate) == 0.0


def test_truthfulqa_answer_helpers_and_metrics() -> None:
    true = split_answers("correct; also correct.")
    false = split_answers("wrong; very wrong")
    metrics = calculate_mc_metrics(
        [-1.0, -2.0], [-3.0, -4.0], true, format_best("correct")
    )
    assert metrics["mc1"] == 1.0
    assert metrics["mc2"] > 0.8
    assert metrics["mc3"] == 1.0
    assert metrics["valid_scores"] == 1.0


def test_nonfinite_truthfulqa_scores_are_invalid() -> None:
    metrics = calculate_mc_metrics(
        [float("nan")], [-1.0], ["correct."], "correct."
    )
    assert metrics["valid_scores"] == 0.0
    assert metrics["mc2"] != metrics["mc2"]


def test_distinct_ngrams() -> None:
    assert distinct_ngram_fraction(["a b a"], 1) == 2 / 3
    assert distinct_ngram_fraction(["a b a"], 2) == 1.0


def test_geometric_mix_respects_kl_budget() -> None:
    perturbed = torch.tensor([[4.0, 0.0, -1.0]])
    reference = torch.tensor([[0.0, 1.0, -1.0]])
    probabilities, scale, divergence = _geometric_probability_mix(
        perturbed,
        reference,
        temperature=1.0,
        maximum_scale=1.0,
        maximum_kl=0.01,
    )
    reference_probabilities = reference.softmax(dim=-1)
    measured = float(_categorical_kl(probabilities, reference_probabilities))
    assert 0.0 < scale < 1.0
    assert measured <= 0.0101
    assert abs(measured - divergence) < 1e-6


def test_token_kl_budget_increases_only_for_difficult_margin() -> None:
    assert _select_token_kl_budget(-4.1, 1.0, -4.0, 2.0) == 2.0
    assert _select_token_kl_budget(-4.0, 1.0, -4.0, 2.0) == 1.0
    assert _select_token_kl_budget(0.0, 1.0, None, None) == 1.0


def test_cache_gradient_mask_selects_late_values() -> None:
    cache = tuple((torch.ones(1), torch.ones(1)) for _ in range(3))
    gradients = tuple(torch.ones(1) for _ in range(6))
    selected = _mask_cache_gradients(gradients, cache, "value", 2)
    assert [float(item) for item in selected] == [0, 0, 0, 1, 0, 1]
