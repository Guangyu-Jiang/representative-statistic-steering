import torch
from transformers import LlamaConfig, LlamaForCausalLM

from repstat_steering.lookback_control import (
    LookbackAttentionController,
    LookbackClassifier,
    LookbackGenerationConfig,
    LookbackNQExperiment,
    NQExample,
    best_subspan_exact_match,
    bm25_relevance_scores,
    biased_lookback_ratios,
    matched_random_mask,
    solve_lookback_bias,
)


def test_minimum_norm_rerank_is_answer_blind_and_selects_highest_score() -> None:
    experiment = object.__new__(LookbackNQExperiment)
    observed_seeds: list[int] = []

    def fake_generate(_example, config):
        observed_seeds.append(config.seed)
        score = [0.2, 0.9, 0.5][len(observed_seeds) - 1]
        return {
            "method": config.method,
            "seed": config.seed,
            "mean_factual_probability": score,
            "final_factual_probability": score,
            "exact_match": float(len(observed_seeds) == 3),
            "response": f"candidate-{len(observed_seeds)}",
            "generated_token_ids": [len(observed_seeds)],
            "generated_tokens": 1,
            "mean_bias_rms": 0.1,
            "mean_output_kl": 0.01,
            "mean_target_logit_error": 0.2,
            "mean_actual_target_logit_error": 0.3,
        }

    experiment.generate_minimum_norm = fake_generate
    experiment._score_generated_tokens = (
        lambda _example, tokens, _window_size: [[0.1], [0.8], [0.3]][tokens[0] - 1]
    )
    example = NQExample(0, "question", ("answer",), "prompt")
    config = LookbackGenerationConfig(
        method="minimum_norm_rerank", num_candidates=3, seed=17
    )
    result = experiment.generate_minimum_norm_rerank(example, config)

    assert observed_seeds == [17, 1_000_020, 2_000_023]
    assert result["response"] == "candidate-2"
    assert result["selected_candidate_index"] == 1
    assert result["mean_factual_probability"] == 0.8
    assert result["controlled_mean_factual_probability"] == 0.9
    assert result["method"] == "minimum_norm_rerank"
    assert result["seed"] == 17


def test_baseline_rerank_uses_unsteered_candidates() -> None:
    experiment = object.__new__(LookbackNQExperiment)

    def fake_generate(_example, config):
        candidate_index = 0 if config.seed == 23 else 1
        return {
            "method": config.method,
            "seed": config.seed,
            "mean_factual_probability": [0.3, 0.4][candidate_index],
            "final_factual_probability": [0.3, 0.4][candidate_index],
            "exact_match": float(candidate_index == 1),
            "response": f"candidate-{candidate_index}",
            "generated_token_ids": [candidate_index + 1],
            "generated_tokens": 1,
            "mean_bias_rms": 0.0,
            "mean_output_kl": 0.0,
            "mean_target_logit_error": 0.0,
            "mean_actual_target_logit_error": 0.0,
        }

    experiment.generate_baseline = fake_generate
    experiment._score_generated_tokens = (
        lambda _example, tokens, _window_size: [0.2, 0.8][tokens[0] - 1 : tokens[0]]
    )
    result = experiment.generate_baseline_rerank(
        NQExample(0, "question", ("answer",), "prompt"),
        LookbackGenerationConfig(
            method="baseline_rerank", num_candidates=2, seed=23
        ),
    )

    assert result["method"] == "baseline_rerank"
    assert result["candidate_generation_method"] == "baseline"
    assert result["selected_candidate_index"] == 1
    assert result["exact_match"] == 1.0


def test_bm25_relevance_scores_select_question_specific_unit() -> None:
    units = [
        "A passage about basketball teams and championship games.",
        "The Yakima River is a tributary of the Columbia River in Washington.",
        "A passage about the history of European railways.",
    ]
    scores = bm25_relevance_scores("Where does the Yakima River flow?", units)
    assert scores.index(max(scores)) == 1
    assert scores[1] > scores[0]


def test_attention_controller_preserves_unsteered_causal_decoding() -> None:
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    input_ids = torch.tensor([[1, 5, 7, 9, 4]])
    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
    controller = LookbackAttentionController(model)
    try:
        with torch.no_grad():
            actual = model(input_ids, use_cache=False).logits
    finally:
        controller.close()
    torch.testing.assert_close(actual, expected)


def test_uniform_context_bias_adds_to_lookback_logit() -> None:
    ratios = torch.tensor([0.1, 0.4, 0.8])
    bias = torch.tensor([0.7, -0.2, 1.1])
    controlled = biased_lookback_ratios(ratios, bias)
    torch.testing.assert_close(torch.logit(controlled), torch.logit(ratios) + bias)


def test_focused_context_bias_has_exact_mass_fraction_transform() -> None:
    ratios = torch.tensor([0.2, 0.6, 0.9])
    bias = torch.tensor([1.2, -0.7, 0.3])
    fraction = torch.tensor([0.1, 0.4, 0.8])
    controlled = biased_lookback_ratios(ratios, bias, fraction)
    expected_log_scale = torch.log((1 - fraction) + fraction * bias.exp())
    torch.testing.assert_close(
        torch.logit(controlled), torch.logit(ratios) + expected_log_scale
    )


def test_matched_random_mask_is_deterministic_and_cardinality_matched() -> None:
    reference = torch.tensor([False, True, True, False, True, False])
    eligible = torch.tensor([True, True, False, True, True, True])
    first = matched_random_mask(reference, eligible, seed=17)
    second = matched_random_mask(reference, eligible, seed=17)
    torch.testing.assert_close(first, second)
    assert int(first.sum()) == int(reference.sum())
    assert torch.all(~first | eligible)


def test_minimum_norm_bias_increases_classifier_probability() -> None:
    classifier = LookbackClassifier(
        torch.tensor([2.0, -1.0, 2.0]), torch.tensor(-1.0), 0.5
    )
    base = torch.tensor([0.2, 0.5, 0.1])
    bias, diagnostics = solve_lookback_bias(
        base,
        [],
        classifier,
        target_probability=0.8,
        window_size=8,
        ridge=1e-4,
        steps=20,
        damping=1.0,
        maximum_bias_rms=4.0,
    )
    assert diagnostics.predicted_probability > diagnostics.initial_probability
    assert diagnostics.predicted_probability > 0.75
    assert bias.norm() > 0


def test_minimum_norm_bias_follows_signed_statistic_gradient() -> None:
    classifier = LookbackClassifier(
        torch.tensor([2.0, -3.0, 0.5]), torch.tensor(-0.4), 0.5
    )
    base = torch.tensor([0.2, 0.6, 0.8])
    point = torch.zeros_like(base).requires_grad_(True)
    statistic = classifier.logit(biased_lookback_ratios(base, point))
    gradient = torch.autograd.grad(statistic, point)[0]

    bias, _ = solve_lookback_bias(
        base,
        [],
        classifier,
        target_logit=float(statistic.detach() + 0.05),
        window_size=1,
        ridge=0.0,
        steps=1,
        damping=1.0,
        maximum_bias_rms=None,
    )

    assert torch.dot(bias, gradient) > 0
    assert bias[0] > 0
    assert bias[1] < 0
    assert bias[2] > 0
    scale = bias / gradient
    torch.testing.assert_close(
        scale,
        torch.full_like(scale, float(scale.mean())),
        rtol=1e-5,
        atol=1e-6,
    )


def test_minimum_norm_bias_is_one_sided() -> None:
    classifier = LookbackClassifier(torch.ones(2), torch.tensor(5.0), 0.5)
    bias, diagnostics = solve_lookback_bias(
        torch.tensor([0.5, 0.5]),
        [],
        classifier,
        target_probability=0.8,
        window_size=8,
        ridge=0.1,
        steps=5,
        damping=1.0,
        maximum_bias_rms=1.0,
    )
    torch.testing.assert_close(bias, torch.zeros_like(bias))
    assert diagnostics.initial_probability > 0.8


def test_minimum_norm_bias_can_be_sparse_and_nonnegative() -> None:
    classifier = LookbackClassifier(
        torch.tensor([4.0, -3.0, 2.0, -1.0]), torch.tensor(-2.0), 0.5
    )
    bias, diagnostics = solve_lookback_bias(
        torch.tensor([0.2, 0.3, 0.4, 0.5]),
        [],
        classifier,
        target_logit=1.0,
        window_size=1,
        ridge=0.0,
        steps=20,
        damping=1.0,
        maximum_bias_rms=4.0,
        active_control_count=2,
        bias_constraint="nonnegative",
    )
    assert torch.count_nonzero(bias) <= 2
    assert torch.all(bias >= 0)
    assert diagnostics.predicted_probability > diagnostics.initial_probability


def test_sparse_nonnegative_solver_selects_positive_gradient_heads() -> None:
    classifier = LookbackClassifier(
        torch.tensor([-100.0, 4.0, 3.0, 2.0]), torch.tensor(-2.0), 0.5
    )
    bias, diagnostics = solve_lookback_bias(
        torch.tensor([0.2, 0.2, 0.2, 0.2]),
        [],
        classifier,
        target_logit=0.0,
        window_size=1,
        ridge=0.0,
        steps=20,
        damping=1.0,
        maximum_bias_rms=4.0,
        active_control_count=2,
        bias_constraint="nonnegative",
    )
    assert bias[0] == 0
    assert torch.count_nonzero(bias) == 2
    assert torch.all(bias >= 0)
    assert diagnostics.predicted_probability > diagnostics.initial_probability


def test_regularization_penalizes_accumulated_bias() -> None:
    classifier = LookbackClassifier(torch.tensor([3.0]), torch.tensor(-1.0), 0.5)
    base = torch.tensor([0.2])
    weak_bias, _ = solve_lookback_bias(
        base,
        [],
        classifier,
        target_probability=0.95,
        window_size=1,
        ridge=1e-4,
        steps=20,
        damping=0.5,
        maximum_bias_rms=None,
    )
    strong_bias, strong = solve_lookback_bias(
        base,
        [],
        classifier,
        target_probability=0.95,
        window_size=1,
        ridge=1.0,
        steps=20,
        damping=0.5,
        maximum_bias_rms=None,
    )
    assert strong_bias.norm() < weak_bias.norm()
    assert strong.control_objective >= 0


def test_solver_monotonically_improves_control_objective() -> None:
    classifier = LookbackClassifier(
        torch.tensor([2.0, -4.0]), torch.tensor(-0.5), 0.5
    )
    base = torch.tensor([0.01, 0.99])
    target_probability = 0.95
    target_logit = torch.logit(torch.tensor(target_probability))
    initial_error = target_logit - classifier.logit(base)
    _, diagnostics = solve_lookback_bias(
        base,
        [],
        classifier,
        target_probability=target_probability,
        window_size=1,
        ridge=0.01,
        steps=20,
        damping=1.0,
        maximum_bias_rms=2.0,
    )
    assert diagnostics.control_objective <= float(initial_error.square()) + 1e-6


def test_solver_accepts_direct_logit_target_without_probability_clamping() -> None:
    classifier = LookbackClassifier(torch.tensor([10.0]), torch.tensor(-5.0), 0.5)
    _, diagnostics = solve_lookback_bias(
        torch.tensor([0.5]),
        [],
        classifier,
        target_logit=4.0,
        window_size=1,
        ridge=0.0,
        steps=30,
        damping=1.0,
        maximum_bias_rms=10.0,
    )
    assert diagnostics.target_logit_error < 1e-3


def test_best_subspan_exact_match_matches_official_metric() -> None:
    assert best_subspan_exact_match(
        "The answer is Wilhelm Conrad Rontgen.", ["Wilhelm Conrad Rontgen"]
    ) == 1.0
    assert best_subspan_exact_match("It was Einstein.", ["Rontgen"]) == 0.0
