from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_exact_h0_hybrid_steering.py"
SPEC = importlib.util.spec_from_file_location("run_exact_h0_hybrid_steering", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_shared_shift_preserves_h0_after_centered_deformation() -> None:
    generator = np.random.default_rng(7)
    cloud = generator.normal(size=(12, 5))
    centered = generator.normal(scale=0.1, size=cloud.shape)
    centered -= centered.mean(axis=0, keepdims=True)
    shared = generator.normal(scale=0.3, size=(1, cloud.shape[1]))

    before_shift = MODULE.exact._exact_features_tensor(
        torch.as_tensor(cloud + centered, dtype=torch.float64)
    )
    after_shift = MODULE.exact._exact_features_tensor(
        torch.as_tensor(cloud + centered + shared, dtype=torch.float64)
    )

    assert torch.allclose(before_shift, after_shift, atol=1e-10, rtol=1e-10)
    assert np.allclose((centered + shared).mean(axis=0), shared[0])


def test_arrow_object_matrix_is_densified() -> None:
    value = np.asarray([np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])], dtype=object)
    result = MODULE._as_float_matrix(value)

    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    assert np.array_equal(result, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_bounded_slug_is_stable_unique_and_within_limit() -> None:
    first = MODULE._bounded_slug("setting_" + "a" * 400, max_length=80)
    second = MODULE._bounded_slug("setting_" + "b" * 400, max_length=80)

    assert len(first) <= 80
    assert first == MODULE._bounded_slug("setting_" + "a" * 400, max_length=80)
    assert first != second


def test_behavior_rank1_controller_changes_topology_with_centered_token_gates() -> None:
    generator = np.random.default_rng(11)
    n_tokens, pca_dim, hidden_dim = 16, 4, 9
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    direction = generator.normal(size=hidden_dim).astype(np.float32)
    direction_unit = direction / np.linalg.norm(direction)
    projected = direction_unit @ components.T
    target_gates = np.linspace(-0.15, 0.15, n_tokens, dtype=np.float32)
    target_cloud = cloud + target_gates[:, None] * projected[None, :]
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(target_cloud, dtype=torch.float64)
    ).numpy()

    gates, delta_y, diagnostics = MODULE._optimize_behavior_rank1_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        direction_np=direction,
        components_np=components,
        lambda_value=1e-4,
        damping=1e-4,
        trust_ratio=0.2,
        gn_steps=16,
        line_search_steps=10,
    )

    assert abs(float(gates.mean())) < 1e-6
    assert np.allclose(delta_y, gates[:, None] * projected[None, :], atol=1e-6)
    assert diagnostics["final_normalized_target_error"] < diagnostics["initial_normalized_target_error"]
    assert diagnostics["behavior_direction_pca_projection_ratio"] > 0.0


def test_self_tokenwise_template_resamples_and_preserves_last_token_direction() -> None:
    rows = pd.DataFrame({"example_id": ["q1"]})
    sequence = np.asarray(
        [[0.0, 1.0], [2.0, 1.0], [4.0, 3.0]],
        dtype=np.float32,
    )

    directions, templates = MODULE._self_tokenwise_directions_and_templates(
        rows,
        {"q1": sequence},
        [5],
        readout="last_token",
    )

    assert np.array_equal(directions[0], sequence[-1])
    assert templates[0].shape == (5, 2)
    scale = templates[0][-1, 1] / sequence[-1, 1]
    assert np.allclose(templates[0][0], scale * sequence[0])
    assert np.allclose(templates[0][-1], scale * sequence[-1])
    assert np.isclose(np.linalg.norm(templates[0]) / np.sqrt(5), 1.0, atol=1e-6)


def test_local_neighbor_tokenwise_template_contrasts_matched_groups() -> None:
    rows = pd.DataFrame(
        {
            "plus_neighbor_ids": ["p0|p1"],
            "minus_neighbor_ids": ["m0|m1"],
        }
    )
    sequences = {
        "p0": np.asarray([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32),
        "p1": np.asarray([[4.0, 1.0], [6.0, 3.0]], dtype=np.float32),
        "m0": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "m1": np.asarray([[2.0, 0.0], [3.0, 1.0]], dtype=np.float32),
    }

    directions, templates = MODULE._local_neighbor_tokenwise_directions_and_templates(
        rows,
        sequences,
        [3],
        readout="mean_pool",
    )

    shape_reference = np.zeros((3, 1), dtype=np.float32)
    plus = np.stack(
        [MODULE._position_cloud_match(shape_reference, sequences[key]) for key in ["p0", "p1"]]
    )
    minus = np.stack(
        [MODULE._position_cloud_match(shape_reference, sequences[key]) for key in ["m0", "m1"]]
    )
    raw_template = plus.mean(axis=0) - minus.mean(axis=0)
    expected_template = raw_template / (np.linalg.norm(raw_template) / np.sqrt(3))

    assert np.allclose(directions[0], raw_template.mean(axis=0))
    assert np.allclose(templates[0], expected_template)
    assert np.isclose(np.linalg.norm(templates[0]) / np.sqrt(3), 1.0, atol=1e-6)


def test_behavior_tokenwise_controller_reduces_reachable_target_error() -> None:
    generator = np.random.default_rng(13)
    n_tokens, pca_dim, hidden_dim = 17, 5, 11
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    template = generator.normal(size=(n_tokens, hidden_dim)).astype(np.float32)
    template /= np.linalg.norm(template) / np.sqrt(n_tokens)
    target_gates = np.linspace(-0.04, 0.06, n_tokens, dtype=np.float32)
    target_hidden = target_gates[:, None] * template
    target_hidden -= target_hidden.mean(axis=0, keepdims=True)
    target_cloud = cloud + target_hidden @ components.T
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(target_cloud, dtype=torch.float64)
    ).numpy()

    gates, delta_y, diagnostics = MODULE._optimize_behavior_tokenwise_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        template_np=template,
        components_np=components,
        lambda_value=1e-4,
        damping=1e-4,
        trust_ratio=0.2,
        gn_steps=16,
        line_search_steps=10,
    )

    optimized_hidden = gates[:, None] * template
    optimized_hidden -= optimized_hidden.mean(axis=0, keepdims=True)
    assert np.linalg.norm(optimized_hidden.mean(axis=0)) < 1e-6
    assert np.allclose(delta_y, optimized_hidden @ components.T, atol=1e-5)
    assert diagnostics["final_normalized_target_error"] < diagnostics["initial_normalized_target_error"]


def test_arrow_object_vector_is_densified() -> None:
    value = np.asarray([np.asarray(1.0), np.asarray(2.0)], dtype=object)
    result = MODULE._as_float_vector(value)

    assert result.dtype == np.float32
    assert np.array_equal(result, np.asarray([1.0, 2.0], dtype=np.float32))


def test_high_precision_behavior_markers_separate_clarification_from_direct_answer() -> None:
    assert MODULE._has_high_precision_behavior_marker(
        "Which season do you mean? Please specify the year."
    )
    assert MODULE._has_high_precision_behavior_marker(
        "I do not have real-time information about the current winner."
    )
    assert not MODULE._has_high_precision_behavior_marker(
        "The South Sudanese Civil War began on December 15, 2013."
    )
    assert not MODULE._has_high_precision_behavior_marker(
        "The wife of Nigeria's president is Aisha Buhari."
    )


def test_targeted_marker_excludes_generic_refusal_and_accepts_specific_clarification() -> None:
    assert MODULE._has_targeted_clarification_marker(
        "Which season and country do you mean? Please specify the version."
    )
    assert MODULE._has_targeted_clarification_marker(
        "Could you provide more context about which municipal stadium you mean?"
    )
    assert not MODULE._has_targeted_clarification_marker(
        "I'm not aware of that information."
    )


def test_audited_positive_quality_gate_rejects_repetitive_generation() -> None:
    looping = (
        "I think you mean Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, "
        "Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed."
    )
    assert MODULE._is_degenerate_response(looping)
    assert not MODULE._is_degenerate_response(
        "Which final do you mean? Please specify the sport, league, and year."
    )


def test_position_cloud_match_preserves_order_and_endpoint_states() -> None:
    query = np.zeros((5, 2), dtype=np.float32)
    reference = np.asarray([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)

    matched = MODULE._position_cloud_match(query, reference)

    assert matched.shape == query.shape
    assert np.allclose(matched[0], reference[0])
    assert np.allclose(matched[-1], reference[-1])
    assert np.allclose(matched[:, 0], np.arange(5, dtype=np.float32))


def test_topology_decode_vector_uses_causal_prompt_suffix() -> None:
    token_deltas = np.arange(24, dtype=np.float32).reshape(6, 4)

    last = MODULE._topology_decode_vector(
        token_deltas,
        mode="last_token",
        scale=0.5,
        suffix_fraction=0.25,
    )
    suffix = MODULE._topology_decode_vector(
        token_deltas,
        mode="suffix_mean",
        scale=2.0,
        suffix_fraction=0.5,
    )
    disabled = MODULE._topology_decode_vector(
        token_deltas,
        mode="none",
        scale=9.0,
        suffix_fraction=0.5,
    )

    assert np.allclose(last, 0.5 * token_deltas[-1])
    assert np.allclose(suffix, 2.0 * token_deltas[-3:].mean(axis=0))
    assert np.array_equal(disabled, np.zeros(4, dtype=np.float32))


def test_shared_target_ratio_normalizes_prompt_relative_frobenius_norm() -> None:
    directions = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    hidden_norms = np.asarray([100.0, 60.0])
    token_counts = np.asarray([25, 9])

    shared, scales = MODULE._scaled_shared_vectors(
        directions,
        hidden_norms,
        token_counts,
        mean_alpha=None,
        target_ratio=0.1,
    )
    achieved = np.sqrt(token_counts) * np.linalg.norm(shared, axis=1) / hidden_norms

    assert np.all(scales > 0.0)
    assert np.allclose(achieved, 0.1, atol=1e-6)


def test_self_contrastive_direction_preserves_query_specific_pairing() -> None:
    rows = pd.DataFrame({"example_id": ["q1", "q2"]})
    differences = {
        "q1": np.asarray([3.0, 4.0, 0.0], dtype=np.float32),
        "q2": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }

    directions, bases = MODULE._self_paired_directions_and_bases(rows, differences)

    assert np.array_equal(directions[0], differences["q1"])
    assert np.array_equal(directions[1], differences["q2"])
    assert np.allclose(bases[0], np.asarray([[0.6, 0.8, 0.0]], dtype=np.float32))
    assert np.array_equal(bases[1], np.zeros((1, 3), dtype=np.float32))


def test_audited_local_behavior_filters_by_confidence_and_margin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    behavior_path = tmp_path / "behavior.parquet"
    label_path = tmp_path / "labels.parquet"
    behavior = pd.DataFrame(
        {
            "example_id": ["high", "low"],
            "split": ["train", "train"],
            "text": ["Which final?", "What capital?"],
            "response_text": ["Which sport and year?", "Please clarify."],
            "judge_label": ["ACCEPTABLE", "ACCEPTABLE"],
        }
    )
    behavior.to_parquet(behavior_path, index=False)
    pair_hashes = [
        MODULE._pair_hash(question, response)
        for question, response in zip(behavior["text"], behavior["response_text"], strict=True)
    ]
    pd.DataFrame(
        {
            "pair_hash": pair_hashes,
            "local_judge_label": ["GROUNDED_ACCEPTABLE", "GROUNDED_ACCEPTABLE"],
            "local_judge_confidence": [0.9, 0.55],
            "local_judge_margin": [0.7, 0.05],
        }
    ).to_parquet(label_path, index=False)
    monkeypatch.setattr(MODULE.exact, "_resolve_behavior_path", lambda *_args: behavior_path)
    args = SimpleNamespace(
        behavior_label_source="audited_local_fourway",
        local_label_path=str(label_path),
        local_label_confidence_min=0.7,
        local_label_margin_min=0.2,
        require_positive_rule_marker=False,
        positive_label_mode="grounded",
    )

    result = MODULE._load_local_behavior(args, {}, "ambigqa")

    assert result["example_id"].tolist() == ["high"]
    assert result["behavior_label"].tolist() == [1]


def test_local_behavior_basis_is_orthonormal_and_starts_with_mean_contrast() -> None:
    rows = pd.DataFrame(
        {"plus_neighbor_ids": ["p0|p1|p2"], "minus_neighbor_ids": ["m0|m1|m2"]}
    )
    vectors = {
        "p0": np.asarray([2.0, 1.0, 0.0, 0.0]),
        "p1": np.asarray([1.0, 2.0, 1.0, 0.0]),
        "p2": np.asarray([1.0, 1.0, 0.0, 1.0]),
        "m0": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "m1": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "m2": np.asarray([0.0, 0.0, 0.0, 0.0]),
    }
    basis = MODULE._local_direction_bases(rows, vectors, rank=3)[0]
    mean_contrast = np.mean([vectors[f"p{i}"] - vectors[f"m{i}"] for i in range(3)], axis=0)
    mean_contrast /= np.linalg.norm(mean_contrast)

    assert basis.shape == (3, 4)
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)
    assert np.allclose(basis[0], mean_contrast, atol=1e-6)


def test_local_paired_direction_uses_same_question_contrast_neighbors() -> None:
    rows = pd.DataFrame(
        {"plus_neighbor_ids": ["p0|p1"], "minus_neighbor_ids": ["m0|p1"]}
    )
    differences = {
        "p0": np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        "p1": np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        "m0": np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
    }

    direction = MODULE._local_paired_directions(rows, differences)[0]
    basis = MODULE._local_paired_direction_bases(rows, differences, rank=3)[0]
    expected = np.mean(np.stack(list(differences.values())), axis=0)

    assert np.allclose(direction, expected)
    assert np.allclose(basis[0], expected / np.linalg.norm(expected), atol=1e-6)
    assert np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-6)


def test_behavior_lowrank_controller_reduces_reachable_target_error() -> None:
    generator = np.random.default_rng(19)
    n_tokens, pca_dim, hidden_dim, rank = 18, 5, 10, 3
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    basis, _ = np.linalg.qr(generator.normal(size=(hidden_dim, rank)))
    basis = basis.T.astype(np.float32)
    coefficients = generator.normal(scale=0.04, size=(n_tokens, rank)).astype(np.float32)
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    target_cloud = cloud + coefficients @ (basis @ components.T)
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(target_cloud, dtype=torch.float64)
    ).numpy()

    optimized, delta_y, diagnostics = MODULE._optimize_behavior_lowrank_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        basis_np=basis,
        components_np=components,
        lambda_value=1e-4,
        damping=1e-4,
        trust_ratio=0.2,
        gn_steps=16,
        line_search_steps=10,
    )

    assert np.linalg.norm(optimized.mean(axis=0)) < 1e-6
    assert np.allclose(delta_y, optimized @ (basis @ components.T), atol=1e-6)
    assert diagnostics["final_normalized_target_error"] < diagnostics["initial_normalized_target_error"]


def test_causal_anchor_moves_final_token_and_bounds_topology_error() -> None:
    generator = np.random.default_rng(23)
    n_tokens, pca_dim, hidden_dim, rank = 14, 5, 11, 3
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    basis, _ = np.linalg.qr(generator.normal(size=(hidden_dim, rank)))
    basis = basis.T.astype(np.float32)
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(cloud, dtype=torch.float64)
    ).numpy()

    optimized, _delta_y, diagnostics = MODULE._optimize_behavior_lowrank_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        basis_np=basis,
        components_np=components,
        lambda_value=0.1,
        damping=0.01,
        trust_ratio=0.1,
        gn_steps=4,
        line_search_steps=8,
        causal_anchor_ratio=0.02,
        causal_anchor_max_error_increase=0.1,
    )

    assert diagnostics["causal_anchor_applied_norm"] > 0.0
    assert diagnostics["final_token_behavior_coefficient"] > 0.0
    assert diagnostics["final_normalized_target_error"] <= 0.1 + 1e-8
    assert np.linalg.norm(optimized.mean(axis=0)) < 1e-6


def test_suffix_causal_anchor_distributes_positive_behavior_shift() -> None:
    generator = np.random.default_rng(29)
    n_tokens, pca_dim, hidden_dim, rank = 16, 5, 12, 3
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    basis, _ = np.linalg.qr(generator.normal(size=(hidden_dim, rank)))
    basis = basis.T.astype(np.float32)
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(cloud, dtype=torch.float64)
    ).numpy()

    optimized, _delta_y, diagnostics = MODULE._optimize_behavior_lowrank_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        basis_np=basis,
        components_np=components,
        lambda_value=0.1,
        damping=0.01,
        trust_ratio=0.1,
        gn_steps=4,
        line_search_steps=8,
        causal_anchor_ratio=0.02,
        causal_anchor_max_error_increase=0.1,
        causal_anchor_suffix_fraction=0.25,
    )

    suffix_count = diagnostics["causal_anchor_suffix_tokens"]
    assert suffix_count == 4
    assert optimized[-suffix_count:, 0].mean() > 0.0
    assert optimized[:-suffix_count, 0].mean() < 0.0
    assert np.linalg.norm(optimized.mean(axis=0)) < 1e-6


def test_position_weighting_concentrates_reachable_change_near_suffix() -> None:
    generator = np.random.default_rng(31)
    n_tokens, pca_dim, hidden_dim, rank = 18, 5, 12, 3
    cloud = generator.normal(size=(n_tokens, pca_dim)).astype(np.float32)
    components, _ = np.linalg.qr(generator.normal(size=(hidden_dim, pca_dim)))
    components = components.T.astype(np.float32)
    basis, _ = np.linalg.qr(generator.normal(size=(hidden_dim, rank)))
    basis = basis.T.astype(np.float32)
    target_coefficients = generator.normal(scale=0.05, size=(n_tokens, rank)).astype(np.float32)
    target_coefficients -= target_coefficients.mean(axis=0, keepdims=True)
    target_cloud = cloud + target_coefficients @ (basis @ components.T)
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(target_cloud, dtype=torch.float64)
    ).numpy()

    _uniform, _uniform_delta, uniform_diagnostics = MODULE._optimize_behavior_lowrank_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        basis_np=basis,
        components_np=components,
        lambda_value=1e-4,
        damping=1e-3,
        trust_ratio=0.2,
        gn_steps=8,
        line_search_steps=8,
        causal_position_beta=0.0,
    )
    _weighted, _weighted_delta, weighted_diagnostics = MODULE._optimize_behavior_lowrank_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        basis_np=basis,
        components_np=components,
        lambda_value=1e-4,
        damping=1e-3,
        trust_ratio=0.2,
        gn_steps=8,
        line_search_steps=8,
        causal_position_beta=4.0,
    )

    assert weighted_diagnostics["position_weight_min"] < 0.02
    assert (
        weighted_diagnostics["suffix_quarter_coefficient_norm_fraction"]
        > uniform_diagnostics["suffix_quarter_coefficient_norm_fraction"]
    )


def test_local_transport_template_is_centered_and_points_toward_positive_cloud() -> None:
    query = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    plus = query + np.asarray([[0.0, 0.1], [0.0, 0.3], [0.0, 0.6]], dtype=np.float32)
    minus = query + np.asarray([[0.0, -0.1], [0.0, -0.1], [0.0, -0.1]], dtype=np.float32)
    rows = pd.DataFrame(
        {
            "example_id": ["q"],
            "plus_neighbor_ids": ["p"],
            "minus_neighbor_ids": ["m"],
        }
    )
    clouds = pd.DataFrame(
        {"example_id": ["q", "p", "m"], "cloud": [query, plus, minus]}
    ).set_index("example_id")

    template = MODULE._local_transport_templates(rows, clouds)[0]

    assert template.shape == query.shape
    assert np.linalg.norm(template.mean(axis=0)) < 1e-7
    assert np.linalg.norm(template) > 0.0


def test_transport_prior_optimizer_reduces_exact_target_error() -> None:
    generator = np.random.default_rng(37)
    cloud = generator.normal(size=(16, 5)).astype(np.float32)
    template = generator.normal(size=cloud.shape).astype(np.float32)
    template -= template.mean(axis=0, keepdims=True)
    desired = template * 0.08
    target = MODULE.exact._exact_features_tensor(
        torch.as_tensor(cloud + desired, dtype=torch.float64)
    ).numpy()

    optimized, diagnostics = MODULE._optimize_transport_one(
        cloud_np=cloud,
        target_np=target,
        feature_std_np=np.ones(len(MODULE.FEATURES), dtype=np.float32),
        hidden_norm=20.0,
        template_np=template,
        prior_ratio=0.03,
        lambda_value=1e-4,
        damping=1e-3,
        trust_ratio=0.2,
        gn_steps=12,
        line_search_steps=8,
    )

    assert np.linalg.norm(optimized.mean(axis=0)) < 1e-6
    assert diagnostics["final_normalized_target_error"] < diagnostics["initial_normalized_target_error"]
    assert diagnostics["transport_prior_applied_norm"] > 0.0


def test_classifier_projection_moves_exact_h0_score_toward_positive_class() -> None:
    rows = []
    examples = [
        *(("train", 0, -1.0 - 0.1 * index) for index in range(8)),
        *(("train", 1, 1.0 + 0.1 * index) for index in range(5)),
        ("test", 0, -1.5),
    ]
    for index, (split, label, offset) in enumerate(examples):
        rows.append(
            {
                "example_id": f"e{index}",
                "split": split,
                "behavior_label": label,
                "response_text": "direct" if label == 0 else "please clarify",
                "judge_label": "UNACCEPTABLE" if label == 0 else "GROUNDED_ACCEPTABLE",
                "retrieval": offset,
                MODULE.FEATURES[0]: offset,
                MODULE.FEATURES[1]: 0.5 * offset,
                MODULE.FEATURES[2]: 0.25 * offset,
            }
        )
    frame = pd.DataFrame(rows)

    selected, current, target, _std, _plus, _minus = MODULE._select_neighbors_and_targets(
        frame=frame,
        retrieval_columns=["retrieval"],
        retrieval_geometry="standard",
        target_mode="classifier_projection",
        k=3,
        eval_n=0,
        limit=None,
        seed=7,
        classifier_target_quantile=0.5,
    )

    assert len(selected) == 1
    assert selected.loc[0, "classifier_target_score"] > selected.loc[0, "classifier_current_score"]
    assert np.linalg.norm(target - current) > 0.0
