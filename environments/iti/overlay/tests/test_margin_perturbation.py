from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT))

from local_truthfulqa_judge import extract_label
from validate_margin_perturbation import build_actions


def realized_shift(
    actions: dict[int, object],
    top_heads: list[tuple[int, int]],
    weights: np.ndarray,
    stds: np.ndarray,
    head_dim: int,
) -> float:
    gradients = weights / (len(top_heads) * stds[:, None])
    total = 0.0
    for index, (layer, head) in enumerate(top_heads):
        action = actions[layer][head * head_dim : (head + 1) * head_dim].numpy()
        total += float(gradients[index] @ action)
    return total


def test_zero_ridge_realizes_linearized_target() -> None:
    top_heads = [(0, 0), (1, 1)]
    weights = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32)
    stds = np.array([2.0, 4.0], dtype=np.float32)
    actions, _norm = build_actions(
        top_heads=top_heads,
        weights=weights,
        stds=stds,
        hidden_size=4,
        head_dim=2,
        requested_shift=0.7,
        ridge_ratio=0.0,
    )
    assert np.isclose(realized_shift(actions, top_heads, weights, stds, 2), 0.7)


def test_relative_ridge_shrinks_shift() -> None:
    top_heads = [(0, 0)]
    weights = np.array([[2.0, -1.0]], dtype=np.float32)
    stds = np.array([3.0], dtype=np.float32)
    actions, _norm = build_actions(
        top_heads=top_heads,
        weights=weights,
        stds=stds,
        hidden_size=2,
        head_dim=2,
        requested_shift=1.0,
        ridge_ratio=1.0,
    )
    assert np.isclose(realized_shift(actions, top_heads, weights, stds, 2), 0.5)


def test_local_judge_parser_is_conservative() -> None:
    assert extract_label("<truthful>YES</truthful>", "truthful") == "YES"
    assert extract_label("yes", "truthful") == "UNKNOWN"
