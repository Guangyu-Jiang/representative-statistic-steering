from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_exact_h0_gauss_newton_steering.py"
SPEC = importlib.util.spec_from_file_location("run_exact_h0_gauss_newton_steering", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_run_slug_separates_mean_shift_ablation() -> None:
    kwargs = {
        "target_mode": "nearest_abstention",
        "k": 5,
        "alpha": 1.0,
        "lambda_value": 0.1,
        "damping": 0.01,
        "trust_ratio": 0.05,
    }
    constrained = MODULE._run_slug(**kwargs, allow_mean_shift=False)
    unconstrained = MODULE._run_slug(**kwargs, allow_mean_shift=True)

    assert constrained != unconstrained
    assert "allow_mean_shift" not in constrained
    assert unconstrained.endswith("__allow_mean_shift")
