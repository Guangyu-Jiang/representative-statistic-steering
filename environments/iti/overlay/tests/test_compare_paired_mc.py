from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from compare_paired_mc import paired_bootstrap


def test_paired_bootstrap_preserves_fold_sizes() -> None:
    difference, low, high, p_nonpositive = paired_bootstrap(
        [np.ones(4), np.ones(6)],
        samples=100,
        seed=7,
    )
    assert difference == 1.0
    assert low == 1.0
    assert high == 1.0
    assert p_nonpositive == 0.0
