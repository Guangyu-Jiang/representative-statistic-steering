from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "judge_exact_h0_gn_local.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("judge_exact_h0_gn_local", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_degenerate_response_gate_rejects_loops_but_not_grounded_clarification() -> None:
    looping = (
        "I think you mean Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, "
        "Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed."
    )
    grounded = (
        "There are several competitions called a final. Which sport, league, and year do you "
        "mean? With that information I can identify the winner."
    )

    assert MODULE._response_is_degenerate("")
    assert MODULE._response_is_degenerate(looping)
    assert not MODULE._response_is_degenerate(grounded)
