from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from mc_scoring import find_answer_token_span, score_answer_logits


TOKENIZER_PATHS = (
    Path.home()
    / ".cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee",
    Path.home()
    / ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct",
)


def llama3_snapshot() -> Path | None:
    snapshots = TOKENIZER_PATHS[1] / "snapshots"
    return next(snapshots.iterdir(), None) if snapshots.exists() else None


@pytest.mark.parametrize("path", (TOKENIZER_PATHS[0], llama3_snapshot()))
def test_answer_span_keeps_first_semantic_token(path: Path | None) -> None:
    if path is None or not path.exists():
        pytest.skip("cached tokenizer is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    answer = "Nauru is the smallest country."
    prompt = f"Q: What is the smallest country?\nA: {answer}"
    span = find_answer_token_span(tokenizer, prompt, answer)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    decoded = tokenizer.decode(ids[0, span.answer_start : span.answer_end])
    assert "Nauru" in decoded


def test_score_uses_exact_answer_span() -> None:
    prompt_ids = torch.tensor([[7, 8, 2, 3]])
    logits = torch.full((1, 4, 10), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    from mc_scoring import AnswerTokenSpan

    score = score_answer_logits(logits, prompt_ids, AnswerTokenSpan(2, 4))
    assert score > -1e-4
