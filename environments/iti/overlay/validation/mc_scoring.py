"""Tokenizer-aware causal scoring helpers for TruthfulQA answer candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnswerTokenSpan:
    """Token span for the answer suffix in a complete candidate prompt."""

    answer_start: int
    answer_end: int

    @property
    def causal_start(self) -> int:
        """First hidden-state position whose logits predict an answer token."""

        return self.answer_start - 1


def find_answer_token_span(tokenizer, prompt: str, answer: str) -> AnswerTokenSpan:
    """Locate an answer suffix without assuming a tokenizer-specific prefix length.

    A token that starts in the separating whitespace and ends inside the answer is
    an answer token. This matters for SentencePiece and byte-level tokenizers,
    which commonly merge the leading answer space into the first answer token.
    """

    if not prompt.endswith(answer):
        raise ValueError("The candidate answer must be the exact suffix of the prompt")
    answer_start_char = len(prompt) - len(answer)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("A fast tokenizer with offset mappings is required")
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0][0], list):
        offsets = offsets[0]

    answer_tokens = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > answer_start_char and end > start
    ]
    if not answer_tokens:
        raise ValueError("The answer suffix did not produce any tokens")
    start = answer_tokens[0]
    end = answer_tokens[-1] + 1
    if start == 0:
        raise ValueError("Cannot causally score an answer beginning at token zero")
    return AnswerTokenSpan(answer_start=start, answer_end=end)


def score_answer_logits(
    logits: torch.Tensor,
    prompt_ids: torch.Tensor,
    span: AnswerTokenSpan,
) -> float:
    """Return the summed teacher-forced log probability of answer tokens."""

    if logits.ndim != 3 or prompt_ids.ndim != 2:
        raise ValueError("Expected logits [batch, tokens, vocab] and ids [batch, tokens]")
    if logits.shape[0] != 1 or prompt_ids.shape[0] != 1:
        raise ValueError("TruthfulQA scoring currently requires batch size one")
    predictions = logits[0, span.answer_start - 1 : span.answer_end - 1].log_softmax(-1)
    targets = prompt_ids[0, span.answer_start : span.answer_end]
    indices = torch.arange(targets.shape[0], device=logits.device)
    return float(predictions[indices, targets].sum().item())
