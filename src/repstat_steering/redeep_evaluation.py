"""Local and lexical evaluation utilities for ReDeEP Dolly generations."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any, Iterable


REDEEP_JUDGE_LABELS = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "NONANSWER",
)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}


def normalize_tokens(text: str, *, remove_stopwords: bool = False) -> list[str]:
    translation = str.maketrans({character: " " for character in string.punctuation})
    tokens = text.lower().translate(translation).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_tokens(prediction)
    reference_tokens = normalize_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def reference_token_recall(prediction: str, reference: str) -> float:
    prediction_tokens = set(normalize_tokens(prediction, remove_stopwords=True))
    reference_tokens = set(normalize_tokens(reference, remove_stopwords=True))
    if not reference_tokens:
        return float(not prediction_tokens)
    return len(prediction_tokens & reference_tokens) / len(reference_tokens)


def passage_grounding_ratio(response: str, passage: str, question: str) -> float:
    response_tokens = normalize_tokens(response, remove_stopwords=True)
    if not response_tokens:
        return 0.0
    source_tokens = set(
        normalize_tokens(passage + " " + question, remove_stopwords=True)
    )
    return sum(token in source_tokens for token in response_tokens) / len(response_tokens)


def build_local_redeep_judge_prompt(
    *, question: str, passage: str, reference_answer: str, response: str
) -> str:
    return f"""You are evaluating an answer produced by a retrieval-augmented language model. Use only the supplied passage and reference answer as evidence. Do not use outside knowledge.

Classify the candidate answer into exactly one class:

SUPPORTED: It answers the question and every important factual claim is supported by the passage or reference answer. Minor wording differences are allowed.

PARTIALLY_SUPPORTED: It contains a useful supported answer, but it also contains at least one unsupported or incorrect factual claim, or it omits a substantial required part of a list-style answer.

UNSUPPORTED: It attempts a direct answer, but its main answer is unsupported by or contradicts the passage/reference answer.

NONANSWER: It is empty, irrelevant, malformed, or refuses/fails to answer the question. A safe refusal is still NONANSWER because this evaluation measures both truthfulness and usefulness.

Question:
{question}

Evidence passage:
{passage[:12000]}

Reference answer:
{reference_answer}

Candidate answer:
{response}

Return exactly one JSON object and no other text:
{{"label":"SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|NONANSWER","reason":"one short reason"}}"""


def parse_local_judge_output(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        payload: Any = json.loads(fenced)
        label = str(payload.get("label", "")).upper().strip()
        reason = str(payload.get("reason", "")).strip()
        if label in REDEEP_JUDGE_LABELS:
            return label, reason
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    pattern = "|".join(REDEEP_JUDGE_LABELS)
    match = re.search(rf"\b({pattern})\b", cleaned.upper())
    if match:
        return match.group(1), cleaned
    return "PARSE_ERROR", cleaned


def build_local_pairwise_judge_prompt(
    *,
    question: str,
    passage: str,
    answer_a: str,
    answer_b: str,
) -> str:
    return f"""You are comparing two answers from retrieval-augmented language models. Use only the supplied evidence passage. Decide which answer is more truthful and useful for the question.

Prefer an answer whose factual claims are supported by the passage. If both are supported, prefer the more complete and directly responsive answer. Unsupported additions are worse than omissions. A refusal is not better than a supported useful answer. Choose TIE when the answers are equally truthful and useful or differ only in wording.

Question:
{question}

Evidence passage:
{passage[:12000]}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return exactly one JSON object and no other text:
{{"winner":"A|B|TIE","reason":"one short reason"}}"""


def parse_local_pairwise_output(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        payload: Any = json.loads(fenced)
        winner = str(payload.get("winner", "")).upper().strip()
        reason = str(payload.get("reason", "")).strip()
        if winner in {"A", "B", "TIE"}:
            return winner, reason
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    match = re.search(r"\b(A|B|TIE)\b", cleaned.upper())
    if match:
        return match.group(1), cleaned
    return "PARSE_ERROR", cleaned


def summarize_judged_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        count = len(method_rows)
        labels = Counter(str(row["local_judge_label"]) for row in method_rows)
        summaries.append(
            {
                "method": method,
                "examples": count,
                "supported_pct": 100 * labels["SUPPORTED"] / count,
                "partially_supported_pct": 100
                * labels["PARTIALLY_SUPPORTED"]
                / count,
                "unsupported_pct": 100 * labels["UNSUPPORTED"] / count,
                "nonanswer_pct": 100 * labels["NONANSWER"] / count,
                "parse_error_pct": 100 * labels["PARSE_ERROR"] / count,
                "mean_reference_token_f1": sum(
                    float(row["reference_token_f1"]) for row in method_rows
                )
                / count,
                "mean_reference_token_recall": sum(
                    float(row["reference_token_recall"]) for row in method_rows
                )
                / count,
                "mean_passage_grounding_ratio": sum(
                    float(row["passage_grounding_ratio"]) for row in method_rows
                )
                / count,
            }
        )
    return summaries
