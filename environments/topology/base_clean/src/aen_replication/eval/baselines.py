"""Prompt and entropy baselines from the paper appendix."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.generation import GenerationResult, generate_batch
from aen_replication.models.hf_model import HFModelBundle
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet

LOGGER = logging.getLogger(__name__)

CLAMBER_ZEROSHOT_TEMPLATE = """Given a query, answer the
question or ask a clarifying
question. The response should
start with "The answer is" or "The
clarifying question is".
Question: {question}"""

CLAM_FEWSHOT_TEMPLATE = """Q: Who was the first woman to make
a solo flight across this ocean?
This question is ambiguous: True.
Q: Who was the first woman to
make a solo flight across the
Atlantic?
This question is ambiguous:
False.
Q: In which city were Rotary Clubs
set up in 1905?
This question is ambiguous:
False.
Q: Who along with Philips
developed the CD in the late 70s?
This question is ambiguous:
False.
Q: Where is the multinational
corporation based?
This question is ambiguous: True.
Q: {question}
This question is ambiguous:"""

CLAMBER_COT_TEMPLATE = """Given a query, answer the
question or ask a clarifying
question. The response should
start with "The answer is" or
"The clarifying question is".
Question: Who played Michael
Myers in Rob Zombie's movie?
Output: In Rob Zombie's
"Halloween" films, the role
of Michael Myers was primarily
played by Tyler Mane. Therefore,
the question is not ambiguous.
The answer is Tyler Mane.
Question: Give me some Mother's
Day gift ideas.
Output: There could be underlying
ambiguities depending on the
interests of the specific mother
in question, the budget, and
the giver's relationship to
the mother. Therefore, the
question is ambiguous. The
clarifying question is: What are
the interests or hobbies of the
mother, and is there a particular
budget range for the gift?
Question: {question}"""

INFOGAIN_TEMPLATE = """Evaluate the clarity of the input
question.
If the question is ambiguous,
enhance it by adding specific
details such as relevant
locations, time periods, or
additional context needed to
resolve the ambiguity.
For clear questions, simply
repeat the query as is.
Example:
Input Question: When did the
Frozen ride open at Epcot?
Disambiguation: When did the
Frozen ride open at Epcot?
Input Question: What is the legal
age of marriage in the USA?
Disambiguation: What is the legal
age of marriage in each state of
the USA, excluding exceptions for
parental consent?
Input Question: {question}
Disambiguation:"""


def _parse_true_false(response_text: str) -> int | None:
    normalized = response_text.strip().lower()
    if normalized.startswith("true"):
        return 1
    if normalized.startswith("false"):
        return 0
    match = re.search(r"\b(true|false)\b", normalized)
    if not match:
        return None
    return 1 if match.group(1) == "true" else 0


def _parse_answer_or_clarify(response_text: str) -> int | None:
    normalized = " ".join(response_text.strip().lower().split())
    if normalized.startswith("the clarifying question is"):
        return 1
    if normalized.startswith("the answer is"):
        return 0
    if "the clarifying question is" in normalized[:120]:
        return 1
    if "the answer is" in normalized[:80]:
        return 0
    return None


def _prediction_metrics(labels: np.ndarray, predictions: list[int | None]) -> dict[str, Any]:
    safe_predictions = np.array([prediction if prediction is not None else 0 for prediction in predictions], dtype=float)
    metrics = binary_classification_metrics(labels, safe_predictions, threshold=0.5)
    metrics["parse_failure_count"] = int(sum(prediction is None for prediction in predictions))
    metrics["parse_failure_rate"] = float(metrics["parse_failure_count"] / len(predictions))
    return metrics


def _coerce_finite_float(value: Any) -> tuple[float, bool]:
    """Return a finite float and whether the input had to be repaired."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, value is not None
    if not np.isfinite(numeric):
        return 0.0, True
    return numeric, False


def _iter_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def _run_prompt_baseline(
    bundle: HFModelBundle,
    df: pd.DataFrame,
    generation_cfg: dict[str, Any],
    prompt_builder: Callable[[str], str],
    parser: Callable[[str], int | None],
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions: list[int | None] = []
    labels = df["label_ambiguous"].to_numpy(dtype=int)
    batch_size = int(generation_cfg.get("batch_size", 8))
    for batch_df in tqdm(_iter_batches(df, batch_size), total=(len(df) + batch_size - 1) // batch_size, desc=method_name, leave=False):
        generations = generate_batch(
            bundle=bundle,
            prompt_texts=[prompt_builder(text) for text in batch_df["text"].tolist()],
            generation_config=generation_cfg,
        )
        for row, generation in zip(batch_df.itertuples(index=False), generations, strict=True):
            prediction = parser(generation.response_text)
            predictions.append(prediction)
            rows.append(
                {
                    "example_id": row.example_id,
                    "pair_id": row.pair_id,
                    "dataset": row.dataset,
                    "split": row.split,
                    "label_ambiguous": int(row.label_ambiguous),
                    "method": method_name,
                    "prompt_text": generation.prompt_text,
                    "response_text": generation.response_text,
                    "prediction": prediction,
                }
            )
    metrics = _prediction_metrics(labels, predictions)
    return pd.DataFrame(rows), metrics


def _run_infogain(
    bundle: HFModelBundle,
    df: pd.DataFrame,
    generation_cfg: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = df["label_ambiguous"].to_numpy(dtype=int)
    batch_size = int(generation_cfg.get("batch_size", 8))
    non_finite_entropy_count = 0
    original_results: list[GenerationResult] = []
    original_prompts = [INFOGAIN_TEMPLATE.format(question=text) for text in df["text"].tolist()]
    for prompt_batch in tqdm(
        [original_prompts[start : start + batch_size] for start in range(0, len(original_prompts), batch_size)],
        total=(len(original_prompts) + batch_size - 1) // batch_size,
        desc="infogain_original",
        leave=False,
    ):
        original_results.extend(
            generate_batch(
                bundle=bundle,
                prompt_texts=prompt_batch,
                generation_config=generation_cfg,
                return_entropy=True,
            )
        )

    refined_prompts = [
        INFOGAIN_TEMPLATE.format(question=(result.response_text or text))
        for result, text in zip(original_results, df["text"].tolist(), strict=True)
    ]
    refined_results: list[GenerationResult] = []
    for prompt_batch in tqdm(
        [refined_prompts[start : start + batch_size] for start in range(0, len(refined_prompts), batch_size)],
        total=(len(refined_prompts) + batch_size - 1) // batch_size,
        desc="infogain_refined",
        leave=False,
    ):
        refined_results.extend(
            generate_batch(
                bundle=bundle,
                prompt_texts=prompt_batch,
                generation_config=generation_cfg,
                return_entropy=True,
            )
        )

    scores: list[float] = []
    for row, original, refined in zip(df.itertuples(index=False), original_results, refined_results, strict=True):
        disambiguated_question = original.response_text or row.text
        entropy_original, repaired_original = _coerce_finite_float(original.average_entropy)
        entropy_refined, repaired_refined = _coerce_finite_float(refined.average_entropy)
        non_finite_entropy_count += int(repaired_original) + int(repaired_refined)
        score = entropy_original - entropy_refined
        scores.append(score)
        rows.append(
            {
                "example_id": row.example_id,
                "pair_id": row.pair_id,
                "dataset": row.dataset,
                "split": row.split,
                "label_ambiguous": int(row.label_ambiguous),
                "method": "infogain",
                "prompt_text": original.prompt_text,
                "response_text": original.response_text,
                "disambiguated_question": disambiguated_question,
                "entropy_original": entropy_original,
                "entropy_refined": entropy_refined,
                "score": score,
                "prediction": int(score > threshold),
            }
        )
    metrics = binary_classification_metrics(labels, np.asarray(scores, dtype=float), threshold=threshold)
    metrics["threshold"] = float(threshold)
    metrics["non_finite_entropy_count"] = int(non_finite_entropy_count)
    return pd.DataFrame(rows), metrics


def run_baselines(
    bundle: HFModelBundle,
    dataset_df: pd.DataFrame,
    config: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    """Evaluate prompt baselines on the paper's test split."""

    test_df = dataset_df.loc[dataset_df["split"] == "test"].reset_index(drop=True)
    baseline_cfg = config["baselines"]
    generation_cfg = config["generation"]
    model_slug = slugify(config["model"]["name"])
    artifact_root = ensure_dir(baseline_cfg["artifact_dir"]) / model_slug / dataset_name
    summary: dict[str, Any] = {"dataset": dataset_name, "model_name": config["model"]["name"], "methods": {}}

    methods: list[tuple[str, Callable[[str], str], Callable[[str], int | None]]] = [
        ("clam_fewshot", lambda question: CLAM_FEWSHOT_TEMPLATE.format(question=question), _parse_true_false),
        ("clamber_zeroshot", lambda question: CLAMBER_ZEROSHOT_TEMPLATE.format(question=question), _parse_answer_or_clarify),
        ("clamber_fewshot_cot", lambda question: CLAMBER_COT_TEMPLATE.format(question=question), _parse_answer_or_clarify),
    ]

    for method_name, builder, parser in methods:
        outputs_path = artifact_root / f"{method_name}.parquet"
        metrics_path = artifact_root / f"{method_name}.json"
        if outputs_path.exists() and metrics_path.exists():
            LOGGER.info("Skipping %s on %s because artifacts already exist", method_name, dataset_name)
            summary["methods"][method_name] = json.loads(metrics_path.read_text(encoding="utf-8"))
            write_json(artifact_root / "summary.json", summary)
            continue
        LOGGER.info("Running %s on %s (%d examples)", method_name, dataset_name, len(test_df))
        outputs_df, metrics = _run_prompt_baseline(
            bundle=bundle,
            df=test_df,
            generation_cfg=generation_cfg,
            prompt_builder=builder,
            parser=parser,
            method_name=method_name,
        )
        write_parquet(outputs_df, outputs_path)
        write_json(metrics_path, metrics)
        summary["methods"][method_name] = metrics
        write_json(artifact_root / "summary.json", summary)

    infogain_outputs_path = artifact_root / "infogain.parquet"
    infogain_metrics_path = artifact_root / "infogain.json"
    if infogain_outputs_path.exists() and infogain_metrics_path.exists():
        LOGGER.info("Skipping infogain on %s because artifacts already exist", dataset_name)
        summary["methods"]["infogain"] = json.loads(infogain_metrics_path.read_text(encoding="utf-8"))
        write_json(artifact_root / "summary.json", summary)
    else:
        LOGGER.info("Running infogain on %s (%d examples)", dataset_name, len(test_df))
        infogain_df, infogain_metrics = _run_infogain(
            bundle=bundle,
            df=test_df,
            generation_cfg=generation_cfg,
            threshold=float(baseline_cfg.get("infogain_threshold", 0.5)),
        )
        write_parquet(infogain_df, infogain_outputs_path)
        write_json(infogain_metrics_path, infogain_metrics)
        summary["methods"]["infogain"] = infogain_metrics
        write_json(artifact_root / "summary.json", summary)
    return summary
