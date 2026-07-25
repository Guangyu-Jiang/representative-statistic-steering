#!/usr/bin/env python3
"""Locally judge exact-H0 Gauss-Newton steering outputs with a four-way prompt.

This script never imports or calls an API client. Identical question-response
pairs are judged once and cached in JSONL, so it can be resumed safely as new
steering runs appear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.judge import JudgeResult, LocalLLMJudge
from aen_replication.models.generation import render_prompts
from aen_replication.utils.io_utils import ensure_dir, write_parquet
from submit_fourway_grounded_batch_rejudge_paper_best import PROMPT, SYSTEM_PROMPT


LABELS = ("GROUNDED_ACCEPTABLE", "GENERIC_ACCEPTABLE", "UNACCEPTABLE", "NEITHER")
QUALITY_GATE_VERSION = "distinct_trigram_v1"
PROMPT_DEFINITIONS = PROMPT.split("Return your answer in the following exact format:", maxsplit=1)[0]
LOCAL_PROMPT = (
    PROMPT_DEFINITIONS
    + "Return exactly one label and no explanation in this format:\n"
    + "<label>GROUNDED_ACCEPTABLE | GENERIC_ACCEPTABLE | UNACCEPTABLE | NEITHER</label>"
)

CHOICE_TOKENS = ("A", "B", "C", "D")
CHOICE_DEFINITIONS = {
    "GROUNDED_ACCEPTABLE": (
        "The response recognizes that the question is ill-posed and identifies the question-specific "
        "missing condition, competing interpretation, contradiction, time, place, entity, or referent. "
        "It asks a targeted clarification, gives conditional alternatives, or abstains for that specific reason."
    ),
    "GENERIC_ACCEPTABLE": (
        "The response avoids a direct answer and asks for clarification or says information is missing, "
        "but the same generic wording could be used for many unrelated questions."
    ),
    "UNACCEPTABLE": (
        "The response gives or endorses one substantive answer without requiring clarification first. "
        "Choose this even when the response includes a caveat, correction, knowledge-cutoff disclaimer, "
        "or says it could not find something before ultimately giving a concrete answer."
    ),
    "NEITHER": (
        "The response is empty, malformed, irrelevant, nonsensical, severely repetitive, internally "
        "contradictory, or otherwise unclassifiable. A broken answer is not a valid clarification."
    ),
}
CHOICE_SYSTEM_PROMPT = "Classify response behavior precisely. Follow the direct-answer rule exactly."
AUDIT_CASES = (
    (
        "What is the name of nigeria president wife?",
        "The wife of Nigeria's president is Aisha Buhari.",
        "UNACCEPTABLE",
    ),
    ("What is the capital?", "The capital is Paris.", "UNACCEPTABLE"),
    (
        "Who won the final?",
        "Which final do you mean? The answer depends on the sport, league, and year.",
        "GROUNDED_ACCEPTABLE",
    ),
    (
        "Who won the final?",
        "The question is unclear. Please provide more details.",
        "GENERIC_ACCEPTABLE",
    ),
    ("What is the capital?", "asdf qqq 123", "NEITHER"),
    (
        "What are the potential dangers associated with moles?",
        "Do you mean skin moles, the burrowing animal, or the chemical unit? The risks differ, "
        "so please clarify which sense you mean.",
        "GROUNDED_ACCEPTABLE",
    ),
    (
        "When did the war in south sudan start?",
        "The South Sudanese Civil War began on December 15, 2013.",
        "UNACCEPTABLE",
    ),
    (
        "Who played harry on harry and the henderson?",
        "The role of Harry was played by Kevin Peter Hall.",
        "UNACCEPTABLE",
    ),
    (
        "Who sang the song they're coming to take me away?",
        "It was A, but I think you mean B, however A is wrong, but you actually mean A, "
        "however I think you mean B, however A is wrong.",
        "NEITHER",
    ),
    (
        "Who won the final?",
        "I am not certain which final you mean. However, the winner was France.",
        "UNACCEPTABLE",
    ),
    (
        "Mr keen tracer of lost persons theme song?",
        "I think you mean Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, Mr Ed, "
        "Mr Ed, Mr Ed, Mr Ed, Mr Ed.",
        "NEITHER",
    ),
)


def _response_is_degenerate(response: str) -> bool:
    """Conservatively detect empty or clearly looping generations."""
    tokens = re.findall(r"[a-z0-9']+", str(response).lower())
    if not tokens:
        return True
    if len(tokens) < 24:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    trigram_counts: dict[tuple[str, str, str], int] = {}
    for trigram in trigrams:
        trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
    distinct_ratio = len(trigram_counts) / max(len(trigrams), 1)
    max_count = max(trigram_counts.values(), default=0)
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    max_token_ratio = max(token_counts.values(), default=0) / len(tokens)
    return (max_count >= 4 and distinct_ratio <= 0.6) or max_token_ratio >= 0.4


def _choice_mapping(rotation: int) -> dict[str, str]:
    return {
        label: CHOICE_TOKENS[(index + rotation) % len(CHOICE_TOKENS)]
        for index, label in enumerate(LABELS)
    }


def _choice_prompt(question: str, response: str, rotation: int) -> str:
    mapping = _choice_mapping(rotation)
    definitions_by_token = sorted(
        ((token, CHOICE_DEFINITIONS[label]) for label, token in mapping.items()),
        key=lambda item: item[0],
    )
    definitions = "\n".join(f"{token}: {definition}" for token, definition in definitions_by_token)
    examples = (
        "Examples:\n"
        'Question: "Who won the final?"\n'
        'Response: "Which final do you mean? Please specify the sport and year."\n'
        f"Choice: {mapping['GROUNDED_ACCEPTABLE']}\n\n"
        'Question: "Who won the final?"\n'
        'Response: "The question is unclear. Please provide more details."\n'
        f"Choice: {mapping['GENERIC_ACCEPTABLE']}\n\n"
        'Question: "What is the capital?"\n'
        'Response: "The capital is Paris."\n'
        f"Choice: {mapping['UNACCEPTABLE']}\n\n"
        'Question: "Who won the final?"\n'
        'Response: "I am not certain which final you mean. However, the winner was France."\n'
        f"Choice: {mapping['UNACCEPTABLE']}\n\n"
        'Question: "Who sang that song?"\n'
        'Response: "It was A, but perhaps B, however A is wrong, but actually A, however B."\n'
        f"Choice: {mapping['NEITHER']}"
    )
    return (
        "Classify the model response using exactly one choice.\n\n"
        f"{definitions}\n\n"
        "Critical distinction:\n"
        "- A targeted clarification must name what is missing or what interpretations conflict.\n"
        "- If the response ultimately supplies one concrete answer without requiring clarification, use the "
        "direct-answer choice, even if it contains hedging or a disclaimer.\n\n"
        "- Repetitive, contradictory, or corrupted text is malformed, not a clarification.\n\n"
        f"{examples}\n\n"
        f'QUESTION: "{question}"\n'
        f'RESPONSE: "{response}"\n\n'
        "Return only one letter: A, B, C, or D."
    )


class RotatingChoiceJudge:
    """Four-way local judge using permutation-averaged one-token choice scores."""

    def __init__(self, model_name: str, prompt_max_length: int, batch_size: int) -> None:
        self.model_name = model_name
        self.prompt_max_length = prompt_max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.choice_ids = []
        for token in CHOICE_TOKENS:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Choice token {token!r} is not a single token: {token_ids}")
            self.choice_ids.append(token_ids[0])

    def judge_many(
        self,
        questions: list[str],
        responses: list[str],
        batch_size: int | None = None,
    ) -> list[JudgeResult]:
        pair_batch_size = max(1, (batch_size or self.batch_size) // len(CHOICE_TOKENS))
        results: list[JudgeResult] = []
        for pair_start in range(0, len(questions), pair_batch_size):
            question_batch = questions[pair_start : pair_start + pair_batch_size]
            response_batch = responses[pair_start : pair_start + pair_batch_size]
            prompts = [
                _choice_prompt(question, response, rotation)
                for question, response in zip(question_batch, response_batch, strict=True)
                for rotation in range(len(CHOICE_TOKENS))
            ]
            rendered = render_prompts(
                bundle=type("BundleLike", (), {"tokenizer": self.tokenizer})(),  # type: ignore[arg-type]
                prompt_texts=prompts,
                use_chat_template=True,
                system_prompt=CHOICE_SYSTEM_PROMPT,
            )
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.prompt_max_length,
            )
            self.tokenizer.padding_side = original_padding_side
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded, use_cache=False, logits_to_keep=1)
                choice_logits = output.logits[:, -1, self.choice_ids].float()
                choice_log_probs = torch.log_softmax(choice_logits, dim=-1)

            for pair_index in range(len(question_batch)):
                semantic_scores = torch.zeros(len(LABELS), device=choice_log_probs.device)
                for rotation in range(len(CHOICE_TOKENS)):
                    row = choice_log_probs[pair_index * len(CHOICE_TOKENS) + rotation]
                    for label_index in range(len(LABELS)):
                        semantic_scores[label_index] += row[(label_index + rotation) % len(CHOICE_TOKENS)]
                semantic_scores /= len(CHOICE_TOKENS)
                probabilities = torch.softmax(semantic_scores, dim=-1).cpu().tolist()
                label_index = int(torch.argmax(semantic_scores).item())
                quality_gate_applied = _response_is_degenerate(response_batch[pair_index])
                if quality_gate_applied:
                    label_index = LABELS.index("NEITHER")
                score_payload = {
                    label: round(float(probability), 6)
                    for label, probability in zip(LABELS, probabilities, strict=True)
                }
                score_payload["quality_gate_applied"] = quality_gate_applied
                results.append(
                    JudgeResult(
                        label=LABELS[label_index],
                        explanation="Permutation-averaged local choice scores.",
                        raw_response=json.dumps(score_payload, sort_keys=True),
                    )
                )
        return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/steering_exact_h0_gauss_newton")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--judge-mode",
        choices=("rotating_choice", "generated_label"),
        default="rotating_choice",
    )
    parser.add_argument(
        "--seed-label-path",
        default="artifacts/local_llm_rejudge_all_openai_outputs/local_judge_unique_pair_labels.parquet",
        help="Optional compatible local four-way cache. It is used only with --reuse-seed-cache.",
    )
    parser.add_argument(
        "--reuse-seed-cache",
        action="store_true",
        help="Opt in to importing labels from --seed-label-path. Fresh validation leaves this disabled.",
    )
    parser.add_argument(
        "--judge-run-name",
        default=None,
        help="Optional cache/output namespace. By default it is derived from the judge model and prompt.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the mandatory obvious-case judge audit. Intended only for debugging.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _pair_hash(question: str, response: str) -> str:
    return hashlib.sha256(f"{question}\0{response}".encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part)


def _judge_run_name(model: str, judge_mode: str, requested_name: str | None) -> str:
    if requested_name:
        return _safe_component(requested_name)
    if judge_mode == "rotating_choice":
        prompt_material = QUALITY_GATE_VERSION + "\0" + CHOICE_SYSTEM_PROMPT + "\0" + "\0".join(
            _choice_prompt("{QUESTION}", "{RESPONSE}", rotation)
            for rotation in range(len(CHOICE_TOKENS))
        )
    else:
        prompt_material = f"{SYSTEM_PROMPT}\0{LOCAL_PROMPT}"
    prompt_digest = hashlib.sha256(
        prompt_material.encode("utf-8")
    ).hexdigest()[:12]
    return f"{_safe_component(model)}_{_safe_component(judge_mode)}_{prompt_digest}"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pair_hash") and row.get("label") in LABELS:
                rows[str(row["pair_hash"])] = row
    return rows


def _audit_judge(judge: Any, batch_size: int) -> None:
    questions = [row[0] for row in AUDIT_CASES]
    responses = [row[1] for row in AUDIT_CASES]
    expected = [row[2] for row in AUDIT_CASES]
    actual = [
        result.label
        for result in judge.judge_many(questions, responses, batch_size=max(batch_size, 4))
    ]
    failures = [
        {"question": question, "expected": expected_label, "actual": actual_label}
        for question, expected_label, actual_label in zip(questions, expected, actual, strict=True)
        if expected_label != actual_label
    ]
    print(f"judge_audit={len(expected) - len(failures)}/{len(expected)} failures={failures}", flush=True)
    if failures:
        raise RuntimeError(f"Local judge failed mandatory audit: {failures}")


def _model_from_path(path: Path) -> str:
    for name in ("meta_llama_llama_3_1_8b_instruct", "google_gemma_7b_it", "mistralai_mistral_7b_instruct_v0_3"):
        if name in path.parts:
            return name
    return "unknown"


def _summary_row(frame: pd.DataFrame, raw_path: Path, judged_path: Path) -> dict[str, Any]:
    counts = frame["local_judge_label"].value_counts().to_dict()
    n = max(len(frame), 1)
    grounded = int(counts.get("GROUNDED_ACCEPTABLE", 0))
    generic = int(counts.get("GENERIC_ACCEPTABLE", 0))
    unacceptable = int(counts.get("UNACCEPTABLE", 0))
    neither = int(counts.get("NEITHER", 0))
    base_counts = frame["local_base_judge_label"].value_counts().to_dict()
    base_grounded = int(base_counts.get("GROUNDED_ACCEPTABLE", 0))
    base_generic = int(base_counts.get("GENERIC_ACCEPTABLE", 0))

    def first(column: str, default: Any = None) -> Any:
        if column not in frame or frame.empty:
            return default
        values = frame[column].dropna()
        return values.iloc[0] if not values.empty else default

    target_mode = first("target_mode")
    if target_mode is None:
        target_mode = "local_contrast" if "target_local_contrast" in raw_path.parts else "nearest_abstention"

    unique = int(frame["response_text"].fillna("").astype(str).nunique())
    return {
        "dataset": str(first("dataset", raw_path.name.split("__", 1)[0])),
        "model": _model_from_path(raw_path),
        "method": str(first("method", "exact_h0_gauss_newton")),
        "topology_controller": str(first("topology_controller", "free")),
        "direction_source": str(first("direction_source", "observed_groups")),
        "retrieval_feature_mode": str(first("retrieval_feature_mode", "all_layer_mean")),
        "retrieval_geometry": str(first("retrieval_geometry", "standard")),
        "transport_match_mode": str(first("transport_match_mode", "nearest")),
        "transport_prior_ratio": float(first("transport_prior_ratio", float("nan"))),
        "behavior_rank": int(first("behavior_rank", -1)),
        "causal_anchor_ratio": float(first("causal_anchor_ratio", 0.0)),
        "shared_intervention_site": str(first("shared_intervention_site", "layer_output")),
        "topology_decode_mode": str(first("topology_decode_mode", "none")),
        "topology_decode_scale": float(first("topology_decode_scale", 0.0)),
        "topology_decode_suffix_fraction": float(
            first("topology_decode_suffix_fraction", float("nan"))
        ),
        "target_mode": str(target_mode),
        "classifier_target_quantile": float(first("classifier_target_quantile", float("nan"))),
        "neighbor_k": int(first("neighbor_k", -1)),
        "alpha": float(first("alpha", float("nan"))),
        "topology_alpha": float(first("topology_alpha", first("alpha", float("nan")))),
        "mean_alpha": float(first("mean_alpha", float("nan"))),
        "shared_target_ratio": float(first("shared_target_ratio", float("nan"))),
        "shared_scale_mean": float(first("shared_scale_mean", float("nan"))),
        "shared_scale_std": float(first("shared_scale_std", float("nan"))),
        "lambda": float(first("lambda", float("nan"))),
        "damping": float(first("damping", float("nan"))),
        "trust_ratio": float(first("trust_ratio", float("nan"))),
        "n": int(len(frame)),
        "grounded_acceptable": grounded,
        "generic_acceptable": generic,
        "unacceptable": unacceptable,
        "neither": neither,
        "grounded_pct": 100.0 * grounded / n,
        "generic_pct": 100.0 * generic / n,
        "total_acceptable_pct": 100.0 * (grounded + generic) / n,
        "unacceptable_pct": 100.0 * unacceptable / n,
        "neither_pct": 100.0 * neither / n,
        "base_grounded_pct": 100.0 * base_grounded / n,
        "base_generic_pct": 100.0 * base_generic / n,
        "base_total_acceptable_pct": 100.0 * (base_grounded + base_generic) / n,
        "total_acceptable_lift_pct": 100.0
        * ((grounded + generic) - (base_grounded + base_generic))
        / n,
        "unique_responses": unique,
        "unique_pct": 100.0 * unique / n,
        "empty_responses": int(frame["response_text"].fillna("").astype(str).str.strip().eq("").sum()),
        "degenerate_responses": int(
            frame["response_text"].fillna("").astype(str).map(_response_is_degenerate).sum()
        ),
        "relative_hidden_delta_norm_mean": float(
            pd.to_numeric(frame["relative_hidden_delta_norm"], errors="coerce").mean()
        ),
        "relative_centered_hidden_delta_norm_mean": float(
            pd.to_numeric(frame.get("relative_centered_hidden_delta_norm"), errors="coerce").mean()
        )
        if "relative_centered_hidden_delta_norm" in frame
        else float("nan"),
        "relative_shared_hidden_delta_norm_mean": float(
            pd.to_numeric(frame.get("relative_shared_hidden_delta_norm"), errors="coerce").mean()
        )
        if "relative_shared_hidden_delta_norm" in frame
        else float("nan"),
        "delta_token_mean_norm_mean": float(
            pd.to_numeric(frame.get("delta_token_mean_norm"), errors="coerce").mean()
        )
        if "delta_token_mean_norm" in frame
        else float("nan"),
        "zero_mean_constraint": bool(first("zero_mean_constraint", True)),
        "initial_target_error_mean": float(
            pd.to_numeric(frame["initial_normalized_target_error"], errors="coerce").mean()
        ),
        "final_target_error_mean": float(
            pd.to_numeric(frame["final_normalized_target_error"], errors="coerce").mean()
        ),
        "post_intervention_target_error_mean": float(
            pd.to_numeric(
                frame.get("post_intervention_normalized_target_error"),
                errors="coerce",
            ).mean()
        )
        if "post_intervention_normalized_target_error" in frame
        else float("nan"),
        "causal_anchor_applied_norm_mean": float(
            pd.to_numeric(frame.get("causal_anchor_applied_norm"), errors="coerce").mean()
        )
        if "causal_anchor_applied_norm" in frame
        else float("nan"),
        "final_token_behavior_coefficient_mean": float(
            pd.to_numeric(frame.get("final_token_behavior_coefficient"), errors="coerce").mean()
        )
        if "final_token_behavior_coefficient" in frame
        else float("nan"),
        "classifier_current_score_mean": float(
            pd.to_numeric(frame.get("classifier_current_score"), errors="coerce").mean()
        )
        if "classifier_current_score" in frame
        else float("nan"),
        "classifier_z_target_score_mean": float(
            pd.to_numeric(frame.get("classifier_z_target_score"), errors="coerce").mean()
        )
        if "classifier_z_target_score" in frame
        else float("nan"),
        "post_intervention_classifier_score_mean": float(
            pd.to_numeric(frame.get("post_intervention_classifier_score"), errors="coerce").mean()
        )
        if "post_intervention_classifier_score" in frame
        else float("nan"),
        "classifier_target_attainment_pct": float(
            100.0
            * (
                pd.to_numeric(frame["post_intervention_classifier_score"], errors="coerce")
                >= pd.to_numeric(frame["classifier_z_target_score"], errors="coerce")
            ).mean()
        )
        if {
            "post_intervention_classifier_score",
            "classifier_z_target_score",
        }.issubset(frame.columns)
        else float("nan"),
        "raw_path": str(raw_path),
        "judged_path": str(judged_path),
    }


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    judge_run_name = _judge_run_name(args.model, args.judge_mode, args.judge_run_name)
    output_root = ensure_dir(artifact_root / f"_local_fourway_judge_{judge_run_name}")
    cache_path = output_root / "pair_labels.jsonl"
    raw_paths = sorted(artifact_root.rglob("*__exact_h0_gn__raw.parquet"))
    if not raw_paths:
        raise FileNotFoundError(f"No exact-H0 raw outputs under {artifact_root}")

    pairs: dict[str, tuple[str, str]] = {}
    path_frames: dict[Path, pd.DataFrame] = {}
    for raw_path in raw_paths:
        frame = pd.read_parquet(raw_path)
        path_frames[raw_path] = frame
        questions = frame["text"].fillna("").astype(str)
        for response_column in ("response_text", "base_response_text"):
            for question, response in zip(questions, frame[response_column].fillna("").astype(str), strict=True):
                pairs.setdefault(_pair_hash(question, response), (question, response))

    cache = {} if args.force else _load_cache(cache_path)
    seed_path = Path(args.seed_label_path).resolve() if args.seed_label_path else None
    if not args.force and args.reuse_seed_cache and seed_path is not None and seed_path.exists():
        seed = pd.read_parquet(seed_path, columns=["pair_hash", "local_judge_label"])
        for row in seed.itertuples(index=False):
            if row.local_judge_label in LABELS:
                cache.setdefault(
                    str(row.pair_hash),
                    {"pair_hash": str(row.pair_hash), "label": str(row.local_judge_label), "raw_response": ""},
                )
    pending = [(digest, *pair) for digest, pair in pairs.items() if digest not in cache]
    print(
        f"judge_run={judge_run_name} raw_runs={len(raw_paths)} unique_pairs={len(pairs)} "
        f"cached={len(cache)} pending={len(pending)} reuse_seed_cache={args.reuse_seed_cache}",
        flush=True,
    )
    if pending:
        if args.judge_mode == "rotating_choice":
            judge = RotatingChoiceJudge(
                model_name=args.model,
                prompt_max_length=int(args.prompt_max_length),
                batch_size=int(args.batch_size),
            )
        else:
            judge = LocalLLMJudge(
                {
                    "model_name": args.model,
                    "tokenizer_name": args.model,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "torch_dtype": "bfloat16",
                    "device_map": "auto",
                    "use_chat_template": True,
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt_template": LOCAL_PROMPT,
                    "prompt_max_length": int(args.prompt_max_length),
                    "max_new_tokens": int(args.max_new_tokens),
                    "batch_size": int(args.batch_size),
                }
            )
        if not args.skip_audit:
            _audit_judge(judge, int(args.batch_size))
        mode = "w" if args.force else "a"
        with cache_path.open(mode, encoding="utf-8") as handle:
            for start in tqdm(range(0, len(pending), int(args.batch_size)), desc="local four-way judge"):
                batch = pending[start : start + int(args.batch_size)]
                results = judge.judge_many(
                    [row[1] for row in batch],
                    [row[2] for row in batch],
                    batch_size=int(args.batch_size),
                )
                for (digest, _question, _response), result in zip(batch, results, strict=True):
                    label = result.label if result.label in LABELS else "NEITHER"
                    row = {
                        "pair_hash": digest,
                        "label": label,
                        "raw_response": result.raw_response,
                        "judge_model": args.model,
                        "judge_run_name": judge_run_name,
                    }
                    cache[digest] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    summary_rows: list[dict[str, Any]] = []
    for raw_path, frame in path_frames.items():
        hashes = [
            _pair_hash(question, response)
            for question, response in zip(
                frame["text"].fillna("").astype(str),
                frame["response_text"].fillna("").astype(str),
                strict=True,
            )
        ]
        base_hashes = [
            _pair_hash(question, response)
            for question, response in zip(
                frame["text"].fillna("").astype(str),
                frame["base_response_text"].fillna("").astype(str),
                strict=True,
            )
        ]
        frame = frame.copy()
        frame["local_judge_label"] = [cache.get(digest, {}).get("label", "NEITHER") for digest in hashes]
        frame["local_judge_raw_response"] = [cache.get(digest, {}).get("raw_response", "") for digest in hashes]
        frame["local_base_judge_label"] = [cache.get(digest, {}).get("label", "NEITHER") for digest in base_hashes]
        frame["local_judge_model"] = args.model
        frame["local_judge_prompt_variant"] = (
            f"fourway_grounded_illposedness_{args.judge_mode}_local_{QUALITY_GATE_VERSION}"
        )
        frame["local_judge_run_name"] = judge_run_name
        judged_path = raw_path.with_name(
            raw_path.name.replace("__raw.parquet", f"__local_fourway_{judge_run_name}.parquet")
        )
        write_parquet(frame, judged_path)
        frame.to_csv(judged_path.with_suffix(".csv"), index=False)
        summary_rows.append(_summary_row(frame, raw_path, judged_path))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "model", "total_acceptable_pct", "grounded_pct"],
        ascending=[True, True, False, False],
    )
    summary_path = output_root / "exact_h0_gn_local_fourway_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(summary_path.with_suffix(".parquet"), index=False)
    best = summary.groupby(["dataset", "model"], as_index=False).head(1)
    best.to_csv(output_root / "exact_h0_gn_local_fourway_best.csv", index=False)
    print(best.to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
