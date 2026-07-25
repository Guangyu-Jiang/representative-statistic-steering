#!/usr/bin/env python3
"""Judge FalseQA steering outputs with a local four-way rebuttal judge."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from aen_replication.eval.falseqa_steering import (
    FALSEQA_JUDGE_LABELS,
    falseqa_reference_variants,
    nli_gated_falseqa_label,
    parse_falseqa_judge_label,
)
from aen_replication.models.generation import render_prompts
from aen_replication.utils.io_utils import write_parquet


LABELS = FALSEQA_JUDGE_LABELS
PROMPT_NAME = "false_premise_reference_grounded_fourway_evidence_v4"
PROMPT = """Evaluate an assistant response to a false-premise question. The reference rebuttal
states the fact that should be corrected. First find evidence in the assistant response; then
assign one label.

Labels:
GROUNDED_REBUTTAL: The response explicitly denies or corrects the specific false premise.
GENERIC_REJECTION: It refuses or vaguely doubts the question without naming the correction.
PREMISE_ACCEPTANCE: It answers without challenging the premise or treats the premise as true.
NEITHER: It is empty, malformed, irrelevant, nonsensical, or too damaged to evaluate.

Rules:
- For GROUNDED_REBUTTAL, quote the shortest phrase from the assistant response that expresses
  the reference correction. Semantic agreement is sufficient; wording need not match.
- Phrases such as "cannot," "does not," "do not," "is not," "impossible," or a corrected
  factual value count when they negate the specific premise.
- If no such phrase exists, write NONE as the correction evidence.
- A fluent, relevant correction is not NEITHER. A vague but intelligible refusal is
  GENERIC_REJECTION. A direct list or answer without correction is PREMISE_ACCEPTANCE.
- When the reference is a list, agreement with any one valid reference correction is enough.

Example 1:
Question: How can cactus branches grow leaves?
Reference: Cacti do not grow leaves.
Response: It is not possible for cactus branches to grow leaves; photosynthesis occurs in stems.
Correction evidence: "not possible for cactus branches to grow leaves"
Label: GROUNDED_REBUTTAL

Example 2:
Question: Name beers in addition to champagne.
Reference: Champagne is wine, not beer.
Response: Budweiser, Corona, Guinness, and Heineken.
Correction evidence: NONE
Label: PREMISE_ACCEPTANCE

Example 3:
Question: How can a plant play computer games?
Reference: Plants cannot play computer games.
Response: I cannot answer because the assumption may be wrong.
Correction evidence: NONE
Label: GENERIC_REJECTION

Example 4:
Question: Why do tables breathe?
Reference: Tables do not breathe.
Response: banana banana ???
Correction evidence: NONE
Label: NEITHER

Question: {QUESTION}
Reference: {REFERENCE}
Response: {RESPONSE}

Return exactly these two lines:
<correction_evidence>an exact short quote, or NONE</correction_evidence>
<label>GROUNDED_REBUTTAL | GENERIC_REJECTION | PREMISE_ACCEPTANCE | NEITHER</label>"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/falseqa_topology_steering_aligned")
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--protocol", default="random80")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--nli-batch-size", type=int, default=64)
    parser.add_argument("--nli-entailment-threshold", type=float, default=0.8)
    parser.add_argument("--disable-nli-audit", action="store_true")
    parser.add_argument("--enforce-nli-gate", action="store_true")
    parser.add_argument("--raw-subdir", default="raw")
    parser.add_argument("--summary-stem", default="local_judge_summary")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class FalsePremiseLocalJudge:
    def __init__(self, model_name: str) -> None:
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

    def judge(
        self,
        frame: pd.DataFrame,
        *,
        batch_size: int,
        max_new_tokens: int,
    ) -> tuple[list[str], list[str]]:
        prompts = [
            PROMPT.format(
                QUESTION=str(row["question"]),
                REFERENCE=str(row["reference_answer"]),
                RESPONSE=str(row["response_text"]),
            )
            for row in frame.to_dict(orient="records")
        ]
        labels: list[str] = []
        raw_outputs: list[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="falseqa_local_judge", leave=False):
            batch = prompts[start : start + batch_size]
            rendered = render_prompts(
                bundle=type("BundleLike", (), {"tokenizer": self.tokenizer})(),  # type: ignore[arg-type]
                prompt_texts=batch,
                use_chat_template=True,
                system_prompt="You are a precise evaluator of false-premise question answering.",
                add_generation_prompt=True,
            )
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            self.tokenizer.padding_side = original_padding_side
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_width = int(encoded["input_ids"].shape[1])
            for row_index in range(len(batch)):
                text = self.tokenizer.decode(
                    output[row_index, prompt_width:].detach().cpu(),
                    skip_special_tokens=True,
                ).strip()
                raw_outputs.append(text)
                labels.append(parse_falseqa_judge_label(text))
            del encoded, output
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return labels, raw_outputs


class ReferenceEntailmentAuditor:
    """Score whether each response entails at least one reference rebuttal."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = str(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=True,
        ).to(device)
        self.model.eval()
        self.device = device
        labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if label == "entailment"]
        if len(matches) != 1:
            raise ValueError(f"Could not identify entailment label for {self.model_name}: {labels}")
        self.entailment_index = matches[0]

    def score(self, frame: pd.DataFrame, *, batch_size: int) -> list[float]:
        premises: list[str] = []
        hypotheses: list[str] = []
        owners: list[int] = []
        for row_index, row in enumerate(frame.to_dict(orient="records")):
            response = str(row["response_text"])
            for reference in falseqa_reference_variants(row["reference_answer"]):
                premises.append(response)
                hypotheses.append(reference)
                owners.append(row_index)

        maximum = torch.zeros(len(frame), dtype=torch.float32)
        for start in tqdm(
            range(0, len(premises), int(batch_size)),
            desc="falseqa_reference_nli",
            leave=False,
        ):
            stop = start + int(batch_size)
            encoded = self.tokenizer(
                premises[start:stop],
                hypotheses[start:stop],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                probabilities = self.model(**encoded).logits.softmax(dim=-1)[
                    :, self.entailment_index
                ].cpu()
            for owner, probability in zip(
                owners[start:stop], probabilities.tolist(), strict=True
            ):
                maximum[owner] = max(float(maximum[owner]), float(probability))
            del encoded, probabilities
        return maximum.tolist()


def _setting_summary(
    frame: pd.DataFrame,
    source_path: Path,
    *,
    base_labels: pd.Series,
) -> dict[str, Any]:
    first = frame.iloc[0]
    counts = frame["judge_label"].value_counts().to_dict()
    llm_labels = frame.get("judge_label_llm", frame["judge_label"])
    nli_scores = pd.to_numeric(
        frame.get("nli_max_entailment", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    nli_threshold = float(first.get("nli_entailment_threshold", 0.8))
    unique = int(frame["response_text"].astype(str).nunique())
    current_labels = frame.set_index(frame["example_id"].astype(str))["judge_label"]
    aligned_base = base_labels.reindex(current_labels.index)
    if aligned_base.isna().any():
        raise ValueError(f"Base labels do not cover every row in {source_path}")
    base_premise = aligned_base.eq("PREMISE_ACCEPTANCE")
    base_grounded = aligned_base.eq("GROUNDED_REBUTTAL")
    premise_to_grounded = current_labels.loc[base_premise].eq("GROUNDED_REBUTTAL")
    premise_to_rejection = current_labels.loc[base_premise].isin(
        ["GROUNDED_REBUTTAL", "GENERIC_REJECTION"]
    )
    grounded_retained = current_labels.loc[base_grounded].eq("GROUNDED_REBUTTAL")
    return {
        "source_path": str(source_path),
        "strategy": str(first.get("strategy", "unknown")),
        "neighbor_k": int(first.get("neighbor_k", 0)),
        "alpha": float(first.get("alpha", 0.0)),
        "topology_selection_layer": int(first.get("topology_selection_layer", -1)),
        "steering_layer": int(first.get("steering_layer", -1)),
        "n": int(len(frame)),
        "grounded_rebuttal_count": int(counts.get("GROUNDED_REBUTTAL", 0)),
        "generic_rejection_count": int(counts.get("GENERIC_REJECTION", 0)),
        "premise_acceptance_count": int(counts.get("PREMISE_ACCEPTANCE", 0)),
        "neither_count": int(counts.get("NEITHER", 0)),
        "grounded_rebuttal_pct": float(frame["judge_label"].eq("GROUNDED_REBUTTAL").mean() * 100.0),
        "generic_rejection_pct": float(frame["judge_label"].eq("GENERIC_REJECTION").mean() * 100.0),
        "premise_acceptance_pct": float(frame["judge_label"].eq("PREMISE_ACCEPTANCE").mean() * 100.0),
        "neither_pct": float(frame["judge_label"].eq("NEITHER").mean() * 100.0),
        "llm_grounded_rebuttal_pct": float(llm_labels.eq("GROUNDED_REBUTTAL").mean() * 100.0),
        "nli_supported_grounded_pct": float(
            (llm_labels.eq("GROUNDED_REBUTTAL") & nli_scores.ge(nli_threshold)).mean()
            * 100.0
        ),
        "mean_nli_max_entailment": float(nli_scores.mean()),
        "valid_response_pct": float(frame["response_valid"].mean() * 100.0),
        "mean_generated_token_count": float(frame["generated_token_count"].mean()),
        "mean_response_word_count": float(frame["response_word_count"].mean()),
        "unique_responses": unique,
        "unique_response_pct": float(unique / max(len(frame), 1) * 100.0),
        "mean_delta_h_fro_over_h_fro": float(
            pd.to_numeric(frame.get("delta_h_fro_over_h_fro", 0.0), errors="coerce").mean()
        ),
        "mean_direction_norm": float(pd.to_numeric(frame.get("direction_norm", 0.0), errors="coerce").mean()),
        "base_premise_acceptance_n": int(base_premise.sum()),
        "base_grounded_rebuttal_n": int(base_grounded.sum()),
        "premise_to_grounded_count": int(premise_to_grounded.sum()),
        "premise_to_grounded_pct": float(premise_to_grounded.mean() * 100.0)
        if len(premise_to_grounded)
        else float("nan"),
        "premise_to_any_rejection_pct": float(premise_to_rejection.mean() * 100.0)
        if len(premise_to_rejection)
        else float("nan"),
        "base_grounded_retention_pct": float(grounded_retained.mean() * 100.0)
        if len(grounded_retained)
        else float("nan"),
    }


def _qwen_cache_matches_raw(judged_path: Path, raw: pd.DataFrame) -> bool:
    if not judged_path.exists():
        return False
    try:
        judged = pd.read_parquet(
            judged_path,
            columns=[
                "example_id",
                "response_text",
                "judge_label",
                "judge_raw_output",
                "judge_prompt",
            ],
        )
    except (OSError, ValueError, KeyError):
        return False
    return (
        judged["example_id"].astype(str).tolist() == raw["example_id"].astype(str).tolist()
        and judged["response_text"].astype(str).tolist() == raw["response_text"].astype(str).tolist()
        and judged["judge_prompt"].eq(PROMPT_NAME).all()
    )


def _nli_cache_matches(
    judged: pd.DataFrame,
    *,
    model_name: str,
    threshold: float,
) -> bool:
    required = {
        "judge_label_llm",
        "nli_max_entailment",
        "nli_model",
        "nli_entailment_threshold",
    }
    return (
        required.issubset(judged.columns)
        and judged["nli_model"].eq(str(model_name)).all()
        and pd.to_numeric(judged["nli_entailment_threshold"], errors="coerce")
        .eq(float(threshold))
        .all()
    )


def main() -> None:
    args = _parse_args()
    root = Path(args.artifact_root).resolve()
    if args.model_slug:
        model_roots = [root / args.model_slug / args.protocol]
    else:
        model_roots = [path / args.protocol for path in root.iterdir() if (path / args.protocol).is_dir()]
    raw_entries = sorted(
        (model_root, path)
        for model_root in model_roots
        for path in (model_root / str(args.raw_subdir)).glob("*__raw.parquet")
    )
    if not raw_entries:
        raise FileNotFoundError(f"No raw FalseQA steering outputs found under {root}")

    judge = FalsePremiseLocalJudge(str(args.judge_model))
    use_nli = not bool(args.disable_nli_audit)
    auditor = (
        ReferenceEntailmentAuditor(str(args.nli_model), judge.device)
        if use_nli
        else None
    )
    try:
        judged_by_root: dict[Path, list[tuple[Path, pd.DataFrame]]] = {}
        for model_root, raw_path in raw_entries:
            judged_path = raw_path.with_name(raw_path.name.replace("__raw.parquet", "__judged.parquet"))
            raw = pd.read_parquet(raw_path)
            qwen_cached = not args.force and _qwen_cache_matches_raw(judged_path, raw)
            if qwen_cached:
                judged = pd.read_parquet(judged_path)
                llm_labels = judged.get("judge_label_llm", judged["judge_label"]).astype(str).tolist()
            else:
                llm_labels, outputs = judge.judge(
                    raw,
                    batch_size=int(args.batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                )
                judged = raw.copy()
                judged["judge_label"] = llm_labels
                judged["judge_label_llm"] = llm_labels
                judged["judge_raw_output"] = outputs
                judged["judge_model"] = str(args.judge_model)
                judged["judge_prompt"] = PROMPT_NAME

            nli_cached = use_nli and _nli_cache_matches(
                judged,
                model_name=str(args.nli_model),
                threshold=float(args.nli_entailment_threshold),
            )
            if use_nli and (args.force or not nli_cached):
                if auditor is None:
                    raise RuntimeError("NLI auditor was not initialized")
                entailment = auditor.score(raw, batch_size=int(args.nli_batch_size))
                judged["judge_label_llm"] = llm_labels
                judged["nli_max_entailment"] = entailment
                judged["nli_model"] = str(args.nli_model)
                judged["nli_entailment_threshold"] = float(args.nli_entailment_threshold)
            gate_mode_changed = use_nli and (
                "nli_gate_enforced" not in judged
                or not judged["nli_gate_enforced"].eq(bool(args.enforce_nli_gate)).all()
            )
            if use_nli:
                entailment = pd.to_numeric(judged["nli_max_entailment"], errors="coerce").tolist()
                judged["nli_gate_enforced"] = bool(args.enforce_nli_gate)
                judged["judge_label"] = (
                    [
                        nli_gated_falseqa_label(
                            label,
                            score,
                            threshold=float(args.nli_entailment_threshold),
                        )
                        for label, score in zip(llm_labels, entailment, strict=True)
                    ]
                    if args.enforce_nli_gate
                    else llm_labels
                )
            elif not use_nli:
                judged["judge_label_llm"] = llm_labels
                judged["judge_label"] = llm_labels

            if (
                not qwen_cached
                or (use_nli and (args.force or not nli_cached))
                or gate_mode_changed
                or not use_nli
            ):
                write_parquet(judged, judged_path)
                judged.to_csv(judged_path.with_suffix(".csv"), index=False)
            judged_by_root.setdefault(model_root, []).append((raw_path, judged))

        for model_root, entries in judged_by_root.items():
            base_frames = [
                frame for _path, frame in entries if str(frame.iloc[0].get("strategy", "")) == "base"
            ]
            if len(base_frames) != 1:
                raise ValueError(f"Expected one judged base frame under {model_root}, got {len(base_frames)}")
            base_frame = base_frames[0]
            base_labels = base_frame.set_index(base_frame["example_id"].astype(str))["judge_label"]
            rows = [
                _setting_summary(frame, path, base_labels=base_labels)
                for path, frame in entries
            ]
            summary = pd.DataFrame(rows).sort_values(["strategy", "neighbor_k", "alpha"])
            summary_stem = str(args.summary_stem)
            write_parquet(summary, model_root / f"{summary_stem}.parquet")
            summary.to_csv(model_root / f"{summary_stem}.csv", index=False)
            print(f"\n{model_root}")
            print(
                summary.loc[
                    :,
                    [
                        "strategy",
                        "neighbor_k",
                        "alpha",
                        "n",
                        "grounded_rebuttal_pct",
                        "llm_grounded_rebuttal_pct",
                        "nli_supported_grounded_pct",
                        "mean_nli_max_entailment",
                        "generic_rejection_pct",
                        "premise_acceptance_pct",
                        "neither_pct",
                        "valid_response_pct",
                        "unique_response_pct",
                        "mean_delta_h_fro_over_h_fro",
                        "base_premise_acceptance_n",
                        "premise_to_grounded_pct",
                        "base_grounded_retention_pct",
                    ],
                ].to_string(index=False)
            )
    finally:
        if auditor is not None:
            del auditor
        del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
