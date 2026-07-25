#!/usr/bin/env python3
"""Run local ReDeEP/AARF/target-conditioned steering on Dolly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repstat_steering.redeep_generation import (
    ReDeEPGenerationConfig,
    format_llama2_chat_prompt,
    generate_with_redeep,
    load_dolly_examples,
    seed_everything,
    select_examples,
    split_dolly_examples,
    split_dolly_steering_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="NousResearch/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/redeep/datasets/source_info_dolly.jsonl"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/redeep/datasets/response_dolly.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/redeep/dolly_generations.jsonl")
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "fixed_aarf", "minimum_norm"],
        default=["baseline", "fixed_aarf", "minimum_norm"],
    )
    parser.add_argument(
        "--split",
        choices=["development", "heldout", "tuning", "evaluation", "all"],
        default="heldout",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tuning-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-ids", type=int, nargs="*")
    parser.add_argument(
        "--exclude-generations",
        type=Path,
        help="JSONL generations whose source IDs should be skipped",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--target-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--target-score", type=float, default=0.4)
    parser.add_argument("--target-score-shift", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--finite-difference-epsilon", type=float, default=0.05)
    parser.add_argument("--solver-damping", type=float, default=1.0)
    parser.add_argument("--maximum-control-rms", type=float, default=0.30)
    parser.add_argument("--maximum-control-abs", type=float, default=0.60)
    parser.add_argument("--jacobian-refresh-interval", type=int, default=0)
    parser.add_argument("--trigger-threshold", type=float)
    parser.add_argument("--intervention-positions", choices=["last", "all"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval().requires_grad_(False)

    all_examples = load_dolly_examples(args.source, args.responses)
    development, heldout = split_dolly_examples(
        all_examples, test_size=args.test_size, seed=args.seed
    )
    tuning, evaluation = split_dolly_steering_evaluation(
        all_examples,
        tuning_size=args.tuning_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_examples = {
        "development": development,
        "heldout": heldout,
        "tuning": tuning,
        "evaluation": evaluation,
        "all": all_examples,
    }[args.split]
    if args.exclude_generations is not None:
        excluded_ids = {
            int(json.loads(line)["source_id"])
            for line in args.exclude_generations.open(encoding="utf-8")
        }
        split_examples = [
            example
            for example in split_examples
            if example.source_id not in excluded_ids
        ]
    examples = select_examples(
        split_examples,
        offset=args.offset,
        limit=args.limit,
        source_ids=args.source_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for example_index, example in enumerate(examples):
            chat_prompt = format_llama2_chat_prompt(tokenizer, example.prompt)
            for method in args.methods:
                config = ReDeEPGenerationConfig(
                    method=method,
                    max_new_tokens=args.max_new_tokens,
                    target_mode=args.target_mode,
                    target_score=args.target_score,
                    target_score_shift=args.target_score_shift,
                    ridge=args.ridge,
                    finite_difference_epsilon=args.finite_difference_epsilon,
                    solver_damping=args.solver_damping,
                    maximum_control_rms=args.maximum_control_rms,
                    maximum_control_abs=args.maximum_control_abs,
                    jacobian_refresh_interval=args.jacobian_refresh_interval,
                    trigger_threshold=args.trigger_threshold,
                    intervention_positions=args.intervention_positions,
                )
                response, diagnostics = generate_with_redeep(
                    model, tokenizer, chat_prompt, config
                )
                row = {
                    "split": args.split,
                    "example_index": example_index,
                    "source_id": example.source_id,
                    "question": example.question,
                    "passage": example.passage,
                    "reference_answer": example.reference_answer,
                    "original_hallucination_label": example.hallucination_label,
                    "method": method,
                    "response": response,
                    "diagnostics": diagnostics,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(
                    f"[{example_index + 1}/{len(examples)}] {example.source_id=} "
                    f"{method=} tokens={diagnostics['generated_tokens']} "
                    f"trigger={diagnostics['trigger_rate']:.3f} "
                    f"hidden_delta={diagnostics['mean_final_hidden_relative_change']:.5f}"
                )

    summary_rows = []
    for method, method_rows in pd.DataFrame(rows).groupby("method"):
        diagnostics = list(method_rows["diagnostics"])
        summary_rows.append(
            {
                "method": method,
                "examples": len(method_rows),
                "mean_trigger_rate": sum(d["trigger_rate"] for d in diagnostics) / len(diagnostics),
                "mean_control_rms": sum(d["mean_control_rms"] for d in diagnostics) / len(diagnostics),
                "mean_final_hidden_relative_change": sum(
                    d["mean_final_hidden_relative_change"] for d in diagnostics
                ) / len(diagnostics),
                "mean_baseline_detector_score": sum(
                    d["mean_baseline_detector_score"] for d in diagnostics
                ) / len(diagnostics),
                "mean_controlled_detector_score": sum(
                    d["mean_controlled_detector_score"] for d in diagnostics
                ) / len(diagnostics),
            }
        )
    summary_path = args.output.with_suffix(".summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
