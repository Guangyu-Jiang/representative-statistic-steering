#!/usr/bin/env python3
"""Evaluate Lookback Lens guided and minimum-norm control on Natural Questions."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from repstat_steering.lookback_control import (
    LookbackGenerationConfig,
    LookbackNQExperiment,
    load_nq_examples,
    summarize_lookback_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee",
    )
    parser.add_argument(
        "--data",
        default="external/Lookback-Lens/data/nq-open-10_total_documents_gold_at_4.jsonl.gz",
    )
    parser.add_argument(
        "--classifier",
        default="external/Lookback-Lens/classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "baseline",
            "baseline_rerank",
            "guided",
            "minimum_norm",
            "minimum_norm_rerank",
        ],
        default=["baseline", "minimum_norm"],
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument(
        "--target-mode", choices=["absolute", "relative"], default="relative"
    )
    parser.add_argument("--target-probability", type=float, default=0.95)
    parser.add_argument("--target-logit-shift", type=float, default=1.0)
    parser.add_argument("--control-trigger-probability", type=float, default=1.0)
    parser.add_argument("--high-confidence-logit-shift", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--solver-steps", type=int, default=8)
    parser.add_argument("--solver-damping", type=float, default=1.0)
    parser.add_argument("--maximum-bias-rms", type=float, default=0.5)
    parser.add_argument(
        "--context-bias-mode",
        choices=[
            "uniform",
            "top_attention",
            "question_overlap",
            "question_top_union",
            "random_overlap",
            "retrieved_passage",
            "retrieved_sentence",
        ],
        default="uniform",
    )
    parser.add_argument("--context-top-fraction", type=float, default=0.25)
    parser.add_argument("--context-overlap-radius", type=int, default=8)
    parser.add_argument("--active-control-count", type=int, default=0)
    parser.add_argument(
        "--bias-constraint",
        choices=["unrestricted", "nonnegative"],
        default="unrestricted",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_nq_examples(
        args.data, offset=args.offset, limit=args.limit, indices=args.indices
    )
    base_config = LookbackGenerationConfig(
        method=args.methods[0],
        max_new_tokens=args.max_new_tokens,
        window_size=args.window_size,
        target_mode=args.target_mode,
        target_probability=args.target_probability,
        target_logit_shift=args.target_logit_shift,
        control_trigger_probability=args.control_trigger_probability,
        high_confidence_logit_shift=args.high_confidence_logit_shift,
        ridge=args.ridge,
        solver_steps=args.solver_steps,
        solver_damping=args.solver_damping,
        maximum_bias_rms=args.maximum_bias_rms,
        context_bias_mode=args.context_bias_mode,
        context_top_fraction=args.context_top_fraction,
        context_overlap_radius=args.context_overlap_radius,
        active_control_count=args.active_control_count,
        bias_constraint=args.bias_constraint,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        chunk_size=args.chunk_size,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )
    metadata = vars(args) | {"n_examples": len(examples)}
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))

    result_path = output_dir / "results.jsonl"
    completed: set[tuple[int, str]] = set()
    rows: list[dict] = []
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            completed.add((row["dataset_index"], row["method"]))

    experiment = LookbackNQExperiment(
        args.model,
        args.classifier,
        device=args.device,
        model_dtype=args.model_dtype,
    )
    try:
        total = len(examples) * len(args.methods)
        progress = len(completed)
        for example in examples:
            for method in args.methods:
                if (example.dataset_index, method) in completed:
                    continue
                config = replace(base_config, method=method)
                started_at = time.perf_counter()
                row = experiment.evaluate(example, config)
                row["generation_seconds"] = time.perf_counter() - started_at
                rows.append(row)
                with result_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                progress += 1
                print(
                    f"[{progress}/{total}] index={example.dataset_index} method={method} "
                    f"em={row['exact_match']:.0f} factual={row['mean_factual_probability']:.3f} "
                    f"bias_rms={row['mean_bias_rms']:.4f} output_kl={row['mean_output_kl']:.4f}",
                    flush=True,
                )
    finally:
        experiment.close()

    summaries = summarize_lookback_rows(rows)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
