#!/usr/bin/env python3
"""Evaluate original and minimum-norm TruthX on TruthfulQA multiple choice."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from repstat_steering.truthx_control import (
    TruthXInterventionConfig,
    TruthXMultipleChoiceExperiment,
    summarize_truthx_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="NousResearch/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--checkpoint-fold1",
        default="checkpoints/truthx/Llama-2-7b-chat-hf/truthx_model.fold1.pt",
    )
    parser.add_argument(
        "--checkpoint-fold2",
        default="checkpoints/truthx/Llama-2-7b-chat-hf/truthx_model.fold2.pt",
    )
    parser.add_argument("--official-truthx", default="external/TruthX/truthx.py")
    parser.add_argument("--fold-yaml", default="external/TruthX/data/truthfulqa_data_fold1.yaml")
    parser.add_argument("--data", default="external/TruthX/data/TruthfulQA.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method", choices=["baseline", "truthx_original", "minimum_norm"], required=True
    )
    parser.add_argument("--top-modules", type=int, default=10)
    parser.add_argument("--original-strength", type=float, default=4.5)
    parser.add_argument(
        "--target-mode",
        choices=[
            "positive_center",
            "centroid_shift",
            "cosine_margin",
            "cosine_margin_decoder",
            "cosine_margin_shift_decoder",
        ],
        default="positive_center",
    )
    parser.add_argument("--target-strength", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--optimization-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--maximum-relative-norm", type=float, default=0.10)
    parser.add_argument("--intervention-margin-threshold", type=float)
    parser.add_argument("--directional-backtracking-steps", type=int, default=0)
    parser.add_argument("--directional-nonnegative", action="store_true")
    parser.add_argument("--two-sided", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TruthXInterventionConfig(
        method=args.method,
        solver_version=(
            "accumulated_linesearch_ray_v4"
            if args.directional_backtracking_steps > 0
            and args.directional_nonnegative
            else (
                "accumulated_linesearch_v3"
                if args.directional_backtracking_steps > 0
                else (
                    "accumulated_ray_v3"
                    if args.directional_nonnegative
                    else "accumulated_v2"
                )
            )
        ),
        top_modules=args.top_modules,
        original_strength=args.original_strength,
        target_mode=args.target_mode,
        target_strength=args.target_strength,
        ridge=args.ridge,
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        maximum_relative_norm=args.maximum_relative_norm,
        one_sided=not args.two_sided,
        intervention_margin_threshold=args.intervention_margin_threshold,
        directional_backtracking_steps=args.directional_backtracking_steps,
        directional_nonnegative=args.directional_nonnegative,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"intervention": asdict(config)}, indent=2)
    )
    data = pd.read_csv(args.data)
    if args.indices:
        indices = args.indices
    else:
        stop = len(data) if args.limit <= 0 else min(len(data), args.offset + args.limit)
        indices = list(range(args.offset, stop))

    experiment = TruthXMultipleChoiceExperiment(
        args.model,
        (args.checkpoint_fold1, args.checkpoint_fold2),
        args.official_truthx,
        args.fold_yaml,
        config,
        args.device,
        args.model_dtype,
    )
    result_path = output_dir / "results.jsonl"
    rows = []
    completed = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_index = int(row["dataset_index"])
            if dataset_index in completed:
                continue
            rows.append(row)
            completed.add(dataset_index)
    try:
        progress = len(completed.intersection(indices))
        for dataset_index in indices:
            if dataset_index in completed:
                continue
            result = experiment.evaluate_question(dataset_index, data.iloc[dataset_index])
            rows.append(result)
            completed.add(dataset_index)
            with result_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            progress += 1
            print(
                f"[{progress}/{len(indices)}] index={dataset_index} "
                f"mc1={result['mc1']:.0f} mc2={result['mc2']:.4f} "
                f"rel={result['mean_relative_action_norm']:.5f}",
                flush=True,
            )
    finally:
        experiment.close()

    summary = summarize_truthx_rows(rows, config)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
