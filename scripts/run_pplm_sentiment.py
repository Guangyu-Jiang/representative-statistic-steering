#!/usr/bin/env python3
"""Run matched PPLM and minimum-norm sentiment-control experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from repstat_steering.pplm_control import (
    PPLMGenerationConfig,
    PPLMSentimentExperiment,
    distinct_ngram_fraction,
)
from repstat_steering.pplm_quantiles import load_quantile_margin_targets


DEFAULT_PREFIXES = [
    "The lake",
    "The chicken",
    "The country",
    "The painting",
    "The movie",
    "The book",
    "The pizza",
    "The potato",
    "The city",
    "The president",
    "The company",
    "The game",
    "The restaurant",
    "The weather",
    "The conversation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/gpt2-medium")
    parser.add_argument("--classifier", default="checkpoints/SST_classifier_head.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["baseline", "pplm", "minimum_norm"])
    parser.add_argument("--targets", nargs="+", default=["positive", "negative"])
    parser.add_argument("--prefixes", nargs="*", default=DEFAULT_PREFIXES)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon-length", type=int, default=1)
    parser.add_argument("--gm-scale", type=float, default=0.95)
    parser.add_argument("--kl-scale", type=float, default=0.01)
    parser.add_argument("--target-probability", type=float, default=0.8)
    parser.add_argument("--target-margin-shift", type=float)
    parser.add_argument("--quantile-targets", type=Path)
    parser.add_argument("--target-quantile", type=float)
    parser.add_argument("--minimum-target-probability", type=float)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--minimum-norm-steps", type=int, default=3)
    parser.add_argument("--minimum-norm-damping", type=float, default=1.0)
    parser.add_argument("--maximum-relative-norm", type=float, default=0.10)
    parser.add_argument("--maximum-token-kl", type=float)
    parser.add_argument("--difficult-margin-threshold", type=float)
    parser.add_argument("--difficult-maximum-token-kl", type=float)
    parser.add_argument("--cache-component", choices=["all", "key", "value"], default="all")
    parser.add_argument("--cache-last-n-layers", type=int)
    parser.add_argument(
        "--statistic-mode", choices=["margin", "distribution"], default="margin"
    )
    parser.add_argument("--gradient-block-normalization", type=float, default=0.0)
    parser.add_argument("--preserve-top-log-probs", type=int, default=0)
    parser.add_argument("--log-probability-preservation-weight", type=float, default=1.0)
    parser.add_argument("--persistent-cache", action="store_true")
    parser.add_argument("--pplm-steps", type=int, default=5)
    parser.add_argument("--pplm-step-size", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.quantile_targets is None) != (args.target_quantile is None):
        raise ValueError("quantile_targets and target_quantile must be set together")
    if args.quantile_targets is not None and args.target_margin_shift is not None:
        raise ValueError("quantile targets cannot be combined with target_margin_shift")
    quantile_margins = (
        load_quantile_margin_targets(args.quantile_targets, args.target_quantile)
        if args.quantile_targets is not None
        else {}
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    experiment = PPLMSentimentExperiment(args.model, args.classifier, args.device)
    rows = []
    completed = set()
    result_path = output_dir / "generations.jsonl"
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["method"],
                row["target_label"],
                row["prefix"],
                int(row["seed"]),
            )
            if key in completed:
                continue
            rows.append(row)
            completed.add(key)
    for method in args.methods:
        base_config = PPLMGenerationConfig(
            method=method,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=not args.greedy,
            horizon_length=args.horizon_length,
            gm_scale=args.gm_scale,
            kl_scale=args.kl_scale,
            target_probability=args.target_probability,
            target_margin_shift=args.target_margin_shift,
            minimum_target_probability=args.minimum_target_probability,
            ridge=args.ridge,
            minimum_norm_steps=args.minimum_norm_steps,
            minimum_norm_damping=args.minimum_norm_damping,
            maximum_relative_norm=args.maximum_relative_norm,
            maximum_token_kl=args.maximum_token_kl,
            difficult_margin_threshold=args.difficult_margin_threshold,
            difficult_maximum_token_kl=args.difficult_maximum_token_kl,
            cache_component=args.cache_component,
            cache_last_n_layers=args.cache_last_n_layers,
            statistic_mode=args.statistic_mode,
            gradient_block_normalization=args.gradient_block_normalization,
            preserve_top_log_probs=args.preserve_top_log_probs,
            log_probability_preservation_weight=args.log_probability_preservation_weight,
            persistent_cache=args.persistent_cache,
            pplm_steps=args.pplm_steps,
            pplm_step_size=args.pplm_step_size,
        )
        for target in args.targets:
            config = replace(
                base_config,
                target_margin=(
                    quantile_margins[target]
                    if method == "minimum_norm" and quantile_margins
                    else None
                ),
            )
            for prefix_index, prefix in enumerate(args.prefixes):
                for seed in args.seeds:
                    effective_seed = seed + 1000 * prefix_index
                    key = (method, target, prefix, effective_seed)
                    if key in completed:
                        continue
                    result = experiment.generate(
                        prefix, target, config, effective_seed
                    )
                    rows.append(result)
                    completed.add(key)
                    with result_path.open("a") as handle:
                        handle.write(json.dumps(result) + "\n")
                    print(
                        method,
                        target,
                        prefix,
                        result[f"{target}_probability"],
                        result["mean_relative_cache_change"],
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "generations.csv", index=False)
    summaries = []
    for (method, target), group in frame.groupby(["method", "target_label"]):
        probability_column = f"{target}_probability"
        summaries.append(
            {
                "method": method,
                "target_label": target,
                "n": len(group),
                "mean_target_probability": group[probability_column].mean(),
                "target_probability_ge_0p5": (group[probability_column] >= 0.5).mean(),
                "mean_relative_cache_change": group["mean_relative_cache_change"].mean(),
                "mean_mix_scale": group["mean_mix_scale"].mean(),
                "mean_token_kl": group["mean_token_kl"].mean(),
                "mean_raw_token_kl": group["mean_raw_token_kl"].mean(),
                "mean_token_kl_budget": group["mean_token_kl_budget"].mean(),
                "distinct_1": distinct_ngram_fraction(group["continuation"], 1),
                "distinct_2": distinct_ngram_fraction(group["continuation"], 2),
                "distinct_3": distinct_ngram_fraction(group["continuation"], 3),
            }
        )
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
