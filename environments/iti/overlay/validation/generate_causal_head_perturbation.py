#!/usr/bin/env python3
"""Generate TruthfulQA answers with online causal attention-head control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import load_truthfulqa_frame
from truthfulqa import utilities
from validate_causal_head_perturbation import (
    AttentionHeadController,
    DIAGNOSTIC_FIELDS,
    Setting,
    build_aggregate_actions,
    configure_controllers,
    diagnostics,
    layer_indices,
    load_statistics,
)
from validate_margin_perturbation import DEFAULT_PREFIX


LAST_POSITION_STOP = 10**9
ONLINE_METHODS = {
    "fixed_com",
    "aggregate_com",
    "aggregate_probe",
    "targeted_iti",
    "targeted_probe_iti",
    "bounded_targeted_probe_iti",
    "headwise_probe_iti",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--statistics-cache", type=Path, required=True)
    parser.add_argument("--settings-file", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--eval-split", choices=("validation", "test"), default="test")
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="reuse matching completed answer/diagnostic columns by dataset_index",
    )
    return parser.parse_args()


def decode_answer(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    text = text.split("Q:", maxsplit=1)[0].strip()
    if "A:" in text:
        text = text.split("A:", maxsplit=1)[-1].strip()
    return text


def collect_last_position(
    controllers: dict[int, AttentionHeadController],
    bank,
) -> None:
    grouped = layer_indices(bank)
    for layer, controller in controllers.items():
        controller.collect(
            [bank.top_heads[index][1] for index in grouped[layer]],
            -1,
            LAST_POSITION_STOP,
        )


def fork_dynamic_cache(cache: DynamicCache | None) -> DynamicCache | None:
    """Fork cache metadata while sharing immutable tensors from past tokens."""

    if cache is None:
        return None
    fork = DynamicCache()
    fork._seen_tokens = cache._seen_tokens
    fork.key_cache = list(cache.key_cache)
    fork.value_cache = list(cache.value_cache)
    return fork


def reuse_generation_columns(
    output: pd.DataFrame,
    prior: pd.DataFrame,
    tags: list[str],
) -> pd.DataFrame:
    """Copy matching generated columns from a row-aligned prior artifact."""

    if prior["dataset_index"].duplicated().any():
        raise ValueError("Reuse artifact has duplicate dataset_index values")
    prior_by_id = prior.set_index("dataset_index")
    missing_ids = sorted(set(output["dataset_index"]).difference(prior_by_id.index))
    if missing_ids:
        raise ValueError(f"Reuse artifact is missing {len(missing_ids)} requested rows")
    result = output.copy()
    for tag in tags:
        prefix = f"{tag}_"
        columns = [column for column in prior if column.startswith(prefix)]
        for column in columns:
            result[column] = result["dataset_index"].map(prior_by_id[column])
    return result


def configure_precomputed_last(
    controllers: dict[int, AttentionHeadController],
    bank,
    actions: dict[int, torch.Tensor],
) -> None:
    grouped = layer_indices(bank)
    for layer, controller in controllers.items():
        controller.precomputed(
            actions[layer],
            [bank.top_heads[index][1] for index in grouped[layer]],
            -1,
            LAST_POSITION_STOP,
        )


def greedy_generate(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    controllers: dict[int, AttentionHeadController],
    bank,
    setting: Setting | None,
    max_new_tokens: int,
) -> tuple[str, dict[str, float]]:
    current_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    metric_rows: list[dict[str, float]] = []
    device = input_ids.device

    for _step in range(max_new_tokens):
        if setting is None:
            for controller in controllers.values():
                controller.off()
            with torch.inference_mode():
                output = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        elif setting.method == "fixed_com":
            configure_controllers(
                controllers,
                bank,
                setting,
                -1,
                LAST_POSITION_STOP,
            )
            with torch.inference_mode():
                output = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            metric_rows.append(diagnostics(controllers))
        else:
            collect_last_position(controllers, bank)
            baseline_cache = fork_dynamic_cache(past_key_values)
            with torch.inference_mode():
                model(
                    input_ids=current_ids,
                    past_key_values=baseline_cache,
                    use_cache=True,
                )
            actions, metrics = build_aggregate_actions(
                controllers=controllers,
                bank=bank,
                setting=setting,
                hidden_size=model.config.hidden_size,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )
            configure_precomputed_last(controllers, bank, actions)
            with torch.inference_mode():
                output = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            metric_rows.append(metrics)

        next_token = int(output.logits[0, -1].argmax().item())
        generated.append(next_token)
        past_key_values = output.past_key_values
        if next_token == tokenizer.eos_token_id:
            break
        current_ids = torch.tensor([[next_token]], device=device)

    metrics = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in DIAGNOSTIC_FIELDS
        if metric_rows and all(key in row for row in metric_rows)
    }
    return decode_answer(tokenizer, generated), metrics


def main() -> None:
    args = parse_args()
    settings = [Setting(**item) for item in json.loads(args.settings_file.read_text())]
    unsupported = sorted({setting.method for setting in settings}.difference(ONLINE_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported online methods: {unsupported}")

    frame = load_truthfulqa_frame()
    folds = list(np.array_split(np.arange(len(frame)), 2))
    if args.eval_split == "test":
        indices = folds[args.fold]
    else:
        development = np.concatenate([folds[index] for index in range(2) if index != args.fold])
        rng = np.random.RandomState(42)
        train = set(rng.choice(development, size=int(len(development) * 0.8), replace=False))
        indices = np.array([index for index in development if index not in train])

    output = frame.iloc[indices].copy().reset_index().rename(columns={"index": "dataset_index"})
    output = output.iloc[args.question_offset :]
    if args.max_questions is not None:
        output = output.iloc[: args.max_questions]
    output = output.copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    bank = load_statistics(args.statistics_cache)
    if any(setting.num_heads != len(bank.top_heads) for setting in settings):
        raise ValueError("Settings and statistics cache use different head counts")

    controllers = {
        layer: AttentionHeadController(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.hidden_size,
        )
        for layer, _head in bank.top_heads
    }
    handles = [
        model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(controller)
        for layer, controller in controllers.items()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fold_{args.fold}_{args.eval_split}_generations.csv"
    metadata_path = args.output_dir / f"fold_{args.fold}_{args.eval_split}_metadata.json"
    if args.resume and output_path.exists():
        prior = pd.read_csv(output_path).fillna("")
        if list(prior["dataset_index"]) != list(output["dataset_index"]):
            raise ValueError("Resume artifact does not match requested rows")
        output = prior

    tags = ["baseline", *(setting.tag for setting in settings)]
    for tag in tags:
        answer_column = f"{tag}_answer"
        if answer_column not in output:
            output[answer_column] = ""
        if tag != "baseline":
            for metric in DIAGNOSTIC_FIELDS:
                column = f"{tag}_{metric}"
                if column not in output:
                    output[column] = np.nan

    if args.reuse_from is not None:
        output = reuse_generation_columns(output, pd.read_csv(args.reuse_from), tags)

    metadata_path.write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "statistics_cache": str(args.statistics_cache),
                "settings": [asdict(setting) | {"tag": setting.tag} for setting in settings],
                "fold": args.fold,
                "eval_split": args.eval_split,
                "max_new_tokens": args.max_new_tokens,
                "decode": "greedy online causal control with steered KV cache",
                "reuse_from": str(args.reuse_from) if args.reuse_from is not None else None,
                "api_used": False,
            },
            indent=2,
        )
        + "\n"
    )

    try:
        for row_index in tqdm(output.index, desc=f"generate fold {args.fold} {args.eval_split}"):
            if all(str(output.loc[row_index, f"{tag}_answer"]).strip() for tag in tags):
                continue
            row = output.loc[row_index]
            prompt = DEFAULT_PREFIX + utilities.format_prompt(row, "qa", format="general")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            for tag, setting in [("baseline", None), *[(item.tag, item) for item in settings]]:
                if str(output.loc[row_index, f"{tag}_answer"]).strip():
                    continue
                answer, metrics = greedy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    controllers=controllers,
                    bank=bank,
                    setting=setting,
                    max_new_tokens=args.max_new_tokens,
                )
                output.loc[row_index, f"{tag}_answer"] = answer
                for metric, value in metrics.items():
                    output.loc[row_index, f"{tag}_{metric}"] = value
            if (row_index + 1) % args.checkpoint_every == 0:
                output.to_csv(output_path, index=False)
    finally:
        for handle in handles:
            handle.remove()

    output.to_csv(output_path, index=False)
    summary = []
    for tag in tags:
        row = {
            "fold": args.fold,
            "setting": tag,
            "n": len(output),
            "unique_answers": output[f"{tag}_answer"].nunique(),
            "empty_answers": output[f"{tag}_answer"].astype(str).str.strip().eq("").sum(),
        }
        if tag != "baseline":
            for metric in DIAGNOSTIC_FIELDS:
                row[metric] = output[f"{tag}_{metric}"].mean()
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(
        args.output_dir / f"fold_{args.fold}_{args.eval_split}_generation_summary.csv",
        index=False,
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
