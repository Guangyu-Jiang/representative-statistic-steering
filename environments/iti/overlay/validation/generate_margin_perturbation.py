#!/usr/bin/env python3
"""Generate TruthfulQA answers with two-pass adaptive margin steering."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyvene as pv

from interveners import wrapper
from utils import get_com_directions, get_separated_activations, load_truthfulqa_frame
from truthfulqa import utilities
from validate_margin_perturbation import (
    DEFAULT_PREFIX,
    MODEL_PATH,
    LayerController,
    aggregate_margin,
    build_actions,
    fold_train_val_indices,
    prepare_probe_statistics,
    selected_head_matrix,
    setting_tag,
    slug_number,
)
from validate_fixed_iti_positions import build_fixed_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[0, 1], required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--ridge-ratio", type=float, required=True)
    parser.add_argument("--target-quantile", type=float, default=0.25)
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--include-fixed-iti",
        action="store_true",
        help="also generate with the official fixed center-of-mass ITI direction",
    )
    parser.add_argument("--fixed-alpha", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("../artifacts/perturbation_margin_generation"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def decode_answer(tokenizer: AutoTokenizer, generated: torch.Tensor, prompt_length: int) -> str:
    text = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
    text = text.split("Q:", maxsplit=1)[0].strip()
    if "A:" in text:
        text = text.split("A:", maxsplit=1)[-1].strip()
    return text


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    frame = load_truthfulqa_frame()
    folds = list(np.array_split(np.arange(len(frame)), 2))
    train_indices, validation_indices, development_indices = fold_train_val_indices(
        args.fold, folds, args.seed, args.val_ratio
    )
    test_indices = folds[args.fold]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    head_activations = rearrange(
        np.load("../features/llama3_8B_instruct_tqa_mc2_head_wise.npy"),
        "b l (h d) -> b l h d",
        h=num_heads,
    )
    labels = np.load("../features/llama3_8B_instruct_tqa_mc2_labels.npy")
    (
        top_heads,
        weights,
        intercepts,
        means,
        stds,
        target,
        _probe_diagnostics,
    ) = prepare_probe_statistics(
        head_activations=head_activations,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        development_indices=development_indices,
        num_layers=num_layers,
        num_heads=num_heads,
        num_selected_heads=args.num_heads,
        seed=args.seed,
        target_quantile=args.target_quantile,
    )

    fixed_actions: dict[int, torch.Tensor] | None = None
    fixed_tag = f"fixed_iti_a{slug_number(args.fixed_alpha)}"
    if args.include_fixed_iti:
        separated_activations, separated_labels, _ = get_separated_activations(
            labels,
            head_activations,
        )
        com_directions = get_com_directions(
            num_layers,
            num_heads,
            train_indices,
            validation_indices,
            separated_activations,
            separated_labels,
        )
        tuning_activations = rearrange(
            np.load("../features/llama3_8B_instruct_tqa_gen_end_q_head_wise.npy"),
            "b l (h d) -> b l h d",
            h=num_heads,
        )
        fixed_actions = build_fixed_actions(
            top_heads=top_heads,
            com_directions=com_directions,
            tuning_activations=tuning_activations,
            num_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            alpha=args.fixed_alpha,
        )

    controllers = {layer: LayerController() for layer, _head in top_heads}
    intervened_model = pv.IntervenableModel(
        [
            {
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(controller),
            }
            for layer, controller in controllers.items()
        ],
        model,
    )

    tag = setting_tag(args.alpha, args.ridge_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fold_{args.fold}_{tag}.csv"
    output = frame.iloc[test_indices].copy().reset_index().rename(columns={"index": "dataset_index"})
    if args.max_questions is not None:
        output = output.iloc[: args.max_questions].copy()
    if args.resume and output_path.exists():
        prior = pd.read_csv(output_path)
        if list(prior["dataset_index"]) != list(output["dataset_index"]):
            raise ValueError("Resume output does not match the requested test slice")
        output = prior
    for column in (
        "baseline_answer",
        f"{tag}_answer",
        "pre_margin",
        "target_margin",
        "post_margin",
        "relative_action_norm",
        "intervened",
    ):
        if column not in output:
            output[column] = np.nan if not column.endswith("answer") else ""
    if args.include_fixed_iti and f"{fixed_tag}_answer" not in output:
        output[f"{fixed_tag}_answer"] = ""

    for row_index in tqdm(output.index, desc=f"generate fold {args.fold} {tag}"):
        adaptive_done = str(output.loc[row_index, f"{tag}_answer"]).strip()
        fixed_done = (
            not args.include_fixed_iti
            or str(output.loc[row_index, f"{fixed_tag}_answer"]).strip()
        )
        if adaptive_done and fixed_done:
            continue
        row = output.loc[row_index]
        prompt = DEFAULT_PREFIX + utilities.format_prompt(row, "qa", format="general")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            baseline_tokens = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        baseline_answer = decode_answer(tokenizer, baseline_tokens, input_ids.shape[-1])

        fixed_answer = ""
        if fixed_actions is not None:
            for layer, controller in controllers.items():
                controller.apply(fixed_actions[layer], -1)
            with torch.inference_mode():
                _, fixed_tokens = intervened_model.generate(
                    {"input_ids": input_ids},
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            fixed_answer = decode_answer(tokenizer, fixed_tokens, input_ids.shape[-1])

        completed_prompt = DEFAULT_PREFIX + utilities.format_prompt_with_answer_strings(
            str(row["Question"]), baseline_answer, "qa", format="general"
        )
        completed_ids = tokenizer(completed_prompt, return_tensors="pt").input_ids.to(device)
        for controller in controllers.values():
            controller.collect()
        with torch.inference_mode():
            intervened_model({"input_ids": completed_ids})
        states = {
            layer: controller.pre_state
            for layer, controller in controllers.items()
            if controller.pre_state is not None
        }
        selected = selected_head_matrix(states, top_heads, head_dim)
        pre_margin = aggregate_margin(selected, weights, intercepts, means, stds)
        requested_shift = args.alpha * max(target - pre_margin, 0.0)
        target_margin = pre_margin + requested_shift
        actions, action_norm = build_actions(
            top_heads=top_heads,
            weights=weights,
            stds=stds,
            hidden_size=hidden_size,
            head_dim=head_dim,
            requested_shift=requested_shift,
            ridge_ratio=args.ridge_ratio,
        )
        if requested_shift == 0.0:
            steered_answer = baseline_answer
            post_margin = pre_margin
        else:
            for layer, controller in controllers.items():
                controller.apply(actions[layer], -1)
            with torch.inference_mode():
                _, steered_tokens = intervened_model.generate(
                    {"input_ids": input_ids},
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            steered_answer = decode_answer(tokenizer, steered_tokens, input_ids.shape[-1])
            post_states = {
                layer: controller.post_state
                for layer, controller in controllers.items()
                if controller.post_state is not None
            }
            post_selected = selected_head_matrix(post_states, top_heads, head_dim)
            post_margin = aggregate_margin(post_selected, weights, intercepts, means, stds)

        output.loc[row_index, "baseline_answer"] = baseline_answer
        output.loc[row_index, f"{tag}_answer"] = steered_answer
        if fixed_actions is not None:
            output.loc[row_index, f"{fixed_tag}_answer"] = fixed_answer
        output.loc[row_index, "pre_margin"] = pre_margin
        output.loc[row_index, "target_margin"] = target_margin
        output.loc[row_index, "post_margin"] = post_margin
        output.loc[row_index, "relative_action_norm"] = action_norm / max(float(np.linalg.norm(selected)), 1e-12)
        output.loc[row_index, "intervened"] = int(requested_shift > 0.0)
        if (row_index + 1) % args.checkpoint_every == 0:
            output.to_csv(output_path, index=False)

    output.to_csv(output_path, index=False)
    print(
        pd.Series(
            {
                "fold": args.fold,
                "setting": tag,
                "n": len(output),
                "intervention_rate": output["intervened"].mean(),
                "relative_action_norm": output["relative_action_norm"].mean(),
                "target_error": (output["post_margin"] - output["target_margin"]).abs().mean(),
                "unique_baseline": output["baseline_answer"].nunique(),
                "unique_steered": output[f"{tag}_answer"].nunique(),
                "unique_fixed": (
                    output[f"{fixed_tag}_answer"].nunique()
                    if fixed_actions is not None
                    else np.nan
                ),
            }
        ).to_string()
    )


if __name__ == "__main__":
    main()
