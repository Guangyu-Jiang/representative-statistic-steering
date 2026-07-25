#!/usr/bin/env python3
"""Compare fixed ITI under modern, legacy, and causal MC application sites."""

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
from utils import (
    get_com_directions,
    get_separated_activations,
    get_top_heads,
    layer_head_to_flattened_idx,
    load_truthfulqa_frame,
)
from truthfulqa import utilities
from truthfulqa.configs import ANSWER_COL, BEST_COL, INCORRECT_COL
from truthfulqa.models import MC_calcs, set_columns
from validate_margin_perturbation import (
    DEFAULT_PREFIX,
    MODEL_PATH,
    LayerController,
    fold_train_val_indices,
    score_answer_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[0, 1], required=True)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("../artifacts/fixed_iti_positions"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=("baseline", "fixed_modern_last", "fixed_legacy_answer", "fixed_causal_answer"),
        default=("baseline", "fixed_modern_last", "fixed_legacy_answer", "fixed_causal_answer"),
        help="Only evaluate the selected application sites.",
    )
    return parser.parse_args()


def build_fixed_actions(
    *,
    top_heads: list[tuple[int, int]],
    com_directions: np.ndarray,
    tuning_activations: np.ndarray,
    num_heads: int,
    hidden_size: int,
    head_dim: int,
    alpha: float,
) -> dict[int, torch.Tensor]:
    actions: dict[int, torch.Tensor] = {}
    for layer, head in top_heads:
        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        projected = tuning_activations[:, layer, head, :] @ direction
        scaled = alpha * float(np.std(projected)) * direction
        action = actions.setdefault(layer, torch.zeros(hidden_size, dtype=torch.float32))
        action[head * head_dim : (head + 1) * head_dim] += torch.from_numpy(scaled).float()
    return actions


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    frame = load_truthfulqa_frame()
    folds = list(np.array_split(np.arange(len(frame)), 2))
    train_indices, validation_indices, _development_indices = fold_train_val_indices(
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
    tuning_activations = rearrange(
        np.load("../features/llama3_8B_instruct_tqa_gen_end_q_head_wise.npy"),
        "b l (h d) -> b l h d",
        h=num_heads,
    )
    separated_activations, separated_labels, _ = get_separated_activations(labels, head_activations)
    com_directions = get_com_directions(
        num_layers,
        num_heads,
        train_indices,
        validation_indices,
        separated_activations,
        separated_labels,
    )
    top_heads, _probes = get_top_heads(
        train_indices,
        validation_indices,
        separated_activations,
        separated_labels,
        num_layers,
        num_heads,
        args.seed,
        args.num_heads,
        False,
    )
    top_heads = [(int(layer), int(head)) for layer, head in top_heads]
    actions = build_fixed_actions(
        top_heads=top_heads,
        com_directions=com_directions,
        tuning_activations=tuning_activations,
        num_heads=num_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        alpha=args.alpha,
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

    tags = tuple(dict.fromkeys(args.settings))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fold_{args.fold}_results.csv"
    output = frame.iloc[test_indices].copy().reset_index().rename(columns={"index": "dataset_index"})
    if args.max_questions is not None:
        output = output.iloc[: args.max_questions].copy()
    if args.resume and output_path.exists():
        prior = pd.read_csv(output_path)
        if list(prior["dataset_index"]) != list(output["dataset_index"]):
            raise ValueError("Resume output does not match the requested test slice")
        output = prior
    for tag in tags:
        set_columns(tag, output)

    for row_index in tqdm(output.index, desc=f"fixed ITI positions fold {args.fold}"):
        if all(pd.notna(output.loc[row_index, f"{tag} MC2"]) for tag in tags):
            continue
        row = output.loc[row_index]
        true_answers = utilities.split_multi_answer(row[ANSWER_COL])
        false_answers = utilities.split_multi_answer(row[INCORRECT_COL])
        best_answer = utilities.format_best(row[BEST_COL])
        scores = {tag: [] for tag in tags}
        input_prompt = DEFAULT_PREFIX + utilities.format_prompt(row, "qa", format="general")
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        input_length = input_ids.shape[-1]

        for answer in [*true_answers, *false_answers]:
            prompt = DEFAULT_PREFIX + utilities.format_prompt_with_answer_strings(
                str(row["Question"]), answer, "qa", format="general"
            )
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if "baseline" in tags:
                for controller in controllers.values():
                    controller.collect()
                with torch.inference_mode():
                    _, baseline_output = intervened_model({"input_ids": prompt_ids})
                scores["baseline"].append(
                    score_answer_logits(baseline_output.logits, prompt_ids, input_length)
                )
            starts = {
                "fixed_modern_last": prompt_ids.shape[-1] - 1,
                "fixed_legacy_answer": min(input_length + 4, prompt_ids.shape[-1] - 1),
                "fixed_causal_answer": input_length - 1,
            }
            for tag, start in starts.items():
                if tag not in tags:
                    continue
                for layer, controller in controllers.items():
                    controller.apply(actions[layer], int(start))
                with torch.inference_mode():
                    _, steered_output = intervened_model({"input_ids": prompt_ids})
                scores[tag].append(
                    score_answer_logits(steered_output.logits, prompt_ids, input_length)
                )

        split = len(true_answers)
        for tag in tags:
            MC_calcs(
                tag,
                output,
                row_index,
                scores[tag][:split],
                scores[tag][split:],
                true_answers,
                best_answer,
            )
        if (row_index + 1) % args.checkpoint_every == 0:
            output.to_csv(output_path, index=False)

    output.to_csv(output_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "fold": args.fold,
                "setting": tag,
                "n": len(output),
                "mc1": output[f"{tag} MC1"].mean(),
                "mc2": output[f"{tag} MC2"].mean(),
            }
            for tag in tags
        ]
    )
    summary.to_csv(args.output_dir / f"fold_{args.fold}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
