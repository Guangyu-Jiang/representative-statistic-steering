from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from caa_perturbation.core import (
    BehaviorStatistics,
    clean_scalar_hinge_delta,
    clean_statistic_shift_delta,
    fit_behavior_statistics,
    fisher_hinge_delta,
    fisher_statistic_shift_delta,
    fixed_caa_delta,
    pca_margin_hinge_delta,
    pca_statistic_shift_delta,
    pca_target_delta,
    relative_norm,
    scalar_hinge_delta,
    scalar_target_delta,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "caa_perturbation"


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def behavior_dir(output_root: Path, model_name: str, behavior: str) -> Path:
    return output_root / model_slug(model_name) / behavior


def load_json(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def generation_data_path(behavior: str) -> Path:
    return REPO_ROOT / "datasets" / "generate" / behavior / "generate_dataset.json"


def test_data_path(behavior: str, open_ended: bool = False) -> Path:
    filename = "test_dataset_open_ended.json" if open_ended else "test_dataset_ab.json"
    return REPO_ROOT / "datasets" / "test" / behavior / filename


def load_model(model_name: str, device: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("The current pipeline expects a decoder model with model.layers")


def chat_prompt_ids(tokenizer, question: str) -> list[int]:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def answer_letter(answer: str) -> str:
    match = re.fullmatch(r"\s*\(([A-Z])\)\s*", answer)
    if match is None:
        raise ValueError(f"Expected a parenthesized answer letter, received {answer!r}")
    return match.group(1)


def option_id(letter: str) -> int:
    return ord(letter) - ord("A")


def answer_ids(tokenizer, letter: str) -> list[int]:
    # Keeping '(' and the answer letter as separate tokens matches the official
    # evaluation, which supplies '(' and scores A versus B as the next token.
    prefix = tokenizer.encode("(", add_special_tokens=False)
    letter_ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(letter_ids) != 1:
        raise ValueError(f"Expected one token for {letter!r}, got {letter_ids}")
    return prefix + letter_ids


def left_pad(sequences: list[list[int]], pad_token_id: int, device: str):
    maximum = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, -len(sequence) :] = torch.tensor(sequence, device=device)
        attention_mask[row, -len(sequence) :] = 1
    return input_ids, attention_mask


def batched(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def extract_activations(args) -> Path:
    model, tokenizer = load_model(args.model, args.device, args.local_files_only)
    layers = decoder_layers(model)
    data = load_json(generation_data_path(args.behavior))
    if args.max_examples is not None:
        data = data[: args.max_examples]

    n_examples = len(data)
    n_layers = len(layers)
    hidden_size = model.config.hidden_size
    positive = torch.empty(n_examples, n_layers, hidden_size, dtype=torch.float16)
    negative = torch.empty_like(positive)
    positive_is_a = torch.empty(n_examples, dtype=torch.bool)
    positive_option_ids = torch.empty(n_examples, dtype=torch.long)
    negative_option_ids = torch.empty(n_examples, dtype=torch.long)
    positive_letters = []
    negative_letters = []

    records = []
    for index, item in enumerate(data):
        pos_letter = answer_letter(item["answer_matching_behavior"])
        neg_letter = answer_letter(item["answer_not_matching_behavior"])
        if pos_letter == neg_letter:
            raise ValueError("Positive and negative answers must use different letters")
        prompt = chat_prompt_ids(tokenizer, item["question"])
        records.append(
            (
                index,
                prompt + answer_ids(tokenizer, pos_letter),
                prompt + answer_ids(tokenizer, neg_letter),
            )
        )
        positive_is_a[index] = pos_letter == "A"
        positive_option_ids[index] = option_id(pos_letter)
        negative_option_ids[index] = option_id(neg_letter)
        positive_letters.append(pos_letter)
        negative_letters.append(neg_letter)

    for batch in tqdm(list(batched(records, args.batch_size)), desc="Extracting pairs"):
        indices = [record[0] for record in batch]
        for destination, sequence_index in ((positive, 1), (negative, 2)):
            sequences = [record[sequence_index] for record in batch]
            input_ids, attention_mask = left_pad(sequences, tokenizer.pad_token_id, args.device)
            with torch.inference_mode():
                output = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            # hidden_states[0] is the embedding output; subsequent entries are
            # decoder block outputs and align with forward hooks on model.layers.
            for layer_index, hidden in enumerate(output.hidden_states[1:]):
                destination[indices, layer_index] = hidden[:, -1, :].detach().cpu().float().half()
            del output, input_ids, attention_mask

    destination_dir = behavior_dir(args.output_root, args.model, args.behavior)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / "train_activations.pt"
    metadata = {
        "model": args.model,
        "behavior": args.behavior,
        "examples": n_examples,
        "layers": n_layers,
        "hidden_size": hidden_size,
        "source": str(generation_data_path(args.behavior)),
        "activation_site": "decoder_block_output_at_answer_letter",
        "positive_option_counts": dict(sorted(Counter(positive_letters).items())),
        "negative_option_counts": dict(sorted(Counter(negative_letters).items())),
    }
    torch.save(
        {
            "positive": positive,
            "negative": negative,
            "positive_is_a": positive_is_a,
            "positive_option_ids": positive_option_ids,
            "negative_option_ids": negative_option_ids,
            "metadata": metadata,
        },
        output_path,
    )
    dump_json(destination_dir / "train_activations_metadata.json", metadata)
    print(output_path)
    return output_path


def cross_validated_layer_diagnostics(artifact: dict, seed: int) -> list[dict]:
    positive = artifact["positive"].float()
    negative = artifact["negative"].float()
    n_examples, n_layers, _ = positive.shape
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_examples, generator=generator)
    split = max(1, int(0.8 * n_examples))
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    rows = []
    for layer in range(n_layers):
        train_difference = positive[train_indices, layer] - negative[train_indices, layer]
        raw_direction = train_difference.mean(dim=0)
        raw_direction_norm = raw_direction.norm()
        direction = raw_direction / raw_direction_norm.clamp_min(1e-8)
        pos_score = positive[validation_indices, layer] @ direction
        neg_score = negative[validation_indices, layer] @ direction
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        auc = roc_auc_score(labels.numpy(), scores.numpy())
        paired_margin = pos_score - neg_score
        rows.append(
            {
                "layer": layer,
                "validation_auc": float(auc),
                "paired_margin_mean": float(paired_margin.mean()),
                "paired_margin_std": float(paired_margin.std(unbiased=False)),
                "caa_vector_norm": float(raw_direction_norm),
            }
        )
    return rows


@dataclass
class InterventionConfig:
    method: str
    strength: float
    ridge: float
    max_relative_norm: float | None
    target_quantile: float = 0.75


class InterventionController:
    def __init__(self, stats: BehaviorStatistics):
        self.stats = stats
        self.config = InterventionConfig("none", 0.0, 0.0, None)
        self.patch_mask: torch.Tensor | None = None
        self.decode_all = False
        self.relative_norm_sum = 0.0
        self.relative_norm_count = 0

    def configure(
        self,
        config: InterventionConfig,
        patch_mask: torch.Tensor | None,
        decode_all: bool = False,
    ) -> None:
        self.config = config
        self.patch_mask = patch_mask
        self.decode_all = decode_all

    def reset_metrics(self) -> None:
        self.relative_norm_sum = 0.0
        self.relative_norm_count = 0

    @property
    def mean_relative_norm(self) -> float:
        if self.relative_norm_count == 0:
            return 0.0
        return self.relative_norm_sum / self.relative_norm_count

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        config = self.config
        if config.method == "caa":
            return fixed_caa_delta(
                hidden, self.stats, config.strength, config.max_relative_norm
            )
        if config.method == "scalar_target":
            return scalar_target_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.max_relative_norm,
            )
        if config.method == "pca_target":
            return pca_target_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.max_relative_norm,
            )
        if config.method == "scalar_hinge":
            return scalar_hinge_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.target_quantile,
                config.max_relative_norm,
            )
        if config.method == "clean_scalar_hinge":
            return clean_scalar_hinge_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.target_quantile,
                config.max_relative_norm,
            )
        if config.method == "fisher_hinge":
            return fisher_hinge_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.target_quantile,
                config.max_relative_norm,
            )
        if config.method == "clean_statistic_shift":
            return clean_statistic_shift_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.max_relative_norm,
            )
        if config.method == "fisher_statistic_shift":
            return fisher_statistic_shift_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.max_relative_norm,
            )
        if config.method == "pca_statistic_shift":
            return pca_statistic_shift_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.max_relative_norm,
            )
        if config.method == "pca_margin_hinge":
            return pca_margin_hinge_delta(
                hidden,
                self.stats,
                config.strength,
                config.ridge,
                config.target_quantile,
                config.max_relative_norm,
            )
        return torch.zeros_like(hidden)

    def hook(self, _module, _inputs, output):
        if self.config.method == "none" or self.config.strength == 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == 1 and self.decode_all:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            mask = self.patch_mask
        if mask is None or mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Patch mask {None if mask is None else tuple(mask.shape)} does not match "
                f"hidden states {tuple(hidden.shape[:2])}"
            )
        selected = hidden[mask]
        if selected.numel() == 0:
            return output
        delta = self._delta(selected)
        ratios = relative_norm(delta, selected)
        self.relative_norm_sum += float(ratios.sum().detach().cpu())
        self.relative_norm_count += ratios.numel()
        updated = hidden.clone()
        updated[mask] = selected + delta
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated


def make_eval_batch(tokenizer, items: list[dict], device: str):
    prefix_ids = tokenizer.encode("(", add_special_tokens=False)
    sequences = []
    prefix_lengths = []
    for item in items:
        prompt = chat_prompt_ids(tokenizer, item["question"])
        sequences.append(prompt + prefix_ids)
        prefix_lengths.append(len(prefix_ids))
    input_ids, attention_mask = left_pad(sequences, tokenizer.pad_token_id, device)
    patch_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, prefix_length in enumerate(prefix_lengths):
        patch_mask[row, -prefix_length:] = True
    return input_ids, attention_mask, patch_mask


def score_configuration(
    model,
    tokenizer,
    test_data: list[dict],
    controller: InterventionController,
    config: InterventionConfig,
    batch_size: int,
    device: str,
):
    option_token_ids = {}
    for item in test_data:
        for key in ("answer_matching_behavior", "answer_not_matching_behavior"):
            letter = answer_letter(item[key])
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"The held-out metric requires one-token answer options; "
                    f"{letter!r} maps to {token_ids}"
                )
            option_token_ids[letter] = token_ids[0]
    behavior_probabilities = []
    correct = []
    predictions = []
    controller.reset_metrics()
    offset = 0
    for batch in batched(test_data, batch_size):
        input_ids, attention_mask, patch_mask = make_eval_batch(tokenizer, batch, device)
        controller.configure(config, patch_mask=patch_mask)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].float()
        for row, item in enumerate(batch):
            matching = answer_letter(item["answer_matching_behavior"])
            not_matching = answer_letter(item["answer_not_matching_behavior"])
            if matching == not_matching:
                raise ValueError("Matching and non-matching options must differ")
            pair_logits = logits[
                row,
                [option_token_ids[matching], option_token_ids[not_matching]],
            ]
            pair_probabilities = pair_logits.softmax(dim=-1).cpu()
            probability = float(pair_probabilities[0])
            predicted_matching = int(pair_probabilities.argmax()) == 0
            behavior_probabilities.append(probability)
            correct.append(predicted_matching)
            predictions.append(
                {
                    "index": offset + row,
                    "question": item["question"],
                    "answer_matching_behavior": item["answer_matching_behavior"],
                    "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                    "matching_option": matching,
                    "not_matching_option": not_matching,
                    "matching_option_probability": probability,
                    "not_matching_option_probability": float(pair_probabilities[1]),
                    "matching_probability": probability,
                    "predicted_matching_behavior": predicted_matching,
                }
            )
        offset += len(batch)
    summary = {
        "n": len(test_data),
        "mean_matching_probability": sum(behavior_probabilities) / len(behavior_probabilities),
        "matching_accuracy": sum(correct) / len(correct),
        "mean_action_relative_norm": controller.mean_relative_norm,
    }
    return summary, predictions


def fit_stats_for_layer(artifact: dict, layer: int, components: int, seed: int):
    return fit_behavior_statistics(
        artifact["positive"][:, layer],
        artifact["negative"][:, layer],
        artifact.get("positive_is_a"),
        n_components=components,
        remove_letter=True,
        seed=seed,
        positive_option_ids=artifact.get("positive_option_ids"),
        negative_option_ids=artifact.get("negative_option_ids"),
    )


def scan_layers(args) -> Path:
    destination_dir = behavior_dir(args.output_root, args.model, args.behavior)
    artifact = torch.load(
        destination_dir / "train_activations.pt", map_location="cpu", weights_only=False
    )
    model, tokenizer = load_model(args.model, args.device, args.local_files_only)
    test_data = load_json(test_data_path(args.behavior))
    baseline_stats = fit_stats_for_layer(artifact, 0, 1, args.seed).to(
        args.device, next(model.parameters()).dtype
    )
    baseline_controller = InterventionController(baseline_stats)
    baseline_result, baseline_predictions = score_configuration(
        model,
        tokenizer,
        test_data,
        baseline_controller,
        InterventionConfig("none", 0.0, 0.0, None),
        args.batch_size,
        args.device,
    )
    rows = [
        {
            "layer": -1,
            "strength": 0.0,
            **baseline_result,
        }
    ]
    predictions = [
        {"layer": -1, "strength": 0.0, **prediction}
        for prediction in baseline_predictions
    ]
    n_layers = artifact["positive"].shape[1]
    selected_layers = args.layers if args.layers else list(range(n_layers))
    for layer in tqdm(selected_layers, desc="CAA layer scan"):
        stats = fit_stats_for_layer(artifact, layer, 1, args.seed).to(
            args.device, next(model.parameters()).dtype
        )
        controller = InterventionController(stats)
        handle = decoder_layers(model)[layer].register_forward_hook(controller.hook)
        try:
            for strength in args.strengths:
                result, setting_predictions = score_configuration(
                    model,
                    tokenizer,
                    test_data,
                    controller,
                    InterventionConfig(
                        "caa", strength, 0.0, args.max_relative_norm
                    ),
                    args.batch_size,
                    args.device,
                )
                rows.append({"layer": layer, "strength": strength, **result})
                predictions.extend(
                    {"layer": layer, "strength": strength, **prediction}
                    for prediction in setting_predictions
                )
        finally:
            handle.remove()
    output_path = destination_dir / "caa_layer_scan.csv"
    write_summary_csv(output_path, rows)
    dump_json(destination_dir / "caa_layer_scan_predictions.json", predictions)
    print(output_path)
    return output_path


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args) -> Path:
    destination_dir = behavior_dir(args.output_root, args.model, args.behavior)
    artifact_path = destination_dir / "train_activations.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    diagnostics = cross_validated_layer_diagnostics(artifact, args.seed)
    dump_json(destination_dir / "layer_diagnostics.json", diagnostics)
    if args.layer is None:
        selected_layer = max(diagnostics, key=lambda row: row["validation_auc"])["layer"]
    else:
        selected_layer = args.layer
    print(f"Selected layer: {selected_layer}")

    model, tokenizer = load_model(args.model, args.device, args.local_files_only)
    test_data = load_json(test_data_path(args.behavior))
    summary_rows = []
    all_predictions = []

    component_values = sorted(set(args.components))
    max_components = max(component_values)
    maximum_stats = fit_stats_for_layer(artifact, selected_layer, max_components, args.seed)
    stats_path = destination_dir / f"stats_layer_{selected_layer}_components_{max_components}.pt"
    torch.save(maximum_stats.state_dict(), stats_path)

    def truncated_stats(component_count: int):
        state = maximum_stats.state_dict()
        for key in (
            "components",
            "component_scale",
            "positive_centroid",
            "negative_centroid",
            "pca_margin_positive_targets",
            "explained_variance_ratio",
        ):
            state[key] = state[key][:component_count]
        return BehaviorStatistics.from_state_dict(state).to(args.device, next(model.parameters()).dtype)

    # All methods share the same layer and hook. PCA dimensions only alter the
    # representative statistic and its target, not the intervention site.
    layer_module = decoder_layers(model)[selected_layer]
    controller = InterventionController(truncated_stats(max_components))
    hook_handle = layer_module.register_forward_hook(controller.hook)

    configurations: list[tuple[InterventionConfig, int]] = [
        (InterventionConfig("none", 0.0, 0.0, None), 0)
    ]
    configurations.extend(
        (InterventionConfig("caa", strength, 0.0, args.max_relative_norm), 0)
        for strength in args.caa_strengths
        if strength != 0
    )
    configurations.extend(
        (InterventionConfig("scalar_target", strength, ridge, args.max_relative_norm), 1)
        for ridge in args.ridges
        for strength in args.target_strengths
    )
    configurations.extend(
        (InterventionConfig("pca_target", strength, ridge, args.max_relative_norm), components)
        for components in component_values
        for ridge in args.ridges
        for strength in args.target_strengths
    )
    if args.include_improved:
        for method in ("scalar_hinge", "clean_scalar_hinge", "fisher_hinge"):
            configurations.extend(
                (
                    InterventionConfig(
                        method,
                        strength,
                        ridge,
                        args.max_relative_norm,
                        target_quantile,
                    ),
                    1,
                )
                for target_quantile in args.target_quantiles
                for ridge in args.improved_ridges
                for strength in args.improved_strengths
            )
        for method in ("clean_statistic_shift", "fisher_statistic_shift"):
            configurations.extend(
                (
                    InterventionConfig(
                        method, strength, ridge, args.max_relative_norm
                    ),
                    1,
                )
                for ridge in args.improved_ridges
                for strength in args.improved_strengths
            )
        for method in ("pca_statistic_shift", "pca_margin_hinge"):
            configurations.extend(
                (
                    InterventionConfig(
                        method,
                        strength,
                        ridge,
                        args.max_relative_norm,
                        target_quantile,
                    ),
                    components,
                )
                for components in component_values
                for target_quantile in (
                    args.target_quantiles
                    if method == "pca_margin_hinge"
                    else [0.75]
                )
                for ridge in args.improved_ridges
                for strength in args.improved_strengths
            )

    try:
        for config, components in tqdm(configurations, desc="Evaluating settings"):
            if config.method in {
                "pca_target",
                "pca_statistic_shift",
                "pca_margin_hinge",
            }:
                controller.stats = truncated_stats(components)
            else:
                controller.stats = truncated_stats(max_components)
            result, predictions = score_configuration(
                model,
                tokenizer,
                test_data,
                controller,
                config,
                args.batch_size,
                args.device,
            )
            setting_id = f"{config.method}__r{components}__strength{config.strength:g}__ridge{config.ridge:g}"
            if config.method in {
                "scalar_hinge",
                "clean_scalar_hinge",
                "fisher_hinge",
                "pca_margin_hinge",
            }:
                setting_id += f"__q{config.target_quantile:g}"
            row = {
                "setting_id": setting_id,
                "model": args.model,
                "behavior": args.behavior,
                "layer": selected_layer,
                "method": config.method,
                "components": components,
                "strength": config.strength,
                "ridge": config.ridge,
                "target_quantile": config.target_quantile,
                "max_relative_norm": args.max_relative_norm,
                **result,
            }
            summary_rows.append(row)
            for prediction in predictions:
                all_predictions.append({"setting_id": setting_id, **prediction})
            print(row)
    finally:
        hook_handle.remove()

    summary_path = destination_dir / "mc_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    dump_json(destination_dir / "mc_predictions.json", all_predictions)
    dump_json(
        destination_dir / "evaluation_metadata.json",
        {
            "model": args.model,
            "behavior": args.behavior,
            "selected_layer": selected_layer,
            "layer_selection": "80/20 training-pair AUC",
            "test_source": str(test_data_path(args.behavior)),
            "test_examples": len(test_data),
            "external_api_used": False,
        },
    )
    print(summary_path)
    return summary_path


def generation_input(tokenizer, question: str, device: str):
    ids = chat_prompt_ids(tokenizer, question)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    # The final assistant-header token is the state that predicts the first
    # generated token. Subsequent cached decode states have sequence length 1.
    patch_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    patch_mask[:, -1] = True
    return input_ids, attention_mask, patch_mask


def generate_open_ended(args) -> Path:
    destination_dir = behavior_dir(args.output_root, args.model, args.behavior)
    artifact = torch.load(
        destination_dir / "train_activations.pt", map_location="cpu", weights_only=False
    )
    stats = fit_stats_for_layer(artifact, args.layer, args.components, args.seed)
    model, tokenizer = load_model(args.model, args.device, args.local_files_only)
    stats = stats.to(args.device, next(model.parameters()).dtype)
    controller = InterventionController(stats)
    hook_handle = decoder_layers(model)[args.layer].register_forward_hook(controller.hook)
    data = load_json(test_data_path(args.behavior, open_ended=True))
    if args.max_examples is not None:
        data = data[: args.max_examples]

    config = InterventionConfig(
        args.method,
        args.strength,
        args.ridge,
        args.max_relative_norm,
        args.target_quantile,
    )
    rows = []
    controller.reset_metrics()
    try:
        for index, item in enumerate(tqdm(data, desc="Generating")):
            input_ids, attention_mask, patch_mask = generation_input(
                tokenizer, item["question"], args.device
            )
            controller.configure(config, patch_mask=patch_mask, decode_all=True)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[0, input_ids.shape[1] :]
            response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            rows.append(
                {
                    "index": index,
                    "question": item["question"],
                    "response": response,
                }
            )
    finally:
        hook_handle.remove()
    suffix = f"{args.method}__r{args.components}__strength{args.strength:g}__ridge{args.ridge:g}"
    if args.method in {
        "scalar_hinge",
        "clean_scalar_hinge",
        "fisher_hinge",
        "pca_margin_hinge",
    }:
        suffix += f"__q{args.target_quantile:g}"
    output_path = destination_dir / "open_ended" / f"{suffix}.json"
    dump_json(output_path, rows)
    dump_json(
        output_path.with_suffix(".metadata.json"),
        {
            "model": args.model,
            "behavior": args.behavior,
            "layer": args.layer,
            "method": args.method,
            "components": args.components,
            "strength": args.strength,
            "ridge": args.ridge,
            "target_quantile": args.target_quantile,
            "mean_action_relative_norm": controller.mean_relative_norm,
            "external_api_used": False,
        },
    )
    print(output_path)
    return output_path


def add_shared_arguments(parser):
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--behavior", default="sycophancy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    add_shared_arguments(extract_parser)
    extract_parser.add_argument("--batch-size", type=int, default=4)
    extract_parser.add_argument("--max-examples", type=int)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_shared_arguments(evaluate_parser)
    evaluate_parser.add_argument("--layer", type=int)
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--components", nargs="+", type=int, default=[1, 2, 4, 8])
    evaluate_parser.add_argument(
        "--caa-strengths", nargs="+", type=float, default=[-2, -1, -0.5, 0.5, 1, 2]
    )
    evaluate_parser.add_argument(
        "--target-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0, 1.5]
    )
    evaluate_parser.add_argument("--ridges", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    evaluate_parser.add_argument(
        "--include-improved", action=argparse.BooleanOptionalAction, default=False
    )
    evaluate_parser.add_argument(
        "--improved-strengths",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    evaluate_parser.add_argument(
        "--improved-ridges", nargs="+", type=float, default=[0.1]
    )
    evaluate_parser.add_argument(
        "--target-quantiles",
        nargs="+",
        type=float,
        choices=[0.5, 0.75, 0.9],
        default=[0.5, 0.75, 0.9],
    )
    evaluate_parser.add_argument("--max-relative-norm", type=float, default=0.5)

    scan_parser = subparsers.add_parser("scan")
    add_shared_arguments(scan_parser)
    scan_parser.add_argument("--layers", nargs="+", type=int)
    scan_parser.add_argument("--strengths", nargs="+", type=float, default=[-1.0, 1.0])
    scan_parser.add_argument("--batch-size", type=int, default=8)
    scan_parser.add_argument("--max-relative-norm", type=float, default=0.5)

    generate_parser = subparsers.add_parser("generate")
    add_shared_arguments(generate_parser)
    generate_parser.add_argument("--layer", type=int, required=True)
    generate_parser.add_argument(
        "--method",
        choices=[
            "none",
            "caa",
            "scalar_target",
            "pca_target",
            "scalar_hinge",
            "clean_scalar_hinge",
            "fisher_hinge",
            "clean_statistic_shift",
            "fisher_statistic_shift",
            "pca_statistic_shift",
            "pca_margin_hinge",
        ],
        required=True,
    )
    generate_parser.add_argument("--components", type=int, default=4)
    generate_parser.add_argument("--strength", type=float, default=1.0)
    generate_parser.add_argument("--ridge", type=float, default=0.1)
    generate_parser.add_argument("--target-quantile", type=float, default=0.75)
    generate_parser.add_argument("--max-relative-norm", type=float, default=0.5)
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.command == "extract":
        extract_activations(args)
    elif args.command == "scan":
        scan_layers(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "generate":
        generate_open_ended(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
