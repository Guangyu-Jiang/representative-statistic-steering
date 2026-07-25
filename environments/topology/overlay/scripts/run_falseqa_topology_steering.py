#!/usr/bin/env python3
"""Steer FalseQA responses with topology-local paired activation contrasts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.eval.falseqa_steering import (
    build_paired_local_directions,
    build_topology_tensor,
    paired_hidden_differences,
    response_quality,
    select_layer_with_train_validation,
    topology_neighbor_matrices,
)
from aen_replication.models.generation import _build_generate_kwargs, render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.steering import _decoder_layers
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet
from aen_replication.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--topology-root", default="artifacts/falseqa_topology_classification")
    parser.add_argument("--artifact-root", default="artifacts/falseqa_topology_steering_aligned")
    parser.add_argument("--protocol", choices=["random80", "official"], default="random80")
    parser.add_argument("--neighbor-ks", nargs="+", type=int, default=[5, 20, 50])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1, 2, 4, 8])
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="H0 output layer; -1 selects it using train-only validation. Steering patches layer+1 input.",
    )
    parser.add_argument("--eval-n", type=int, default=0, help="0 evaluates every held-out false question.")
    parser.add_argument("--activation-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--steering-apply-on",
        choices=["prompt_only", "prompt_and_decode"],
        default="prompt_and_decode",
    )
    parser.add_argument("--force-activations", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    return parser.parse_args()


def _float_slug(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _render_questions(
    bundle: HFModelBundle,
    questions: list[str],
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
) -> list[str]:
    prompts = [f"{question}{prompt_suffix}" if prompt_suffix else str(question) for question in questions]
    return render_prompts(
        bundle=bundle,
        prompt_texts=prompts,
        use_chat_template=bool(generation_cfg.get("use_chat_template", True)),
        system_prompt=generation_cfg.get("system_prompt"),
        add_generation_prompt=True,
    )


def _base_model(bundle: HFModelBundle) -> torch.nn.Module:
    return getattr(bundle.model, "model", bundle.model)


def _extract_layer_input_vectors(
    *,
    bundle: HFModelBundle,
    frame: pd.DataFrame,
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
    layer: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Mean-pool the exact layer-input tensor used by the steering pre-hook."""

    target_layer = _decoder_layers(bundle.model)[layer]
    tokenizer = bundle.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    vectors: list[np.ndarray] = []
    mean_token_norms: list[np.ndarray] = []
    fro_norms: list[np.ndarray] = []
    token_counts: list[np.ndarray] = []
    try:
        for start in tqdm(range(0, len(frame), batch_size), desc="falseqa_layer_input"):
            batch = frame.iloc[start : start + batch_size]
            rendered = _render_questions(
                bundle,
                batch["question"].astype(str).tolist(),
                generation_cfg,
                prompt_suffix,
            )
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(generation_cfg.get("prompt_max_length", 2048)),
            )
            encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
            attention_mask = encoded["attention_mask"].to(dtype=torch.float32)
            captured: dict[str, torch.Tensor] = {}

            def input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                hidden = inputs[0].detach().float()
                mask = attention_mask.unsqueeze(-1)
                counts = attention_mask.sum(dim=1).clamp_min(1.0)
                captured["vectors"] = (hidden * mask).sum(dim=1) / counts[:, None]
                per_token_norm = hidden.norm(dim=-1)
                captured["mean_token_norms"] = (per_token_norm * attention_mask).sum(dim=1) / counts
                captured["fro_norms"] = ((hidden.square() * mask).sum(dim=(1, 2))).sqrt()
                captured["token_counts"] = counts

            handle = target_layer.register_forward_pre_hook(input_hook)
            try:
                with torch.inference_mode():
                    _base_model(bundle)(**encoded, use_cache=False, return_dict=True)
            finally:
                handle.remove()
            if "vectors" not in captured:
                raise RuntimeError(f"Layer-input hook did not run for layer {layer}")
            vectors.append(captured["vectors"].cpu().numpy().astype(np.float32, copy=False))
            mean_token_norms.append(captured["mean_token_norms"].cpu().numpy().astype(np.float32, copy=False))
            fro_norms.append(captured["fro_norms"].cpu().numpy().astype(np.float32, copy=False))
            token_counts.append(captured["token_counts"].cpu().numpy().astype(np.int32, copy=False))
            del encoded, captured
            if bundle.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return {
        "example_ids": np.asarray(frame["example_id"].astype(str).tolist(), dtype=np.str_),
        "vectors": np.vstack(vectors),
        "mean_token_norms": np.concatenate(mean_token_norms),
        "fro_norms": np.concatenate(fro_norms),
        "token_counts": np.concatenate(token_counts),
    }


def _generate(
    *,
    bundle: HFModelBundle,
    questions: list[str],
    directions: np.ndarray | None,
    alpha: float,
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
    layer: int,
    apply_on: str,
) -> tuple[list[str], list[int]]:
    target_layer = _decoder_layers(bundle.model)[layer]
    batch_size = int(generation_cfg.get("batch_size", 8))
    responses: list[str] = []
    generated_counts: list[int] = []
    for start in tqdm(range(0, len(questions), batch_size), desc=f"falseqa_generate_a{alpha:g}", leave=False):
        batch_questions = questions[start : start + batch_size]
        rendered = _render_questions(bundle, batch_questions, generation_cfg, prompt_suffix)
        original_padding_side = bundle.tokenizer.padding_side
        bundle.tokenizer.padding_side = "left"
        encoded = bundle.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(generation_cfg.get("prompt_max_length", 2048)),
        )
        bundle.tokenizer.padding_side = original_padding_side
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
        handle = None
        if directions is not None and float(alpha) != 0.0:
            batch_directions = directions[start : start + len(batch_questions)]
            direction = torch.as_tensor(
                float(alpha) * batch_directions,
                device=bundle.device,
                dtype=next(bundle.model.parameters()).dtype,
            )
            prompt_delta = encoded["attention_mask"].unsqueeze(-1) * direction[:, None, :]
            prefill_applied = False

            def input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                nonlocal prefill_applied
                if not inputs or not torch.is_tensor(inputs[0]):
                    return inputs
                hidden = inputs[0]
                if hidden.shape[:2] == prompt_delta.shape[:2] and not prefill_applied:
                    delta = prompt_delta.to(dtype=hidden.dtype)
                    prefill_applied = True
                elif apply_on == "prompt_and_decode" and prefill_applied:
                    delta = direction[:, None, :].to(dtype=hidden.dtype).expand(
                        hidden.shape[0], hidden.shape[1], -1
                    )
                else:
                    return inputs
                return (hidden + delta,) + inputs[1:]

            handle = target_layer.register_forward_pre_hook(input_hook)
        try:
            kwargs = _build_generate_kwargs(generation_cfg, return_entropy=False)
            kwargs["pad_token_id"] = bundle.tokenizer.pad_token_id
            kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id
            with torch.inference_mode():
                output = bundle.model.generate(**encoded, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        prompt_width = int(encoded["input_ids"].shape[1])
        for row_index in range(len(batch_questions)):
            generated = output.sequences[row_index, prompt_width:].detach().cpu()
            token_ids = generated.tolist()
            if bundle.tokenizer.eos_token_id in token_ids:
                count = token_ids.index(bundle.tokenizer.eos_token_id) + 1
            else:
                count = len(token_ids)
            responses.append(bundle.tokenizer.decode(generated, skip_special_tokens=True).strip())
            generated_counts.append(int(count))
        del encoded, output
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()
    return responses, generated_counts


def _quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    quality = pd.DataFrame([response_quality(value) for value in frame["response_text"]])
    return pd.concat([frame.reset_index(drop=True), quality], axis=1)


def _raw_cache_matches_eval(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        cached = pd.read_parquet(path, columns=["example_id"])
    except (OSError, ValueError, KeyError):
        return False
    return cached["example_id"].astype(str).tolist() == expected_ids


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    set_global_seed(seed)
    generation_cfg = dict(config["generation"])
    generation_cfg["temperature"] = float(args.temperature)
    generation_cfg["do_sample"] = True
    generation_cfg["max_new_tokens"] = int(args.max_new_tokens)
    generation_cfg["batch_size"] = int(args.generation_batch_size)
    prompt_suffix = str(config.get("steering", {}).get("prompt_suffix", "\nAnswer:"))
    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    topology_dir = Path(args.topology_root).resolve() / model_slug
    output_dir = ensure_dir(Path(args.artifact_root).resolve() / model_slug / args.protocol)
    cache_dir = ensure_dir(output_dir / "cache")
    raw_dir = ensure_dir(output_dir / "raw")
    split_column = "split_random80" if args.protocol == "random80" else "split_official"

    metadata = pd.read_parquet(topology_dir / "falseqa_paired_questions.parquet")
    feature_frame = pd.read_parquet(topology_dir / "falseqa_h0_all_layers.parquet")
    topology_tensor, layers, tensor_example_ids = build_topology_tensor(metadata, feature_frame)
    if tensor_example_ids != metadata["example_id"].astype(str).tolist():
        raise ValueError("Topology tensor order differs from metadata order")
    auto_layer, layer_metrics = select_layer_with_train_validation(
        metadata,
        topology_tensor,
        layers,
        split_column=split_column,
        seed=seed,
    )
    topology_layer = int(auto_layer if args.layer < 0 else args.layer)
    if topology_layer not in layers:
        raise ValueError(f"Requested layer {topology_layer} is unavailable; layers={layers}")
    if topology_layer >= max(layers):
        raise ValueError(
            "Layer-input steering requires an H0 output layer with a following decoder layer; "
            f"got topology layer {topology_layer}."
        )
    steering_layer = topology_layer + 1
    write_parquet(layer_metrics, output_dir / "train_validation_layer_selection.parquet")

    train_false, train_topology, all_eval_false, all_eval_topology, _topology_scaler = topology_neighbor_matrices(
        metadata,
        topology_tensor,
        split_column=split_column,
    )
    eval_false = all_eval_false
    eval_topology = all_eval_topology
    if args.eval_n > 0 and args.eval_n < len(eval_false):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(eval_false), size=int(args.eval_n), replace=False))
        eval_false = eval_false.iloc[selected].reset_index(drop=True)
        eval_topology = eval_topology[selected]

    activation_frame = pd.concat(
        [metadata.loc[metadata[split_column].eq("train")], all_eval_false],
        ignore_index=True,
    ).drop_duplicates("example_id").reset_index(drop=True)
    activation_path = cache_dir / (
        f"topology_layer_{topology_layer:02d}__steering_layer_{steering_layer:02d}__"
        "layer_input_prompt_vectors.npz"
    )
    bundle: HFModelBundle | None = None
    if activation_path.exists() and not args.force_activations:
        payload = dict(np.load(activation_path, allow_pickle=False))
    else:
        bundle = load_hf_model(config["model"], generation_cfg)
        payload = _extract_layer_input_vectors(
            bundle=bundle,
            frame=activation_frame,
            generation_cfg=generation_cfg,
            prompt_suffix=prompt_suffix,
            layer=steering_layer,
            batch_size=int(args.activation_batch_size),
        )
        np.savez_compressed(activation_path, **payload)

    hidden_by_example = {
        str(example_id): vector
        for example_id, vector in zip(payload["example_ids"], payload["vectors"], strict=True)
    }
    train_differences = paired_hidden_differences(train_false, hidden_by_example, metadata)
    global_direction = train_differences.mean(axis=0).astype(np.float32, copy=False)
    eval_ids = eval_false["example_id"].astype(str).tolist()
    activation_index = {str(value): index for index, value in enumerate(payload["example_ids"])}
    eval_fro_norms = np.asarray([payload["fro_norms"][activation_index[value]] for value in eval_ids])
    eval_token_counts = np.asarray([payload["token_counts"][activation_index[value]] for value in eval_ids])

    direction_sets: list[tuple[str, int, np.ndarray]] = [
        ("global_paired_mean_diff_raw", 0, np.repeat(global_direction[None, :], len(eval_false), axis=0))
    ]
    for k in args.neighbor_ks:
        directions, indices, distances = build_paired_local_directions(
            train_topology,
            eval_topology,
            train_differences,
            neighbor_k=int(k),
        )
        direction_sets.append(("topology_local_paired_mean_diff_raw", int(k), directions))
        diagnostics = eval_false.loc[:, ["example_id", "pair_id", "question"]].copy()
        diagnostics["neighbor_k"] = int(k)
        diagnostics["neighbor_pair_ids"] = [
            train_false.iloc[row_indices]["pair_id"].astype(str).tolist() for row_indices in indices
        ]
        diagnostics["neighbor_distances"] = [row.tolist() for row in distances]
        diagnostics["direction_norm"] = np.linalg.norm(directions, axis=1)
        write_parquet(diagnostics, cache_dir / f"topology_neighbors_k{int(k)}.parquet")
        np.save(cache_dir / f"topology_local_directions_k{int(k)}.npy", directions)
    np.save(cache_dir / "global_paired_direction.npy", global_direction)

    if bundle is None:
        bundle = load_hf_model(config["model"], generation_cfg)
    try:
        base_path = raw_dir / "base__raw.parquet"
        if not args.force_generation and _raw_cache_matches_eval(base_path, eval_ids):
            base_frame = pd.read_parquet(base_path)
        else:
            set_global_seed(seed)
            responses, counts = _generate(
                bundle=bundle,
                questions=eval_false["question"].astype(str).tolist(),
                directions=None,
                alpha=0.0,
                generation_cfg=generation_cfg,
                prompt_suffix=prompt_suffix,
                layer=steering_layer,
                apply_on=str(args.steering_apply_on),
            )
            base_frame = eval_false.loc[
                :, ["example_id", "pair_id", "question", "reference_answer", "source_split"]
            ].copy()
            base_frame["response_text"] = responses
            base_frame["generated_token_count"] = counts
            base_frame["strategy"] = "base"
            base_frame["neighbor_k"] = 0
            base_frame["alpha"] = 0.0
            base_frame["topology_selection_layer"] = topology_layer
            base_frame["steering_layer"] = steering_layer
            base_frame["direction_norm"] = 0.0
            base_frame["delta_h_fro_over_h_fro"] = 0.0
            base_frame = _quality_columns(base_frame)
            write_parquet(base_frame, base_path)

        base_lookup = base_frame.set_index("example_id")["response_text"]
        for strategy, neighbor_k, directions in direction_sets:
            direction_norms = np.linalg.norm(directions, axis=1)
            for alpha in args.alphas:
                output_path = raw_dir / (
                    f"{strategy}__k{neighbor_k}__alpha{_float_slug(float(alpha))}__raw.parquet"
                )
                if not args.force_generation and _raw_cache_matches_eval(output_path, eval_ids):
                    continue
                set_global_seed(seed)
                responses, counts = _generate(
                    bundle=bundle,
                    questions=eval_false["question"].astype(str).tolist(),
                    directions=directions,
                    alpha=float(alpha),
                    generation_cfg=generation_cfg,
                    prompt_suffix=prompt_suffix,
                    layer=steering_layer,
                    apply_on=str(args.steering_apply_on),
                )
                frame = eval_false.loc[
                    :, ["example_id", "pair_id", "question", "reference_answer", "source_split"]
                ].copy()
                frame["base_response_text"] = frame["example_id"].map(base_lookup)
                frame["response_text"] = responses
                frame["generated_token_count"] = counts
                frame["strategy"] = strategy
                frame["neighbor_k"] = int(neighbor_k)
                frame["alpha"] = float(alpha)
                frame["topology_selection_layer"] = int(topology_layer)
                frame["steering_layer"] = int(steering_layer)
                frame["direction_norm"] = direction_norms
                delta_fro = abs(float(alpha)) * np.sqrt(eval_token_counts) * direction_norms
                frame["delta_h_fro_over_h_fro"] = delta_fro / np.maximum(eval_fro_norms, 1e-12)
                frame = _quality_columns(frame)
                write_parquet(frame, output_path)
    finally:
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(
        output_dir / "metadata.json",
        {
            "model": model_name,
            "protocol": args.protocol,
            "split_column": split_column,
            "train_pair_count": int(train_false["pair_id"].nunique()),
            "eval_false_question_count": int(len(eval_false)),
            "topology_features": list((
                "h0_mean_persistence",
                "h0_persistence_entropy",
                "h0_top5_persistence_fraction",
            )),
            "topology_layers": layers,
            "topology_selection_layer": topology_layer,
            "steering_layer": steering_layer,
            "auto_selected_layer": auto_layer,
            "direction": "mean(h_false - h_corrected) over topology-nearest training pairs",
            "neighbor_ks": [int(value) for value in args.neighbor_ks],
            "alphas": [float(value) for value in args.alphas],
            "intervention_site": f"decoder_layer_{steering_layer}_input_equals_layer_{topology_layer}_output",
            "steering_apply_on": str(args.steering_apply_on),
            "temperature": float(args.temperature),
            "max_new_tokens": int(args.max_new_tokens),
            "judge": "not run by this generation script",
        },
    )
    print(
        f"wrote {output_dir} topology_layer={topology_layer} "
        f"steering_layer={steering_layer} eval_n={len(eval_false)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
