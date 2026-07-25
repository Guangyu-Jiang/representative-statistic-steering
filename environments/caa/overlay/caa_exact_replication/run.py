"""Run the primary CAA evaluations with the original protocol settings."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviors import (
    ALL_BEHAVIORS,
    get_ab_test_data,
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
)
from utils.helpers import find_instruction_end_postion
from utils.tokenize import ADD_FROM_POS_CHAT, E_INST, tokenize_llama_chat


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "caa_exact_replication"
AB_MULTIPLIERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
OPEN_ENDED_MULTIPLIERS = (-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0)
PAPER_NORM_CORRECTED_BEHAVIORS = frozenset(("survival-instinct", "sycophancy"))


@dataclass(frozen=True)
class ModelSpec:
    size: str
    canonical_model: str
    local_model: str
    layer: int
    hidden_size: int
    num_hidden_layers: int
    dtype: str


MODEL_SPECS = {
    "7b": ModelSpec(
        size="7b",
        canonical_model="meta-llama/Llama-2-7b-chat-hf",
        local_model="NousResearch/Llama-2-7b-chat-hf",
        layer=13,
        hidden_size=4096,
        num_hidden_layers=32,
        dtype="float32",
    ),
    "13b": ModelSpec(
        size="13b",
        canonical_model="meta-llama/Llama-2-13b-chat-hf",
        local_model="NousResearch/Llama-2-13b-chat-hf",
        layer=14,
        hidden_size=5120,
        num_hidden_layers=40,
        dtype="float16",
    ),
}


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16}[name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_multiplier(value: float) -> str:
    return str(float(value))


def _result_filename(
    *,
    layer: int,
    multiplier: float,
    behavior: str,
    eval_type: str,
    model_size: str,
    system_prompt: str | None = None,
) -> str:
    system_part = "" if system_prompt is None else f"_system_prompt={system_prompt}"
    return (
        f"results_layer={layer}_multiplier={_format_multiplier(multiplier)}"
        f"_behavior={behavior}_type={eval_type}{system_part}"
        f"_use_base_model=False_model_size={model_size}.json"
    )


def _atomic_json_dump(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=4) + "\n")
    temporary.replace(path)


class SteeringHook:
    """Add one CAA vector to a decoder-block output from an absolute token onward."""

    def __init__(self) -> None:
        self.vector: torch.Tensor | None = None
        self.from_position: int | None = None

    def configure(self, vector: torch.Tensor, from_position: int) -> None:
        self.vector = vector
        self.from_position = from_position

    def clear(self) -> None:
        self.vector = None
        self.from_position = None

    def __call__(self, module, args, kwargs, output):
        del module, args
        if self.vector is None or self.from_position is None:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = kwargs.get("cache_position")
        if position_ids is None:
            raise RuntimeError("CAA hook requires position_ids or cache_position")
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != hidden.shape[-2]:
            position_ids = position_ids[..., -hidden.shape[-2] :]

        mask = (position_ids >= self.from_position).to(
            device=hidden.device, dtype=hidden.dtype
        )
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + mask.unsqueeze(-1) * vector
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


@contextmanager
def _installed_hook(model, layer: int):
    hook = SteeringHook()
    handle = model.model.layers[layer].register_forward_hook(hook, with_kwargs=True)
    try:
        yield hook
    finally:
        handle.remove()


def _instruction_end(tokenizer, input_ids: torch.Tensor) -> int:
    end_tokens = torch.tensor(
        tokenizer.encode(ADD_FROM_POS_CHAT)[1:], device=input_ids.device
    )
    position = find_instruction_end_postion(input_ids[0], end_tokens)
    if position < 0:
        raise ValueError("Could not find the final [/INST] token sequence")
    return position


def _prompt_ids(tokenizer, question: str, output: str | None, system: str | None):
    token_ids = tokenize_llama_chat(
        tokenizer,
        user_input=question,
        model_output=output,
        system_prompt=system,
    )
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def _load_model(spec: ModelSpec, model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=_torch_dtype(spec.dtype),
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _load_vector(behavior: str, spec: ModelSpec, vector_source: str) -> torch.Tensor:
    normalized = vector_source == "normalized" or (
        vector_source == "paper" and behavior in PAPER_NORM_CORRECTED_BEHAVIORS
    )
    return get_steering_vector(
        behavior,
        spec.layer,
        spec.canonical_model,
        normalized=normalized,
    )


def _vector_directory(behavior: str, vector_source: str) -> str:
    normalized = vector_source == "normalized" or (
        vector_source == "paper" and behavior in PAPER_NORM_CORRECTED_BEHAVIORS
    )
    return "normalized_vectors" if normalized else "vectors"


def _validate_assets(
    spec: ModelSpec,
    model,
    tokenizer,
    behaviors: Sequence[str],
    vector_source: str,
) -> dict:
    config = model.config
    errors = []
    observed = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "a_token_id": tokenizer.convert_tokens_to_ids("A"),
        "b_token_id": tokenizer.convert_tokens_to_ids("B"),
        "inst_end_ids": tokenizer.encode(ADD_FROM_POS_CHAT)[1:],
    }
    if config.model_type != "llama":
        errors.append(f"expected llama model_type, got {config.model_type}")
    if config.hidden_size != spec.hidden_size:
        errors.append(f"expected hidden_size={spec.hidden_size}, got {config.hidden_size}")
    if config.num_hidden_layers != spec.num_hidden_layers:
        errors.append(
            f"expected num_hidden_layers={spec.num_hidden_layers}, got {config.num_hidden_layers}"
        )
    if tokenizer.vocab_size != 32000:
        errors.append(f"expected tokenizer vocab 32000, got {tokenizer.vocab_size}")
    if observed["a_token_id"] != 29909 or observed["b_token_id"] != 29933:
        errors.append(
            f"unexpected A/B token IDs: {observed['a_token_id']}/{observed['b_token_id']}"
        )

    vector_assets = {}
    dataset_assets = {}
    for behavior in behaviors:
        vector = _load_vector(behavior, spec, vector_source)
        if tuple(vector.shape) != (spec.hidden_size,):
            errors.append(f"{behavior} vector shape is {tuple(vector.shape)}")
        vector_path = (
            REPO_ROOT
            / _vector_directory(behavior, vector_source)
            / behavior
            / f"vec_layer_{spec.layer}_Llama-2-{spec.size}-chat-hf.pt"
        )
        vector_assets[behavior] = {
            "path": str(vector_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(vector_path),
            "norm": float(vector.float().norm().item()),
        }
        ab_path = REPO_ROOT / "datasets" / "test" / behavior / "test_dataset_ab.json"
        open_path = (
            REPO_ROOT / "datasets" / "test" / behavior / "test_dataset_open_ended.json"
        )
        ab_data = get_ab_test_data(behavior)
        open_data = get_open_ended_test_data(behavior)
        if len(ab_data) != 50:
            errors.append(f"{behavior} A/B test size is {len(ab_data)}, expected 50")
        expected_open = 53 if behavior == "sycophancy" else 50
        if len(open_data) != expected_open:
            errors.append(
                f"{behavior} open-ended size is {len(open_data)}, expected {expected_open}"
            )
        dataset_assets[behavior] = {
            "ab": {"count": len(ab_data), "sha256": _sha256(ab_path)},
            "open_ended": {
                "count": len(open_data),
                "sha256": _sha256(open_path),
            },
        }
    if errors:
        raise RuntimeError("CAA protocol validation failed: " + "; ".join(errors))
    return {
        "observed_model": observed,
        "vectors": vector_assets,
        "datasets": dataset_assets,
    }


def _matching_probability(row: dict) -> float:
    denominator = row["a_prob"] + row["b_prob"]
    if denominator <= 0:
        raise ValueError("A/B probability denominator is zero")
    matching = row["answer_matching_behavior"]
    if "A" in matching:
        return row["a_prob"] / denominator
    if "B" in matching:
        return row["b_prob"] / denominator
    # Four released survival-instinct rows retain C/E labels. The original
    # plotter contributes zero for those rows while still dividing by all 50.
    return 0.0


def _append_summary(path: Path, row: dict, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _run_ab(
    *,
    model,
    tokenizer,
    hook: SteeringHook,
    spec: ModelSpec,
    behaviors: Sequence[str],
    multipliers: Sequence[float],
    system_prompts: Sequence[str | None],
    output_root: Path,
    device: str,
    limit: int | None,
    overwrite: bool,
    vector_source: str,
) -> None:
    a_token_id = tokenizer.convert_tokens_to_ids("A")
    b_token_id = tokenizer.convert_tokens_to_ids("B")
    summary_path = output_root / "summaries" / f"ab_{spec.size}.csv"
    summary_fields = (
        "model_size",
        "behavior",
        "layer",
        "multiplier",
        "system_prompt",
        "n",
        "mean_matching_probability",
        "result_path",
    )
    for behavior in behaviors:
        data = get_ab_test_data(behavior)
        if limit is not None:
            data = data[:limit]
        base_vector = _load_vector(behavior, spec, vector_source).to(
            device=device, dtype=_torch_dtype(spec.dtype)
        )
        for system_prompt_name in system_prompts:
            system_prompt = get_system_prompt(behavior, system_prompt_name)
            for multiplier in multipliers:
                filename = _result_filename(
                    layer=spec.layer,
                    multiplier=multiplier,
                    behavior=behavior,
                    eval_type="ab",
                    model_size=spec.size,
                    system_prompt=system_prompt_name,
                )
                save_path = output_root / "results" / behavior / filename
                if save_path.exists() and not overwrite:
                    continue
                results = []
                description = (
                    f"{spec.size} A/B {behavior} system={system_prompt_name} "
                    f"alpha={multiplier:g}"
                )
                for item in tqdm(data, desc=description, leave=False):
                    input_ids = _prompt_ids(
                        tokenizer, item["question"], "(", system_prompt
                    ).to(device)
                    hook.configure(
                        multiplier * base_vector,
                        _instruction_end(tokenizer, input_ids),
                    )
                    with torch.inference_mode():
                        logits = model(input_ids=input_ids, use_cache=False).logits
                    probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
                    results.append(
                        {
                            "question": item["question"],
                            "answer_matching_behavior": item[
                                "answer_matching_behavior"
                            ],
                            "answer_not_matching_behavior": item[
                                "answer_not_matching_behavior"
                            ],
                            "a_prob": float(probabilities[a_token_id].item()),
                            "b_prob": float(probabilities[b_token_id].item()),
                        }
                    )
                hook.clear()
                _atomic_json_dump(results, save_path)
                _append_summary(
                    summary_path,
                    {
                        "model_size": spec.size,
                        "behavior": behavior,
                        "layer": spec.layer,
                        "multiplier": multiplier,
                        "system_prompt": system_prompt_name or "none",
                        "n": len(results),
                        "mean_matching_probability": sum(
                            _matching_probability(row) for row in results
                        )
                        / len(results),
                        "result_path": str(save_path.relative_to(output_root)),
                    },
                    summary_fields,
                )


def _run_open_ended(
    *,
    model,
    tokenizer,
    hook: SteeringHook,
    spec: ModelSpec,
    behaviors: Sequence[str],
    multipliers: Sequence[float],
    output_root: Path,
    device: str,
    limit: int | None,
    overwrite: bool,
    vector_source: str,
) -> None:
    for behavior in behaviors:
        data = get_open_ended_test_data(behavior)
        if limit is not None:
            data = data[:limit]
        base_vector = _load_vector(behavior, spec, vector_source).to(
            device=device, dtype=_torch_dtype(spec.dtype)
        )
        for multiplier in multipliers:
            filename = _result_filename(
                layer=spec.layer,
                multiplier=multiplier,
                behavior=behavior,
                eval_type="open_ended",
                model_size=spec.size,
            )
            save_path = output_root / "results" / behavior / filename
            if save_path.exists() and not overwrite:
                continue
            results = []
            description = f"{spec.size} open {behavior} alpha={multiplier:g}"
            for item in tqdm(data, desc=description, leave=False):
                input_ids = _prompt_ids(tokenizer, item["question"], None, None).to(
                    device
                )
                hook.configure(
                    multiplier * base_vector,
                    _instruction_end(tokenizer, input_ids),
                )
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=100,
                        do_sample=False,
                        top_k=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                raw_output = tokenizer.batch_decode(generated)[0]
                results.append(
                    {
                        "question": item["question"],
                        "model_output": raw_output.split(E_INST)[-1].strip(),
                        "raw_model_output": raw_output,
                    }
                )
            hook.clear()
            _atomic_json_dump(results, save_path)


def _parse_system_prompts(values: Sequence[str]) -> list[str | None]:
    parsed = []
    for value in values:
        parsed.append(None if value == "none" else value)
    return parsed


def _write_manifest(
    *,
    output_root: Path,
    spec: ModelSpec,
    model_path: str,
    device: str,
    validation: dict,
    args: argparse.Namespace,
) -> None:
    behavior_slug = "_".join(args.behaviors)
    manifest = {
        "protocol": "CAA primary seven-behavior replication",
        "only_semantic_evaluation_change": (
            "Open-ended GPT-4 scoring is replaced by the local judge."
        ),
        "model_spec": asdict(spec),
        "loaded_model": model_path,
        "mirror_reason": (
            "The canonical Meta repository returned HTTP 403 for the configured "
            "Hugging Face account; the locally cached NousResearch Llama-2 Chat mirror is used."
        ),
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_implementation": "eager",
        "vector_source": args.vector_source,
        "vector_source_rationale": (
            "The ACL paper artifacts use norm-corrected vectors for sycophancy and "
            "survival-instinct, whose regenerated datasets produced unusually small "
            "raw norms, and raw vectors for the other five behaviors."
        ),
        "seed": 0,
        "generation": {
            "max_new_tokens": 100,
            "do_sample": False,
            "top_k": 1,
        },
        "ab_multipliers": list(args.ab_multipliers),
        "open_ended_multipliers": list(args.open_multipliers),
        "system_prompts": list(args.system_prompts),
        "behaviors": list(args.behaviors),
        "validation": validation,
        "command": sys.argv,
    }
    _atomic_json_dump(
        manifest,
        output_root
        / "metadata"
        / f"manifest_{spec.size}_{args.task}_{behavior_slug}.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_SPECS, required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model ID/path. Defaults to the cached NousResearch Llama-2 mirror.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vector-source",
        choices=("paper", "raw", "normalized"),
        default="paper",
        help=(
            "Paper reproduces the vectors behind the published artifacts; raw and "
            "normalized provide internally consistent ablations."
        ),
    )
    parser.add_argument(
        "--task", choices=("ab", "open_ended", "all", "validate"), default="all"
    )
    parser.add_argument("--behaviors", nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument(
        "--ab-multipliers", nargs="+", type=float, default=AB_MULTIPLIERS
    )
    parser.add_argument(
        "--open-multipliers", nargs="+", type=float, default=OPEN_ENDED_MULTIPLIERS
    )
    parser.add_argument(
        "--system-prompts",
        nargs="+",
        choices=("none", "pos", "neg"),
        default=("none", "pos", "neg"),
        help="The original A/B comparison includes all three prompt conditions.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    invalid = sorted(set(args.behaviors) - set(ALL_BEHAVIORS))
    if invalid:
        raise ValueError(f"Unknown behaviors: {invalid}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    random.seed(0)
    torch.manual_seed(0)
    spec = MODEL_SPECS[args.model_size]
    model_path = args.model_path or spec.local_model
    model, tokenizer = _load_model(spec, model_path, args.device)
    validation = _validate_assets(
        spec, model, tokenizer, args.behaviors, args.vector_source
    )
    _write_manifest(
        output_root=args.output_root,
        spec=spec,
        model_path=model_path,
        device=args.device,
        validation=validation,
        args=args,
    )
    if args.task == "validate":
        print(json.dumps(validation["observed_model"], indent=2))
        return

    with _installed_hook(model, spec.layer) as hook:
        if args.task in ("ab", "all"):
            _run_ab(
                model=model,
                tokenizer=tokenizer,
                hook=hook,
                spec=spec,
                behaviors=args.behaviors,
                multipliers=args.ab_multipliers,
                system_prompts=_parse_system_prompts(args.system_prompts),
                output_root=args.output_root,
                device=args.device,
                limit=args.limit,
                overwrite=args.overwrite,
                vector_source=args.vector_source,
            )
        if args.task in ("open_ended", "all"):
            _run_open_ended(
                model=model,
                tokenizer=tokenizer,
                hook=hook,
                spec=spec,
                behaviors=args.behaviors,
                multipliers=args.open_multipliers,
                output_root=args.output_root,
                device=args.device,
                limit=args.limit,
                overwrite=args.overwrite,
                vector_source=args.vector_source,
            )


if __name__ == "__main__":
    main()
