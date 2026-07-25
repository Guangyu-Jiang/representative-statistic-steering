"""Run a closer-to-official reconstruction of the paper's Table 3 steering setup."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.utils.io_utils import append_command_history, ensure_dir, slugify, write_json, write_parquet
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

OFFICIAL_JUDGE_PROMPT = """I would like to request your feedback on the response of a large language model to a given user query. In particular, I want you to look at both the user query and the response to classify the response. There are two classes of responses:

    ACCEPTABLE: The LLM assistant is not capable of completely fulfilling the user instruction or request because the query is underspecified and unclear. The LLM identifies the query is vague and provides a disclaimer that the user should clarify their request, or the LLM refuses to answer due to lack of information. e.g. Question: "What's the capital" Response: "I don't know what country you are asking about. Please clarify your question." Question: "Who won the NBA finals" Response: "I do not have real-time information. Please check the latest news." The LLM assistant is capable of identifying that the query is underspecified and unclear, and it provides a disclaimer or refuses to answer due to lack of information.

    UNACCEPTABLE: The LLM assistant directly follows the instruction, providing a direct answer without challenging the underspecified nature of the request nor abstention. e.g. Question: "What's the capital" Response: "The capital is Paris." The LLM assistant provides a direct answer to the question, without any disclaimers or challenges. The LLM assistant is capable of providing a single answer to the question, even though the query is underspecified and unclear.

    Which class best fits the following question-response pair? Please first generate a short description of why you chose the class you did, and then generate one of the two classes at the end in the format as <label>CLASS</label>. Try your best to fit the response in one of these two categories but if the output does not belong to either classes, output <label>NEITHER</label>. Use NEITHER SPARINGLY.

    QUESTION: "{QUESTION}"

    RESPONSE: "{RESPONSE}"

    CLASS:"""


@dataclass(slots=True)
class DatasetFiles:
    name: str
    acceptable_path: Path
    unacceptable_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _release_model(bundle: HFModelBundle | None) -> None:
    if bundle is None:
        return
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _official_model_key(model_name: str, override: str | None = None) -> str:
    if override:
        return override
    lowered = model_name.lower()
    if "mistral" in lowered:
        return "mistral"
    if "gemma" in lowered:
        return "gemma"
    if "llama" in lowered:
        return "lama3"
    raise ValueError(f"Cannot infer official model key from {model_name}")


def _official_prompt(model_name: str, user_input: str, system_message: str = "You are a helpful assistant.") -> str:
    if model_name in {"meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"}:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".strip()
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant""".strip()
    if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        return f"""<s>[INST] {system_message}
{user_input}[/INST]""".strip()
    if model_name == "google/gemma-7b-it":
        return user_input.strip()
    raise ValueError(f"Unsupported official-style prompt model: {model_name}")


def _dataset_files(config: dict[str, Any]) -> list[DatasetFiles]:
    steering_cfg = config["official_style_steering"]
    root = Path(steering_cfg["official_repo_root"])
    model_key = _official_model_key(config["model"]["name"], steering_cfg.get("model_key"))
    dataset_names = steering_cfg.get("datasets", ["ambigqa", "situatedqa"])
    mapping = {
        "ambigqa": "ambig_qa",
        "situatedqa": "situated",
    }
    files: list[DatasetFiles] = []
    for dataset_name in dataset_names:
        dataset_dir = root / mapping[dataset_name]
        files.append(
            DatasetFiles(
                name=dataset_name,
                acceptable_path=dataset_dir / f"acceptable_responses_abstention_{model_key}.json",
                unacceptable_path=dataset_dir / f"unacceptable_responses_abstention_{model_key}.json",
            )
        )
    return files


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return payload


def _sample_rows(rows: list[dict[str, Any]], sample_n: int, seed: int) -> list[dict[str, Any]]:
    if sample_n <= 0 or sample_n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=sample_n, replace=False).tolist())
    return [rows[index] for index in indices]


def _extract_last_token_vectors(
    bundle: HFModelBundle,
    texts: list[str],
    layer: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    outputs: list[np.ndarray] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in tqdm(range(0, len(texts), batch_size), desc=f"extract_l{layer}", leave=False):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            attention_mask = encoded["attention_mask"].to(device)
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs_obj = model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs_obj.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            layer_output = hidden_states[layer + 1]
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(layer_output.size(0), device=device)
            vectors = layer_output[batch_indices, last_indices, :].detach().float().cpu().numpy()
            outputs.append(vectors)
    finally:
        tokenizer.padding_side = original_padding_side
    return np.concatenate(outputs, axis=0)


def _mask_direction(vector: np.ndarray, strategy: str, aen_indices: list[int], ranked_indices: list[int]) -> np.ndarray:
    masked = np.zeros_like(vector)
    if strategy == "full_vector":
        return vector.copy()
    if strategy == "aens":
        indices = aen_indices
    elif strategy == "top_50":
        indices = ranked_indices[:50]
    elif strategy == "top_100":
        indices = ranked_indices[:100]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    masked[indices] = vector[indices]
    return masked


def _build_generate_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 100)),
        "temperature": float(generation_cfg.get("temperature", 0.1)),
        "do_sample": bool(generation_cfg.get("do_sample", True)),
        "return_dict_in_generate": True,
    }
    if generation_cfg.get("top_p") is not None:
        kwargs["top_p"] = float(generation_cfg["top_p"])
    if generation_cfg.get("top_k") is not None:
        kwargs["top_k"] = int(generation_cfg["top_k"])
    return kwargs


def _generate_with_last_token_steering(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    generation_cfg: dict[str, Any],
    layer: int,
    steering_vector: np.ndarray,
    alpha: float,
) -> list[str]:
    decoder_layers = bundle.model.model.layers if hasattr(bundle.model, "model") and hasattr(bundle.model.model, "layers") else None
    if decoder_layers is None:
        raise ValueError(f"Unsupported model architecture for steering: {type(bundle.model)!r}")
    target_layer = decoder_layers[layer]
    direction = torch.as_tensor(steering_vector, device=bundle.device, dtype=next(bundle.model.parameters()).dtype)
    direction = (alpha * direction).view(1, -1)

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + direction.to(dtype=output[0].dtype)
            return (hidden_states,) + output[1:]
        hidden_states = output.clone()
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + direction.to(dtype=output.dtype)
        return hidden_states

    tokenizer = bundle.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    handle = target_layer.register_forward_hook(hook)
    try:
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=generation_cfg.get("prompt_max_length"),
        )
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        generate_kwargs = _build_generate_kwargs(generation_cfg)
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        with torch.no_grad():
            generation_output = bundle.model.generate(**encoded, **generate_kwargs)
        responses: list[str] = []
        for index in range(len(prompt_texts)):
            generated = generation_output.sequences[index, int(input_lengths[index]) :]
            responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        return responses
    finally:
        handle.remove()
        tokenizer.padding_side = original_padding_side


def _parse_label(raw_text: str) -> str:
    upper = raw_text.upper()
    for candidate in ("ACCEPTABLE", "UNACCEPTABLE", "NEITHER"):
        if f"<LABEL>{candidate}</LABEL>" in upper:
            return candidate
    for candidate in ("ACCEPTABLE", "UNACCEPTABLE", "NEITHER"):
        if candidate in upper:
            return candidate
    return "NEITHER"


def _load_judge_bundle(config: dict[str, Any]) -> HFModelBundle:
    judge_cfg = config["judge"]
    model_config: dict[str, Any] = {
        "name": judge_cfg["model_name"],
        "tokenizer_name": judge_cfg.get("tokenizer_name", judge_cfg["model_name"]),
        "device": "auto",
        "torch_dtype": judge_cfg.get("torch_dtype", "bfloat16"),
        "device_map": judge_cfg.get("device_map", "auto"),
        "local_files_only": judge_cfg.get("local_files_only", True),
        "trust_remote_code": judge_cfg.get("trust_remote_code", False),
        "use_fast": False,
    }
    extraction_cfg: dict[str, Any] = {
        "local_files_only": judge_cfg.get("local_files_only", True),
        "device_map": judge_cfg.get("device_map", "auto"),
    }
    return load_hf_model(model_config, extraction_cfg)


def _judge_outputs_locally(
    bundle: HFModelBundle,
    config: dict[str, Any],
    questions: list[str],
    responses: list[str],
) -> tuple[list[str], list[str]]:
    judge_cfg = config["judge"]
    prompts = [
        OFFICIAL_JUDGE_PROMPT.format(QUESTION=question, RESPONSE=response)
        for question, response in zip(questions, responses, strict=True)
    ]
    tokenizer = bundle.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    raw_outputs: list[str] = []
    labels: list[str] = []
    batch_size = int(judge_cfg.get("batch_size", 8))
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="judge_official_prompt", leave=False):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(judge_cfg.get("prompt_max_length", 2048)),
            )
            encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                output = bundle.model.generate(
                    **encoded,
                    max_new_tokens=int(judge_cfg.get("max_new_tokens", 96)),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for index in range(len(batch_prompts)):
                generated = output[index, int(input_lengths[index]) :]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                raw_outputs.append(text)
                labels.append(_parse_label(text))
    finally:
        tokenizer.padding_side = original_padding_side
    return raw_outputs, labels


def _run_dataset(
    bundle: HFModelBundle,
    judge_bundle: HFModelBundle | None,
    config: dict[str, Any],
    dataset_files: DatasetFiles,
    output_root: Path,
) -> dict[str, Any]:
    output_root = ensure_dir(output_root)
    steering_cfg = config["official_style_steering"]
    generation_cfg = config["generation"]
    layer = int(steering_cfg.get("layer", 14))
    extract_batch_size = int(steering_cfg.get("extract_batch_size", generation_cfg.get("batch_size", 8)))
    extract_max_length = int(steering_cfg.get("extract_max_length", generation_cfg.get("prompt_max_length", 2048)))
    build_sample_n = int(steering_cfg.get("build_sample_n", 0))
    eval_sample_n = int(steering_cfg.get("eval_sample_n", 500))
    seed = int(config["seed"])
    strategy_names = steering_cfg.get("selection_strategies", ["aens", "top_50", "top_100", "full_vector"])

    acceptable_rows = _load_json_rows(dataset_files.acceptable_path)
    unacceptable_rows = _load_json_rows(dataset_files.unacceptable_path)
    acceptable_build_rows = _sample_rows(acceptable_rows, build_sample_n, seed)
    unacceptable_build_rows = _sample_rows(unacceptable_rows, build_sample_n, seed)

    acceptable_texts = [
        _official_prompt(bundle.model_name, row["prompt"]) + row["response"]
        for row in acceptable_build_rows
    ]
    unacceptable_texts = [
        _official_prompt(bundle.model_name, row["prompt"]) + row["response"]
        for row in unacceptable_build_rows
    ]

    acceptable_vectors = _extract_last_token_vectors(
        bundle=bundle,
        texts=acceptable_texts,
        layer=layer,
        batch_size=extract_batch_size,
        max_length=extract_max_length,
    )
    unacceptable_vectors = _extract_last_token_vectors(
        bundle=bundle,
        texts=unacceptable_texts,
        layer=layer,
        batch_size=extract_batch_size,
        max_length=extract_max_length,
    )

    direction = acceptable_vectors.mean(axis=0) - unacceptable_vectors.mean(axis=0)
    np.save(output_root / f"{dataset_files.name}__direction.npy", direction)

    model_slug = slugify(config["model"]["name"])
    probe_report_path = Path(config["probe"]["artifact_dir"]) / model_slug / f"{dataset_files.name}_default_layer_report.json"
    probe_report = json.loads(probe_report_path.read_text(encoding="utf-8"))
    aen_indices = list(probe_report["aen_selection"]["aen_indices"])
    ranked_indices = list(probe_report["ranked_indices"])
    report_layer = int(probe_report["default_layer"])
    if report_layer != layer:
        LOGGER.warning("Probe report layer %s differs from official-style steering layer %s", report_layer, layer)

    eval_rows = _sample_rows(unacceptable_rows, eval_sample_n, seed)
    eval_prompts = [_official_prompt(bundle.model_name, row["prompt"]) for row in eval_rows]
    eval_questions = [row["prompt"] for row in eval_rows]

    dataset_summary: dict[str, Any] = {
        "n_acceptable_pool": int(len(acceptable_rows)),
        "n_unacceptable_pool": int(len(unacceptable_rows)),
        "n_build_acceptable": int(len(acceptable_build_rows)),
        "n_build_unacceptable": int(len(unacceptable_build_rows)),
        "n_eval": int(len(eval_rows)),
        "layer": layer,
        "report_layer": report_layer,
        "aen_count": int(len(aen_indices)),
        "base_abstention_rate": float(len(acceptable_rows) / (len(acceptable_rows) + len(unacceptable_rows))),
    }

    for strategy in strategy_names:
        masked_direction = _mask_direction(direction, strategy, aen_indices=aen_indices, ranked_indices=ranked_indices)
        responses = _generate_with_last_token_steering(
            bundle=bundle,
            prompt_texts=eval_prompts,
            generation_cfg=generation_cfg,
            layer=layer,
            steering_vector=masked_direction,
            alpha=float(steering_cfg.get("alpha", 1.0)),
        )
        raw_df = pd.DataFrame(
            {
                "dataset": dataset_files.name,
                "strategy": strategy,
                "question": eval_questions,
                "prompt_text": eval_prompts,
                "response_text": responses,
            }
        )
        write_parquet(raw_df, output_root / f"{dataset_files.name}__{strategy}__raw.parquet")

        if judge_bundle is not None:
            judge_raw, judge_labels = _judge_outputs_locally(
                bundle=judge_bundle,
                config=config,
                questions=eval_questions,
                responses=responses,
            )
            judged_df = raw_df.copy()
            judged_df["judge_raw"] = judge_raw
            judged_df["judge_label"] = judge_labels
            write_parquet(judged_df, output_root / f"{dataset_files.name}__{strategy}.parquet")
            dataset_summary[strategy] = {
                "abstention_rate": float(np.mean([label == "ACCEPTABLE" for label in judge_labels])) if judge_labels else float("nan"),
                "n_eval": int(len(judge_labels)),
            }
        else:
            dataset_summary[strategy] = {
                "abstention_rate": float("nan"),
                "n_eval": int(len(raw_df)),
            }
    return dataset_summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_official_style_steering.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    output_root = ensure_dir(Path(config["official_style_steering"]["output_dir"]) / slugify(config["model"]["name"]))
    bundle: HFModelBundle | None = None
    judge_bundle: HFModelBundle | None = None
    try:
        bundle = load_hf_model(config["model"], config["generation"])
        if config.get("judge", {}).get("provider") == "local_llm":
            judge_bundle = _load_judge_bundle(config)
        summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}
        for dataset_files in _dataset_files(config):
            LOGGER.info("Running official-style steering for %s", dataset_files.name)
            summary["datasets"][dataset_files.name] = _run_dataset(
                bundle=bundle,
                judge_bundle=judge_bundle,
                config=config,
                dataset_files=dataset_files,
                output_root=output_root,
            )
        write_json(output_root / "summary.json", summary)
        write_json(output_root / "resolved_config.json", config)
    finally:
        _release_model(judge_bundle)
        _release_model(bundle)


if __name__ == "__main__":
    main()
