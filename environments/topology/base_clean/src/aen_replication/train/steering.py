"""Activation-steering experiments for ambiguity behavior."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from aen_replication.eval.judge import JudgeResult, LocalLLMJudge, RuleBasedAbstentionJudge
from aen_replication.models.generation import _build_generate_kwargs, generate_batch, render_prompts
from aen_replication.models.hidden_state_extractor import HiddenStateExtractor
from aen_replication.models.hf_model import HFModelBundle
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet

LOGGER = logging.getLogger(__name__)

OFFICIAL_MODEL_ALIASES = {
    "meta-llama/Llama-3.1-8B-Instruct": "lama3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "lama3",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "google/gemma-7b-it": "gemma",
}

OFFICIAL_DATASET_DIRS = {
    "ambigqa": "ambig_qa",
    "situatedqa": "situated",
}

OFFICIAL_CLEAR_FILES = {
    "ambigqa": "clean_questions.json",
    "situatedqa": "clean_combined_question.json",
}


@dataclass(slots=True)
class SteeringDirection:
    """A contrastive steering direction and associated neuron masks."""

    vector: np.ndarray
    aen_indices: list[int]
    ranked_indices: list[int]
    layer: int


JudgeBackend = RuleBasedAbstentionJudge | LocalLLMJudge


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported model architecture for steering: {type(model)!r}")


def _prompt_texts(
    bundle: HFModelBundle,
    texts: list[str],
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
) -> list[str]:
    prompts = [f"{text}{prompt_suffix}" if prompt_suffix else text for text in texts]
    return render_prompts(
        bundle=bundle,
        prompt_texts=prompts,
        use_chat_template=bool(generation_cfg.get("use_chat_template", True)),
        system_prompt=generation_cfg.get("system_prompt"),
    )


def _extract_prompt_vectors(
    bundle: HFModelBundle,
    prompts: list[str],
    extraction_cfg: dict[str, Any],
    layer: int,
) -> np.ndarray:
    extractor = HiddenStateExtractor(
        bundle=bundle,
        batch_size=int(extraction_cfg["batch_size"]),
        max_length=int(extraction_cfg["max_length"]),
        use_mixed_precision=bool(extraction_cfg.get("use_mixed_precision", False)),
    )
    df = pd.DataFrame({"prompt_text": prompts})
    vectors = extractor.extract(
        df=df,
        layers=[layer],
        readouts=["mean_pool"],
        text_column="prompt_text",
    )
    return vectors[(layer, "mean_pool")]


def _paired_vectors(
    plus_vectors: np.ndarray,
    minus_vectors: np.ndarray,
    seed: int,
    pairing: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_pairs = min(len(plus_vectors), len(minus_vectors))
    if n_pairs <= 0:
        raise ValueError("Need at least one positive and one negative example to build a steering direction.")

    rng = np.random.default_rng(seed)
    plus_indices = rng.choice(len(plus_vectors), size=n_pairs, replace=False)

    if pairing == "random":
        minus_indices = rng.choice(len(minus_vectors), size=n_pairs, replace=False)
        return plus_vectors[plus_indices], minus_vectors[minus_indices]

    if pairing == "nearest":
        selected_plus = plus_vectors[plus_indices]
        plus_norm = selected_plus / np.maximum(np.linalg.norm(selected_plus, axis=1, keepdims=True), 1e-12)
        minus_norm = minus_vectors / np.maximum(np.linalg.norm(minus_vectors, axis=1, keepdims=True), 1e-12)
        similarities = plus_norm @ minus_norm.T
        chosen_minus: list[int] = []
        used_minus: set[int] = set()
        for row in np.argsort(-similarities, axis=1):
            for candidate in row:
                candidate_int = int(candidate)
                if candidate_int not in used_minus:
                    used_minus.add(candidate_int)
                    chosen_minus.append(candidate_int)
                    break
        return selected_plus[: len(chosen_minus)], minus_vectors[chosen_minus]

    raise ValueError(f"Unknown direction pairing method: {pairing}")


def _build_direction(
    plus_vectors: np.ndarray,
    minus_vectors: np.ndarray,
    seed: int,
    method: str = "pca_global_center",
    pairing: str = "random",
) -> np.ndarray:
    if method in {"pca_global", "pca_global_center"}:
        mean_center = 0.5 * (plus_vectors.mean(axis=0) + minus_vectors.mean(axis=0))
        train = np.vstack([plus_vectors - mean_center, minus_vectors - mean_center])
        sign_plus = plus_vectors
        sign_minus = minus_vectors
    elif method == "pca_center":
        sign_plus, sign_minus = _paired_vectors(plus_vectors, minus_vectors, seed=seed, pairing=pairing)
        h = np.empty((len(sign_plus) * 2, sign_plus.shape[1]), dtype=np.float32)
        h[::2] = sign_plus
        h[1::2] = sign_minus
        train = h - h.mean(axis=0)
    elif method == "pca_diff":
        sign_plus, sign_minus = _paired_vectors(plus_vectors, minus_vectors, seed=seed, pairing=pairing)
        train = sign_plus - sign_minus
    elif method == "pca_pairwise":
        sign_plus, sign_minus = _paired_vectors(plus_vectors, minus_vectors, seed=seed, pairing=pairing)
        h = np.empty((len(sign_plus) * 2, sign_plus.shape[1]), dtype=np.float32)
        h[::2] = sign_plus
        h[1::2] = sign_minus
        center = (h[::2] + h[1::2]) / 2
        train = h.copy()
        train[::2] -= center
        train[1::2] -= center
    elif method == "mean_diff":
        direction = (plus_vectors.mean(axis=0) - minus_vectors.mean(axis=0)).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("Mean-difference steering direction has zero norm.")
        return direction / norm
    elif method in {"mean_diff_raw", "raw_mean_diff"}:
        direction = (plus_vectors.mean(axis=0) - minus_vectors.mean(axis=0)).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("Raw mean-difference steering direction has zero norm.")
        return direction
    else:
        raise ValueError(f"Unknown steering direction method: {method}")

    pca = PCA(n_components=1, random_state=seed)
    pca.fit(train)
    direction = pca.components_[0].astype(np.float32)
    if len(sign_plus) == len(sign_minus):
        plus_projection = sign_plus @ direction
        minus_projection = sign_minus @ direction
        should_flip = float(np.mean(plus_projection > minus_projection)) < float(np.mean(plus_projection < minus_projection))
    else:
        should_flip = float((sign_plus.mean(axis=0) - sign_minus.mean(axis=0)) @ direction) < 0
    if should_flip:
        direction = -direction
    return direction


def _masked_vector(direction: SteeringDirection, strategy: str) -> np.ndarray:
    vector = np.zeros_like(direction.vector)
    if strategy == "full_vector":
        return direction.vector.copy()
    if strategy == "aens":
        indices = direction.aen_indices
    elif strategy == "top_50":
        indices = direction.ranked_indices[:50]
    elif strategy == "top_100":
        indices = direction.ranked_indices[:100]
    else:
        raise ValueError(f"Unknown steering strategy: {strategy}")
    vector[indices] = direction.vector[indices]
    return vector


def _generate_with_steering(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    generation_cfg: dict[str, Any],
    layer: int,
    steering_vector: np.ndarray,
    alpha: float,
    intervention_site: str = "layer_output",
    apply_on: str = "prompt_only",
    preserve_token_norm: bool = False,
) -> list[str]:
    decoder_layers = _decoder_layers(bundle.model)
    target_layer = decoder_layers[layer]
    direction = torch.as_tensor(steering_vector, device=bundle.device, dtype=next(bundle.model.parameters()).dtype)
    direction = (alpha * direction).view(1, 1, -1)

    use_chat_template = bool(generation_cfg.get("use_chat_template", True))
    system_prompt = generation_cfg.get("system_prompt")
    rendered_prompts = render_prompts(
        bundle=bundle,
        prompt_texts=prompt_texts,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        add_generation_prompt=True,
    )
    original_padding_side = bundle.tokenizer.padding_side
    bundle.tokenizer.padding_side = "left"
    encoded = bundle.tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=generation_cfg.get("prompt_max_length"),
    )
    bundle.tokenizer.padding_side = original_padding_side
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    prompt_delta = encoded["attention_mask"].unsqueeze(-1).to(bundle.device) * direction
    prefill_applied = False

    def apply_delta(hidden_states: torch.Tensor) -> torch.Tensor:
        nonlocal prefill_applied
        if hidden_states.shape[:2] == prompt_delta.shape[:2] and not prefill_applied:
            delta = prompt_delta.to(dtype=hidden_states.dtype)
            prefill_applied = True
        elif apply_on in {"prompt_and_decode", "all"} and prefill_applied:
            delta = direction.to(dtype=hidden_states.dtype).expand(hidden_states.shape[0], hidden_states.shape[1], -1)
        else:
            return hidden_states

        steered = hidden_states + delta
        if not preserve_token_norm:
            return steered
        original_norm = hidden_states.norm(dim=-1, keepdim=True)
        steered_norm = steered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return steered * (original_norm / steered_norm)

    def output_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden_states = apply_delta(output[0].clone())
            return (hidden_states,) + output[1:]
        return apply_delta(output.clone())

    def input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            return inputs
        hidden_states = inputs[0]
        if not torch.is_tensor(hidden_states):
            return inputs
        return (apply_delta(hidden_states.clone()),) + inputs[1:]

    if intervention_site == "layer_output":
        handle = target_layer.register_forward_hook(output_hook)
    elif intervention_site == "layer_input":
        handle = target_layer.register_forward_pre_hook(input_hook)
    else:
        raise ValueError(f"Unknown steering intervention site: {intervention_site}")
    try:
        generate_kwargs = _build_generate_kwargs(generation_cfg, return_entropy=False)
        generate_kwargs["pad_token_id"] = bundle.tokenizer.pad_token_id
        generate_kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id
        with torch.no_grad():
            generation_output = bundle.model.generate(**encoded, **generate_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        responses: list[str] = []
        for index in range(len(rendered_prompts)):
            generated_ids = generation_output.sequences[index, prompt_length:]
            response_text = bundle.tokenizer.decode(generated_ids.detach().cpu(), skip_special_tokens=True).strip()
            responses.append(response_text)
        return responses
    finally:
        handle.remove()


def _iter_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def _behavior_table_raw(
    bundle: HFModelBundle,
    df: pd.DataFrame,
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    batch_size = int(generation_cfg.get("batch_size", 8))
    for batch_df in tqdm(_iter_batches(df, batch_size), total=(len(df) + batch_size - 1) // batch_size, desc="base_behavior", leave=False):
        prompts = [
            f"{text}{prompt_suffix}" if prompt_suffix else text
            for text in batch_df["text"].tolist()
        ]
        generations = generate_batch(bundle=bundle, prompt_texts=prompts, generation_config=generation_cfg)
        for row, generation in zip(batch_df.itertuples(index=False), generations, strict=True):
            rows.append(
                {
                    "example_id": row.example_id,
                    "pair_id": row.pair_id,
                    "dataset": row.dataset,
                    "split": row.split,
                    "label_ambiguous": int(row.label_ambiguous),
                    "text": row.text,
                    "prompt_text": generation.prompt_text,
                    "response_text": generation.response_text,
                }
            )
    return pd.DataFrame(rows)


def _strategy_raw_path(output_root: Path, dataset_name: str, strategy: str) -> Path:
    return output_root / f"{dataset_name}__{strategy}__raw.parquet"


def _strategy_judged_path(output_root: Path, dataset_name: str, strategy: str) -> Path:
    return output_root / f"{dataset_name}__{strategy}.parquet"


def _direction_path(output_root: Path, dataset_name: str, config: dict[str, Any]) -> Path:
    steering_cfg = config["steering"]
    if not any(
        key in steering_cfg
        for key in ("direction_method", "direction_negative_source", "direction_pairing")
    ):
        return output_root / f"{dataset_name}__direction.npy"
    method = steering_cfg.get("direction_method", "pca_global_center")
    source = steering_cfg.get("direction_negative_source", "clear")
    pairing = steering_cfg.get("direction_pairing", "random")
    return output_root / f"{dataset_name}__direction__{slugify(f'{method}_{source}_{pairing}')}.npy"


def _judge_table(judge: JudgeBackend, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    results: list[JudgeResult] = []
    for batch_df in tqdm(_iter_batches(df, batch_size), total=(len(df) + batch_size - 1) // batch_size, desc="judge", leave=False):
        if hasattr(judge, "judge_many"):
            batch_results = judge.judge_many(batch_df["text"].tolist(), batch_df["response_text"].tolist(), batch_size=batch_size)
        else:
            batch_results = [judge.judge(row.text, row.response_text) for row in batch_df.itertuples(index=False)]
        results.extend(batch_results)
    judged_df = df.copy()
    judged_df["judge_label"] = [result.label for result in results]
    judged_df["judge_explanation"] = [result.explanation for result in results]
    judged_df["judge_raw_response"] = [result.raw_response for result in results]
    return judged_df


def _official_annotation_source(config: dict[str, Any]) -> bool:
    return config["steering"].get("base_behavior_source") == "official_annotations"


def _load_prompt_entries(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            prompts.append(entry)
        elif isinstance(entry, dict) and "prompt" in entry:
            prompts.append(str(entry["prompt"]))
        else:
            raise ValueError(f"Unsupported prompt entry in {path}: {entry!r}")
    return prompts


def _official_dataset_root(config: dict[str, Any], dataset_name: str) -> Path:
    steering_cfg = config["steering"]
    repo_root = Path(steering_cfg.get("official_repo_root", "/home/ubuntu/Internal_State_Detect_Ambiguity"))
    dataset_dir = OFFICIAL_DATASET_DIRS[dataset_name]
    return repo_root / dataset_dir


def _official_model_alias(config: dict[str, Any]) -> str:
    model_name = config["model"]["name"]
    try:
        return OFFICIAL_MODEL_ALIASES[model_name]
    except KeyError as exc:
        raise KeyError(f"No official annotation alias configured for model {model_name!r}") from exc


def _official_behavior_files(config: dict[str, Any], dataset_name: str) -> dict[str, Path]:
    steering_cfg = config["steering"]
    mode = steering_cfg.get("official_annotation_mode", "abstention")
    model_alias = _official_model_alias(config)
    root = _official_dataset_root(config, dataset_name)
    files = {
        "acceptable": root / f"acceptable_responses_{mode}_{model_alias}.json",
        "unacceptable": root / f"unacceptable_responses_{mode}_{model_alias}.json",
        "neither": root / f"neither_responses_{mode}_{model_alias}.json",
    }
    missing = [label for label, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official {mode} annotation files are missing for dataset={dataset_name}, model={model_alias}: {missing}"
        )
    return files


def _load_official_behavior_df(config: dict[str, Any], dataset_name: str) -> pd.DataFrame:
    files = _official_behavior_files(config, dataset_name)
    rows: list[dict[str, Any]] = []
    for label, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        judge_label = label.upper() if label != "acceptable" else "ACCEPTABLE"
        if label == "unacceptable":
            judge_label = "UNACCEPTABLE"
        elif label == "neither":
            judge_label = "NEITHER"
        for record in data:
            rows.append(
                {
                    "example_id": None,
                    "pair_id": None,
                    "dataset": dataset_name,
                    "split": "official",
                    "label_ambiguous": 1,
                    "text": str(record["prompt"]),
                    "prompt_text": str(record["prompt"]),
                    "response_text": str(record["response"]),
                    "judge_label": judge_label,
                    "judge_explanation": str(record.get("llm_evaluation", "")),
                    "judge_raw_response": str(record.get("llm_evaluation", "")),
                }
            )
    return pd.DataFrame(rows)


def _load_official_clear_df(config: dict[str, Any], dataset_name: str) -> pd.DataFrame:
    root = _official_dataset_root(config, dataset_name)
    clear_path = root / OFFICIAL_CLEAR_FILES[dataset_name]
    prompts = _load_prompt_entries(clear_path)
    return pd.DataFrame({"text": prompts})


def _direction_from_probe_report(config: dict[str, Any], dataset_name: str) -> SteeringDirection:
    model_slug = slugify(config["model"]["name"])
    probe_report_path = Path(config["probe"]["artifact_dir"]) / model_slug / f"{dataset_name}_default_layer_report.json"
    payload = json.loads(probe_report_path.read_text(encoding="utf-8"))
    return SteeringDirection(
        vector=np.empty(0, dtype=np.float32),
        aen_indices=list(payload["aen_selection"]["aen_indices"]),
        ranked_indices=list(payload["ranked_indices"]),
        layer=int(payload["default_layer"]),
    )


def _dataset_direction(
    bundle: HFModelBundle,
    config: dict[str, Any],
    dataset_name: str,
    clear_df: pd.DataFrame,
    abstain_df: pd.DataFrame,
    direct_answer_df: pd.DataFrame | None = None,
) -> SteeringDirection:
    steering_cfg = config["steering"]
    seed = int(config["seed"])
    descriptor = _direction_from_probe_report(config, dataset_name)
    prompt_suffix = steering_cfg.get("prompt_suffix", "")
    negative_source = steering_cfg.get("direction_negative_source", "clear")
    if negative_source == "clear":
        negative_df = clear_df
        negative_n_key = "build_clear_n"
    elif negative_source in {"direct_answer", "answer", "unacceptable"}:
        if direct_answer_df is None:
            raise ValueError("direction_negative_source=direct_answer requires direct_answer_df.")
        negative_df = direct_answer_df
        negative_n_key = "build_direct_answer_n"
    else:
        raise ValueError(f"Unknown direction negative source: {negative_source}")

    n_plus = min(int(steering_cfg.get("build_abstain_n", 100)), len(abstain_df))
    n_minus = min(int(steering_cfg.get(negative_n_key, steering_cfg.get("build_clear_n", 100))), len(negative_df))
    abstain_sample = abstain_df.sample(n=n_plus, random_state=seed).reset_index(drop=True)
    negative_sample = negative_df.sample(n=n_minus, random_state=seed).reset_index(drop=True)
    plus_prompts = _prompt_texts(bundle, abstain_sample["text"].tolist(), config["generation"], prompt_suffix)
    minus_prompts = _prompt_texts(bundle, negative_sample["text"].tolist(), config["generation"], prompt_suffix)
    plus_vectors = _extract_prompt_vectors(bundle, plus_prompts, config["extraction"], descriptor.layer)
    minus_vectors = _extract_prompt_vectors(bundle, minus_prompts, config["extraction"], descriptor.layer)
    descriptor.vector = _build_direction(
        plus_vectors=plus_vectors,
        minus_vectors=minus_vectors,
        seed=seed,
        method=steering_cfg.get("direction_method", "pca_global_center"),
        pairing=steering_cfg.get("direction_pairing", "random"),
    )
    return descriptor


def _evaluate_strategy(
    bundle: HFModelBundle,
    df: pd.DataFrame,
    generation_cfg: dict[str, Any],
    steering_cfg: dict[str, Any],
    prompt_suffix: str,
    direction: SteeringDirection,
    strategy: str,
    alpha: float,
    reverse: bool = False,
) -> pd.DataFrame:
    steer_vector = _masked_vector(direction, strategy)
    if reverse:
        steer_vector = -steer_vector
    rows: list[dict[str, Any]] = []
    batch_size = int(generation_cfg.get("batch_size", 8))
    for batch_df in tqdm(_iter_batches(df, batch_size), total=(len(df) + batch_size - 1) // batch_size, desc=f"{strategy}_{'reverse' if reverse else 'forward'}", leave=False):
        prompts = [
            f"{text}{prompt_suffix}" if prompt_suffix else text
            for text in batch_df["text"].tolist()
        ]
        responses = _generate_with_steering(
            bundle=bundle,
            prompt_texts=prompts,
            generation_cfg=generation_cfg,
            layer=direction.layer,
            steering_vector=steer_vector,
            alpha=alpha,
            intervention_site=steering_cfg.get("intervention_site", "layer_output"),
            apply_on=steering_cfg.get("steering_apply_on", "prompt_only"),
            preserve_token_norm=bool(steering_cfg.get("preserve_token_norm", False)),
        )
        for row, prompt_text, response_text in zip(batch_df.itertuples(index=False), prompts, responses, strict=True):
            rows.append(
                {
                    "example_id": getattr(row, "example_id", None),
                    "pair_id": getattr(row, "pair_id", None),
                    "dataset": getattr(row, "dataset", None),
                    "text": row.text,
                    "prompt_text": prompt_text,
                    "strategy": strategy,
                    "reverse": reverse,
                    "response_text": response_text,
                }
            )
    return pd.DataFrame(rows)


def _release_model(*objects: Any) -> None:
    for obj in objects:
        if hasattr(obj, "model"):
            delattr(obj, "model")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_base_behavior_outputs(
    bundle: HFModelBundle | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate base responses on ambiguous test prompts."""

    steering_cfg = config["steering"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(steering_cfg["artifact_dir"]) / model_slug)
    summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}

    if _official_annotation_source(config):
        for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
            raw_path = output_root / f"{dataset_name}__base_behavior_raw.parquet"
            judged_path = output_root / f"{dataset_name}__base_behavior.parquet"
            if raw_path.exists() and judged_path.exists():
                behavior_df = pd.read_parquet(judged_path)
                summary["datasets"][dataset_name] = {
                    "n_ambiguous_test": int(len(behavior_df)),
                    "base_behavior_source": "official_annotations",
                }
                continue
            behavior_df = _load_official_behavior_df(config, dataset_name)
            raw_df = behavior_df.drop(columns=["judge_label", "judge_explanation", "judge_raw_response"])
            write_parquet(raw_df, raw_path)
            write_parquet(behavior_df, judged_path)
            summary["datasets"][dataset_name] = {
                "n_ambiguous_test": int(len(behavior_df)),
                "base_behavior_source": "official_annotations",
            }
        write_json(output_root / "base_behavior_generation_summary.json", summary)
        return summary

    for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
        raw_path = output_root / f"{dataset_name}__base_behavior_raw.parquet"
        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            summary["datasets"][dataset_name] = {"n_ambiguous_test": int(len(raw_df))}
            continue
        pairs_path = Path(config["data"]["pair_output_dir"]) / f"{dataset_name}_pairs.parquet"
        dataset_df = pd.read_parquet(pairs_path)
        ambiguous_test = dataset_df.loc[
            (dataset_df["split"] == "test") & (dataset_df["label_ambiguous"] == 1)
        ].reset_index(drop=True)
        behavior_df = _behavior_table_raw(
            bundle=bundle,
            df=ambiguous_test,
            generation_cfg=config["generation"],
            prompt_suffix=steering_cfg.get("prompt_suffix", ""),
        )
        write_parquet(behavior_df, output_root / f"{dataset_name}__base_behavior_raw.parquet")
        summary["datasets"][dataset_name] = {"n_ambiguous_test": int(len(ambiguous_test))}
    write_json(output_root / "base_behavior_generation_summary.json", summary)
    return summary


def judge_base_behavior_outputs(
    config: dict[str, Any],
    judge: JudgeBackend,
) -> dict[str, Any]:
    """Judge base-model responses on ambiguous test prompts."""

    steering_cfg = config["steering"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(steering_cfg["artifact_dir"]) / model_slug)
    summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}

    if _official_annotation_source(config):
        for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
            judged_path = output_root / f"{dataset_name}__base_behavior.parquet"
            behavior_df = pd.read_parquet(judged_path)
            summary["datasets"][dataset_name] = {
                "abstention_rate": float(behavior_df["judge_label"].eq("ACCEPTABLE").mean()),
                "n_abstain": int(behavior_df["judge_label"].eq("ACCEPTABLE").sum()),
                "n_answer": int(behavior_df["judge_label"].eq("UNACCEPTABLE").sum()),
                "n_neither": int(behavior_df["judge_label"].eq("NEITHER").sum()),
                "base_behavior_source": "official_annotations",
            }
        write_json(output_root / "base_behavior_judged_summary.json", summary)
        return summary

    for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
        raw_path = output_root / f"{dataset_name}__base_behavior_raw.parquet"
        judged_path = output_root / f"{dataset_name}__base_behavior.parquet"
        if judged_path.exists():
            behavior_df = pd.read_parquet(judged_path)
        else:
            behavior_df = pd.read_parquet(raw_path)
            behavior_df = _judge_table(judge, behavior_df, batch_size=int(config["judge"].get("batch_size", 8)))
            write_parquet(behavior_df, judged_path)
        summary["datasets"][dataset_name] = {
            "abstention_rate": float(behavior_df["judge_label"].eq("ACCEPTABLE").mean()),
            "n_abstain": int(behavior_df["judge_label"].eq("ACCEPTABLE").sum()),
            "n_answer": int(behavior_df["judge_label"].eq("UNACCEPTABLE").sum()),
        }
    write_json(output_root / "base_behavior_judged_summary.json", summary)
    return summary


def generate_steering_outputs(
    bundle: HFModelBundle,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate steered responses after judging base behavior."""

    steering_cfg = config["steering"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(steering_cfg["artifact_dir"]) / model_slug)
    summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}

    dataset_tables: dict[str, pd.DataFrame] = {}
    behavior_tables: dict[str, pd.DataFrame] = {}
    directions: dict[str, SteeringDirection] = {}

    if _official_annotation_source(config):
        for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
            clear_df = _load_official_clear_df(config, dataset_name)
            behavior_df = pd.read_parquet(output_root / f"{dataset_name}__base_behavior.parquet")
            behavior_tables[dataset_name] = behavior_df

            abstain_joined = behavior_df.loc[behavior_df["judge_label"] == "ACCEPTABLE", ["text"]].drop_duplicates().reset_index(drop=True)
            answer_joined = behavior_df.loc[behavior_df["judge_label"] == "UNACCEPTABLE", ["text"]].drop_duplicates().reset_index(drop=True)
            directions[dataset_name] = _dataset_direction(
                bundle=bundle,
                config=config,
                dataset_name=dataset_name,
                clear_df=clear_df,
                abstain_df=abstain_joined,
                direct_answer_df=answer_joined,
            )
            direction_path = _direction_path(output_root, dataset_name, config)
            if direction_path.exists():
                directions[dataset_name].vector = np.load(direction_path)
            else:
                np.save(direction_path, directions[dataset_name].vector)

            eval_n = min(int(steering_cfg.get("eval_direct_answer_n", 500)), len(answer_joined))
            eval_df = answer_joined.sample(n=eval_n, random_state=int(config["seed"])).reset_index(drop=True)
            strategies_summary: dict[str, Any] = {
                "n_abstain_pool": int(len(abstain_joined)),
                "n_answer_pool": int(len(answer_joined)),
                "n_eval": int(eval_n),
                "base_behavior_source": "official_annotations",
            }
            for strategy in steering_cfg.get("selection_strategies", ["aens", "top_50", "top_100", "full_vector"]):
                steered_df = _evaluate_strategy(
                    bundle=bundle,
                    df=eval_df,
                    generation_cfg=config["generation"],
                    steering_cfg=steering_cfg,
                    prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                    direction=directions[dataset_name],
                    strategy=strategy,
                    alpha=float(steering_cfg.get("alpha", 1.0)),
                ) if not _strategy_raw_path(output_root, dataset_name, strategy).exists() else pd.read_parquet(_strategy_raw_path(output_root, dataset_name, strategy))
                if not _strategy_raw_path(output_root, dataset_name, strategy).exists():
                    write_parquet(steered_df, _strategy_raw_path(output_root, dataset_name, strategy))

            consistency_pool = abstain_joined.sample(
                n=min(int(steering_cfg.get("eval_consistency_n", 500)), len(abstain_joined)),
                random_state=int(config["seed"]),
            ).reset_index(drop=True)
            consistency_raw_path = output_root / f"{dataset_name}__aens_consistency__raw.parquet"
            if not consistency_raw_path.exists():
                consistency_df = _evaluate_strategy(
                    bundle=bundle,
                    df=consistency_pool,
                    generation_cfg=config["generation"],
                    steering_cfg=steering_cfg,
                    prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                    direction=directions[dataset_name],
                    strategy="aens",
                    alpha=float(steering_cfg.get("alpha", 1.0)),
                    reverse=False,
                )
                write_parquet(consistency_df, consistency_raw_path)

            reverse_pool = abstain_joined.sample(
                n=min(int(steering_cfg.get("eval_abstain_n", 500)), len(abstain_joined)),
                random_state=int(config["seed"]),
            ).reset_index(drop=True)
            reverse_raw_path = output_root / f"{dataset_name}__aens_reverse__raw.parquet"
            if not reverse_raw_path.exists():
                reverse_df = _evaluate_strategy(
                    bundle=bundle,
                    df=reverse_pool,
                    generation_cfg=config["generation"],
                    steering_cfg=steering_cfg,
                    prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                    direction=directions[dataset_name],
                    strategy="aens",
                    alpha=float(steering_cfg.get("alpha", 1.0)),
                    reverse=True,
                )
                write_parquet(reverse_df, reverse_raw_path)
            summary["datasets"][dataset_name] = strategies_summary

        for train_dataset, test_dataset in (("ambigqa", "situatedqa"), ("situatedqa", "ambigqa")):
            if train_dataset not in directions or test_dataset not in behavior_tables:
                continue
            answer_df = behavior_tables[test_dataset].loc[
                behavior_tables[test_dataset]["judge_label"] == "UNACCEPTABLE", ["text"]
            ].drop_duplicates().reset_index(drop=True)
            if answer_df.empty:
                summary["datasets"][f"{train_dataset}_to_{test_dataset}"] = {"abstention_rate": float("nan")}
                continue
            eval_df = answer_df.sample(
                n=min(int(steering_cfg.get("eval_direct_answer_n", 500)), len(answer_df)),
                random_state=int(config["seed"]),
            ).reset_index(drop=True)
            transfer_df = _evaluate_strategy(
                bundle=bundle,
                df=eval_df,
                generation_cfg=config["generation"],
                steering_cfg=steering_cfg,
                prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                direction=directions[train_dataset],
                strategy="aens",
                alpha=float(steering_cfg.get("alpha", 1.0)),
            )
            write_parquet(transfer_df, output_root / f"{train_dataset}__to__{test_dataset}__aens__raw.parquet")
            summary["datasets"][f"{train_dataset}_to_{test_dataset}"] = {"n_eval": int(len(eval_df))}

        write_json(output_root / "steering_generation_summary.json", summary)
        return summary

    for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
        pairs_path = Path(config["data"]["pair_output_dir"]) / f"{dataset_name}_pairs.parquet"
        dataset_df = pd.read_parquet(pairs_path)
        dataset_tables[dataset_name] = dataset_df
        behavior_df = pd.read_parquet(output_root / f"{dataset_name}__base_behavior.parquet")
        behavior_tables[dataset_name] = behavior_df

        clear_test = dataset_df.loc[
            (dataset_df["split"] == "test") & (dataset_df["label_ambiguous"] == 0)
        ].reset_index(drop=True)
        ambiguous_test = dataset_df.loc[
            (dataset_df["split"] == "test") & (dataset_df["label_ambiguous"] == 1)
        ].reset_index(drop=True)
        abstain_df = behavior_df.loc[behavior_df["judge_label"] == "ACCEPTABLE", ["example_id", "text"]]
        answer_df = behavior_df.loc[behavior_df["judge_label"] == "UNACCEPTABLE", ["example_id", "text"]]
        abstain_joined = ambiguous_test.merge(abstain_df, on=["example_id", "text"], how="inner")
        answer_joined = ambiguous_test.merge(answer_df, on=["example_id", "text"], how="inner")
        directions[dataset_name] = _dataset_direction(
            bundle=bundle,
            config=config,
            dataset_name=dataset_name,
            clear_df=clear_test,
            abstain_df=abstain_joined,
            direct_answer_df=answer_joined,
        )
        direction_path = _direction_path(output_root, dataset_name, config)
        if direction_path.exists():
            directions[dataset_name].vector = np.load(direction_path)
        else:
            np.save(direction_path, directions[dataset_name].vector)

        eval_n = min(int(steering_cfg.get("eval_direct_answer_n", 500)), len(answer_joined))
        eval_df = answer_joined.sample(n=eval_n, random_state=int(config["seed"])).reset_index(drop=True)
        strategies_summary: dict[str, Any] = {
            "n_abstain_pool": int(len(abstain_joined)),
            "n_answer_pool": int(len(answer_joined)),
            "n_eval": int(eval_n),
        }
        for strategy in steering_cfg.get("selection_strategies", ["aens", "top_50", "top_100", "full_vector"]):
            raw_path = _strategy_raw_path(output_root, dataset_name, strategy)
            if not raw_path.exists():
                steered_df = _evaluate_strategy(
                    bundle=bundle,
                    df=eval_df,
                    generation_cfg=config["generation"],
                    steering_cfg=steering_cfg,
                    prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                    direction=directions[dataset_name],
                    strategy=strategy,
                    alpha=float(steering_cfg.get("alpha", 1.0)),
                )
                write_parquet(steered_df, raw_path)

        consistency_pool = abstain_joined.sample(
            n=min(int(steering_cfg.get("eval_consistency_n", 500)), len(abstain_joined)),
            random_state=int(config["seed"]),
        ).reset_index(drop=True)
        consistency_raw_path = output_root / f"{dataset_name}__aens_consistency__raw.parquet"
        if not consistency_raw_path.exists():
            consistency_df = _evaluate_strategy(
                bundle=bundle,
                df=consistency_pool,
                generation_cfg=config["generation"],
                steering_cfg=steering_cfg,
                prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                direction=directions[dataset_name],
                strategy="aens",
                alpha=float(steering_cfg.get("alpha", 1.0)),
                reverse=False,
            )
            write_parquet(consistency_df, consistency_raw_path)

        reverse_pool = abstain_joined.sample(
            n=min(int(steering_cfg.get("eval_abstain_n", 500)), len(abstain_joined)),
            random_state=int(config["seed"]),
        ).reset_index(drop=True)
        reverse_raw_path = output_root / f"{dataset_name}__aens_reverse__raw.parquet"
        if not reverse_raw_path.exists():
            reverse_df = _evaluate_strategy(
                bundle=bundle,
                df=reverse_pool,
                generation_cfg=config["generation"],
                steering_cfg=steering_cfg,
                prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                direction=directions[dataset_name],
                strategy="aens",
                alpha=float(steering_cfg.get("alpha", 1.0)),
                reverse=True,
            )
            write_parquet(reverse_df, reverse_raw_path)
        summary["datasets"][dataset_name] = strategies_summary

    available_datasets = set(steering_cfg.get("datasets", ["ambigqa", "situatedqa"]))
    for train_dataset, test_dataset in (("ambigqa", "situatedqa"), ("situatedqa", "ambigqa")):
        if train_dataset not in available_datasets or test_dataset not in available_datasets:
            continue
        answer_df = behavior_tables[test_dataset].loc[behavior_tables[test_dataset]["judge_label"] == "UNACCEPTABLE", ["example_id", "text"]]
        test_ambiguous = dataset_tables[test_dataset].loc[
            (dataset_tables[test_dataset]["split"] == "test") & (dataset_tables[test_dataset]["label_ambiguous"] == 1)
        ].merge(answer_df, on=["example_id", "text"], how="inner")
        if test_ambiguous.empty:
            summary["datasets"][f"{train_dataset}_to_{test_dataset}"] = {"abstention_rate": float("nan")}
            continue
        eval_df = test_ambiguous.sample(
            n=min(int(steering_cfg.get("eval_direct_answer_n", 500)), len(test_ambiguous)),
            random_state=int(config["seed"]),
        ).reset_index(drop=True)
        transfer_raw_path = output_root / f"{train_dataset}__to__{test_dataset}__aens__raw.parquet"
        if not transfer_raw_path.exists():
            transfer_df = _evaluate_strategy(
                bundle=bundle,
                df=eval_df,
                generation_cfg=config["generation"],
                steering_cfg=steering_cfg,
                prompt_suffix=steering_cfg.get("prompt_suffix", ""),
                direction=directions[train_dataset],
                strategy="aens",
                alpha=float(steering_cfg.get("alpha", 1.0)),
            )
            write_parquet(transfer_df, transfer_raw_path)
        summary["datasets"][f"{train_dataset}_to_{test_dataset}"] = {"n_eval": int(len(eval_df))}

    write_json(output_root / "steering_generation_summary.json", summary)
    return summary


def judge_steering_outputs_and_summarize(
    config: dict[str, Any],
    judge: JudgeBackend,
) -> dict[str, Any]:
    """Judge steered outputs and write the final steering summary."""

    steering_cfg = config["steering"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(steering_cfg["artifact_dir"]) / model_slug)
    summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}
    judge_batch_size = int(config["judge"].get("batch_size", 8))

    for dataset_name in steering_cfg.get("datasets", ["ambigqa", "situatedqa"]):
        base_df = pd.read_parquet(output_root / f"{dataset_name}__base_behavior.parquet")
        dataset_summary: dict[str, Any] = {
            "base_abstention_rate": float(base_df["judge_label"].eq("ACCEPTABLE").mean()),
            "n_abstain_pool": int(base_df["judge_label"].eq("ACCEPTABLE").sum()),
            "n_answer_pool": int(base_df["judge_label"].eq("UNACCEPTABLE").sum()),
        }
        for strategy in steering_cfg.get("selection_strategies", ["aens", "top_50", "top_100", "full_vector"]):
            judged_path = _strategy_judged_path(output_root, dataset_name, strategy)
            if judged_path.exists():
                judged_df = pd.read_parquet(judged_path)
            else:
                raw_df = pd.read_parquet(_strategy_raw_path(output_root, dataset_name, strategy))
                judged_df = _judge_table(judge, raw_df, batch_size=judge_batch_size)
                write_parquet(judged_df, judged_path)
            dataset_summary[strategy] = {
                "abstention_rate": float(judged_df["judge_label"].eq("ACCEPTABLE").mean()) if not judged_df.empty else float("nan"),
                "n_eval": int(len(judged_df)),
            }

        consistency_judged_path = output_root / f"{dataset_name}__aens_consistency.parquet"
        if consistency_judged_path.exists():
            consistency_df = pd.read_parquet(consistency_judged_path)
        else:
            consistency_raw = pd.read_parquet(output_root / f"{dataset_name}__aens_consistency__raw.parquet")
            consistency_df = _judge_table(judge, consistency_raw, batch_size=judge_batch_size)
            write_parquet(consistency_df, consistency_judged_path)
        dataset_summary["aens_consistency"] = {
            "abstention_consistency": float(consistency_df["judge_label"].eq("ACCEPTABLE").mean()) if not consistency_df.empty else float("nan"),
            "n_eval": int(len(consistency_df)),
        }

        reverse_judged_path = output_root / f"{dataset_name}__aens_reverse.parquet"
        if reverse_judged_path.exists():
            reverse_df = pd.read_parquet(reverse_judged_path)
        else:
            reverse_raw = pd.read_parquet(output_root / f"{dataset_name}__aens_reverse__raw.parquet")
            reverse_df = _judge_table(judge, reverse_raw, batch_size=judge_batch_size)
            write_parquet(reverse_df, reverse_judged_path)
        dataset_summary["aens_reverse"] = {
            "direct_answer_rate": float(reverse_df["judge_label"].eq("UNACCEPTABLE").mean()) if not reverse_df.empty else float("nan"),
            "n_eval": int(len(reverse_df)),
        }
        summary["datasets"][dataset_name] = dataset_summary

    for train_dataset, test_dataset in (("ambigqa", "situatedqa"), ("situatedqa", "ambigqa")):
        raw_path = output_root / f"{train_dataset}__to__{test_dataset}__aens__raw.parquet"
        if not raw_path.exists():
            continue
        judged_path = output_root / f"{train_dataset}__to__{test_dataset}__aens.parquet"
        if judged_path.exists():
            judged_df = pd.read_parquet(judged_path)
        else:
            raw_df = pd.read_parquet(raw_path)
            judged_df = _judge_table(judge, raw_df, batch_size=judge_batch_size)
            write_parquet(judged_df, judged_path)
        summary["datasets"][f"{train_dataset}_to_{test_dataset}"] = {
            "abstention_rate": float(judged_df["judge_label"].eq("ACCEPTABLE").mean()) if not judged_df.empty else float("nan"),
            "n_eval": int(len(judged_df)),
        }

    write_json(output_root / "summary.json", summary)
    return summary
