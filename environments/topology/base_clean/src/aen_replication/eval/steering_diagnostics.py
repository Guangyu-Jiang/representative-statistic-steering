"""Focused diagnostics for the Table 3 steering mismatch."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from aen_replication.eval.judge import JudgeResult, LocalLLMJudge, OpenAIAPIJudge, RuleBasedAbstentionJudge
from aen_replication.models.generation import generate_batch
from aen_replication.models.hf_model import HFModelBundle
from aen_replication.train.steering import (
    OFFICIAL_CLEAR_FILES,
    OFFICIAL_DATASET_DIRS,
    SteeringDirection,
    _build_direction,
    _decoder_layers,
    _direction_from_probe_report,
    _extract_prompt_vectors,
    _judge_table,
    _load_prompt_entries,
    _prompt_texts,
)
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet

LOGGER = logging.getLogger(__name__)

JudgeBackend = RuleBasedAbstentionJudge | LocalLLMJudge | OpenAIAPIJudge


@dataclass(slots=True)
class DiagnosticVariant:
    name: str
    direction_type: str
    alpha: float
    renorm_mode: str
    position_mode: str


def _official_dataset_root(config: dict[str, Any], dataset_name: str) -> Path:
    repo_root = Path(config["steering"].get("official_repo_root", "/home/ubuntu/Internal_State_Detect_Ambiguity"))
    return repo_root / OFFICIAL_DATASET_DIRS[dataset_name]


def _load_clear_df(config: dict[str, Any], dataset_name: str) -> pd.DataFrame:
    root = _official_dataset_root(config, dataset_name)
    clear_path = root / OFFICIAL_CLEAR_FILES[dataset_name]
    prompts = _load_prompt_entries(clear_path)
    return pd.DataFrame({"text": prompts})


def _baseline_artifact_root(config: dict[str, Any]) -> Path:
    steering_cfg = config["steering_diagnostics"]
    model_slug = slugify(config["model"]["name"])
    return Path(steering_cfg["baseline_artifact_dir"]) / model_slug


def _output_root(config: dict[str, Any]) -> Path:
    steering_cfg = config["steering_diagnostics"]
    model_slug = slugify(config["model"]["name"])
    return ensure_dir(Path(steering_cfg["artifact_dir"]) / model_slug)


def _build_mean_difference(plus_vectors: np.ndarray, minus_vectors: np.ndarray) -> np.ndarray:
    return (plus_vectors.mean(axis=0) - minus_vectors.mean(axis=0)).astype(np.float32)


def _direction_vector(
    plus_vectors: np.ndarray,
    minus_vectors: np.ndarray,
    direction_type: str,
    seed: int,
) -> np.ndarray:
    if direction_type == "pca":
        return _build_direction(plus_vectors=plus_vectors, minus_vectors=minus_vectors, seed=seed)
    if direction_type == "mean_diff":
        return _build_mean_difference(plus_vectors=plus_vectors, minus_vectors=minus_vectors)
    raise ValueError(f"Unsupported direction_type: {direction_type}")


def _masked_vector_with_renorm(
    direction: SteeringDirection,
    strategy: str,
    renorm_mode: str,
) -> np.ndarray:
    vector = np.zeros_like(direction.vector)
    if strategy == "full_vector":
        vector = direction.vector.copy()
    elif strategy == "aens":
        vector[direction.aen_indices] = direction.vector[direction.aen_indices]
    elif strategy == "top_50":
        vector[direction.ranked_indices[:50]] = direction.vector[direction.ranked_indices[:50]]
    elif strategy == "top_100":
        vector[direction.ranked_indices[:100]] = direction.vector[direction.ranked_indices[:100]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if renorm_mode == "none":
        return vector

    masked_norm = float(np.linalg.norm(vector))
    if masked_norm <= 0:
        return vector

    if renorm_mode == "unit":
        target_norm = 1.0
    elif renorm_mode == "full_norm":
        target_norm = float(np.linalg.norm(direction.vector))
    else:
        raise ValueError(f"Unknown renorm_mode: {renorm_mode}")

    if target_norm <= 0:
        return vector
    return (vector * (target_norm / masked_norm)).astype(np.float32)


def _iter_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def _generate_with_position_steering(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    generation_cfg: dict[str, Any],
    layer: int,
    steering_vector: np.ndarray,
    alpha: float,
    position_mode: str,
) -> list[str]:
    decoder_layers = _decoder_layers(bundle.model)
    target_layer = decoder_layers[layer]
    direction = torch.as_tensor(
        steering_vector,
        device=bundle.device,
        dtype=next(bundle.model.parameters()).dtype,
    )
    direction = (alpha * direction).view(1, 1, -1)

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            if position_mode == "last_token":
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + direction.to(dtype=output[0].dtype).view(1, -1)
            elif position_mode == "all_tokens":
                hidden_states = hidden_states + direction.to(dtype=output[0].dtype)
            else:
                raise ValueError(f"Unknown position_mode: {position_mode}")
            return (hidden_states,) + output[1:]

        hidden_states = output.clone()
        if position_mode == "last_token":
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + direction.to(dtype=output.dtype).view(1, -1)
        elif position_mode == "all_tokens":
            hidden_states = hidden_states + direction.to(dtype=output.dtype)
        else:
            raise ValueError(f"Unknown position_mode: {position_mode}")
        return hidden_states

    handle = target_layer.register_forward_hook(hook)
    try:
        return [
            generation.response_text
            for generation in generate_batch(
                bundle=bundle,
                prompt_texts=prompt_texts,
                generation_config=generation_cfg,
            )
        ]
    finally:
        handle.remove()


def _evaluate_variant(
    bundle: HFModelBundle,
    eval_df: pd.DataFrame,
    generation_cfg: dict[str, Any],
    prompt_suffix: str,
    direction: SteeringDirection,
    strategy: str,
    variant: DiagnosticVariant,
) -> pd.DataFrame:
    steer_vector = _masked_vector_with_renorm(direction, strategy=strategy, renorm_mode=variant.renorm_mode)
    rows: list[dict[str, Any]] = []
    batch_size = int(generation_cfg.get("batch_size", 8))
    for batch_df in tqdm(
        _iter_batches(eval_df, batch_size),
        total=(len(eval_df) + batch_size - 1) // batch_size,
        desc=variant.name,
        leave=False,
    ):
        prompts = [
            f"{text}{prompt_suffix}" if prompt_suffix else text
            for text in batch_df["text"].tolist()
        ]
        responses = _generate_with_position_steering(
            bundle=bundle,
            prompt_texts=prompts,
            generation_cfg=generation_cfg,
            layer=direction.layer,
            steering_vector=steer_vector,
            alpha=variant.alpha,
            position_mode=variant.position_mode,
        )
        for row, prompt_text, response_text in zip(batch_df.itertuples(index=False), prompts, responses, strict=True):
            rows.append(
                {
                    "text": row.text,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "variant": variant.name,
                    "direction_type": variant.direction_type,
                    "alpha": variant.alpha,
                    "renorm_mode": variant.renorm_mode,
                    "position_mode": variant.position_mode,
                    "strategy": strategy,
                }
            )
    return pd.DataFrame(rows)


def _variant_specs(config: dict[str, Any]) -> list[DiagnosticVariant]:
    diag_cfg = config["steering_diagnostics"]
    variants: list[DiagnosticVariant] = []
    for alpha in diag_cfg.get("alpha_values", [0.5, 1.0, 2.0, 4.0, 8.0]):
        for renorm_mode in diag_cfg.get("renorm_modes", ["none", "full_norm"]):
            name = f"pca_last_token_alpha{str(alpha).replace('.', 'p')}_{renorm_mode}"
            variants.append(
                DiagnosticVariant(
                    name=name,
                    direction_type="pca",
                    alpha=float(alpha),
                    renorm_mode=str(renorm_mode),
                    position_mode="last_token",
                )
            )

    compare_alpha = float(diag_cfg.get("compare_alpha", 4.0))
    compare_renorm = str(diag_cfg.get("compare_renorm_mode", "full_norm"))
    variants.append(
        DiagnosticVariant(
            name=f"pca_all_tokens_alpha{str(compare_alpha).replace('.', 'p')}_{compare_renorm}",
            direction_type="pca",
            alpha=compare_alpha,
            renorm_mode=compare_renorm,
            position_mode="all_tokens",
        )
    )
    variants.append(
        DiagnosticVariant(
            name=f"mean_diff_last_token_alpha{str(compare_alpha).replace('.', 'p')}_{compare_renorm}",
            direction_type="mean_diff",
            alpha=compare_alpha,
            renorm_mode=compare_renorm,
            position_mode="last_token",
        )
    )
    return variants


def run_steering_diagnostics(
    bundle: HFModelBundle,
    config: dict[str, Any],
    judge: JudgeBackend,
) -> dict[str, Any]:
    diag_cfg = config["steering_diagnostics"]
    dataset_name = str(diag_cfg.get("dataset", "ambigqa"))
    strategy = str(diag_cfg.get("strategy", "aens"))
    seed = int(config["seed"])

    baseline_root = _baseline_artifact_root(config)
    behavior_df = pd.read_parquet(baseline_root / f"{dataset_name}__base_behavior.parquet")
    output_root = _output_root(config)

    clear_df = _load_clear_df(config, dataset_name)
    abstain_df = behavior_df.loc[behavior_df["judge_label"] == "ACCEPTABLE", ["text"]].drop_duplicates().reset_index(drop=True)
    answer_df = behavior_df.loc[behavior_df["judge_label"] == "UNACCEPTABLE", ["text"]].drop_duplicates().reset_index(drop=True)

    probe_descriptor = _direction_from_probe_report(config, dataset_name)
    build_abstain_n = min(int(diag_cfg.get("build_abstain_n", 100)), len(abstain_df))
    build_clear_n = min(int(diag_cfg.get("build_clear_n", 100)), len(clear_df))
    eval_n = min(int(diag_cfg.get("eval_direct_answer_n", 500)), len(answer_df))

    abstain_sample = abstain_df.sample(n=build_abstain_n, random_state=seed).reset_index(drop=True)
    clear_sample = clear_df.sample(n=build_clear_n, random_state=seed).reset_index(drop=True)
    eval_df = answer_df.sample(n=eval_n, random_state=seed).reset_index(drop=True)

    prompt_suffix = config["steering"].get("prompt_suffix", "")
    plus_prompts = _prompt_texts(bundle, abstain_sample["text"].tolist(), config["generation"], prompt_suffix)
    minus_prompts = _prompt_texts(bundle, clear_sample["text"].tolist(), config["generation"], prompt_suffix)
    plus_vectors = _extract_prompt_vectors(bundle, plus_prompts, config["extraction"], probe_descriptor.layer)
    minus_vectors = _extract_prompt_vectors(bundle, minus_prompts, config["extraction"], probe_descriptor.layer)

    direction_map: dict[str, SteeringDirection] = {}
    for direction_type in sorted({variant.direction_type for variant in _variant_specs(config)}):
        vector = _direction_vector(plus_vectors, minus_vectors, direction_type=direction_type, seed=seed)
        direction_map[direction_type] = SteeringDirection(
            vector=vector,
            aen_indices=probe_descriptor.aen_indices,
            ranked_indices=probe_descriptor.ranked_indices,
            layer=probe_descriptor.layer,
        )
        np.save(output_root / f"{dataset_name}__direction__{direction_type}.npy", vector)

    summary_rows: list[dict[str, Any]] = []
    for variant in _variant_specs(config):
        raw_path = output_root / f"{dataset_name}__{variant.name}__raw.parquet"
        judged_path = output_root / f"{dataset_name}__{variant.name}.parquet"

        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
        else:
            raw_df = _evaluate_variant(
                bundle=bundle,
                eval_df=eval_df,
                generation_cfg=config["generation"],
                prompt_suffix=prompt_suffix,
                direction=direction_map[variant.direction_type],
                strategy=strategy,
                variant=variant,
            )
            write_parquet(raw_df, raw_path)

        if judged_path.exists():
            judged_df = pd.read_parquet(judged_path)
        else:
            judged_df = _judge_table(
                judge,
                raw_df,
                batch_size=int(config["judge"].get("batch_size", 8)),
            )
            write_parquet(judged_df, judged_path)

        abstention_rate = float(judged_df["judge_label"].eq("ACCEPTABLE").mean()) if not judged_df.empty else float("nan")
        summary_rows.append(
            {
                "variant": variant.name,
                "direction_type": variant.direction_type,
                "alpha": variant.alpha,
                "renorm_mode": variant.renorm_mode,
                "position_mode": variant.position_mode,
                "abstention_rate": abstention_rate,
                "n_eval": int(len(judged_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["abstention_rate", "alpha", "variant"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    write_parquet(summary_df, output_root / f"{dataset_name}__diagnostic_summary.parquet")
    payload = {
        "model_name": config["model"]["name"],
        "dataset": dataset_name,
        "strategy": strategy,
        "build_abstain_n": build_abstain_n,
        "build_clear_n": build_clear_n,
        "eval_n": eval_n,
        "baseline_artifact_dir": str(baseline_root),
        "results": summary_df.to_dict(orient="records"),
    }
    write_json(output_root / f"{dataset_name}__diagnostic_summary.json", payload)
    return payload
