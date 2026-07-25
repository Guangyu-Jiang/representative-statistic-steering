"""CLAMBER steering with topology-surrogate-guided prompt perturbations."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from aen_replication.eval.judge import JudgeResult, load_judge
from aen_replication.models.generation import _build_generate_kwargs, render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.clamber_subclass_classification import _build_clamber_token_cloud_features
from aen_replication.train.clamber_topology_surrogate import DeepSetTopologySurrogate
from aen_replication.train.token_cloud_topology_classifier import (
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
    _valid_token_mask,
)
from aen_replication.utils.io_utils import ensure_dir, write_json, write_markdown, write_parquet
from aen_replication.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

_JOIN_COLUMNS = ["example_id", "pair_id", "dataset", "split", "label_ambiguous"]


@dataclass(slots=True)
class _PCAProjector:
    mean: torch.Tensor
    components_t: torch.Tensor

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        centered = hidden_states - self.mean.view(1, 1, -1)
        return torch.matmul(centered, self.components_t)


def _resolve_token_cfg(config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    classifier_cfg = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    token_cfg = {
        **classifier_cfg,
        "batch_size": int(subclass_cfg.get("token_cloud_batch_size", classifier_cfg.get("batch_size", 8))),
        "max_length": int(subclass_cfg.get("token_cloud_max_length", classifier_cfg.get("max_length", 64))),
        "parallel_jobs": int(subclass_cfg.get("token_cloud_parallel_jobs", classifier_cfg.get("parallel_jobs", 12))),
        "pca_components": int(subclass_cfg.get("token_cloud_pca_components", classifier_cfg.get("pca_components", 16))),
        "topology_components": int(subclass_cfg.get("token_cloud_topology_components", classifier_cfg.get("topology_components", 16))),
        "prototype_token_cap": int(subclass_cfg.get("token_cloud_prototype_token_cap", classifier_cfg.get("prototype_token_cap", 192))),
        "distance_feature_mode": str(
            subclass_cfg.get("token_cloud_distance_feature_mode", classifier_cfg.get("distance_feature_mode", "none"))
        ),
        "distance_feature_k": int(
            subclass_cfg.get("token_cloud_distance_feature_k", classifier_cfg.get("distance_feature_k", 8))
        ),
        "distance_feature_chunk_size": int(
            subclass_cfg.get("token_cloud_distance_feature_chunk_size", classifier_cfg.get("distance_feature_chunk_size", 24))
        ),
        "subclass_distance_max_workers": int(
            subclass_cfg.get(
                "token_cloud_subclass_distance_max_workers",
                classifier_cfg.get("subclass_distance_max_workers", 2),
            )
        ),
        "subclass_distance_executor": str(
            subclass_cfg.get(
                "token_cloud_subclass_distance_executor",
                classifier_cfg.get("subclass_distance_executor", "process"),
            )
        ),
        "betti_grid_size": int(subclass_cfg.get("token_cloud_betti_grid_size", classifier_cfg.get("betti_grid_size", 24))),
        "persistence_image_grid_side": int(
            subclass_cfg.get("token_cloud_persistence_image_grid_side", classifier_cfg.get("persistence_image_grid_side", 3))
        ),
        "_seed": seed,
    }
    return classifier_cfg, token_cfg


def _load_behavior_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {"example_id", "pair_id", "split", "text", "response_text", "judge_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Behavior table missing required columns {sorted(missing)}: {path}")
    return df


def _load_surrogate_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[DeepSetTopologySurrogate, list[str], int, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    config = dict(payload["config"])
    model = DeepSetTopologySurrogate(
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        token_hidden_dim=int(config["token_hidden_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
        dropout=float(config["dropout"]),
        include_token_count_input=bool(config["include_token_count_input"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    target_mean = torch.tensor(payload["target_mean"], dtype=torch.float32, device=device).view(1, -1)
    target_std = torch.tensor(payload["target_std"], dtype=torch.float32, device=device).view(1, -1)
    return model, list(payload["target_columns"]), int(payload["layer"]), target_mean, target_std


def _select_reliable_target_columns(
    *,
    checkpoint_path: Path,
    layer: int,
    target_columns: list[str],
    min_r2: float,
    min_test_std: float,
) -> tuple[list[str], pd.DataFrame]:
    root = checkpoint_path.parent
    feature_metrics_path = root / "clamber_topology_surrogate_feature_metrics.parquet"
    test_predictions_path = root / f"layer_{int(layer):02d}__test_predictions.parquet"
    if not feature_metrics_path.exists() or not test_predictions_path.exists():
        LOGGER.warning(
            "Reliable-target selection artifacts are missing; using all surrogate targets. feature_metrics=%s test_predictions=%s",
            feature_metrics_path,
            test_predictions_path,
        )
        return target_columns, pd.DataFrame()

    feature_metrics = pd.read_parquet(feature_metrics_path)
    feature_metrics = feature_metrics.loc[
        feature_metrics["split"].eq("test") & feature_metrics["layer"].eq(int(layer))
    ].copy()
    feature_metrics = feature_metrics.loc[feature_metrics["feature"].isin(target_columns)].copy()
    if feature_metrics.empty:
        return target_columns, pd.DataFrame()

    test_predictions = pd.read_parquet(test_predictions_path)
    std_rows: list[dict[str, Any]] = []
    for feature in target_columns:
        column = f"target__{feature}"
        if column not in test_predictions.columns:
            continue
        std_rows.append(
            {
                "feature": feature,
                "test_std": float(np.std(test_predictions[column].to_numpy(dtype=np.float32))),
            }
        )
    std_df = pd.DataFrame(std_rows)
    selected_df = feature_metrics.merge(std_df, on="feature", how="left")
    selected_df["test_std"] = selected_df["test_std"].fillna(0.0)
    selected_df["selected"] = selected_df["r2"].ge(float(min_r2)) & selected_df["test_std"].gt(float(min_test_std))
    selected_columns = selected_df.loc[selected_df["selected"], "feature"].astype(str).tolist()
    if not selected_columns:
        raise ValueError(
            "Reliable-target selection kept zero features. "
            f"min_r2={float(min_r2):.4f} min_test_std={float(min_test_std):.4e}"
        )
    return selected_columns, selected_df.sort_values(["selected", "r2", "feature"], ascending=[False, False, True]).reset_index(drop=True)


def _fit_or_load_reducer(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    token_cfg: dict[str, Any],
    layer: int,
    output_path: Path,
    seed: int,
) -> PCA:
    if output_path.exists():
        reducer = joblib.load(output_path)
        if not isinstance(reducer, PCA):
            raise ValueError(f"Reducer cache is not a PCA instance: {output_path}")
        return reducer

    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    dataset_df = pd.read_parquet(dataset_path)
    bundle_for_prepare = bundle
    prepared_df, prepared_text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle_for_prepare,
        text_column=str(token_cfg.get("text_column", "text")),
        use_chat_template=bool(token_cfg.get("use_chat_template", False)),
        system_prompt=token_cfg.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
    train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=[int(layer)],
        config=token_cfg,
    )
    reducers = _fit_layer_reducers(token_matrices, config=token_cfg, seed=seed)
    reducer = reducers[int(layer)]
    if reducer is None:
        raise ValueError(f"Expected PCA reducer for layer {layer}, but PCA is disabled.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, output_path)
    return reducer


def _build_projector(reducer: PCA, *, device: torch.device) -> _PCAProjector:
    mean = torch.tensor(np.asarray(reducer.mean_, dtype=np.float32), device=device)
    components_t = torch.tensor(np.asarray(reducer.components_, dtype=np.float32).T, device=device)
    return _PCAProjector(mean=mean, components_t=components_t)


def _merge_behavior_features(
    behavior_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    layer: int,
    target_columns: list[str],
) -> pd.DataFrame:
    layer_features = feature_df.loc[feature_df["layer"].eq(int(layer)), _JOIN_COLUMNS + target_columns].copy()
    merged = behavior_df.merge(layer_features, on=_JOIN_COLUMNS, how="inner")
    if merged.empty:
        raise ValueError("No overlap between CLAMBER behavior table and topology features at the steering layer.")
    return merged


def _sample_group(df: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _delta_z_from_behavior(
    merged_df: pd.DataFrame,
    *,
    target_columns: list[str],
    build_split: str,
    build_abstain_n: int,
    build_direct_n: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    build_df = merged_df.loc[merged_df["split"].eq(build_split)].copy()
    abstain_df = build_df.loc[build_df["judge_label"].eq("ACCEPTABLE")].copy()
    direct_df = build_df.loc[build_df["judge_label"].eq("UNACCEPTABLE")].copy()
    abstain_sample = _sample_group(abstain_df, n=build_abstain_n, seed=seed)
    direct_sample = _sample_group(direct_df, n=build_direct_n, seed=seed)
    if abstain_sample.empty or direct_sample.empty:
        raise ValueError("Need non-empty abstain and direct pools to build topology steering shift.")
    mu_abstain = abstain_sample.loc[:, target_columns].to_numpy(dtype=np.float32).mean(axis=0)
    mu_direct = direct_sample.loc[:, target_columns].to_numpy(dtype=np.float32).mean(axis=0)
    delta_z = mu_abstain - mu_direct
    summary = {
        "build_split": build_split,
        "n_build_abstain_pool": int(len(abstain_df)),
        "n_build_direct_pool": int(len(direct_df)),
        "n_build_abstain_used": int(len(abstain_sample)),
        "n_build_direct_used": int(len(direct_sample)),
        "delta_z_l2_norm": float(np.linalg.norm(delta_z)),
    }
    return delta_z, summary


def _eval_pool(
    merged_df: pd.DataFrame,
    *,
    eval_split: str,
    eval_direct_answer_n: int,
    seed: int,
) -> pd.DataFrame:
    eval_df = merged_df.loc[
        merged_df["split"].eq(eval_split) & merged_df["judge_label"].eq("UNACCEPTABLE")
    ].copy()
    return _sample_group(eval_df, n=eval_direct_answer_n, seed=seed)


def _encode_rendered_prompts(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    *,
    generation_cfg: dict[str, Any],
) -> tuple[list[str], dict[str, torch.Tensor]]:
    rendered_prompts = render_prompts(
        bundle=bundle,
        prompt_texts=prompt_texts,
        use_chat_template=bool(generation_cfg.get("use_chat_template", True)),
        system_prompt=generation_cfg.get("system_prompt"),
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
    return rendered_prompts, encoded


def _valid_masks_from_encoded(
    bundle: HFModelBundle,
    encoded: dict[str, torch.Tensor],
    *,
    drop_special_tokens: bool,
) -> torch.Tensor:
    special_ids = set(int(token_id) for token_id in getattr(bundle.tokenizer, "all_special_ids", []) if token_id is not None)
    input_ids_cpu = encoded["input_ids"].detach().cpu()
    attention_mask_cpu = encoded["attention_mask"].detach().cpu()
    masks = []
    for row_index in range(input_ids_cpu.shape[0]):
        masks.append(
            _valid_token_mask(
                input_ids_cpu[row_index],
                attention_mask_cpu[row_index],
                special_ids=special_ids,
                drop_special_tokens=drop_special_tokens,
            )
        )
    return torch.stack(masks, dim=0).to(bundle.device)


def _prompt_layer_hidden_states(
    bundle: HFModelBundle,
    encoded: dict[str, torch.Tensor],
    *,
    layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = bundle.model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states for topology steering.")
    return hidden_states[layer + 1].detach().to(torch.float32)


def _predict_topology_targets(
    surrogate: DeepSetTopologySurrogate,
    projector: _PCAProjector,
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    reduced = projector.project(hidden_states)
    token_counts = valid_mask.sum(dim=1).to(hidden_states.dtype)
    prediction_norm = surrogate(reduced, valid_mask, token_counts)
    return prediction_norm * target_std + target_mean


def _optimize_delta_h(
    *,
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
    surrogate: DeepSetTopologySurrogate,
    projector: _PCAProjector,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_indices: torch.Tensor,
    delta_z: torch.Tensor,
    alpha: float,
    lambda_l2: float,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, float], dict[str, list[float]]]:
    batch_size = int(hidden_states.shape[0])
    with torch.no_grad():
        current_z = _predict_topology_targets(
            surrogate,
            projector,
            hidden_states,
            valid_mask,
            target_mean,
            target_std,
        ).index_select(1, target_indices)
    target_z = current_z + float(alpha) * delta_z.view(1, -1).expand(batch_size, -1)
    delta = torch.nn.Parameter(torch.zeros_like(hidden_states))
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    mask_f = valid_mask.unsqueeze(-1).float()
    losses: list[float] = []
    fit_losses: list[float] = []
    reg_losses: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        masked_delta = delta * mask_f
        predicted = _predict_topology_targets(
            surrogate,
            projector,
            hidden_states + masked_delta,
            valid_mask,
            target_mean,
            target_std,
        ).index_select(1, target_indices)
        fit_loss = F.mse_loss(predicted, target_z)
        reg_loss = masked_delta.pow(2).mean()
        loss = fit_loss + float(lambda_l2) * reg_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        fit_losses.append(float(fit_loss.item()))
        reg_losses.append(float(reg_loss.item()))
    final_delta = (delta.detach() * mask_f).to(hidden_states.dtype)
    with torch.no_grad():
        final_z = _predict_topology_targets(
            surrogate,
            projector,
            hidden_states + final_delta,
            valid_mask,
            target_mean,
            target_std,
        ).index_select(1, target_indices)
        error_initial = current_z - target_z
        error_final = final_z - target_z
        per_example_mse_initial = error_initial.pow(2).mean(dim=1).detach().cpu().numpy().tolist()
        per_example_mse_final = error_final.pow(2).mean(dim=1).detach().cpu().numpy().tolist()
        per_example_l2_initial = error_initial.pow(2).sum(dim=1).sqrt().detach().cpu().numpy().tolist()
        per_example_l2_final = error_final.pow(2).sum(dim=1).sqrt().detach().cpu().numpy().tolist()
    stats = {
        "objective_initial": float(losses[0]) if losses else 0.0,
        "objective_final": float(losses[-1]) if losses else 0.0,
        "fit_loss_initial": float(fit_losses[0]) if fit_losses else 0.0,
        "fit_loss_final": float(fit_losses[-1]) if fit_losses else 0.0,
        "reg_loss_initial": float(reg_losses[0]) if reg_losses else 0.0,
        "reg_loss_final": float(reg_losses[-1]) if reg_losses else 0.0,
        "delta_l2_norm": float(final_delta.pow(2).sum().sqrt().item()),
        "surrogate_target_mse_initial_mean": float(np.mean(per_example_mse_initial)) if per_example_mse_initial else 0.0,
        "surrogate_target_mse_final_mean": float(np.mean(per_example_mse_final)) if per_example_mse_final else 0.0,
        "surrogate_target_l2_initial_mean": float(np.mean(per_example_l2_initial)) if per_example_l2_initial else 0.0,
        "surrogate_target_l2_final_mean": float(np.mean(per_example_l2_final)) if per_example_l2_final else 0.0,
    }
    diagnostics = {
        "surrogate_target_mse_initial": per_example_mse_initial,
        "surrogate_target_mse_final": per_example_mse_final,
        "surrogate_target_l2_initial": per_example_l2_initial,
        "surrogate_target_l2_final": per_example_l2_final,
    }
    return final_delta, stats, diagnostics


def _decode_generation_output(
    bundle: HFModelBundle,
    rendered_prompts: list[str],
    encoded: dict[str, torch.Tensor],
    generation_output: Any,
) -> list[dict[str, Any]]:
    prompt_length = int(encoded["input_ids"].shape[1])
    results: list[dict[str, Any]] = []
    pad_token_id = bundle.tokenizer.pad_token_id
    eos_token_id = bundle.tokenizer.eos_token_id
    for index, rendered_prompt in enumerate(rendered_prompts):
        generated_ids = generation_output.sequences[index, prompt_length:]
        generated_ids_cpu = generated_ids.detach().cpu()
        token_ids = generated_ids_cpu.tolist()
        effective_token_count = len(token_ids)
        if pad_token_id is not None:
            effective_token_count = sum(token_id != pad_token_id for token_id in token_ids)
        if eos_token_id is not None and eos_token_id in token_ids:
            effective_token_count = min(effective_token_count, token_ids.index(eos_token_id) + 1)
        response_text = bundle.tokenizer.decode(generated_ids_cpu, skip_special_tokens=True).strip()
        results.append(
            {
                "prompt_text": rendered_prompt,
                "response_text": response_text,
                "generated_token_count": int(effective_token_count),
            }
        )
    return results


def _generate_with_delta_h(
    *,
    bundle: HFModelBundle,
    encoded: dict[str, torch.Tensor],
    rendered_prompts: list[str],
    layer: int,
    delta_h: torch.Tensor,
    generation_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    model = bundle.model
    decoder_layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else None
    if decoder_layers is None:
        raise ValueError(f"Unsupported architecture for topology steering: {type(model)!r}")
    target_layer = decoder_layers[int(layer)]
    delta_h = delta_h.to(device=bundle.device, dtype=next(model.parameters()).dtype)
    apply_once = {"done": False}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if apply_once["done"]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[:2] != delta_h.shape[:2]:
            return output
        steered = hidden.clone()
        steered = steered + delta_h.to(dtype=hidden.dtype)
        apply_once["done"] = True
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    handle = target_layer.register_forward_hook(hook)
    try:
        generate_kwargs = _build_generate_kwargs(generation_cfg, return_entropy=False)
        generate_kwargs["pad_token_id"] = bundle.tokenizer.pad_token_id
        generate_kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id
        with torch.no_grad():
            generation_output = model.generate(**encoded, **generate_kwargs)
        return _decode_generation_output(bundle, rendered_prompts, encoded, generation_output)
    finally:
        handle.remove()


def _judge_outputs(config: dict[str, Any], raw_df: pd.DataFrame) -> pd.DataFrame:
    judge = load_judge(config)
    batch_size = int(config["judge"].get("batch_size", 8))
    try:
        results: list[JudgeResult] = []
        for start in range(0, len(raw_df), batch_size):
            batch = raw_df.iloc[start : start + batch_size]
            if hasattr(judge, "judge_many"):
                batch_results = judge.judge_many(batch["text"].tolist(), batch["response_text"].tolist(), batch_size=batch_size)
            else:
                batch_results = [judge.judge(row.text, row.response_text) for row in batch.itertuples(index=False)]
            results.extend(batch_results)
    finally:
        if hasattr(judge, "model"):
            delattr(judge, "model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    judged_df = raw_df.copy()
    judged_df["judge_label"] = [result.label for result in results]
    judged_df["judge_explanation"] = [result.explanation for result in results]
    judged_df["judge_raw_response"] = [result.raw_response for result in results]
    return judged_df


def run_clamber_topology_steering(*, config: dict[str, Any], seed: int) -> dict[str, str]:
    set_global_seed(seed)
    topo_cfg = dict(config["clamber_topology_steering"])
    output_root = ensure_dir(Path(topo_cfg.get("output_dir", "artifacts/reports/clamber_topology_steering")) / config["model"]["name"].replace("/", "_").replace("-", "_"))
    classifier_cfg, token_cfg = _resolve_token_cfg(config, seed)
    subclass_cfg = dict(config["clamber_subclass_classification"])
    subclass_cfg["token_cloud_distance_feature_mode"] = "none"
    if topo_cfg.get("token_cloud_feature_cache_path"):
        subclass_cfg["token_cloud_feature_cache_path"] = str(Path(topo_cfg["token_cloud_feature_cache_path"]))
    if topo_cfg.get("token_cloud_forward_cache_path"):
        subclass_cfg["token_cloud_forward_cache_path"] = str(Path(topo_cfg["token_cloud_forward_cache_path"]))

    checkpoint_path = Path(topo_cfg["surrogate_checkpoint_path"])
    bundle: HFModelBundle | None = None
    try:
        model_name, feature_df, reused_feature_cache = _build_clamber_token_cloud_features(
            config=config,
            classifier_config=classifier_cfg,
            subclass_cfg=subclass_cfg,
            seed=seed,
        )
        bundle = load_hf_model(config["model"], config["generation"])
        target_columns_all, layer = None, None
        surrogate, target_columns_all, layer, target_mean, target_std = _load_surrogate_checkpoint(checkpoint_path, device=bundle.device)
        target_columns, selected_target_df = _select_reliable_target_columns(
            checkpoint_path=checkpoint_path,
            layer=layer,
            target_columns=target_columns_all,
            min_r2=float(topo_cfg.get("min_target_r2", 0.8)),
            min_test_std=float(topo_cfg.get("min_target_test_std", 1e-8)),
        ) if bool(topo_cfg.get("use_reliable_target_subset", True)) else (target_columns_all, pd.DataFrame())
        target_indices = torch.tensor(
            [target_columns_all.index(column) for column in target_columns],
            dtype=torch.long,
            device=bundle.device,
        )
        reducer_cache_path = Path(topo_cfg.get("reducer_cache_path", output_root / f"layer_{layer:02d}__pca_reducer.joblib"))
        reducer = _fit_or_load_reducer(
            bundle=bundle,
            config=config,
            token_cfg=token_cfg,
            layer=layer,
            output_path=reducer_cache_path,
            seed=seed,
        )
        projector = _build_projector(reducer, device=bundle.device)

        behavior_df = _load_behavior_df(Path(topo_cfg["base_behavior_path"]))
        merged_df = _merge_behavior_features(behavior_df, feature_df, layer=layer, target_columns=target_columns)
        delta_z_np, build_summary = _delta_z_from_behavior(
            merged_df,
            target_columns=target_columns,
            build_split=str(topo_cfg.get("build_split", "train")),
            build_abstain_n=int(topo_cfg.get("build_abstain_n", 100)),
            build_direct_n=int(topo_cfg.get("build_direct_n", 100)),
            seed=seed,
        )
        delta_z = torch.tensor(delta_z_np, dtype=torch.float32, device=bundle.device)
        eval_df = _eval_pool(
            merged_df,
            eval_split=str(topo_cfg.get("eval_split", "test")),
            eval_direct_answer_n=int(topo_cfg.get("eval_direct_answer_n", 200)),
            seed=seed,
        )
        if eval_df.empty:
            raise ValueError("Topology steering evaluation pool is empty.")

        raw_rows: list[dict[str, Any]] = []
        batch_size = int(topo_cfg.get("generation_batch_size", 4))
        drop_special_tokens = bool(token_cfg.get("drop_special_tokens", True))
        for start in tqdm(range(0, len(eval_df), batch_size), desc="topology_steering", leave=False):
            batch_df = eval_df.iloc[start : start + batch_size].reset_index(drop=True)
            prompt_texts = batch_df["text"].astype(str).tolist()
            rendered_prompts, encoded = _encode_rendered_prompts(bundle, prompt_texts, generation_cfg=config["generation"])
            valid_mask = _valid_masks_from_encoded(bundle, encoded, drop_special_tokens=drop_special_tokens)
            hidden_states = _prompt_layer_hidden_states(bundle, encoded, layer=layer)
            delta_h, opt_stats, opt_diag = _optimize_delta_h(
                hidden_states=hidden_states,
                valid_mask=valid_mask,
                surrogate=surrogate,
                projector=projector,
                target_mean=target_mean,
                target_std=target_std,
                target_indices=target_indices,
                delta_z=delta_z,
                alpha=float(topo_cfg.get("alpha", 1.0)),
                lambda_l2=float(topo_cfg.get("lambda_l2", 1e-4)),
                steps=int(topo_cfg.get("opt_steps", 80)),
                lr=float(topo_cfg.get("opt_lr", 0.05)),
            )
            generated = _generate_with_delta_h(
                bundle=bundle,
                encoded=encoded,
                rendered_prompts=rendered_prompts,
                layer=layer,
                delta_h=delta_h,
                generation_cfg=config["generation"],
            )
            per_example_delta_norm = delta_h.pow(2).sum(dim=(1, 2)).sqrt().detach().cpu().numpy().tolist()
            for row, generation_row, delta_norm, mse_before, mse_after, l2_before, l2_after in zip(
                batch_df.itertuples(index=False),
                generated,
                per_example_delta_norm,
                opt_diag["surrogate_target_mse_initial"],
                opt_diag["surrogate_target_mse_final"],
                opt_diag["surrogate_target_l2_initial"],
                opt_diag["surrogate_target_l2_final"],
                strict=True,
            ):
                raw_rows.append(
                    {
                        "example_id": row.example_id,
                        "pair_id": row.pair_id,
                        "dataset": row.dataset,
                        "split": row.split,
                        "label_ambiguous": int(row.label_ambiguous),
                        "text": row.text,
                        "prompt_text": generation_row["prompt_text"],
                        "response_text": generation_row["response_text"],
                        "generated_token_count": int(generation_row["generated_token_count"]),
                        "layer": int(layer),
                        "alpha": float(topo_cfg.get("alpha", 1.0)),
                        "lambda_l2": float(topo_cfg.get("lambda_l2", 1e-4)),
                        "opt_steps": int(topo_cfg.get("opt_steps", 80)),
                        "opt_lr": float(topo_cfg.get("opt_lr", 0.05)),
                        "delta_h_l2_norm": float(delta_norm),
                        "delta_z_l2_norm": float(np.linalg.norm(delta_z_np)),
                        "objective_initial": float(opt_stats["objective_initial"]),
                        "objective_final": float(opt_stats["objective_final"]),
                        "fit_loss_initial": float(opt_stats["fit_loss_initial"]),
                        "fit_loss_final": float(opt_stats["fit_loss_final"]),
                        "surrogate_target_mse_initial": float(mse_before),
                        "surrogate_target_mse_final": float(mse_after),
                        "surrogate_target_l2_initial": float(l2_before),
                        "surrogate_target_l2_final": float(l2_after),
                    }
                )
        raw_df = pd.DataFrame(raw_rows)
        raw_path = output_root / "clamber__topology_steering__raw.parquet"
        write_parquet(raw_df, raw_path)

        outputs: dict[str, str] = {
            "output_root": str(output_root),
            "raw_path": str(raw_path),
        }
        summary: dict[str, Any] = {
            "model_name": model_name,
            "layer": int(layer),
            "n_eval": int(len(raw_df)),
            "target_dim": int(len(target_columns)),
            "surrogate_target_dim": int(len(target_columns_all)),
            "surrogate_target_mse_initial_mean": float(raw_df["surrogate_target_mse_initial"].mean()),
            "surrogate_target_mse_final_mean": float(raw_df["surrogate_target_mse_final"].mean()),
            "surrogate_target_l2_initial_mean": float(raw_df["surrogate_target_l2_initial"].mean()),
            "surrogate_target_l2_final_mean": float(raw_df["surrogate_target_l2_final"].mean()),
            "reused_feature_cache": bool(reused_feature_cache),
            "surrogate_checkpoint_path": str(checkpoint_path),
            "reducer_cache_path": str(reducer_cache_path),
            **build_summary,
        }
        if not selected_target_df.empty:
            selected_target_path = output_root / "selected_target_metrics.parquet"
            write_parquet(selected_target_df, selected_target_path)
            summary["selected_target_metrics_path"] = str(selected_target_path)
            summary["selected_target_columns"] = [str(column) for column in target_columns]
            summary["min_target_r2"] = float(topo_cfg.get("min_target_r2", 0.8))
            summary["min_target_test_std"] = float(topo_cfg.get("min_target_test_std", 1e-8))
            outputs["selected_target_metrics_path"] = str(selected_target_path)

        if bool(topo_cfg.get("judge_outputs", False)):
            judged_df = _judge_outputs(config, raw_df)
            judged_path = output_root / "clamber__topology_steering.parquet"
            write_parquet(judged_df, judged_path)
            summary["abstention_rate"] = float(judged_df["judge_label"].eq("ACCEPTABLE").mean())
            summary["judge_label_counts"] = {
                str(label): int(count)
                for label, count in judged_df["judge_label"].value_counts(dropna=False).to_dict().items()
            }
            outputs["judged_path"] = str(judged_path)

        write_json(output_root / "summary.json", summary)
        report_lines = [
            "# CLAMBER Topology Steering",
            "",
            f"- Model: `{model_name}`",
            f"- Layer: `{layer}`",
            f"- Eval examples: `{int(len(raw_df))}`",
            f"- Target feature dim: `{len(target_columns)}`",
            f"- Delta-z L2 norm: `{float(np.linalg.norm(delta_z_np)):.4f}`",
            f"- Build abstain/direct used: `{build_summary['n_build_abstain_used']}/{build_summary['n_build_direct_used']}`",
        ]
        if "abstention_rate" in summary:
            report_lines.append(f"- Judged abstention rate: `{float(summary['abstention_rate']):.4f}`")
        write_markdown(output_root / "summary.md", "\n".join(report_lines) + "\n")
        outputs["summary_json"] = str(output_root / "summary.json")
        outputs["summary_md"] = str(output_root / "summary.md")
        return outputs
    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
