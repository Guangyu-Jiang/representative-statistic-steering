"""Low-label and cross-dataset transfer study under the author-style pipeline.

This script fixes the input construction to the first author's prompt format and
layer-14 hidden-state extraction, then compares:

- full hidden-state probes
- official super-neuron probes
- intrinsic token-cloud topology probes

The main goal is to test whether topology is more sample-efficient or more
transferable than sparse neuron probes, even if it is weaker on in-domain IID
accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.author_repo_eval import (
    DATASET_SPECS,
    MODEL_SPECS,
    _extract_author_style_hidden_states,
    _load_author_prompts,
    _shuffle_and_slice_author_style,
)
from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.eval.tune_token_cloud_feature_tables import _feature_subset_columns
from aen_replication.train.independent_topology_classifier import _fit_classifier, _transform_with_scaler
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


CLASSIFIER_CONFIG: dict[str, Any] = {
    "family": "logistic",
    "penalty": "l2",
    "solver": "liblinear",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 4000,
    "standardize": True,
}

TOKEN_CLOUD_ROOTS: dict[str, Path] = {
    "llama31_8b_instruct": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_author_prompts_large_1000_2000_llama/meta_llama_Llama_3.1_8B_Instruct/token_cloud_topology_features.parquet"
    ),
    "mistral_7b_instruct_v03": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_author_prompts_large_1000_2000_mistral/mistralai_Mistral_7B_Instruct_v0.3/token_cloud_topology_features.parquet"
    ),
    "gemma_7b_it": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_author_prompts_large_1000_2000_gemma/google_gemma_7b_it/token_cloud_topology_features.parquet"
    ),
}

INTRINSIC_SUBSET = "descriptors_only"


def _predict_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(matrix), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    predictions = np.asarray(model.predict(matrix), dtype=float)
    return np.clip(predictions, 0.0, 1.0)


def _threshold_grid(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    quantiles = np.linspace(0.02, 0.98, 97)
    grid = np.unique(np.clip(np.quantile(probs, quantiles), 0.0, 1.0))
    if grid.size == 0:
        return np.array([0.5], dtype=float)
    return np.unique(np.concatenate([grid, np.array([0.5], dtype=float)]))


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = binary_classification_metrics(y_true, probabilities, threshold=best_threshold)
    best_score = (
        float(best_metrics["accuracy"]),
        float(best_metrics["f1"]),
        -abs(best_threshold - 0.5),
    )
    for threshold in _threshold_grid(probabilities):
        metrics = binary_classification_metrics(y_true, probabilities, threshold=float(threshold))
        score = (
            float(metrics["accuracy"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def _stratified_train_val_split(
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in [0, 1]:
        label_idx = np.where(labels == label)[0]
        shuffled = rng.permutation(label_idx)
        val_size = max(1, int(round(len(label_idx) * val_fraction)))
        val_parts.append(np.sort(shuffled[:val_size]))
        train_parts.append(np.sort(shuffled[val_size:]))
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    return train_idx, val_idx


def _evaluate_binary(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> dict[str, float]:
    inner_train_idx, val_idx = _stratified_train_val_split(train_y, val_fraction=val_fraction, seed=seed)
    fit_X = train_X[inner_train_idx]
    fit_y = train_y[inner_train_idx]
    val_X = train_X[val_idx]
    val_y = train_y[val_idx]
    clf, scaler = _fit_classifier(fit_X, fit_y, config=CLASSIFIER_CONFIG, seed=seed)
    val_matrix = _transform_with_scaler(val_X, scaler)
    test_matrix = _transform_with_scaler(test_X, scaler)
    val_scores = _predict_scores(clf, val_matrix)
    test_scores = _predict_scores(clf, test_matrix)
    threshold, _ = _select_threshold(val_y, val_scores)
    metrics = binary_classification_metrics(test_y, test_scores, threshold=threshold)
    return {
        "accuracy": float(metrics["accuracy"]),
        "auroc": float(roc_auc_score(test_y, test_scores)),
        "f1": float(metrics["f1"]),
        "threshold": float(threshold),
    }


def _load_or_extract_matrix(
    *,
    cache_dir: Path,
    author_repo_root: Path,
    model_key: str,
    dataset_name: str,
    train_per_class: int,
    test_per_class: int,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{dataset_name}__author_layer14_{train_per_class}_{test_per_class}.npz"
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return {
            "train_X": payload["train_X"],
            "train_y": payload["train_y"],
            "test_X": payload["test_X"],
            "test_y": payload["test_y"],
            "avg_token_length": float(payload["avg_token_length"]),
        }

    spec = MODEL_SPECS[model_key]
    model_path = snapshot_download(repo_id=spec.load_repo, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    try:
        ambig_prompts, clear_prompts = _load_author_prompts(
            author_repo_root=author_repo_root,
            dataset_name=dataset_name,
            prompt_model_name=spec.prompt_model_name,
        )
        split_payload = _shuffle_and_slice_author_style(
            ambig_prompts,
            clear_prompts,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
        )
        texts = (
            split_payload["train_ambig"]
            + split_payload["train_clear"]
            + split_payload["test_ambig"]
            + split_payload["test_clear"]
        )
        matrix, lengths = _extract_author_style_hidden_states(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            layer_index=14,
            batch_size=batch_size,
        )
        train_end = train_per_class * 2
        test_ambig_start = train_end
        test_clear_start = train_end + test_per_class
        train_X = np.vstack([matrix[:train_per_class], matrix[train_per_class:train_end]])
        test_X = np.vstack(
            [
                matrix[test_ambig_start:test_clear_start],
                matrix[test_clear_start:test_clear_start + test_per_class],
            ]
        )
        train_y = np.array([0] * train_per_class + [1] * train_per_class, dtype=int)
        test_y = np.array([0] * test_per_class + [1] * test_per_class, dtype=int)
        avg_token_length = float(np.mean(lengths))
        np.savez_compressed(
            cache_path,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            avg_token_length=avg_token_length,
        )
        return {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "avg_token_length": avg_token_length,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_token_cloud_frames(
    *,
    model_key: str,
    dataset_names: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    feature_table_path = TOKEN_CLOUD_ROOTS[model_key]
    feature_df = pd.read_parquet(feature_table_path)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset_name in dataset_names:
        dataset_df = feature_df.loc[
            feature_df["dataset"].eq(dataset_name) & feature_df["feature_variant"].eq("single_layer")
        ].copy()
        if dataset_df.empty:
            raise ValueError(f"No token-cloud single-layer rows for {model_key} / {dataset_name}")
        feature_columns = _feature_subset_columns(dataset_df, INTRINSIC_SUBSET)
        if not feature_columns:
            raise ValueError(f"No intrinsic token-cloud columns for {model_key} / {dataset_name}")
        dataset_df = dataset_df.loc[:, ["example_id", "split", "label_ambiguous"] + feature_columns].copy()
        train_df = dataset_df.loc[dataset_df["split"].eq("train")].copy().reset_index(drop=True)
        test_df = dataset_df.loc[dataset_df["split"].eq("test")].copy().reset_index(drop=True)
        out[dataset_name] = {
            "train": train_df,
            "test": test_df,
            "feature_columns": feature_columns,
        }
    return out


def _subsample_classwise(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_per_class: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in [0, 1]:
        indices = np.where(labels == label)[0]
        if sample_per_class > len(indices):
            raise ValueError(f"Requested {sample_per_class} per class, only {len(indices)} available for label {label}")
        chosen = np.sort(rng.choice(indices, size=sample_per_class, replace=False))
        selected.append(chosen)
    subset_idx = np.concatenate(selected)
    return matrix[subset_idx], labels[subset_idx]


def _subsample_token_cloud_train(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    sample_per_class: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []
    for label in [0, 1]:
        label_df = train_df.loc[train_df["label_ambiguous"].eq(label)].copy()
        if sample_per_class > len(label_df):
            raise ValueError(f"Requested {sample_per_class} per class, only {len(label_df)} rows available for label {label}")
        selected_idx = rng.choice(label_df.index.to_numpy(), size=sample_per_class, replace=False)
        selected_parts.append(label_df.loc[np.sort(selected_idx)].copy())
    subset_df = pd.concat(selected_parts, ignore_index=True)
    return (
        subset_df.loc[:, feature_columns].to_numpy(dtype=float),
        subset_df["label_ambiguous"].to_numpy(dtype=int),
    )


def run_study(
    *,
    author_repo_root: Path,
    output_root: Path,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int,
    test_per_class: int,
    low_label_sizes: list[int],
    seed: int,
    batch_size: int,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        model_dir = ensure_dir(output_root / spec.output_dir_name)
        cache_dir = ensure_dir(model_dir / "cached_author_matrices")
        matrices = {
            dataset_name: _load_or_extract_matrix(
                cache_dir=cache_dir,
                author_repo_root=author_repo_root,
                model_key=model_key,
                dataset_name=dataset_name,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
                batch_size=batch_size,
            )
            for dataset_name in dataset_names
        }
        token_cloud_frames = _load_token_cloud_frames(model_key=model_key, dataset_names=dataset_names)

        for dataset_name in dataset_names:
            payload = matrices[dataset_name]
            feature_columns = token_cloud_frames[dataset_name]["feature_columns"]
            token_train = token_cloud_frames[dataset_name]["train"]
            token_test = token_cloud_frames[dataset_name]["test"]
            x_token_test = token_test.loc[:, feature_columns].to_numpy(dtype=float)
            y_token_test = token_test["label_ambiguous"].to_numpy(dtype=int)

            for sample_per_class in low_label_sizes:
                full_train_X, full_train_y = _subsample_classwise(
                    payload["train_X"], payload["train_y"], sample_per_class, seed=seed + sample_per_class
                )
                aen_train_X = full_train_X[:, spec.super_neurons]
                token_train_X, token_train_y = _subsample_token_cloud_train(
                    token_train, feature_columns, sample_per_class, seed=seed + sample_per_class
                )

                full_metrics = _evaluate_binary(
                    full_train_X,
                    full_train_y,
                    payload["test_X"],
                    payload["test_y"],
                    seed=seed,
                )
                aen_metrics = _evaluate_binary(
                    aen_train_X,
                    full_train_y,
                    payload["test_X"][:, spec.super_neurons],
                    payload["test_y"],
                    seed=seed,
                )
                topo_metrics = _evaluate_binary(
                    token_train_X,
                    token_train_y,
                    x_token_test,
                    y_token_test,
                    seed=seed,
                )
                rows.extend(
                    [
                        {
                            "model": spec.label,
                            "dataset": dataset_name,
                            "experiment": "low_label_in_domain",
                            "train_per_class": int(sample_per_class),
                            "method": "full_neurons",
                            "accuracy": full_metrics["accuracy"],
                            "auroc": full_metrics["auroc"],
                            "f1": full_metrics["f1"],
                        },
                        {
                            "model": spec.label,
                            "dataset": dataset_name,
                            "experiment": "low_label_in_domain",
                            "train_per_class": int(sample_per_class),
                            "method": "official_super_neurons",
                            "accuracy": aen_metrics["accuracy"],
                            "auroc": aen_metrics["auroc"],
                            "f1": aen_metrics["f1"],
                        },
                        {
                            "model": spec.label,
                            "dataset": dataset_name,
                            "experiment": "low_label_in_domain",
                            "train_per_class": int(sample_per_class),
                            "method": "token_cloud_intrinsic",
                            "accuracy": topo_metrics["accuracy"],
                            "auroc": topo_metrics["auroc"],
                            "f1": topo_metrics["f1"],
                        },
                    ]
                )

        for source_dataset in dataset_names:
            for target_dataset in dataset_names:
                if source_dataset == target_dataset:
                    continue
                source_payload = matrices[source_dataset]
                target_payload = matrices[target_dataset]
                source_token = token_cloud_frames[source_dataset]
                target_token = token_cloud_frames[target_dataset]
                common_columns = [col for col in source_token["feature_columns"] if col in set(target_token["feature_columns"])]
                full_metrics = _evaluate_binary(
                    source_payload["train_X"],
                    source_payload["train_y"],
                    target_payload["test_X"],
                    target_payload["test_y"],
                    seed=seed,
                )
                aen_metrics = _evaluate_binary(
                    source_payload["train_X"][:, spec.super_neurons],
                    source_payload["train_y"],
                    target_payload["test_X"][:, spec.super_neurons],
                    target_payload["test_y"],
                    seed=seed,
                )
                topo_metrics = _evaluate_binary(
                    source_token["train"].loc[:, common_columns].to_numpy(dtype=float),
                    source_token["train"]["label_ambiguous"].to_numpy(dtype=int),
                    target_token["test"].loc[:, common_columns].to_numpy(dtype=float),
                    target_token["test"]["label_ambiguous"].to_numpy(dtype=int),
                    seed=seed,
                )
                rows.extend(
                    [
                        {
                            "model": spec.label,
                            "dataset": f"{source_dataset}_to_{target_dataset}",
                            "experiment": "cross_dataset_transfer",
                            "train_per_class": int(train_per_class),
                            "method": "full_neurons",
                            "accuracy": full_metrics["accuracy"],
                            "auroc": full_metrics["auroc"],
                            "f1": full_metrics["f1"],
                        },
                        {
                            "model": spec.label,
                            "dataset": f"{source_dataset}_to_{target_dataset}",
                            "experiment": "cross_dataset_transfer",
                            "train_per_class": int(train_per_class),
                            "method": "official_super_neurons",
                            "accuracy": aen_metrics["accuracy"],
                            "auroc": aen_metrics["auroc"],
                            "f1": aen_metrics["f1"],
                        },
                        {
                            "model": spec.label,
                            "dataset": f"{source_dataset}_to_{target_dataset}",
                            "experiment": "cross_dataset_transfer",
                            "train_per_class": int(train_per_class),
                            "method": "token_cloud_intrinsic",
                            "accuracy": topo_metrics["accuracy"],
                            "auroc": topo_metrics["auroc"],
                            "f1": topo_metrics["f1"],
                        },
                    ]
                )

    result_df = pd.DataFrame(rows).sort_values(
        ["experiment", "model", "dataset", "train_per_class", "method"]
    ).reset_index(drop=True)
    result_path = output_root / "author_transfer_lowlabel_results.parquet"
    write_parquet(result_df, result_path)

    lines = [
        "# Author-Style Low-Label and Transfer Study",
        "",
        "Setting: official author prompt formatting, layer-14 hidden states, logistic regression for all methods.",
        f"- Train per class (full setting): `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        f"- Low-label sizes: `{low_label_sizes}`",
        f"- Token-cloud subset: `{INTRINSIC_SUBSET}`",
        "",
        "## Cross-Dataset Transfer",
        "",
        "| Model | Transfer | Method | AUROC | Acc |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    transfer_df = result_df.loc[result_df["experiment"].eq("cross_dataset_transfer")].copy()
    for row in transfer_df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['method']} | {row['auroc']:.4f} | {row['accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Low-Label In-Domain",
        "",
        "| Model | Dataset | Train/Class | Method | AUROC | Acc |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    low_df = result_df.loc[result_df["experiment"].eq("low_label_in_domain")].copy()
    for row in low_df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {int(row['train_per_class'])} | {row['method']} | {row['auroc']:.4f} | {row['accuracy']:.4f} |"
        )
    report_path = output_root / "author_transfer_lowlabel_report.md"
    write_markdown(report_path, "\n".join(lines) + "\n")

    metadata_path = output_root / "author_transfer_lowlabel_metadata.json"
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "author_repo_root": str(author_repo_root),
            "output_root": str(output_root),
            "model_keys": model_keys,
            "dataset_names": dataset_names,
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "low_label_sizes": [int(value) for value in low_label_sizes],
            "token_cloud_subset": INTRINSIC_SUBSET,
            "batch_size": int(batch_size),
            "results_path": str(result_path),
            "report_path": str(report_path),
        },
    )
    return {
        "results_path": str(result_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/author_transfer_lowlabel_study",
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()))
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--low-label-sizes", nargs="*", type=int, default=[50, 100, 200, 400, 800, 1000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
        low_label_sizes=[int(value) for value in args.low_label_sizes],
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )
