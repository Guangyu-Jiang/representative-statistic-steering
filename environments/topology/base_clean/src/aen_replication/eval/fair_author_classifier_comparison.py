"""Fair classifier comparison across token-cloud and neuron features.

This script keeps the author-style prompt construction and classwise split logic
fixed, then evaluates the same classifier families on:

- full hidden states at the author layer
- official super-neuron subsets from the EMNLP public repo
- class-prototype distance features in neuron space
- token-cloud topology features from the same author-style prompts

Classifier family selection is performed with an inner validation split, and the
decision threshold is tuned on the validation probabilities for accuracy.
"""

from __future__ import annotations

import argparse
import json
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
from aen_replication.train.independent_topology_classifier import (
    _fit_classifier,
    _group_train_val_split,
    _transform_with_scaler,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic": {
        "family": "logistic",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
    "rf200_leaf4": {
        "family": "random_forest",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "standardize": False,
    },
    "et200_leaf4": {
        "family": "extra_trees",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "standardize": False,
    },
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

INTRINSIC_TOPOLOGY_SUBSETS = ["no_distance", "descriptors_only"]
PROTOTYPE_AUGMENTED_TOPOLOGY_SUBSETS = [
    "all_topology",
    "no_pimg",
    "no_bottleneck",
    "core_no_pimg",
    "distance_only",
]


def _predict_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
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


def _evaluate_matrix(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    *,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    clf, scaler = _fit_classifier(train_X, train_y, config=classifier_config, seed=seed)
    val_matrix = _transform_with_scaler(val_X, scaler)
    test_matrix = _transform_with_scaler(test_X, scaler)
    val_probs = _predict_probabilities(clf, val_matrix)
    test_probs = _predict_probabilities(clf, test_matrix)
    threshold, val_metrics = _select_threshold(val_y, val_probs)
    test_metrics = binary_classification_metrics(test_y, test_probs, threshold=threshold)
    return {
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_auroc": float(roc_auc_score(test_y, test_probs)),
    }


def _split_matrix(
    matrix: np.ndarray,
    train_per_class: int,
    test_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return train_X, train_y, test_X, test_y


def _prototype_distance_features(
    train_X: np.ndarray,
    labels_train: np.ndarray,
    eval_X: np.ndarray,
) -> np.ndarray:
    clear_centroid = np.mean(train_X[labels_train == 0], axis=0)
    ambig_centroid = np.mean(train_X[labels_train == 1], axis=0)

    def _l2(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.linalg.norm(matrix - center[None, :], axis=1)

    def _cosine_distance(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
        center_norm = float(np.linalg.norm(center))
        matrix_norm = np.linalg.norm(matrix, axis=1)
        denom = np.maximum(matrix_norm * max(center_norm, 1e-12), 1e-12)
        cosine = np.sum(matrix * center[None, :], axis=1) / denom
        return 1.0 - cosine

    l2_clear = _l2(eval_X, clear_centroid)
    l2_ambig = _l2(eval_X, ambig_centroid)
    cos_clear = _cosine_distance(eval_X, clear_centroid)
    cos_ambig = _cosine_distance(eval_X, ambig_centroid)
    return np.column_stack(
        [
            l2_clear,
            l2_ambig,
            cos_clear,
            cos_ambig,
            l2_clear - l2_ambig,
            cos_clear - cos_ambig,
        ]
    ).astype(float, copy=False)


def _random_train_val_split(n_train: int, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_train)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_train * val_fraction)))
    val_idx = np.sort(indices[:val_size])
    inner_train_idx = np.sort(indices[val_size:])
    return inner_train_idx, val_idx


def _evaluate_neuron_feature_sets(
    *,
    matrix: np.ndarray,
    super_neurons: list[int],
    train_per_class: int,
    test_per_class: int,
    seed: int,
    val_fraction: float,
) -> list[dict[str, Any]]:
    train_X, train_y, test_X, test_y = _split_matrix(matrix, train_per_class=train_per_class, test_per_class=test_per_class)
    n_train = train_X.shape[0]
    inner_train_idx, val_idx = _random_train_val_split(n_train, seed=seed, val_fraction=val_fraction)

    raw_feature_sets = {
        "full_neurons": (train_X, test_X),
        "official_super_neurons": (train_X[:, super_neurons], test_X[:, super_neurons]),
    }
    rows: list[dict[str, Any]] = []
    for feature_set_name, (full_train, full_test) in raw_feature_sets.items():
        inner_train_X = full_train[inner_train_idx]
        inner_train_y = train_y[inner_train_idx]
        val_X = full_train[val_idx]
        val_y = train_y[val_idx]
        for family_name, family_config in FAMILY_CONFIGS.items():
            payload = _evaluate_matrix(
                inner_train_X,
                inner_train_y,
                val_X,
                val_y,
                full_test,
                test_y,
                classifier_config=family_config,
                seed=seed,
            )
            rows.append(
                {
                    "feature_source": feature_set_name,
                    "feature_variant": "single_layer",
                    "feature_subset": "raw",
                    "feature_count": int(full_train.shape[1]),
                    "config_name": family_name,
                    "threshold": float(payload["threshold"]),
                    "val_auroc": float(payload["val_metrics"]["auroc"]),
                    "val_accuracy": float(payload["val_metrics"]["accuracy"]),
                    "test_auroc": float(payload["test_auroc"]),
                    "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                    "test_f1": float(payload["test_metrics"]["f1"]),
                }
            )
        prototype_train = _prototype_distance_features(inner_train_X, inner_train_y, inner_train_X)
        prototype_val = _prototype_distance_features(inner_train_X, inner_train_y, val_X)
        prototype_test = _prototype_distance_features(inner_train_X, inner_train_y, full_test)
        for family_name, family_config in FAMILY_CONFIGS.items():
            payload = _evaluate_matrix(
                prototype_train,
                inner_train_y,
                prototype_val,
                val_y,
                prototype_test,
                test_y,
                classifier_config=family_config,
                seed=seed,
            )
            rows.append(
                {
                    "feature_source": f"{feature_set_name}_prototype_distances",
                    "feature_variant": "single_layer",
                    "feature_subset": "centroid_distance",
                    "feature_count": int(prototype_train.shape[1]),
                    "config_name": family_name,
                    "threshold": float(payload["threshold"]),
                    "val_auroc": float(payload["val_metrics"]["auroc"]),
                    "val_accuracy": float(payload["val_metrics"]["accuracy"]),
                    "test_auroc": float(payload["test_auroc"]),
                    "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                    "test_f1": float(payload["test_metrics"]["f1"]),
                }
            )
    return rows


def _evaluate_token_cloud_feature_table(
    *,
    feature_table_path: Path,
    dataset_name: str,
    seed: int,
    val_fraction: float,
) -> list[dict[str, Any]]:
    frame = pd.read_parquet(feature_table_path)
    subset = frame.loc[frame["dataset"].eq(dataset_name)].copy()
    if subset.empty:
        return []
    rows: list[dict[str, Any]] = []
    for feature_variant in sorted(subset["feature_variant"].dropna().unique()):
        variant_df = subset.loc[subset["feature_variant"].eq(feature_variant)].copy()
        train_df = variant_df.loc[variant_df["split"].eq("train")].copy()
        test_df = variant_df.loc[variant_df["split"].eq("test")].copy()
        if train_df.empty or test_df.empty:
            continue
        if "pair_id" not in train_df.columns or train_df["pair_id"].isna().all():
            train_df["pair_id"] = train_df["example_id"].astype(str)
        inner_train_ids, val_ids = _group_train_val_split(train_df, val_fraction=val_fraction, seed=seed)
        inner_train = train_df.loc[train_df["example_id"].astype(str).isin(inner_train_ids)].copy()
        val_df = train_df.loc[train_df["example_id"].astype(str).isin(val_ids)].copy()
        if inner_train.empty or val_df.empty:
            continue

        subset_groups = [
            ("token_cloud_intrinsic", INTRINSIC_TOPOLOGY_SUBSETS),
            ("token_cloud_prototype_augmented", PROTOTYPE_AUGMENTED_TOPOLOGY_SUBSETS),
        ]
        for feature_source, subset_names in subset_groups:
            for subset_name in subset_names:
                feature_columns = _feature_subset_columns(variant_df, subset_name)
                if not feature_columns:
                    continue
                x_train = inner_train.loc[:, feature_columns].to_numpy(dtype=float)
                y_train = inner_train["label_ambiguous"].to_numpy(dtype=int)
                x_val = val_df.loc[:, feature_columns].to_numpy(dtype=float)
                y_val = val_df["label_ambiguous"].to_numpy(dtype=int)
                x_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
                y_test = test_df["label_ambiguous"].to_numpy(dtype=int)
                for family_name, family_config in FAMILY_CONFIGS.items():
                    payload = _evaluate_matrix(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        x_test,
                        y_test,
                        classifier_config=family_config,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "feature_source": feature_source,
                            "feature_variant": str(feature_variant),
                            "feature_subset": str(subset_name),
                            "feature_count": int(len(feature_columns)),
                            "config_name": family_name,
                            "threshold": float(payload["threshold"]),
                            "val_auroc": float(payload["val_metrics"]["auroc"]),
                            "val_accuracy": float(payload["val_metrics"]["accuracy"]),
                            "test_auroc": float(payload["test_auroc"]),
                            "test_accuracy": float(payload["test_metrics"]["accuracy"]),
                            "test_f1": float(payload["test_metrics"]["f1"]),
                        }
                    )
    return rows


def run_fair_comparison(
    *,
    author_repo_root: Path,
    output_root: Path,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int,
    test_per_class: int,
    val_fraction: float,
    seed: int,
    batch_size: int,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    run_outputs: dict[str, str] = {}

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        feature_table_path = TOKEN_CLOUD_ROOTS[model_key]
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

        model_output_dir = ensure_dir(output_root / spec.output_dir_name)
        for dataset_name in dataset_names:
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
            dataset_rows = _evaluate_neuron_feature_sets(
                matrix=matrix,
                super_neurons=spec.super_neurons,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
                seed=seed,
                val_fraction=val_fraction,
            )
            dataset_rows.extend(
                _evaluate_token_cloud_feature_table(
                    feature_table_path=feature_table_path,
                    dataset_name=dataset_name,
                    seed=seed,
                    val_fraction=val_fraction,
                )
            )
            for row in dataset_rows:
                row.update(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "avg_prompt_token_length": float(np.mean(lengths)),
                    }
                )
                candidate_rows.append(row)

            dataset_df = pd.DataFrame([row for row in candidate_rows if row["model"] == spec.label and row["dataset"] == dataset_name])
            for feature_source in [
                "full_neurons",
                "official_super_neurons",
                "full_neurons_prototype_distances",
                "official_super_neurons_prototype_distances",
                "token_cloud_intrinsic",
                "token_cloud_prototype_augmented",
            ]:
                source_df = dataset_df.loc[dataset_df["feature_source"].eq(feature_source)].copy()
                if source_df.empty:
                    continue
                source_df = source_df.sort_values(
                    ["val_auroc", "val_accuracy", "test_auroc", "test_accuracy", "config_name"],
                    ascending=[False, False, False, False, True],
                ).reset_index(drop=True)
                best_row = source_df.iloc[0].to_dict()
                best_rows.append(best_row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_outputs[spec.output_dir_name] = str(model_output_dir)

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["model", "dataset", "feature_source", "val_auroc", "test_auroc"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    best_df = pd.DataFrame(best_rows).sort_values(["model", "dataset", "feature_source"]).reset_index(drop=True)

    candidate_path = output_root / "fair_author_classifier_candidates.parquet"
    best_path = output_root / "fair_author_classifier_best.parquet"
    write_parquet(candidate_df, candidate_path)
    write_parquet(best_df, best_path)

    lines = [
        "# Fair Author-Style Classifier Comparison",
        "",
        "This report keeps the author-style prompts and split counts fixed, then evaluates the same",
        "classifier family grid on raw neurons, neuron-prototype features, and token-cloud topology",
        "features. Each feature source selects its best config by inner validation AUROC.",
        "",
        f"- Train per class: `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        f"- Validation fraction: `{val_fraction}`",
        f"- Classifier families: `{', '.join(FAMILY_CONFIGS.keys())}`",
        "",
        "| Model | Dataset | Feature Source | Best Config | Variant | Subset | AUROC | Acc |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in best_df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['feature_source']} | {row['config_name']} | "
            f"{row['feature_variant']} | {row['feature_subset']} | {row['test_auroc']:.4f} | {row['test_accuracy']:.4f} |"
        )
    report_path = output_root / "fair_author_classifier_comparison.md"
    write_markdown(report_path, "\n".join(lines) + "\n")

    metadata_path = output_root / "fair_author_classifier_comparison_metadata.json"
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
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "candidate_parquet": str(candidate_path),
            "best_parquet": str(best_path),
            "report_path": str(report_path),
        },
    )
    return {
        "candidate_path": str(candidate_path),
        "best_path": str(best_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/fair_author_classifier_comparison",
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()))
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fair_comparison(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )
