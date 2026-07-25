#!/usr/bin/env python3
"""Classify FalseQA false premises from all-layer prompt-token H0 topology.

The primary evaluations are (1) ordinary false-vs-corrected classification
and (2) paired orientation classification using the signed difference between
the two members of each FalseQA pair. PCA and feature scaling are fit only on
training questions. Full H0 lifetime vectors are cached for later ablations.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.eval.falseqa_topology import (
    assert_group_disjoint,
    assign_evaluation_splits,
    h0_features_from_cloud,
    load_falseqa_pairs,
    paired_index_plan,
)
from aen_replication.models.generation import render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.token_cloud_topology_classifier import _valid_token_mask
from aen_replication.utils.io_utils import ensure_dir, slugify, utc_now_iso, write_json, write_markdown, write_parquet
from aen_replication.utils.seed import set_global_seed


H0_FEATURES = (
    "h0_mean_persistence",
    "h0_persistence_entropy",
    "h0_top5_persistence_fraction",
)
PROTOCOLS = {
    "random80": "split_random80",
    "official": "split_official",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Model run config.")
    parser.add_argument("--falseqa-root", default="datasets/FalseQA/dataset")
    parser.add_argument("--output-root", default="artifacts/falseqa_topology_classification")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--pca-fit-token-cap", type=int, default=8000)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=-1, help="Inclusive; -1 uses every decoder layer.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Development-only pair cap; 0 uses all pairs.")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    return parser.parse_args()


def _selected_layers(total_layers: int, start: int, end: int) -> list[int]:
    upper = total_layers - 1 if end < 0 else min(end, total_layers - 1)
    layers = list(range(max(0, start), upper + 1))
    if not layers:
        raise ValueError(f"No layers selected from total={total_layers}, start={start}, end={end}")
    return layers


def _iter_batches(frame: pd.DataFrame, batch_size: int):
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def _render_questions(
    bundle: HFModelBundle,
    config: dict[str, Any],
    questions: list[str],
    *,
    use_chat_template: bool,
) -> list[str]:
    if not use_chat_template:
        return questions
    generation = dict(config.get("generation", {}))
    return render_prompts(
        bundle=bundle,
        prompt_texts=questions,
        use_chat_template=True,
        system_prompt=generation.get("system_prompt"),
        add_generation_prompt=True,
    )


def _forward_hidden_states(bundle: HFModelBundle, model_inputs: dict[str, torch.Tensor]) -> Any:
    """Avoid materializing vocabulary logits when the architecture exposes its base model."""

    base_model = getattr(bundle.model, "model", None)
    if base_model is not None:
        return base_model(**model_inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    return bundle.model(**model_inputs, output_hidden_states=True, use_cache=False, return_dict=True)


def _batch_masks(bundle: HFModelBundle, encoded: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    special_ids = {
        int(token_id)
        for token_id in getattr(bundle.tokenizer, "all_special_ids", [])
        if token_id is not None
    }
    input_ids = encoded["input_ids"].detach().cpu()
    attention_mask = encoded["attention_mask"].detach().cpu()
    return [
        _valid_token_mask(
            input_ids[row],
            attention_mask[row],
            special_ids=special_ids,
            drop_special_tokens=True,
        )
        for row in range(len(input_ids))
    ]


def _fit_pca_reducers(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    layers: list[int],
    batch_size: int,
    max_length: int,
    token_cap: int,
    pca_components: int,
    seed: int,
    use_chat_template: bool,
) -> dict[int, PCA]:
    """Fit one PCA per layer on a shuffled, training-only token sample."""

    shuffled = train_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    token_chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    counts = {layer: 0 for layer in layers}
    tokenizer = bundle.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for batch in tqdm(_iter_batches(shuffled, batch_size), desc="falseqa_pca_tokens", leave=False):
            prompts = _render_questions(
                bundle,
                config,
                batch["question"].astype(str).tolist(),
                use_chat_template=use_chat_template,
            )
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            masks = _batch_masks(bundle, encoded)
            model_inputs = {key: value.to(bundle.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = _forward_hidden_states(bundle, model_inputs)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            for layer in layers:
                needed = token_cap - counts[layer]
                if needed <= 0:
                    continue
                layer_output = hidden_states[layer + 1].detach().to(dtype=torch.float16).cpu()
                chunks = [layer_output[row][masks[row]].numpy() for row in range(len(masks))]
                chunks = [chunk for chunk in chunks if len(chunk)]
                if chunks:
                    selected = np.vstack(chunks)[:needed]
                    token_chunks[layer].append(selected)
                    counts[layer] += len(selected)
            del outputs, hidden_states, model_inputs, encoded
            if bundle.device.type == "cuda":
                torch.cuda.empty_cache()
            if all(counts[layer] >= token_cap for layer in layers):
                break
    finally:
        tokenizer.padding_side = original_padding_side

    reducers: dict[int, PCA] = {}
    for layer in tqdm(layers, desc="falseqa_pca_fit", leave=False):
        if not token_chunks[layer]:
            raise ValueError(f"No PCA tokens collected at layer {layer}")
        matrix = np.vstack(token_chunks[layer]).astype(np.float32, copy=False)
        components = min(int(pca_components), matrix.shape[1], max(1, len(matrix) - 1))
        reducer = PCA(n_components=components, svd_solver="randomized", random_state=seed + layer)
        reducer.fit(matrix)
        reducers[layer] = reducer
    return reducers


def _extract_features(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    frame: pd.DataFrame,
    layers: list[int],
    reducers: dict[int, PCA],
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
) -> pd.DataFrame:
    tokenizer = bundle.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    rows: list[dict[str, Any]] = []
    try:
        for batch in tqdm(_iter_batches(frame, batch_size), total=(len(frame) + batch_size - 1) // batch_size, desc="falseqa_h0"):
            prompts = _render_questions(
                bundle,
                config,
                batch["question"].astype(str).tolist(),
                use_chat_template=use_chat_template,
            )
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            masks = _batch_masks(bundle, encoded)
            model_inputs = {key: value.to(bundle.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = _forward_hidden_states(bundle, model_inputs)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            records = batch.reset_index(drop=True).to_dict(orient="records")
            token_counts = [int(mask.sum().item()) for mask in masks]
            for layer in layers:
                layer_output = hidden_states[layer + 1].detach().float().cpu()
                chunks = [layer_output[row][masks[row]].numpy() for row in range(len(masks))]
                nonempty = [chunk for chunk in chunks if len(chunk)]
                if not nonempty:
                    continue
                reduced_matrix = reducers[layer].transform(np.vstack(nonempty)).astype(np.float32, copy=False)
                offset = 0
                for record, token_count in zip(records, token_counts, strict=True):
                    cloud = reduced_matrix[offset : offset + token_count]
                    offset += token_count
                    feature_row = {
                        "example_id": str(record["example_id"]),
                        "pair_id": str(record["pair_id"]),
                        "source_split": str(record["source_split"]),
                        "variant": str(record["variant"]),
                        "split_random80": str(record["split_random80"]),
                        "split_official": str(record["split_official"]),
                        "pca_fit": bool(record["pca_fit"]),
                        "label_false_premise": int(record["label_false_premise"]),
                        "layer": int(layer),
                        "token_count": int(token_count),
                        "word_count": int(len(str(record["question"]).split())),
                        "char_count": int(len(str(record["question"]))),
                    }
                    feature_row.update(h0_features_from_cloud(cloud))
                    rows.append(feature_row)
            del outputs, hidden_states, model_inputs, encoded
            if bundle.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return pd.DataFrame(rows)


def _build_feature_sets(
    metadata: pd.DataFrame,
    feature_frame: pd.DataFrame,
    layers: list[int],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    order = metadata["example_id"].astype(str).tolist()
    feature_sets: dict[str, tuple[np.ndarray, list[str]]] = {}
    stat_matrices: dict[str, np.ndarray] = {}
    stat_names: dict[str, list[str]] = {}
    for statistic in H0_FEATURES:
        pivot = feature_frame.pivot(index="example_id", columns="layer", values=statistic)
        pivot = pivot.reindex(index=order, columns=layers)
        if pivot.isna().any().any():
            raise ValueError(f"Missing {statistic} values after pivot")
        stat_matrices[statistic] = pivot.to_numpy(dtype=float)
        stat_names[statistic] = [f"{statistic}__l{layer:02d}" for layer in layers]

    count_rows = feature_frame.loc[feature_frame["layer"].eq(layers[0])].set_index("example_id").reindex(order)
    length_matrix = count_rows.loc[:, ["token_count", "word_count", "char_count"]].to_numpy(dtype=float)
    length_names = ["token_count", "word_count", "char_count"]
    mean_matrix = stat_matrices["h0_mean_persistence"]
    mean_names = stat_names["h0_mean_persistence"]
    topology3_matrix = np.hstack([stat_matrices[name] for name in H0_FEATURES])
    topology3_names = [feature for name in H0_FEATURES for feature in stat_names[name]]

    feature_sets["length_controls"] = (length_matrix, length_names)
    feature_sets["h0_mean_all_layers"] = (mean_matrix, mean_names)
    feature_sets["h0_three_all_layers"] = (topology3_matrix, topology3_names)
    feature_sets["h0_mean_all_layers_plus_length"] = (
        np.hstack([mean_matrix, length_matrix]),
        mean_names + length_names,
    )
    feature_sets["h0_three_all_layers_plus_length"] = (
        np.hstack([topology3_matrix, length_matrix]),
        topology3_names + length_names,
    )
    last = len(layers) - 1
    feature_sets["h0_mean_last_layer"] = (
        mean_matrix[:, [last]],
        [mean_names[last]],
    )
    last_three_indices = [last, len(layers) + last, 2 * len(layers) + last]
    feature_sets["h0_three_last_layer"] = (
        topology3_matrix[:, last_three_indices],
        [topology3_names[index] for index in last_three_indices],
    )
    for layer_index, layer in enumerate(layers):
        feature_sets[f"h0_mean_layer_{layer:02d}"] = (
            mean_matrix[:, [layer_index]],
            [mean_names[layer_index]],
        )
        indices = [layer_index, len(layers) + layer_index, 2 * len(layers) + layer_index]
        feature_sets[f"h0_three_layer_{layer:02d}"] = (
            topology3_matrix[:, indices],
            [topology3_names[index] for index in indices],
        )
    return feature_sets


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.0).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _fit_logistic_dense(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    test_matrix: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    test_scaled = scaler.transform(test_matrix)
    classifier = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        classifier.fit(train_scaled, train_labels)
    return (
        classifier.decision_function(train_scaled),
        classifier.decision_function(test_scaled),
        classifier,
        scaler,
    )


def _evaluate_dense_set(
    *,
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    task: str,
    protocol: str,
    split_column: str,
    seed: int,
    pair_plan: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if task == "ordinary":
        task_metadata = metadata.reset_index(drop=True).copy()
        task_matrix = np.asarray(matrix, dtype=float)
        label_column = "label_false_premise"
    elif task == "paired_orientation":
        task_metadata, first_indices, second_indices = pair_plan
        task_matrix = np.asarray(matrix, dtype=float)[first_indices] - np.asarray(matrix, dtype=float)[second_indices]
        label_column = "label_false_first"
    else:
        raise ValueError(f"Unknown task: {task}")

    train_mask = task_metadata[split_column].eq("train").to_numpy()
    test_mask = task_metadata[split_column].eq("test").to_numpy()
    train_labels = task_metadata.loc[train_mask, label_column].to_numpy(dtype=int)
    test_labels = task_metadata.loc[test_mask, label_column].to_numpy(dtype=int)
    train_scores, test_scores, classifier, _scaler = _fit_logistic_dense(
        task_matrix[train_mask],
        train_labels,
        task_matrix[test_mask],
        seed=seed,
    )
    train_metrics = _classification_metrics(train_labels, train_scores)
    test_metrics = _classification_metrics(test_labels, test_scores)
    row: dict[str, Any] = {
        "protocol": protocol,
        "task": task,
        "feature_set": feature_set,
        "feature_dim": int(task_matrix.shape[1]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
        "feature_names": ",".join(feature_names),
    }
    predictions = task_metadata.loc[test_mask].copy()
    predictions["protocol"] = protocol
    predictions["task"] = task
    predictions["feature_set"] = feature_set
    predictions["label"] = test_labels
    predictions["score"] = test_scores
    predictions["probability"] = 1.0 / (1.0 + np.exp(-np.clip(test_scores, -40.0, 40.0)))
    predictions["prediction"] = (test_scores >= 0.0).astype(int)
    predictions["correct"] = predictions["prediction"].to_numpy() == test_labels
    return row, predictions


def _evaluate_tfidf(
    metadata: pd.DataFrame,
    *,
    protocol: str,
    split_column: str,
    task: str,
    seed: int,
    pair_plan: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_examples = metadata.loc[metadata[split_column].eq("train")]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    vectorizer.fit(train_examples["question"].astype(str))
    question_matrix = vectorizer.transform(metadata["question"].astype(str))
    if task == "ordinary":
        task_metadata = metadata.reset_index(drop=True).copy()
        task_matrix = question_matrix
        label_column = "label_false_premise"
    else:
        task_metadata, first_indices, second_indices = pair_plan
        task_matrix = (question_matrix[first_indices] - question_matrix[second_indices]).tocsr()
        label_column = "label_false_first"
    train_mask = task_metadata[split_column].eq("train").to_numpy()
    test_mask = task_metadata[split_column].eq("test").to_numpy()
    train_labels = task_metadata.loc[train_mask, label_column].to_numpy(dtype=int)
    test_labels = task_metadata.loc[test_mask, label_column].to_numpy(dtype=int)
    classifier = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
    )
    classifier.fit(task_matrix[train_mask], train_labels)
    train_scores = classifier.decision_function(task_matrix[train_mask])
    test_scores = classifier.decision_function(task_matrix[test_mask])
    row: dict[str, Any] = {
        "protocol": protocol,
        "task": task,
        "feature_set": "tfidf_word_unigram_bigram",
        "feature_dim": int(task_matrix.shape[1]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        **{f"train_{key}": value for key, value in _classification_metrics(train_labels, train_scores).items()},
        **{f"test_{key}": value for key, value in _classification_metrics(test_labels, test_scores).items()},
        "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
        "feature_names": "tfidf vocabulary fit on training questions only",
    }
    predictions = task_metadata.loc[test_mask].copy()
    predictions["protocol"] = protocol
    predictions["task"] = task
    predictions["feature_set"] = "tfidf_word_unigram_bigram"
    predictions["label"] = test_labels
    predictions["score"] = test_scores
    predictions["probability"] = 1.0 / (1.0 + np.exp(-np.clip(test_scores, -40.0, 40.0)))
    predictions["prediction"] = (test_scores >= 0.0).astype(int)
    predictions["correct"] = predictions["prediction"].to_numpy() == test_labels
    return row, predictions


def _summary_markdown(metrics: pd.DataFrame, *, model_name: str, metadata: pd.DataFrame) -> str:
    primary_names = {
        "length_controls",
        "h0_mean_all_layers",
        "h0_three_all_layers",
        "h0_mean_all_layers_plus_length",
        "h0_three_all_layers_plus_length",
        "h0_mean_last_layer",
        "h0_three_last_layer",
        "tfidf_word_unigram_bigram",
    }
    primary = metrics.loc[metrics["feature_set"].isin(primary_names)].copy()
    columns = [
        "protocol",
        "task",
        "feature_set",
        "feature_dim",
        "n_train",
        "n_test",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_auroc",
    ]
    table = primary.loc[:, columns].sort_values(["protocol", "task", "feature_set"]).to_markdown(index=False)
    return (
        "# FalseQA H0 Topology Classification\n\n"
        f"- Model: `{model_name}`\n"
        f"- Questions: {len(metadata):,} from {metadata['pair_id'].nunique():,} false/corrected pairs\n"
        "- Primary statistic sets: all-layer H0 mean persistence and all-layer three-feature H0 vectors\n"
        "- Classifier: train-standardized, class-balanced L2 logistic regression\n"
        "- Paired task: signed feature difference after deterministic random pair orientation\n"
        "- PCA fit set: intersection of random-80 training and official training pairs\n\n"
        "## Primary results\n\n"
        f"{table}\n\n"
        "Single-layer rows are retained in `classification_metrics.parquet` as diagnostic ablations; "
        "the pre-specified all-layer rows should be used for primary claims.\n"
    )


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    seed = int(config["seed"] if args.seed is None else args.seed)
    set_global_seed(seed)
    model_name = str(config["model"]["name"])
    output_dir = ensure_dir(Path(args.output_root).resolve() / slugify(model_name))
    feature_path = output_dir / "falseqa_h0_all_layers.parquet"
    reducer_path = output_dir / "falseqa_pca16_reducers.joblib"
    data_path = output_dir / "falseqa_paired_questions.parquet"

    metadata = load_falseqa_pairs(args.falseqa_root)
    metadata = assign_evaluation_splits(
        metadata,
        train_fraction=float(args.train_fraction),
        seed=seed,
        max_pairs=int(args.max_pairs),
    )
    for split_column in PROTOCOLS.values():
        assert_group_disjoint(metadata, split_column)
    write_parquet(metadata, data_path)

    if feature_path.exists() and reducer_path.exists() and not args.force_features:
        print(f"loading cached features: {feature_path}", flush=True)
        feature_frame = pd.read_parquet(feature_path)
        reducers = joblib.load(reducer_path)
        layers = sorted(int(layer) for layer in reducers)
    else:
        bundle = load_hf_model(config["model"], config.get("token_cloud_topology_classifier", {}))
        try:
            total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
            layers = _selected_layers(total_layers, int(args.layer_start), int(args.layer_end))
            pca_train = metadata.loc[metadata["pca_fit"]].reset_index(drop=True)
            if pca_train["pair_id"].nunique() < 2:
                raise ValueError("PCA training intersection has fewer than two pairs.")
            reducers = _fit_pca_reducers(
                bundle=bundle,
                config=config,
                train_frame=pca_train,
                layers=layers,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                token_cap=int(args.pca_fit_token_cap),
                pca_components=int(args.pca_components),
                seed=seed,
                use_chat_template=bool(args.use_chat_template),
            )
            joblib.dump(reducers, reducer_path, compress=3)
            feature_frame = _extract_features(
                bundle=bundle,
                config=config,
                frame=metadata,
                layers=layers,
                reducers=reducers,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                use_chat_template=bool(args.use_chat_template),
            )
            write_parquet(feature_frame, feature_path)
        finally:
            del bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_rows = len(metadata) * len(layers)
    if len(feature_frame) != expected_rows:
        raise ValueError(f"Feature cache has {len(feature_frame)} rows; expected {expected_rows}.")
    if set(feature_frame["example_id"].astype(str)) != set(metadata["example_id"].astype(str)):
        raise ValueError("Feature cache example IDs do not match the current FalseQA frame.")
    feature_sets = _build_feature_sets(metadata, feature_frame, layers)
    pair_plan = paired_index_plan(metadata, seed=seed)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for protocol, split_column in PROTOCOLS.items():
        for task in ("ordinary", "paired_orientation"):
            for feature_set, (matrix, feature_names) in tqdm(
                feature_sets.items(),
                desc=f"classify_{protocol}_{task}",
                leave=False,
            ):
                metric_row, predictions = _evaluate_dense_set(
                    metadata=metadata,
                    matrix=matrix,
                    feature_names=feature_names,
                    feature_set=feature_set,
                    task=task,
                    protocol=protocol,
                    split_column=split_column,
                    seed=seed,
                    pair_plan=pair_plan,
                )
                metric_rows.append(metric_row)
                prediction_frames.append(predictions)
            if not args.skip_tfidf:
                metric_row, predictions = _evaluate_tfidf(
                    metadata,
                    protocol=protocol,
                    split_column=split_column,
                    task=task,
                    seed=seed,
                    pair_plan=pair_plan,
                )
                metric_rows.append(metric_row)
                prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    write_parquet(metrics, output_dir / "classification_metrics.parquet")
    metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
    write_parquet(predictions, output_dir / "test_predictions.parquet")
    write_markdown(
        output_dir / "summary.md",
        _summary_markdown(metrics, model_name=model_name, metadata=metadata),
    )
    metadata_payload = {
        "created_at": utc_now_iso(),
        "model": model_name,
        "falseqa_root": str(Path(args.falseqa_root).resolve()),
        "question_count": int(len(metadata)),
        "pair_count": int(metadata["pair_id"].nunique()),
        "layers": layers,
        "pca_components": int(args.pca_components),
        "pca_fit_token_cap": int(args.pca_fit_token_cap),
        "pca_fit_question_count": int(metadata["pca_fit"].sum()),
        "train_fraction": float(args.train_fraction),
        "seed": seed,
        "prompt_mode": "chat_generation" if args.use_chat_template else "raw_question",
        "max_length": int(args.max_length),
        "feature_cache": str(feature_path),
        "full_h0_lifetimes_cached": True,
    }
    write_json(output_dir / "metadata.json", metadata_payload)
    primary = metrics.loc[
        metrics["feature_set"].isin(["h0_mean_all_layers", "h0_three_all_layers", "tfidf_word_unigram_bigram"]),
        ["protocol", "task", "feature_set", "n_test", "test_accuracy", "test_macro_f1", "test_auroc"],
    ].sort_values(["protocol", "task", "feature_set"])
    print(primary.to_string(index=False), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
