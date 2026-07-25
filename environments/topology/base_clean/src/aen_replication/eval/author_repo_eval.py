"""Run the first author's public detection code path against cached exact models.

This audit script reproduces the effective logic in:
  - Internal_State_Detect_Ambiguity/extract_first_hidden_state.py
  - Internal_State_Detect_Ambiguity/single_layer_classifier.py

Differences from the upstream repo:
  - model weights are loaded from locally cached Hugging Face snapshots
  - outputs are written into this repo's artifact tree
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.paper_audit import PAPER_TABLE_1
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


@dataclass(frozen=True)
class AuthorEvalModelSpec:
    label: str
    load_repo: str
    prompt_model_name: str
    output_dir_name: str
    super_neurons: list[int]


MODEL_SPECS: dict[str, AuthorEvalModelSpec] = {
    "llama31_8b_instruct": AuthorEvalModelSpec(
        label="LLaMA 3.1 8B",
        load_repo="meta-llama/Llama-3.1-8B-Instruct",
        prompt_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_dir_name="meta_llama_llama_3_1_8b_instruct",
        super_neurons=[788, 4062, 1384],
    ),
    "mistral_7b_instruct_v03": AuthorEvalModelSpec(
        label="Mistral 7B",
        load_repo="mistralai/Mistral-7B-Instruct-v0.3",
        prompt_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        output_dir_name="mistralai_mistral_7b_instruct_v0_3",
        super_neurons=[2070],
    ),
    "gemma_7b_it": AuthorEvalModelSpec(
        label="Gemma 7B",
        load_repo="google/gemma-7b-it",
        prompt_model_name="google/gemma-7b-it",
        output_dir_name="google_gemma_7b_it",
        super_neurons=[1995],
    ),
}


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "ambigqa": {
        "paper_key": "ambigqa",
        "author_dir": "ambig_qa",
        "ambig_file": "ambig_questions.json",
        "clear_file": "clean_questions.json",
    },
    "situatedqa": {
        "paper_key": "situatedqa",
        "author_dir": "situated",
        "ambig_file": "ambi_combined_question.json",
        "clear_file": "clean_combined_question.json",
    },
}


def _format_prompt(user_input: str, model_name: str, system_message: str = "You are a helpful assistant.") -> str:
    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 23 July 2024\n\n"
            f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ).strip()
    if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        return f"<s>[INST] {system_message}\n            {user_input}[/INST]".strip()
    if model_name == "google/gemma-7b-it":
        return user_input.strip()
    raise ValueError(f"Unsupported author-style prompt model: {model_name}")


def _load_author_prompts(*, author_repo_root: Path, dataset_name: str, prompt_model_name: str) -> tuple[list[str], list[str]]:
    spec = DATASET_SPECS[dataset_name]
    ambig_path = author_repo_root / spec["author_dir"] / spec["ambig_file"]
    clear_path = author_repo_root / spec["author_dir"] / spec["clear_file"]
    with ambig_path.open("r", encoding="utf-8") as handle:
        ambig_payload = json.load(handle)
    with clear_path.open("r", encoding="utf-8") as handle:
        clear_payload = json.load(handle)
    ambig_prompts = [_format_prompt(str(item["prompt"]), prompt_model_name) for item in ambig_payload]
    clear_prompts = [_format_prompt(str(item["prompt"]), prompt_model_name) for item in clear_payload]
    return ambig_prompts, clear_prompts


def _shuffle_and_slice_author_style(
    ambig_prompts: list[str],
    clear_prompts: list[str],
    *,
    train_per_class: int = 400,
    test_per_class: int = 1000,
) -> dict[str, list[str]]:
    train_ambig = list(ambig_prompts)
    train_clear = list(clear_prompts)
    test_ambig = list(ambig_prompts)
    test_clear = list(clear_prompts)
    if train_per_class <= 0 or test_per_class <= 0:
        raise ValueError("train_per_class and test_per_class must be positive integers.")
    if train_per_class > len(train_ambig) or train_per_class > len(train_clear):
        raise ValueError(
            f"Requested train_per_class={train_per_class} exceeds available prompts "
            f"({len(train_ambig)} ambiguous, {len(train_clear)} clear)."
        )
    if test_per_class > len(test_ambig) or test_per_class > len(test_clear):
        raise ValueError(
            f"Requested test_per_class={test_per_class} exceeds available prompts "
            f"({len(test_ambig)} ambiguous, {len(test_clear)} clear)."
        )
    for seed, values in [(11, train_ambig), (12, train_clear), (13, test_ambig), (14, test_clear)]:
        np.random.seed(seed)
        np.random.shuffle(values)
    return {
        "train_ambig": train_ambig[:train_per_class],
        "train_clear": train_clear[:train_per_class],
        "test_ambig": test_ambig[-test_per_class:],
        "test_clear": test_clear[-test_per_class:],
    }


def _extract_author_style_hidden_states(
    *,
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_index: int = 14,
    batch_size: int = 20,
) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        attention_mask = encoded["attention_mask"].to(model.device)
        model_inputs = {key: value.to(model.device) for key, value in encoded.items()}
        lengths.extend(attention_mask.sum(dim=1).tolist())
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        # Faithful to the author repo: direct hidden_states[layer_index] and unmasked mean over padded tokens.
        layer_hidden = outputs.hidden_states[layer_index]
        vectors.append(layer_hidden.mean(dim=1).float().cpu().numpy())
        del outputs
    return np.concatenate(vectors, axis=0), lengths


def _evaluate_probe(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    proba = clf.predict_proba(test_X)[:, 1]
    weights = clf.coef_[0]
    ranked = np.argsort(np.abs(weights))[::-1]
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "auroc": float(roc_auc_score(test_y, proba)),
        "precision": float(precision_score(test_y, pred)),
        "recall": float(recall_score(test_y, pred)),
        "f1": float(f1_score(test_y, pred)),
        "top10": ranked[:10].tolist(),
        "classifier": clf,
    }


def _load_best_token_cloud_metrics(token_cloud_root: Path, model_dir_name: str, dataset_name: str) -> dict[str, Any] | None:
    model_dir_map = {
        "meta_llama_llama_3_1_8b_instruct": "meta_llama_Llama_3.1_8B_Instruct",
        "mistralai_mistral_7b_instruct_v0_3": "mistralai_Mistral_7B_Instruct_v0.3",
        "google_gemma_7b_it": "google_gemma_7b_it",
    }
    final_path = token_cloud_root / model_dir_map[model_dir_name] / "token_cloud_topology_final_metrics.parquet"
    if not final_path.exists():
        return None
    final_df = pd.read_parquet(final_path)
    subset = final_df.loc[final_df["dataset"].eq(dataset_name)].copy()
    if subset.empty:
        return None
    best_row = subset.sort_values(["test_auroc", "test_accuracy"], ascending=[False, False]).iloc[0]
    return {
        "feature_set": str(best_row["feature_set"]),
        "accuracy": float(best_row["test_accuracy"]),
        "auroc": float(best_row["test_auroc"]),
        "selection_mode": str(best_row["selection_mode"]),
        "selection_signature": str(best_row["selection_signature"]),
    }


def run_author_repo_eval(
    *,
    author_repo_root: Path,
    output_root: Path,
    token_cloud_root: Path,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int = 400,
    test_per_class: int = 1000,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    rows: list[dict[str, Any]] = []
    run_artifacts: dict[str, str] = {}

    for model_key in model_keys:
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

        model_output_dir = ensure_dir(output_root / spec.output_dir_name)
        combo_outputs: list[str] = []

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
            X, lengths = _extract_author_style_hidden_states(texts=texts, tokenizer=tokenizer, model=model)
            train_end = train_per_class * 2
            test_ambig_start = train_end
            test_clear_start = train_end + test_per_class
            train_X = np.vstack([X[:train_per_class], X[train_per_class:train_end]])
            test_X = np.vstack([X[test_ambig_start:test_clear_start], X[test_clear_start:test_clear_start + test_per_class]])
            train_y = np.array([0] * train_per_class + [1] * train_per_class)
            test_y = np.array([0] * test_per_class + [1] * test_per_class)

            full = _evaluate_probe(train_X, train_y, test_X, test_y)
            super_probe = _evaluate_probe(
                train_X[:, spec.super_neurons],
                train_y,
                test_X[:, spec.super_neurons],
                test_y,
            )
            token_cloud = _load_best_token_cloud_metrics(token_cloud_root, spec.output_dir_name, dataset_name)
            paper = PAPER_TABLE_1[(dataset_name, spec.label)]

            payload = {
                "created_at": utc_now_iso(),
                "model_name": spec.label,
                "load_repo": spec.load_repo,
                "prompt_model_name": spec.prompt_model_name,
                "dataset": dataset_name,
                "author_style": {
                    "layer_index": 14,
                    "pooling": "unmasked_mean_over_padded_sequence",
                    "train_per_class": int(train_per_class),
                    "test_per_class": int(test_per_class),
                    "shuffle_seeds": {"train_ambig": 11, "train_clear": 12, "test_ambig": 13, "test_clear": 14},
                    "avg_token_length": float(np.mean(lengths)),
                    "median_token_length": float(np.median(lengths)),
                },
                "full_probe": {key: value for key, value in full.items() if key != "classifier"},
                "super_neurons": spec.super_neurons,
                "super_probe": {key: value for key, value in super_probe.items() if key != "classifier"},
                "paper_claim": paper,
                "token_cloud_best": token_cloud,
            }
            out_path = model_output_dir / f"{dataset_name}__author_repo_eval.json"
            write_json(out_path, payload)
            combo_outputs.append(str(out_path))

            row = {
                "model": spec.label,
                "dataset": dataset_name,
                "paper_accuracy": float(paper["accuracy"]) / 100.0,
                "paper_f1": float(paper["f1"]) / 100.0,
                "author_full_accuracy": float(full["accuracy"]),
                "author_full_auroc": float(full["auroc"]),
                "author_full_f1": float(full["f1"]),
                "author_full_gap_to_paper_acc": float(full["accuracy"] - float(paper["accuracy"]) / 100.0),
                "author_super_accuracy": float(super_probe["accuracy"]),
                "author_super_auroc": float(super_probe["auroc"]),
                "author_super_f1": float(super_probe["f1"]),
                "token_cloud_feature_set": None if token_cloud is None else token_cloud["feature_set"],
                "token_cloud_accuracy": np.nan if token_cloud is None else float(token_cloud["accuracy"]),
                "token_cloud_auroc": np.nan if token_cloud is None else float(token_cloud["auroc"]),
                "author_full_minus_token_cloud_acc": np.nan if token_cloud is None else float(full["accuracy"] - token_cloud["accuracy"]),
                "author_full_minus_token_cloud_auroc": np.nan if token_cloud is None else float(full["auroc"] - token_cloud["auroc"]),
                "author_super_minus_token_cloud_acc": np.nan if token_cloud is None else float(super_probe["accuracy"] - token_cloud["accuracy"]),
                "author_super_minus_token_cloud_auroc": np.nan if token_cloud is None else float(super_probe["auroc"] - token_cloud["auroc"]),
                "avg_token_length": float(np.mean(lengths)),
                "top10_author_full": full["top10"],
                "super_neurons": spec.super_neurons,
            }
            rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_artifacts[spec.output_dir_name] = str(model_output_dir)

    summary_df = pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    summary_path = output_root / "author_repo_eval_summary.parquet"
    write_parquet(summary_df, summary_path)

    lines = [
        "# Author Repo Evaluation Summary",
        "",
        "This report runs the first author's public code path against the locally cached exact models,",
        "then compares those numbers against the paper claims and the token-cloud topology classifier.",
        "",
        f"- Train per class: `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        "",
        "| Model | Dataset | Paper Acc | Author Full Acc | Gap | Author Super Acc | Token-Cloud Acc | Author Full AUROC | Token-Cloud AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.to_dict(orient="records"):
        token_acc = "NA" if not np.isfinite(row["token_cloud_accuracy"]) else f"{row['token_cloud_accuracy']:.4f}"
        token_auroc = "NA" if not np.isfinite(row["token_cloud_auroc"]) else f"{row['token_cloud_auroc']:.4f}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['paper_accuracy']:.4f} | "
            f"{row['author_full_accuracy']:.4f} | {row['author_full_gap_to_paper_acc']:+.4f} | "
            f"{row['author_super_accuracy']:.4f} | {token_acc} | {row['author_full_auroc']:.4f} | {token_auroc} |"
        )
    report_path = output_root / "author_repo_eval_summary.md"
    write_markdown(report_path, "\n".join(lines) + "\n")

    metadata_path = output_root / "author_repo_eval_metadata.json"
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "author_repo_root": str(author_repo_root),
            "output_root": str(output_root),
            "token_cloud_root": str(token_cloud_root),
            "model_keys": model_keys,
            "dataset_names": dataset_names,
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "outputs": {
                "summary_parquet": str(summary_path),
                "summary_markdown": str(report_path),
                "per_model_dirs": run_artifacts,
            },
        },
    )
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/author_repo_eval",
    )
    parser.add_argument(
        "--token-cloud-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_classifier_all_datasets",
    )
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_author_repo_eval(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        token_cloud_root=Path(args.token_cloud_root),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
    )
