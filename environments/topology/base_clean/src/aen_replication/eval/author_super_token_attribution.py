"""Trace official super-neuron ambiguity signals back to question words.

This script follows the first author's prompt formatting and public super-neuron
indices, trains a super-neuron-only probe, and then attributes the resulting
ambiguity score back to the user-question tokens/words inside each prompt.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.author_repo_eval import (
    DATASET_SPECS,
    MODEL_SPECS,
    _extract_author_style_hidden_states,
    _format_prompt,
)
from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.train.aen import select_aens
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


def _load_question_records(
    *,
    author_repo_root: Path,
    dataset_name: str,
    prompt_model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = DATASET_SPECS[dataset_name]
    ambig_path = author_repo_root / spec["author_dir"] / spec["ambig_file"]
    clear_path = author_repo_root / spec["author_dir"] / spec["clear_file"]
    ambig_payload = json.loads(ambig_path.read_text(encoding="utf-8"))
    clear_payload = json.loads(clear_path.read_text(encoding="utf-8"))

    def _records(payload: list[dict[str, Any]], label_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in payload:
            question = str(item["prompt"]).strip()
            rows.append(
                {
                    "id": int(item["id"]),
                    "group": label_name,
                    "question": question,
                    "formatted": _format_prompt(question, prompt_model_name),
                }
            )
        return rows

    return _records(ambig_payload, "ambiguous"), _records(clear_payload, "clear")


def _shuffle_and_slice_records(
    records: list[dict[str, Any]],
    *,
    train_seed: int,
    test_seed: int,
    train_per_class: int,
    test_per_class: int,
) -> dict[str, list[dict[str, Any]]]:
    if train_per_class > len(records) or test_per_class > len(records):
        raise ValueError(
            f"Requested train/test counts ({train_per_class}, {test_per_class}) exceed available record count {len(records)}."
        )
    train_records = list(records)
    test_records = list(records)
    train_rng = np.random.default_rng(train_seed)
    test_rng = np.random.default_rng(test_seed)
    train_rng.shuffle(train_records)
    test_rng.shuffle(test_records)
    return {
        "train": train_records[:train_per_class],
        "test": test_records[-test_per_class:],
    }


def _fit_super_probe(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    train_ambig: list[dict[str, Any]],
    train_clear: list[dict[str, Any]],
    super_neurons: list[int],
    layer_index: int,
    batch_size: int,
) -> dict[str, Any]:
    texts = [row["formatted"] for row in train_ambig] + [row["formatted"] for row in train_clear]
    x_train, _ = _extract_author_style_hidden_states(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        layer_index=layer_index,
        batch_size=batch_size,
    )
    y_train = np.array([1] * len(train_ambig) + [0] * len(train_clear), dtype=int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train[:, super_neurons], y_train)
    train_scores = clf.decision_function(x_train[:, super_neurons])
    return {
        "classifier": clf,
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "weights": clf.coef_[0].astype(float),
        "bias": float(clf.intercept_[0]),
    }


def _fit_probe_from_matrix(
    *,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    selected = train_matrix if indices is None else train_matrix[:, indices]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(selected, train_labels)
    train_scores = clf.decision_function(selected)
    return {
        "classifier": clf,
        "train_metrics": binary_classification_metrics(train_labels, train_scores),
        "weights": clf.coef_[0].astype(float),
        "bias": float(clf.intercept_[0]),
    }


def _build_author_style_full_probe(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    train_ambig: list[dict[str, Any]],
    train_clear: list[dict[str, Any]],
    test_ambig: list[dict[str, Any]],
    test_clear: list[dict[str, Any]],
    layer_index: int,
    batch_size: int,
) -> dict[str, Any]:
    train_texts = [row["formatted"] for row in train_ambig] + [row["formatted"] for row in train_clear]
    test_texts = [row["formatted"] for row in test_ambig] + [row["formatted"] for row in test_clear]
    train_matrix, _ = _extract_author_style_hidden_states(
        texts=train_texts,
        tokenizer=tokenizer,
        model=model,
        layer_index=layer_index,
        batch_size=batch_size,
    )
    test_matrix, _ = _extract_author_style_hidden_states(
        texts=test_texts,
        tokenizer=tokenizer,
        model=model,
        layer_index=layer_index,
        batch_size=batch_size,
    )
    train_labels = np.array([1] * len(train_ambig) + [0] * len(train_clear), dtype=int)
    test_labels = np.array([1] * len(test_ambig) + [0] * len(test_clear), dtype=int)
    full_probe = _fit_probe_from_matrix(
        train_matrix=train_matrix,
        train_labels=train_labels,
        indices=None,
    )
    full_probe["coefficients"] = full_probe["classifier"].coef_.ravel().astype(float)
    full_probe["splits"] = {
        "train": {"matrix": train_matrix, "labels": train_labels},
        "test": {"matrix": test_matrix, "labels": test_labels},
    }
    full_probe["test_metrics"] = binary_classification_metrics(
        test_labels,
        full_probe["classifier"].decision_function(test_matrix),
    )
    return full_probe


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _aggregate_word_contributions(
    *,
    question: str,
    local_offsets: list[tuple[int, int]],
    token_scores: np.ndarray,
) -> list[dict[str, Any]]:
    spans = _word_spans(question)
    if not spans:
        return []
    values = np.zeros(len(spans), dtype=float)
    for (start, end), score in zip(local_offsets, token_scores.tolist(), strict=False):
        if end <= start:
            continue
        for idx, (_, word_start, word_end) in enumerate(spans):
            if end <= word_start or start >= word_end:
                continue
            values[idx] += float(score)
    positive_mass = float(np.maximum(values, 0.0).sum())
    rows: list[dict[str, Any]] = []
    for idx, (word, start, end) in enumerate(spans):
        score = float(values[idx])
        rows.append(
            {
                "word_index": idx,
                "word": word,
                "start": start,
                "end": end,
                "score": score,
                "positive_share": 0.0 if positive_mass <= 0.0 else max(score, 0.0) / positive_mass,
            }
        )
    return rows


def _attribute_records(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    records: list[dict[str, Any]],
    super_neurons: list[int],
    weights: np.ndarray,
    layer_index: int,
    batch_size: int,
    top_k_words: int,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        texts = [row["formatted"] for row in batch_records]
        encoded = tokenizer(texts, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoded["offset_mapping"].cpu().numpy()
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        model_inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        layer_hidden = outputs.hidden_states[layer_index].detach().float().cpu().numpy()
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        for row_idx, record in enumerate(batch_records):
            question = str(record["question"])
            formatted = str(record["formatted"])
            start_char = formatted.index(question)
            end_char = start_char + len(question)
            token_rows: list[dict[str, Any]] = []
            local_offsets: list[tuple[int, int]] = []
            token_scores: list[float] = []
            for tok_idx in range(layer_hidden.shape[1]):
                if int(attention_mask_np[row_idx, tok_idx]) == 0:
                    continue
                token_id = int(input_ids_np[row_idx, tok_idx])
                if token_id in special_ids:
                    continue
                tok_start, tok_end = (int(value) for value in offset_mapping[row_idx, tok_idx].tolist())
                if tok_end <= tok_start:
                    continue
                if tok_end <= start_char or tok_start >= end_char:
                    continue
                local_start = max(tok_start, start_char) - start_char
                local_end = min(tok_end, end_char) - start_char
                hidden = layer_hidden[row_idx, tok_idx, super_neurons]
                score = float(np.dot(hidden, weights))
                token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
                token_rows.append(
                    {
                        "token": token_text,
                        "start": local_start,
                        "end": local_end,
                        "score": score,
                    }
                )
                local_offsets.append((local_start, local_end))
                token_scores.append(score)

            word_rows = _aggregate_word_contributions(
                question=question,
                local_offsets=local_offsets,
                token_scores=np.asarray(token_scores, dtype=float),
            )
            top_words = sorted(word_rows, key=lambda item: item["score"], reverse=True)[:top_k_words]
            top_positive_word = top_words[0]["word"] if top_words else ""
            top_positive_score = float(top_words[0]["score"]) if top_words else 0.0
            rows.append(
                {
                    "question_id": int(record["id"]),
                    "group": str(record["group"]),
                    "question": question,
                    "top_positive_word": top_positive_word,
                    "top_positive_score": top_positive_score,
                    "top_words": json.dumps(top_words, ensure_ascii=True),
                    "token_attributions": json.dumps(token_rows, ensure_ascii=True),
                    "question_token_count": len(token_rows),
                }
            )
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _summarize_top_words(example_df: pd.DataFrame, *, top_k_words: int) -> pd.DataFrame:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in example_df.to_dict(orient="records"):
        group = str(row["group"])
        top_words = json.loads(str(row["top_words"]))
        for rank, word_row in enumerate(top_words[:top_k_words], start=1):
            word = str(word_row["word"]).strip()
            if not word:
                continue
            key = (group, word.lower())
            current = aggregate.setdefault(
                key,
                {
                    "group": group,
                    "word": word.lower(),
                    "count": 0,
                    "mean_score_sum": 0.0,
                    "mean_positive_share_sum": 0.0,
                    "best_rank": rank,
                },
            )
            current["count"] += 1
            current["mean_score_sum"] += float(word_row["score"])
            current["mean_positive_share_sum"] += float(word_row.get("positive_share", 0.0))
            current["best_rank"] = min(int(current["best_rank"]), int(rank))
    rows: list[dict[str, Any]] = []
    for payload in aggregate.values():
        count = max(int(payload["count"]), 1)
        rows.append(
            {
                "group": payload["group"],
                "word": payload["word"],
                "count": count,
                "best_rank": int(payload["best_rank"]),
                "mean_score": float(payload["mean_score_sum"]) / count,
                "mean_positive_share": float(payload["mean_positive_share_sum"]) / count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["group", "mean_positive_share", "count", "mean_score", "word"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)


def run_author_super_token_attribution(
    *,
    author_repo_root: Path,
    output_root: Path,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int,
    test_per_class: int,
    layer_index: int,
    batch_size: int,
    top_k_words: int,
    neuron_mode: str,
    perturb_top_k: list[int],
    perturb_sigma: float,
    perturb_trials: int,
    seed: int,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    summary_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        model_output_dir = ensure_dir(output_root / spec.output_dir_name)
        model_path = snapshot_download(repo_id=spec.load_repo, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True,
        )
        model.eval()

        try:
            for dataset_name in dataset_names:
                ambig_records, clear_records = _load_question_records(
                    author_repo_root=author_repo_root,
                    dataset_name=dataset_name,
                    prompt_model_name=spec.prompt_model_name,
                )
                ambig_split = _shuffle_and_slice_records(
                    ambig_records,
                    train_seed=11,
                    test_seed=13,
                    train_per_class=train_per_class,
                    test_per_class=test_per_class,
                )
                clear_split = _shuffle_and_slice_records(
                    clear_records,
                    train_seed=12,
                    test_seed=14,
                    train_per_class=train_per_class,
                    test_per_class=test_per_class,
                )
                if neuron_mode == "official_super":
                    selected_neurons = list(spec.super_neurons)
                    probe = _fit_super_probe(
                        tokenizer=tokenizer,
                        model=model,
                        train_ambig=ambig_split["train"],
                        train_clear=clear_split["train"],
                        super_neurons=selected_neurons,
                        layer_index=layer_index,
                        batch_size=batch_size,
                    )
                    selection_payload: dict[str, Any] = {
                        "mode": neuron_mode,
                        "selected_neurons": selected_neurons,
                    }
                elif neuron_mode == "dynamic_aen":
                    full_probe = _build_author_style_full_probe(
                        tokenizer=tokenizer,
                        model=model,
                        train_ambig=ambig_split["train"],
                        train_clear=clear_split["train"],
                        test_ambig=ambig_split["test"],
                        test_clear=clear_split["test"],
                        layer_index=layer_index,
                        batch_size=batch_size,
                    )
                    aen_selection = select_aens(
                        full_probe=full_probe,
                        perturb_top_k=[int(value) for value in perturb_top_k],
                        sigma=float(perturb_sigma),
                        trials=int(perturb_trials),
                        seed=int(seed),
                    )
                    selected_neurons = [int(value) for value in aen_selection["aen_indices"]]
                    probe = _fit_probe_from_matrix(
                        train_matrix=full_probe["splits"]["train"]["matrix"],
                        train_labels=full_probe["splits"]["train"]["labels"],
                        indices=selected_neurons,
                    )
                    probe["test_metrics"] = binary_classification_metrics(
                        full_probe["splits"]["test"]["labels"],
                        probe["classifier"].decision_function(full_probe["splits"]["test"]["matrix"][:, selected_neurons]),
                    )
                    selection_payload = {
                        "mode": neuron_mode,
                        "selected_neurons": selected_neurons,
                        "aen_selection": aen_selection,
                        "full_probe_train_metrics": full_probe["train_metrics"],
                        "full_probe_test_metrics": full_probe["test_metrics"],
                    }
                else:
                    raise ValueError(f"Unsupported neuron_mode: {neuron_mode}")
                ambig_df = _attribute_records(
                    tokenizer=tokenizer,
                    model=model,
                    records=ambig_split["test"],
                    super_neurons=selected_neurons,
                    weights=probe["weights"],
                    layer_index=layer_index,
                    batch_size=batch_size,
                    top_k_words=top_k_words,
                )
                clear_df = _attribute_records(
                    tokenizer=tokenizer,
                    model=model,
                    records=clear_split["test"],
                    super_neurons=selected_neurons,
                    weights=probe["weights"],
                    layer_index=layer_index,
                    batch_size=batch_size,
                    top_k_words=top_k_words,
                )
                example_df = pd.concat([ambig_df, clear_df], ignore_index=True)
                summary_df = _summarize_top_words(example_df, top_k_words=top_k_words)

                base_name = f"{dataset_name}__{neuron_mode}_word_attribution"
                examples_path = model_output_dir / f"{base_name}__examples.parquet"
                summary_path = model_output_dir / f"{base_name}__summary.parquet"
                report_path = model_output_dir / f"{base_name}.md"
                metadata_path = model_output_dir / f"{base_name}.metadata.json"
                write_parquet(example_df, examples_path)
                write_parquet(summary_df, summary_path)

                top_ambig = summary_df.loc[summary_df["group"].eq("ambiguous")].head(15)
                top_clear = summary_df.loc[summary_df["group"].eq("clear")].head(15)
                sample_ambig = example_df.loc[example_df["group"].eq("ambiguous")].head(5)
                lines = [
                    f"# Author Token Attribution: {spec.label} / {dataset_name} / {neuron_mode}",
                    "",
                    f"- Created at: `{utc_now_iso()}`",
                    f"- Train per class: `{train_per_class}`",
                    f"- Test per class: `{test_per_class}`",
                    f"- Layer index: `{layer_index}`",
                    f"- Neuron mode: `{neuron_mode}`",
                    f"- Selected neurons: `{selected_neurons}`",
                    f"- Train accuracy: `{probe['train_metrics']['accuracy']:.4f}`",
                    f"- Train AUROC: `{probe['train_metrics']['auroc']:.4f}`",
                    (
                        f"- Test accuracy: `{probe['test_metrics']['accuracy']:.4f}`, "
                        f"test AUROC: `{probe['test_metrics']['auroc']:.4f}`"
                        if "test_metrics" in probe
                        else ""
                    ),
                    "",
                    "## Top Ambiguous Words",
                    "",
                ]
                lines = [line for line in lines if line]
                for row in top_ambig.to_dict(orient="records"):
                    lines.append(
                        f"- `{row['word']}`: count `{row['count']}`, mean share `{row['mean_positive_share']:.4f}`, "
                        f"mean score `{row['mean_score']:.4f}`, best rank `{row['best_rank']}`"
                    )
                lines.extend(["", "## Top Clear Words", ""])
                for row in top_clear.to_dict(orient="records"):
                    lines.append(
                        f"- `{row['word']}`: count `{row['count']}`, mean share `{row['mean_positive_share']:.4f}`, "
                        f"mean score `{row['mean_score']:.4f}`, best rank `{row['best_rank']}`"
                    )
                lines.extend(["", "## Example Ambiguous Questions", ""])
                for row in sample_ambig.to_dict(orient="records"):
                    lines.append(f"- Question: `{row['question']}`")
                    lines.append(f"  Top words: `{row['top_words']}`")
                write_markdown(report_path, "\n".join(lines) + "\n")
                write_json(
                    metadata_path,
                    {
                        "created_at": utc_now_iso(),
                        "model": spec.label,
                        "dataset": dataset_name,
                        "train_per_class": int(train_per_class),
                        "test_per_class": int(test_per_class),
                        "layer_index": int(layer_index),
                        "neuron_mode": neuron_mode,
                        "selected_neurons": selected_neurons,
                        "train_metrics": probe["train_metrics"],
                        "test_metrics": probe.get("test_metrics"),
                        "selection_payload": selection_payload,
                        "output_artifacts": {
                            "examples_parquet": str(examples_path),
                            "summary_parquet": str(summary_path),
                            "report": str(report_path),
                        },
                    },
                )

                best_ambig = top_ambig.iloc[0] if not top_ambig.empty else None
                summary_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "train_per_class": int(train_per_class),
                        "test_per_class": int(test_per_class),
                        "layer_index": int(layer_index),
                        "neuron_mode": neuron_mode,
                        "selected_neurons": json.dumps(selected_neurons),
                        "selected_k": int(len(selected_neurons)),
                        "train_accuracy": float(probe["train_metrics"]["accuracy"]),
                        "train_auroc": float(probe["train_metrics"]["auroc"]),
                        "test_accuracy": float(probe["test_metrics"]["accuracy"]) if "test_metrics" in probe else None,
                        "test_auroc": float(probe["test_metrics"]["auroc"]) if "test_metrics" in probe else None,
                        "top_ambiguous_word": None if best_ambig is None else str(best_ambig["word"]),
                        "top_ambiguous_word_count": None if best_ambig is None else int(best_ambig["count"]),
                        "top_ambiguous_word_mean_share": None if best_ambig is None else float(best_ambig["mean_positive_share"]),
                        "report_path": str(report_path),
                        "examples_path": str(examples_path),
                        "summary_path": str(summary_path),
                    }
                )
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overall_df = pd.DataFrame(summary_rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    overall_summary_path = output_root / f"{neuron_mode}_token_attribution_summary.parquet"
    overall_report_path = output_root / f"{neuron_mode}_token_attribution_summary.md"
    overall_metadata_path = output_root / f"{neuron_mode}_token_attribution_metadata.json"
    write_parquet(overall_df, overall_summary_path)

    lines = [
        "# Author Token Attribution Summary",
        "",
        f"- Created at: `{utc_now_iso()}`",
        f"- Train per class: `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        f"- Neuron mode: `{neuron_mode}`",
        "",
    ]
    for row in overall_df.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['model']} / {row['dataset']}",
                "",
                f"- Selected k `{row['selected_k']}`, train accuracy `{row['train_accuracy']:.4f}`, AUROC `{row['train_auroc']:.4f}`",
                (
                    f"- Test accuracy `{row['test_accuracy']:.4f}`, AUROC `{row['test_auroc']:.4f}`"
                    if row.get("test_accuracy") is not None
                    else ""
                ),
                f"- Top ambiguous word: `{row['top_ambiguous_word']}` "
                f"(count `{row['top_ambiguous_word_count']}`, mean share `{row['top_ambiguous_word_mean_share']:.4f}`)",
                f"- Report: `{row['report_path']}`",
                "",
            ]
        )
    lines = [line for line in lines if line]
    write_markdown(overall_report_path, "\n".join(lines) + "\n")
    write_json(
        overall_metadata_path,
        {
            "created_at": utc_now_iso(),
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "layer_index": int(layer_index),
            "neuron_mode": neuron_mode,
            "perturb_top_k": [int(value) for value in perturb_top_k],
            "perturb_sigma": float(perturb_sigma),
            "perturb_trials": int(perturb_trials),
            "seed": int(seed),
            "output_artifacts": {
                "summary_parquet": str(overall_summary_path),
                "report": str(overall_report_path),
            },
        },
    )
    return {
        "summary_parquet": str(overall_summary_path),
        "report": str(overall_report_path),
        "metadata": str(overall_metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/author_super_token_attribution",
    )
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--layer-index", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--top-k-words", type=int, default=5)
    parser.add_argument("--neuron-mode", choices=["official_super", "dynamic_aen"], default="official_super")
    parser.add_argument("--perturb-top-k", type=int, nargs="*", default=[1, 2, 3, 5, 10, 20])
    parser.add_argument("--perturb-sigma", type=float, default=1.0)
    parser.add_argument("--perturb-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama31_8b_instruct", "mistral_7b_instruct_v03", "gemma_7b_it"],
    )
    parser.add_argument("--datasets", nargs="*", default=["ambigqa", "situatedqa"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_author_super_token_attribution(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
        layer_index=int(args.layer_index),
        batch_size=int(args.batch_size),
        top_k_words=int(args.top_k_words),
        neuron_mode=str(args.neuron_mode),
        perturb_top_k=[int(value) for value in args.perturb_top_k],
        perturb_sigma=float(args.perturb_sigma),
        perturb_trials=int(args.perturb_trials),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
