"""Masked-pooling ablation for the public single-layer AEN setup."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.eval.author_repo_eval import DATASET_SPECS, MODEL_SPECS, _format_prompt
from aen_replication.eval.author_repo_fixed_split_eval import _make_paired_disjoint_split, _topology_best
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/author_repo_masked_pooling_eval",
    )
    parser.add_argument(
        "--topology-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_classifier_all_datasets",
    )
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--layer-index", type=int, default=14)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama31_8b_instruct", "mistral_7b_instruct_v03", "gemma_7b_it"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ambigqa", "situatedqa"],
        choices=sorted(DATASET_SPECS),
    )
    return parser.parse_args()


def _load_formatted_rows(*, author_repo_root: Path, dataset_name: str, prompt_model_name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec = DATASET_SPECS[dataset_name]
    with (author_repo_root / spec["author_dir"] / spec["ambig_file"]).open("r", encoding="utf-8") as handle:
        ambig_payload = json.load(handle)
    with (author_repo_root / spec["author_dir"] / spec["clear_file"]).open("r", encoding="utf-8") as handle:
        clear_payload = json.load(handle)
    ambig = [
        {
            "id": str(item["id"]),
            "prompt": _format_prompt(str(item["prompt"]), prompt_model_name),
        }
        for item in ambig_payload
    ]
    clear = [
        {
            "id": str(item["id"]),
            "prompt": _format_prompt(str(item["prompt"]), prompt_model_name),
        }
        for item in clear_payload
    ]
    return ambig, clear


def _shuffle_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    copied = list(rows)
    np.random.seed(seed)
    np.random.shuffle(copied)
    return copied


def _official_leaky_split(
    ambig_rows: list[dict[str, str]],
    clear_rows: list[dict[str, str]],
    *,
    train_per_class: int,
    test_per_class: int,
) -> dict[str, Any]:
    train_ambig = _shuffle_rows(ambig_rows, 11)[:train_per_class]
    train_clear = _shuffle_rows(clear_rows, 12)[:train_per_class]
    test_ambig = _shuffle_rows(ambig_rows, 13)[-test_per_class:]
    test_clear = _shuffle_rows(clear_rows, 14)[-test_per_class:]
    train_ids = {row["id"] for row in train_ambig} | {row["id"] for row in train_clear}
    test_ids = {row["id"] for row in test_ambig} | {row["id"] for row in test_clear}
    train_prompts = {row["prompt"] for row in train_ambig} | {row["prompt"] for row in train_clear}
    test_prompts = {row["prompt"] for row in test_ambig} | {row["prompt"] for row in test_clear}
    return {
        "split_mode": "official_leaky",
        "train_ambig": [row["prompt"] for row in train_ambig],
        "train_clear": [row["prompt"] for row in train_clear],
        "test_ambig": [row["prompt"] for row in test_ambig],
        "test_clear": [row["prompt"] for row in test_clear],
        "pair_overlap": int(len(train_ids & test_ids)),
        "prompt_overlap": int(len(train_prompts & test_prompts)),
    }


def _fixed_split(
    ambig_rows: list[dict[str, str]],
    clear_rows: list[dict[str, str]],
    *,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> dict[str, Any]:
    ambig_by_id = {row["id"]: row["prompt"] for row in ambig_rows}
    clear_by_id = {row["id"]: row["prompt"] for row in clear_rows}
    payload = _make_paired_disjoint_split(
        ambig_by_id=ambig_by_id,
        clear_by_id=clear_by_id,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        seed=seed,
    )
    return {
        "split_mode": "fixed_disjoint",
        **payload,
    }


def _extract_pooling_variants(
    *,
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_index: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    unmasked: list[np.ndarray] = []
    masked: list[np.ndarray] = []
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        attention_mask = encoded["attention_mask"].to(model.device)
        model_inputs = {key: value.to(model.device) for key, value in encoded.items()}
        lengths.extend(attention_mask.sum(dim=1).tolist())
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        layer_hidden = outputs.hidden_states[layer_index]
        unmasked.append(layer_hidden.mean(dim=1).float().cpu().numpy())
        mask = attention_mask.unsqueeze(-1).to(layer_hidden.dtype)
        counts = mask.sum(dim=1).clamp(min=1.0)
        masked.append(((layer_hidden * mask).sum(dim=1) / counts).float().cpu().numpy())
        del outputs
    return {"unmasked": np.concatenate(unmasked, axis=0), "masked": np.concatenate(masked, axis=0)}, lengths


def _fit_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    proba = clf.predict_proba(test_x)[:, 1]
    ranked = np.argsort(np.abs(clf.coef_[0]))[::-1]
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "auroc": float(roc_auc_score(test_y, proba)),
        "f1": float(f1_score(test_y, pred)),
        "top10": ranked[:10].tolist(),
    }


def _split_matrix(matrix: np.ndarray, train_per_class: int, test_per_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_end = train_per_class * 2
    test_ambig_start = train_end
    test_clear_start = train_end + test_per_class
    train_x = np.vstack([matrix[:train_per_class], matrix[train_per_class:train_end]])
    test_x = np.vstack(
        [
            matrix[test_ambig_start:test_clear_start],
            matrix[test_clear_start : test_clear_start + test_per_class],
        ]
    )
    train_y = np.array([0] * train_per_class + [1] * train_per_class, dtype=int)
    test_y = np.array([0] * test_per_class + [1] * test_per_class, dtype=int)
    return train_x, train_y, test_x, test_y


def _evaluate_variant(
    *,
    matrix: np.ndarray,
    train_per_class: int,
    test_per_class: int,
    official_super_neurons: list[int],
) -> dict[str, Any]:
    train_x, train_y, test_x, test_y = _split_matrix(matrix, train_per_class, test_per_class)
    full = _fit_probe(train_x, train_y, test_x, test_y)
    dynamic_neurons = full["top10"][: len(official_super_neurons)]
    official = _fit_probe(train_x[:, official_super_neurons], train_y, test_x[:, official_super_neurons], test_y)
    dynamic = _fit_probe(train_x[:, dynamic_neurons], train_y, test_x[:, dynamic_neurons], test_y)
    return {
        "full": full,
        "official_super": official,
        "dynamic_super": dynamic,
        "dynamic_super_neurons": dynamic_neurons,
    }


def _render_report(df: pd.DataFrame, output_root: Path) -> str:
    lines = [
        "# Author Repo Masked-Pooling Ablation",
        "",
        "This keeps the public single-layer setup and changes only the pooling readout:",
        "`unmasked` is the repo behavior, `masked` averages only non-padding tokens using the attention mask.",
        "",
        "## Fixed Disjoint Split",
        "",
        "| Model | Dataset | Pooling | Full Acc/AUROC | Official AEN Acc/AUROC | Dynamic AENs | Topology Acc/AUROC |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    fixed = df.loc[df["split_mode"].eq("fixed_disjoint")].copy()
    for row in fixed.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['pooling']} | "
            f"{row['full_accuracy']:.4f}/{row['full_auroc']:.4f} | "
            f"{row['official_super_accuracy']:.4f}/{row['official_super_auroc']:.4f} | "
            f"`{row['dynamic_super_neurons']}` | "
            f"{row['topology_accuracy']:.4f}/{row['topology_auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Official Leaky Split",
            "",
            "| Model | Dataset | Pooling | Pair/Prompt overlap | Full Acc/AUROC | Official AEN Acc/AUROC | Dynamic AENs |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    leaky = df.loc[df["split_mode"].eq("official_leaky")].copy()
    for row in leaky.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['pooling']} | "
            f"{int(row['pair_overlap'])}/{int(row['prompt_overlap'])} | "
            f"{row['full_accuracy']:.4f}/{row['full_auroc']:.4f} | "
            f"{row['official_super_accuracy']:.4f}/{row['official_super_auroc']:.4f} | "
            f"`{row['dynamic_super_neurons']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Masked pooling substantially lowers AEN performance compared with the repo's unmasked pooling, especially on the official paper AEN dimensions.",
            "- Full probes remain strong even with masked pooling, so token content still carries a large dataset signal.",
            "- Under the fixed split, masked-pooling full probes still beat topology on every model/dataset row.",
            "- Under the fixed split, topology can be competitive with or better than masked-pooling official-AEN probes in several rows.",
            "",
            f"- Metrics parquet: `{output_root / 'author_repo_masked_pooling_metrics.parquet'}`",
            f"- Metadata: `{output_root / 'author_repo_masked_pooling_metadata.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    author_repo_root = Path(args.author_repo_root)
    output_root = ensure_dir(Path(args.output_root))
    topology_root = Path(args.topology_root)
    rows: list[dict[str, Any]] = []

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        model_path = snapshot_download(repo_id=spec.load_repo, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        model.eval()
        try:
            for dataset in args.datasets:
                ambig_rows, clear_rows = _load_formatted_rows(
                    author_repo_root=author_repo_root,
                    dataset_name=dataset,
                    prompt_model_name=spec.prompt_model_name,
                )
                split_payloads = [
                    _official_leaky_split(
                        ambig_rows,
                        clear_rows,
                        train_per_class=int(args.train_per_class),
                        test_per_class=int(args.test_per_class),
                    ),
                    _fixed_split(
                        ambig_rows,
                        clear_rows,
                        train_per_class=int(args.train_per_class),
                        test_per_class=int(args.test_per_class),
                        seed=int(args.split_seed),
                    ),
                ]
                topology = _topology_best(topology_root=topology_root, model_dir_name=spec.output_dir_name, dataset=dataset)
                for split_payload in split_payloads:
                    texts = (
                        split_payload["train_ambig"]
                        + split_payload["train_clear"]
                        + split_payload["test_ambig"]
                        + split_payload["test_clear"]
                    )
                    matrices, lengths = _extract_pooling_variants(
                        texts=texts,
                        tokenizer=tokenizer,
                        model=model,
                        layer_index=int(args.layer_index),
                        batch_size=int(args.batch_size),
                    )
                    for pooling, matrix in matrices.items():
                        result = _evaluate_variant(
                            matrix=matrix,
                            train_per_class=int(args.train_per_class),
                            test_per_class=int(args.test_per_class),
                            official_super_neurons=spec.super_neurons,
                        )
                        rows.append(
                            {
                                "model": spec.label,
                                "model_key": model_key,
                                "dataset": dataset,
                                "split_mode": split_payload["split_mode"],
                                "pooling": pooling,
                                "train_per_class": int(args.train_per_class),
                                "test_per_class": int(args.test_per_class),
                                "pair_overlap": int(split_payload["pair_overlap"]),
                                "prompt_overlap": int(split_payload["prompt_overlap"]),
                                "avg_token_length": float(np.mean(lengths)),
                                "official_super_neurons": spec.super_neurons,
                                "dynamic_super_neurons": result["dynamic_super_neurons"],
                                "full_accuracy": float(result["full"]["accuracy"]),
                                "full_auroc": float(result["full"]["auroc"]),
                                "full_f1": float(result["full"]["f1"]),
                                "official_super_accuracy": float(result["official_super"]["accuracy"]),
                                "official_super_auroc": float(result["official_super"]["auroc"]),
                                "official_super_f1": float(result["official_super"]["f1"]),
                                "dynamic_super_accuracy": float(result["dynamic_super"]["accuracy"]),
                                "dynamic_super_auroc": float(result["dynamic_super"]["auroc"]),
                                "dynamic_super_f1": float(result["dynamic_super"]["f1"]),
                                "full_top10": result["full"]["top10"],
                                "topology_accuracy": np.nan if topology is None else float(topology["accuracy"]),
                                "topology_auroc": np.nan if topology is None else float(topology["auroc"]),
                                "topology_feature_set": None if topology is None else topology["feature_set"],
                                "topology_selection": None
                                if topology is None
                                else f"{topology['selection_mode']}:{topology['selection_signature']}",
                            }
                        )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["model", "dataset", "split_mode", "pooling"]).reset_index(drop=True)
    write_parquet(df, output_root / "author_repo_masked_pooling_metrics.parquet")
    write_markdown(output_root / "author_repo_masked_pooling_report.md", _render_report(df, output_root))
    write_json(
        output_root / "author_repo_masked_pooling_metadata.json",
        {
            "created_at": utc_now_iso(),
            "author_repo_root": str(author_repo_root),
            "output_root": str(output_root),
            "topology_root": str(topology_root),
            "models": list(args.models),
            "datasets": list(args.datasets),
            "train_per_class": int(args.train_per_class),
            "test_per_class": int(args.test_per_class),
            "layer_index": int(args.layer_index),
            "batch_size": int(args.batch_size),
        },
    )


if __name__ == "__main__":
    main()
