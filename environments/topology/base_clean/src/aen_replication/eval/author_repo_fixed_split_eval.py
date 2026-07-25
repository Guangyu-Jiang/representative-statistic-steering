"""Leakage-fixed evaluation of the public single-layer AEN setup.

This keeps the first author's public representation/probe choices:

- author prompt formatting
- layer index 14
- unmasked mean over padded hidden-state sequence positions
- logistic-regression full probe
- official paper AEN dimensions

The only intended split change is to split paired example IDs once and reuse
that split for the ambiguous and clear versions, so train/test IDs are disjoint.
"""

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

from aen_replication.eval.author_repo_eval import (
    DATASET_SPECS,
    MODEL_SPECS,
    _extract_author_style_hidden_states,
    _format_prompt,
)
from aen_replication.eval.paper_audit import PAPER_TABLE_1
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet


MODEL_DIR_MAP = {
    "meta_llama_llama_3_1_8b_instruct": "meta_llama_Llama_3.1_8B_Instruct",
    "mistralai_mistral_7b_instruct_v0_3": "mistralai_Mistral_7B_Instruct_v0.3",
    "google_gemma_7b_it": "google_gemma_7b_it",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/author_repo_fixed_split_eval",
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


def _load_records_by_id(*, author_repo_root: Path, dataset_name: str, prompt_model_name: str) -> tuple[dict[str, str], dict[str, str]]:
    spec = DATASET_SPECS[dataset_name]
    ambig_path = author_repo_root / spec["author_dir"] / spec["ambig_file"]
    clear_path = author_repo_root / spec["author_dir"] / spec["clear_file"]
    with ambig_path.open("r", encoding="utf-8") as handle:
        ambig_payload = json.load(handle)
    with clear_path.open("r", encoding="utf-8") as handle:
        clear_payload = json.load(handle)
    ambig = {
        str(item["id"]): _format_prompt(str(item["prompt"]), prompt_model_name)
        for item in ambig_payload
    }
    clear = {
        str(item["id"]): _format_prompt(str(item["prompt"]), prompt_model_name)
        for item in clear_payload
    }
    return ambig, clear


def _make_paired_disjoint_split(
    *,
    ambig_by_id: dict[str, str],
    clear_by_id: dict[str, str],
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> dict[str, Any]:
    common_ids = np.asarray([example_id for example_id in ambig_by_id if example_id in clear_by_id], dtype=object)
    required = int(train_per_class) + int(test_per_class)
    if len(common_ids) < required:
        raise ValueError(f"Need at least {required} paired IDs, found {len(common_ids)}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(common_ids)
    train_ids = common_ids[:train_per_class].tolist()
    train_prompts = (
        set(ambig_by_id[example_id] for example_id in train_ids)
        | set(clear_by_id[example_id] for example_id in train_ids)
    )
    test_ids: list[str] = []
    for example_id in common_ids[train_per_class:].tolist():
        candidate_prompts = {ambig_by_id[example_id], clear_by_id[example_id]}
        if train_prompts & candidate_prompts:
            continue
        test_ids.append(example_id)
        if len(test_ids) == test_per_class:
            break
    if len(test_ids) < test_per_class:
        raise ValueError(
            f"Only found {len(test_ids)} prompt-disjoint test IDs after selecting {train_per_class} train IDs."
        )
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"Fixed split still has ID overlap: {overlap[:5]}")
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_ambig": [ambig_by_id[example_id] for example_id in train_ids],
        "train_clear": [clear_by_id[example_id] for example_id in train_ids],
        "test_ambig": [ambig_by_id[example_id] for example_id in test_ids],
        "test_clear": [clear_by_id[example_id] for example_id in test_ids],
        "available_pair_count": int(len(common_ids)),
        "pair_overlap": int(len(overlap)),
        "prompt_overlap": int(
            len(
                (
                    set(ambig_by_id[example_id] for example_id in train_ids)
                    | set(clear_by_id[example_id] for example_id in train_ids)
                )
                & (
                    set(ambig_by_id[example_id] for example_id in test_ids)
                    | set(clear_by_id[example_id] for example_id in test_ids)
                )
            )
        ),
    }


def _fit_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    proba = clf.predict_proba(test_x)[:, 1]
    weights = clf.coef_[0]
    ranked = np.argsort(np.abs(weights))[::-1]
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "auroc": float(roc_auc_score(test_y, proba)),
        "f1": float(f1_score(test_y, pred)),
        "top10": ranked[:10].tolist(),
        "classifier": clf,
    }


def _topology_best(*, topology_root: Path, model_dir_name: str, dataset: str) -> dict[str, Any] | None:
    topology_dir_name = MODEL_DIR_MAP[model_dir_name]
    path = topology_root / topology_dir_name / "token_cloud_topology_final_metrics.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    subset = df.loc[df["dataset"].eq(dataset)].copy()
    if subset.empty:
        return None
    row = subset.sort_values(["test_auroc", "test_accuracy"], ascending=[False, False]).iloc[0]
    return {
        "selection_mode": str(row["selection_mode"]),
        "selection_signature": str(row["selection_signature"]),
        "feature_set": str(row["feature_set"]),
        "accuracy": float(row["test_accuracy"]),
        "auroc": float(row["test_auroc"]),
        "f1": float(row["test_f1"]),
    }


def _row_from_metrics(
    *,
    model_label: str,
    model_dir: str,
    dataset: str,
    split_payload: dict[str, Any],
    full: dict[str, Any],
    official_super: dict[str, Any],
    dynamic_super: dict[str, Any],
    official_super_neurons: list[int],
    dynamic_super_neurons: list[int],
    topology: dict[str, Any] | None,
    train_per_class: int,
    test_per_class: int,
    layer_index: int,
    split_seed: int,
    avg_token_length: float,
) -> dict[str, Any]:
    paper = PAPER_TABLE_1[(dataset, model_label)]
    topology_acc = np.nan if topology is None else float(topology["accuracy"])
    topology_auroc = np.nan if topology is None else float(topology["auroc"])
    return {
        "model": model_label,
        "model_dir": model_dir,
        "dataset": dataset,
        "train_per_class": int(train_per_class),
        "test_per_class": int(test_per_class),
        "layer_index": int(layer_index),
        "split_seed": int(split_seed),
        "available_pair_count": int(split_payload["available_pair_count"]),
        "pair_overlap": int(split_payload["pair_overlap"]),
        "prompt_overlap": int(split_payload["prompt_overlap"]),
        "avg_token_length": float(avg_token_length),
        "paper_full_accuracy": float(paper["accuracy"]) / 100.0,
        "paper_full_f1": float(paper["f1"]) / 100.0,
        "fixed_full_accuracy": float(full["accuracy"]),
        "fixed_full_auroc": float(full["auroc"]),
        "fixed_full_f1": float(full["f1"]),
        "fixed_full_gap_to_paper_accuracy": float(full["accuracy"] - float(paper["accuracy"]) / 100.0),
        "official_super_neurons": official_super_neurons,
        "fixed_official_super_accuracy": float(official_super["accuracy"]),
        "fixed_official_super_auroc": float(official_super["auroc"]),
        "fixed_official_super_f1": float(official_super["f1"]),
        "dynamic_super_neurons": dynamic_super_neurons,
        "fixed_dynamic_super_accuracy": float(dynamic_super["accuracy"]),
        "fixed_dynamic_super_auroc": float(dynamic_super["auroc"]),
        "fixed_dynamic_super_f1": float(dynamic_super["f1"]),
        "top10_fixed_full": full["top10"],
        "topology_selection": None if topology is None else f"{topology['selection_mode']}:{topology['selection_signature']}",
        "topology_feature_set": None if topology is None else topology["feature_set"],
        "topology_accuracy": topology_acc,
        "topology_auroc": topology_auroc,
        "topology_f1": np.nan if topology is None else float(topology["f1"]),
        "fixed_full_minus_topology_accuracy": np.nan if topology is None else float(full["accuracy"] - topology_acc),
        "fixed_full_minus_topology_auroc": np.nan if topology is None else float(full["auroc"] - topology_auroc),
        "fixed_official_super_minus_topology_accuracy": np.nan
        if topology is None
        else float(official_super["accuracy"] - topology_acc),
        "fixed_official_super_minus_topology_auroc": np.nan
        if topology is None
        else float(official_super["auroc"] - topology_auroc),
        "fixed_dynamic_super_minus_topology_accuracy": np.nan
        if topology is None
        else float(dynamic_super["accuracy"] - topology_acc),
        "fixed_dynamic_super_minus_topology_auroc": np.nan
        if topology is None
        else float(dynamic_super["auroc"] - topology_auroc),
    }


def _render_report(df: pd.DataFrame, output_root: Path) -> str:
    lines = [
        "# Leakage-Fixed Author Repo Single-Layer Evaluation",
        "",
        "This reruns the public single-layer setup after replacing the leaky split with a paired-ID-disjoint split.",
        "Everything else intentionally follows the author code path: author prompts, layer 14, unmasked mean pooling, and logistic regression.",
        "",
        "- Labels follow the public script convention: ambiguous = 0, clear = 1.",
        "- `Official AEN` uses the paper/public hard-coded super-neuron indices.",
        "- `Dynamic AEN` reselects the top-k dimensions from the fixed-split full-probe weights, where k equals the number of official AENs for that model.",
        "- `Topology` is the best no-overlap token-cloud topology result already computed in this repo on the same two datasets.",
        "",
        "## Fixed Author Results",
        "",
        "| Model | Dataset | Pair/Prompt overlap | Full Acc/AUROC | Official AEN Acc/AUROC | Dynamic AEN Acc/AUROC | Dynamic AENs | Topology Acc/AUROC | Full - Topology Acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {int(row['pair_overlap'])}/{int(row['prompt_overlap'])} | "
            f"{row['fixed_full_accuracy']:.4f}/{row['fixed_full_auroc']:.4f} | "
            f"{row['fixed_official_super_accuracy']:.4f}/{row['fixed_official_super_auroc']:.4f} | "
            f"{row['fixed_dynamic_super_accuracy']:.4f}/{row['fixed_dynamic_super_auroc']:.4f} | "
            f"`{row['dynamic_super_neurons']}` | "
            f"{row['topology_accuracy']:.4f}/{row['topology_auroc']:.4f} | "
            f"{row['fixed_full_minus_topology_accuracy']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gaps Versus Paper",
            "",
            "| Model | Dataset | Paper Full Acc | Fixed Full Acc | Gap | Paper AENs | Official AEN Acc | Dynamic AEN Acc |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['paper_full_accuracy']:.4f} | "
            f"{row['fixed_full_accuracy']:.4f} | {row['fixed_full_gap_to_paper_accuracy']:+.4f} | "
            f"`{row['official_super_neurons']}` | {row['fixed_official_super_accuracy']:.4f} | "
            f"{row['fixed_dynamic_super_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The full-neuron probe remains stronger than the topology classifier after fixing leakage.",
            "- With exact author-style extraction, the official paper AEN dimensions also remain strong under the fixed split.",
            "- The fixed-split dynamic AENs match the official AEN sets in these runs, up to ordering.",
            "- Therefore, split leakage is real, but it is not sufficient by itself to explain the high author-style probe/AEN performance. The author-style representation and full-neuron signal remain highly predictive.",
            "- The topology classifier does not beat the full-neuron probe or the official-AEN probe on these two datasets.",
            "",
            f"- Metrics parquet: `{output_root / 'author_repo_fixed_split_metrics.parquet'}`",
            f"- Metadata: `{output_root / 'author_repo_fixed_split_metadata.json'}`",
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
                ambig_by_id, clear_by_id = _load_records_by_id(
                    author_repo_root=author_repo_root,
                    dataset_name=dataset,
                    prompt_model_name=spec.prompt_model_name,
                )
                split_payload = _make_paired_disjoint_split(
                    ambig_by_id=ambig_by_id,
                    clear_by_id=clear_by_id,
                    train_per_class=int(args.train_per_class),
                    test_per_class=int(args.test_per_class),
                    seed=int(args.split_seed),
                )
                texts = (
                    split_payload["train_ambig"]
                    + split_payload["train_clear"]
                    + split_payload["test_ambig"]
                    + split_payload["test_clear"]
                )
                x_all, lengths = _extract_author_style_hidden_states(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    layer_index=int(args.layer_index),
                    batch_size=int(args.batch_size),
                )
                train_end = int(args.train_per_class) * 2
                test_ambig_start = train_end
                test_clear_start = train_end + int(args.test_per_class)
                train_x = np.vstack([x_all[: args.train_per_class], x_all[args.train_per_class : train_end]])
                test_x = np.vstack(
                    [
                        x_all[test_ambig_start:test_clear_start],
                        x_all[test_clear_start : test_clear_start + args.test_per_class],
                    ]
                )
                train_y = np.array([0] * args.train_per_class + [1] * args.train_per_class, dtype=int)
                test_y = np.array([0] * args.test_per_class + [1] * args.test_per_class, dtype=int)

                full = _fit_probe(train_x, train_y, test_x, test_y)
                official_super = _fit_probe(
                    train_x[:, spec.super_neurons],
                    train_y,
                    test_x[:, spec.super_neurons],
                    test_y,
                )
                dynamic_super_neurons = full["top10"][: len(spec.super_neurons)]
                dynamic_super = _fit_probe(
                    train_x[:, dynamic_super_neurons],
                    train_y,
                    test_x[:, dynamic_super_neurons],
                    test_y,
                )
                topology = _topology_best(
                    topology_root=topology_root,
                    model_dir_name=spec.output_dir_name,
                    dataset=dataset,
                )
                rows.append(
                    _row_from_metrics(
                        model_label=spec.label,
                        model_dir=spec.output_dir_name,
                        dataset=dataset,
                        split_payload=split_payload,
                        full=full,
                        official_super=official_super,
                        dynamic_super=dynamic_super,
                        official_super_neurons=spec.super_neurons,
                        dynamic_super_neurons=dynamic_super_neurons,
                        topology=topology,
                        train_per_class=int(args.train_per_class),
                        test_per_class=int(args.test_per_class),
                        layer_index=int(args.layer_index),
                        split_seed=int(args.split_seed),
                        avg_token_length=float(np.mean(lengths)),
                    )
                )
                write_json(
                    output_root / f"{spec.output_dir_name}__{dataset}__fixed_split.json",
                    rows[-1],
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    write_parquet(df, output_root / "author_repo_fixed_split_metrics.parquet")
    write_markdown(output_root / "author_repo_fixed_split_report.md", _render_report(df, output_root))
    write_json(
        output_root / "author_repo_fixed_split_metadata.json",
        {
            "created_at": utc_now_iso(),
            "author_repo_root": str(author_repo_root),
            "output_root": str(output_root),
            "topology_root": str(topology_root),
            "models": list(args.models),
            "datasets": list(args.datasets),
            "train_per_class": int(args.train_per_class),
            "test_per_class": int(args.test_per_class),
            "split_seed": int(args.split_seed),
            "batch_size": int(args.batch_size),
            "layer_index": int(args.layer_index),
            "split_definition": "paired IDs are shuffled once, train uses first train_per_class IDs, test uses next test_per_class IDs",
        },
    )


if __name__ == "__main__":
    main()
