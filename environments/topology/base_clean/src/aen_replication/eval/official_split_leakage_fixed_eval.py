"""Audit official AEN split leakage and evaluate a leakage-free split.

The public AEN code builds train/test from the same JSON files with independent
shuffle seeds. This script reproduces that split construction to count overlap,
then evaluates the paper's hard-coded AENs on our cached leakage-free split.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.eval.paper_audit import PAPER_TABLE_1, PAPER_TABLE_8
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet


MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
        "super_neurons": [788, 4062, 1384],
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
        "super_neurons": [2070],
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
        "super_neurons": [1995],
    },
]

DATASET_FILES = {
    "ambigqa": {
        "ambiguous": "ambig_qa/ambig_questions.json",
        "clear": "ambig_qa/clean_questions.json",
    },
    "situatedqa": {
        "ambiguous": "situated/ambi_combined_question.json",
        "clear": "situated/clean_combined_question.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/official_split_leakage_fixed_eval",
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=1000)
    return parser.parse_args()


def _load_prompts(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue
        rows.append(
            {
                "row_index": str(index),
                "id": str(item.get("id", index)),
                "prompt": prompt,
            }
        )
    return rows


def _shuffle_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    shuffled = list(rows)
    np.random.seed(seed)
    np.random.shuffle(shuffled)
    return shuffled


def _official_split_overlap(
    *,
    author_repo_root: Path,
    dataset: str,
    train_per_class: int,
    test_per_class: int,
) -> list[dict[str, Any]]:
    files = DATASET_FILES[dataset]
    ambiguous = _load_prompts(author_repo_root / files["ambiguous"])
    clear = _load_prompts(author_repo_root / files["clear"])
    class_payloads = [
        ("ambiguous", ambiguous, 11, 13),
        ("clear", clear, 12, 14),
    ]
    rows: list[dict[str, Any]] = []
    for label_name, prompts, train_seed, test_seed in class_payloads:
        train_rows = _shuffle_rows(prompts, train_seed)[:train_per_class]
        test_rows = _shuffle_rows(prompts, test_seed)[-test_per_class:]
        train_prompts = {row["prompt"] for row in train_rows}
        test_prompts = {row["prompt"] for row in test_rows}
        train_ids = {row["id"] for row in train_rows}
        test_ids = {row["id"] for row in test_rows}
        prompt_overlap = train_prompts & test_prompts
        id_overlap = train_ids & test_ids
        fixed_test_candidates = [row for row in _shuffle_rows(prompts, test_seed) if row["prompt"] not in train_prompts]
        fixed_test_rows = fixed_test_candidates[-test_per_class:]
        rows.append(
            {
                "dataset": dataset,
                "class": label_name,
                "available": int(len(prompts)),
                "official_train_n": int(len(train_rows)),
                "official_test_n": int(len(test_rows)),
                "official_prompt_overlap": int(len(prompt_overlap)),
                "official_id_overlap": int(len(id_overlap)),
                "official_prompt_overlap_rate_vs_train": float(len(prompt_overlap) / max(1, len(train_rows))),
                "official_prompt_overlap_rate_vs_test": float(len(prompt_overlap) / max(1, len(test_rows))),
                "fixed_test_n_after_removing_train_prompts": int(len(fixed_test_rows)),
                "fixed_prompt_overlap": int(len({row["prompt"] for row in train_rows} & {row["prompt"] for row in fixed_test_rows})),
            }
        )
    return rows


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*", category=UserWarning)
        clf.fit(x_train, y_train)
    return clf


def _evaluate_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    clf = _fit_probe(x_train, y_train)
    scores = clf.decision_function(x_test)
    return binary_classification_metrics(y_test, scores)


def _evaluate_fixed_split_model(spec: dict[str, Any], dataset: str, layer: int) -> dict[str, Any]:
    path = Path(spec["hidden_root"]) / f"{dataset}__layer_{layer:02d}__mean_pool.parquet"
    metadata, matrix = load_hidden_state_table(path)
    train_mask = metadata["split"].astype(str).eq("train").to_numpy()
    test_mask = metadata["split"].astype(str).eq("test").to_numpy()
    train_df = metadata.loc[train_mask].reset_index(drop=True)
    test_df = metadata.loc[test_mask].reset_index(drop=True)
    x_train = np.asarray(matrix[train_mask], dtype=np.float32)
    x_test = np.asarray(matrix[test_mask], dtype=np.float32)
    y_train = train_df["label_ambiguous"].to_numpy(dtype=int)
    y_test = test_df["label_ambiguous"].to_numpy(dtype=int)

    pair_overlap = 0
    example_overlap = 0
    if "pair_id" in metadata.columns:
        pair_overlap = len(set(train_df["pair_id"].astype(str)) & set(test_df["pair_id"].astype(str)))
    if "example_id" in metadata.columns:
        example_overlap = len(set(train_df["example_id"].astype(str)) & set(test_df["example_id"].astype(str)))

    super_neurons = list(spec["super_neurons"])
    full_metrics = _evaluate_probe(x_train, y_train, x_test, y_test)
    super_metrics = _evaluate_probe(x_train[:, super_neurons], y_train, x_test[:, super_neurons], y_test)

    paper_full = PAPER_TABLE_1[(dataset, spec["label"])]
    paper_aen_acc, paper_aen_f1 = PAPER_TABLE_8[(dataset, "Ambiguity-Encoding Neurons only", spec["label"])]

    return {
        "model": spec["slug"],
        "model_label": spec["label"],
        "dataset": dataset,
        "layer": int(layer),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
        "pair_overlap": int(pair_overlap),
        "example_overlap": int(example_overlap),
        "official_super_neurons": super_neurons,
        "fixed_full_accuracy": float(full_metrics["accuracy"]),
        "fixed_full_auroc": float(full_metrics["auroc"]),
        "fixed_full_macro_f1": float(full_metrics["macro_f1"]),
        "fixed_full_confusion_matrix": full_metrics["confusion_matrix"],
        "fixed_official_super_accuracy": float(super_metrics["accuracy"]),
        "fixed_official_super_auroc": float(super_metrics["auroc"]),
        "fixed_official_super_macro_f1": float(super_metrics["macro_f1"]),
        "fixed_official_super_confusion_matrix": super_metrics["confusion_matrix"],
        "paper_full_accuracy": float(paper_full["accuracy"]) / 100.0,
        "paper_full_f1": float(paper_full["f1"]) / 100.0,
        "paper_aen_accuracy": float(paper_aen_acc) / 100.0,
        "paper_aen_f1": float(paper_aen_f1) / 100.0,
        "fixed_full_gap_to_paper_accuracy": float(full_metrics["accuracy"] - float(paper_full["accuracy"]) / 100.0),
        "fixed_aen_gap_to_paper_accuracy": float(super_metrics["accuracy"] - float(paper_aen_acc) / 100.0),
    }


def _render_report(overlap_df: pd.DataFrame, fixed_df: pd.DataFrame, output_root: Path) -> str:
    lines = [
        "# Official Split Leakage And Fixed-Split AEN Check",
        "",
        "## Official Split Audit",
        "",
        "The public code uses the same class file for train and test, shuffles with different seeds, uses the first train slice and the last test slice. Exact prompt overlap is therefore possible.",
        "",
        "| Dataset | Class | Available | Train | Test | Prompt overlap | ID overlap | Fixed test after removing train prompts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in overlap_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['class']} | {int(row['available'])} | "
            f"{int(row['official_train_n'])} | {int(row['official_test_n'])} | "
            f"{int(row['official_prompt_overlap'])} | {int(row['official_id_overlap'])} | "
            f"{int(row['fixed_test_n_after_removing_train_prompts'])} |"
        )

    lines.extend(
        [
            "",
            "## Leakage-Free Cached Split Evaluation",
            "",
            "This uses our cached split with zero `pair_id`/`example_id` overlap and the paper's hard-coded AENs. It fixes the split leakage, but it is not a bit-for-bit rerun of the official extractor because our cached hidden states use the repo pipeline's extraction settings.",
            "",
            "| Model | Dataset | Overlap pair/example | Paper full acc | Fixed full acc | Gap | Paper AEN acc | Fixed official-AEN acc | Gap | Official AENs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in fixed_df.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['dataset']} | {int(row['pair_overlap'])}/{int(row['example_overlap'])} | "
            f"{row['paper_full_accuracy']:.4f} | {row['fixed_full_accuracy']:.4f} | {row['fixed_full_gap_to_paper_accuracy']:+.4f} | "
            f"{row['paper_aen_accuracy']:.4f} | {row['fixed_official_super_accuracy']:.4f} | {row['fixed_aen_gap_to_paper_accuracy']:+.4f} | "
            f"`{row['official_super_neurons']}` |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Yes: the official within-dataset split has train/test overlap.",
            "- After enforcing a no-overlap split in the cached pipeline, the results do not match the paper numbers; they are consistently lower, especially for the hard-coded AEN-only probes.",
            "- Because this fixed-split check uses cached hidden states rather than re-running the exact official extractor, the drop is attributable to the corrected split plus extraction/data-processing differences. A bit-for-bit fixed official rerun would require regenerating author-style hidden states with the corrected split.",
            "",
            f"- Overlap rows: `{output_root / 'official_split_overlap.parquet'}`",
            f"- Fixed-split metrics: `{output_root / 'fixed_split_official_aen_metrics.parquet'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    author_repo_root = Path(args.author_repo_root)

    overlap_rows: list[dict[str, Any]] = []
    for dataset in DATASET_FILES:
        overlap_rows.extend(
            _official_split_overlap(
                author_repo_root=author_repo_root,
                dataset=dataset,
                train_per_class=int(args.train_per_class),
                test_per_class=int(args.test_per_class),
            )
        )

    fixed_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for dataset in DATASET_FILES:
            fixed_rows.append(_evaluate_fixed_split_model(spec, dataset, int(args.layer)))

    overlap_df = pd.DataFrame(overlap_rows).sort_values(["dataset", "class"]).reset_index(drop=True)
    fixed_df = pd.DataFrame(fixed_rows).sort_values(["model_label", "dataset"]).reset_index(drop=True)

    write_parquet(overlap_df, output_root / "official_split_overlap.parquet")
    write_parquet(fixed_df, output_root / "fixed_split_official_aen_metrics.parquet")
    write_markdown(
        output_root / "official_split_leakage_fixed_eval_report.md",
        _render_report(overlap_df, fixed_df, output_root),
    )


if __name__ == "__main__":
    main()
