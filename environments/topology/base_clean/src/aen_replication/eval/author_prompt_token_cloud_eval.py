"""Evaluate token-cloud topology classifiers on the author's public prompts.

This script keeps the token-cloud classifier unchanged and only swaps the input
text source to the first author's public prompt files and classwise shuffle
logic, so we can measure how much of the token-cloud gap is due to prompt
formatting / split construction.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from aen_replication.eval.author_repo_eval import (
    MODEL_SPECS,
    _load_author_prompts,
    _load_best_token_cloud_metrics,
    _shuffle_and_slice_author_style,
)
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.token_cloud_topology_classifier import (
    _distance_feature_mode,
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prototype_diagrams_from_clouds,
    _resolve_candidate_layers,
    build_token_cloud_feature_frame,
    run_token_cloud_topology_classifier_from_features,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet
from aen_replication.utils.seed import set_global_seed


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama31_8b_instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "local_files_only": True,
        "use_fast": False,
        "model_class": "causal_lm",
    },
    "mistral_7b_instruct_v03": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "local_files_only": True,
        "use_fast": False,
        "model_class": "causal_lm",
    },
    "gemma_7b_it": {
        "name": "google/gemma-7b-it",
        "tokenizer_name": "google/gemma-7b-it",
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "local_files_only": True,
        "use_fast": True,
        "model_class": "causal_lm",
    },
}


TOKEN_CLOUD_CONFIG: dict[str, Any] = {
    "output_dir": "artifacts/token_cloud_topology_author_prompts",
    "datasets": ["ambigqa", "situatedqa"],
    "text_column": "text",
    "batch_size": 8,
    "max_length": 64,
    "parallel_jobs": 16,
    "candidate_layers": [0, 14, 31],
    "layer_selection_strategy": "evenly_spaced",
    "max_candidate_layers": 3,
    "drop_special_tokens": True,
    "pca_fit_token_cap": 16000,
    "pca_components": 8,
    "pca_whiten": False,
    "topology_components": 6,
    "prototype_token_cap": 192,
    "distance_metric": "euclidean",
    "betti_grid_size": 24,
    "persistence_image_grid_side": 3,
    "maxdim": 1,
    "coeff": 2,
    "val_fraction": 0.2,
    "multilayer_enabled": True,
    "multilayer_top_k": 2,
    "local_files_only": True,
    "feature_table_filename": "token_cloud_topology_features.parquet",
    "candidate_metrics_filename": "token_cloud_topology_candidate_metrics.parquet",
    "final_metrics_filename": "token_cloud_topology_final_metrics.parquet",
    "selected_candidates_filename": "token_cloud_topology_selected_candidates.parquet",
    "report_filename": "token_cloud_topology_summary.md",
    "metadata_filename": "token_cloud_topology_metadata.json",
    "classifier": {
        "penalty": "l2",
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 4000,
        "standardize": True,
    },
}


def _build_author_prompt_frame(
    *,
    author_repo_root: Path,
    dataset_name: str,
    prompt_model_name: str,
    train_per_class: int,
    test_per_class: int,
) -> pd.DataFrame:
    ambig_prompts, clear_prompts = _load_author_prompts(
        author_repo_root=author_repo_root,
        dataset_name=dataset_name,
        prompt_model_name=prompt_model_name,
    )
    split_payload = _shuffle_and_slice_author_style(
        ambig_prompts,
        clear_prompts,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
    )
    rows: list[dict[str, Any]] = []
    for split_name, prompts, label_ambiguous in [
        ("train", split_payload["train_ambig"], 1),
        ("train", split_payload["train_clear"], 0),
        ("test", split_payload["test_ambig"], 1),
        ("test", split_payload["test_clear"], 0),
    ]:
        label_name = "ambiguous" if label_ambiguous else "clear"
        for index, prompt in enumerate(prompts):
            example_id = f"{dataset_name}__{split_name}__{label_name}__{index:04d}"
            rows.append(
                {
                    "example_id": example_id,
                    "pair_id": example_id,
                    "dataset": dataset_name,
                    "split": split_name,
                    "label_ambiguous": label_ambiguous,
                    "text": prompt,
                }
            )
    return pd.DataFrame(rows)


def _load_author_full_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(summary_path)
    return df.loc[:, ["model", "dataset", "author_full_accuracy", "author_full_auroc", "author_super_accuracy", "author_super_auroc"]]


def run_eval(
    *,
    author_repo_root: Path,
    output_root: Path,
    raw_token_cloud_root: Path,
    author_full_summary_path: Path,
    seed: int,
    model_keys: list[str],
    dataset_names: list[str],
    train_per_class: int,
    test_per_class: int,
) -> dict[str, str]:
    output_root = ensure_dir(output_root)
    author_full_df = _load_author_full_summary(author_full_summary_path)
    summary_rows: list[dict[str, Any]] = []
    set_global_seed(seed)

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        model_config = MODEL_CONFIGS[model_key]
        classifier_config = {**TOKEN_CLOUD_CONFIG, "datasets": dataset_names, "output_dir": str(output_root)}
        bundle = load_hf_model(model_config, classifier_config)
        try:
            total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
            layers = _resolve_candidate_layers(total_layers, classifier_config)
            dataset_frames = [
                _build_author_prompt_frame(
                    author_repo_root=author_repo_root,
                    dataset_name=dataset_name,
                    prompt_model_name=spec.prompt_model_name,
                    train_per_class=train_per_class,
                    test_per_class=test_per_class,
                )
                for dataset_name in dataset_names
            ]
            full_df = pd.concat(dataset_frames, ignore_index=True)
            train_df = full_df.loc[full_df["split"].eq("train")].copy().reset_index(drop=True)

            token_matrices = _extract_train_token_matrices(
                bundle=bundle,
                train_df=train_df,
                text_column="text",
                layers=layers,
                config={**classifier_config, "_seed": seed},
            )
            reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
            cloud_df = _extract_reduced_clouds(
                bundle=bundle,
                df=full_df,
                text_column="text",
                layers=layers,
                reducers=reducers,
                config=classifier_config,
            )
            prototype_map = None
            if _distance_feature_mode(classifier_config) == "prototype":
                prototype_map = _prototype_diagrams_from_clouds(cloud_df, layers=layers, config=classifier_config, seed=seed)
            feature_df = build_token_cloud_feature_frame(cloud_df, prototype_map=prototype_map, config=classifier_config)
            artifact_map = run_token_cloud_topology_classifier_from_features(
                model_name=model_config["name"],
                feature_df=feature_df,
                classifier_config=classifier_config,
                seed=seed,
            )

            final_df = pd.read_parquet(artifact_map["final_metrics_path"])
            for dataset_name in dataset_names:
                subset = final_df.loc[final_df["dataset"].eq(dataset_name)].copy()
                if subset.empty:
                    continue
                best_row = subset.sort_values(["test_auroc", "test_accuracy"], ascending=[False, False]).iloc[0]
                raw_metrics = _load_best_token_cloud_metrics(raw_token_cloud_root, spec.output_dir_name, dataset_name)
                author_metrics = author_full_df.loc[
                    author_full_df["model"].eq(spec.label) & author_full_df["dataset"].eq(dataset_name)
                ].iloc[0]
                summary_rows.append(
                    {
                        "model": spec.label,
                        "dataset": dataset_name,
                        "author_prompt_feature_set": str(best_row["feature_set"]),
                        "author_prompt_selection_mode": str(best_row["selection_mode"]),
                        "author_prompt_selection_signature": str(best_row["selection_signature"]),
                        "author_prompt_accuracy": float(best_row["test_accuracy"]),
                        "author_prompt_auroc": float(best_row["test_auroc"]),
                        "raw_token_cloud_feature_set": None if raw_metrics is None else raw_metrics["feature_set"],
                        "raw_token_cloud_accuracy": None if raw_metrics is None else float(raw_metrics["accuracy"]),
                        "raw_token_cloud_auroc": None if raw_metrics is None else float(raw_metrics["auroc"]),
                        "author_prompt_minus_raw_acc": None
                        if raw_metrics is None
                        else float(best_row["test_accuracy"]) - float(raw_metrics["accuracy"]),
                        "author_prompt_minus_raw_auroc": None
                        if raw_metrics is None
                        else float(best_row["test_auroc"]) - float(raw_metrics["auroc"]),
                        "official_full_accuracy": float(author_metrics["author_full_accuracy"]),
                        "official_full_auroc": float(author_metrics["author_full_auroc"]),
                        "official_super_accuracy": float(author_metrics["author_super_accuracy"]),
                        "official_super_auroc": float(author_metrics["author_super_auroc"]),
                        "official_full_minus_author_prompt_acc": float(author_metrics["author_full_accuracy"])
                        - float(best_row["test_accuracy"]),
                        "official_full_minus_author_prompt_auroc": float(author_metrics["author_full_auroc"])
                        - float(best_row["test_auroc"]),
                        "train_per_class": int(train_per_class),
                        "test_per_class": int(test_per_class),
                    }
                )
        finally:
            model = bundle.model
            tokenizer = bundle.tokenizer
            del model, tokenizer, bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    summary_path = output_root / "author_prompt_token_cloud_summary.parquet"
    report_path = output_root / "author_prompt_token_cloud_summary.md"
    metadata_path = output_root / "author_prompt_token_cloud_metadata.json"
    write_parquet(summary_df, summary_path)

    lines = [
        "# Author-Prompt Token-Cloud Evaluation",
        "",
        f"- Created at: `{utc_now_iso()}`",
        f"- Author repo: `{author_repo_root}`",
        f"- Train per class: `{train_per_class}`",
        f"- Test per class: `{test_per_class}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.extend(
            [
                f"### {row['model']} / {row['dataset']}",
                "",
                f"- Author-prompt token-cloud: `{row['author_prompt_feature_set']}` "
                f"(AUROC `{row['author_prompt_auroc']:.4f}`, accuracy `{row['author_prompt_accuracy']:.4f}`, "
                f"selection `{row['author_prompt_selection_signature']}`)",
                f"- Raw-question token-cloud: `{row['raw_token_cloud_feature_set']}` "
                f"(AUROC `{row['raw_token_cloud_auroc']:.4f}`, accuracy `{row['raw_token_cloud_accuracy']:.4f}`)",
                f"- Delta from raw-question token-cloud: accuracy `{row['author_prompt_minus_raw_acc']:+.4f}`, "
                f"AUROC `{row['author_prompt_minus_raw_auroc']:+.4f}`",
                f"- Official full probe: accuracy `{row['official_full_accuracy']:.4f}`, AUROC `{row['official_full_auroc']:.4f}`",
                f"- Official super-neuron probe: accuracy `{row['official_super_accuracy']:.4f}`, AUROC `{row['official_super_auroc']:.4f}`",
                "",
            ]
        )
    write_markdown(report_path, "\n".join(lines) + "\n")
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "output_artifacts": {
                "summary_parquet": str(summary_path),
                "report": str(report_path),
            },
        },
    )
    return {
        "summary_parquet": str(summary_path),
        "report": str(report_path),
        "metadata": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo-root", default="/home/ubuntu/Internal_State_Detect_Ambiguity")
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_author_prompts",
    )
    parser.add_argument(
        "--raw-token-cloud-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/token_cloud_topology_classifier_all_datasets",
    )
    parser.add_argument(
        "--author-full-summary-path",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/author_repo_eval/author_repo_eval_summary.parquet",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--parallel-jobs", type=int, default=16)
    parser.add_argument("--candidate-layers", nargs="*", type=int)
    parser.add_argument("--multilayer-enabled", choices=["true", "false"], default="true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama31_8b_instruct", "mistral_7b_instruct_v03", "gemma_7b_it"],
    )
    parser.add_argument("--datasets", nargs="*", default=["ambigqa", "situatedqa"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TOKEN_CLOUD_CONFIG["parallel_jobs"] = int(args.parallel_jobs)
    if args.candidate_layers:
        TOKEN_CLOUD_CONFIG["candidate_layers"] = [int(layer) for layer in args.candidate_layers]
    TOKEN_CLOUD_CONFIG["multilayer_enabled"] = args.multilayer_enabled == "true"
    run_eval(
        author_repo_root=Path(args.author_repo_root),
        output_root=Path(args.output_root),
        raw_token_cloud_root=Path(args.raw_token_cloud_root),
        author_full_summary_path=Path(args.author_full_summary_path),
        seed=int(args.seed),
        model_keys=list(args.models),
        dataset_names=list(args.datasets),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
    )


if __name__ == "__main__":
    main()
