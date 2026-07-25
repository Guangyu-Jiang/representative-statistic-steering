#!/usr/bin/env python3
"""Aggregate FalseQA topology classification results across local LLMs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet


MODEL_NAMES = {
    "meta_llama_llama_3_1_8b_instruct": "LLaMA-3.1-8B-Instruct",
    "google_gemma_7b_it": "Gemma-7B-it",
    "mistralai_mistral_7b_instruct_v0_3": "Mistral-7B-Instruct-v0.3",
}
PRIMARY_FEATURES = (
    "length_controls",
    "h0_mean_last_layer",
    "h0_three_last_layer",
    "h0_mean_all_layers",
    "h0_three_all_layers",
    "tfidf_word_unigram_bigram",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/falseqa_topology_classification")
    parser.add_argument("--output-dir", default="artifacts/reports/falseqa_topology_classification")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load(artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for slug, model_name in MODEL_NAMES.items():
        model_root = artifact_root / slug
        metrics = pd.read_parquet(model_root / "classification_metrics.parquet")
        predictions = pd.read_parquet(model_root / "test_predictions.parquet")
        metrics.insert(0, "model", model_name)
        predictions.insert(0, "model", model_name)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    return (
        pd.concat(metric_frames, ignore_index=True, sort=False),
        pd.concat(prediction_frames, ignore_index=True, sort=False),
    )


def _bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pair_ids = frame["pair_id"].astype(str).unique()
    grouped = {pair_id: group.index.to_numpy() for pair_id, group in frame.groupby("pair_id")}
    accuracy_values: list[float] = []
    auroc_values: list[float] = []
    for _ in range(repeats):
        sampled_pairs = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled_indices = np.concatenate([grouped[str(pair_id)] for pair_id in sampled_pairs])
        sampled = frame.loc[sampled_indices]
        accuracy_values.append(float(sampled["correct"].mean()))
        if sampled["label"].nunique() == 2:
            auroc_values.append(float(roc_auc_score(sampled["label"], sampled["score"])))
    accuracy_low, accuracy_high = np.quantile(accuracy_values, [0.025, 0.975])
    auroc_low, auroc_high = np.quantile(auroc_values, [0.025, 0.975])
    return {
        "accuracy_ci_low": float(accuracy_low),
        "accuracy_ci_high": float(accuracy_high),
        "auroc_ci_low": float(auroc_low),
        "auroc_ci_high": float(auroc_high),
    }


def _matched_comparison(
    predictions: pd.DataFrame,
    *,
    model: str,
    protocol: str,
    task: str,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    keys = ["pair_id", "example_id"]
    candidate_frame = predictions.loc[
        predictions["model"].eq(model)
        & predictions["protocol"].eq(protocol)
        & predictions["task"].eq(task)
        & predictions["feature_set"].eq(candidate),
        keys + ["correct"],
    ].rename(columns={"correct": "candidate_correct"})
    baseline_frame = predictions.loc[
        predictions["model"].eq(model)
        & predictions["protocol"].eq(protocol)
        & predictions["task"].eq(task)
        & predictions["feature_set"].eq(baseline),
        keys + ["correct"],
    ].rename(columns={"correct": "baseline_correct"})
    merged = candidate_frame.merge(baseline_frame, on=keys, how="inner", validate="one_to_one")
    candidate_only = int((merged["candidate_correct"] & ~merged["baseline_correct"]).sum())
    baseline_only = int((~merged["candidate_correct"] & merged["baseline_correct"]).sum())
    discordant = candidate_only + baseline_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(candidate_only, n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "model": model,
        "protocol": protocol,
        "task": task,
        "candidate": candidate,
        "baseline": baseline,
        "n": int(len(merged)),
        "candidate_accuracy": float(merged["candidate_correct"].mean()),
        "baseline_accuracy": float(merged["baseline_correct"].mean()),
        "accuracy_delta": float(merged["candidate_correct"].mean() - merged["baseline_correct"].mean()),
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "mcnemar_exact_p": p_value,
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    metrics, predictions = _load(artifact_root)
    write_parquet(metrics, output_dir / "all_metrics.parquet")
    metrics.to_csv(output_dir / "all_metrics.csv", index=False)

    primary = metrics.loc[metrics["feature_set"].isin(PRIMARY_FEATURES)].copy()
    interval_rows: list[dict[str, Any]] = []
    for row in primary.to_dict(orient="records"):
        subset = predictions.loc[
            predictions["model"].eq(row["model"])
            & predictions["protocol"].eq(row["protocol"])
            & predictions["task"].eq(row["task"])
            & predictions["feature_set"].eq(row["feature_set"])
        ]
        intervals = _bootstrap_intervals(
            subset,
            repeats=int(args.bootstrap_repeats),
            seed=int(args.seed),
        )
        interval_rows.append({**row, **intervals})
    primary_intervals = pd.DataFrame(interval_rows)
    write_parquet(primary_intervals, output_dir / "primary_metrics_with_pair_bootstrap_ci.parquet")
    primary_intervals.to_csv(output_dir / "primary_metrics_with_pair_bootstrap_ci.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES.values():
        for protocol in ("random80", "official"):
            for task in ("ordinary", "paired_orientation"):
                for baseline in ("tfidf_word_unigram_bigram", "length_controls", "h0_three_last_layer"):
                    comparison_rows.append(
                        _matched_comparison(
                            predictions,
                            model=model,
                            protocol=protocol,
                            task=task,
                            candidate="h0_three_all_layers",
                            baseline=baseline,
                        )
                    )
    comparisons = pd.DataFrame(comparison_rows)
    write_parquet(comparisons, output_dir / "matched_comparisons.parquet")
    comparisons.to_csv(output_dir / "matched_comparisons.csv", index=False)

    focus = primary_intervals.loc[
        primary_intervals["protocol"].eq("random80")
        & primary_intervals["task"].eq("paired_orientation")
        & primary_intervals["feature_set"].isin(
            ["h0_three_all_layers", "h0_mean_all_layers", "h0_three_last_layer", "length_controls", "tfidf_word_unigram_bigram"]
        )
    ].copy()
    focus["accuracy (95% CI)"] = focus.apply(
        lambda row: f"{_format_percent(row['test_accuracy'])} [{_format_percent(row['accuracy_ci_low'])}, {_format_percent(row['accuracy_ci_high'])}]",
        axis=1,
    )
    focus["AUROC (95% CI)"] = focus.apply(
        lambda row: f"{row['test_auroc']:.3f} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]",
        axis=1,
    )
    focus_table = focus.loc[:, ["model", "feature_set", "n_test", "accuracy (95% CI)", "AUROC (95% CI)"]]
    focus_table = focus_table.sort_values(["model", "feature_set"])

    tfidf_comparison = comparisons.loc[
        comparisons["protocol"].eq("random80")
        & comparisons["task"].eq("paired_orientation")
        & comparisons["baseline"].eq("tfidf_word_unigram_bigram")
    ].copy()
    tfidf_comparison["H0-three"] = tfidf_comparison["candidate_accuracy"].map(_format_percent)
    tfidf_comparison["TF-IDF"] = tfidf_comparison["baseline_accuracy"].map(_format_percent)
    tfidf_comparison["delta"] = tfidf_comparison["accuracy_delta"].map(
        lambda value: f"{100.0 * value:+.1f} pp"
    )
    tfidf_table = tfidf_comparison.loc[:, ["model", "H0-three", "TF-IDF", "delta", "mcnemar_exact_p"]]

    report = (
        "# FalseQA False-Premise Topology Probe\n\n"
        "The primary leakage-controlled task randomly orders each false/corrected question pair and predicts "
        "whether the first question contains the false premise from the signed topology-vector difference. "
        "All PCA reducers and scalers are fit on training questions only. Confidence intervals resample pairs.\n\n"
        "## Random 80/20 paired results\n\n"
        f"{focus_table.to_markdown(index=False)}\n\n"
        "## Matched comparison with lexical control\n\n"
        f"{tfidf_table.to_markdown(index=False)}\n\n"
        "The single-layer rows are diagnostic. The all-layer feature sets were specified before evaluation and "
        "are the appropriate primary comparison.\n"
    )
    write_markdown(output_dir / "comparison.md", report)
    print(report)


if __name__ == "__main__":
    main()
