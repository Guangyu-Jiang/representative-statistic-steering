#!/usr/bin/env python3
"""Aggregate locally judged FalseQA global and topology-local steering results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from aen_replication.utils.io_utils import ensure_dir, write_parquet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/falseqa_topology_steering_aligned")
    parser.add_argument("--output-root", default="artifacts/reports/falseqa_topology_steering")
    parser.add_argument("--protocol", default="random80")
    parser.add_argument("--expected-n", type=int, default=473)
    parser.add_argument("--min-valid-pct", type=float, default=95.0)
    parser.add_argument("--min-unique-pct", type=float, default=90.0)
    return parser.parse_args()


def _method_name(strategy: str) -> str:
    return {
        "base": "Unsteered",
        "global_paired_mean_diff_raw": "Global paired mean-diff",
        "topology_local_paired_mean_diff_raw": "Topology-local paired mean-diff",
    }.get(str(strategy), str(strategy))


def _subset_metrics(
    frame: pd.DataFrame,
    *,
    example_ids: set[str],
    base_labels: pd.Series,
) -> dict[str, float | int]:
    subset = frame.loc[frame["example_id"].astype(str).isin(example_ids)].copy()
    labels = subset.set_index(subset["example_id"].astype(str))["judge_label"]
    llm_labels = subset.get("judge_label_llm", subset["judge_label"])
    nli_scores = pd.to_numeric(
        subset.get("nli_max_entailment", pd.Series(float("nan"), index=subset.index)),
        errors="coerce",
    )
    nli_threshold = float(
        subset.get(
            "nli_entailment_threshold",
            pd.Series(0.8, index=subset.index),
        ).iloc[0]
    )
    aligned_base = base_labels.reindex(labels.index)
    base_premise = aligned_base.eq("PREMISE_ACCEPTANCE")
    base_grounded = aligned_base.eq("GROUNDED_REBUTTAL")
    unique = int(subset["response_text"].astype(str).nunique())
    return {
        "n": int(len(subset)),
        "grounded_rebuttal_pct": float(labels.eq("GROUNDED_REBUTTAL").mean() * 100.0),
        "llm_grounded_rebuttal_pct": float(
            llm_labels.eq("GROUNDED_REBUTTAL").mean() * 100.0
        ),
        "nli_supported_grounded_pct": float(
            (llm_labels.eq("GROUNDED_REBUTTAL") & nli_scores.ge(nli_threshold)).mean()
            * 100.0
        ),
        "mean_nli_max_entailment": float(nli_scores.mean()),
        "generic_rejection_pct": float(labels.eq("GENERIC_REJECTION").mean() * 100.0),
        "premise_acceptance_pct": float(labels.eq("PREMISE_ACCEPTANCE").mean() * 100.0),
        "neither_pct": float(labels.eq("NEITHER").mean() * 100.0),
        "valid_response_pct": float(subset["response_valid"].mean() * 100.0),
        "mean_generated_token_count": float(subset["generated_token_count"].mean()),
        "mean_response_word_count": float(subset["response_word_count"].mean()),
        "unique_response_pct": float(unique / max(len(subset), 1) * 100.0),
        "base_premise_acceptance_n": int(base_premise.sum()),
        "premise_to_grounded_pct": float(
            labels.loc[base_premise].eq("GROUNDED_REBUTTAL").mean() * 100.0
        )
        if base_premise.any()
        else float("nan"),
        "base_grounded_retention_pct": float(
            labels.loc[base_grounded].eq("GROUNDED_REBUTTAL").mean() * 100.0
        )
        if base_grounded.any()
        else float("nan"),
        "mean_delta_h_fro_over_h_fro": float(subset["delta_h_fro_over_h_fro"].mean()),
    }


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    output_root = ensure_dir(Path(args.output_root).resolve())
    rows: list[pd.DataFrame] = []
    for summary_path in sorted(artifact_root.glob(f"*/{args.protocol}/local_judge_summary.csv")):
        frame = pd.read_csv(summary_path)
        if frame.empty or not frame["n"].eq(int(args.expected_n)).all():
            continue
        frame.insert(0, "model", summary_path.parents[1].name)
        frame.insert(1, "protocol", str(args.protocol))
        frame.insert(2, "method", frame["strategy"].map(_method_name))
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(
            f"No complete n={int(args.expected_n)} local-judge summaries under {artifact_root}"
        )

    comprehensive = pd.concat(rows, ignore_index=True)
    comprehensive = comprehensive.sort_values(["model", "strategy", "neighbor_k", "alpha"])
    write_parquet(comprehensive, output_root / "falseqa_steering_comprehensive.parquet")
    comprehensive.to_csv(output_root / "falseqa_steering_comprehensive.csv", index=False)

    eligible = comprehensive.loc[
        comprehensive["valid_response_pct"].ge(float(args.min_valid_pct))
        & comprehensive["unique_response_pct"].ge(float(args.min_unique_pct))
    ].copy()
    best_rows: list[pd.Series] = []
    for (_model, _method), group in eligible.groupby(["model", "method"], sort=True):
        ranked = group.sort_values(
            [
                "grounded_rebuttal_pct",
                "premise_to_grounded_pct",
                "base_grounded_retention_pct",
                "neither_pct",
                "mean_delta_h_fro_over_h_fro",
            ],
            ascending=[False, False, False, True, True],
        )
        best_rows.append(ranked.iloc[0])
    best = pd.DataFrame(best_rows).sort_values(["model", "method"])
    write_parquet(best, output_root / "falseqa_steering_best.parquet")
    best.to_csv(output_root / "falseqa_steering_best.csv", index=False)

    matched_rows: list[dict[str, object]] = []
    for model, model_frame in eligible.groupby("model", sort=True):
        global_rows = model_frame.loc[model_frame["strategy"].eq("global_paired_mean_diff_raw")]
        local_rows = model_frame.loc[
            model_frame["strategy"].eq("topology_local_paired_mean_diff_raw")
        ]
        if global_rows.empty:
            continue
        global_norms = global_rows["mean_delta_h_fro_over_h_fro"].to_numpy(dtype=float)
        for _, local in local_rows.iterrows():
            local_norm = float(local["mean_delta_h_fro_over_h_fro"])
            match_index = int(abs(global_norms - local_norm).argmin())
            global_row = global_rows.iloc[match_index]
            matched_rows.append(
                {
                    "model": model,
                    "local_neighbor_k": int(local["neighbor_k"]),
                    "local_alpha": float(local["alpha"]),
                    "local_delta_h_over_h": local_norm,
                    "local_grounded_rebuttal_pct": float(local["grounded_rebuttal_pct"]),
                    "local_premise_to_grounded_pct": float(local["premise_to_grounded_pct"]),
                    "global_alpha": float(global_row["alpha"]),
                    "global_delta_h_over_h": float(global_row["mean_delta_h_fro_over_h_fro"]),
                    "global_grounded_rebuttal_pct": float(global_row["grounded_rebuttal_pct"]),
                    "global_premise_to_grounded_pct": float(global_row["premise_to_grounded_pct"]),
                    "relative_norm_gap": abs(
                        local_norm - float(global_row["mean_delta_h_fro_over_h_fro"])
                    ),
                }
            )
    matched = pd.DataFrame(matched_rows).sort_values(
        ["model", "local_neighbor_k", "local_alpha"]
    )
    write_parquet(matched, output_root / "falseqa_steering_matched_strength.parquet")
    matched.to_csv(output_root / "falseqa_steering_matched_strength.csv", index=False)

    nested_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for model_root in sorted(artifact_root.glob(f"*/{args.protocol}")):
        pilot_path = model_root / "pilot64_example_ids.parquet"
        pilot_summary_path = model_root / "pilot64_local_judge_summary.csv"
        raw_root = model_root / "raw"
        if not pilot_path.exists() or not pilot_summary_path.exists():
            continue
        judged_paths = sorted(raw_root.glob("*__judged.parquet"))
        judged_frames = [pd.read_parquet(path) for path in judged_paths]
        judged_frames = [frame for frame in judged_frames if len(frame) == int(args.expected_n)]
        if not judged_frames:
            continue
        base_candidates = [frame for frame in judged_frames if frame.iloc[0]["strategy"] == "base"]
        if len(base_candidates) != 1:
            raise ValueError(f"Expected one complete base frame under {model_root}")
        base_frame = base_candidates[0]
        all_ids = set(base_frame["example_id"].astype(str))
        validation_ids = set(pd.read_parquet(pilot_path)["example_id"].astype(str))
        final_ids = all_ids.difference(validation_ids)
        base_labels = base_frame.set_index(base_frame["example_id"].astype(str))["judge_label"]

        frame_by_key: dict[tuple[str, int, float], pd.DataFrame] = {}
        for frame in judged_frames:
            first = frame.iloc[0]
            key = (str(first["strategy"]), int(first["neighbor_k"]), float(first["alpha"]))
            frame_by_key[key] = frame
        # Select hyperparameters only from the frozen pilot run. Full-run responses
        # for the same IDs can differ because stochastic decoding uses new batches.
        validation = pd.read_csv(pilot_summary_path)
        if not validation["n"].eq(len(validation_ids)).all():
            raise ValueError(
                f"Pilot summary row counts do not match {len(validation_ids)} IDs under {model_root}"
            )
        validation["method"] = validation["strategy"].map(_method_name)
        validation = validation.loc[
            validation["valid_response_pct"].ge(float(args.min_valid_pct))
            & validation["unique_response_pct"].ge(float(args.min_unique_pct))
        ]
        selected_frames: dict[str, pd.DataFrame] = {}
        for method, group in validation.groupby("method", sort=True):
            selected = group.sort_values(
                [
                    "grounded_rebuttal_pct",
                    "premise_to_grounded_pct",
                    "base_grounded_retention_pct",
                    "neither_pct",
                    "mean_delta_h_fro_over_h_fro",
                ],
                ascending=[False, False, False, True, True],
            ).iloc[0]
            key = (str(selected["strategy"]), int(selected["neighbor_k"]), float(selected["alpha"]))
            if key not in frame_by_key:
                raise ValueError(f"Pilot-selected setting {key} is absent from the full run")
            final_metrics = _subset_metrics(
                frame_by_key[key],
                example_ids=final_ids,
                base_labels=base_labels,
            )
            row: dict[str, object] = {
                "model": model_root.parent.name,
                "method": method,
                "strategy": key[0],
                "neighbor_k": key[1],
                "alpha": key[2],
            }
            row.update({f"validation_{name}": value for name, value in selected.items() if name not in row})
            row.update({f"final_{name}": value for name, value in final_metrics.items()})
            nested_rows.append(row)
            selected_frames[method] = frame_by_key[key]
        global_name = "Global paired mean-diff"
        local_name = "Topology-local paired mean-diff"
        if global_name in selected_frames and local_name in selected_frames:
            global_labels = (
                selected_frames[global_name]
                .set_index(selected_frames[global_name]["example_id"].astype(str))["judge_label"]
                .reindex(sorted(final_ids))
            )
            local_labels = (
                selected_frames[local_name]
                .set_index(selected_frames[local_name]["example_id"].astype(str))["judge_label"]
                .reindex(global_labels.index)
            )
            global_grounded = global_labels.eq("GROUNDED_REBUTTAL").to_numpy(dtype=float)
            local_grounded = local_labels.eq("GROUNDED_REBUTTAL").to_numpy(dtype=float)
            local_only = int(((local_grounded == 1) & (global_grounded == 0)).sum())
            global_only = int(((local_grounded == 0) & (global_grounded == 1)).sum())
            discordant = local_only + global_only
            rng = np.random.default_rng(0)
            differences = local_grounded - global_grounded
            bootstrap_indices = rng.integers(0, len(differences), size=(5000, len(differences)))
            bootstrap_delta = differences[bootstrap_indices].mean(axis=1) * 100.0
            paired_rows.append(
                {
                    "model": model_root.parent.name,
                    "n_final": int(len(differences)),
                    "topology_minus_global_grounded_pp": float(differences.mean() * 100.0),
                    "delta_ci95_low": float(np.quantile(bootstrap_delta, 0.025)),
                    "delta_ci95_high": float(np.quantile(bootstrap_delta, 0.975)),
                    "topology_grounded_global_not_count": local_only,
                    "global_grounded_topology_not_count": global_only,
                    "mcnemar_exact_p": float(
                        binomtest(local_only, n=discordant, p=0.5).pvalue
                    )
                    if discordant
                    else 1.0,
                }
            )
    if nested_rows:
        nested = pd.DataFrame(nested_rows).sort_values(["model", "method"])
        write_parquet(nested, output_root / "falseqa_steering_nested_selection.parquet")
        nested.to_csv(output_root / "falseqa_steering_nested_selection.csv", index=False)
    if paired_rows:
        paired = pd.DataFrame(paired_rows).sort_values("model")
        write_parquet(paired, output_root / "falseqa_steering_nested_paired_comparison.parquet")
        paired.to_csv(output_root / "falseqa_steering_nested_paired_comparison.csv", index=False)

    columns = [
        "model",
        "method",
        "neighbor_k",
        "alpha",
        "n",
        "grounded_rebuttal_pct",
        "llm_grounded_rebuttal_pct",
        "nli_supported_grounded_pct",
        "mean_nli_max_entailment",
        "generic_rejection_pct",
        "premise_acceptance_pct",
        "neither_pct",
        "premise_to_grounded_pct",
        "base_grounded_retention_pct",
        "valid_response_pct",
        "unique_response_pct",
        "mean_generated_token_count",
        "mean_delta_h_fro_over_h_fro",
    ]
    printable = best.loc[:, columns].copy()
    printable.to_markdown(output_root / "falseqa_steering_best.md", index=False, floatfmt=".3f")
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
