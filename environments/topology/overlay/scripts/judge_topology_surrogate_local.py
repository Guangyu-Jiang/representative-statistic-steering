#!/usr/bin/env python3
"""Judge topology-surrogate steering raw outputs with a non-API judge.

This script is intentionally provider-guarded: it refuses to run if the loaded
config points at the OpenAI API. Use ``judge.provider: local_llm`` for a local
model judge or ``judge.provider: rules`` for a deterministic smoke check.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from aen_replication.config import load_config
from aen_replication.eval.judge import load_judge
from aen_replication.train.steering import _judge_table
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet


STRATEGIES = ("topology3_surrogate", "topology_steering")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runs/llama_steering_paper_style.yaml")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["ambigqa", "situatedqa", "clamber"])
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--lambdas", nargs="+", type=float, default=None)
    parser.add_argument("--judge-batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _slug_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _parse_run(path: Path) -> tuple[float | None, float | None]:
    match = re.search(r"alpha_([^/]+)__lambda_([^/]+)", str(path))
    if match:
        alpha = float(match.group(1).replace("p", ".").replace("m", "-"))
        lambda_value = float(match.group(2).replace("p", ".").replace("m", "-"))
        return alpha, lambda_value
    alpha = None
    lambda_value = None
    if "alpha" in path.parent.name:
        alpha_match = re.search(r"alpha[_=]?([^_/]+)", path.parent.name)
        if alpha_match:
            alpha = float(alpha_match.group(1).replace("p", ".").replace("m", "-"))
    return alpha, lambda_value


def _strategy_from_path(path: Path) -> str:
    for strategy in STRATEGIES:
        if f"__{strategy}__raw.parquet" in path.name:
            return strategy
    return "unknown"


def _discover_raw_paths(
    *,
    artifact_root: Path,
    model_slug: str,
    datasets: list[str],
    alphas: list[float] | None,
    lambdas: list[float] | None,
) -> list[Path]:
    alpha_set = {_slug_float(value) for value in alphas} if alphas is not None else None
    lambda_set = {_slug_float(value) for value in lambdas} if lambdas is not None else None
    paths: list[Path] = []
    for dataset in datasets:
        search_roots = [
            artifact_root / dataset / model_slug,
            artifact_root / model_slug,
            artifact_root / dataset,
            artifact_root,
        ]
        seen_roots: set[Path] = set()
        for root in search_roots:
            if root in seen_roots or not root.exists():
                continue
            seen_roots.add(root)
            for raw_path in sorted(root.rglob(f"{dataset}__*__raw.parquet")):
                if not any(f"__{strategy}__raw.parquet" in raw_path.name for strategy in STRATEGIES):
                    continue
                run_dir = raw_path.parent.name
                match = re.fullmatch(r"alpha_(.+)__lambda_(.+)", run_dir)
                if alpha_set is not None:
                    if not match or match.group(1) not in alpha_set:
                        continue
                if lambda_set is not None:
                    if not match or match.group(2) not in lambda_set:
                        continue
                paths.append(raw_path)
    return sorted(set(paths))


def _counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["judge_label"].value_counts(dropna=False).to_dict().items()}


def _mean_or_none(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def _summary_row(judged_df: pd.DataFrame, raw_path: Path, config: dict[str, Any], judge_batch_size: int) -> dict[str, Any]:
    dataset = raw_path.name.split("__", maxsplit=1)[0]
    alpha, lambda_value = _parse_run(raw_path)
    counts = _counts(judged_df)
    n = int(len(judged_df))
    unique_responses = int(judged_df["response_text"].fillna("").astype(str).nunique()) if "response_text" in judged_df else 0
    response_nonempty = judged_df["response_text"].fillna("").astype(str).str.strip().ne("").mean() if "response_text" in judged_df else 0.0
    return {
        "dataset": dataset,
        "strategy": _strategy_from_path(raw_path),
        "alpha": alpha,
        "lambda": lambda_value,
        "judge_provider": str(config["judge"]["provider"]),
        "judge_model": str(config["judge"].get("model_name", config["judge"]["provider"])),
        "judge_batch_size": int(judge_batch_size),
        "n_eval": n,
        "acceptable": int(counts.get("ACCEPTABLE", 0)),
        "unacceptable": int(counts.get("UNACCEPTABLE", 0)),
        "neither": int(counts.get("NEITHER", 0)),
        "acceptable_rate": float(counts.get("ACCEPTABLE", 0) / max(n, 1)),
        "unacceptable_rate": float(counts.get("UNACCEPTABLE", 0) / max(n, 1)),
        "neither_rate": float(counts.get("NEITHER", 0) / max(n, 1)),
        "unique_responses": unique_responses,
        "unique_response_rate": float(unique_responses / max(n, 1)),
        "nonempty_response_rate": float(response_nonempty),
        "relative_hidden_delta_norm_mean": _mean_or_none(judged_df, "relative_hidden_delta_norm"),
        "hidden_delta_norm_mean": _mean_or_none(judged_df, "hidden_delta_norm"),
        "delta_h_l2_norm_mean": _mean_or_none(judged_df, "delta_h_l2_norm"),
        "surrogate_target_l2_error_mean": _mean_or_none(judged_df, "surrogate_target_l2_error"),
        "surrogate_target_l2_error_normalized_mean": _mean_or_none(
            judged_df, "surrogate_target_l2_error_normalized"
        ),
        "exact_target_l2_error_mean": _mean_or_none(judged_df, "exact_target_l2_error"),
        "exact_target_l2_error_normalized_mean": _mean_or_none(judged_df, "exact_target_l2_error_normalized"),
        "surrogate_target_mse_initial_mean": _mean_or_none(judged_df, "surrogate_target_mse_initial"),
        "surrogate_target_mse_final_mean": _mean_or_none(judged_df, "surrogate_target_mse_final"),
        "label_counts": json.dumps(counts, sort_keys=True),
        "judged_path": str(raw_path.with_name(raw_path.name.replace("__raw.parquet", "__local_judge.parquet"))),
        "raw_path": str(raw_path),
    }


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    provider = str(config.get("judge", {}).get("provider", "rules"))
    if provider == "openai_api":
        raise RuntimeError("Refusing to run an API judge. Use judge.provider: local_llm or rules.")

    judge_batch_size = int(args.judge_batch_size or config["judge"].get("batch_size", 4))
    config["judge"]["batch_size"] = judge_batch_size

    artifact_root = Path(args.artifact_root).resolve()
    model_slug = slugify(config["model"]["name"])
    raw_paths = _discover_raw_paths(
        artifact_root=artifact_root,
        model_slug=model_slug,
        datasets=[str(dataset) for dataset in args.datasets],
        alphas=args.alphas,
        lambdas=args.lambdas,
    )
    if not raw_paths:
        raise FileNotFoundError("No topology surrogate raw files matched the requested filters.")

    rows: list[dict[str, Any]] = []
    judge = load_judge(config)
    try:
        for raw_path in raw_paths:
            judged_path = raw_path.with_name(raw_path.name.replace("__raw.parquet", "__local_judge.parquet"))
            dataset = raw_path.name.split("__", maxsplit=1)[0]
            alpha, lambda_value = _parse_run(raw_path)
            if judged_path.exists() and not args.force:
                judged_df = pd.read_parquet(judged_path)
                print(f"[{dataset} alpha={alpha} lambda={lambda_value}] local judged exists", flush=True)
            else:
                print(f"[{dataset} alpha={alpha} lambda={lambda_value}] local judging {raw_path}", flush=True)
                raw_df = pd.read_parquet(raw_path)
                judged_df = _judge_table(judge, raw_df, batch_size=judge_batch_size)
                judged_df["judge_rerun_note"] = f"local_{provider}_topology_surrogate"
                write_parquet(judged_df, judged_path)
                judged_df.to_csv(judged_path.with_suffix(".csv"), index=False)
            rows.append(_summary_row(judged_df, raw_path, config, judge_batch_size))
    finally:
        if hasattr(judge, "model"):
            delattr(judge, "model")

    summary_df = pd.DataFrame(rows).sort_values(["dataset", "strategy", "alpha", "lambda"], na_position="last")
    summary_dir = ensure_dir(artifact_root / "_local_judge_summaries")
    summary_path = summary_dir / "topology_surrogate_local_judge_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_parquet(summary_path.with_suffix(".parquet"), index=False)
    write_json(summary_dir / "topology_surrogate_local_judge_summary.json", summary_df.to_dict(orient="records"))
    print("SUMMARY", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"Wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
