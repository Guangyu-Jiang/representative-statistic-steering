"""Group-conditioned CLAMBER steering using predicted 9-way subclasses.

This experiment conditions steering directions on the predicted CLAMBER 9-way
subclass for each ambiguous prompt. Directions are built *within each
predicted group* using only examples in that predicted group:

    plus  = train ambiguous prompts in the group judged ACCEPTABLE
    minus = train ambiguous prompts in the group judged UNACCEPTABLE

The resulting direction is then applied only to test ambiguous prompts in the
same predicted group whose base response was judged UNACCEPTABLE.
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

from aen_replication.config import load_config
from aen_replication.eval.judge import load_judge
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.clamber_subclass_classification import _fit_multiclass_logistic
from aen_replication.train.steering import (
    SteeringDirection,
    _build_direction,
    _evaluate_strategy,
    _extract_prompt_vectors,
    _judge_table,
    _prompt_texts,
)
from aen_replication.utils.io_utils import append_command_history, ensure_dir, slugify, write_json, write_markdown, write_parquet
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _normalize_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    return [int(v) for v in list(value)]


def _release_model(bundle: HFModelBundle | None) -> None:
    if bundle is None:
        return
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_response_table(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No CLAMBER responses found at {path}")
    df = df.rename(columns={"prompt": "text", "response": "response_text"}).copy()
    keep = [
        "example_id",
        "pair_id",
        "split",
        "dataset",
        "text",
        "response_text",
        "subclass",
        "group4",
        "category",
        "label_ambiguous",
        "require_clarification",
        "source_question",
        "clarifying_question",
        "context",
    ]
    return df.loc[:, [col for col in keep if col in df.columns]].copy()


def _judge_or_load_base_behavior(config: dict[str, Any], output_root: Path) -> pd.DataFrame:
    cfg = config["clamber_conditioned_steering"]
    raw_path = output_root / "clamber_base_behavior_raw.parquet"
    judged_path = output_root / "clamber_base_behavior.parquet"
    summary_path = output_root / "clamber_base_behavior_summary.json"

    if judged_path.exists():
        return pd.read_parquet(judged_path)

    response_df = _load_response_table(Path(cfg["responses_json"]))
    write_parquet(response_df, raw_path)

    judge = load_judge(config)
    try:
        judged_df = _judge_table(judge, response_df.loc[:, ["example_id", "pair_id", "split", "text", "response_text"]], batch_size=int(config["judge"].get("batch_size", 8)))
    finally:
        if hasattr(judge, "model"):
            delattr(judge, "model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judged_df = response_df.merge(
        judged_df.loc[:, ["example_id", "pair_id", "split", "judge_label", "judge_explanation", "judge_raw_response"]],
        on=["example_id", "pair_id", "split"],
        how="left",
    )
    write_parquet(judged_df, judged_path)
    summary = {
        "n_rows": int(len(judged_df)),
        "judge_label_counts": {str(k): int(v) for k, v in judged_df["judge_label"].value_counts(dropna=False).to_dict().items()},
        "split_counts": {str(k): int(v) for k, v in judged_df["split"].value_counts(dropna=False).to_dict().items()},
        "subclass_counts": {str(k): int(v) for k, v in judged_df["subclass"].value_counts(dropna=False).to_dict().items()},
    }
    write_json(summary_path, summary)
    return judged_df


def _load_clear_pool(config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["clamber_conditioned_steering"]
    pairs_path = Path(cfg.get("clear_pairs_path", config["data"]["pair_output_dir"] + "/clamber_pairs.parquet"))
    df = pd.read_parquet(pairs_path).copy()
    build_split = str(cfg.get("build_split", "train"))
    clear_df = df.loc[
        df["split"].eq(build_split) & df["label_ambiguous"].eq(0),
        ["example_id", "pair_id", "split", "text", "subclass", "category", "source_question", "context"],
    ].reset_index(drop=True)
    if clear_df.empty:
        raise ValueError(f"No clear CLAMBER prompts found in {pairs_path} for split={build_split}")
    return clear_df


def _load_best_rows(metrics_path: Path, model_slug: str) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(metrics_path)
    subset = df.loc[df["model"].eq(model_slug)].copy()
    if subset.empty:
        raise ValueError(f"No CLAMBER 9-way metrics found for model slug {model_slug}")
    result: dict[str, dict[str, Any]] = {}
    for method in ("full_probe", "aen_only"):
        row = subset.loc[subset["method"].eq(method)].iloc[0].to_dict()
        row["aen_indices"] = _normalize_indices(row.get("aen_indices"))
        row["layer"] = int(row["layer"])
        result[method] = row
    return result


def _fit_prediction_table(
    *,
    hidden_root: Path,
    layer: int,
    aen_indices: list[int],
    seed: int,
    max_iter: int,
    c_value: float,
) -> pd.DataFrame:
    meta, matrix = load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    train_mask = meta["split"].eq("train").to_numpy()
    x_train = matrix[train_mask]
    y_train = meta.loc[train_mask, "subclass"].astype(str).to_numpy()
    x_all = matrix
    if aen_indices:
        x_train = x_train[:, aen_indices]
        x_all = x_all[:, aen_indices]
    clf, scaler = _fit_multiclass_logistic(
        x_train=x_train,
        y_train=y_train,
        max_iter=max_iter,
        c_value=c_value,
        seed=seed,
    )
    predictions = clf.predict(scaler.transform(x_all))
    out = meta.loc[:, ["example_id", "pair_id", "split", "subclass"]].copy()
    out["predicted_subclass"] = predictions
    out["layer"] = int(layer)
    out["feature_count"] = int(x_train.shape[1])
    return out


def _prediction_tables(config: dict[str, Any], output_root: Path) -> dict[str, pd.DataFrame]:
    cfg = config["clamber_conditioned_steering"]
    pred_path = output_root / "prediction_tables.parquet"
    if pred_path.exists():
        pred_df = pd.read_parquet(pred_path)
    else:
        model_slug = slugify(config["model"]["name"])
        best_rows = _load_best_rows(Path(cfg["metrics_path"]), model_slug)
        hidden_root = Path(cfg["hidden_root"])
        max_iter = int(cfg.get("max_iter", 4000))
        c_value = float(cfg.get("c_value", 1.0))
        frames: list[pd.DataFrame] = []
        for source in cfg.get("prediction_sources", ["full_probe", "aen_only"]):
            row = best_rows[source]
            pred_df_source = _fit_prediction_table(
                hidden_root=hidden_root,
                layer=int(row["layer"]),
                aen_indices=list(row["aen_indices"]) if source == "aen_only" else [],
                seed=int(config["seed"]) + (0 if source == "full_probe" else 1000),
                max_iter=max_iter,
                c_value=c_value,
            )
            pred_df_source["prediction_source"] = source
            pred_df_source["aen_k"] = int(row["aen_k"]) if source == "aen_only" and row.get("aen_k") is not None and not pd.isna(row.get("aen_k")) else np.nan
            pred_df_source["aen_indices"] = [list(row["aen_indices"]) if source == "aen_only" else []] * len(pred_df_source)
            frames.append(pred_df_source)
        pred_df = pd.concat(frames, ignore_index=True)
        write_parquet(pred_df, pred_path)

    return {
        str(source): pred_df.loc[pred_df["prediction_source"].eq(source)].reset_index(drop=True)
        for source in pred_df["prediction_source"].drop_duplicates().tolist()
    }


def _strategy_for_source(source: str) -> str:
    return "full_vector" if source == "full_probe" else "aens"


def _stable_group_seed(base_seed: int, source: str, group_name: str) -> int:
    payload = f"{source}::{group_name}"
    return int(base_seed + sum(ord(ch) for ch in payload))


def _direction_layer_and_indices(pred_df: pd.DataFrame) -> tuple[int, list[int]]:
    layer_values = pred_df["layer"].dropna().unique().tolist()
    if len(layer_values) != 1:
        raise ValueError(f"Expected one steering layer per prediction source, found {layer_values}")
    aen_indices = []
    if "aen_indices" in pred_df.columns and not pred_df["aen_indices"].empty:
        sample = pred_df["aen_indices"].iloc[0]
        aen_indices = _normalize_indices(sample)
    return int(layer_values[0]), aen_indices


def _direction_for_group(
    *,
    bundle: HFModelBundle,
    config: dict[str, Any],
    layer: int,
    aen_indices: list[int],
    group_name: str,
    build_df: pd.DataFrame,
    clear_df: pd.DataFrame,
    seed: int,
) -> SteeringDirection | None:
    cfg = config["clamber_conditioned_steering"]
    plus_df = build_df.loc[build_df["judge_label"].eq("ACCEPTABLE")].reset_index(drop=True)
    min_plus = int(cfg.get("min_plus_per_group", 5))
    min_clear = int(cfg.get("min_clear_pool", 20))
    if len(plus_df) < min_plus or len(clear_df) < min_clear:
        return None

    max_plus = int(cfg.get("build_max_plus_per_group", 100))
    max_minus = int(cfg.get("build_clear_n", cfg.get("build_max_minus_per_group", 100)))
    plus_sample = plus_df.sample(n=min(max_plus, len(plus_df)), random_state=seed).reset_index(drop=True)
    minus_sample = clear_df.sample(n=min(max_minus, len(clear_df)), random_state=seed).reset_index(drop=True)

    prompt_suffix = str(cfg.get("prompt_suffix", ""))
    plus_prompts = _prompt_texts(bundle, plus_sample["text"].astype(str).tolist(), config["generation"], prompt_suffix)
    minus_prompts = _prompt_texts(bundle, minus_sample["text"].astype(str).tolist(), config["generation"], prompt_suffix)
    plus_vectors = _extract_prompt_vectors(bundle, plus_prompts, config["extraction"], layer)
    minus_vectors = _extract_prompt_vectors(bundle, minus_prompts, config["extraction"], layer)
    vector = _build_direction(plus_vectors=plus_vectors, minus_vectors=minus_vectors, seed=seed)
    ranked_indices = np.argsort(-np.abs(vector)).astype(int).tolist()
    return SteeringDirection(
        vector=vector,
        aen_indices=list(aen_indices),
        ranked_indices=ranked_indices,
        layer=int(layer),
    )


def _summary_and_report(summary_df: pd.DataFrame, output_root: Path) -> None:
    write_parquet(summary_df, output_root / "summary.parquet")
    lines = ["# CLAMBER Conditioned Steering", ""]
    for source in summary_df["prediction_source"].drop_duplicates().tolist():
        source_df = summary_df.loc[summary_df["prediction_source"].eq(source)].copy()
        lines.append(f"## {source}")
        lines.append("")
        lines.append("| Predicted group | Build + | Build - | Eval n | Abstention rate | Strategy | Layer |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
        for row in source_df.sort_values("predicted_group").itertuples(index=False):
            abstention = "nan" if pd.isna(row.abstention_rate) else f"{float(row.abstention_rate):.4f}"
            lines.append(
                f"| {row.predicted_group} | {int(row.n_build_plus)} | {int(row.n_build_minus)} | {int(row.n_eval)} | {abstention} | {row.strategy} | {int(row.layer)} |"
            )
        overall_eval = int(source_df["n_eval"].sum())
        if overall_eval > 0:
            weighted = float((source_df["abstention_rate"].fillna(0.0) * source_df["n_eval"]).sum() / overall_eval)
            lines.append("")
            lines.append(f"Overall weighted abstention rate: `{weighted:.4f}` over `{overall_eval}` eval prompts.")
        lines.append("")
    write_markdown(output_root / "report.md", lines)


def run_conditioned_steering(config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["clamber_conditioned_steering"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(cfg["output_dir"]) / model_slug)

    judged_base = _judge_or_load_base_behavior(config, output_root)
    clear_pool = _load_clear_pool(config)
    prediction_tables = _prediction_tables(config, output_root)

    bundle: HFModelBundle | None = None
    summary_rows: list[dict[str, Any]] = []
    try:
        bundle = load_hf_model(config["model"], config["generation"])
        for source in cfg.get("prediction_sources", ["full_probe", "aen_only"]):
            pred_df = prediction_tables[source]
            merged = judged_base.merge(
                pred_df.loc[:, ["example_id", "pair_id", "split", "predicted_subclass", "layer", "aen_indices"]],
                on=["example_id", "pair_id", "split"],
                how="inner",
            )
            write_parquet(merged, output_root / f"{source}__base_with_predictions.parquet")

            layer, aen_indices = _direction_layer_and_indices(pred_df)
            strategy = _strategy_for_source(source)
            build_split = str(cfg.get("build_split", "train"))
            eval_split = str(cfg.get("eval_split", "test"))

            for group_name in sorted(merged["predicted_subclass"].astype(str).unique().tolist()):
                group_slug = slugify(group_name)
                group_build = merged.loc[
                    merged["split"].eq(build_split) & merged["predicted_subclass"].astype(str).eq(group_name)
                ].reset_index(drop=True)
                direction = _direction_for_group(
                    bundle=bundle,
                    config=config,
                    layer=layer,
                    aen_indices=aen_indices,
                    group_name=group_name,
                    build_df=group_build,
                    clear_df=clear_pool,
                    seed=_stable_group_seed(int(config["seed"]), source, group_name),
                )
                direction_path = output_root / f"{source}__{group_slug}__direction.npy"
                raw_path = output_root / f"{source}__{group_slug}__raw.parquet"
                judged_path = output_root / f"{source}__{group_slug}.parquet"

                n_plus = int(group_build["judge_label"].eq("ACCEPTABLE").sum())
                n_minus = int(min(int(cfg.get("build_clear_n", cfg.get("build_max_minus_per_group", 100))), len(clear_pool)))
                eval_df = merged.loc[
                    merged["split"].eq(eval_split)
                    & merged["predicted_subclass"].astype(str).eq(group_name)
                    & merged["judge_label"].eq("UNACCEPTABLE")
                ].reset_index(drop=True)

                if direction is None:
                    summary_rows.append(
                        {
                            "prediction_source": source,
                            "predicted_group": group_name,
                            "strategy": strategy,
                            "layer": int(layer),
                            "n_build_plus": n_plus,
                            "n_build_minus": n_minus,
                            "n_eval": int(len(eval_df)),
                            "abstention_rate": np.nan,
                            "status": "skipped_insufficient_build_pool",
                        }
                    )
                    continue

                np.save(direction_path, direction.vector)
                if raw_path.exists():
                    steered_raw = pd.read_parquet(raw_path)
                else:
                    steer_input = eval_df.loc[:, ["example_id", "pair_id", "split", "text", "subclass", "predicted_subclass"]].copy()
                    steered_raw = _evaluate_strategy(
                        bundle=bundle,
                        df=steer_input,
                        generation_cfg=config["generation"],
                        steering_cfg=config.get("steering", {}),
                        prompt_suffix=str(cfg.get("prompt_suffix", "")),
                        direction=direction,
                        strategy=strategy,
                        alpha=float(cfg.get("alpha", 1.0)),
                    )
                    if "split" not in steered_raw.columns:
                        steered_raw = steered_raw.merge(
                            steer_input.loc[:, ["example_id", "pair_id", "split"]],
                            on=["example_id", "pair_id"],
                            how="left",
                        )
                    steered_raw = steered_raw.merge(
                        eval_df.loc[:, ["example_id", "pair_id", "split", "subclass", "predicted_subclass", "judge_label"]].rename(columns={"judge_label": "base_judge_label"}),
                        on=["example_id", "pair_id", "split"],
                        how="left",
                    )
                    steered_raw["prediction_source"] = source
                    steered_raw["predicted_group"] = group_name
                    write_parquet(steered_raw, raw_path)

                if judged_path.exists():
                    judged_df = pd.read_parquet(judged_path)
                else:
                    judge = load_judge(config)
                    try:
                        judge_input = steered_raw.loc[:, ["example_id", "pair_id", "text", "response_text"]].copy()
                        if "split" in steered_raw.columns:
                            judge_input["split"] = steered_raw["split"]
                        judged_df = _judge_table(judge, judge_input, batch_size=int(config["judge"].get("batch_size", 8)))
                    finally:
                        if hasattr(judge, "model"):
                            delattr(judge, "model")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if "split" not in judged_df.columns and "split" in steered_raw.columns:
                        judged_df = judged_df.merge(
                            steered_raw.loc[:, ["example_id", "pair_id", "split"]].drop_duplicates(),
                            on=["example_id", "pair_id"],
                            how="left",
                        )
                    judged_df = steered_raw.merge(
                        judged_df.loc[:, ["example_id", "pair_id", "split", "judge_label", "judge_explanation", "judge_raw_response"]],
                        on=["example_id", "pair_id", "split"],
                        how="left",
                    )
                    write_parquet(judged_df, judged_path)

                abstention_rate = float(judged_df["judge_label"].eq("ACCEPTABLE").mean()) if not judged_df.empty else float("nan")
                summary_rows.append(
                    {
                        "prediction_source": source,
                        "predicted_group": group_name,
                        "strategy": strategy,
                        "layer": int(layer),
                        "n_build_plus": n_plus,
                        "n_build_minus": n_minus,
                        "n_eval": int(len(judged_df)),
                        "abstention_rate": abstention_rate,
                        "status": "ok",
                    }
                )
    finally:
        _release_model(bundle)

    summary_df = pd.DataFrame(summary_rows)
    _summary_and_report(summary_df, output_root)
    write_json(
        output_root / "metadata.json",
        {
            "model_name": config["model"]["name"],
            "prediction_sources": list(cfg.get("prediction_sources", ["full_probe", "aen_only"])),
            "responses_json": str(cfg["responses_json"]),
            "metrics_path": str(cfg["metrics_path"]),
            "hidden_root": str(cfg["hidden_root"]),
            "build_split": str(cfg.get("build_split", "train")),
            "eval_split": str(cfg.get("eval_split", "test")),
            "clear_pool_size": int(len(clear_pool)),
        },
    )
    return summary_df


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_clamber_conditioned_steering.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], ["run_clamber_conditioned_steering", "--config", args.config])
    run_conditioned_steering(config)


if __name__ == "__main__":
    main()
