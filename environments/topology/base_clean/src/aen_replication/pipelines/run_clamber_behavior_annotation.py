"""Annotate CLAMBER ambiguous prompts with abstention-vs-direct labels using a local judge."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from aen_replication.config import load_config
from aen_replication.eval.judge import load_judge
from aen_replication.models.generation import generate_batch
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.utils.io_utils import append_command_history, ensure_dir, slugify, write_json, write_parquet
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

GROUP4_MAP = {
    "polysemy": "ambiguity",
    "co-reference": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "ICL": "conflicting_condition",
    "none": "clear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _iter_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + batch_size] for start in range(0, len(df), batch_size)]


def _release_model(bundle: HFModelBundle | None) -> None:
    if bundle is None:
        return
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_clamber_ambiguous(config: dict[str, Any]) -> pd.DataFrame:
    annotation_cfg = config["clamber_behavior_annotation"]
    input_path = Path(annotation_cfg.get("input_path", "data/processed/clamber_pairs.parquet"))
    df = pd.read_parquet(input_path).copy()
    if bool(annotation_cfg.get("ambiguous_only", True)):
        df = df.loc[df["label_ambiguous"].eq(1)].copy()
    sample_n = int(annotation_cfg.get("sample_n", 0))
    if sample_n > 0 and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=int(config["seed"])).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df


def _generate_raw_annotations(
    bundle: HFModelBundle,
    config: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    annotation_cfg = config["clamber_behavior_annotation"]
    generation_cfg = config["generation"]
    prompt_suffix = str(annotation_cfg.get("prompt_suffix", ""))
    batch_size = int(generation_cfg.get("batch_size", 8))
    rows: list[dict[str, Any]] = []
    for batch_df in tqdm(
        _iter_batches(df, batch_size),
        total=(len(df) + batch_size - 1) // batch_size,
        desc="clamber_generate",
        leave=False,
    ):
        prompts = [f"{text}{prompt_suffix}" if prompt_suffix else str(text) for text in batch_df["text"].tolist()]
        generations = generate_batch(bundle=bundle, prompt_texts=prompts, generation_config=generation_cfg)
        for row, generation in zip(batch_df.itertuples(index=False), generations, strict=True):
            rows.append(
                {
                    "example_id": row.example_id,
                    "pair_id": row.pair_id,
                    "split": row.split,
                    "text": row.text,
                    "source_question": row.source_question,
                    "context": row.context,
                    "subclass": row.subclass,
                    "group4": row.group4,
                    "category": row.category,
                    "clarifying_question": row.clarifying_question,
                    "prompt_suffix": prompt_suffix,
                    "prompt_text": generation.prompt_text,
                    "response_text": generation.response_text,
                    "generated_token_count": int(generation.generated_token_count),
                    "average_entropy": generation.average_entropy,
                }
            )
    return pd.DataFrame(rows)


def _judge_annotations(
    judge: Any,
    df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    results = []
    for batch_df in tqdm(
        _iter_batches(df, batch_size),
        total=(len(df) + batch_size - 1) // batch_size,
        desc="clamber_judge",
        leave=False,
    ):
        if hasattr(judge, "judge_many"):
            batch_results = judge.judge_many(
                batch_df["text"].tolist(),
                batch_df["response_text"].tolist(),
                batch_size=batch_size,
            )
        else:
            batch_results = [judge.judge(row.text, row.response_text) for row in batch_df.itertuples(index=False)]
        results.extend(batch_results)
    judged = df.copy()
    judged["judge_label"] = [result.label for result in results]
    judged["judge_explanation"] = [result.explanation for result in results]
    judged["judge_raw_response"] = [result.raw_response for result in results]
    return judged


def _summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_df in df.groupby("split", dropna=False, sort=True):
        for label, label_df in split_df.groupby("judge_label", dropna=False, sort=True):
            rows.append(
                {
                    "group_field": "__overall__",
                    "group_value": "__all__",
                    "split": str(split),
                    "judge_label": str(label),
                    "count": int(len(label_df)),
                }
            )
        for group_field in ("subclass", "group4"):
            valid_df = split_df.loc[split_df[group_field].notna()].copy()
            for (group_value, label), subset in valid_df.groupby([group_field, "judge_label"], dropna=False, sort=True):
                rows.append(
                    {
                        "group_field": group_field,
                        "group_value": str(group_value),
                        "split": str(split),
                        "judge_label": str(label),
                        "count": int(len(subset)),
                    }
                )
    return pd.DataFrame(rows)


def _write_label_splits(df: pd.DataFrame, output_root: Path, group_fields: list[str]) -> None:
    label_root = ensure_dir(output_root / "label_splits")
    for label in ("ACCEPTABLE", "UNACCEPTABLE", "NEITHER"):
        subset = df.loc[df["judge_label"].eq(label)].reset_index(drop=True)
        if not subset.empty:
            write_parquet(subset, label_root / f"{label.lower()}_all.parquet")
            (label_root / f"{label.lower()}_all.json").write_text(
                subset.to_json(orient="records", force_ascii=False, indent=2),
                encoding="utf-8",
            )

    grouped_root = ensure_dir(output_root / "group_splits")
    for group_field in group_fields:
        field_root = ensure_dir(grouped_root / group_field)
        valid_df = df.loc[df[group_field].notna()].copy()
        for (group_value, label), subset in valid_df.groupby([group_field, "judge_label"], dropna=False, sort=True):
            safe_value = slugify(str(group_value))
            safe_label = str(label).lower()
            if subset.empty:
                continue
            write_parquet(subset.reset_index(drop=True), field_root / f"{safe_value}__{safe_label}.parquet")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_clamber_behavior_annotation.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    annotation_cfg = config["clamber_behavior_annotation"]
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(annotation_cfg["output_dir"]) / model_slug)
    raw_path = output_root / "clamber_ambiguous_behavior_raw.parquet"
    judged_path = output_root / "clamber_ambiguous_behavior.parquet"
    summary_path = output_root / "clamber_ambiguous_behavior_summary.parquet"
    metadata_path = output_root / "metadata.json"

    df = _load_clamber_ambiguous(config)
    LOGGER.info("Loaded %s CLAMBER ambiguous prompts for annotation.", len(df))

    bundle: HFModelBundle | None = None
    try:
        bundle = load_hf_model(config["model"], config["generation"])
        raw_df = _generate_raw_annotations(bundle=bundle, config=config, df=df)
        write_parquet(raw_df, raw_path)
    finally:
        _release_model(bundle)

    judge = load_judge(config)
    try:
        judged_df = _judge_annotations(judge=judge, df=raw_df, batch_size=int(config["judge"].get("batch_size", 4)))
    finally:
        if hasattr(judge, "model"):
            delattr(judge, "model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_parquet(judged_df, judged_path)
    summary_df = _summary_rows(judged_df)
    write_parquet(summary_df, summary_path)
    _write_label_splits(
        judged_df,
        output_root=output_root,
        group_fields=list(annotation_cfg.get("group_fields", ["subclass", "group4"])),
    )

    metadata = {
        "model_name": config["model"]["name"],
        "judge_model_name": config["judge"]["model_name"],
        "n_rows": int(len(judged_df)),
        "n_train": int(judged_df["split"].eq("train").sum()),
        "n_test": int(judged_df["split"].eq("test").sum()),
        "prompt_suffix": str(annotation_cfg.get("prompt_suffix", "")),
        "group4_counts": {str(key): int(value) for key, value in judged_df["group4"].value_counts(dropna=False).to_dict().items()},
        "subclass_counts": {str(key): int(value) for key, value in judged_df["subclass"].value_counts(dropna=False).to_dict().items()},
        "judge_label_counts": {str(key): int(value) for key, value in judged_df["judge_label"].value_counts(dropna=False).to_dict().items()},
    }
    write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
