#!/usr/bin/env python3
"""Extract rolling Lookback features by replaying saved NQ responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repstat_steering.lookback_control import LookbackNQExperiment, load_nq_examples


DEFAULT_RESULTS = [
    "artifacts/lookback_nq/development_n60/baseline_guided/results.jsonl",
    "artifacts/lookback_nq/heldout_offset60_n100_max256/baseline_greedy/results.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-dir", default="artifacts/lookback_nq/nq_replay_features"
    )
    parser.add_argument(
        "--model",
        default="NousResearch/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--classifier",
        default="external/Lookback-Lens/classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl",
    )
    parser.add_argument(
        "--data",
        default="external/Lookback-Lens/data/nq-open-10_total_documents_gold_at_4.jsonl.gz",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", default="float16")
    return parser.parse_args()


def read_rows(paths: list[str]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path_string in paths:
        path = Path(path_string)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") != "baseline":
                continue
            rows[int(row["dataset_index"])] = row
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    rows = read_rows(args.results)
    examples = {
        example.dataset_index: example
        for example in load_nq_examples(args.data, indices=sorted(rows))
    }
    completed: set[int] = set()
    if manifest_path.exists():
        completed = {
            int(json.loads(line)["dataset_index"])
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        }

    experiment = LookbackNQExperiment(
        args.model,
        args.classifier,
        device=args.device,
        model_dtype=args.model_dtype,
    )
    try:
        for progress, dataset_index in enumerate(sorted(rows), start=1):
            if dataset_index in completed:
                continue
            row = rows[dataset_index]
            features = experiment.replay_response_features(
                examples[dataset_index], row["response"], args.window_size
            )
            if not features:
                print(f"[{progress}/{len(rows)}] index={dataset_index} empty", flush=True)
                continue
            feature_path = feature_dir / f"{dataset_index:05d}.npz"
            matrix = torch.stack(features).numpy().astype(np.float16)
            np.savez_compressed(feature_path, features=matrix)
            record = {
                "dataset_index": dataset_index,
                "feature_path": str(feature_path),
                "token_windows": int(matrix.shape[0]),
                "feature_count": int(matrix.shape[1]),
                "exact_match": float(row["exact_match"]),
                "response": row["response"],
                "replay_source": "retokenized_saved_response",
                "window_size": args.window_size,
            }
            with manifest_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[{progress}/{len(rows)}] index={dataset_index} "
                f"windows={matrix.shape[0]} label={row['exact_match']:.0f}",
                flush=True,
            )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
