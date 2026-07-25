#!/usr/bin/env python3
"""Calibrate class-conditional PPLM margin targets on SST-5 training text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from repstat_steering.pplm_control import (
    PPLM_SENTIMENT_LABELS,
    PPLMSentimentExperiment,
)
from repstat_steering.pplm_quantiles import (
    PPLM_SST5_DATASET_LABELS,
    quantile_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SetFit/sst5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="checkpoints/gpt2-medium")
    parser.add_argument("--classifier", default="checkpoints/SST_classifier_head.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantiles", nargs="+", type=float, default=[0.5, 0.7, 0.75, 0.8, 0.9]
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantiles = sorted(set(float(value) for value in args.quantiles))
    for quantile in quantiles:
        quantile_key(quantile)

    dataset = load_dataset(args.dataset, split=args.split)
    experiment = PPLMSentimentExperiment(args.model, args.classifier, args.device)
    rows: list[dict[str, object]] = []
    targets: dict[str, dict[str, object]] = {}
    for target_label in ("positive", "negative"):
        dataset_label = PPLM_SST5_DATASET_LABELS[target_label]
        indices = [
            index
            for index, label in enumerate(dataset["label"])
            if int(label) == dataset_label
        ]
        texts = [str(dataset[index]["text"]) for index in indices]
        classifier_logits = experiment.classifier_logits_for_texts(
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        classifier_class = PPLM_SENTIMENT_LABELS[target_label]
        margins = experiment._margin(classifier_logits, classifier_class).numpy()
        probabilities = classifier_logits.softmax(dim=-1)[:, classifier_class].numpy()
        for source_index, text, margin, probability in zip(
            indices, texts, margins, probabilities, strict=True
        ):
            rows.append(
                {
                    "source_index": source_index,
                    "target_label": target_label,
                    "dataset_label": dataset_label,
                    "classifier_class": classifier_class,
                    "text": text,
                    "margin": float(margin),
                    "target_probability": float(probability),
                }
            )
        targets[target_label] = {
            "dataset_label": dataset_label,
            "classifier_class": classifier_class,
            "count": len(margins),
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "minimum": float(np.min(margins)),
            "maximum": float(np.max(margins)),
            "quantiles": {
                quantile_key(quantile): float(np.quantile(margins, quantile))
                for quantile in quantiles
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    margins_path = args.output_dir / "sst5_train_margins.csv"
    calibration_path = args.output_dir / "quantile_targets.json"
    pd.DataFrame(rows).to_csv(margins_path, index=False)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "calibration_unit": "complete_sentence_mean_pooled_hidden_state",
        "model": args.model,
        "classifier": args.classifier,
        "max_length": args.max_length,
        "quantile_method": "numpy_linear",
        "targets": targets,
    }
    calibration_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {margins_path} and {calibration_path}")


if __name__ == "__main__":
    main()
