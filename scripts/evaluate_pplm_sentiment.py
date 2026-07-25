#!/usr/bin/env python3
"""Evaluate PPLM generations with an external local sentiment model and LM NLL."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sentiment-model",
        default="checkpoints/distilbert-sst2",
    )
    parser.add_argument("--language-model", default="checkpoints/gpt2-medium")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sentiment_probabilities(
    texts: list[str], model_path: str, device: torch.device
) -> list[dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to(device)
    model.eval()
    outputs = []
    for start in range(0, len(texts), 32):
        batch = tokenizer(
            texts[start : start + 32],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            probabilities = model(**batch).logits.softmax(dim=-1).cpu()
        for row in probabilities:
            outputs.append(
                {
                    str(model.config.id2label[index]).lower(): float(value)
                    for index, value in enumerate(row)
                }
            )
    del model
    torch.cuda.empty_cache()
    return outputs


def continuation_perplexities(
    frame: pd.DataFrame, model_path: str, device: torch.device
) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to(device)
    model.eval()
    values = []
    for row in frame.itertuples(index=False):
        full_ids = tokenizer(row.text, return_tensors="pt").input_ids.to(device)
        prefix_length = tokenizer(row.prefix, return_tensors="pt").input_ids.shape[1]
        with torch.no_grad():
            logits = model(full_ids, use_cache=False).logits
            log_probabilities = logits[0, prefix_length - 1 : -1].log_softmax(dim=-1)
        targets = full_ids[0, prefix_length:]
        indices = torch.arange(targets.shape[0], device=device)
        mean_nll = -log_probabilities[indices, targets].mean()
        values.append(float(mean_nll.exp()))
    return values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    device = torch.device(args.device)
    sentiment = sentiment_probabilities(frame["text"].tolist(), args.sentiment_model, device)
    frame["external_positive_probability"] = [row.get("positive", 0.0) for row in sentiment]
    frame["external_negative_probability"] = [row.get("negative", 0.0) for row in sentiment]
    frame["perplexity"] = continuation_perplexities(frame, args.language_model, device)
    frame["external_target_probability"] = [
        positive if target == "positive" else negative
        for target, positive, negative in zip(
            frame["target_label"],
            frame["external_positive_probability"],
            frame["external_negative_probability"],
        )
    ]
    frame.to_csv(output_dir / "evaluated_generations.csv", index=False)
    grouping = ["method", "target_label"]
    if "run_dir" in frame.columns:
        grouping.insert(0, "run_dir")
    summary = (
        frame.groupby(grouping, as_index=False)
        .agg(
            n=("text", "size"),
            external_target_probability=("external_target_probability", "mean"),
            external_success=(
                "external_target_probability", lambda values: (values >= 0.5).mean()
            ),
            perplexity=("perplexity", "mean"),
            mean_relative_cache_change=("mean_relative_cache_change", "mean"),
        )
    )
    for column in ("mean_mix_scale", "mean_token_kl", "mean_raw_token_kl"):
        if column in frame.columns:
            summary[column] = frame.groupby(grouping)[column].mean().to_numpy()
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
