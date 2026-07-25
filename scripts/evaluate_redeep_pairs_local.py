#!/usr/bin/env python3
"""Run a local paper-style pairwise ReDeEP truthfulness judge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repstat_steering.redeep_evaluation import (
    build_local_pairwise_judge_prompt,
    parse_local_pairwise_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def batched(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(
        args.input.stem + ".local_pairwise.jsonl"
    )
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    by_source = {
        int(row["source_id"]): row
        for row in rows
        if row["method"] == "baseline"
    }
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row["method"] == "baseline":
            continue
        baseline = by_source[int(row["source_id"])]
        pairs.append(
            {
                "source_id": row["source_id"],
                "method": row["method"],
                "question": row["question"],
                "passage": row["passage"],
                "baseline_response": baseline["response"],
                "controlled_response": row["response"],
            }
        )

    ordered_pairs: list[dict[str, object]] = []
    for pair in pairs:
        for controlled_is_a in (True, False):
            ordered = dict(pair)
            ordered.update(
                {
                    "controlled_is_a": controlled_is_a,
                    "answer_a": (
                        pair["controlled_response"]
                        if controlled_is_a
                        else pair["baseline_response"]
                    ),
                    "answer_b": (
                        pair["baseline_response"]
                        if controlled_is_a
                        else pair["controlled_response"]
                    ),
                }
            )
            ordered_pairs.append(ordered)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval().requires_grad_(False)

    ordered_judgments: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batched(ordered_pairs, args.batch_size)):
        prompts = [
            build_local_pairwise_judge_prompt(
                question=str(pair["question"]),
                passage=str(pair["passage"]),
                answer_a=str(pair["answer_a"]),
                answer_b=str(pair["answer_b"]),
            )
            for pair in batch
        ]
        conversations = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        for pair, raw in zip(batch, texts, strict=True):
            winner, reason = parse_local_pairwise_output(raw)
            if winner == "TIE":
                mapped = "tie"
            elif winner == "PARSE_ERROR":
                mapped = "parse_error"
            else:
                winner_is_a = winner == "A"
                mapped = (
                    "controlled"
                    if winner_is_a == bool(pair["controlled_is_a"])
                    else "baseline"
                )
            result = dict(pair)
            result.update(
                {
                    "local_judge_model": args.model,
                    "winner": mapped,
                    "reason": reason,
                    "raw_judgment": raw,
                }
            )
            ordered_judgments.append(result)
        print(
            f"judged {min((batch_index + 1) * args.batch_size, len(ordered_pairs))}/{len(ordered_pairs)}"
        )

    by_pair: dict[tuple[int, str], list[dict[str, object]]] = {}
    for judgment in ordered_judgments:
        key = (int(judgment["source_id"]), str(judgment["method"]))
        by_pair.setdefault(key, []).append(judgment)
    judged: list[dict[str, object]] = []
    for pair in pairs:
        key = (int(pair["source_id"]), str(pair["method"]))
        votes = by_pair[key]
        if len(votes) != 2:
            raise RuntimeError(f"expected two order judgments for {key}")
        mapped_votes = [str(vote["winner"]) for vote in votes]
        if "parse_error" in mapped_votes:
            winner = "parse_error"
        elif mapped_votes[0] == mapped_votes[1]:
            winner = mapped_votes[0]
        else:
            # Any order disagreement is treated conservatively as a tie.
            winner = "tie"
        result = dict(pair)
        result.update(
            {
                "local_judge_model": args.model,
                "winner": winner,
                "order_votes": mapped_votes,
                "order_judgments": [
                    {
                        "controlled_is_a": vote["controlled_is_a"],
                        "winner": vote["winner"],
                        "reason": vote["reason"],
                        "raw_judgment": vote["raw_judgment"],
                    }
                    for vote in votes
                ],
            }
        )
        judged.append(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = []
    for method in sorted({str(row["method"]) for row in judged}):
        selected = [row for row in judged if row["method"] == method]
        counts = Counter(str(row["winner"]) for row in selected)
        summaries.append(
            {
                "method": method,
                "pairs": len(selected),
                "controlled_win_pct": 100 * counts["controlled"] / len(selected),
                "baseline_win_pct": 100 * counts["baseline"] / len(selected),
                "tie_pct": 100 * counts["tie"] / len(selected),
                "parse_error_pct": 100 * counts["parse_error"] / len(selected),
            }
        )
    summary = pd.DataFrame(summaries)
    summary_path = output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
