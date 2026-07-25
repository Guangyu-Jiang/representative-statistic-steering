#!/usr/bin/env python3
"""Evaluate ReDeEP Dolly generations with a local instruction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repstat_steering.redeep_evaluation import (
    build_local_redeep_judge_prompt,
    parse_local_judge_output,
    passage_grounding_ratio,
    reference_token_recall,
    summarize_judged_rows,
    token_f1,
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
        args.input.stem + ".local_judged.jsonl"
    )
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
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

    judged: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batched(rows, args.batch_size)):
        prompts = [
            build_local_redeep_judge_prompt(
                question=str(row["question"]),
                passage=str(row["passage"]),
                reference_answer=str(row["reference_answer"]),
                response=str(row["response"]),
            )
            for row in batch
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
        outputs = generated[:, encoded.input_ids.shape[1] :]
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row, raw_judgment in zip(batch, texts, strict=True):
            label, reason = parse_local_judge_output(raw_judgment)
            response = str(row["response"])
            evaluated = dict(row)
            evaluated.update(
                {
                    "local_judge_model": args.model,
                    "local_judge_label": label,
                    "local_judge_reason": reason,
                    "local_judge_raw": raw_judgment,
                    "reference_token_f1": token_f1(
                        response, str(row["reference_answer"])
                    ),
                    "reference_token_recall": reference_token_recall(
                        response, str(row["reference_answer"])
                    ),
                    "passage_grounding_ratio": passage_grounding_ratio(
                        response, str(row["passage"]), str(row["question"])
                    ),
                }
            )
            judged.append(evaluated)
        print(
            f"judged {min((batch_index + 1) * args.batch_size, len(rows))}/{len(rows)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = pd.DataFrame(summarize_judged_rows(judged))
    summary_path = output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
