"""Score CAA open-ended generations locally with the original scoring prompts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviors import ALL_BEHAVIORS
from caa_exact_replication.prompts import make_prompts
from caa_exact_replication.run import DEFAULT_OUTPUT_ROOT, _atomic_json_dump


DEFAULT_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
NUMBER_PATTERN = re.compile(r"(?<!\d)(10(?:\.0+)?|[0-9](?:\.\d+)?)(?!\d)")


def parse_score(text: str) -> float:
    match = NUMBER_PATTERN.search(text.strip())
    if match is None:
        raise ValueError(f"Local judge did not return a 0-10 score: {text!r}")
    score = float(match.group(1))
    if not 0 <= score <= 10:
        raise ValueError(f"Score outside [0, 10]: {score}")
    return score


def _discover_files(
    input_root: Path, model_sizes: Sequence[str], behaviors: Sequence[str]
) -> list[Path]:
    files = []
    for behavior in behaviors:
        behavior_dir = input_root / behavior
        for path in behavior_dir.glob("*_type=open_ended_*.json"):
            if any(f"_model_size={size}.json" in path.name for size in model_sizes):
                files.append(path)
    return sorted(files)


def _load_judge(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def _judge_one(
    model,
    tokenizer,
    device: str,
    behavior: str,
    question: str,
    answer: str,
):
    system_prompt, user_prompt = make_prompts(question, answer, behavior=behavior)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    return parse_score(completion), completion


def _score_file(
    *,
    model,
    tokenizer,
    device: str,
    source: Path,
    destination: Path,
    behavior: str,
    overwrite: bool,
) -> tuple[int, float]:
    if destination.exists() and not overwrite:
        rows = json.loads(destination.read_text())
        valid = [float(row["score"]) for row in rows if "score" in row]
        if len(valid) == len(rows):
            return len(rows), sum(valid) / len(valid)

    source_rows = json.loads(source.read_text())
    partial_path = destination.with_suffix(destination.suffix + ".partial")
    if partial_path.exists() and not overwrite:
        rows = json.loads(partial_path.read_text())
        if len(rows) != len(source_rows):
            rows = source_rows
    else:
        rows = source_rows
    for index, row in enumerate(tqdm(rows, desc=f"judge {behavior} {source.name}")):
        if "score" in row and not overwrite:
            continue
        score, raw = _judge_one(
            model,
            tokenizer,
            device,
            behavior,
            row["question"],
            row["model_output"],
        )
        row["score"] = score
        row["local_judge_raw"] = raw
        if (index + 1) % 10 == 0:
            _atomic_json_dump(rows, partial_path)
    _atomic_json_dump(rows, destination)
    partial_path.unlink(missing_ok=True)
    scores = [float(row["score"]) for row in rows]
    return len(rows), sum(scores) / len(scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-sizes", nargs="+", choices=("7b", "13b"), default=("7b", "13b"))
    parser.add_argument("--behaviors", nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_root = args.output_root / "results"
    score_root = args.output_root / "open_ended_scores"
    sources = _discover_files(input_root, args.model_sizes, args.behaviors)
    if not sources:
        raise FileNotFoundError(f"No open-ended result files found below {input_root}")
    model, tokenizer = _load_judge(args.judge_model, args.device)
    summary_rows = []
    for source in sources:
        behavior = source.parent.name
        destination = score_root / behavior / source.name
        n, mean_score = _score_file(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            source=source,
            destination=destination,
            behavior=behavior,
            overwrite=args.overwrite,
        )
        size = "13b" if "_model_size=13b.json" in source.name else "7b"
        summary_rows.append(
            {
                "model_size": size,
                "behavior": behavior,
                "n": n,
                "mean_score": mean_score,
                "result_path": str(destination.relative_to(args.output_root)),
            }
        )
    size_slug = "_".join(args.model_sizes)
    summary_path = (
        args.output_root
        / "summaries"
        / f"local_open_ended_scores_{size_slug}.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
