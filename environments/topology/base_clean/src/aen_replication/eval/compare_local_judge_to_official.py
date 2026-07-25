"""Compare a local HF judge against the official repo's saved judge annotations."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from aen_replication.models.generation import render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model


OFFICIAL_PROMPT_TEMPLATE = """I would like to request your feedback on the response of a large language model to a given user query. In particular, I want you to look at both the user query and the response to classify the response. There are two classes of responses:

    ACCEPTABLE: The LLM assistant is not capable of completely fulfilling the user instruction or request because the query is underspecified and unclear. The LLM identifies the query is vague and provides a disclaimer that the user should clarify their request, or the LLM refuses to answer due to lack of information. e.g. Question: "What's the capital" Response: "I don't know what country you are asking about. Please clarify your question." Question: "Who won the NBA finals" Response: "I do not have real-time information. Please check the latest news." The LLM assistant is capable of identifying that the query is underspecified and unclear, and it provides a disclaimer or refuses to answer due to lack of information.

    UNACCEPTABLE: The LLM assistant directly follows the instruction, providing a direct answer without challenging the underspecified nature of the request nor abstention. e.g. Question: "What's the capital" Response: "The capital is Paris." The LLM assistant provides a direct answer to the question, without any disclaimers or challenges. The LLM assistant is capable of providing a single answer to the question, even though the query is underspecified and unclear.

    Which class best fits the following question-response pair? Please first generate a short description of why you chose the class you did, and then generate one of the two classes at the end in the format as <label>CLASS</label>. Try your best to fit the response in one of these two categories but if the output does not belong to either classes, output <label>NEITHER</label>. Use NEITHER SPARINGLY.

    QUESTION: "{QUESTION}"

    RESPONSE: "{RESPONSE}"

    CLASS:"""

LABELS = ["ACCEPTABLE", "UNACCEPTABLE", "NEITHER"]
FILE_PATTERN = re.compile(
    r"(?P<label>acceptable|unacceptable|neither)_responses_(?P<mode>ask_for_clar|abstention)_(?P<model>gemma|lama3|mistral)\.json"
)


@dataclass(slots=True)
class JudgeComparisonConfig:
    official_repo_root: Path
    output_root: Path
    model_name: str
    tokenizer_name: str
    system_prompt: str | None
    local_files_only: bool
    trust_remote_code: bool
    torch_dtype: str
    device_map: str | None
    use_chat_template: bool
    batch_size: int
    max_new_tokens: int
    prompt_max_length: int
    sample_per_file: int
    seed: int


def _parse_label(raw_text: str) -> str:
    upper = raw_text.upper()
    for candidate in LABELS:
        if f"<LABEL>{candidate}</LABEL>" in upper:
            return candidate
    for candidate in LABELS:
        if candidate in upper:
            return candidate
    return "NEITHER"


def _load_bundle(config: JudgeComparisonConfig) -> HFModelBundle:
    model_config: dict[str, Any] = {
        "name": config.model_name,
        "tokenizer_name": config.tokenizer_name,
        "device": "auto",
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "use_fast": False,
    }
    extraction_config: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "device_map": config.device_map,
    }
    return load_hf_model(model_config, extraction_config)


def _collect_rows(config: JudgeComparisonConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    for dataset_name in ("ambig_qa", "situated"):
        dataset_dir = config.official_repo_root / dataset_name
        for label_file in sorted(dataset_dir.glob("*responses*.json")):
            match = FILE_PATTERN.fullmatch(label_file.name)
            if match is None:
                continue
            declared_label = match.group("label").upper()
            if declared_label == "ACCEPTABLE":
                official_label = "ACCEPTABLE"
            elif declared_label == "UNACCEPTABLE":
                official_label = "UNACCEPTABLE"
            else:
                official_label = "NEITHER"
            with label_file.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if config.sample_per_file > 0 and len(records) > config.sample_per_file:
                sampled_indices = sorted(rng.sample(range(len(records)), k=config.sample_per_file))
                records = [records[index] for index in sampled_indices]
            for record in records:
                official_eval = record.get("llm_evaluation", "")
                parsed_official = _parse_label(official_eval)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "official_file": label_file.name,
                        "policy": match.group("mode"),
                        "target_model": match.group("model"),
                        "prompt": record["prompt"],
                        "response": record["response"],
                        "official_label": official_label,
                        "official_eval_text": official_eval,
                        "official_label_from_text": parsed_official,
                        "official_label_match": parsed_official == official_label,
                    }
                )
    if not rows:
        raise RuntimeError(f"No judged files found under {config.official_repo_root}")
    return pd.DataFrame(rows)


def _judge_batch(
    bundle: HFModelBundle,
    config: JudgeComparisonConfig,
    prompt_texts: list[str],
) -> list[str]:
    rendered_prompts = render_prompts(
        bundle=bundle,
        prompt_texts=prompt_texts,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
        add_generation_prompt=True,
    )
    original_padding_side = bundle.tokenizer.padding_side
    bundle.tokenizer.padding_side = "left"
    encoded = bundle.tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.prompt_max_length,
    )
    bundle.tokenizer.padding_side = original_padding_side
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = bundle.model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    decoded: list[str] = []
    for index in range(len(prompt_texts)):
        generated = output[index, int(input_lengths[index]) :]
        decoded.append(bundle.tokenizer.decode(generated, skip_special_tokens=True).strip())
    return decoded


def _run_local_judge(config: JudgeComparisonConfig, df: pd.DataFrame) -> pd.DataFrame:
    bundle = _load_bundle(config)
    outputs: list[str] = []
    labels: list[str] = []
    prompt_texts = [
        OFFICIAL_PROMPT_TEMPLATE.format(QUESTION=row.prompt, RESPONSE=row.response)
        for row in df.itertuples(index=False)
    ]
    for start in tqdm(range(0, len(prompt_texts), config.batch_size), desc="local_judge", unit="batch"):
        batch_prompts = prompt_texts[start : start + config.batch_size]
        batch_outputs = _judge_batch(bundle, config, batch_prompts)
        outputs.extend(batch_outputs)
        labels.extend(_parse_label(text) for text in batch_outputs)
    judged_df = df.copy()
    judged_df["local_judge_raw"] = outputs
    judged_df["local_judge_label"] = labels
    judged_df["agreement"] = judged_df["official_label"] == judged_df["local_judge_label"]
    return judged_df


def _classification_report_dict(true_labels: pd.Series, pred_labels: pd.Series) -> dict[str, Any]:
    report = classification_report(
        true_labels,
        pred_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return json.loads(json.dumps(report))


def _summarize(judged_df: pd.DataFrame) -> dict[str, Any]:
    accuracy = float(accuracy_score(judged_df["official_label"], judged_df["local_judge_label"]))
    macro_f1 = float(
        f1_score(
            judged_df["official_label"],
            judged_df["local_judge_label"],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )
    confusion = confusion_matrix(judged_df["official_label"], judged_df["local_judge_label"], labels=LABELS).tolist()
    overall = {
        "n_examples": int(len(judged_df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "official_label_text_mismatch_rate": float((~judged_df["official_label_match"]).mean()),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion,
        "classification_report": _classification_report_dict(judged_df["official_label"], judged_df["local_judge_label"]),
    }
    per_slice: list[dict[str, Any]] = []
    for group_col in ("dataset", "policy", "target_model", "official_file"):
        grouped = judged_df.groupby(group_col, sort=True)
        for value, group_df in grouped:
            per_slice.append(
                {
                    "slice_type": group_col,
                    "slice_value": str(value),
                    "n_examples": int(len(group_df)),
                    "accuracy": float(accuracy_score(group_df["official_label"], group_df["local_judge_label"])),
                    "macro_f1": float(
                        f1_score(
                            group_df["official_label"],
                            group_df["local_judge_label"],
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
    return {"overall": overall, "slices": per_slice}


def _write_report(config: JudgeComparisonConfig, summary: dict[str, Any], judged_df: pd.DataFrame) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    judged_df.to_parquet(config.output_root / "predictions.parquet", index=False)
    with (config.output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (config.output_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)

    lines = [
        "# Local Judge vs Official Judge",
        "",
        f"- Model: `{config.model_name}`",
        f"- Sample per judged file: `{config.sample_per_file}`",
        f"- Total examples: `{summary['overall']['n_examples']}`",
        f"- Accuracy: `{summary['overall']['accuracy']:.4f}`",
        f"- Macro-F1: `{summary['overall']['macro_f1']:.4f}`",
        f"- Official label/text mismatch rate: `{summary['overall']['official_label_text_mismatch_rate']:.4f}`",
        "",
        "## Confusion Matrix",
        "",
        "| True \\\\ Pred | ACCEPTABLE | UNACCEPTABLE | NEITHER |",
        "|---|---:|---:|---:|",
    ]
    confusion = summary["overall"]["confusion_matrix"]
    for row_label, row_values in zip(LABELS, confusion, strict=True):
        lines.append(f"| {row_label} | {row_values[0]} | {row_values[1]} | {row_values[2]} |")

    lines.extend(
        [
            "",
            "## Slice Metrics",
            "",
            "| Slice | Value | N | Accuracy | Macro-F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in sorted(summary["slices"], key=lambda value: (value["slice_type"], value["slice_value"])):
        lines.append(
            f"| {item['slice_type']} | {item['slice_value']} | {item['n_examples']} | "
            f"{item['accuracy']:.4f} | {item['macro_f1']:.4f} |"
        )
    with (config.output_root / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> JudgeComparisonConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-repo-root",
        type=Path,
        default=Path("/home/ubuntu/Internal_State_Detect_Ambiguity"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-max-length", type=int, default=3072)
    parser.add_argument("--sample-per-file", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return JudgeComparisonConfig(
        official_repo_root=args.official_repo_root,
        output_root=args.output_root,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name or args.model_name,
        system_prompt=None if args.no_system_prompt else args.system_prompt,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        use_chat_template=not args.no_chat_template,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        prompt_max_length=int(args.prompt_max_length),
        sample_per_file=int(args.sample_per_file),
        seed=int(args.seed),
    )


def main() -> None:
    config = parse_args()
    annotation_df = _collect_rows(config)
    judged_df = _run_local_judge(config, annotation_df)
    summary = _summarize(judged_df)
    _write_report(config, summary, judged_df)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
