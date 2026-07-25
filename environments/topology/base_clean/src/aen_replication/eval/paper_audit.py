"""Generate a paper-to-artifact replication audit."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader

from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, read_json, write_markdown

LOGGER = logging.getLogger(__name__)

MODEL_ORDER = ["Mistral 7B", "LLaMA 3.1 8B", "Gemma 7B"]
DATASET_ORDER = ["ambigqa", "situatedqa"]
STEERING_METHOD_ORDER = ["AENs", "Top 50 Neurons", "Top 100 Neurons", "Full Vector"]
BASELINE_METHOD_ORDER = [
    "CLAM-FewShot",
    "CLAMBER-ZeroShot",
    "CLAMBER-FewShotWithCoT",
    "INFOGAIN",
    "Ambiguity-Encoding Neurons only",
    "Full probe",
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    exact_detection_slug: str
    closest_baseline_slug: str
    closest_steering_slug: str


MODEL_SPECS = {
    "Mistral 7B": ModelSpec(
        label="Mistral 7B",
        exact_detection_slug="mistralai_mistral_7b_instruct_v0_3",
        closest_baseline_slug="mistralai_mistral_7b_instruct_v0_2",
        closest_steering_slug="mistralai_mistral_7b_instruct_v0_2",
    ),
    "LLaMA 3.1 8B": ModelSpec(
        label="LLaMA 3.1 8B",
        exact_detection_slug="meta_llama_llama_3_1_8b_instruct",
        closest_baseline_slug="meta_llama_meta_llama_3_8b_instruct",
        closest_steering_slug="meta_llama_meta_llama_3_8b_instruct",
    ),
    "Gemma 7B": ModelSpec(
        label="Gemma 7B",
        exact_detection_slug="google_gemma_7b_it",
        closest_baseline_slug="google_gemma_7b_it",
        closest_steering_slug="google_gemma_7b_it",
    ),
}


@dataclass(frozen=True)
class ArtifactResolution:
    slug: str
    exact: bool


PAPER_TABLE_1 = {
    ("ambigqa", "Mistral 7B"): {"accuracy": 93.30, "precision": 93.48, "recall": 93.30, "f1": 93.29},
    ("ambigqa", "LLaMA 3.1 8B"): {"accuracy": 90.65, "precision": 91.79, "recall": 90.65, "f1": 90.59},
    ("ambigqa", "Gemma 7B"): {"accuracy": 95.25, "precision": 95.53, "recall": 95.25, "f1": 95.24},
    ("situatedqa", "Mistral 7B"): {"accuracy": 94.14, "precision": 94.57, "recall": 94.15, "f1": 94.14},
    ("situatedqa", "LLaMA 3.1 8B"): {"accuracy": 95.40, "precision": 95.74, "recall": 95.40, "f1": 95.39},
    ("situatedqa", "Gemma 7B"): {"accuracy": 97.10, "precision": 97.12, "recall": 97.10, "f1": 97.10},
}

PAPER_TABLE_2 = {
    ("ambigqa", "Mistral 7B"): [2070, 3240, 2043, 1909, 1372],
    ("situatedqa", "Mistral 7B"): [2070, 2388, 2078, 53, 2083],
    ("ambigqa", "LLaMA 3.1 8B"): [788, 1384, 4062, 4055, 1298],
    ("situatedqa", "LLaMA 3.1 8B"): [788, 1384, 4062, 4055, 3231],
    ("ambigqa", "Gemma 7B"): [1995, 1963, 1496, 1288, 2217],
    ("situatedqa", "Gemma 7B"): [1995, 1258, 1355, 1884, 155],
}

PAPER_TABLE_3 = {
    ("ambigqa", "AENs", "Mistral 7B"): 18.0,
    ("ambigqa", "AENs", "LLaMA 3.1 8B"): 52.0,
    ("ambigqa", "AENs", "Gemma 7B"): 13.2,
    ("ambigqa", "Top 50 Neurons", "Mistral 7B"): 27.4,
    ("ambigqa", "Top 50 Neurons", "LLaMA 3.1 8B"): 54.6,
    ("ambigqa", "Top 50 Neurons", "Gemma 7B"): 20.0,
    ("ambigqa", "Top 100 Neurons", "Mistral 7B"): 38.4,
    ("ambigqa", "Top 100 Neurons", "LLaMA 3.1 8B"): 58.2,
    ("ambigqa", "Top 100 Neurons", "Gemma 7B"): 28.8,
    ("ambigqa", "Full Vector", "Mistral 7B"): 68.8,
    ("ambigqa", "Full Vector", "LLaMA 3.1 8B"): 62.8,
    ("ambigqa", "Full Vector", "Gemma 7B"): 53.6,
    ("situatedqa", "AENs", "Mistral 7B"): 23.8,
    ("situatedqa", "AENs", "LLaMA 3.1 8B"): 50.4,
    ("situatedqa", "AENs", "Gemma 7B"): 11.6,
    ("situatedqa", "Top 50 Neurons", "Mistral 7B"): 32.8,
    ("situatedqa", "Top 50 Neurons", "LLaMA 3.1 8B"): 62.6,
    ("situatedqa", "Top 50 Neurons", "Gemma 7B"): 16.0,
    ("situatedqa", "Top 100 Neurons", "Mistral 7B"): 35.4,
    ("situatedqa", "Top 100 Neurons", "LLaMA 3.1 8B"): 74.0,
    ("situatedqa", "Top 100 Neurons", "Gemma 7B"): 17.6,
    ("situatedqa", "Full Vector", "Mistral 7B"): 73.6,
    ("situatedqa", "Full Vector", "LLaMA 3.1 8B"): 93.2,
    ("situatedqa", "Full Vector", "Gemma 7B"): 56.8,
}

PAPER_TABLE_4 = {
    ("ambigqa", "LLaMA 3.1 8B"): 89.9,
    ("ambigqa", "Mistral 7B"): 98.5,
    ("ambigqa", "Gemma 7B"): 89.2,
    ("situatedqa", "LLaMA 3.1 8B"): 90.6,
    ("situatedqa", "Mistral 7B"): 96.0,
    ("situatedqa", "Gemma 7B"): 88.7,
}

PAPER_TABLE_5 = {
    ("ambigqa", "LLaMA 3.1 8B"): 98.8,
    ("ambigqa", "Mistral 7B"): 94.6,
    ("ambigqa", "Gemma 7B"): 97.0,
    ("situatedqa", "LLaMA 3.1 8B"): 95.2,
    ("situatedqa", "Mistral 7B"): 92.6,
    ("situatedqa", "Gemma 7B"): 95.8,
}

PAPER_TABLE_6 = {
    ("ambigqa", "LLaMA 3.1 8B"): 56.2,
    ("ambigqa", "Mistral 7B"): 20.2,
    ("ambigqa", "Gemma 7B"): 18.4,
    ("situatedqa", "LLaMA 3.1 8B"): 52.6,
    ("situatedqa", "Mistral 7B"): 22.6,
    ("situatedqa", "Gemma 7B"): 16.6,
}

PAPER_TABLE_8 = {
    ("ambigqa", "CLAM-FewShot", "Mistral 7B"): (52.98, 45.25),
    ("ambigqa", "CLAM-FewShot", "LLaMA 3.1 8B"): (60.28, 58.26),
    ("ambigqa", "CLAM-FewShot", "Gemma 7B"): (49.33, 35.72),
    ("ambigqa", "CLAMBER-ZeroShot", "Mistral 7B"): (49.59, 34.36),
    ("ambigqa", "CLAMBER-ZeroShot", "LLaMA 3.1 8B"): (52.60, 52.19),
    ("ambigqa", "CLAMBER-ZeroShot", "Gemma 7B"): (51.93, 44.50),
    ("ambigqa", "CLAMBER-FewShotWithCoT", "Mistral 7B"): (50.88, 37.83),
    ("ambigqa", "CLAMBER-FewShotWithCoT", "LLaMA 3.1 8B"): (52.00, 42.80),
    ("ambigqa", "CLAMBER-FewShotWithCoT", "Gemma 7B"): (48.42, 48.25),
    ("ambigqa", "INFOGAIN", "Mistral 7B"): (59.50, 59.18),
    ("ambigqa", "INFOGAIN", "LLaMA 3.1 8B"): (54.25, 45.19),
    ("ambigqa", "INFOGAIN", "Gemma 7B"): (55.75, 55.19),
    ("ambigqa", "Ambiguity-Encoding Neurons only", "Mistral 7B"): (90.30, 90.28),
    ("ambigqa", "Ambiguity-Encoding Neurons only", "LLaMA 3.1 8B"): (88.60, 88.55),
    ("ambigqa", "Ambiguity-Encoding Neurons only", "Gemma 7B"): (92.00, 91.97),
    ("ambigqa", "Full probe", "Mistral 7B"): (93.30, 93.29),
    ("ambigqa", "Full probe", "LLaMA 3.1 8B"): (90.65, 90.59),
    ("ambigqa", "Full probe", "Gemma 7B"): (95.25, 95.24),
    ("situatedqa", "CLAM-FewShot", "Mistral 7B"): (58.53, 54.02),
    ("situatedqa", "CLAM-FewShot", "LLaMA 3.1 8B"): (50.30, 46.04),
    ("situatedqa", "CLAM-FewShot", "Gemma 7B"): (48.34, 32.80),
    ("situatedqa", "CLAMBER-ZeroShot", "Mistral 7B"): (51.32, 38.75),
    ("situatedqa", "CLAMBER-ZeroShot", "LLaMA 3.1 8B"): (54.65, 54.39),
    ("situatedqa", "CLAMBER-ZeroShot", "Gemma 7B"): (50.40, 40.62),
    ("situatedqa", "CLAMBER-FewShotWithCoT", "Mistral 7B"): (47.21, 45.95),
    ("situatedqa", "CLAMBER-FewShotWithCoT", "LLaMA 3.1 8B"): (50.68, 44.20),
    ("situatedqa", "CLAMBER-FewShotWithCoT", "Gemma 7B"): (47.10, 46.91),
    ("situatedqa", "INFOGAIN", "Mistral 7B"): (62.10, 61.85),
    ("situatedqa", "INFOGAIN", "LLaMA 3.1 8B"): (55.75, 47.88),
    ("situatedqa", "INFOGAIN", "Gemma 7B"): (61.30, 61.05),
    ("situatedqa", "Ambiguity-Encoding Neurons only", "Mistral 7B"): (92.35, 92.32),
    ("situatedqa", "Ambiguity-Encoding Neurons only", "LLaMA 3.1 8B"): (94.00, 93.98),
    ("situatedqa", "Ambiguity-Encoding Neurons only", "Gemma 7B"): (96.90, 96.90),
    ("situatedqa", "Full probe", "Mistral 7B"): (94.14, 94.14),
    ("situatedqa", "Full probe", "LLaMA 3.1 8B"): (95.40, 95.39),
    ("situatedqa", "Full probe", "Gemma 7B"): (97.10, 97.10),
}


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value)


def _flatten_text(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _dataset_label(dataset: str) -> str:
    return "AmbigQA" if dataset == "ambigqa" else "SituatedQA"


def _method_key_to_label(method_key: str) -> str:
    mapping = {
        "clam_fewshot": "CLAM-FewShot",
        "clamber_zeroshot": "CLAMBER-ZeroShot",
        "clamber_fewshot_cot": "CLAMBER-FewShotWithCoT",
        "infogain": "INFOGAIN",
    }
    return mapping[method_key]


def _read_detection_summary(project_root: Path, slug: str) -> dict[str, Any]:
    return read_json(project_root / "artifacts" / "reports" / slug / "detection_summary.json")


def _read_default_layer_report(project_root: Path, slug: str, dataset: str) -> dict[str, Any]:
    return read_json(project_root / "artifacts" / "probes" / slug / f"{dataset}_default_layer_report.json")


def _read_baseline_summary(project_root: Path, slug: str, dataset: str) -> dict[str, Any]:
    return read_json(project_root / "artifacts" / "baselines" / slug / dataset / "summary.json")


def _read_steering_summary(project_root: Path, slug: str) -> dict[str, Any]:
    return read_json(project_root / "artifacts" / "steering" / slug / "summary.json")


def _resolve_baseline_artifact(project_root: Path, spec: ModelSpec, dataset: str) -> ArtifactResolution:
    for slug in [spec.exact_detection_slug, spec.closest_baseline_slug]:
        path = project_root / "artifacts" / "baselines" / slug / dataset / "summary.json"
        if path.exists():
            return ArtifactResolution(slug=slug, exact=(slug == spec.exact_detection_slug))
    raise FileNotFoundError(
        f"Could not locate baseline summary for dataset={dataset} using slugs "
        f"{[spec.exact_detection_slug, spec.closest_baseline_slug]}"
    )


def _resolve_steering_artifact(project_root: Path, spec: ModelSpec) -> ArtifactResolution:
    for slug in [spec.exact_detection_slug, spec.closest_steering_slug]:
        path = project_root / "artifacts" / "steering" / slug / "summary.json"
        if path.exists():
            return ArtifactResolution(slug=slug, exact=(slug == spec.exact_detection_slug))
    raise FileNotFoundError(
        f"Could not locate steering summary using slugs {[spec.exact_detection_slug, spec.closest_steering_slug]}"
    )


def _infer_hidden_size(project_root: Path, preferred_slugs: list[str]) -> int:
    hidden_root = project_root / "artifacts" / "hidden_states"
    for slug in preferred_slugs:
        slug_root = hidden_root / slug
        if not slug_root.exists():
            continue
        for metadata_path in sorted(slug_root.glob("*.metadata.json")):
            metadata = read_json(metadata_path)
            hidden_size = metadata.get("hidden_size")
            if hidden_size is not None:
                return int(hidden_size)
    raise FileNotFoundError(f"Could not infer hidden size from hidden-state caches for {preferred_slugs}")


def extract_paper_inventory(pdf_path: str | Path) -> pd.DataFrame:
    """Extract a simple figure/table inventory from the local paper PDF."""

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    items: list[dict[str, Any]] = []
    for item_type, max_index in (("Figure", 10), ("Table", 8)):
        for index in range(1, max_index + 1):
            token = f"{item_type} {index}"
            match = re.search(rf"{re.escape(token)}:?\s*(.+)", text)
            snippet = ""
            if match:
                snippet = _flatten_text(match.group(0), limit=260)
            items.append(
                {
                    "item_type": item_type.lower(),
                    "item_number": index,
                    "item_id": f"{item_type.lower()}_{index}",
                    "token": token,
                    "snippet": snippet,
                    "found_in_pdf": bool(match),
                }
            )
    return pd.DataFrame(items)


def build_table_1(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        summary = _read_detection_summary(project_root, spec.exact_detection_slug)
        for dataset in DATASET_ORDER:
            paper = PAPER_TABLE_1[(dataset, model_label)]
            payload = summary["datasets"][dataset]["full_probe_test"]
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "paper_accuracy": paper["accuracy"],
                    "rep_accuracy": _pct(payload["accuracy"]),
                    "accuracy_diff": _pct(payload["accuracy"]) - paper["accuracy"],
                    "paper_precision": paper["precision"],
                    "rep_precision": _pct(payload["precision"]),
                    "precision_diff": _pct(payload["precision"]) - paper["precision"],
                    "paper_recall": paper["recall"],
                    "rep_recall": _pct(payload["recall"]),
                    "recall_diff": _pct(payload["recall"]) - paper["recall"],
                    "paper_f1": paper["f1"],
                    "rep_f1": _pct(payload["f1"]),
                    "f1_diff": _pct(payload["f1"]) - paper["f1"],
                    "artifact_model_exact": True,
                }
            )
    return pd.DataFrame(rows)


def build_table_2(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        for dataset in DATASET_ORDER:
            report = _read_default_layer_report(project_root, spec.exact_detection_slug, dataset)
            rep_top5 = [int(value) for value in report["top_5_weights"]]
            paper_top5 = PAPER_TABLE_2[(dataset, model_label)]
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "paper_top_5": paper_top5,
                    "rep_top_5": rep_top5,
                    "paper_rep_overlap": sorted(set(paper_top5) & set(rep_top5)),
                    "paper_rep_overlap_count": len(set(paper_top5) & set(rep_top5)),
                    "rep_aen_indices": [int(value) for value in report["aen_selection"]["aen_indices"]],
                    "rep_aen_k": int(report["aen_selection"]["aen_k"]),
                }
            )
    return pd.DataFrame(rows)


def build_table_3(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_by_model: dict[str, tuple[ArtifactResolution, dict[str, Any]]] = {}
    for model_label, spec in MODEL_SPECS.items():
        resolution = _resolve_steering_artifact(project_root, spec)
        summary_by_model[model_label] = (resolution, _read_steering_summary(project_root, resolution.slug))
    method_to_key = {
        "AENs": "aens",
        "Top 50 Neurons": "top_50",
        "Top 100 Neurons": "top_100",
        "Full Vector": "full_vector",
    }
    for dataset in DATASET_ORDER:
        for method_label in STEERING_METHOD_ORDER:
            for model_label in MODEL_ORDER:
                spec = MODEL_SPECS[model_label]
                resolution, summary = summary_by_model[model_label]
                rep_payload = summary["datasets"][dataset][method_to_key[method_label]]
                rep_value = _pct(rep_payload["abstention_rate"])
                paper_value = PAPER_TABLE_3[(dataset, method_label, model_label)]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method_label,
                        "model_label": model_label,
                        "paper_abstention_rate": paper_value,
                        "rep_abstention_rate": rep_value,
                        "difference": rep_value - paper_value,
                        "artifact_model_name": summary["model_name"],
                        "artifact_model_exact": resolution.exact,
                    }
                )
    return pd.DataFrame(rows)


def build_table_4(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        summary_path = project_root / "artifacts" / "reports" / spec.exact_detection_slug / "triviaqa_aen_false_positive.json"
        summary = read_json(summary_path) if summary_path.exists() else None
        for dataset in DATASET_ORDER:
            if summary is None or dataset not in summary.get("train_datasets", {}):
                rows.append(
                    {
                        "dataset": dataset,
                        "model_label": model_label,
                        "paper_accuracy": PAPER_TABLE_4[(dataset, model_label)],
                        "rep_accuracy": np.nan,
                        "difference": np.nan,
                        "artifact_model_exact": False,
                        "status": "missing_no_triviaqa_evaluation_artifact",
                    }
                )
                continue
            rep_accuracy = _pct(summary["train_datasets"][dataset]["accuracy"])
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "paper_accuracy": PAPER_TABLE_4[(dataset, model_label)],
                    "rep_accuracy": rep_accuracy,
                    "difference": rep_accuracy - PAPER_TABLE_4[(dataset, model_label)],
                    "artifact_model_exact": True,
                    "status": "generated",
                }
            )
    return pd.DataFrame(rows)


def build_table_5(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        summary = _read_steering_summary(project_root, resolution.slug)
        for dataset in DATASET_ORDER:
            rep_value = _pct(summary["datasets"][dataset]["aens_consistency"]["abstention_consistency"])
            paper_value = PAPER_TABLE_5[(dataset, model_label)]
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "paper_consistency": paper_value,
                    "rep_consistency": rep_value,
                    "difference": rep_value - paper_value,
                    "artifact_model_name": summary["model_name"],
                    "artifact_model_exact": resolution.exact,
                }
            )
    return pd.DataFrame(rows)


def build_table_6(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        summary = _read_steering_summary(project_root, resolution.slug)
        for dataset in DATASET_ORDER:
            rep_value = _pct(summary["datasets"][dataset]["aens_reverse"]["direct_answer_rate"])
            paper_value = PAPER_TABLE_6[(dataset, model_label)]
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "paper_direct_answer_rate": paper_value,
                    "rep_direct_answer_rate": rep_value,
                    "difference": rep_value - paper_value,
                    "artifact_model_name": summary["model_name"],
                    "artifact_model_exact": resolution.exact,
                }
            )
    return pd.DataFrame(rows)


def build_table_8(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        detection = _read_detection_summary(project_root, spec.exact_detection_slug)
        for dataset in DATASET_ORDER:
            resolution = _resolve_baseline_artifact(project_root, spec, dataset)
            baseline_summary = _read_baseline_summary(project_root, resolution.slug, dataset)
            for method_label in BASELINE_METHOD_ORDER:
                if method_label == "Ambiguity-Encoding Neurons only":
                    metrics = detection["datasets"][dataset]["aen_probe_test"]
                    rep_accuracy = _pct(metrics["accuracy"])
                    rep_macro_f1 = _pct(metrics["macro_f1"])
                    artifact_exact = True
                    artifact_model_name = detection["model_name"]
                elif method_label == "Full probe":
                    metrics = detection["datasets"][dataset]["full_probe_test"]
                    rep_accuracy = _pct(metrics["accuracy"])
                    rep_macro_f1 = _pct(metrics["macro_f1"])
                    artifact_exact = True
                    artifact_model_name = detection["model_name"]
                else:
                    payload = baseline_summary["methods"][
                        {
                            "CLAM-FewShot": "clam_fewshot",
                            "CLAMBER-ZeroShot": "clamber_zeroshot",
                            "CLAMBER-FewShotWithCoT": "clamber_fewshot_cot",
                            "INFOGAIN": "infogain",
                        }[method_label]
                    ]
                    rep_accuracy = _pct(payload["accuracy"])
                    rep_macro_f1 = _pct(payload["macro_f1"])
                    artifact_exact = resolution.exact
                    artifact_model_name = baseline_summary["model_name"]
                paper_accuracy, paper_macro_f1 = PAPER_TABLE_8[(dataset, method_label, model_label)]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method_label,
                        "model_label": model_label,
                        "paper_accuracy": paper_accuracy,
                        "rep_accuracy": rep_accuracy,
                        "accuracy_diff": rep_accuracy - paper_accuracy,
                        "paper_macro_f1": paper_macro_f1,
                        "rep_macro_f1": rep_macro_f1,
                        "macro_f1_diff": rep_macro_f1 - paper_macro_f1,
                        "artifact_model_name": artifact_model_name,
                        "artifact_model_exact": artifact_exact,
                    }
                )
    return pd.DataFrame(rows)


def build_table_7(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        steering_root = project_root / "artifacts" / "steering" / resolution.slug
        found = False
        for dataset in DATASET_ORDER:
            base_path = steering_root / f"{dataset}__base_behavior.parquet"
            steered_path = steering_root / f"{dataset}__aens.parquet"
            if not (base_path.exists() and steered_path.exists()):
                continue
            base_df = pd.read_parquet(base_path)
            steered_df = pd.read_parquet(steered_path)
            merged = base_df.merge(
                steered_df[["example_id", "response_text", "judge_label"]],
                on="example_id",
                suffixes=("_before", "_after"),
            )
            candidates = merged.loc[
                merged["judge_label_before"].eq("UNACCEPTABLE") & merged["judge_label_after"].eq("ACCEPTABLE")
            ]
            if candidates.empty:
                continue
            row = candidates.iloc[0]
            rows.append(
                {
                    "model_label": model_label,
                    "dataset": dataset,
                    "artifact_model_exact": resolution.exact,
                    "question": _flatten_text(str(row["text"]), limit=120),
                    "before_response": _flatten_text(str(row["response_text_before"]), limit=180),
                    "after_response": _flatten_text(str(row["response_text_after"]), limit=180),
                }
            )
            found = True
            break
        if not found:
            rows.append(
                {
                    "model_label": model_label,
                    "dataset": "",
                    "artifact_model_exact": resolution.exact,
                    "question": "No improved example found in current steering artifacts.",
                    "before_response": "",
                    "after_response": "",
                }
            )
    return pd.DataFrame(rows)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure_2(project_root: Path, output_path: Path) -> None:
    table8 = build_table_8(project_root)
    methods = BASELINE_METHOD_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(methods)))
    x = np.arange(len(MODEL_ORDER))
    width = 0.12
    for ax, dataset in zip(axes, DATASET_ORDER, strict=True):
        subset = table8.loc[table8["dataset"].eq(dataset)]
        for idx, method in enumerate(methods):
            method_subset = subset.loc[subset["method"].eq(method)].set_index("model_label").reindex(MODEL_ORDER)
            ax.bar(x + (idx - 2.5) * width, method_subset["rep_accuracy"], width=width, label=method, color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel("Accuracy (%)")
        _style_axes(ax, f"Figure 2 Replication: {_dataset_label(dataset)}")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.02, 1.25), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_3(project_root: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = [0, 1, 2, 3, 5]
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        for dataset in DATASET_ORDER:
            report = _read_default_layer_report(project_root, spec.exact_detection_slug, dataset)
            aen_selection = report["aen_selection"]
            accuracies = [100.0 * float(aen_selection["baseline_accuracy"])]
            perturb_results = {int(item["k"]): 100.0 * float(item["accuracy_after_perturb"]) for item in aen_selection["results"]}
            accuracies.extend(perturb_results[k] for k in [1, 2, 3, 5])
            ax.plot(x_positions, accuracies, marker="o", label=f"{model_label} ({_dataset_label(dataset)})")
    ax.set_xlabel("Number of Top Neurons Perturbed")
    ax.set_ylabel("Probe Accuracy (%)")
    _style_axes(ax, "Figure 3 Replication: Perturbation Sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_4(project_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model_label in zip(axes, MODEL_ORDER, strict=True):
        spec = MODEL_SPECS[model_label]
        summary = _read_detection_summary(project_root, spec.exact_detection_slug)
        matrix = np.array(
            [
                [
                    _pct(summary["datasets"]["ambigqa"]["aen_probe_test"]["accuracy"]),
                    _pct(summary["datasets"]["ambigqa_to_situatedqa"]["accuracy"]),
                ],
                [
                    _pct(summary["datasets"]["situatedqa_to_ambigqa"]["accuracy"]),
                    _pct(summary["datasets"]["situatedqa"]["aen_probe_test"]["accuracy"]),
                ],
            ]
        )
        image = ax.imshow(matrix, cmap="Blues", vmin=50, vmax=100)
        ax.set_xticks([0, 1], labels=["AmbigQA", "SituatedQA"])
        ax.set_yticks([0, 1], labels=["AmbigQA", "SituatedQA"])
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title(model_label)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Accuracy (%)")
    fig.suptitle("Figure 4 Replication: Cross-Domain AEN Probe Accuracy", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_aen_k(project_root: Path, slug: str, dataset: str) -> int:
    report = _read_default_layer_report(project_root, slug, dataset)
    return int(report["aen_selection"]["aen_k"])


def plot_figure_5(project_root: Path, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        steering_summary = _read_steering_summary(project_root, resolution.slug)
        hidden_size = _infer_hidden_size(project_root, [resolution.slug, spec.exact_detection_slug])
        for dataset in DATASET_ORDER:
            aen_k = _load_aen_k(project_root, resolution.slug, dataset)
            for method_label, method_key, neuron_count in (
                ("AENs", "aens", aen_k),
                ("Top 50 Neurons", "top_50", 50),
                ("Top 100 Neurons", "top_100", 100),
                ("Full Vector", "full_vector", hidden_size),
            ):
                abstention_rate = _pct(steering_summary["datasets"][dataset][method_key]["abstention_rate"])
                rows.append(
                    {
                        "dataset": dataset,
                        "model_label": model_label,
                        "method": method_label,
                        "gain_per_neuron": abstention_rate / neuron_count,
                    }
                )
    gain_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(STEERING_METHOD_ORDER)))
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    for ax, dataset in zip(axes, DATASET_ORDER, strict=True):
        subset = gain_df.loc[gain_df["dataset"].eq(dataset)]
        for idx, method in enumerate(STEERING_METHOD_ORDER):
            method_subset = subset.loc[subset["method"].eq(method)].set_index("model_label").reindex(MODEL_ORDER)
            ax.bar(x + (idx - 1.5) * width, method_subset["gain_per_neuron"], width=width, label=method, color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel("Abstention gain per neuron (%)")
        _style_axes(ax, f"Figure 5 Replication: {_dataset_label(dataset)}")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.22), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_6(project_root: Path, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        summary = _read_steering_summary(project_root, resolution.slug)
        for dataset in DATASET_ORDER:
            aen_rate = float(summary["datasets"][dataset]["aens"]["abstention_rate"])
            full_rate = float(summary["datasets"][dataset]["full_vector"]["abstention_rate"])
            proportion = 0.0 if full_rate <= 1e-12 else 100.0 * (aen_rate / full_rate)
            rows.append(
                {
                    "label": f"{_dataset_label(dataset)}\n{model_label}",
                    "aen_share": proportion,
                    "other_share": max(0.0, 100.0 - proportion),
                }
            )
    share_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(share_df))
    ax.barh(y, share_df["aen_share"], color="#4C78A8", label="AENs")
    ax.barh(y, share_df["other_share"], left=share_df["aen_share"], color="#E45756", label="Other Neurons")
    ax.set_yticks(y, share_df["label"])
    ax.set_xlabel("Proportion of Full-Vector Abstention Effect (%)")
    _style_axes(ax, "Figure 6 Replication: AEN Share of Full-Vector Steering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_7(project_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, dataset in enumerate(DATASET_ORDER):
        for col_index, model_label in enumerate(MODEL_ORDER):
            spec = MODEL_SPECS[model_label]
            layerwise_path = project_root / "artifacts" / "reports" / spec.exact_detection_slug / f"{dataset}_layerwise.csv"
            df = pd.read_csv(layerwise_path)
            ax = axes[row_index, col_index]
            ax.plot(df["layer"], 100.0 * df["full_accuracy"], label="Full probe", linewidth=2)
            ax.plot(df["layer"], 100.0 * df["aen_accuracy"], label="AENs-only", linestyle="--", linewidth=2)
            ax.set_title(f"{model_label} — {_dataset_label(dataset)}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Figure 7 Replication: Layerwise Probing")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_distribution_inputs(project_root: Path, slug: str, dataset: str) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    report = _read_default_layer_report(project_root, slug, dataset)
    cache_path = project_root / "artifacts" / "hidden_states" / slug / f"{dataset}__layer_14__mean_pool.parquet"
    metadata, matrix = load_hidden_state_table(cache_path)
    labels = metadata["label_ambiguous"].to_numpy(dtype=int)
    ranked = [int(value) for value in report["ranked_indices"]]
    aen_indices = [int(value) for value in report["aen_selection"]["aen_indices"]]
    aen_k = int(report["aen_selection"]["aen_k"])
    neighbor_index = ranked[aen_k] if aen_k < len(ranked) else ranked[-1]
    return matrix, labels, aen_indices, neighbor_index


def _plot_density(ax: plt.Axes, clear_values: np.ndarray, ambig_values: np.ndarray, title: str) -> None:
    bins = 40
    ax.hist(clear_values, bins=bins, density=True, histtype="step", linewidth=2, label="Clear")
    ax.hist(ambig_values, bins=bins, density=True, histtype="step", linewidth=2, label="Ambiguous")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)


def plot_figure_8(project_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for row_index, model_label in enumerate(MODEL_ORDER):
        spec = MODEL_SPECS[model_label]
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            matrix, labels, aen_indices, neighbor_index = _load_distribution_inputs(
                project_root, spec.exact_detection_slug, dataset
            )
            aen_index = aen_indices[0]
            for local_col, neuron_index, prefix in (
                (0, aen_index, "AEN"),
                (1, neighbor_index, "Neighbor"),
            ):
                ax = axes[row_index, dataset_index * 2 + local_col]
                clear_values = matrix[labels == 0, neuron_index]
                ambig_values = matrix[labels == 1, neuron_index]
                delta_mu = float(np.mean(ambig_values) - np.mean(clear_values))
                _plot_density(
                    ax,
                    clear_values=clear_values,
                    ambig_values=ambig_values,
                    title=f"{model_label} — {_dataset_label(dataset)}\n{prefix} {neuron_index} | Δμ={delta_mu:.3f}",
                )
                if row_index == 0 and dataset_index == 0 and local_col == 0:
                    ax.legend()
    fig.suptitle("Figure 8 Replication: Activation Distributions for AEN vs Neighbor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_9(project_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for row_index, model_label in enumerate(MODEL_ORDER):
        spec = MODEL_SPECS[model_label]
        for col_index, dataset in enumerate(DATASET_ORDER):
            report = _read_default_layer_report(project_root, spec.exact_detection_slug, dataset)
            cache_path = project_root / "artifacts" / "hidden_states" / spec.exact_detection_slug / f"{dataset}__layer_14__mean_pool.parquet"
            metadata, matrix = load_hidden_state_table(cache_path)
            labels = metadata["label_ambiguous"].to_numpy(dtype=int)
            ranked = [int(value) for value in report["ranked_indices"][:50]]
            aen_indices = {int(value) for value in report["aen_selection"]["aen_indices"]}
            clear = matrix[labels == 0]
            ambig = matrix[labels == 1]
            delta_mu = np.abs(np.mean(clear[:, ranked], axis=0) - np.mean(ambig[:, ranked], axis=0))
            ax = axes[row_index, col_index]
            ax.plot(np.arange(1, len(ranked) + 1), delta_mu, linewidth=1.5, color="#4C78A8")
            highlighted_x = [idx + 1 for idx, neuron in enumerate(ranked) if neuron in aen_indices]
            highlighted_y = [delta_mu[idx] for idx, neuron in enumerate(ranked) if neuron in aen_indices]
            if highlighted_x:
                ax.scatter(highlighted_x, highlighted_y, color="#E45756", label="AEN", zorder=3)
            ax.set_yscale("log")
            ax.set_title(f"{model_label} — {_dataset_label(dataset)}")
            ax.set_xlabel("Neuron Rank")
            ax.set_ylabel("|Δμ|")
            ax.grid(True, alpha=0.25)
            if row_index == 0 and col_index == 0:
                ax.legend()
    fig.suptitle("Figure 9 Replication: |Δμ| Across Top-50 Ranked Neurons")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_10(project_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model_label in zip(axes, MODEL_ORDER, strict=True):
        spec = MODEL_SPECS[model_label]
        resolution = _resolve_steering_artifact(project_root, spec)
        summary = _read_steering_summary(project_root, resolution.slug)
        matrix = np.array(
            [
                [
                    _pct(summary["datasets"]["ambigqa"]["aens"]["abstention_rate"]),
                    _pct(summary["datasets"]["ambigqa_to_situatedqa"]["abstention_rate"]),
                ],
                [
                    _pct(summary["datasets"]["situatedqa_to_ambigqa"]["abstention_rate"]),
                    _pct(summary["datasets"]["situatedqa"]["aens"]["abstention_rate"]),
                ],
            ]
        )
        image = ax.imshow(matrix, cmap="Oranges", vmin=0, vmax=max(10.0, float(np.max(matrix))))
        ax.set_xticks([0, 1], labels=["AmbigQA", "SituatedQA"])
        ax.set_yticks([0, 1], labels=["AmbigQA", "SituatedQA"])
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Vector Extraction Dataset")
        ax.set_title(model_label + ("" if resolution.exact else "\n(closest available model run)"))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Abstention rate (%)")
    fig.suptitle("Figure 10 Replication: Cross-Domain AEN Steering", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _status_rows(table3: pd.DataFrame, table4: pd.DataFrame, table5: pd.DataFrame, table6: pd.DataFrame, table8: pd.DataFrame) -> list[dict[str, Any]]:
    baseline_exact = bool(
        table8.loc[
            ~table8["method"].isin(["Ambiguity-Encoding Neurons only", "Full probe"]),
            "artifact_model_exact",
        ].all()
    )
    steering_exact = bool(table3["artifact_model_exact"].all() and table5["artifact_model_exact"].all() and table6["artifact_model_exact"].all())
    table4_generated = bool(table4["status"].eq("generated").all())
    return [
        {"item_id": "figure_1", "status": "not_generated", "exactness": "n/a", "note": "Conceptual overview figure; no quantitative artifact to recreate directly."},
        {"item_id": "table_1", "status": "generated", "exactness": "exact", "note": "Exact-model detection metrics available."},
        {"item_id": "table_2", "status": "generated", "exactness": "exact", "note": "Exact-model top-5 neuron rankings available."},
        {
            "item_id": "figure_2",
            "status": "generated",
            "exactness": "exact" if baseline_exact else "partial",
            "note": "Prompt baselines are all exact-model artifacts." if baseline_exact else "Gemma exact; LLaMA/Mistral baselines come from closest available non-exact runs.",
        },
        {"item_id": "figure_3", "status": "generated", "exactness": "exact", "note": "Exact-model perturbation curves available from detection reports."},
        {"item_id": "figure_4", "status": "generated", "exactness": "exact", "note": "Exact-model cross-domain AEN probe results available."},
        {
            "item_id": "table_3",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Steering summaries are exact-model artifacts." if steering_exact else "Steering summaries available, but LLaMA/Mistral use older model variants.",
        },
        {
            "item_id": "figure_5",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Built from exact-model steering summaries and AEN counts." if steering_exact else "Built from current steering summaries and AEN counts.",
        },
        {
            "item_id": "figure_6",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Built from exact-model steering summaries." if steering_exact else "Built from current steering summaries.",
        },
        {
            "item_id": "table_4",
            "status": "generated" if table4_generated else "missing",
            "exactness": "exact" if table4_generated else "missing",
            "note": "Exact-model TriviaQA false-positive evaluation available." if table4_generated else "No TriviaQA side-effect evaluation artifact present.",
        },
        {
            "item_id": "table_5",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Consistency available from exact-model steering summaries." if steering_exact else "Consistency available from current steering summaries.",
        },
        {
            "item_id": "table_6",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Reverse steering available from exact-model steering summaries." if steering_exact else "Reverse steering available from current steering summaries.",
        },
        {
            "item_id": "table_7",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Qualitative examples sampled from exact-model steering outputs." if steering_exact else "Qualitative examples sampled from current steering outputs.",
        },
        {
            "item_id": "table_8",
            "status": "generated",
            "exactness": "exact" if baseline_exact else "partial",
            "note": "Detection and baselines are exact-model artifacts." if baseline_exact else "Detection exact; baselines partial due model mismatch on LLaMA/Mistral.",
        },
        {"item_id": "figure_7", "status": "generated", "exactness": "exact", "note": "Exact-model layerwise probing curves available."},
        {"item_id": "figure_8", "status": "generated", "exactness": "exact", "note": "Exact-model hidden-state distributions available at layer 14."},
        {"item_id": "figure_9", "status": "generated", "exactness": "exact", "note": "Exact-model |Δμ| rankings available from cached hidden states."},
        {
            "item_id": "figure_10",
            "status": "generated",
            "exactness": "exact" if steering_exact else "partial",
            "note": "Cross-domain steering available with exact-model runs." if steering_exact else "Cross-domain steering available, but not with exact LLaMA/Mistral models.",
        },
    ]


def _write_table7_markdown(table7: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Table 7 Replication",
        "",
        "| Model | Dataset | Exact Model Match | Question | Before | After |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in table7.to_dict(orient="records"):
        lines.append(
            "| {model} | {dataset} | {exact} | {question} | {before} | {after} |".format(
                model=row["model_label"],
                dataset=row["dataset"] or "-",
                exact="yes" if row["artifact_model_exact"] else "no",
                question=row["question"].replace("|", "\\|"),
                before=row["before_response"].replace("|", "\\|"),
                after=row["after_response"].replace("|", "\\|"),
            )
        )
    write_markdown(path, "\n".join(lines) + "\n")


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    if isinstance(value, (list, tuple, set)):
        return str(list(value)).replace("|", "\\|")
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _df_to_markdown(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = [_format_markdown_value(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_markdown_report(
    output_root: Path,
    inventory: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    table8: pd.DataFrame,
) -> None:
    baseline_exact = bool(
        table8.loc[
            ~table8["method"].isin(["Ambiguity-Encoding Neurons only", "Full probe"]),
            "artifact_model_exact",
        ].all()
    )
    steering_exact = bool(table3["artifact_model_exact"].all() and table5["artifact_model_exact"].all() and table6["artifact_model_exact"].all())
    status_df = pd.DataFrame(_status_rows(table3=table3, table4=table4, table5=table5, table6=table6, table8=table8))
    lines = [
        "# Paper Replication Audit",
        "",
        "This report maps the local paper PDF against the currently available replication artifacts.",
        "",
        "## Status",
        "",
    ]
    lines.extend(_df_to_markdown(status_df))
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Table 1 exact detection remains below paper values for all models. The largest accuracy gap is `{table1['accuracy_diff'].min():.2f}` points.",
            f"- Table 2 only partially matches the paper neuron rankings. Mean overlap with the paper top-5 is `{table2['paper_rep_overlap_count'].mean():.2f}` neurons.",
            (
                "- Table 8 baseline comparisons now use exact-model prompt-baseline artifacts."
                if baseline_exact
                else "- Table 8 is still partial because LLaMA/Mistral prompt baselines come from older model variants."
            ),
            (
                "- Steering tables/figures still remain partial because exact LLaMA 3.1 / Mistral v0.3 steering runs are missing."
                if not steering_exact
                else "- Steering tables/figures now use exact-model steering artifacts."
            ),
            (
                f"- Table 4 now has exact-model TriviaQA results."
                if bool(table4["status"].eq("generated").all())
                else f"- Table 4 cannot be checked yet because no TriviaQA side-effect artifact exists in the workspace."
            ),
            "",
            "## Paper Inventory",
            "",
        ]
    )
    lines.extend(_df_to_markdown(inventory))
    lines.extend(
        [
            "",
            "## Largest Numerical Gaps",
            "",
            "### Table 1",
            "",
        ]
    )
    lines.extend(
        _df_to_markdown(
            table1.loc[:, ["dataset", "model_label", "paper_accuracy", "rep_accuracy", "accuracy_diff", "paper_f1", "rep_f1", "f1_diff"]]
        )
    )
    lines.extend(["", "### Table 3", ""])
    lines.extend(
        _df_to_markdown(
            table3.loc[:, ["dataset", "method", "model_label", "paper_abstention_rate", "rep_abstention_rate", "difference", "artifact_model_exact"]]
        )
    )
    lines.extend(["", "### Table 4", ""])
    lines.extend(_df_to_markdown(table4))
    lines.extend(["", "### Table 5", ""])
    lines.extend(_df_to_markdown(table5))
    lines.extend(["", "### Table 6", ""])
    lines.extend(_df_to_markdown(table6))
    lines.extend(["", "### Table 8", ""])
    lines.extend(
        _df_to_markdown(
            table8.loc[:, ["dataset", "method", "model_label", "paper_accuracy", "rep_accuracy", "accuracy_diff", "artifact_model_exact"]]
        )
    )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `tables/table_1_detection_comparison.csv`",
            "- `tables/table_2_top_neuron_comparison.csv`",
            "- `tables/table_3_steering_abstention_comparison.csv`",
            "- `tables/table_4_triviaqa_side_effects_comparison.csv`",
            "- `tables/table_5_consistency_comparison.csv`",
            "- `tables/table_6_reverse_steering_comparison.csv`",
            "- `tables/table_7_qualitative_examples.md`",
            "- `tables/table_8_baseline_probe_comparison.csv`",
            "- `figures/figure_2_replication.png` through `figures/figure_10_replication.png`",
            "",
        ]
    )
    write_markdown(output_root / "replication_audit.md", "\n".join(lines) + "\n")


def generate_paper_replication_audit(
    project_root: str | Path,
    paper_pdf_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Generate a paper inventory plus recreated figures/tables from local artifacts."""

    project_root = Path(project_root)
    output_root = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")

    inventory = extract_paper_inventory(paper_pdf_path)
    table1 = build_table_1(project_root)
    table2 = build_table_2(project_root)
    table3 = build_table_3(project_root)
    table4 = build_table_4(project_root)
    table5 = build_table_5(project_root)
    table6 = build_table_6(project_root)
    table7 = build_table_7(project_root)
    table8 = build_table_8(project_root)

    _save_csv(inventory, tables_dir / "paper_inventory.csv")
    _save_csv(table1, tables_dir / "table_1_detection_comparison.csv")
    _save_csv(table2, tables_dir / "table_2_top_neuron_comparison.csv")
    _save_csv(table3, tables_dir / "table_3_steering_abstention_comparison.csv")
    _save_csv(table4, tables_dir / "table_4_triviaqa_side_effects_comparison.csv")
    _save_csv(table5, tables_dir / "table_5_consistency_comparison.csv")
    _save_csv(table6, tables_dir / "table_6_reverse_steering_comparison.csv")
    _save_csv(table7, tables_dir / "table_7_qualitative_examples.csv")
    _write_table7_markdown(table7, tables_dir / "table_7_qualitative_examples.md")
    _save_csv(table8, tables_dir / "table_8_baseline_probe_comparison.csv")

    plot_figure_2(project_root, figures_dir / "figure_2_replication.png")
    plot_figure_3(project_root, figures_dir / "figure_3_replication.png")
    plot_figure_4(project_root, figures_dir / "figure_4_replication.png")
    plot_figure_5(project_root, figures_dir / "figure_5_replication.png")
    plot_figure_6(project_root, figures_dir / "figure_6_replication.png")
    plot_figure_7(project_root, figures_dir / "figure_7_replication.png")
    plot_figure_8(project_root, figures_dir / "figure_8_replication.png")
    plot_figure_9(project_root, figures_dir / "figure_9_replication.png")
    plot_figure_10(project_root, figures_dir / "figure_10_replication.png")

    _render_markdown_report(
        output_root=output_root,
        inventory=inventory,
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
        table5=table5,
        table6=table6,
        table8=table8,
    )

    LOGGER.info("Saved paper replication audit to %s", output_root)
    return {
        "output_root": str(output_root),
        "report_path": str(output_root / "replication_audit.md"),
        "inventory_path": str(tables_dir / "paper_inventory.csv"),
        "table1_path": str(tables_dir / "table_1_detection_comparison.csv"),
        "table8_path": str(tables_dir / "table_8_baseline_probe_comparison.csv"),
        "figure2_path": str(figures_dir / "figure_2_replication.png"),
        "figure10_path": str(figures_dir / "figure_10_replication.png"),
    }
