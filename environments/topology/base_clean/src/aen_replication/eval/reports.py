"""Markdown report helpers for the replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Replication Summary",
        "",
        f"- Model: `{summary['model_name']}`",
        "",
        "## Detection Results",
    ]
    for name, payload in summary["datasets"].items():
        lines.append(f"### {name}")
        if "full_probe_test" in payload:
            lines.append(f"- Full probe accuracy: `{payload['full_probe_test']['accuracy']:.4f}`")
            lines.append(f"- Full probe F1: `{payload['full_probe_test']['f1']:.4f}`")
            lines.append(f"- Top-5 weighted neurons: `{payload['top_5_weights']}`")
            lines.append(f"- AEN k: `{payload['aen_selection']['aen_k']}`")
            lines.append(f"- AEN indices: `{payload['aen_selection']['aen_indices']}`")
            lines.append(f"- AEN probe accuracy: `{payload['aen_probe_test']['accuracy']:.4f}`")
            lines.append(f"- AEN probe F1: `{payload['aen_probe_test']['f1']:.4f}`")
        else:
            lines.append(f"- Transfer accuracy: `{payload['accuracy']:.4f}`")
            lines.append(f"- Transfer F1: `{payload['f1']:.4f}`")
        lines.append("")
    overlap = summary.get("cross_dataset_overlap")
    if overlap:
        lines.extend(
            [
                "## Cross-Dataset Overlap",
                "",
                f"- Default layer: `{overlap['default_layer']}`",
                f"- Top-5 overlap: `{overlap['top_5_overlap']}`",
                f"- Top-10 overlap: `{overlap['top_10_overlap']}`",
                "",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
