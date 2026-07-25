"""CLAMBER-specific AEN evaluation using the existing ambiguity probe stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.train.aen import evaluate_full_probe, evaluate_sparse_probe, select_aens
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_markdown, write_parquet


def _layerwise_probe_report(
    manifest: dict[str, Any],
    dataset_name: str,
    aen_indices: list[int],
    probe_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for file_record in manifest["files"]:
        if file_record["readout"] != "mean_pool" or file_record["dataset"] != dataset_name:
            continue
        metadata, matrix = load_hidden_state_table(file_record["parquet_path"])
        full_probe = evaluate_full_probe(metadata=metadata, matrix=matrix, probe_cfg=probe_cfg, seed=seed)
        sparse_probe = evaluate_sparse_probe(full_probe=full_probe, indices=aen_indices, probe_cfg=probe_cfg, seed=seed)
        rows.append(
            {
                "dataset": dataset_name,
                "layer": int(file_record["layer"]),
                "full_accuracy": full_probe["test_metrics"]["accuracy"],
                "full_f1": full_probe["test_metrics"]["f1"],
                "full_auroc": full_probe["test_metrics"]["auroc"],
                "aen_accuracy": sparse_probe["test_metrics"]["accuracy"],
                "aen_f1": sparse_probe["test_metrics"]["f1"],
                "aen_auroc": sparse_probe["test_metrics"]["auroc"],
            }
        )
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def _subclass_metrics(metadata: pd.DataFrame, decision_scores: np.ndarray) -> pd.DataFrame:
    test_df = metadata.loc[metadata["split"].eq("test")].reset_index(drop=True).copy()
    if len(test_df) != len(decision_scores):
        raise ValueError("Subclass metric shape mismatch between metadata and decision scores.")
    predicted = (decision_scores >= 0.0).astype(int)
    test_df["decision_score"] = decision_scores.astype(float)
    test_df["predicted_ambiguous"] = predicted

    rows: list[dict[str, Any]] = []
    for subclass, group in test_df.groupby("subclass", dropna=False, sort=True):
        y_true = group["label_ambiguous"].to_numpy(dtype=int)
        y_score = group["decision_score"].to_numpy(dtype=float)
        y_pred = group["predicted_ambiguous"].to_numpy(dtype=int)
        metrics = binary_classification_metrics(y_true, y_score)
        rows.append(
            {
                "subclass": str(subclass),
                "n_examples": int(len(group)),
                "positive_rate": float(y_true.mean()) if len(group) else float("nan"),
                "predicted_ambiguous_rate": float(y_pred.mean()) if len(group) else float("nan"),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "auroc": float(metrics["auroc"]),
            }
        )
    return pd.DataFrame(rows).sort_values("subclass").reset_index(drop=True)


def _render_report(
    *,
    model_name: str,
    dataset_name: str,
    default_layer: int,
    full_metrics: dict[str, Any],
    aen_selection: dict[str, Any],
    aen_metrics: dict[str, Any],
    layerwise_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    lines = [
        "# CLAMBER Binary Ambiguity Detection",
        "",
        f"- Model: `{model_name}`",
        f"- Dataset: `{dataset_name}`",
        f"- Default layer: `{default_layer}`",
        "",
        "## Default Layer Metrics",
        "",
        f"- Full probe: AUROC `{full_metrics['auroc']:.4f}`, accuracy `{full_metrics['accuracy']:.4f}`, F1 `{full_metrics['f1']:.4f}`.",
        (
            f"- AEN-only probe: AUROC `{aen_metrics['auroc']:.4f}`, accuracy `{aen_metrics['accuracy']:.4f}`, "
            f"F1 `{aen_metrics['f1']:.4f}` using `k={int(aen_selection['aen_k'])}` neurons."
        ),
        f"- Selected AEN indices: `{aen_selection['aen_indices']}`",
        "",
        "## Layerwise Best Results",
        "",
    ]
    if not layerwise_df.empty:
        best_full = layerwise_df.sort_values(["full_auroc", "full_accuracy"], ascending=False).iloc[0]
        best_aen = layerwise_df.sort_values(["aen_auroc", "aen_accuracy"], ascending=False).iloc[0]
        lines.append(
            f"- Best full-probe layer: `{int(best_full['layer'])}` "
            f"(AUROC `{best_full['full_auroc']:.4f}`, accuracy `{best_full['full_accuracy']:.4f}`)."
        )
        lines.append(
            f"- Best AEN layer: `{int(best_aen['layer'])}` "
            f"(AUROC `{best_aen['aen_auroc']:.4f}`, accuracy `{best_aen['aen_accuracy']:.4f}`)."
        )
    write_markdown(output_path, "\n".join(lines) + "\n")


def run_clamber_detection_experiment(config: dict[str, Any]) -> dict[str, str]:
    dataset_name = str(config.get("clamber_detection", {}).get("dataset", "clamber"))
    seed = int(config["seed"])
    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    cache_root = Path(config["extraction"]["cache_dir"]) / model_slug
    manifest_path = cache_root / f"{dataset_name}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CLAMBER manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    output_root = ensure_dir(Path(config["clamber_detection"]["output_dir"]) / model_slug)
    default_layer = int(config["extraction"]["default_layer"])
    default_record = next(
        record
        for record in manifest["files"]
        if record["readout"] == "mean_pool" and int(record["layer"]) == default_layer and record["dataset"] == dataset_name
    )
    metadata, matrix = load_hidden_state_table(default_record["parquet_path"])
    full_probe = evaluate_full_probe(metadata=metadata, matrix=matrix, probe_cfg=config["probe"], seed=seed)
    aen_selection = select_aens(
        full_probe=full_probe,
        perturb_top_k=list(config["probe"]["perturb_top_k"]),
        sigma=float(config["probe"]["perturb_sigma"]),
        trials=int(config["probe"]["perturb_trials"]),
        seed=seed,
    )
    sparse_probe = evaluate_sparse_probe(
        full_probe=full_probe,
        indices=list(aen_selection["aen_indices"]),
        probe_cfg=config["probe"],
        seed=seed,
    )

    test_labels = full_probe["splits"]["test"]["labels"]
    full_test_scores = full_probe["classifier"].decision_function(full_probe["splits"]["test"]["matrix"])
    sparse_test_features = full_probe["splits"]["test"]["matrix"][:, list(aen_selection["aen_indices"])]
    if sparse_probe["scaler"] is not None:
        sparse_test_features = sparse_probe["scaler"].transform(sparse_test_features)
    sparse_test_scores = sparse_probe["classifier"].decision_function(sparse_test_features)

    layerwise_df = _layerwise_probe_report(
        manifest=manifest,
        dataset_name=dataset_name,
        aen_indices=list(aen_selection["aen_indices"]),
        probe_cfg=config["probe"],
        seed=seed,
    )
    full_subclass_df = _subclass_metrics(
        metadata=metadata.loc[metadata["split"].eq("test")].reset_index(drop=True).assign(split="test"),
        decision_scores=full_test_scores,
    )
    aen_subclass_df = _subclass_metrics(
        metadata=metadata.loc[metadata["split"].eq("test")].reset_index(drop=True).assign(split="test"),
        decision_scores=sparse_test_scores,
    )
    subclass_df = full_subclass_df.merge(
        aen_subclass_df,
        on=["subclass", "n_examples", "positive_rate"],
        suffixes=("_full", "_aen"),
    )

    layerwise_path = output_root / str(config["clamber_detection"]["layerwise_filename"])
    subclass_path = output_root / str(config["clamber_detection"]["subclass_metrics_filename"])
    summary_json_path = output_root / str(config["clamber_detection"]["summary_json_filename"])
    report_path = output_root / str(config["clamber_detection"]["report_filename"])

    layerwise_df.to_csv(layerwise_path, index=False)
    write_parquet(subclass_df, subclass_path)
    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "default_layer": default_layer,
        "n_test": int(len(test_labels)),
        "full_probe_test": full_probe["test_metrics"],
        "aen_selection": aen_selection,
        "aen_probe_test": sparse_probe["test_metrics"],
    }
    write_json(summary_json_path, summary)
    _render_report(
        model_name=model_name,
        dataset_name=dataset_name,
        default_layer=default_layer,
        full_metrics=full_probe["test_metrics"],
        aen_selection=aen_selection,
        aen_metrics=sparse_probe["test_metrics"],
        layerwise_df=layerwise_df,
        output_path=report_path,
    )
    return {
        "summary_json_path": str(summary_json_path),
        "layerwise_path": str(layerwise_path),
        "subclass_metrics_path": str(subclass_path),
        "report_path": str(report_path),
    }
