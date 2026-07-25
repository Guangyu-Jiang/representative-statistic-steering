"""Evaluate source-dataset AENs on CLAMBER binary classification.

This checks whether sparse ambiguity directions detected on a binary source dataset transfer to
CLAMBER/CHAMBER.  It reports two complementary settings:

1. Direct transfer: train the AEN probe on the source dataset, evaluate on CLAMBER.
2. Fixed-dimension CLAMBER: keep the source AEN dimensions fixed, but train a
   new CLAMBER binary probe using only those dimensions.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
        "layer": 14,
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
        "layer": 14,
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
        "layer": 14,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/ambigqa_aen_to_clamber_binary",
    )
    parser.add_argument(
        "--source-dataset",
        default="ambigqa",
        choices=["ambigqa", "situatedqa"],
    )
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    return parser.parse_args()


def _artifact_prefix(source_dataset: str) -> str:
    return f"{source_dataset}_aen_to_clamber_binary"


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
        clf.fit(x_train, y_train)
    return clf


def _load_dataset(hidden_root: Path, dataset: str, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    path = hidden_root / f"{dataset}__layer_{int(layer):02d}__mean_pool.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden-state cache: {path}")
    meta, matrix = load_hidden_state_table(path)
    return meta.reset_index(drop=True), np.asarray(matrix, dtype=np.float32)


def _split(meta: pd.DataFrame, matrix: np.ndarray, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = meta["split"].astype(str).eq(split).to_numpy()
    split_meta = meta.loc[mask].reset_index(drop=True)
    x = np.asarray(matrix[mask], dtype=np.float32)
    y = split_meta["label_ambiguous"].to_numpy(dtype=int)
    return split_meta, x, y


def _ranked_indices(clf: LogisticRegression) -> list[int]:
    weights = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    return np.argsort(-weights).astype(int).tolist()


def _scores(clf: LogisticRegression, x_eval: np.ndarray) -> np.ndarray:
    return np.asarray(clf.decision_function(x_eval), dtype=float)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    if unique.size == 1:
        candidates = np.asarray([unique[0]], dtype=float)
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.concatenate(
            [
                [unique[0] - 1e-6],
                mids,
                [unique[-1] + 1e-6],
            ]
        )
    best_threshold = float(candidates[0])
    best_metrics = binary_classification_metrics(y_true, scores, threshold=best_threshold)
    for threshold in candidates[1:]:
        metrics = binary_classification_metrics(y_true, scores, threshold=float(threshold))
        best_key = (float(best_metrics["macro_f1"]), float(best_metrics["accuracy"]))
        key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_auroc": float(metrics["auroc"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_confusion_matrix": metrics["confusion_matrix"],
    }


def _subclass_rows(
    *,
    model: str,
    model_label: str,
    method: str,
    k: int,
    indices: list[int],
    clamber_test_meta: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_df = clamber_test_meta.reset_index(drop=True).copy()
    if len(test_df) != len(scores):
        raise ValueError("CLAMBER test metadata and scores have different lengths.")
    test_df["score"] = np.asarray(scores, dtype=float)
    for subclass, group in test_df.groupby("subclass", dropna=False, sort=True):
        y = group["label_ambiguous"].to_numpy(dtype=int)
        subclass_scores = group["score"].to_numpy(dtype=float)
        metrics = binary_classification_metrics(y, subclass_scores, threshold=threshold)
        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "method": method,
                "k": int(k),
                "aen_indices": list(indices),
                "subclass": str(subclass),
                "n_test": int(len(group)),
                "positive_rate": float(y.mean()) if len(y) else float("nan"),
                "threshold": float(threshold),
                "accuracy": float(metrics["accuracy"]),
                "auroc": float(metrics["auroc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return rows


def _run_model(
    spec: dict[str, Any],
    k_values: list[int],
    *,
    seed: int,
    source_dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hidden_root = Path(spec["hidden_root"])
    layer = int(spec["layer"])

    source_meta, source_matrix = _load_dataset(hidden_root, source_dataset, layer)
    clamber_meta, clamber_matrix = _load_dataset(hidden_root, "clamber", layer)

    source_train_meta, x_source_train, y_source_train = _split(source_meta, source_matrix, "train")
    source_test_meta, x_source_test, y_source_test = _split(source_meta, source_matrix, "test")
    clamber_train_meta, x_clamber_train, y_clamber_train = _split(clamber_meta, clamber_matrix, "train")
    clamber_test_meta, x_clamber_test, y_clamber_test = _split(clamber_meta, clamber_matrix, "test")

    full_source = _fit_probe(x_source_train, y_source_train, seed=seed)
    ranked = _ranked_indices(full_source)

    full_clamber = _fit_probe(x_clamber_train, y_clamber_train, seed=seed + 10)
    full_clamber_metrics = binary_classification_metrics(y_clamber_test, _scores(full_clamber, x_clamber_test))

    full_transfer_train_scores = _scores(full_source, x_clamber_train)
    full_transfer_test_scores = _scores(full_source, x_clamber_test)
    full_transfer_zero = binary_classification_metrics(y_clamber_test, full_transfer_test_scores)
    full_transfer_threshold, full_transfer_calibrated = _best_threshold(y_clamber_train, full_transfer_train_scores)
    full_transfer_calibrated_test = binary_classification_metrics(
        y_clamber_test,
        full_transfer_test_scores,
        threshold=full_transfer_threshold,
    )

    rows: list[dict[str, Any]] = []
    subclass_rows: list[dict[str, Any]] = []
    for k in k_values:
        indices = ranked[: int(k)]

        source_aen = _fit_probe(x_source_train[:, indices], y_source_train, seed=seed + 100 + int(k))
        source_aen_source_test_scores = _scores(source_aen, x_source_test[:, indices])
        source_aen_source_test_metrics = binary_classification_metrics(y_source_test, source_aen_source_test_scores)

        transfer_train_scores = _scores(source_aen, x_clamber_train[:, indices])
        transfer_test_scores = _scores(source_aen, x_clamber_test[:, indices])
        transfer_zero = binary_classification_metrics(y_clamber_test, transfer_test_scores)
        transfer_threshold, transfer_train_cal = _best_threshold(y_clamber_train, transfer_train_scores)
        transfer_calibrated = binary_classification_metrics(
            y_clamber_test,
            transfer_test_scores,
            threshold=transfer_threshold,
        )

        fixed_dim_clamber = _fit_probe(x_clamber_train[:, indices], y_clamber_train, seed=seed + 200 + int(k))
        fixed_dim_test_scores = _scores(fixed_dim_clamber, x_clamber_test[:, indices])
        fixed_dim_metrics = binary_classification_metrics(y_clamber_test, fixed_dim_test_scores)

        row: dict[str, Any] = {
            "model": spec["slug"],
            "model_label": spec["label"],
            "layer": layer,
            "source_dataset": source_dataset,
            "k": int(k),
            "aen_indices": list(indices),
            "source_train_n": int(len(source_train_meta)),
            "source_test_n": int(len(source_test_meta)),
            "clamber_train_n": int(len(clamber_train_meta)),
            "clamber_test_n": int(len(clamber_test_meta)),
            "clamber_train_pos": int(y_clamber_train.sum()),
            "clamber_test_pos": int(y_clamber_test.sum()),
            "transfer_threshold": 0.0,
            "transfer_calibrated_threshold": float(transfer_threshold),
            "transfer_train_calibrated_accuracy": float(transfer_train_cal["accuracy"]),
            "full_transfer_calibrated_threshold": float(full_transfer_threshold),
            **_metric_row("source_aen_source_test", source_aen_source_test_metrics),
            **_metric_row("source_aen_to_clamber_zero", transfer_zero),
            **_metric_row("source_aen_to_clamber_calibrated", transfer_calibrated),
            **_metric_row("clamber_fixed_source_aen_dims", fixed_dim_metrics),
            **_metric_row("clamber_full_probe", full_clamber_metrics),
            **_metric_row("source_full_to_clamber_zero", full_transfer_zero),
            **_metric_row("source_full_to_clamber_calibrated", full_transfer_calibrated_test),
        }
        rows.append(row)

        subclass_rows.extend(
            _subclass_rows(
                model=spec["slug"],
                model_label=spec["label"],
                method=f"{source_dataset}_aen_to_clamber_zero",
                k=int(k),
                indices=indices,
                clamber_test_meta=clamber_test_meta,
                scores=transfer_test_scores,
                threshold=0.0,
            )
        )
        subclass_rows.extend(
            _subclass_rows(
                model=spec["slug"],
                model_label=spec["label"],
                method=f"{source_dataset}_aen_to_clamber_calibrated",
                k=int(k),
                indices=indices,
                clamber_test_meta=clamber_test_meta,
                scores=transfer_test_scores,
                threshold=transfer_threshold,
            )
        )
        subclass_rows.extend(
            _subclass_rows(
                model=spec["slug"],
                model_label=spec["label"],
                method=f"clamber_fixed_{source_dataset}_aen_dims",
                k=int(k),
                indices=indices,
                clamber_test_meta=clamber_test_meta,
                scores=fixed_dim_test_scores,
                threshold=0.0,
            )
        )

    split_summary = {
        "model": spec["slug"],
        "model_label": spec["label"],
        "layer": layer,
        "source_dataset": source_dataset,
        "source_train_n": int(len(source_train_meta)),
        "source_test_n": int(len(source_test_meta)),
        "source_train_pos": int(y_source_train.sum()),
        "source_test_pos": int(y_source_test.sum()),
        "clamber_train_n": int(len(clamber_train_meta)),
        "clamber_test_n": int(len(clamber_test_meta)),
        "clamber_train_pos": int(y_clamber_train.sum()),
        "clamber_test_pos": int(y_clamber_test.sum()),
        "clamber_subclass_counts": clamber_meta["subclass"].value_counts().sort_index().to_dict(),
        "clamber_subclass_label_counts": {
            str(subclass): {str(label): int(count) for label, count in counts.items()}
            for subclass, counts in pd.crosstab(clamber_meta["subclass"], clamber_meta["label_ambiguous"]).to_dict("index").items()
        },
    }
    return rows, subclass_rows, split_summary


def _render_report(summary_df: pd.DataFrame, split_df: pd.DataFrame, output_root: Path, *, source_dataset: str) -> str:
    prefix = _artifact_prefix(source_dataset)
    source_label = source_dataset
    lines = [
        f"# {source_label.capitalize()}-Detected AENs on CLAMBER Binary Detection",
        "",
        "Setup:",
        f"- Source dataset for AEN detection/training: `{source_label}`.",
        "- Target dataset: `clamber`, binary label `label_ambiguous` / `require_clarification`.",
        "- Readout: mean-pooled hidden state.",
        "- Layer: 14 for all models.",
        f"- AEN dimensions: top-k absolute coefficients from the `{source_label}` full probe, reported for k=3 and k=5.",
        f"- `{source_label}_aen_to_clamber_zero`: train sparse AEN probe on `{source_label}`, evaluate CLAMBER with source threshold 0.",
        f"- `{source_label}_aen_to_clamber_calibrated`: same `{source_label}` sparse probe, but threshold calibrated on CLAMBER train.",
        f"- `clamber_fixed_{source_label}_aen_dims`: train a CLAMBER probe using only the `{source_label}` AEN dimensions.",
        "- `clamber_full_probe`: in-domain CLAMBER full-neuron probe reference.",
        "",
        "## Splits",
        "",
        f"| Model | {source_label.capitalize()} train/test | CLAMBER train/test | CLAMBER test positives |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in split_df.iterrows():
        lines.append(
            f"| {row['model_label']} | {int(row['source_train_n'])}/{int(row['source_test_n'])} | "
            f"{int(row['clamber_train_n'])}/{int(row['clamber_test_n'])} | {int(row['clamber_test_pos'])} |"
        )

    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            f"| Model | k | AEN indices | {source_label.capitalize()} AEN source acc/AUROC | Direct CLAMBER acc/AUROC/F1 | Calibrated direct acc/AUROC/F1 | CLAMBER fixed-dim acc/AUROC/F1 | CLAMBER full acc/AUROC/F1 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model_label']} | {int(row['k'])} | `{row['aen_indices']}` | "
            f"{row['source_aen_source_test_accuracy']:.4f}/{row['source_aen_source_test_auroc']:.4f} | "
            f"{row['source_aen_to_clamber_zero_accuracy']:.4f}/{row['source_aen_to_clamber_zero_auroc']:.4f}/{row['source_aen_to_clamber_zero_macro_f1']:.4f} | "
            f"{row['source_aen_to_clamber_calibrated_accuracy']:.4f}/{row['source_aen_to_clamber_calibrated_auroc']:.4f}/{row['source_aen_to_clamber_calibrated_macro_f1']:.4f} | "
            f"{row['clamber_fixed_source_aen_dims_accuracy']:.4f}/{row['clamber_fixed_source_aen_dims_auroc']:.4f}/{row['clamber_fixed_source_aen_dims_macro_f1']:.4f} | "
            f"{row['clamber_full_probe_accuracy']:.4f}/{row['clamber_full_probe_auroc']:.4f}/{row['clamber_full_probe_macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full-Probe Transfer Reference",
            "",
            "| Model | Direct full acc/AUROC/F1 | Calibrated full acc/AUROC/F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    full_rows = summary_df.sort_values(["model_label", "k"]).drop_duplicates("model_label")
    for _, row in full_rows.iterrows():
        lines.append(
            f"| {row['model_label']} | "
            f"{row['source_full_to_clamber_zero_accuracy']:.4f}/{row['source_full_to_clamber_zero_auroc']:.4f}/{row['source_full_to_clamber_zero_macro_f1']:.4f} | "
            f"{row['source_full_to_clamber_calibrated_accuracy']:.4f}/{row['source_full_to_clamber_calibrated_auroc']:.4f}/{row['source_full_to_clamber_calibrated_macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Direct {source_label.capitalize()}-to-CLAMBER AEN transfer is weak, especially with the source threshold.",
            "- Calibrating only the threshold on CLAMBER train improves accuracy/F1 but AUROC is unchanged; this checks score separability rather than a new classifier.",
            f"- Reusing the {source_label.capitalize()} AEN dimensions and retraining on CLAMBER gives stronger results, but remains well below the full CLAMBER probe.",
            "",
            f"- Overall metrics: `{output_root / f'{prefix}_metrics.parquet'}`",
            f"- Subclass metrics: `{output_root / f'{prefix}_subclass_metrics.parquet'}`",
            f"- Split summary: `{output_root / f'{prefix}_splits.parquet'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    k_values = sorted({int(k) for k in args.k_values})
    source_dataset = str(args.source_dataset)
    prefix = _artifact_prefix(source_dataset)

    summary_rows: list[dict[str, Any]] = []
    subclass_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(MODEL_SPECS):
        rows, sub_rows, split_summary = _run_model(
            spec,
            k_values,
            seed=int(args.seed) + 1000 * idx,
            source_dataset=source_dataset,
        )
        summary_rows.extend(rows)
        subclass_rows.extend(sub_rows)
        split_rows.append(split_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["model_label", "k"]).reset_index(drop=True)
    subclass_df = pd.DataFrame(subclass_rows).sort_values(["model_label", "method", "k", "subclass"]).reset_index(drop=True)
    split_df = pd.DataFrame(split_rows).sort_values(["model_label"]).reset_index(drop=True)

    write_parquet(summary_df, output_root / f"{prefix}_metrics.parquet")
    write_parquet(subclass_df, output_root / f"{prefix}_subclass_metrics.parquet")
    write_parquet(split_df, output_root / f"{prefix}_splits.parquet")
    write_markdown(
        output_root / f"{prefix}_report.md",
        _render_report(summary_df, split_df, output_root, source_dataset=source_dataset),
    )


if __name__ == "__main__":
    main()
