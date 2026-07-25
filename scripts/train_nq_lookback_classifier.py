#!/usr/bin/env python3
"""Train and evaluate an NQ-domain classifier over Lookback features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir", default="artifacts/lookback_nq/nq_replay_features"
    )
    parser.add_argument(
        "--official-classifier",
        default="external/Lookback-Lens/classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl",
    )
    parser.add_argument(
        "--output", default="checkpoints/lookback_nq_domain_classifier.pkl"
    )
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    return parser.parse_args()


def read_manifest(root: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["dataset_index"])] = row
    return rows


def load_examples(rows: dict[int, dict], indices: list[int]):
    matrices = []
    labels = []
    groups = []
    weights = []
    response_labels = np.asarray([rows[index]["exact_match"] for index in indices])
    class_counts = {
        label: max(1, int((response_labels == label).sum()))
        for label in np.unique(response_labels)
    }
    for index in indices:
        matrix = np.load(rows[index]["feature_path"])["features"].astype(np.float32)
        label = float(rows[index]["exact_match"])
        class_weight = len(indices) / (len(class_counts) * class_counts[label])
        matrices.append(matrix)
        labels.append(np.full(matrix.shape[0], label, dtype=np.int64))
        groups.append(np.full(matrix.shape[0], index, dtype=np.int64))
        weights.append(
            np.full(matrix.shape[0], class_weight / matrix.shape[0], dtype=np.float64)
        )
    return (
        np.concatenate(matrices),
        np.concatenate(labels),
        np.concatenate(groups),
        np.concatenate(weights),
    )


def response_probabilities(
    classifier, matrix: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    token_probabilities = classifier.predict_proba(matrix)[:, 1]
    unique_groups = np.unique(groups)
    probabilities = np.asarray(
        [token_probabilities[groups == group].mean() for group in unique_groups]
    )
    return unique_groups, probabilities


def labels_for_groups(
    labels: np.ndarray, groups: np.ndarray, unique_groups: np.ndarray
) -> np.ndarray:
    return np.asarray([labels[groups == group][0] for group in unique_groups])


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        balanced_accuracy_score(labels, probabilities >= threshold)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(classifier, rows: dict[int, dict], indices: list[int], threshold: float):
    matrix, labels, groups, _ = load_examples(rows, indices)
    unique_groups, probabilities = response_probabilities(classifier, matrix, groups)
    response_labels = labels_for_groups(labels, groups, unique_groups)
    return {
        "n": len(unique_groups),
        "positive_rate": float(response_labels.mean()),
        "roc_auc": float(roc_auc_score(response_labels, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(response_labels, probabilities >= threshold)
        ),
        "mean_probability": float(probabilities.mean()),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_dir)
    rows = read_manifest(root)
    required = set(range(160))
    missing = sorted(required - rows.keys())
    if missing:
        raise RuntimeError(f"missing replay features for indices: {missing[:10]}")

    train_indices = list(range(0, 40))
    tuning_indices = list(range(40, 60))
    refit_indices = list(range(0, 60))
    validation_indices = list(range(60, 110))
    final_indices = list(range(110, 160))
    train_matrix, train_labels, _, train_weights = load_examples(rows, train_indices)
    tuning_matrix, tuning_labels, tuning_groups, _ = load_examples(rows, tuning_indices)

    tuning_records = []
    for c_value in args.c_values:
        classifier = LogisticRegression(C=c_value, max_iter=2_000, solver="liblinear")
        classifier.fit(train_matrix, train_labels, sample_weight=train_weights)
        groups, probabilities = response_probabilities(
            classifier, tuning_matrix, tuning_groups
        )
        response_labels = labels_for_groups(tuning_labels, tuning_groups, groups)
        tuning_records.append(
            {
                "c": c_value,
                "roc_auc": float(roc_auc_score(response_labels, probabilities)),
                "threshold": best_threshold(response_labels, probabilities),
            }
        )
    selected = max(tuning_records, key=lambda row: (row["roc_auc"], -row["c"]))
    selected_threshold = 0.5

    refit_matrix, refit_labels, _, refit_weights = load_examples(rows, refit_indices)
    classifier = LogisticRegression(
        C=selected["c"], max_iter=2_000, solver="liblinear"
    )
    classifier.fit(refit_matrix, refit_labels, sample_weight=refit_weights)

    with Path(args.official_classifier).open("rb") as handle:
        official_payload = pickle.load(handle)
    official = official_payload["clf"]
    report = {
        "split_protocol": {
            "fit": [0, 39],
            "tune": [40, 59],
            "refit": [0, 59],
            "validation": [60, 109],
            "final": [110, 159],
        },
        "tuning": tuning_records,
        "selected_c": selected["c"],
        "selected_threshold": selected_threshold,
        "selected_threshold_source": "fixed_balanced_response_prior",
        "domain_classifier": {
            "validation": evaluate(
                classifier, rows, validation_indices, selected_threshold
            ),
            "final": evaluate(
                classifier, rows, final_indices, selected_threshold
            ),
        },
        "official_classifier": {
            "validation": evaluate(
                official,
                rows,
                validation_indices,
                float(official_payload["best_threshold"]),
            ),
            "final": evaluate(
                official,
                rows,
                final_indices,
                float(official_payload["best_threshold"]),
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(
            {
                "clf": classifier,
                "best_threshold": selected_threshold,
                "metadata": report,
            },
            handle,
        )
    report_path = root / "classifier_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
