"""Hierarchical topology classification for regrouped 4-way CLAMBER.

Stage 1: binary classification (`clear` vs `ill_posed`)
Stage 2: 3-way classification within the ill-posed subset
         (`ambiguity`, `missing_condition`, `conflicting_condition`)

This script uses only cached token-cloud topology features.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

GROUP4_MAP = {
    "polysemy": "ambiguity",
    "co-reference": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "ICL": "conflicting_condition",
    "none": "clear",
}

ILLPOSED_GROUPS = ["ambiguity", "missing_condition", "conflicting_condition"]
FINAL_LABELS = ["ambiguity", "clear", "conflicting_condition", "missing_condition"]

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/meta_llama_llama_3_1_8b_instruct",
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/mistralai_mistral_7b_instruct_v0_3",
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/google_gemma_7b_it",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_hierarchical_topology",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _load_feature_df(subclass_root: Path) -> pd.DataFrame:
    df = pd.read_parquet(subclass_root / "clamber_token_cloud_all_layer_features.parquet").copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    df["binary_label"] = np.where(df["group4"].eq("clear"), "clear", "ill_posed")
    return df.reset_index(drop=True)


def _compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=FINAL_LABELS).tolist(),
        "labels": list(FINAL_LABELS),
    }


def _fit_and_predict_hierarchical(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    x_binary_train = train_df.loc[:, feature_cols].to_numpy(dtype=float)
    y_binary_train = train_df["binary_label"].astype(str).to_numpy()
    binary_clf, binary_scaler = _fit_logistic(x_binary_train, y_binary_train, seed=seed)

    x_eval = eval_df.loc[:, feature_cols].to_numpy(dtype=float)
    binary_scores = binary_clf.decision_function(binary_scaler.transform(x_eval))
    binary_pred = np.where(binary_scores >= 0.0, "ill_posed", "clear")

    ill_train = train_df.loc[train_df["group4"].ne("clear")].copy()
    x_ill_train = ill_train.loc[:, feature_cols].to_numpy(dtype=float)
    y_ill_train = ill_train["group4"].astype(str).to_numpy()
    tri_clf, tri_scaler = _fit_logistic(x_ill_train, y_ill_train, seed=seed + 1)

    final_pred = np.full(len(eval_df), "clear", dtype=object)
    ill_mask = binary_pred == "ill_posed"
    if np.any(ill_mask):
        x_ill_eval = x_eval[ill_mask]
        final_pred[ill_mask] = tri_clf.predict(tri_scaler.transform(x_ill_eval))

    y_true = eval_df["group4"].astype(str).to_numpy()
    binary_true = eval_df["binary_label"].astype(str).to_numpy()
    binary_prob = 1.0 / (1.0 + np.exp(-binary_scores))
    binary_auroc = float(roc_auc_score((binary_true == "ill_posed").astype(int), binary_prob))
    binary_acc = float(accuracy_score(binary_true, binary_pred))

    ill_eval = eval_df.loc[eval_df["group4"].ne("clear")].copy()
    x_ill_eval = ill_eval.loc[:, feature_cols].to_numpy(dtype=float)
    y_ill_eval = ill_eval["group4"].astype(str).to_numpy()
    tri_pred = tri_clf.predict(tri_scaler.transform(x_ill_eval))
    tri_acc = float(accuracy_score(y_ill_eval, tri_pred))
    tri_macro_f1 = float(f1_score(y_ill_eval, tri_pred, average="macro", zero_division=0))

    overall = _compute_multiclass_metrics(y_true, np.asarray(final_pred, dtype=object))
    return {
        "binary_accuracy": binary_acc,
        "binary_auroc": binary_auroc,
        "illposed_accuracy": tri_acc,
        "illposed_macro_f1": tri_macro_f1,
        "overall_accuracy": float(overall["accuracy"]),
        "overall_macro_f1": float(overall["macro_f1"]),
        "confusion_matrix": overall["confusion_matrix"],
        "labels": overall["labels"],
    }


def _make_multilayer_frame(feature_df: pd.DataFrame, layers: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    lookup = feature_df.loc[:, ["example_id", "group4", "binary_label"]].drop_duplicates()
    train_df = train_df.merge(lookup, on="example_id", how="left")
    test_df = test_df.merge(lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    return train_df, test_df, feature_cols


def _evaluate_single_layers(
    feature_df: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_all = feature_df.loc[feature_df["split"].eq("train")].copy()
    test_all = feature_df.loc[feature_df["split"].eq("test")].copy()

    for layer in sorted(feature_df["layer"].unique().tolist()):
        train_df = train_all.loc[train_all["layer"].eq(layer)].copy()
        tr_idx, val_idx = _split_indices(train_df["group4"], val_fraction=val_fraction, seed=seed + int(layer))
        feature_cols = _topology_feature_columns(train_df)
        metrics = _fit_and_predict_hierarchical(
            train_df.iloc[tr_idx].copy(),
            train_df.iloc[val_idx].copy(),
            feature_cols,
            seed=seed + 100 + int(layer),
        )
        rows.append(
            {
                "method": "token_cloud_single",
                "layer": int(layer),
                "feature_count": int(len(feature_cols)),
                "val_binary_accuracy": float(metrics["binary_accuracy"]),
                "val_binary_auroc": float(metrics["binary_auroc"]),
                "val_illposed_accuracy": float(metrics["illposed_accuracy"]),
                "val_illposed_macro_f1": float(metrics["illposed_macro_f1"]),
                "val_overall_accuracy": float(metrics["overall_accuracy"]),
                "val_overall_macro_f1": float(metrics["overall_macro_f1"]),
            }
        )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["val_overall_macro_f1", "val_overall_accuracy", "val_binary_auroc"],
        ascending=False,
    ).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best_layer = int(best["layer"])
    train_df = train_all.loc[train_all["layer"].eq(best_layer)].copy()
    test_df = test_all.loc[test_all["layer"].eq(best_layer)].copy()
    feature_cols = _topology_feature_columns(train_df)
    metrics = _fit_and_predict_hierarchical(train_df, test_df, feature_cols, seed=seed + 999)
    final = {
        "method": "token_cloud_single",
        "layer": best_layer,
        "selection_signature": None,
        "feature_count": int(len(feature_cols)),
        **metrics,
    }
    return candidate_df, final


def _evaluate_multilayer(
    feature_df: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    layers = sorted(feature_df["layer"].unique().tolist())
    rows: list[dict[str, Any]] = []

    for width in [2, min(3, len(layers))]:
        if width > len(layers):
            continue
        for combo in itertools.combinations(layers, width):
            train_df, test_df, feature_cols = _make_multilayer_frame(feature_df, list(combo))
            tr_idx, val_idx = _split_indices(train_df["group4"], val_fraction=val_fraction, seed=seed + sum(combo) + width)
            metrics = _fit_and_predict_hierarchical(
                train_df.iloc[tr_idx].copy(),
                train_df.iloc[val_idx].copy(),
                feature_cols,
                seed=seed + 200 + sum(combo),
            )
            rows.append(
                {
                    "method": "token_cloud_multilayer",
                    "selection_signature": " | ".join(str(item) for item in combo),
                    "feature_count": int(len(feature_cols)),
                    "val_binary_accuracy": float(metrics["binary_accuracy"]),
                    "val_binary_auroc": float(metrics["binary_auroc"]),
                    "val_illposed_accuracy": float(metrics["illposed_accuracy"]),
                    "val_illposed_macro_f1": float(metrics["illposed_macro_f1"]),
                    "val_overall_accuracy": float(metrics["overall_accuracy"]),
                    "val_overall_macro_f1": float(metrics["overall_macro_f1"]),
                }
            )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["val_overall_macro_f1", "val_overall_accuracy", "val_binary_auroc"],
        ascending=False,
    ).reset_index(drop=True)
    best = candidate_df.iloc[0].to_dict()
    best_layers = [int(item.strip()) for item in str(best["selection_signature"]).split("|")]
    train_df, test_df, feature_cols = _make_multilayer_frame(feature_df, best_layers)
    metrics = _fit_and_predict_hierarchical(train_df, test_df, feature_cols, seed=seed + 1999)
    final = {
        "method": "token_cloud_multilayer",
        "layer": -1,
        "selection_signature": str(best["selection_signature"]),
        "feature_count": int(len(feature_cols)),
        **metrics,
    }
    return candidate_df, final


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    report_lines = [
        "# CLAMBER Regrouped 4-Way Hierarchical Topology Study",
        "",
        "Hierarchical protocol:",
        "1. Binary classifier: `clear` vs `ill_posed`",
        "2. 3-way classifier inside `ill_posed`: `ambiguity`, `missing_condition`, `conflicting_condition`",
        "",
    ]

    for model_index, spec in enumerate(MODEL_SPECS):
        feature_df = _load_feature_df(Path(spec["subclass_root"]))
        single_candidates, single_final = _evaluate_single_layers(
            feature_df,
            seed=args.seed + 100 * model_index,
            val_fraction=args.val_fraction,
        )
        multi_candidates, multi_final = _evaluate_multilayer(
            feature_df,
            seed=args.seed + 100 * model_index,
            val_fraction=args.val_fraction,
        )

        single_candidates["model"] = spec["slug"]
        single_candidates["model_label"] = spec["label"]
        multi_candidates["model"] = spec["slug"]
        multi_candidates["model_label"] = spec["label"]
        candidate_rows.extend(single_candidates.to_dict(orient="records"))
        candidate_rows.extend(multi_candidates.to_dict(orient="records"))

        for final in [single_final, multi_final]:
            final_rows.append(
                {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    **final,
                }
            )

        report_lines.extend(
            [
                f"## {spec['label']}",
                "",
                "| Method | Binary AUROC | Binary Acc | 3-way Ill-posed Macro-F1 | Overall 4-way Macro-F1 | Overall 4-way Acc | View |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| Topology Single | {single_final['binary_auroc']:.4f} | {single_final['binary_accuracy']:.4f} | "
                f"{single_final['illposed_macro_f1']:.4f} | {single_final['overall_macro_f1']:.4f} | {single_final['overall_accuracy']:.4f} | layer {int(single_final['layer'])} |",
                f"| Topology Multi | {multi_final['binary_auroc']:.4f} | {multi_final['binary_accuracy']:.4f} | "
                f"{multi_final['illposed_macro_f1']:.4f} | {multi_final['overall_macro_f1']:.4f} | {multi_final['overall_accuracy']:.4f} | layers {multi_final['selection_signature']} |",
                "",
            ]
        )

    candidate_df = pd.DataFrame(candidate_rows)
    final_df = pd.DataFrame(final_rows)
    write_parquet(candidate_df, output_root / "clamber_regrouped_4way_hierarchical_topology_candidates.parquet")
    write_parquet(final_df, output_root / "clamber_regrouped_4way_hierarchical_topology_final_metrics.parquet")
    write_markdown(output_root / "clamber_regrouped_4way_hierarchical_topology_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
