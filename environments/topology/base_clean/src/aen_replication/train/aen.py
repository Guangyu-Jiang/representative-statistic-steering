"""Paper-style probe training, AEN selection, and evaluation."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json

LOGGER = logging.getLogger(__name__)


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    probe_cfg: dict[str, Any],
    seed: int,
) -> tuple[LogisticRegression, StandardScaler | None]:
    scaler = None
    x_fit = x_train
    if probe_cfg.get("standardize", True):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        penalty=probe_cfg.get("penalty", "l1"),
        solver=probe_cfg.get("solver", "liblinear"),
        C=float(probe_cfg.get("C", 1.0)),
        max_iter=int(probe_cfg.get("max_iter", 4000)),
        class_weight=probe_cfg.get("class_weight"),
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*penalty=l1 with l1_ratio=0.0.*", category=UserWarning)
        clf.fit(x_fit, y_train)
    return clf, scaler


def _transform(matrix: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    return scaler.transform(matrix) if scaler is not None else matrix


def _split(metadata: pd.DataFrame, matrix: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split in ("train", "test"):
        split_df = metadata.loc[metadata["split"] == split].reset_index(drop=True)
        split_matrix = matrix[metadata["split"].eq(split).to_numpy()]
        labels = split_df["label_ambiguous"].to_numpy(dtype=int)
        out[split] = (split_df, split_matrix, labels)
    return out


def evaluate_full_probe(metadata: pd.DataFrame, matrix: np.ndarray, probe_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Train and evaluate the paper's full-vector ambiguity probe."""

    splits = _split(metadata, matrix)
    train_df, x_train, y_train = splits["train"]
    test_df, x_test, y_test = splits["test"]
    clf, scaler = _fit_probe(x_train=x_train, y_train=y_train, probe_cfg=probe_cfg, seed=seed)
    x_train_t = _transform(x_train, scaler)
    x_test_t = _transform(x_test, scaler)
    train_scores = clf.decision_function(x_train_t)
    test_scores = clf.decision_function(x_test_t)
    return {
        "classifier": clf,
        "scaler": scaler,
        "coefficients": clf.coef_.ravel().astype(float),
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "test_metrics": binary_classification_metrics(y_test, test_scores),
        "splits": {
            "train": {"df": train_df, "matrix": x_train_t, "labels": y_train},
            "test": {"df": test_df, "matrix": x_test_t, "labels": y_test},
        },
    }


def select_aens(
    full_probe: dict[str, Any],
    perturb_top_k: list[int],
    sigma: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Select Ambiguity-Encoding Neurons by perturbation-induced accuracy drop."""

    rng = np.random.default_rng(seed)
    clf = full_probe["classifier"]
    test_matrix = full_probe["splits"]["test"]["matrix"]
    test_labels = full_probe["splits"]["test"]["labels"]
    weights = np.abs(full_probe["coefficients"])
    ranked = np.argsort(-weights)
    baseline = float(binary_classification_metrics(test_labels, clf.decision_function(test_matrix))["accuracy"])
    results: list[dict[str, Any]] = []
    for k in perturb_top_k:
        top_indices = ranked[:k]
        trial_accs: list[float] = []
        for _ in range(trials):
            perturbed = test_matrix.copy()
            perturbed[:, top_indices] += rng.normal(0.0, sigma, size=(perturbed.shape[0], k))
            scores = clf.decision_function(perturbed)
            trial_accs.append(float(binary_classification_metrics(test_labels, scores)["accuracy"]))
        mean_acc = float(np.mean(trial_accs))
        results.append(
            {
                "k": int(k),
                "indices": top_indices.tolist(),
                "accuracy_after_perturb": mean_acc,
                "accuracy_drop": baseline - mean_acc,
            }
        )
    best = max(results, key=lambda item: item["accuracy_drop"])
    return {"baseline_accuracy": baseline, "results": results, "aen_indices": best["indices"], "aen_k": best["k"]}


def evaluate_sparse_probe(
    full_probe: dict[str, Any],
    indices: list[int],
    probe_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Retrain a probe using only selected neuron indices."""

    train = full_probe["splits"]["train"]
    test = full_probe["splits"]["test"]
    x_train = train["matrix"][:, indices]
    x_test = test["matrix"][:, indices]
    clf, scaler = _fit_probe(x_train=x_train, y_train=train["labels"], probe_cfg=probe_cfg, seed=seed)
    x_train_t = _transform(x_train, scaler)
    x_test_t = _transform(x_test, scaler)
    return {
        "indices": indices,
        "classifier": clf,
        "scaler": scaler,
        "train_metrics": binary_classification_metrics(train["labels"], clf.decision_function(x_train_t)),
        "test_metrics": binary_classification_metrics(test["labels"], clf.decision_function(x_test_t)),
    }


def layerwise_probe_report(
    hidden_state_manifest: dict[str, Any],
    dataset_name: str,
    aen_indices: list[int],
    probe_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Train full and AEN-only probes at every available layer."""

    rows: list[dict[str, Any]] = []
    for file_record in hidden_state_manifest["files"]:
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
                "aen_accuracy": sparse_probe["test_metrics"]["accuracy"],
                "aen_f1": sparse_probe["test_metrics"]["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def run_detection_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the paper's ambiguity-detection experiment for both datasets."""

    seed = int(config["seed"])
    cache_root = Path(config["extraction"]["cache_dir"])
    model_slug = slugify(config["model"]["name"])
    report_root = ensure_dir(Path(config["reports"]["output_dir"]) / model_slug)
    probe_root = ensure_dir(Path(config["probe"]["artifact_dir"]) / model_slug)
    summary: dict[str, Any] = {"model_name": config["model"]["name"], "datasets": {}}

    manifests: dict[str, dict[str, Any]] = {}
    for dataset_name in ("ambigqa", "situatedqa"):
        manifest_path = cache_root / model_slug / f"{dataset_name}_manifest.json"
        manifests[dataset_name] = json.loads(manifest_path.read_text(encoding="utf-8"))

    default_layer = int(config["extraction"]["default_layer"])
    ranked_indices_by_dataset: dict[str, list[int]] = {}
    for dataset_name, manifest in manifests.items():
        default_record = next(
            record for record in manifest["files"] if record["readout"] == "mean_pool" and int(record["layer"]) == default_layer
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
        layerwise = layerwise_probe_report(
            hidden_state_manifest=manifest,
            dataset_name=dataset_name,
            aen_indices=list(aen_selection["aen_indices"]),
            probe_cfg=config["probe"],
            seed=seed,
        )
        layerwise.to_csv(report_root / f"{dataset_name}_layerwise.csv", index=False)
        write_json(
            probe_root / f"{dataset_name}_default_layer_report.json",
            {
                "dataset": dataset_name,
                "default_layer": default_layer,
                "full_probe_test": full_probe["test_metrics"],
                "aen_selection": aen_selection,
                "aen_probe_test": sparse_probe["test_metrics"],
                "top_5_weights": np.argsort(-np.abs(full_probe["coefficients"]))[:5].tolist(),
                "ranked_indices": np.argsort(-np.abs(full_probe["coefficients"])).tolist(),
            },
        )
        ranked_indices_by_dataset[dataset_name] = np.argsort(-np.abs(full_probe["coefficients"])).tolist()
        summary["datasets"][dataset_name] = {
            "full_probe_test": full_probe["test_metrics"],
            "aen_selection": aen_selection,
            "aen_probe_test": sparse_probe["test_metrics"],
            "top_5_weights": ranked_indices_by_dataset[dataset_name][:5],
        }

    ambig_top = ranked_indices_by_dataset["ambigqa"]
    situated_top = ranked_indices_by_dataset["situatedqa"]
    summary["cross_dataset_overlap"] = {
        "default_layer": default_layer,
        "top_5_overlap": sorted(set(ambig_top[:5]) & set(situated_top[:5])),
        "top_10_overlap": sorted(set(ambig_top[:10]) & set(situated_top[:10])),
    }

    # Cross-domain AEN-only transfer at default layer.
    for train_name, test_name in (("ambigqa", "situatedqa"), ("situatedqa", "ambigqa")):
        train_report = json.loads((probe_root / f"{train_name}_default_layer_report.json").read_text(encoding="utf-8"))
        train_record = next(
            record
            for record in manifests[train_name]["files"]
            if record["readout"] == "mean_pool" and int(record["layer"]) == default_layer
        )
        test_record = next(
            record
            for record in manifests[test_name]["files"]
            if record["readout"] == "mean_pool" and int(record["layer"]) == default_layer
        )
        train_metadata, train_matrix = load_hidden_state_table(train_record["parquet_path"])
        test_metadata, test_matrix = load_hidden_state_table(test_record["parquet_path"])
        train_full = evaluate_full_probe(train_metadata, train_matrix, config["probe"], seed)
        indices = list(train_report["aen_selection"]["aen_indices"])
        sparse_probe = evaluate_sparse_probe(train_full, indices, config["probe"], seed)
        test_mask = test_metadata["split"].eq("test").to_numpy()
        test_features = test_matrix[test_mask]
        if train_full["scaler"] is not None:
            test_features = train_full["scaler"].transform(test_features)
        test_features = test_features[:, indices]
        if sparse_probe["scaler"] is not None:
            test_features = sparse_probe["scaler"].transform(test_features)
        test_labels = test_metadata.loc[test_mask, "label_ambiguous"].to_numpy(dtype=int)
        transfer_metrics = binary_classification_metrics(
            test_labels,
            sparse_probe["classifier"].decision_function(test_features),
        )
        summary["datasets"][f"{train_name}_to_{test_name}"] = transfer_metrics

    write_json(report_root / "detection_summary.json", summary)
    return summary
