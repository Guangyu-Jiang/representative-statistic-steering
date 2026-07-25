"""Sparse probe training and sweep utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, get_git_commit, read_json, slugify, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


def _prepare_split(
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = metadata["split"] == split
    split_df = metadata.loc[mask].reset_index(drop=True)
    split_matrix = matrix[mask.to_numpy()]
    labels = split_df["label_ambiguous"].to_numpy(dtype=int)
    return split_df, split_matrix, labels


def _select_nonzero_indices(coefficients: np.ndarray) -> tuple[list[int], str | None]:
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-8).tolist()
    if nonzero:
        return nonzero, None
    fallback = [int(np.argmax(np.abs(coefficients)))]
    return fallback, "all_zero_probe_coefficients"


def train_sparse_probe(
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    probe_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit a sparse logistic probe and compute split-wise metrics."""

    train_df, x_train, y_train = _prepare_split(metadata, matrix, split="train")
    val_df, x_val, y_val = _prepare_split(metadata, matrix, split="val")
    test_df, x_test, y_test = _prepare_split(metadata, matrix, split="test")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Each split must contain at least one row for probe training.")

    scaler = None
    x_train_input = x_train
    x_val_input = x_val
    x_test_input = x_test
    if probe_config.get("standardize", True):
        scaler = StandardScaler()
        x_train_input = scaler.fit_transform(x_train)
        x_val_input = scaler.transform(x_val)
        x_test_input = scaler.transform(x_test)

    classifier = LogisticRegression(
        penalty=probe_config.get("penalty", "l1"),
        solver=probe_config.get("solver", "liblinear"),
        class_weight=probe_config.get("class_weight"),
        C=float(probe_config.get("C", 1.0)),
        max_iter=int(probe_config.get("max_iter", 2000)),
        random_state=seed,
    )
    classifier.fit(x_train_input, y_train)

    train_scores = classifier.decision_function(x_train_input)
    val_scores = classifier.decision_function(x_val_input)
    test_scores = classifier.decision_function(x_test_input)
    threshold = float(probe_config.get("score_threshold", 0.0))

    coefficients = classifier.coef_.ravel().astype(float)
    nonzero_indices, selection_note = _select_nonzero_indices(coefficients)

    metrics_train = binary_classification_metrics(y_train, train_scores, threshold=threshold)
    metrics_val = binary_classification_metrics(y_val, val_scores, threshold=threshold)
    metrics_test = binary_classification_metrics(y_test, test_scores, threshold=threshold)

    return {
        "classifier": classifier,
        "scaler": scaler,
        "coefficients": coefficients,
        "intercept": float(classifier.intercept_[0]),
        "selected_indices": nonzero_indices,
        "selection_note": selection_note,
        "threshold": threshold,
        "splits": {
            "train": {"df": train_df, "scores": train_scores, "labels": y_train, "metrics": metrics_train},
            "val": {"df": val_df, "scores": val_scores, "labels": y_val, "metrics": metrics_val},
            "test": {"df": test_df, "scores": test_scores, "labels": y_test, "metrics": metrics_test},
        },
    }


def save_probe_artifacts(
    output_dir: str | Path,
    model_name: str,
    layer: int,
    readout: str,
    result: dict[str, Any],
    probe_config: dict[str, Any],
    hidden_state_path: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Persist a trained probe and its metadata."""

    artifact_root = ensure_dir(output_dir) / slugify(model_name) / f"layer_{layer:02d}__{readout}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    coefficients_path = artifact_root / "coefficients.npy"
    intercept_path = artifact_root / "intercept.npy"
    indices_path = artifact_root / "selected_indices.npy"
    bundle_path = artifact_root / "probe.joblib"
    metadata_path = artifact_root / "metadata.json"

    np.save(coefficients_path, result["coefficients"])
    np.save(intercept_path, np.asarray([result["intercept"]], dtype=float))
    np.save(indices_path, np.asarray(result["selected_indices"], dtype=int))

    bundle = {
        "model_name": model_name,
        "layer": layer,
        "readout": readout,
        "classifier": result["classifier"],
        "scaler": result["scaler"],
        "selected_indices": result["selected_indices"],
        "coefficients": result["coefficients"],
        "intercept": result["intercept"],
        "probe_config": probe_config,
        "hidden_state_path": hidden_state_path,
    }
    joblib.dump(bundle, bundle_path)

    for split_name, split_payload in result["splits"].items():
        scores_df = split_payload["df"].copy()
        scores_df["decision_value"] = split_payload["scores"]
        scores_df["predicted_label"] = (split_payload["scores"] >= result["threshold"]).astype(int)
        scores_df.to_parquet(artifact_root / f"scores_{split_name}.parquet", index=False)

    metadata = {
        "model_name": model_name,
        "layer": layer,
        "readout": readout,
        "n_nonzero": len(result["selected_indices"]),
        "selected_indices_path": str(indices_path),
        "coefficients_path": str(coefficients_path),
        "intercept_path": str(intercept_path),
        "selection_note": result["selection_note"],
        "probe_config": probe_config,
        "hidden_state_path": hidden_state_path,
        "created_at": utc_now_iso(),
        "git_commit": get_git_commit(project_root),
        "metrics": {
            split_name: split_payload["metrics"] for split_name, split_payload in result["splits"].items()
        },
    }
    write_json(metadata_path, metadata)
    LOGGER.info("Saved probe artifacts to %s", artifact_root)
    return {
        "artifact_dir": str(artifact_root),
        "bundle_path": str(bundle_path),
        "metadata_path": str(metadata_path),
    }


def build_layer_sweep_row(
    model_name: str,
    layer: int,
    readout: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build one layer-sweep row from split-wise probe metrics."""

    metrics_val = result["splits"]["val"]["metrics"]
    metrics_test = result["splits"]["test"]["metrics"]
    return {
        "model_name": model_name,
        "layer": layer,
        "readout": readout,
        "train_n": int(len(result["splits"]["train"]["df"])),
        "val_n": int(len(result["splits"]["val"]["df"])),
        "test_n": int(len(result["splits"]["test"]["df"])),
        "auroc_val": metrics_val["auroc"],
        "f1_val": metrics_val["f1"],
        "accuracy_val": metrics_val["accuracy"],
        "auroc_test": metrics_test["auroc"],
        "f1_test": metrics_test["f1"],
        "accuracy_test": metrics_test["accuracy"],
        "n_nonzero": int(len(result["selected_indices"])),
        "selected_as_best": False,
    }


def train_probe_sweep(
    manifest_path: str | Path,
    probe_config: dict[str, Any],
    output_dir: str | Path,
    selection_metric: str,
    seed: int,
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train probes for all cached layer/readout combinations and select the best one."""

    manifest = read_json(manifest_path)
    model_name = manifest["model_name"]
    sweep_rows: list[dict[str, Any]] = []
    trained_probes: list[dict[str, Any]] = []

    for file_record in manifest["files"]:
        hidden_state_path = file_record["parquet_path"]
        layer = int(file_record["layer"])
        readout = str(file_record["readout"])

        metadata, matrix = load_hidden_state_table(hidden_state_path)
        result = train_sparse_probe(metadata=metadata, matrix=matrix, probe_config=probe_config, seed=seed)
        artifact_info = save_probe_artifacts(
            output_dir=output_dir,
            model_name=model_name,
            layer=layer,
            readout=readout,
            result=result,
            probe_config=probe_config,
            hidden_state_path=hidden_state_path,
            project_root=project_root,
        )
        sweep_row = build_layer_sweep_row(model_name=model_name, layer=layer, readout=readout, result=result)
        sweep_rows.append(sweep_row)
        trained_probes.append(
            {
                "row": sweep_row,
                "result": result,
                "artifact_info": artifact_info,
                "hidden_state_path": hidden_state_path,
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    metric_values = sweep_df[selection_metric].fillna(float("-inf"))
    best_index = int(metric_values.idxmax())
    sweep_df.loc[best_index, "selected_as_best"] = True

    best_probe = trained_probes[best_index]
    best_bundle = joblib.load(best_probe["artifact_info"]["bundle_path"])
    best_metadata = read_json(best_probe["artifact_info"]["metadata_path"])

    best_probe_path = Path(output_dir) / "best_probe.joblib"
    best_metadata_path = Path(output_dir) / "best_probe_metadata.json"
    best_payload = {
        **best_bundle,
        "selection_metric": selection_metric,
    }
    joblib.dump(best_payload, best_probe_path)
    write_json(
        best_metadata_path,
        {
            **best_metadata,
            "selection_metric": selection_metric,
            "selected_as_best": True,
        },
    )
    LOGGER.info("Selected best probe: layer=%s readout=%s", best_bundle["layer"], best_bundle["readout"])
    sweep_df = sweep_df.sort_values(["layer", "readout"]).reset_index(drop=True)
    return sweep_df, {
        "bundle_path": str(best_probe_path),
        "metadata_path": str(best_metadata_path),
    }
