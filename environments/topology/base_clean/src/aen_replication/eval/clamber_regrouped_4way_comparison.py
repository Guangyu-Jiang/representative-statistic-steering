"""Fair 4-way CLAMBER regrouped comparison across readout/probe families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from aen_replication.config import load_config
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.train.aen import _fit_probe, _transform
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

FINAL_LABELS = ["ambiguity", "clear", "conflicting_condition", "missing_condition"]

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_clamber.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/mistral_clamber_pca16.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/gemma_clamber_pca16.yaml",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
    },
]

METHOD_LABELS = {
    "mean_pool_full_probe": "Mean-pool Full Probe",
    "mean_pool_aen": "Mean-pool AEN",
    "last_token_full_probe": "Last-token Full Probe",
    "last_token_aen": "Last-token AEN",
    "token_cloud_single": "Topology Single",
    "token_cloud_multilayer": "Topology Multi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_comparison",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def _fit_multiclass_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    max_iter: int = 4000,
    c_value: float = 1.0,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        C=float(c_value),
        max_iter=int(max_iter),
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


def _select_train_only_aens(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    probe_cfg: dict[str, Any],
    val_fraction: float,
    perturb_top_k: list[int],
    sigma: float,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    train_idx, val_idx = _split_indices(pd.Series(y_train), val_fraction=val_fraction, seed=seed)
    clf, scaler = _fit_probe(
        x_train=x_train[train_idx],
        y_train=y_train[train_idx],
        probe_cfg=probe_cfg,
        seed=seed,
    )
    x_val = _transform(x_train[val_idx], scaler)
    y_val = y_train[val_idx]
    weights = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    ranked = np.argsort(-weights)
    baseline_pred = clf.decision_function(x_val)
    baseline_accuracy = float(np.mean((baseline_pred >= 0.0).astype(int) == y_val))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for k in perturb_top_k:
        k = min(int(k), x_val.shape[1])
        if k <= 0:
            continue
        indices = ranked[:k]
        trial_accs: list[float] = []
        for _ in range(max(1, int(trials))):
            perturbed = x_val.copy()
            perturbed[:, indices] += rng.normal(0.0, float(sigma), size=(perturbed.shape[0], len(indices)))
            scores = clf.decision_function(perturbed)
            trial_accs.append(float(np.mean((scores >= 0.0).astype(int) == y_val)))
        mean_acc = float(np.mean(trial_accs))
        results.append(
            {
                "k": k,
                "indices": indices.tolist(),
                "accuracy_after_perturb": mean_acc,
                "accuracy_drop": baseline_accuracy - mean_acc,
            }
        )
    best = max(results, key=lambda row: float(row["accuracy_drop"]))
    return {
        "baseline_accuracy": baseline_accuracy,
        "aen_indices": list(best["indices"]),
        "aen_k": int(best["k"]),
        "results": results,
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=FINAL_LABELS).tolist(),
        "labels": list(FINAL_LABELS),
    }


def _load_group4_hidden_state(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta, matrix = load_hidden_state_table(path)
    meta = meta.copy()
    meta = meta.loc[meta["subclass"].isin(GROUP4_MAP)].reset_index(drop=True)
    matrix = matrix[meta.index.to_numpy()]  # wrong if reset before indexing
    return meta, matrix


def _load_group4_hidden_state_safe(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta, matrix = load_hidden_state_table(path)
    mask = meta["subclass"].isin(GROUP4_MAP).to_numpy()
    meta = meta.loc[mask].reset_index(drop=True)
    matrix = matrix[mask]
    meta["group4"] = meta["subclass"].map(GROUP4_MAP)
    meta["binary_group"] = np.where(meta["group4"].eq("clear"), "clear", "ill_posed")
    return meta, matrix


def _available_layers(hidden_root: Path, readout: str) -> list[int]:
    layers = []
    for path in sorted(hidden_root.glob(f"clamber__layer_*__{readout}.parquet")):
        layer = int(path.name.split("__")[1].split("_")[1])
        layers.append(layer)
    return layers


def _evaluate_readout(
    *,
    hidden_root: Path,
    readout: str,
    config: dict[str, Any],
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    layers = _available_layers(hidden_root, readout)
    if not layers:
        raise FileNotFoundError(f"No cached layers found for readout={readout} under {hidden_root}")

    probe_cfg = dict(config["probe"])
    subclass_cfg = dict(config.get("clamber_subclass_classification", {}))
    rows: list[dict[str, Any]] = []

    for layer in layers:
        path = hidden_root / f"clamber__layer_{int(layer):02d}__{readout}.parquet"
        meta, matrix = _load_group4_hidden_state_safe(path)
        train_mask = meta["split"].eq("train").to_numpy()
        train_meta = meta.loc[train_mask].reset_index(drop=True)
        train_matrix = matrix[train_mask]
        tr_idx, val_idx = _split_indices(train_meta["group4"], val_fraction=val_fraction, seed=seed + int(layer))

        x_train = train_matrix[tr_idx]
        y_train = train_meta.iloc[tr_idx]["group4"].astype(str).to_numpy()
        x_val = train_matrix[val_idx]
        y_val = train_meta.iloc[val_idx]["group4"].astype(str).to_numpy()
        clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 100 + int(layer))
        y_pred = clf.predict(scaler.transform(x_val))
        full_metrics = _compute_metrics(y_val, y_pred)

        aen_info = _select_train_only_aens(
            x_train=train_matrix,
            y_train=(train_meta["group4"].ne("clear")).to_numpy(dtype=int),
            probe_cfg=probe_cfg,
            val_fraction=val_fraction,
            perturb_top_k=list(subclass_cfg.get("perturb_top_k", [1, 2, 3, 5, 10, 20])),
            sigma=float(subclass_cfg.get("perturb_sigma", 0.15)),
            trials=int(subclass_cfg.get("perturb_trials", 8)),
            seed=seed + 200 + int(layer),
        )
        indices = list(aen_info["aen_indices"])
        clf_aen, scaler_aen = _fit_multiclass_logistic(x_train[:, indices], y_train, seed=seed + 300 + int(layer))
        y_pred_aen = clf_aen.predict(scaler_aen.transform(x_val[:, indices]))
        aen_metrics = _compute_metrics(y_val, y_pred_aen)

        rows.extend(
            [
                {
                    "method": f"{readout}_full_probe",
                    "readout": readout,
                    "layer": int(layer),
                    "val_accuracy": float(full_metrics["accuracy"]),
                    "val_macro_f1": float(full_metrics["macro_f1"]),
                    "feature_count": int(x_train.shape[1]),
                    "aen_k": np.nan,
                },
                {
                    "method": f"{readout}_aen",
                    "readout": readout,
                    "layer": int(layer),
                    "val_accuracy": float(aen_metrics["accuracy"]),
                    "val_macro_f1": float(aen_metrics["macro_f1"]),
                    "feature_count": int(len(indices)),
                    "aen_k": int(aen_info["aen_k"]),
                },
            ]
        )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["method", "val_macro_f1", "val_accuracy", "layer"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    best_full = candidate_df.loc[candidate_df["method"].eq(f"{readout}_full_probe")].iloc[0].to_dict()
    best_aen = candidate_df.loc[candidate_df["method"].eq(f"{readout}_aen")].iloc[0].to_dict()
    return candidate_df, best_full, best_aen


def _finalize_readout_result(
    *,
    hidden_root: Path,
    readout: str,
    layer: int,
    config: dict[str, Any],
    seed: int,
    method: str,
) -> dict[str, Any]:
    meta, matrix = _load_group4_hidden_state_safe(hidden_root / f"clamber__layer_{int(layer):02d}__{readout}.parquet")
    train_mask = meta["split"].eq("train").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    x_train = matrix[train_mask]
    x_test = matrix[test_mask]
    y_train = train_meta["group4"].astype(str).to_numpy()
    y_test = test_meta["group4"].astype(str).to_numpy()
    indices = None
    if method.endswith("_aen"):
        subclass_cfg = dict(config.get("clamber_subclass_classification", {}))
        aen_info = _select_train_only_aens(
            x_train=x_train,
            y_train=(train_meta["group4"].ne("clear")).to_numpy(dtype=int),
            probe_cfg=dict(config["probe"]),
            val_fraction=float(subclass_cfg.get("val_fraction", 0.2)),
            perturb_top_k=list(subclass_cfg.get("perturb_top_k", [1, 2, 3, 5, 10, 20])),
            sigma=float(subclass_cfg.get("perturb_sigma", 0.15)),
            trials=int(subclass_cfg.get("perturb_trials", 8)),
            seed=seed + 1000 + int(layer),
        )
        indices = list(aen_info["aen_indices"])
        x_train = x_train[:, indices]
        x_test = x_test[:, indices]
    clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed + 2000 + int(layer))
    y_pred = clf.predict(scaler.transform(x_test))
    metrics = _compute_metrics(y_test, y_pred)
    result = {
        "method": method,
        "readout": readout,
        "layer": int(layer),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
        "feature_count": int(x_train.shape[1]),
        "selection_signature": None,
    }
    if indices is not None:
        result["aen_k"] = int(len(indices))
        result["aen_indices"] = indices
    return result


def _load_topology_results() -> pd.DataFrame:
    path = Path("/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet")
    df = pd.read_parquet(path).copy()
    return df


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    report_lines = [
        "# CLAMBER Regrouped 4-Way Comparison",
        "",
        "Methods:",
        "- mean-pool full probe",
        "- mean-pool AEN",
        "- last-token full probe",
        "- last-token AEN",
        "- token-cloud topology single",
        "- token-cloud topology multi",
        "",
    ]

    for model_index, spec in enumerate(MODEL_SPECS):
        config = load_config(spec["config_path"])
        hidden_root = Path(spec["hidden_root"])
        model_rows: list[dict[str, Any]] = []

        for readout in ["mean_pool", "last_token"]:
            candidate_df, best_full, best_aen = _evaluate_readout(
                hidden_root=hidden_root,
                readout=readout,
                config=config,
                seed=args.seed + 100 * model_index,
                val_fraction=args.val_fraction,
            )
            candidate_df["model"] = spec["slug"]
            candidate_df["model_label"] = spec["label"]
            candidate_rows.extend(candidate_df.to_dict(orient="records"))

            for best in [best_full, best_aen]:
                result = _finalize_readout_result(
                    hidden_root=hidden_root,
                    readout=readout,
                    layer=int(best["layer"]),
                    config=config,
                    seed=args.seed + 100 * model_index,
                    method=str(best["method"]),
                )
                row = {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    **result,
                }
                final_rows.append(row)
                model_rows.append(row)

        topo = _load_topology_results()
        topo_rows = topo.loc[topo["model"].eq(spec["slug"])].copy()
        for _, row in topo_rows.iterrows():
            record = {
                "model": spec["slug"],
                "model_label": spec["label"],
                "method": str(row["method"]),
                "readout": "token_cloud",
                "layer": int(row["layer"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "confusion_matrix": row["confusion_matrix"],
                "labels": row["labels"],
                "feature_count": int(row["feature_count"]),
                "selection_signature": row.get("selection_signature"),
            }
            final_rows.append(record)
            model_rows.append(record)

        model_df = pd.DataFrame(model_rows)
        order = [
            "mean_pool_full_probe",
            "mean_pool_aen",
            "last_token_full_probe",
            "last_token_aen",
            "token_cloud_single",
            "token_cloud_multilayer",
        ]
        report_lines.extend(
            [
                f"## {spec['label']}",
                "",
                "| Method | Macro-F1 | Acc | Features | View |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for method in order:
            row = model_df.loc[model_df["method"].eq(method)].iloc[0]
            if method == "token_cloud_multilayer":
                view = f"layers {row['selection_signature']}"
            else:
                view = f"{row['readout']}, layer {int(row['layer'])}"
            report_lines.append(
                f"| {METHOD_LABELS[method]} | {float(row['macro_f1']):.4f} | {float(row['accuracy']):.4f} | "
                f"{int(row['feature_count'])} | {view} |"
            )
        report_lines.append("")

    candidate_df = pd.DataFrame(candidate_rows)
    final_df = pd.DataFrame(final_rows)
    write_parquet(candidate_df, output_root / "clamber_regrouped_4way_comparison_candidates.parquet")
    write_parquet(final_df, output_root / "clamber_regrouped_4way_comparison_final_metrics.parquet")
    write_markdown(output_root / "clamber_regrouped_4way_comparison_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
