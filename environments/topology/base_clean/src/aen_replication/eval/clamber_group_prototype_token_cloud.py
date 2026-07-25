"""Evaluate group-specific prototype distances for CLAMBER token-cloud topology."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from aen_replication.config import load_config
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import (
    _diagram_descriptors,
    _persistence_image_features,
    _safe_bottleneck,
    _safe_wasserstein,
)
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _compute_diagrams,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

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
GROUP4_LABELS = ["ambiguity", "clear", "conflicting_condition", "missing_condition"]

MODEL_SPECS = {
    "meta_llama_llama_3_1_8b_instruct": {
        "label": "LLaMA 3.1 8B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    },
    "mistralai_mistral_7b_instruct_v0_3": {
        "label": "Mistral 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/mistral_clamber_pca16.yaml",
    },
    "google_gemma_7b_it": {
        "label": "Gemma 7B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/gemma_clamber_pca16.yaml",
    },
}

BASELINE_4WAY = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet"
)
BASELINE_9WAY = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_9way_comparison/clamber_9way_comparison_final_metrics.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-slugs",
        nargs="+",
        default=["meta_llama_llama_3_1_8b_instruct"],
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument("--label-spaces", nargs="+", default=["4way", "9way"], choices=["4way", "9way"])
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_group_prototype_token_cloud",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _fit_multiclass_logistic(
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def _split_indices(labels: pd.Series, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy = np.zeros(len(labels), dtype=int)
    train_idx, val_idx = next(splitter.split(dummy, labels.astype(str)))
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _prepare_label_space(feature_df: pd.DataFrame, *, label_space: str) -> tuple[pd.DataFrame, list[str]]:
    df = feature_df.copy()
    if label_space == "4way":
        df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
        df["target_label"] = df["subclass"].map(GROUP4_MAP)
        labels = list(GROUP4_LABELS)
    elif label_space == "9way":
        df["target_label"] = df["subclass"].astype(str)
        labels = sorted(df["target_label"].unique().tolist())
    else:
        raise ValueError(f"Unsupported label space: {label_space}")
    return df.reset_index(drop=True), labels


def _prototype_diagrams_from_group_clouds(
    cloud_df: pd.DataFrame,
    *,
    layers: list[int],
    label_values: list[str],
    config: dict[str, Any],
    seed: int,
) -> dict[tuple[int, str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    prototype_cap = int(config.get("prototype_token_cap", 128))
    distance_metric = str(config.get("distance_metric", "euclidean"))
    maxdim = int(config.get("maxdim", 1))
    coeff = int(config.get("coeff", 2))
    prototypes: dict[tuple[int, str], list[np.ndarray]] = {}
    train_df = cloud_df.loc[cloud_df["split"].eq("train")].copy()
    for layer in layers:
        layer_df = train_df.loc[train_df["layer"].eq(layer)]
        for label_name in label_values:
            label_df = layer_df.loc[layer_df["target_label"].astype(str).eq(str(label_name))]
            if label_df.empty:
                prototypes[(layer, str(label_name))] = [np.zeros((0, 2), dtype=float) for _ in range(maxdim + 1)]
                continue
            token_matrix = np.vstack(label_df["cloud"].to_list()).astype(np.float32, copy=False)
            if len(token_matrix) > prototype_cap:
                selected = np.sort(rng.choice(len(token_matrix), size=prototype_cap, replace=False))
                token_matrix = token_matrix[selected]
            prototypes[(layer, str(label_name))] = _compute_diagrams(
                token_matrix,
                maxdim=maxdim,
                coeff=coeff,
                distance_metric=distance_metric,
            )
    return prototypes


def _precompute_diagram_rows(
    cloud_df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    distance_metric = str(config.get("distance_metric", "euclidean"))
    grid_size = int(config.get("betti_grid_size", 32))
    pimg_side = int(config.get("persistence_image_grid_side", 4))
    rows: list[dict[str, Any]] = []
    for row in tqdm(cloud_df.to_dict(orient="records"), desc="group_proto_diagrams"):
        cloud = np.asarray(row["cloud"], dtype=float)
        diagrams = _compute_diagrams(
            cloud,
            maxdim=int(config.get("maxdim", 1)),
            coeff=int(config.get("coeff", 2)),
            distance_metric=distance_metric,
        )
        feature_row: dict[str, Any] = {
            "example_id": str(row["example_id"]),
            "pair_id": str(row["pair_id"]),
            "dataset": str(row["dataset"]),
            "split": str(row["split"]),
            "label_ambiguous": int(row["label_ambiguous"]),
            "layer": int(row["layer"]),
            "token_count": int(row["token_count"]),
            "subclass": str(row["subclass"]),
            "__h0_diagram": diagrams[0] if len(diagrams) > 0 else np.zeros((0, 2), dtype=float),
            "__h1_diagram": diagrams[1] if len(diagrams) > 1 else np.zeros((0, 2), dtype=float),
        }
        for homology_dim, prefix in [(0, "h0"), (1, "h1")]:
            diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.zeros((0, 2), dtype=float)
            feature_row.update(_diagram_descriptors(diagram, prefix=prefix, grid_size=grid_size))
            feature_row.update(_persistence_image_features(diagram, prefix=prefix, grid_side=pimg_side))
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _group_diagram_feature_row(
    row: dict[str, Any],
    *,
    prototype_map: dict[tuple[int, str], list[np.ndarray]],
    group_labels: list[str],
) -> dict[str, Any]:
    layer = int(row["layer"])
    feature_row: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "layer": int(layer),
        "token_count": int(row["token_count"]),
        "subclass": str(row["subclass"]),
        "target_label": str(row["target_label"]),
    }
    for prefix in ["h0", "h1"]:
        for key, value in row.items():
            if key.startswith(f"{prefix}_"):
                feature_row[key] = value
    # The user's proposed extension names the four prototype-distance families
    # directly, which in our current CLAMBER analysis are the H0 versions.
    diagram = np.asarray(row["__h0_diagram"], dtype=float)
    for label_name in group_labels:
        proto = prototype_map[(layer, str(label_name))][0]
        label_slug = str(label_name).replace("-", "_").replace(" ", "_")
        feature_row[f"h0_wasserstein_to_{label_slug}"] = _safe_wasserstein(diagram, proto)
        feature_row[f"h0_bottleneck_to_{label_slug}"] = _safe_bottleneck(diagram, proto)
    return feature_row


def _build_group_feature_frame(
    diagram_df: pd.DataFrame,
    *,
    prototype_map: dict[tuple[int, str], list[np.ndarray]],
    group_labels: list[str],
    parallel_jobs: int,
) -> pd.DataFrame:
    rows = diagram_df.to_dict(orient="records")
    feature_rows = joblib.Parallel(n_jobs=max(1, parallel_jobs), backend="threading")(
        joblib.delayed(_group_diagram_feature_row)(
            row,
            prototype_map=prototype_map,
            group_labels=group_labels,
        )
        for row in tqdm(rows, desc="group_proto_features")
    )
    return pd.DataFrame(feature_rows)


def _evaluate_label_space(
    feature_df: pd.DataFrame,
    *,
    labels: list[str],
    seed: int,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_df = feature_df.loc[feature_df["split"].eq("train")].copy()
    test_df = feature_df.loc[feature_df["split"].eq("test")].copy()
    unique_train = train_df.drop_duplicates("example_id").reset_index(drop=True)
    tr_idx, val_idx = _split_indices(unique_train["target_label"], val_fraction=0.2, seed=seed)
    inner_train_ids = set(unique_train.iloc[tr_idx]["example_id"].astype(str))
    val_ids = set(unique_train.iloc[val_idx]["example_id"].astype(str))

    rows: list[dict[str, Any]] = []
    for layer in sorted(train_df["layer"].unique()):
        layer_train = train_df.loc[train_df["layer"].eq(layer)].copy()
        inner_train = layer_train.loc[layer_train["example_id"].astype(str).isin(inner_train_ids)].copy()
        val_layer = layer_train.loc[layer_train["example_id"].astype(str).isin(val_ids)].copy()
        if inner_train.empty or val_layer.empty:
            continue
        columns = _topology_feature_columns(layer_train)
        clf, scaler = _fit_multiclass_logistic(
            inner_train.loc[:, columns].to_numpy(dtype=float),
            inner_train["target_label"].astype(str).to_numpy(),
            seed=seed + int(layer),
        )
        y_val = val_layer["target_label"].astype(str).to_numpy()
        pred = clf.predict(scaler.transform(val_layer.loc[:, columns].to_numpy(dtype=float)))
        metrics = _metrics(y_val, pred, labels)
        rows.append(
            {
                "method": "token_cloud_single_group_proto",
                "layer": int(layer),
                "val_accuracy": float(metrics["accuracy"]),
                "val_macro_f1": float(metrics["macro_f1"]),
                "feature_count": int(len(columns)),
            }
        )
    candidate_df = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy", "layer"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_single = candidate_df.iloc[0].to_dict()

    best_layer = int(best_single["layer"])
    final_train = train_df.loc[train_df["layer"].eq(best_layer)].copy()
    final_test = test_df.loc[test_df["layer"].eq(best_layer)].copy()
    columns = _topology_feature_columns(final_train)
    clf, scaler = _fit_multiclass_logistic(
        final_train.loc[:, columns].to_numpy(dtype=float),
        final_train["target_label"].astype(str).to_numpy(),
        seed=seed + 1000 + best_layer,
    )
    pred = clf.predict(scaler.transform(final_test.loc[:, columns].to_numpy(dtype=float)))
    single_metrics = _metrics(final_test["target_label"].astype(str).to_numpy(), pred, labels)
    single_result = {
        "method": "token_cloud_single_group_proto",
        "layer": best_layer,
        "selection_signature": str(best_layer),
        "feature_count": int(len(columns)),
        "accuracy": float(single_metrics["accuracy"]),
        "macro_f1": float(single_metrics["macro_f1"]),
        "confusion_matrix": single_metrics["confusion_matrix"],
        "labels": single_metrics["labels"],
    }

    selections = [
        {
            "layer": int(row["layer"]),
            "val_auroc": float(row["val_macro_f1"]),
            "val_accuracy": float(row["val_accuracy"]),
            "val_f1": float(row["val_macro_f1"]),
        }
        for row in candidate_df.head(min(int(top_k), len(candidate_df))).to_dict(orient="records")
    ]
    multi_train, multi_test, multi_meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    label_lookup = feature_df.loc[:, ["example_id", "target_label"]].drop_duplicates()
    multi_train = multi_train.merge(label_lookup, on="example_id", how="left")
    multi_test = multi_test.merge(label_lookup, on="example_id", how="left")
    multi_columns = list(multi_meta["topology_columns"]) + list(multi_meta["topology_summary_columns"])
    clf, scaler = _fit_multiclass_logistic(
        multi_train.loc[:, multi_columns].to_numpy(dtype=float),
        multi_train["target_label"].astype(str).to_numpy(),
        seed=seed + 2000,
    )
    pred = clf.predict(scaler.transform(multi_test.loc[:, multi_columns].to_numpy(dtype=float)))
    multi_metrics = _metrics(multi_test["target_label"].astype(str).to_numpy(), pred, labels)
    multi_result = {
        "method": "token_cloud_multilayer_group_proto",
        "layer": -1,
        "selection_signature": " | ".join(str(int(item["layer"])) for item in selections),
        "feature_count": int(len(multi_columns)),
        "accuracy": float(multi_metrics["accuracy"]),
        "macro_f1": float(multi_metrics["macro_f1"]),
        "confusion_matrix": multi_metrics["confusion_matrix"],
        "labels": multi_metrics["labels"],
    }
    return candidate_df, single_result, multi_result


def _baseline_rows(model_slug: str, label_space: str) -> list[dict[str, Any]]:
    if label_space == "4way":
        df = pd.read_parquet(BASELINE_4WAY)
        df = df.loc[df["model"].eq(model_slug) & df["method"].isin(["token_cloud_single", "token_cloud_multilayer"])].copy()
        mapping = {
            "token_cloud_single": "baseline_token_cloud_single",
            "token_cloud_multilayer": "baseline_token_cloud_multilayer",
        }
        rows = []
        for row in df.to_dict(orient="records"):
            rows.append(
                {
                    "method": mapping[str(row["method"])],
                    "layer": int(row["layer"]),
                    "selection_signature": row["selection_signature"],
                    "feature_count": int(row["feature_count"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "confusion_matrix": row["confusion_matrix"],
                    "labels": row["labels"],
                }
            )
        return rows
    if label_space == "9way":
        df = pd.read_parquet(BASELINE_9WAY)
        df = df.loc[df["model"].eq(model_slug) & df["method"].isin(["token_cloud_single", "token_cloud_multilayer"])].copy()
        mapping = {
            "token_cloud_single": "baseline_token_cloud_single",
            "token_cloud_multilayer": "baseline_token_cloud_multilayer",
        }
        rows = []
        for row in df.to_dict(orient="records"):
            rows.append(
                {
                    "method": mapping[str(row["method"])],
                    "layer": int(row["layer"]),
                    "selection_signature": row["selection_signature"],
                    "feature_count": int(row["feature_count"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "confusion_matrix": row["confusion_matrix"],
                    "labels": row["labels"],
                }
            )
        return rows
    raise ValueError(label_space)


def _render_report(results_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# CLAMBER Group-Prototype Token-Cloud Evaluation",
        "",
        "This compares the current binary-prototype token-cloud method against a group-specific prototype-distance variant.",
        "",
    ]
    for (model_label, label_space), group_df in results_df.groupby(["model_label", "label_space"], dropna=False):
        lines.extend([f"## {model_label} / {label_space}", ""])
        ordered = group_df.sort_values(["macro_f1", "accuracy"], ascending=False)
        for row in ordered.to_dict(orient="records"):
            lines.append(
                f"- `{row['method']}`: accuracy `{row['accuracy']:.4f}`, macro-F1 `{row['macro_f1']:.4f}`, "
                f"selection `{row['selection_signature']}`."
            )
        lines.append("")
    write_markdown(output_path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for model_slug in args.model_slugs:
        spec = MODEL_SPECS[model_slug]
        config = load_config(spec["config_path"])
        subclass_cfg = dict(config["clamber_subclass_classification"])
        classifier_config = dict(config["token_cloud_topology_classifier"])
        print(f"[group-proto] model={model_slug} start", flush=True)

        # rebuild raw clouds using the same extraction path but keep cloud dataframe
        bundle: HFModelBundle = load_hf_model(config["model"], classifier_config)
        dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
        dataset_df = pd.read_parquet(dataset_path)
        from aen_replication.train.token_cloud_topology_classifier import (
            _extract_reduced_clouds,
            _extract_train_token_matrices,
            _fit_layer_reducers,
            _prepare_prompt_frame,
        )
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=str(classifier_config.get("text_column", "text")),
            use_chat_template=bool(classifier_config.get("use_chat_template", False)),
            system_prompt=classifier_config.get("system_prompt"),
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
        layers = [int(layer) for layer in subclass_cfg.get("token_cloud_candidate_layers", classifier_config.get("candidate_layers", [0, 14, 31]))]
        train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
        token_cfg = {
            **classifier_config,
            "batch_size": int(subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8))),
            "max_length": int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64))),
            "parallel_jobs": int(subclass_cfg.get("token_cloud_parallel_jobs", classifier_config.get("parallel_jobs", 12))),
            "pca_components": int(subclass_cfg.get("token_cloud_pca_components", classifier_config.get("pca_components", 16))),
            "topology_components": int(subclass_cfg.get("token_cloud_topology_components", classifier_config.get("topology_components", 16))),
            "prototype_token_cap": int(subclass_cfg.get("token_cloud_prototype_token_cap", classifier_config.get("prototype_token_cap", 192))),
            "betti_grid_size": int(subclass_cfg.get("token_cloud_betti_grid_size", classifier_config.get("betti_grid_size", 24))),
            "persistence_image_grid_side": int(subclass_cfg.get("token_cloud_persistence_image_grid_side", classifier_config.get("persistence_image_grid_side", 3))),
            "_seed": args.seed,
        }
        token_matrices = _extract_train_token_matrices(
            bundle=bundle,
            train_df=train_df,
            text_column="_token_cloud_text",
            layers=layers,
            config=token_cfg,
        )
        reducers = _fit_layer_reducers(token_matrices, config=token_cfg, seed=args.seed)
        cloud_df = _extract_reduced_clouds(
            bundle=bundle,
            df=prepared_df,
            text_column="_token_cloud_text",
            layers=layers,
            reducers=reducers,
            config=token_cfg,
        )
        join_meta = prepared_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
        cloud_df = cloud_df.merge(join_meta, on="example_id", how="left")
        print(f"[group-proto] model={model_slug} reduced_cloud_rows={len(cloud_df)}", flush=True)
        diagram_df = _precompute_diagram_rows(cloud_df, config=token_cfg)
        print(f"[group-proto] model={model_slug} diagram_rows={len(diagram_df)}", flush=True)

        for label_space in args.label_spaces:
            print(f"[group-proto] model={model_slug} label_space={label_space}", flush=True)
            label_cloud_df, labels = _prepare_label_space(cloud_df, label_space=label_space)
            label_diagram_df, _ = _prepare_label_space(diagram_df, label_space=label_space)
            prototype_map = _prototype_diagrams_from_group_clouds(
                label_cloud_df,
                layers=layers,
                label_values=labels,
                config=token_cfg,
                seed=args.seed,
            )
            feature_df = _build_group_feature_frame(
                label_diagram_df,
                prototype_map=prototype_map,
                group_labels=labels,
                parallel_jobs=max(1, int(token_cfg.get("parallel_jobs", 8))),
            )
            candidate_df, single_result, multi_result = _evaluate_label_space(
                feature_df,
                labels=labels,
                seed=args.seed,
                top_k=2,
            )
            print(
                f"[group-proto] model={model_slug} label_space={label_space} "
                f"single={single_result['macro_f1']:.4f} multi={multi_result['macro_f1']:.4f}",
                flush=True,
            )
            candidate_df["model"] = model_slug
            candidate_df["model_label"] = spec["label"]
            candidate_df["label_space"] = label_space
            candidate_rows.extend(candidate_df.to_dict(orient="records"))
            for row in _baseline_rows(model_slug, label_space):
                row["model"] = model_slug
                row["model_label"] = spec["label"]
                row["label_space"] = label_space
                final_rows.append(row)
            for row in [single_result, multi_result]:
                row["model"] = model_slug
                row["model_label"] = spec["label"]
                row["label_space"] = label_space
                final_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows)
    results_df = pd.DataFrame(final_rows).sort_values(
        ["model_label", "label_space", "macro_f1", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    candidate_path = output_root / "clamber_group_prototype_token_cloud_candidates.parquet"
    results_path = output_root / "clamber_group_prototype_token_cloud_results.parquet"
    report_path = output_root / "clamber_group_prototype_token_cloud_report.md"
    metadata_path = output_root / "clamber_group_prototype_token_cloud_metadata.json"
    write_parquet(candidate_df, candidate_path)
    write_parquet(results_df, results_path)
    _render_report(results_df, report_path)
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "model_slugs": list(args.model_slugs),
            "label_spaces": list(args.label_spaces),
            "candidate_path": str(candidate_path),
            "results_path": str(results_path),
            "report_path": str(report_path),
        },
    )


if __name__ == "__main__":
    main()
