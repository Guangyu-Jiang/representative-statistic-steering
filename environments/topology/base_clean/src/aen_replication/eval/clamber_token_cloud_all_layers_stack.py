"""Direct all-layer token-cloud stack vs fixed 0|14 on CLAMBER.

This builds token-cloud topology features for every transformer layer of the
LLaMA 3.1 8B CLAMBER setup and compares:

- fixed current multilayer stack: 0 | 14
- direct all-layer stack: 0 | 1 | ... | 31

The main comparison is the current baseline feature set. A secondary ablation
also reruns the same two stacks after removing the pooled prototype-distance
features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from aen_replication.config import load_config
from aen_replication.eval.clamber_token_cloud_classavg_wasserstein import (
    GROUP4_MAP,
    MODEL_LABEL,
    MODEL_SLUG,
    _drop_prototype_distance_features,
)
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _distance_feature_mode,
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
    _prototype_diagrams_from_clouds,
    build_token_cloud_feature_frame,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

BASELINE_PATHS = {
    "4way": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
        "clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet"
    ),
    "9way": Path(
        "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
        "clamber_9way_comparison/clamber_9way_comparison_final_metrics.parquet"
    ),
}

NO_PROTO_PATH = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/"
    "clamber_topology_no_prototype/clamber_topology_no_prototype_final_metrics.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_token_cloud_clamber_pca16.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_token_cloud_all_layers_stack",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--parallel-jobs", type=int, default=12)
    parser.add_argument("--force-recompute", action="store_true")
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _prepare_task_frame(feature_df: pd.DataFrame, *, task: str) -> tuple[pd.DataFrame, str]:
    if task == "4way":
        df = feature_df.loc[feature_df["subclass"].isin(GROUP4_MAP)].copy()
        df["group4"] = df["subclass"].map(GROUP4_MAP)
        return df.reset_index(drop=True), "group4"
    if task == "9way":
        return feature_df.copy().reset_index(drop=True), "subclass"
    raise ValueError(f"Unsupported task: {task}")


def _evaluate_fixed_stack(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    layers: list[int],
    seed: int,
) -> dict[str, Any]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    label_lookup = feature_df.loc[:, ["example_id", label_col]].drop_duplicates()
    train_df = train_df.merge(label_lookup, on="example_id", how="left")
    test_df = test_df.merge(label_lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    labels = sorted(set(train_df[label_col].astype(str).tolist()) | set(test_df[label_col].astype(str).tolist()))
    clf, scaler = _fit_multiclass_logistic(
        train_df.loc[:, feature_cols].to_numpy(dtype=float),
        train_df[label_col].astype(str).to_numpy(),
        seed=seed,
    )
    y_pred = clf.predict(scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=float)))
    metrics = _compute_metrics(test_df[label_col].astype(str).to_numpy(), y_pred, labels)
    return {
        "selection_signature": " | ".join(str(int(layer)) for layer in layers),
        "selection_size": int(len(layers)),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
    }


def _load_previous_current_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for task, path in BASELINE_PATHS.items():
        df = pd.read_parquet(path).copy()
        df = df.loc[(df["model"].eq(MODEL_SLUG)) & (df["method"].eq("token_cloud_multilayer"))].copy()
        df["task"] = task
        rows.append(df.loc[:, ["task", "accuracy", "macro_f1", "selection_signature"]])
    out = pd.concat(rows, ignore_index=True)
    return out.rename(
        columns={
            "accuracy": "previous_accuracy",
            "macro_f1": "previous_macro_f1",
            "selection_signature": "previous_selection_signature",
        }
    )


def _load_previous_no_proto_metrics() -> pd.DataFrame:
    df = pd.read_parquet(NO_PROTO_PATH).copy()
    df = df.loc[(df["model"].eq(MODEL_SLUG)) & (df["method"].eq("token_cloud_multilayer"))].copy()
    return df.loc[:, ["task", "accuracy", "macro_f1", "selection_signature"]].rename(
        columns={
            "accuracy": "previous_accuracy",
            "macro_f1": "previous_macro_f1",
            "selection_signature": "previous_selection_signature",
        }
    )


def _compute_all_layer_feature_table(
    *,
    config: dict[str, Any],
    output_root: Path,
    seed: int,
    parallel_jobs: int,
    force_recompute: bool,
) -> pd.DataFrame:
    feature_path = output_root / "llama_clamber_token_cloud_all_layers_features.parquet"
    if feature_path.exists() and not force_recompute:
        return pd.read_parquet(feature_path).copy()

    classifier_config = dict(config["token_cloud_topology_classifier"])
    classifier_config["_seed"] = int(seed)
    classifier_config["parallel_jobs"] = int(parallel_jobs)

    bundle = load_hf_model(config["model"], classifier_config)
    total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = list(range(total_layers))

    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    dataset_df = pd.read_parquet(dataset_path).copy()
    prepared_df, prepared_text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle,
        text_column=str(classifier_config.get("text_column", "text")),
        use_chat_template=bool(classifier_config.get("use_chat_template", False)),
        system_prompt=classifier_config.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
    train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)

    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=layers,
        config=classifier_config,
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
    cloud_df = _extract_reduced_clouds(
        bundle=bundle,
        df=prepared_df.reset_index(drop=True),
        text_column="_token_cloud_text",
        layers=layers,
        reducers=reducers,
        config=classifier_config,
    )
    meta = prepared_df.loc[:, ["example_id", "subclass"]].drop_duplicates()
    cloud_df = cloud_df.merge(meta, on="example_id", how="left")

    prototype_map = None
    if _distance_feature_mode(classifier_config) == "prototype":
        prototype_map = _prototype_diagrams_from_clouds(cloud_df, layers=layers, config=classifier_config, seed=seed)

    parts: list[pd.DataFrame] = []
    for layer in layers:
        layer_path = feature_path.with_name(f"{feature_path.stem}__layer_{int(layer):02d}{feature_path.suffix}")
        if layer_path.exists() and not force_recompute:
            layer_features = pd.read_parquet(layer_path).copy()
        else:
            layer_clouds = cloud_df.loc[cloud_df["layer"].eq(layer)].copy()
            layer_features = build_token_cloud_feature_frame(
                layer_clouds,
                prototype_map=prototype_map,
                config=classifier_config,
            ).merge(
                layer_clouds.loc[:, ["example_id", "subclass"]].drop_duplicates(),
                on="example_id",
                how="left",
            )
            write_parquet(layer_features, layer_path)
        parts.append(layer_features)

    feature_df = pd.concat(parts, ignore_index=True)
    write_parquet(feature_df, feature_path)
    return feature_df


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    config = load_config(args.config)

    feature_df = _compute_all_layer_feature_table(
        config=config,
        output_root=output_root,
        seed=args.seed,
        parallel_jobs=args.parallel_jobs,
        force_recompute=bool(args.force_recompute),
    )

    all_layers = sorted(feature_df["layer"].unique().tolist())
    if all_layers != list(range(len(all_layers))):
        raise ValueError(f"Unexpected all-layer set: {all_layers}")

    results: list[dict[str, Any]] = []
    current_previous = _load_previous_current_metrics()
    no_proto_previous = _load_previous_no_proto_metrics()

    for task in ["4way", "9way"]:
        task_df, label_col = _prepare_task_frame(feature_df, task=task)
        current_row = current_previous.loc[current_previous["task"].eq(task)].iloc[0]

        for selection_name, layers in [("current_0_14", [0, 14]), ("all_layers", all_layers)]:
            payload = _evaluate_fixed_stack(
                task_df,
                label_col=label_col,
                layers=layers,
                seed=args.seed + (0 if task == "4way" else 100),
            )
            results.append(
                {
                    "task": task,
                    "feature_variant": "baseline",
                    "selection_name": selection_name,
                    "selection_signature": payload["selection_signature"],
                    "selection_size": int(payload["selection_size"]),
                    "feature_count": int(payload["feature_count"]),
                    "accuracy": float(payload["accuracy"]),
                    "macro_f1": float(payload["macro_f1"]),
                    "delta_accuracy_vs_current_stored": float(payload["accuracy"] - float(current_row["previous_accuracy"])),
                    "delta_macro_f1_vs_current_stored": float(payload["macro_f1"] - float(current_row["previous_macro_f1"])),
                    "comparison_signature": str(current_row["previous_selection_signature"]),
                    "confusion_matrix": payload["confusion_matrix"],
                    "labels": payload["labels"],
                }
            )

        no_proto_df, label_col = _prepare_task_frame(_drop_prototype_distance_features(feature_df), task=task)
        no_proto_row = no_proto_previous.loc[no_proto_previous["task"].eq(task)].iloc[0]
        for selection_name, layers in [("current_0_14", [0, 14]), ("all_layers", all_layers)]:
            payload = _evaluate_fixed_stack(
                no_proto_df,
                label_col=label_col,
                layers=layers,
                seed=args.seed + 1000 + (0 if task == "4way" else 100),
            )
            results.append(
                {
                    "task": task,
                    "feature_variant": "no_prototype",
                    "selection_name": selection_name,
                    "selection_signature": payload["selection_signature"],
                    "selection_size": int(payload["selection_size"]),
                    "feature_count": int(payload["feature_count"]),
                    "accuracy": float(payload["accuracy"]),
                    "macro_f1": float(payload["macro_f1"]),
                    "delta_accuracy_vs_current_stored": float(payload["accuracy"] - float(no_proto_row["previous_accuracy"])),
                    "delta_macro_f1_vs_current_stored": float(payload["macro_f1"] - float(no_proto_row["previous_macro_f1"])),
                    "comparison_signature": str(no_proto_row["previous_selection_signature"]),
                    "confusion_matrix": payload["confusion_matrix"],
                    "labels": payload["labels"],
                }
            )

    result_df = pd.DataFrame(results).sort_values(["task", "feature_variant", "selection_name"]).reset_index(drop=True)
    write_parquet(result_df, output_root / "clamber_token_cloud_all_layers_stack_results.parquet")

    report_lines = [
        "# CLAMBER Token-Cloud All-Layer Stack",
        "",
        f"- Model: `{MODEL_LABEL}`",
        f"- Direct all-layer stack: `{len(all_layers)} layers` = `{all_layers[0]} | ... | {all_layers[-1]}`",
        "- Main question: does stacking every layer help over the current `0 | 14` token-cloud stack?",
        "",
    ]

    for task in ["4way", "9way"]:
        task_df = result_df.loc[result_df["task"].eq(task)].copy()
        report_lines.extend(
            [
                f"## {task.upper()}",
                "",
                "| Feature Variant | Stack | Layers | Features | Acc | Macro-F1 | Delta Acc vs stored current | Delta Macro-F1 vs stored current | Reference |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in task_df.iterrows():
            report_lines.append(
                f"| {row['feature_variant']} | {row['selection_name']} | `{row['selection_signature']}` | "
                f"{int(row['feature_count'])} | {float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
                f"{float(row['delta_accuracy_vs_current_stored']):+.4f} | {float(row['delta_macro_f1_vs_current_stored']):+.4f} | "
                f"`{row['comparison_signature']}` |"
            )
        report_lines.append("")

    write_markdown(output_root / "clamber_token_cloud_all_layers_stack_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
