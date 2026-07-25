"""Mapper-based token-cloud classifier from per-token hidden states.

This stage treats each question as a token point cloud, builds a local Mapper
graph from the reduced token coordinates, extracts graph descriptors, and
trains ambiguity classifiers from those Mapper-derived features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.features.mapper_analysis import MapperSetting, _build_mapper_graph
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.independent_topology_classifier import (
    _extract_model_signal,
    _fit_classifier,
    _group_train_val_split,
    _predict_scores,
    _select_multilayer_candidates,
    _selection_order,
    _stacked_summary_features,
    _transform_with_scaler,
)
from aen_replication.train.token_cloud_topology_classifier import (
    BASE_KEY_COLUMNS,
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
    _resolve_candidate_layers,
)
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

LOGGER = logging.getLogger(__name__)

MAPPER_PREFIX = "mapper_"
MAPPER_STAT_KEYS = [
    "node_count",
    "edge_count",
    "connected_components",
    "largest_component_size",
    "average_degree",
    "branch_node_count",
    "singleton_node_count",
    "mean_node_size",
    "max_node_size",
]
MAPPER_SUMMARY_KEYS = [
    "node_count",
    "edge_count",
    "connected_components",
    "average_degree",
    "branch_node_fraction",
    "largest_component_fraction",
    "graph_density",
]


def _slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _mapper_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in feature_df.columns if column.startswith(MAPPER_PREFIX)]


def _setting_column_slug(setting: MapperSetting) -> str:
    overlap = int(round(setting.overlap * 100))
    eps = int(round(setting.eps * 100))
    return f"{setting.lens_name}__n{setting.n_intervals}__o{overlap:02d}__eps{eps:03d}__m{setting.min_samples}"


def _build_mapper_point_frame(
    cloud: np.ndarray,
    *,
    example_id: str,
) -> pd.DataFrame:
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2:
        raise ValueError(f"Expected 2D cloud array, got shape {cloud.shape}.")
    if cloud.shape[0] == 0:
        return pd.DataFrame(columns=["example_id", "z_0", "z_1", "radius"])
    if cloud.shape[1] == 1:
        coords = np.column_stack([cloud[:, 0], np.zeros(cloud.shape[0], dtype=float)])
    else:
        coords = cloud[:, :2]
    centroid = coords.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(coords - centroid, axis=1)
    return pd.DataFrame(
        {
            "example_id": [f"{example_id}__t{index:03d}" for index in range(len(coords))],
            "z_0": coords[:, 0],
            "z_1": coords[:, 1],
            "radius": radius,
        }
    )


def _mapper_settings_from_config(config: dict[str, Any]) -> list[MapperSetting]:
    lenses = [str(name) for name in config.get("mapper_lenses", ["z_0", "radius"])]
    n_intervals = [int(value) for value in config.get("mapper_n_intervals", [4, 6])]
    overlaps = [float(value) for value in config.get("mapper_overlaps", [0.3, 0.5])]
    eps_values = config.get("mapper_dbscan_eps_values")
    if eps_values is None:
        eps_values = [float(config.get("mapper_dbscan_eps", 0.6))]
    else:
        eps_values = [float(value) for value in eps_values]
    min_samples_values = config.get("mapper_min_samples_values")
    if min_samples_values is None:
        min_samples_values = [int(config.get("mapper_min_samples", 2))]
    else:
        min_samples_values = [int(value) for value in min_samples_values]
    settings: list[MapperSetting] = []
    for lens_name in lenses:
        for interval_count in n_intervals:
            for overlap in overlaps:
                for eps in eps_values:
                    for min_samples in min_samples_values:
                        settings.append(
                            MapperSetting(
                                lens_name=lens_name,
                                n_intervals=interval_count,
                                overlap=overlap,
                                eps=eps,
                                min_samples=min_samples,
                            )
                        )
    return settings


def _normalize_mapper_stats(stats: dict[str, Any], *, token_count: int) -> dict[str, float]:
    token_scale = max(int(token_count), 1)
    node_count = float(stats["node_count"])
    edge_count = float(stats["edge_count"])
    component_count = float(stats["connected_components"])
    largest_component = float(stats["largest_component_size"])
    branch_count = float(stats["branch_node_count"])
    singleton_count = float(stats["singleton_node_count"])
    mean_node_size = float(stats["mean_node_size"])
    max_node_size = float(stats["max_node_size"])
    if node_count <= 1.0:
        graph_density = 0.0
    else:
        graph_density = float((2.0 * edge_count) / max(node_count * (node_count - 1.0), 1.0))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "connected_components": component_count,
        "largest_component_size": largest_component,
        "average_degree": float(stats["average_degree"]),
        "branch_node_count": branch_count,
        "singleton_node_count": singleton_count,
        "mean_node_size": mean_node_size,
        "max_node_size": max_node_size,
        "graph_density": graph_density,
        "node_count_norm": node_count / token_scale,
        "edge_count_norm": edge_count / token_scale,
        "branch_node_fraction": branch_count / max(node_count, 1.0),
        "singleton_node_fraction": singleton_count / max(node_count, 1.0),
        "largest_component_fraction": largest_component / max(node_count, 1.0),
        "mean_node_size_norm": mean_node_size / token_scale,
        "max_node_size_norm": max_node_size / token_scale,
    }


def _aggregate_mapper_features(per_setting_rows: list[dict[str, float]]) -> dict[str, float]:
    if not per_setting_rows:
        return {}
    aggregate: dict[str, float] = {}
    for metric_name in MAPPER_SUMMARY_KEYS:
        values = np.asarray([row[metric_name] for row in per_setting_rows], dtype=float)
        aggregate[f"{MAPPER_PREFIX}summary__{metric_name}__mean"] = float(np.mean(values))
        aggregate[f"{MAPPER_PREFIX}summary__{metric_name}__std"] = float(np.std(values))
        aggregate[f"{MAPPER_PREFIX}summary__{metric_name}__max"] = float(np.max(values))
    return aggregate


def _cloud_mapper_feature_row(
    row: dict[str, Any],
    *,
    settings: list[MapperSetting],
) -> dict[str, Any]:
    cloud = np.asarray(row["cloud"], dtype=float)
    token_count = int(row["token_count"])
    point_frame = _build_mapper_point_frame(cloud, example_id=str(row["example_id"]))
    feature_row: dict[str, Any] = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "layer": int(row["layer"]),
        "token_count": token_count,
    }
    per_setting_rows: list[dict[str, float]] = []
    for setting in settings:
        _, _, stats = _build_mapper_graph(point_frame, setting)
        normalized = _normalize_mapper_stats(stats, token_count=token_count)
        per_setting_rows.append(normalized)
        setting_slug = _setting_column_slug(setting)
        for metric_name, value in normalized.items():
            feature_row[f"{MAPPER_PREFIX}{setting_slug}__{metric_name}"] = float(value)
    feature_row.update(_aggregate_mapper_features(per_setting_rows))
    return feature_row


def build_token_cloud_mapper_feature_frame(
    cloud_df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    if cloud_df.empty:
        return pd.DataFrame()
    settings = _mapper_settings_from_config(config)
    rows = cloud_df.to_dict(orient="records")
    parallel_jobs = max(1, int(config.get("parallel_jobs", 1)))
    feature_rows = joblib.Parallel(n_jobs=parallel_jobs, backend="loky")(
        joblib.delayed(_cloud_mapper_feature_row)(row, settings=settings) for row in rows
    )
    return pd.DataFrame(feature_rows)


def _evaluate_feature_set(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    *,
    feature_columns: list[str],
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train = train_features.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_features["label_ambiguous"].to_numpy(dtype=int)
    x_eval = eval_features.loc[:, feature_columns].to_numpy(dtype=float)
    y_eval = eval_features["label_ambiguous"].to_numpy(dtype=int)
    clf, scaler = _fit_classifier(x_train, y_train, config=classifier_config, seed=seed)
    train_scores = _predict_scores(clf, _transform_with_scaler(x_train, scaler))
    eval_scores = _predict_scores(clf, _transform_with_scaler(x_eval, scaler))
    coefficients, intercept = _extract_model_signal(clf)
    payload = {
        "classifier": clf,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "train_metrics": binary_classification_metrics(y_train, train_scores),
        "eval_metrics": binary_classification_metrics(y_eval, eval_scores),
        "coefficients": coefficients,
        "intercept": intercept,
    }
    return payload["train_metrics"], payload


def _prepare_layer_feature_frame(feature_df: pd.DataFrame, *, layer: int) -> tuple[pd.DataFrame, list[str]]:
    mapper_columns = _mapper_feature_columns(feature_df)
    suffix = f"l{int(layer):02d}"
    renamed = feature_df.loc[:, BASE_KEY_COLUMNS + mapper_columns].copy()
    rename_map = {column: f"{column}__{suffix}" for column in mapper_columns}
    renamed = renamed.rename(columns=rename_map)
    return renamed, [rename_map[column] for column in mapper_columns]


def _build_multilayer_feature_frames(
    feature_df: pd.DataFrame,
    *,
    dataset: str,
    selections: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_merged: pd.DataFrame | None = None
    test_merged: pd.DataFrame | None = None
    mapper_columns: list[str] = []
    selection_specs: list[dict[str, Any]] = []
    dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
    for rank, selection in enumerate(selections, start=1):
        layer = int(selection["layer"])
        selection_specs.append({"rank": rank, "layer": layer, "val_auroc": float(selection["val_auroc"])})
        layer_df = dataset_df.loc[dataset_df["layer"].eq(layer)].copy()
        train_layer = layer_df.loc[layer_df["split"].eq("train")].copy()
        test_layer = layer_df.loc[layer_df["split"].eq("test")].copy()
        train_prepared, train_mapper = _prepare_layer_feature_frame(train_layer, layer=layer)
        test_prepared, _ = _prepare_layer_feature_frame(test_layer, layer=layer)
        mapper_columns.extend(train_mapper)
        if train_merged is None:
            train_merged = train_prepared
            test_merged = test_prepared
        else:
            train_merged = train_merged.merge(train_prepared, on=BASE_KEY_COLUMNS, how="inner")
            test_merged = test_merged.merge(test_prepared, on=BASE_KEY_COLUMNS, how="inner")
    if train_merged is None or test_merged is None or train_merged.empty or test_merged.empty:
        raise ValueError("Failed to build non-empty Mapper multilayer features.")
    mapper_columns = [column for column in mapper_columns if column in train_merged.columns]
    train_summary, train_groups = _stacked_summary_features(train_merged, metric_groups={"mapper": mapper_columns})
    test_summary, _ = _stacked_summary_features(test_merged, metric_groups={"mapper": mapper_columns})
    train_multilayer = pd.concat([train_merged.reset_index(drop=True), train_summary], axis=1)
    test_multilayer = pd.concat([test_merged.reset_index(drop=True), test_summary], axis=1)
    return train_multilayer, test_multilayer, {
        "selections": selection_specs,
        "mapper_columns": mapper_columns,
        "mapper_summary_columns": train_groups["mapper"],
    }


def run_token_cloud_mapper_classifier_from_features(
    *,
    model_name: str,
    feature_df: pd.DataFrame,
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    if feature_df.empty:
        raise ValueError("Token-cloud Mapper classifier received an empty feature table.")

    output_root = ensure_dir(Path(classifier_config["output_dir"]) / _slugify_model_name(model_name))
    models_root = ensure_dir(output_root / "models")

    datasets = list(classifier_config.get("datasets", sorted(feature_df["dataset"].unique())))
    classifier_section = dict(classifier_config.get("classifier", {}))
    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_tables: list[pd.DataFrame] = []

    for dataset in datasets:
        dataset_df = feature_df.loc[feature_df["dataset"].eq(dataset)].copy()
        train_df = dataset_df.loc[dataset_df["split"].eq("train")].copy()
        test_df = dataset_df.loc[dataset_df["split"].eq("test")].copy()
        if train_df.empty or test_df.empty:
            LOGGER.warning("Skipping Mapper token-cloud dataset %s because train/test rows are missing.", dataset)
            continue
        inner_train_ids, val_ids = _group_train_val_split(
            train_df,
            val_fraction=float(classifier_config.get("val_fraction", 0.2)),
            seed=seed,
        )
        for layer in sorted(dataset_df["layer"].unique()):
            layer_train = train_df.loc[train_df["layer"].eq(layer)].copy()
            inner_train = layer_train.loc[layer_train["example_id"].astype(str).isin(inner_train_ids)].copy()
            val_df = layer_train.loc[layer_train["example_id"].astype(str).isin(val_ids)].copy()
            if inner_train.empty or val_df.empty:
                continue
            columns = _mapper_feature_columns(layer_train)
            _, payload = _evaluate_feature_set(
                train_features=inner_train,
                eval_features=val_df,
                feature_columns=columns,
                classifier_config=classifier_section,
                seed=seed,
            )
            metrics = payload["eval_metrics"]
            candidate_rows.append(
                {
                    "dataset": dataset,
                    "layer": int(layer),
                    "selection_mode": "single_layer",
                    "feature_set": "mapper_only",
                    "val_auroc": float(metrics["auroc"]),
                    "val_accuracy": float(metrics["accuracy"]),
                    "val_f1": float(metrics["f1"]),
                    "feature_count": int(len(columns)),
                }
            )

        candidate_df = pd.DataFrame(candidate_rows)
        dataset_candidates = candidate_df.loc[candidate_df["dataset"].eq(dataset)].copy()
        if dataset_candidates.empty:
            continue
        best_row = _selection_order(dataset_candidates).iloc[0]
        selected_rows.append({**best_row.to_dict(), "component_rank": 1})

        best_layer = int(best_row["layer"])
        final_train = train_df.loc[train_df["layer"].eq(best_layer)].copy()
        final_test = test_df.loc[test_df["layer"].eq(best_layer)].copy()
        feature_columns = _mapper_feature_columns(final_train)
        feature_tables.extend([final_train.assign(feature_variant="single_layer"), final_test.assign(feature_variant="single_layer")])
        _, payload = _evaluate_feature_set(
            train_features=final_train,
            eval_features=final_test,
            feature_columns=feature_columns,
            classifier_config=classifier_section,
            seed=seed,
        )
        metrics = payload["eval_metrics"]
        final_rows.append(
            {
                "dataset": dataset,
                "selection_mode": "single_layer",
                "layer": best_layer,
                "selection_signature": str(best_layer),
                "selection_size": 1,
                "feature_set": "mapper_only",
                "test_auroc": float(metrics["auroc"]),
                "test_accuracy": float(metrics["accuracy"]),
                "test_f1": float(metrics["f1"]),
                "feature_count": int(len(feature_columns)),
            }
        )
        joblib.dump(
            {
                "classifier": payload["classifier"],
                "scaler": payload["scaler"],
                "feature_columns": feature_columns,
                "dataset": dataset,
                "selection_mode": "single_layer",
                "layer": best_layer,
                "train_metrics": payload["train_metrics"],
                "test_metrics": payload["eval_metrics"],
            },
            models_root / f"{dataset}__mapper_only.joblib",
        )

        if bool(classifier_config.get("multilayer_enabled", True)):
            selections = _select_multilayer_candidates(
                dataset_candidates,
                top_k=int(classifier_config.get("multilayer_top_k", 3)),
            )
            for rank, row in enumerate(selections, start=1):
                selected_rows.append(
                    {
                        **row,
                        "feature_set": "mapper_multilayer",
                        "selection_mode": "multilayer_component",
                        "component_rank": rank,
                    }
                )
            multi_train, multi_test, multi_meta = _build_multilayer_feature_frames(
                feature_df,
                dataset=dataset,
                selections=selections,
            )
            multi_columns = multi_meta["mapper_columns"] + multi_meta["mapper_summary_columns"]
            feature_tables.extend([multi_train.assign(feature_variant="multilayer"), multi_test.assign(feature_variant="multilayer")])
            _, payload = _evaluate_feature_set(
                train_features=multi_train,
                eval_features=multi_test,
                feature_columns=multi_columns,
                classifier_config=classifier_section,
                seed=seed,
            )
            metrics = payload["eval_metrics"]
            selection_signature = " | ".join(str(int(item["layer"])) for item in multi_meta["selections"])
            final_rows.append(
                {
                    "dataset": dataset,
                    "selection_mode": "multilayer",
                    "layer": -1,
                    "selection_signature": selection_signature,
                    "selection_size": int(len(multi_meta["selections"])),
                    "feature_set": "mapper_multilayer",
                    "test_auroc": float(metrics["auroc"]),
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_f1": float(metrics["f1"]),
                    "feature_count": int(len(multi_columns)),
                }
            )
            joblib.dump(
                {
                    "classifier": payload["classifier"],
                    "scaler": payload["scaler"],
                    "feature_columns": multi_columns,
                    "dataset": dataset,
                    "selection_mode": "multilayer",
                    "selections": multi_meta["selections"],
                    "train_metrics": payload["train_metrics"],
                    "test_metrics": payload["eval_metrics"],
                },
                models_root / f"{dataset}__mapper_multilayer.joblib",
            )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["dataset", "layer"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["dataset", "feature_set"]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values(["dataset", "feature_set", "selection_mode", "component_rank"]).reset_index(drop=True)
    feature_table = pd.concat(feature_tables, ignore_index=True, sort=False) if feature_tables else pd.DataFrame()

    candidate_path = output_root / str(classifier_config["candidate_metrics_filename"])
    final_path = output_root / str(classifier_config["final_metrics_filename"])
    selected_path = output_root / str(classifier_config["selected_candidates_filename"])
    feature_path = output_root / str(classifier_config["feature_table_filename"])
    report_path = output_root / str(classifier_config["report_filename"])
    metadata_path = output_root / str(classifier_config["metadata_filename"])

    write_parquet(candidate_df, candidate_path)
    write_parquet(final_df, final_path)
    write_parquet(selected_df, selected_path)
    if not feature_table.empty:
        write_parquet(feature_table, feature_path)

    lines = [
        "# Token-Cloud Mapper Classifier",
        "",
        f"- Model: `{model_name}`",
        "",
        "## Final Results",
        "",
    ]
    if not final_df.empty:
        for dataset in sorted(final_df["dataset"].unique()):
            lines.append(f"### {dataset}")
            lines.append("")
            dataset_df = final_df.loc[final_df["dataset"].eq(dataset)]
            for row in dataset_df.to_dict(orient="records"):
                lines.append(
                    f"- `{row['feature_set']}`: AUROC `{row['test_auroc']:.4f}`, "
                    f"accuracy `{row['test_accuracy']:.4f}`, selection `{row['selection_signature']}`"
                )
            lines.append("")
    write_markdown(report_path, "\n".join(lines) + "\n")
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "created_at": utc_now_iso(),
            "datasets": datasets,
            "mapper_settings": [setting.setting_id for setting in _mapper_settings_from_config(classifier_config)],
            "output_artifacts": {
                "candidate_metrics": str(candidate_path),
                "final_metrics": str(final_path),
                "selected_candidates": str(selected_path),
                "feature_table": str(feature_path),
                "report": str(report_path),
            },
        },
    )
    return {
        "candidate_metrics_path": str(candidate_path),
        "final_metrics_path": str(final_path),
        "selected_candidates_path": str(selected_path),
        "feature_table_path": str(feature_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }


def run_token_cloud_mapper_classifier_analysis(
    *,
    config: dict[str, Any],
    classifier_config: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    model_name = str(config["model"]["name"])
    bundle = load_hf_model(config["model"], classifier_config)
    total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = _resolve_candidate_layers(total_layers, classifier_config)
    LOGGER.info("Token-cloud Mapper candidate layers: %s", layers)

    dataset_frames: list[pd.DataFrame] = []
    pair_output_dir = Path(config["data"]["pair_output_dir"])
    text_column = str(classifier_config.get("text_column", "text"))
    use_chat_template = bool(classifier_config.get("use_chat_template", False))
    system_prompt = classifier_config.get("system_prompt")
    datasets = list(classifier_config.get("datasets", ["ambigqa"]))
    for dataset in datasets:
        path = pair_output_dir / f"{dataset}_pairs.parquet"
        dataset_df = pd.read_parquet(path)
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=text_column,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
        dataset_frames.append(prepared_df)
    full_df = pd.concat(dataset_frames, ignore_index=True)
    train_df = full_df.loc[full_df["split"].eq("train")].copy().reset_index(drop=True)

    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=layers,
        config={**classifier_config, "_seed": seed},
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=seed)
    cloud_df = _extract_reduced_clouds(
        bundle=bundle,
        df=full_df,
        text_column="_token_cloud_text",
        layers=layers,
        reducers=reducers,
        config=classifier_config,
    )
    feature_df = build_token_cloud_mapper_feature_frame(cloud_df, config=classifier_config)
    return run_token_cloud_mapper_classifier_from_features(
        model_name=model_name,
        feature_df=feature_df,
        classifier_config=classifier_config,
        seed=seed,
    )
