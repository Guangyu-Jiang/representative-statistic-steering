"""Raw-space all-layer H0 mean persistence for CLAMBER 9-way classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import warnings

from aen_replication.config import load_config
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.clamber_subclass_classification import _evaluate_multiclass
from aen_replication.train.token_cloud_topology_classifier import _prepare_prompt_frame, _valid_token_mask
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json


DEFAULT_CONFIGS = [
    "configs/runs/gemma_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
    "configs/runs/llama_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
    "configs/runs/mistral_clamber_pca16_9way_nodist_fulllayers_nohybrid_topofeaturev2.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="artifacts/reports/clamber_h0_no_pca_all_layers")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _h0_mean_from_distance_matrix(distances: np.ndarray) -> float:
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return 0.0
    weights = np.asarray(minimum_spanning_tree(matrix).data, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if weights.size == 0:
        return 0.0
    return float(weights.mean())


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size].copy()


def _raw_h0_feature_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / slugify(model_name) / "clamber_h0_mean_persistence_no_pca_all_layers.parquet"


def _extract_raw_h0_features(
    *,
    config: dict[str, Any],
    output_dir: Path,
    batch_size_override: int | None,
    force_recompute: bool,
) -> tuple[pd.DataFrame, bool]:
    model_name = str(config["model"]["name"])
    feature_path = _raw_h0_feature_path(output_dir, model_name)
    metadata_path = feature_path.with_suffix(".metadata.json")
    if feature_path.exists() and not force_recompute:
        return pd.read_parquet(feature_path).copy(), True

    classifier_config = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    if batch_size_override is not None:
        batch_size = int(batch_size_override)
    else:
        batch_size = int(subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8)))
    max_length = int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64)))
    drop_special_tokens = bool(classifier_config.get("drop_special_tokens", True))

    bundle = load_hf_model(config["model"], classifier_config)
    total_layers = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = list(range(total_layers))
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)

    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    dataset_df = pd.read_parquet(dataset_path).copy()
    prepared_df, prepared_text_column = _prepare_prompt_frame(
        dataset_df,
        bundle=bundle,
        text_column=str(classifier_config.get("text_column", "text")),
        use_chat_template=bool(classifier_config.get("use_chat_template", False)),
        system_prompt=classifier_config.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column].astype(str)

    rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(prepared_df), desc=f"{slugify(model_name)}_raw_h0", unit="example")
    try:
        for batch_df in _iter_batches(prepared_df.reset_index(drop=True), batch_size):
            encoded = tokenizer(
                batch_df["_token_cloud_text"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states for raw H0 extraction.")

            input_ids_cpu = input_ids.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu()
            batch_records = batch_df.reset_index(drop=True).to_dict(orient="records")
            valid_masks_cpu = [
                _valid_token_mask(
                    input_ids_cpu[row_index],
                    attention_mask_cpu[row_index],
                    special_ids=special_ids,
                    drop_special_tokens=drop_special_tokens,
                )
                for row_index in range(len(batch_records))
            ]

            for layer in layers:
                layer_output = hidden_states[layer + 1].detach().float()
                distance_batch = torch.cdist(layer_output, layer_output, p=2).detach().cpu().numpy()
                for row_index, row in enumerate(batch_records):
                    valid = valid_masks_cpu[row_index].numpy().astype(bool)
                    token_count = int(valid.sum())
                    if token_count <= 1:
                        h0_mean = 0.0
                    else:
                        row_distances = distance_batch[row_index][np.ix_(valid, valid)]
                        h0_mean = _h0_mean_from_distance_matrix(row_distances)
                    rows.append(
                        {
                            "example_id": str(row["example_id"]),
                            "pair_id": str(row["pair_id"]),
                            "dataset": str(row["dataset"]),
                            "split": str(row["split"]),
                            "label_ambiguous": int(row["label_ambiguous"]),
                            "subclass": str(row["subclass"]),
                            "layer": int(layer),
                            "token_count": token_count,
                            "h0_mean_persistence": np.float32(h0_mean),
                        }
                    )

            del outputs, hidden_states, model_inputs, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update(len(batch_records))
    finally:
        progress.close()

    feature_df = pd.DataFrame(rows).sort_values(["split", "example_id", "layer"]).reset_index(drop=True)
    ensure_dir(feature_path.parent)
    feature_df.to_parquet(feature_path, index=False)
    write_json(
        metadata_path,
        {
            "model_name": model_name,
            "dataset_path": str(dataset_path),
            "rows": int(len(feature_df)),
            "examples": int(feature_df["example_id"].nunique()),
            "layers": layers,
            "feature": "h0_mean_persistence",
            "pca_used": False,
            "distance_metric": "euclidean",
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "drop_special_tokens": bool(drop_special_tokens),
        },
    )
    return feature_df, False


def _all_layer_wide_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    index_columns = ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "subclass"]
    wide = (
        feature_df.pivot_table(
            index=index_columns,
            columns="layer",
            values="h0_mean_persistence",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    layer_columns = sorted(column for column in wide.columns if isinstance(column, (int, np.integer)))
    rename_map = {layer: f"h0_mean_persistence__l{int(layer):02d}" for layer in layer_columns}
    wide = wide.rename(columns=rename_map)
    feature_columns = [rename_map[layer] for layer in layer_columns]
    return wide, feature_columns


def _evaluate_all_layer_h0(
    *,
    model_name: str,
    feature_df: pd.DataFrame,
    seed: int,
    max_iter: int,
    c_value: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    wide_df, feature_columns = _all_layer_wide_frame(feature_df)
    train_df = wide_df.loc[wide_df["split"].eq("train")].copy()
    test_df = wide_df.loc[wide_df["split"].eq("test")].copy()
    payload = _evaluate_multiclass(
        x_train=train_df.loc[:, feature_columns].to_numpy(dtype=float),
        y_train=train_df["subclass"].astype(str).to_numpy(),
        x_eval=test_df.loc[:, feature_columns].to_numpy(dtype=float),
        y_eval=test_df["subclass"].astype(str).to_numpy(),
        max_iter=max_iter,
        c_value=c_value,
        seed=seed,
    )
    metrics = payload["metrics"]
    coefficients = np.asarray(payload["classifier"].coef_, dtype=float)
    classes = [str(item) for item in payload["classifier"].classes_.tolist()]
    importance = pd.DataFrame(
        {
            "model": slugify(model_name),
            "model_name": model_name,
            "feature": feature_columns,
            "layer": [int(column.rsplit("l", 1)[1]) for column in feature_columns],
            "mean_abs_coefficient": np.mean(np.abs(coefficients), axis=0),
            "max_abs_coefficient": np.max(np.abs(coefficients), axis=0),
        }
    )
    for class_index, class_name in enumerate(classes):
        importance[f"coef_{slugify(class_name)}"] = coefficients[class_index]
    row = {
        "model": slugify(model_name),
        "model_name": model_name,
        "dataset": "clamber",
        "label_space": "9_subclasses",
        "method": "h0_mean_persistence_no_pca_all_layers",
        "layer": "all",
        "selection_signature": " | ".join(str(index) for index in range(len(feature_columns))),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "test_confusion_matrix": metrics["confusion_matrix"],
        "test_labels": metrics["labels"],
    }
    return row, importance.sort_values("mean_abs_coefficient", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = ensure_dir(Path(args.output_dir))
    metric_rows: list[dict[str, Any]] = []
    importance_parts: list[pd.DataFrame] = []
    cache_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        subclass_cfg = dict(config["clamber_subclass_classification"])
        feature_df, reused = _extract_raw_h0_features(
            config=config,
            output_dir=output_dir,
            batch_size_override=args.batch_size,
            force_recompute=bool(args.force_recompute),
        )
        row, importance = _evaluate_all_layer_h0(
            model_name=model_name,
            feature_df=feature_df,
            seed=int(args.seed),
            max_iter=int(subclass_cfg.get("max_iter", 4000)),
            c_value=float(subclass_cfg.get("token_cloud_classifier_C", 1.0)),
        )
        metric_rows.append(row)
        importance_parts.append(importance)
        cache_rows.append(
            {
                "model": slugify(model_name),
                "model_name": model_name,
                "raw_h0_feature_cache_path": str(_raw_h0_feature_path(output_dir, model_name)),
                "raw_h0_feature_cache_reused": bool(reused),
                "pca_used": False,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("model").reset_index(drop=True)
    importance_df = (
        pd.concat(importance_parts, ignore_index=True).sort_values(["model", "mean_abs_coefficient"], ascending=[True, False]).reset_index(drop=True)
        if importance_parts
        else pd.DataFrame()
    )
    cache_df = pd.DataFrame(cache_rows).sort_values("model").reset_index(drop=True)

    metrics_path = output_dir / "clamber_h0_no_pca_all_layers_metrics.csv"
    importance_path = output_dir / "clamber_h0_no_pca_all_layers_importance.csv"
    cache_path = output_dir / "clamber_h0_no_pca_all_layers_cache_status.csv"
    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    cache_df.to_csv(cache_path, index=False)
    write_json(
        output_dir / "clamber_h0_no_pca_all_layers_outputs.json",
        {
            "metrics_csv": str(metrics_path),
            "importance_csv": str(importance_path),
            "cache_status_csv": str(cache_path),
        },
    )

    print(metrics_df.to_string(index=False))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {importance_path}")
    print(f"Wrote {cache_path}")


if __name__ == "__main__":
    main()
