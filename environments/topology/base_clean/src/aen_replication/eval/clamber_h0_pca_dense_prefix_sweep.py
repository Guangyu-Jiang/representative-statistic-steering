"""Dense PCA-prefix H0 sweep for CLAMBER all-layer subclass classification.

This script is designed for durable long runs. It separates GPU-bound PCA
projection from CPU-bound H0 computation and classifier evaluation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import warnings

from aen_replication.config import load_config
from aen_replication.eval.clamber_h0_pca_dim_sweep import (
    DEFAULT_CONFIGS,
    _cached_forward_path,
    _fit_or_load_reducers,
    _iter_batches,
    _load_cached_cloud_frame,
    _reducers_path,
)
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.clamber_subclass_classification import _evaluate_multiclass
from aen_replication.train.token_cloud_topology_classifier import _prepare_prompt_frame, _valid_token_mask
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json


DEFAULT_OUTPUT_DIR = "artifacts/reports/clamber_h0_pca_dense_prefix_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-sweep-dir", default="artifacts/reports/clamber_h0_pca_dim_sweep")
    parser.add_argument("--stage", choices=["fit", "project", "h0", "evaluate", "all"], default="all")
    parser.add_argument("--max-dim", default="512", help="'512', 'full', or an integer PCA prefix limit.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pca-fit-token-cap", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--force-project", action="store_true")
    parser.add_argument("--force-h0", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _model_root(output_dir: Path, model_name: str, max_dim: int) -> Path:
    return output_dir / slugify(model_name) / f"pca_prefix_{int(max_dim):04d}"


def _infer_hidden_size_from_reducers(reducers: dict[int, Any]) -> int:
    first = reducers[sorted(reducers)[0]]
    return int(np.asarray(first.components_).shape[1])


def _resolve_max_dim(config: dict[str, Any], source_sweep_dir: Path, output_dir: Path, requested: str) -> int:
    model_name = str(config["model"]["name"])
    if requested != "full":
        return int(requested)
    p512 = source_sweep_dir / slugify(model_name) / "pca_reducers_max512.joblib"
    if p512.exists():
        return _infer_hidden_size_from_reducers(joblib.load(p512))
    full_candidates = sorted((output_dir / slugify(model_name)).glob("pca_prefix_*/pca_reducers_max*.joblib"))
    if full_candidates:
        return _infer_hidden_size_from_reducers(joblib.load(full_candidates[-1]))
    raise FileNotFoundError(f"Cannot infer hidden size for {model_name}; expected {p512}")


def _copy_or_link_reducers(config: dict[str, Any], source_sweep_dir: Path, model_root: Path, max_dim: int) -> Path:
    model_name = str(config["model"]["name"])
    source = source_sweep_dir / slugify(model_name) / f"pca_reducers_max{int(max_dim):03d}.joblib"
    dest = model_root / f"pca_reducers_max{int(max_dim):03d}.joblib"
    if dest.exists():
        return dest
    if not source.exists():
        return dest
    ensure_dir(dest.parent)
    try:
        dest.symlink_to(source.resolve())
    except FileExistsError:
        pass
    except OSError:
        reducers = joblib.load(source)
        joblib.dump(reducers, dest)
    source_meta = source.with_suffix(".metadata.json")
    if source_meta.exists():
        dest_meta = dest.with_suffix(".metadata.json")
        if not dest_meta.exists():
            try:
                dest_meta.symlink_to(source_meta.resolve())
            except OSError:
                dest_meta.write_text(source_meta.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def _compile_h0_library(output_dir: Path) -> Path:
    source = Path(__file__).with_name("_h0_prefix_mst.cpp")
    build_dir = ensure_dir(output_dir / "_build")
    library = build_dir / "libh0_prefix_mst.so"
    if library.exists() and library.stat().st_mtime >= source.stat().st_mtime:
        return library
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    return library


def _load_h0_function(library_path: str | Path):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.h0_prefix_means
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    fn.restype = ctypes.c_int
    return fn


def _prepare_projection_layout(config: dict[str, Any], model_root: Path, max_dim: int) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    model_name = str(config["model"]["name"])
    cache_path = _cached_forward_path(config, model_name)
    if not cache_path.exists():
        raise FileNotFoundError(f"Existing token-cloud forward cache is required for metadata layout: {cache_path}")
    cloud_df, _ = _load_cached_cloud_frame(cache_path)
    dataset_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    subclass_lookup = pd.read_parquet(dataset_path).loc[:, ["example_id", "subclass"]].drop_duplicates()
    meta = cloud_df.loc[
        :,
        ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "layer", "token_count"],
    ].copy()
    meta["example_id"] = meta["example_id"].astype(str)
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta["dataset"] = meta["dataset"].astype(str)
    meta["split"] = meta["split"].astype(str)
    meta["label_ambiguous"] = meta["label_ambiguous"].astype(int)
    meta["layer"] = meta["layer"].astype(int)
    meta["token_count"] = meta["token_count"].astype(int)
    meta = meta.merge(subclass_lookup.assign(example_id=lambda df: df["example_id"].astype(str)), on="example_id", how="left")
    layers = sorted(int(layer) for layer in meta["layer"].unique())
    example_columns = ["example_id", "pair_id", "dataset", "split", "label_ambiguous", "subclass"]
    examples = meta.loc[:, example_columns].drop_duplicates("example_id").reset_index(drop=True)
    examples["example_index"] = np.arange(len(examples), dtype=np.int32)
    meta = meta.merge(examples.loc[:, ["example_id", "example_index"]], on="example_id", how="left")

    ensure_dir(model_root)
    examples.to_parquet(model_root / "examples.parquet", index=False)
    layer_rows = []
    for layer in layers:
        layer_meta = meta.loc[meta["layer"].eq(layer)].copy().reset_index(drop=True)
        layer_meta["layer_row"] = np.arange(len(layer_meta), dtype=np.int32)
        offsets = np.zeros(len(layer_meta), dtype=np.int64)
        if len(layer_meta) > 1:
            offsets[1:] = np.cumsum(layer_meta["token_count"].to_numpy(dtype=np.int64)[:-1])
        layer_meta["token_offset"] = offsets
        layer_meta.to_parquet(model_root / f"layer_{layer:02d}_meta.parquet", index=False)
        token_total = int(layer_meta["token_count"].sum())
        layer_rows.append({"layer": int(layer), "rows": int(len(layer_meta)), "tokens": token_total})
        token_path = model_root / f"layer_{layer:02d}_projected_tokens_float32.dat"
        if not token_path.exists():
            fp = np.memmap(token_path, mode="w+", dtype=np.float32, shape=(token_total, int(max_dim)))
            fp.flush()
            del fp
    pd.DataFrame(layer_rows).to_csv(model_root / "layer_token_layout.csv", index=False)
    write_json(
        model_root / "projection_layout.metadata.json",
        {
            "model_name": model_name,
            "max_dim": int(max_dim),
            "token_cloud_forward_cache": str(cache_path),
            "examples": int(len(examples)),
            "layers": layers,
            "layer_rows": layer_rows,
        },
    )
    return examples, meta, layers


def fit_reducers_if_needed(
    *,
    config: dict[str, Any],
    output_dir: Path,
    source_sweep_dir: Path,
    max_dim: int,
    pca_fit_token_cap: int | None,
) -> Path:
    model_name = str(config["model"]["name"])
    model_root = _model_root(output_dir, model_name, max_dim)
    reducers_path = _copy_or_link_reducers(config, source_sweep_dir, model_root, max_dim)
    if reducers_path.exists():
        return reducers_path
    fit_config = config
    if pca_fit_token_cap is not None:
        fit_config = dict(config)
        fit_config["token_cloud_topology_classifier"] = dict(config["token_cloud_topology_classifier"])
        fit_config["token_cloud_topology_classifier"]["pca_fit_token_cap"] = int(pca_fit_token_cap)
    _fit_or_load_reducers(
        config=fit_config,
        output_dir=output_dir,
        model_name=model_name,
        max_dim=max_dim,
        force_recompute=False,
    )
    produced = _reducers_path(output_dir, model_name, max_dim)
    if not produced.exists():
        raise FileNotFoundError(f"Reducer fitting did not produce expected cache: {produced}")
    try:
        reducers_path.symlink_to(produced.resolve())
    except FileExistsError:
        pass
    except OSError:
        joblib.dump(joblib.load(produced), reducers_path)
    produced_meta = produced.with_suffix(".metadata.json")
    if produced_meta.exists():
        dest_meta = reducers_path.with_suffix(".metadata.json")
        if not dest_meta.exists():
            try:
                dest_meta.symlink_to(produced_meta.resolve())
            except OSError:
                dest_meta.write_text(produced_meta.read_text(encoding="utf-8"), encoding="utf-8")
    return reducers_path


def project_model(
    *,
    config: dict[str, Any],
    output_dir: Path,
    source_sweep_dir: Path,
    max_dim: int,
    batch_size_override: int | None,
    pca_fit_token_cap: int | None,
    force: bool,
) -> Path:
    model_name = str(config["model"]["name"])
    model_root = _model_root(output_dir, model_name, max_dim)
    done_path = model_root / "projection.done.json"
    if done_path.exists() and not force:
        return model_root

    reducers_path = fit_reducers_if_needed(
        config=config,
        output_dir=output_dir,
        source_sweep_dir=source_sweep_dir,
        max_dim=max_dim,
        pca_fit_token_cap=pca_fit_token_cap,
    )
    reducers = joblib.load(reducers_path)
    examples, _, layers = _prepare_projection_layout(config, model_root, max_dim)

    classifier_config = dict(config["token_cloud_topology_classifier"])
    subclass_cfg = dict(config["clamber_subclass_classification"])
    bundle = load_hf_model(config["model"], classifier_config)
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
    prepared_df["example_id"] = prepared_df["example_id"].astype(str)

    batch_size = int(batch_size_override or subclass_cfg.get("token_cloud_batch_size", classifier_config.get("batch_size", 8)))
    max_length = int(subclass_cfg.get("token_cloud_max_length", classifier_config.get("max_length", 64)))
    drop_special_tokens = bool(classifier_config.get("drop_special_tokens", True))
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)

    layer_meta = {layer: pd.read_parquet(model_root / f"layer_{layer:02d}_meta.parquet") for layer in layers}
    layer_lookup = {
        layer: {
            str(row.example_id): (int(row.token_offset), int(row.token_count))
            for row in meta.itertuples(index=False)
        }
        for layer, meta in layer_meta.items()
    }
    token_maps = {}
    for layer, meta in layer_meta.items():
        token_total = int(meta["token_count"].sum())
        token_maps[layer] = np.memmap(
            model_root / f"layer_{layer:02d}_projected_tokens_float32.dat",
            mode="r+",
            dtype=np.float32,
            shape=(token_total, int(max_dim)),
        )

    progress = tqdm(total=len(prepared_df), desc=f"{slugify(model_name)}_project_pca{max_dim}", unit="example")
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
                raise RuntimeError("Model did not return hidden states for dense PCA projection.")
            input_ids_cpu = input_ids.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu()
            records = batch_df.reset_index(drop=True).to_dict(orient="records")
            valid_masks = [
                _valid_token_mask(
                    input_ids_cpu[row_index],
                    attention_mask_cpu[row_index],
                    special_ids=special_ids,
                    drop_special_tokens=drop_special_tokens,
                )
                for row_index in range(len(records))
            ]
            for layer in layers:
                reducer = reducers[int(layer)]
                layer_output = hidden_states[layer + 1].detach().float().cpu()
                token_chunks: list[np.ndarray] = []
                chunk_meta: list[tuple[str, int]] = []
                for row_index, row in enumerate(records):
                    example_id = str(row["example_id"])
                    token_vectors = layer_output[row_index][valid_masks[row_index]].numpy()
                    if token_vectors.size == 0:
                        continue
                    expected_offset, expected_count = layer_lookup[layer][example_id]
                    if int(token_vectors.shape[0]) != expected_count:
                        raise ValueError(
                            f"Token count mismatch for {model_name} layer {layer} example {example_id}: "
                            f"projected={token_vectors.shape[0]} layout={expected_count}"
                        )
                    token_chunks.append(token_vectors)
                    chunk_meta.append((example_id, expected_count))
                if not token_chunks:
                    continue
                token_matrix = np.vstack(token_chunks)
                reduced_matrix = reducer.transform(token_matrix).astype(np.float32, copy=False)
                if reduced_matrix.shape[1] < max_dim:
                    raise ValueError(f"Reducer {reducers_path} returned {reduced_matrix.shape[1]} dims, expected {max_dim}")
                reduced_matrix = reduced_matrix[:, :max_dim]
                offset = 0
                for example_id, token_count in chunk_meta:
                    target_offset, _ = layer_lookup[layer][example_id]
                    token_maps[layer][target_offset : target_offset + token_count, :] = reduced_matrix[offset : offset + token_count]
                    offset += token_count
            del outputs, hidden_states, model_inputs, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update(len(records))
    finally:
        progress.close()
        for token_map in token_maps.values():
            token_map.flush()
            del token_map

    write_json(
        done_path,
        {
            "model_name": model_name,
            "max_dim": int(max_dim),
            "reducers_path": str(reducers_path),
            "examples": int(len(examples)),
            "layers": layers,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
        },
    )
    return model_root


def _compute_h0_layer_worker(
    *,
    library_path: str,
    model_root: str,
    layer: int,
    n_examples: int,
    n_layers: int,
    max_dim: int,
    force: bool,
) -> dict[str, Any]:
    root = Path(model_root)
    layer_done = root / f"layer_{layer:02d}_h0.done.json"
    if layer_done.exists() and not force:
        return {"layer": int(layer), "status": "reused"}
    fn = _load_h0_function(library_path)
    meta = pd.read_parquet(root / f"layer_{layer:02d}_meta.parquet")
    token_total = int(meta["token_count"].sum())
    tokens = np.memmap(
        root / f"layer_{layer:02d}_projected_tokens_float32.dat",
        mode="r",
        dtype=np.float32,
        shape=(token_total, int(max_dim)),
    )
    tensor = np.memmap(
        root / "h0_tensor_float32.dat",
        mode="r+",
        dtype=np.float32,
        shape=(int(n_examples), int(n_layers), int(max_dim)),
    )
    out = np.empty(int(max_dim), dtype=np.float32)
    layer_index = int(layer)
    for row in meta.itertuples(index=False):
        start = int(row.token_offset)
        count = int(row.token_count)
        cloud = np.asarray(tokens[start : start + count, :], dtype=np.float32, order="C")
        code = fn(
            cloud.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(count),
            ctypes.c_int(max_dim),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if code != 0:
            raise RuntimeError(f"h0_prefix_means failed with code {code} for layer {layer}")
        tensor[int(row.example_index), layer_index, :] = out
    tensor.flush()
    write_json(layer_done, {"layer": int(layer), "rows": int(len(meta)), "max_dim": int(max_dim)})
    return {"layer": int(layer), "status": "computed", "rows": int(len(meta))}


def compute_h0_model(*, output_dir: Path, model_name: str, max_dim: int, library_path: Path, workers: int, force: bool) -> Path:
    model_root = _model_root(output_dir, model_name, max_dim)
    done_path = model_root / "h0.done.json"
    if done_path.exists() and not force:
        return model_root
    projection_meta = _read_json(model_root / "projection_layout.metadata.json")
    layers = [int(layer) for layer in projection_meta["layers"]]
    examples = pd.read_parquet(model_root / "examples.parquet")
    n_examples = int(len(examples))
    n_layers = max(layers) + 1
    tensor_path = model_root / "h0_tensor_float32.dat"
    if not tensor_path.exists() or force:
        tensor = np.memmap(tensor_path, mode="w+", dtype=np.float32, shape=(n_examples, n_layers, int(max_dim)))
        tensor[:] = np.nan
        tensor.flush()
        del tensor

    results = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [
            pool.submit(
                _compute_h0_layer_worker,
                library_path=str(library_path),
                model_root=str(model_root),
                layer=int(layer),
                n_examples=n_examples,
                n_layers=n_layers,
                max_dim=int(max_dim),
                force=bool(force),
            )
            for layer in layers
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{slugify(model_name)}_h0_layers", unit="layer"):
            results.append(future.result())
    write_json(done_path, {"model_name": model_name, "max_dim": int(max_dim), "layers": results})
    return model_root


def _evaluate_one_dim(
    *,
    tensor_path: str,
    examples_path: str,
    n_examples: int,
    n_layers: int,
    max_dim: int,
    dim: int,
    model_name: str,
    seed: int,
    max_iter: int,
    c_value: float,
) -> dict[str, Any]:
    examples = pd.read_parquet(examples_path)
    tensor = np.memmap(tensor_path, mode="r", dtype=np.float32, shape=(n_examples, n_layers, max_dim))
    x = np.asarray(tensor[:, :, int(dim) - 1], dtype=float)
    keep_layers = ~np.isnan(x).all(axis=0)
    x = x[:, keep_layers]
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    train_mask = examples["split"].astype(str).eq("train").to_numpy()
    test_mask = examples["split"].astype(str).eq("test").to_numpy()
    payload = _evaluate_multiclass(
        x_train=x[train_mask],
        y_train=examples.loc[train_mask, "subclass"].astype(str).to_numpy(),
        x_eval=x[test_mask],
        y_eval=examples.loc[test_mask, "subclass"].astype(str).to_numpy(),
        max_iter=int(max_iter),
        c_value=float(c_value),
        seed=int(seed) + int(dim),
    )
    metrics = payload["metrics"]
    return {
        "model": slugify(model_name),
        "model_name": model_name,
        "dataset": "clamber",
        "label_space": "9_subclasses",
        "method": "h0_mean_persistence_pca_dense_prefix_all_layers",
        "pca_dim": int(dim),
        "layer": "all",
        "feature_count": int(x.shape[1]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def evaluate_model(
    *,
    config: dict[str, Any],
    output_dir: Path,
    max_dim: int,
    workers: int,
    seed: int,
    force: bool,
) -> Path:
    model_name = str(config["model"]["name"])
    model_root = _model_root(output_dir, model_name, max_dim)
    metrics_path = model_root / "dense_prefix_metrics.csv"
    if metrics_path.exists() and not force:
        return metrics_path
    examples_path = model_root / "examples.parquet"
    examples = pd.read_parquet(examples_path)
    projection_meta = _read_json(model_root / "projection_layout.metadata.json")
    layers = [int(layer) for layer in projection_meta["layers"]]
    n_layers = max(layers) + 1
    subclass_cfg = dict(config["clamber_subclass_classification"])
    max_iter = int(subclass_cfg.get("max_iter", 4000))
    c_value = float(subclass_cfg.get("token_cloud_classifier_C", 1.0))
    tensor_path = model_root / "h0_tensor_float32.dat"

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [
            pool.submit(
                _evaluate_one_dim,
                tensor_path=str(tensor_path),
                examples_path=str(examples_path),
                n_examples=int(len(examples)),
                n_layers=int(n_layers),
                max_dim=int(max_dim),
                dim=int(dim),
                model_name=model_name,
                seed=int(seed),
                max_iter=max_iter,
                c_value=c_value,
            )
            for dim in range(1, int(max_dim) + 1)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{slugify(model_name)}_evaluate_dims", unit="dim"):
            rows.append(future.result())
    metrics_df = pd.DataFrame(rows).sort_values("pca_dim").reset_index(drop=True)
    metrics_df.to_csv(metrics_path, index=False)
    best_path = model_root / "dense_prefix_best_metrics.csv"
    best_rows = []
    for key in ["accuracy", "macro_f1"]:
        best = metrics_df.sort_values([key, "macro_f1" if key == "accuracy" else "accuracy"], ascending=False).iloc[0].to_dict()
        best["best_by"] = key
        best_rows.append(best)
    pd.DataFrame(best_rows).to_csv(best_path, index=False)
    return metrics_path


def _write_existing_combined_metrics(output_dir: Path) -> None:
    """Rebuild combined CSVs from all completed model runs.

    This keeps parallel one-model-per-GPU launches from leaving a combined file
    that only reflects whichever process finished last.
    """
    metric_paths = sorted(output_dir.glob("*/pca_prefix_*/dense_prefix_metrics.csv"))
    if metric_paths:
        combined = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
        combined_path = output_dir / "dense_prefix_metrics_combined_all_models.csv"
        tmp_path = output_dir / f".{combined_path.name}.tmp.{os.getpid()}"
        combined.to_csv(tmp_path, index=False)
        os.replace(tmp_path, combined_path)
        print(f"Wrote {combined_path}")

    best_paths = sorted(output_dir.glob("*/pca_prefix_*/dense_prefix_best_metrics.csv"))
    if best_paths:
        combined_best = pd.concat([pd.read_csv(path) for path in best_paths], ignore_index=True)
        combined_best_path = output_dir / "dense_prefix_best_metrics_all_models.csv"
        tmp_best_path = output_dir / f".{combined_best_path.name}.tmp.{os.getpid()}"
        combined_best.to_csv(tmp_best_path, index=False)
        os.replace(tmp_best_path, combined_best_path)
        print(f"Wrote {combined_best_path}")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = ensure_dir(Path(args.output_dir))
    source_sweep_dir = Path(args.source_sweep_dir)
    library_path = _compile_h0_library(output_dir)
    if args.compile_only:
        print(f"Compiled {library_path}")
        return

    metric_paths = []
    for config_path in args.configs:
        config = load_config(config_path)
        model_name = str(config["model"]["name"])
        max_dim = _resolve_max_dim(config, source_sweep_dir, output_dir, str(args.max_dim))
        model_root = _model_root(output_dir, model_name, max_dim)
        ensure_dir(model_root)
        write_json(
            model_root / "run_request.json",
            {
                "config_path": str(config_path),
                "model_name": model_name,
                "max_dim": int(max_dim),
                "stage": str(args.stage),
                "workers": int(args.workers),
            },
        )
        if args.stage in {"fit", "all"}:
            fit_reducers_if_needed(
                config=config,
                output_dir=output_dir,
                source_sweep_dir=source_sweep_dir,
                max_dim=max_dim,
                pca_fit_token_cap=args.pca_fit_token_cap,
            )
        if args.stage in {"project", "all"}:
            project_model(
                config=config,
                output_dir=output_dir,
                source_sweep_dir=source_sweep_dir,
                max_dim=max_dim,
                batch_size_override=args.batch_size,
                pca_fit_token_cap=args.pca_fit_token_cap,
                force=bool(args.force_project),
            )
        if args.stage in {"h0", "all"}:
            compute_h0_model(
                output_dir=output_dir,
                model_name=model_name,
                max_dim=max_dim,
                library_path=library_path,
                workers=int(args.workers),
                force=bool(args.force_h0),
            )
        if args.stage in {"evaluate", "all"}:
            metric_paths.append(
                evaluate_model(
                    config=config,
                    output_dir=output_dir,
                    max_dim=max_dim,
                    workers=int(args.workers),
                    seed=int(args.seed),
                    force=bool(args.force_evaluate),
                )
            )

    if metric_paths:
        _write_existing_combined_metrics(output_dir)


if __name__ == "__main__":
    main()
