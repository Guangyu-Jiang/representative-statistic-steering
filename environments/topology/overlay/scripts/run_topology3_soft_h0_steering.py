"""Differentiable soft-H0 topology steering for ambiguity abstention.

This script reuses the cached topology3 clouds/features produced by
run_topology3_surrogate_steering.py, but replaces the learned DeepSets
surrogate with a differentiable graph objective for H0 persistence:

- soft_mst: a soft minimum-spanning-tree approximation using Gibbs spanning
  tree edge marginals.
- soft_nn: a cheaper soft nearest-neighbor approximation.
- hard_mst: exact H0 MST edges are recomputed from the current cloud, and the
  selected edge lengths remain differentiable with respect to token positions.

The proxy is linearly calibrated to exact PH features on natural last-layer
clouds before optimization. This keeps the optimization target in the same
units as the exact H0 features.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_topology3_surrogate_steering as topo3  # noqa: E402

from aen_replication.config import load_config  # noqa: E402
from aen_replication.models.hf_model import load_hf_model  # noqa: E402
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet  # noqa: E402
from aen_replication.utils.seed import set_global_seed  # noqa: E402


FEATURES = topo3.FEATURES
MODEL_SUBDIR = "meta_llama_llama_3_1_8b_instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runs/llama_steering_paper_style_openai_ibm.yaml")
    parser.add_argument("--datasets", nargs="+", default=["ambigqa", "situatedqa"])
    parser.add_argument("--source-root", default="artifacts/steering_paper_style_openai_ibm")
    parser.add_argument("--feature-root", default="artifacts/steering_topology3_surrogate")
    parser.add_argument("--artifact-root", default="artifacts/steering_topology3_soft_h0")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Layer whose PCA token cloud is optimized and whose hidden state is steered. Defaults to final layer.",
    )
    parser.add_argument("--pca-fit-token-cap", type=int, default=32000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=8)
    parser.add_argument("--proxy", choices=["soft_mst", "soft_nn", "hard_mst"], default="soft_mst")
    parser.add_argument(
        "--tau-scale",
        type=float,
        default=0.5,
        help="Softness as a multiplier of each cloud's median nonzero pairwise distance.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mst-recompute-every",
        type=int,
        default=1,
        help="For hard_mst, recompute exact MST edges every N optimization steps.",
    )
    parser.add_argument("--calibration-n", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument("--opt-steps", type=int, default=80)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-batch-size", type=int, default=8)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--eval-n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit after sampling D- eval rows.")
    parser.add_argument(
        "--behavior-limit-per-label",
        type=int,
        default=None,
        help="Optional smoke-test cap per D+/D- behavior label before feature extraction.",
    )
    parser.add_argument(
        "--apply-on",
        choices=["prefill_only", "prompt_and_decode_mean"],
        default="prompt_and_decode_mean",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-neighbors", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slug_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _run_slug(proxy: str, tau_scale: float, alpha: float, lambda_value: float) -> str:
    return (
        f"proxy_{proxy}__tau_{_slug_float(tau_scale)}__"
        f"alpha_{_slug_float(alpha)}__lambda_{_slug_float(lambda_value)}"
    )


def _load_features(
    *,
    config: dict[str, Any],
    dataset: str,
    behavior: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Any], list[int]]:
    model_slug = slugify(config["model"]["name"])
    feature_root = Path(args.feature_root).resolve()
    output_root = feature_root / dataset / model_slug
    cloud_path = output_root / f"{dataset}__topology3_clouds.joblib"
    feature_path = output_root / f"{dataset}__topology3_features.parquet"
    reducer_path = output_root / f"{dataset}__topology3_reducers.joblib"

    if cloud_path.exists() and feature_path.exists() and reducer_path.exists() and not args.force_features:
        payload = joblib.load(cloud_path)
        cloud_df = pd.DataFrame(payload["cloud_df"])
        feature_df = pd.read_parquet(feature_path)
        reducers = joblib.load(reducer_path)
        layers = sorted(int(layer) for layer in reducers.keys())
        return cloud_df, feature_df, reducers, layers

    return topo3._build_or_load_features(
        config=config,
        dataset=dataset,
        behavior=behavior,
        artifact_root=feature_root,
        args=args,
        seed=seed,
    )


def _feature_std(feature_df: pd.DataFrame, *, layer: int) -> np.ndarray:
    values = (
        feature_df.loc[feature_df["layer"].eq(layer), list(FEATURES)]
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    std = values.std(axis=0)
    return np.where(std > 1e-6, std, 1.0).astype(np.float32)


def _entropy_from_values(values: torch.Tensor, *, normalizer: int) -> torch.Tensor:
    total = values.sum().clamp_min(1e-12)
    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
    denom = float(np.log(max(int(normalizer), 2)))
    return entropy / max(denom, 1e-12)


def _features_from_soft_lifetimes(values: torch.Tensor, *, selected_edge_count: int, top_k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(3)
    values = values.clamp_min(0.0)
    total = values.sum().clamp_min(1e-12)
    mean_persistence = total / max(int(selected_edge_count), 1)
    entropy = _entropy_from_values(values, normalizer=max(int(selected_edge_count), 2))
    k = min(max(int(top_k), 1), int(values.numel()))
    top_fraction = torch.topk(values, k=k).values.sum() / total
    return torch.stack([mean_persistence, entropy, top_fraction])


def _cloud_tau(distances: torch.Tensor, *, tau_scale: float) -> torch.Tensor:
    n = int(distances.shape[0])
    if n <= 1:
        return distances.new_tensor(1.0)
    indices = torch.triu_indices(n, n, offset=1, device=distances.device)
    nonzero = distances[indices[0], indices[1]]
    nonzero = nonzero[nonzero > 1e-8]
    if nonzero.numel() == 0:
        return distances.new_tensor(1.0)
    return (nonzero.detach().median() * float(tau_scale)).clamp_min(1e-6)


def _soft_nn_features_one(cloud: torch.Tensor, *, tau_scale: float, top_k: int) -> torch.Tensor:
    n = int(cloud.shape[0])
    if n <= 1:
        return cloud.new_zeros(3)
    distances = torch.cdist(cloud.unsqueeze(0), cloud.unsqueeze(0), p=2).squeeze(0)
    tau = _cloud_tau(distances, tau_scale=tau_scale)
    eye = torch.eye(n, device=cloud.device, dtype=torch.bool)
    scores = -distances / tau
    scores = scores.masked_fill(eye, -1e9)
    probabilities = torch.softmax(scores, dim=1)
    soft_nn_lifetimes = (probabilities * distances).sum(dim=1)
    return _features_from_soft_lifetimes(soft_nn_lifetimes, selected_edge_count=n - 1, top_k=top_k)


def _soft_mst_features_one(cloud: torch.Tensor, *, tau_scale: float, top_k: int) -> torch.Tensor:
    n = int(cloud.shape[0])
    if n <= 1:
        return cloud.new_zeros(3)
    distances = torch.cdist(cloud.unsqueeze(0), cloud.unsqueeze(0), p=2).squeeze(0)
    tau = _cloud_tau(distances, tau_scale=tau_scale)
    eye = torch.eye(n, device=cloud.device, dtype=cloud.dtype)
    weights = torch.exp(-distances / tau) * (1.0 - eye)
    degree = weights.sum(dim=1)
    laplacian = torch.diag(degree) - weights
    pinv = torch.linalg.pinv(laplacian, hermitian=True)
    diag = torch.diag(pinv)
    resistance = (diag[:, None] + diag[None, :] - 2.0 * pinv).clamp_min(0.0)
    edge_marginals = (weights * resistance).clamp_min(0.0)
    indices = torch.triu_indices(n, n, offset=1, device=cloud.device)
    contributions = edge_marginals[indices[0], indices[1]] * distances[indices[0], indices[1]]
    return _features_from_soft_lifetimes(contributions, selected_edge_count=n - 1, top_k=top_k)


def _mst_edges_from_cloud(cloud: torch.Tensor) -> torch.Tensor:
    n = int(cloud.shape[0])
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=cloud.device)
    distances = torch.cdist(cloud.detach().float().unsqueeze(0), cloud.detach().float().unsqueeze(0), p=2)
    matrix = distances.squeeze(0).cpu().numpy().astype(np.float64, copy=False)
    off_diag = ~np.eye(n, dtype=bool)
    matrix[off_diag] = np.maximum(matrix[off_diag], 1e-8)
    matrix[~off_diag] = 0.0
    tree = minimum_spanning_tree(matrix).tocoo()
    if tree.row.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=cloud.device)
    edge_index = np.vstack([tree.row, tree.col]).astype(np.int64, copy=False)
    return torch.as_tensor(edge_index, device=cloud.device, dtype=torch.long)


def _hard_mst_edges_for_batch(clouds: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    edges: list[torch.Tensor] = []
    for row_index in range(int(clouds.shape[0])):
        valid_cloud = clouds[row_index][mask[row_index]]
        edges.append(_mst_edges_from_cloud(valid_cloud))
    return edges


def _hard_mst_features_one(cloud: torch.Tensor, edge_index: torch.Tensor, *, top_k: int) -> torch.Tensor:
    n = int(cloud.shape[0])
    if n <= 1 or edge_index.numel() == 0:
        return cloud.new_zeros(3)
    lifetimes = torch.linalg.norm(cloud[edge_index[0]] - cloud[edge_index[1]], dim=1)
    return _features_from_soft_lifetimes(lifetimes, selected_edge_count=n - 1, top_k=top_k)


def _soft_h0_features_batch(
    clouds: torch.Tensor,
    mask: torch.Tensor,
    *,
    proxy: str,
    tau_scale: float,
    top_k: int,
    mst_edges: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row_index in range(int(clouds.shape[0])):
        valid_cloud = clouds[row_index][mask[row_index]]
        if proxy == "soft_mst":
            rows.append(_soft_mst_features_one(valid_cloud, tau_scale=tau_scale, top_k=top_k))
        elif proxy == "soft_nn":
            rows.append(_soft_nn_features_one(valid_cloud, tau_scale=tau_scale, top_k=top_k))
        elif proxy == "hard_mst":
            if mst_edges is None:
                edge_index = _mst_edges_from_cloud(valid_cloud)
            else:
                edge_index = mst_edges[row_index]
            rows.append(_hard_mst_features_one(valid_cloud, edge_index, top_k=top_k))
        else:
            raise ValueError(f"Unknown proxy: {proxy}")
    return torch.stack(rows, dim=0)


def _proxy_features_numpy(
    clouds: list[np.ndarray],
    *,
    proxy: str,
    tau_scale: float,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(clouds), batch_size), desc=f"{proxy}_calibration", leave=False):
            batch_clouds = clouds[start : start + batch_size]
            padded, mask, _counts = topo3._pad_clouds(batch_clouds)
            padded = padded.to(device)
            mask = mask.to(device)
            mst_edges = _hard_mst_edges_for_batch(padded, mask) if proxy == "hard_mst" else None
            features = _soft_h0_features_batch(
                padded,
                mask,
                proxy=proxy,
                tau_scale=tau_scale,
                top_k=top_k,
                mst_edges=mst_edges,
            )
            outputs.append(features.detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32, copy=False)


def _fit_proxy_calibration(
    *,
    cloud_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    layer: int,
    proxy: str,
    tau_scale: float,
    top_k: int,
    calibration_n: int,
    calibration_batch_size: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    layer_cloud = (
        cloud_df.loc[cloud_df["layer"].eq(layer), ["example_id", "cloud", "token_count", "behavior_label"]]
        .drop_duplicates("example_id")
        .reset_index(drop=True)
    )
    layer_features = (
        feature_df.loc[feature_df["layer"].eq(layer), ["example_id", *FEATURES]]
        .drop_duplicates("example_id")
        .reset_index(drop=True)
    )
    frame = layer_cloud.merge(layer_features, on="example_id", how="inner").reset_index(drop=True)
    if calibration_n is not None and int(calibration_n) > 0 and len(frame) > int(calibration_n):
        frame = frame.sample(n=int(calibration_n), random_state=seed).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clouds = [np.asarray(value, dtype=np.float32) for value in frame["cloud"].tolist()]
    proxy_values = _proxy_features_numpy(
        clouds,
        proxy=proxy,
        tau_scale=tau_scale,
        top_k=top_k,
        batch_size=int(calibration_batch_size),
        device=device,
    )
    exact_values = frame.loc[:, FEATURES].to_numpy(dtype=np.float32)

    slopes: list[float] = []
    intercepts: list[float] = []
    rows: list[dict[str, Any]] = []
    calibrated = np.zeros_like(exact_values)
    for feature_index, feature in enumerate(FEATURES):
        model = LinearRegression()
        x = proxy_values[:, [feature_index]]
        y = exact_values[:, feature_index]
        model.fit(x, y)
        prediction = model.predict(x)
        calibrated[:, feature_index] = prediction.astype(np.float32)
        residual = prediction - y
        sse = float(np.sum(np.square(residual)))
        sst = float(np.sum(np.square(y - y.mean())))
        r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
        slopes.append(float(model.coef_[0]))
        intercepts.append(float(model.intercept_))
        rows.append(
            {
                "feature": feature,
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "r2": float(r2),
                "mae": float(np.mean(np.abs(residual))),
                "proxy_mean": float(proxy_values[:, feature_index].mean()),
                "proxy_std": float(proxy_values[:, feature_index].std()),
                "exact_mean": float(y.mean()),
                "exact_std": float(y.std()),
            }
        )

    calibration_df = pd.DataFrame(rows)
    calibration_path = output_root / f"layer_{layer:02d}__{proxy}_calibration.csv"
    calibration_df.to_csv(calibration_path, index=False)
    calibration_df.to_parquet(calibration_path.with_suffix(".parquet"), index=False)
    return {
        "proxy": proxy,
        "tau_scale": float(tau_scale),
        "top_k": int(top_k),
        "layer": int(layer),
        "n": int(len(frame)),
        "slopes": slopes,
        "intercepts": intercepts,
        "rows": rows,
    }


def _optimize_soft_h0_deltas(
    *,
    current_clouds: list[np.ndarray],
    target_features: np.ndarray,
    target_std: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    proxy: str,
    tau_scale: float,
    top_k: int,
    mst_recompute_every: int,
    lambda_value: float,
    steps: int,
    lr: float,
    batch_size: int,
    parallel_jobs: int,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_std_t = torch.as_tensor(target_std, device=device, dtype=torch.float32)
    slopes_t = torch.as_tensor(slopes, device=device, dtype=torch.float32)
    intercepts_t = torch.as_tensor(intercepts, device=device, dtype=torch.float32)
    deltas: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(current_clouds), batch_size), desc=f"{proxy}_opt", leave=False):
        batch_clouds = current_clouds[start : start + batch_size]
        batch_targets = target_features[start : start + batch_size]
        clouds, mask, _counts = topo3._pad_clouds(batch_clouds)
        clouds = clouds.to(device)
        mask = mask.to(device)
        target = torch.as_tensor(batch_targets, device=device, dtype=torch.float32)
        raw_delta = torch.zeros_like(clouds, requires_grad=True)
        optimizer = torch.optim.Adam([raw_delta], lr=float(lr))
        mask_f = mask.unsqueeze(-1).float()
        denom = clouds.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)
        last_topo_loss = 0.0
        last_reg = 0.0
        mst_edges: list[torch.Tensor] | None = None

        for step in range(int(steps)):
            delta = raw_delta * mask_f
            delta_mean = delta.sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            delta = (delta - delta_mean) * mask_f
            if proxy == "hard_mst" and (
                mst_edges is None or step % max(int(mst_recompute_every), 1) == 0
            ):
                mst_edges = _hard_mst_edges_for_batch(clouds + delta, mask)
            proxy_raw = _soft_h0_features_batch(
                clouds + delta,
                mask,
                proxy=proxy,
                tau_scale=tau_scale,
                top_k=top_k,
                mst_edges=mst_edges,
            )
            proxy_calibrated = proxy_raw * slopes_t + intercepts_t
            topo_loss_per = torch.mean(((proxy_calibrated - target) / target_std_t).pow(2), dim=1)
            reg_per = delta.pow(2).sum(dim=(1, 2)) / denom
            loss = torch.mean(topo_loss_per + float(lambda_value) * reg_per)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_delta], 1.0)
            optimizer.step()
            last_topo_loss = float(topo_loss_per.mean().detach().cpu())
            last_reg = float(reg_per.mean().detach().cpu())

        with torch.no_grad():
            delta = raw_delta * mask_f
            delta_mean = delta.sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            delta = (delta - delta_mean) * mask_f
            proxy_raw = _soft_h0_features_batch(
                clouds + delta,
                mask,
                proxy=proxy,
                tau_scale=tau_scale,
                top_k=top_k,
                mst_edges=_hard_mst_edges_for_batch(clouds + delta, mask) if proxy == "hard_mst" else None,
            )
            proxy_calibrated = proxy_raw * slopes_t + intercepts_t
            delta_cpu = delta.detach().cpu().numpy()
            pred_cpu = proxy_calibrated.detach().cpu().numpy()
            clouds_cpu = clouds.detach().cpu().numpy()
            mask_cpu = mask.detach().cpu().numpy()

        steered_clouds: list[np.ndarray] = []
        batch_rows: list[dict[str, Any]] = []
        for local_index in range(len(batch_clouds)):
            token_count = int(mask_cpu[local_index].sum())
            delta_valid = delta_cpu[local_index, :token_count].astype(np.float32, copy=False)
            current_valid = clouds_cpu[local_index, :token_count].astype(np.float32, copy=False)
            deltas.append(delta_valid)
            steered_clouds.append(current_valid + delta_valid)
            delta_norm = float(np.linalg.norm(delta_valid))
            cloud_norm = float(np.linalg.norm(current_valid))
            target_error = float(np.linalg.norm(pred_cpu[local_index] - batch_targets[local_index]))
            batch_rows.append(
                {
                    "row_index": int(start + local_index),
                    "soft_proxy_target_l2_error": target_error,
                    "soft_proxy_topology_loss": last_topo_loss,
                    "pca_delta_norm": delta_norm,
                    "pca_cloud_norm": cloud_norm,
                    "relative_pca_delta_norm": float(delta_norm / max(cloud_norm, 1e-12)),
                    "regularization_value": last_reg,
                    **{
                        f"soft_proxy_pred__{feature}": float(pred_cpu[local_index, feature_index])
                        for feature_index, feature in enumerate(FEATURES)
                    },
                }
            )

        exact = topo3._exact_features_for_clouds(steered_clouds, parallel_jobs=parallel_jobs)
        for local_index, row in enumerate(batch_rows):
            row["exact_target_l2_error"] = float(np.linalg.norm(exact[local_index] - batch_targets[local_index]))
            for feature_index, feature in enumerate(FEATURES):
                row[f"exact_steered__{feature}"] = float(exact[local_index, feature_index])
            rows.append(row)

    return deltas, pd.DataFrame(rows)


def _run_dataset(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> None:
    seed = int(config["seed"])
    source_root = Path(args.source_root).resolve()
    artifact_root = ensure_dir(Path(args.artifact_root).resolve())
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(artifact_root / dataset / model_slug)

    behavior = topo3._load_behavior(config, dataset, source_root)
    if args.behavior_limit_per_label is not None:
        capped_parts: list[pd.DataFrame] = []
        for label_value, label_df in behavior.groupby("behavior_label", sort=True):
            n_rows = min(int(args.behavior_limit_per_label), len(label_df))
            capped_parts.append(label_df.sample(n=n_rows, random_state=seed + int(label_value)).reset_index(drop=True))
        behavior = pd.concat(capped_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(
        f"[{dataset}] behavior rows={len(behavior)} D+={int(behavior['behavior_label'].sum())} "
        f"D-={int((1 - behavior['behavior_label']).sum())}",
        flush=True,
    )

    cloud_df, feature_df, reducers, layers = _load_features(
        config=config,
        dataset=dataset,
        behavior=behavior,
        args=args,
        seed=seed,
    )
    last_layer = int(layers[-1])
    steering_layer = int(last_layer if args.steering_layer is None else args.steering_layer)
    if steering_layer not in set(int(layer) for layer in layers):
        raise ValueError(f"Requested --steering-layer {steering_layer} is unavailable; layers={layers}")
    wide_df, all_columns, _last_layer_columns = topo3._wide_features(feature_df, layers=layers)

    eval_n = int(args.eval_n or config["steering"].get("eval_direct_answer_n", 500))
    neighbor_path = output_root / f"{dataset}__topology3_soft_h0_neighbors.parquet"
    if neighbor_path.exists() and not args.force_neighbors and args.limit is None:
        neighbor_df = pd.read_parquet(neighbor_path)
    else:
        neighbor_df = topo3._nearest_targets(
            wide_df=wide_df,
            behavior=behavior,
            all_columns=all_columns,
            feature_df=feature_df,
            layer=steering_layer,
            eval_n=eval_n,
            seed=seed,
            limit=args.limit,
        )
        write_parquet(neighbor_df, neighbor_path)
        neighbor_df.to_csv(neighbor_path.with_suffix(".csv"), index=False)

    cloud_by_id = (
        cloud_df.loc[cloud_df["layer"].eq(steering_layer), ["example_id", "cloud", "hidden_fro_norm"]]
        .drop_duplicates("example_id")
        .set_index("example_id")
    )
    current_clouds = [
        np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32)
        for example_id in neighbor_df["example_id"].astype(str)
    ]
    current_features = neighbor_df.loc[:, [f"current__{feature}" for feature in FEATURES]].to_numpy(dtype=np.float32)
    target_features = neighbor_df.loc[:, [f"target__{feature}" for feature in FEATURES]].to_numpy(dtype=np.float32)
    target_std = _feature_std(feature_df, layer=steering_layer)

    calibration = _fit_proxy_calibration(
        cloud_df=cloud_df,
        feature_df=feature_df,
        layer=steering_layer,
        proxy=str(args.proxy),
        tau_scale=float(args.tau_scale),
        top_k=int(args.top_k),
        calibration_n=int(args.calibration_n),
        calibration_batch_size=int(args.calibration_batch_size),
        seed=seed,
        output_root=output_root,
    )
    write_json(output_root / f"layer_{steering_layer:02d}__{args.proxy}_calibration.json", calibration)
    slopes = np.asarray(calibration["slopes"], dtype=np.float32)
    intercepts = np.asarray(calibration["intercepts"], dtype=np.float32)

    reducer = reducers[steering_layer]
    components = reducer.components_[: int(args.pca_components)].astype(np.float32, copy=False)

    bundle = None
    if not args.skip_generate:
        bundle = load_hf_model(config["model"], config["generation"])
    try:
        for alpha in args.alphas:
            for lambda_value in args.lambdas:
                run = _run_slug(str(args.proxy), float(args.tau_scale), float(alpha), float(lambda_value))
                run_root = ensure_dir(output_root / run)
                raw_path = run_root / f"{dataset}__topology3_soft_h0__raw.parquet"
                opt_path = run_root / f"{dataset}__topology3_soft_h0__optimization.parquet"
                if raw_path.exists() and not args.force_generate and not args.skip_generate:
                    print(f"[{dataset} {run}] raw exists: {raw_path}", flush=True)
                    continue
                if opt_path.exists() and args.skip_generate and not args.force_generate:
                    print(f"[{dataset} {run}] optimization exists: {opt_path}", flush=True)
                    continue

                z_target = current_features + float(alpha) * (target_features - current_features)
                delta_y, opt_df = _optimize_soft_h0_deltas(
                    current_clouds=current_clouds,
                    target_features=z_target,
                    target_std=target_std,
                    slopes=slopes,
                    intercepts=intercepts,
                    proxy=str(args.proxy),
                    tau_scale=float(args.tau_scale),
                    top_k=int(args.top_k),
                    mst_recompute_every=int(args.mst_recompute_every),
                    lambda_value=float(lambda_value),
                    steps=int(args.opt_steps),
                    lr=float(args.opt_lr),
                    batch_size=int(args.opt_batch_size),
                    parallel_jobs=int(args.parallel_jobs),
                )
                delta_h = [(delta @ components).astype(np.float32, copy=False) for delta in delta_y]
                delta_h_norm = np.asarray([np.linalg.norm(delta) for delta in delta_h], dtype=float)
                hidden_norm = np.asarray(
                    [
                        float(cloud_by_id.loc[example_id, "hidden_fro_norm"])
                        for example_id in neighbor_df["example_id"].astype(str)
                    ],
                    dtype=float,
                )

                run_df = neighbor_df.copy()
                run_df["prompt_text"] = topo3._prompt_texts(config, run_df["text"].astype(str).tolist())
                run_df["strategy"] = "topology3_soft_h0"
                run_df["proxy"] = str(args.proxy)
                run_df["tau_scale"] = float(args.tau_scale)
                run_df["alpha"] = float(alpha)
                run_df["lambda"] = float(lambda_value)
                run_df["layer"] = int(steering_layer)
                run_df["last_layer"] = int(last_layer)
                run_df["apply_on"] = str(args.apply_on)
                run_df["hidden_delta_norm"] = delta_h_norm
                run_df["hidden_state_norm"] = hidden_norm
                run_df["relative_hidden_delta_norm"] = delta_h_norm / np.maximum(hidden_norm, 1e-12)
                run_df = pd.concat([run_df.reset_index(drop=True), opt_df.drop(columns=["row_index"]).reset_index(drop=True)], axis=1)
                for feature_index, feature in enumerate(FEATURES):
                    run_df[f"z_target__{feature}"] = z_target[:, feature_index].astype(float)
                    run_df[f"target_delta__{feature}"] = (z_target[:, feature_index] - current_features[:, feature_index]).astype(float)

                if args.skip_generate:
                    write_parquet(run_df, opt_path)
                    run_df.to_csv(opt_path.with_suffix(".csv"), index=False)
                else:
                    assert bundle is not None
                    responses = topo3._generate_with_token_deltas(
                        bundle=bundle,
                        config=config,
                        rows_df=neighbor_df,
                        token_deltas_h=delta_h,
                        layer=steering_layer,
                        max_length=int(args.max_length),
                        apply_on=str(args.apply_on),
                    )
                    run_df["response_text"] = responses
                    write_parquet(run_df, raw_path)
                    run_df.to_csv(raw_path.with_suffix(".csv"), index=False)

                summary = {
                    "dataset": dataset,
                    "run": run,
                    "strategy": "topology3_soft_h0",
                    "proxy": str(args.proxy),
                    "tau_scale": float(args.tau_scale),
                    "mst_recompute_every": int(args.mst_recompute_every),
                    "alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "n_eval": int(len(run_df)),
                    "layer": int(steering_layer),
                    "last_layer": int(last_layer),
                    "apply_on": str(args.apply_on),
                    "skip_generate": bool(args.skip_generate),
                    "relative_hidden_delta_norm_mean": float(run_df["relative_hidden_delta_norm"].mean()),
                    "relative_hidden_delta_norm_median": float(run_df["relative_hidden_delta_norm"].median()),
                    "soft_proxy_target_l2_error_mean": float(run_df["soft_proxy_target_l2_error"].mean()),
                    "exact_target_l2_error_mean": float(run_df["exact_target_l2_error"].mean()),
                    "exact_current_to_target_l2_error_mean": float(
                        np.linalg.norm(z_target - current_features, axis=1).mean()
                    ),
                    "calibration": calibration,
                }
                write_json(run_root / f"{dataset}__topology3_soft_h0_summary.json", summary)
    finally:
        if bundle is not None:
            del bundle
        _release()

    metadata = {
        "dataset": dataset,
        "model": config["model"]["name"],
        "features": list(FEATURES),
        "layers": [int(layer) for layer in layers],
        "last_layer": int(last_layer),
        "steering_layer": int(steering_layer),
        "pca_components": int(args.pca_components),
        "proxy": str(args.proxy),
        "tau_scale": float(args.tau_scale),
        "top_k": int(args.top_k),
        "mst_recompute_every": int(args.mst_recompute_every),
        "calibration_n": int(args.calibration_n),
    }
    write_json(output_root / f"{dataset}__topology3_soft_h0_metadata.json", metadata)
    print(f"[{dataset}] topology3 soft-H0 steering complete", flush=True)


def main() -> None:
    args = _parse_args()
    set_global_seed(0)
    config = load_config(args.config)
    for dataset in args.datasets:
        _run_dataset(args, config, dataset)


if __name__ == "__main__":
    main()
