"""Exact H0 topology steering with no learned surrogate.

The method represents prompt tokens from one decoder layer as a PCA cloud,
computes three exact H0 persistence statistics from its Euclidean MST, and
iteratively solves a damped Gauss-Newton problem in PCA space.  The MST is
recomputed after every accepted update, so the optimized and verified
statistics are the same exact statistics used to define the target.

Only training rows are used to fit PCA, normalize topology features, and
select abstention targets.  Evaluation is performed on held-out test rows
whose base response was judged as a direct answer.
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
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_topology3_soft_h0_steering as soft_h0  # noqa: E402
import run_topology3_surrogate_steering as topo3  # noqa: E402

from aen_replication.config import load_config  # noqa: E402
from aen_replication.models.hf_model import load_hf_model  # noqa: E402
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet  # noqa: E402
from aen_replication.utils.seed import set_global_seed  # noqa: E402


FEATURES = topo3.FEATURES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", default=["ambigqa", "situatedqa", "clamber"])
    parser.add_argument(
        "--source-root",
        default="artifacts/steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw",
    )
    parser.add_argument(
        "--clamber-source-root",
        default="artifacts/reports/clamber_conditioned_steering",
    )
    parser.add_argument("--feature-root", default="artifacts/steering_exact_h0_gn_features")
    parser.add_argument("--artifact-root", default="artifacts/steering_exact_h0_gauss_newton")
    parser.add_argument("--steering-layer", type=int, default=14)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--pca-fit-token-cap", type=int, default=64000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-jobs", type=int, default=8)
    parser.add_argument("--optimization-jobs", type=int, default=8)
    parser.add_argument("--neighbor-ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument(
        "--target-mode",
        choices=["nearest_abstention", "local_contrast", "global_contrast"],
        default="nearest_abstention",
        help=(
            "nearest_abstention interpolates toward local D+ topology; local_contrast uses the local "
            "mean(D+) - mean(D-) shift; global_contrast uses the corresponding training-set class shift."
        ),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--dampings", nargs="+", type=float, default=[0.01])
    parser.add_argument("--trust-ratios", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--gn-steps", type=int, default=8)
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument(
        "--allow-mean-shift",
        action="store_true",
        help=(
            "Do not project token perturbations to zero mean. This ablates the "
            "cloud-centroid constraint while retaining norm regularization and the trust region."
        ),
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=100,
        help="Number of eligible held-out direct-answer rows to evaluate; use 0 for all rows.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--behavior-limit-per-split-label",
        type=int,
        default=None,
        help="Optional smoke-test cap for each split/judge-label group before feature extraction.",
    )
    parser.add_argument(
        "--apply-on",
        choices=["prefill_only", "prompt_and_decode_mean"],
        default="prefill_only",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-optimize", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slug_float(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _run_slug(
    *,
    target_mode: str,
    k: int,
    alpha: float,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    allow_mean_shift: bool = False,
) -> str:
    slug = (
        f"target_{target_mode}__k_{int(k)}__alpha_{_slug_float(alpha)}__lambda_{_slug_float(lambda_value)}__"
        f"damping_{_slug_float(damping)}__trust_{_slug_float(trust_ratio)}"
    )
    if allow_mean_shift:
        slug += "__allow_mean_shift"
    return slug


def _resolve_behavior_path(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> Path:
    model_slug = slugify(config["model"]["name"])
    if dataset == "clamber":
        return Path(args.clamber_source_root).resolve() / model_slug / "clamber_base_behavior.parquet"

    root = Path(args.source_root).resolve()
    candidates = [
        root / model_slug / dataset / "_base" / model_slug / f"{dataset}__base_behavior.parquet",
        root / dataset / "_base" / model_slug / f"{dataset}__base_behavior.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {dataset} behavior table; checked: {candidates}")


def _load_behavior(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> pd.DataFrame:
    path = _resolve_behavior_path(args, config, dataset)
    behavior = pd.read_parquet(path)
    behavior = behavior.loc[behavior["judge_label"].isin(["ACCEPTABLE", "UNACCEPTABLE"])].copy()
    behavior["example_id"] = behavior["example_id"].astype(str)
    behavior["behavior_label"] = behavior["judge_label"].eq("ACCEPTABLE").astype(int)
    if "pair_id" not in behavior:
        behavior["pair_id"] = behavior["example_id"]
    if "dataset" not in behavior:
        behavior["dataset"] = dataset
    if "label_ambiguous" not in behavior:
        behavior["label_ambiguous"] = 1
    required = {"example_id", "split", "text", "response_text", "behavior_label"}
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior table {path} is missing columns: {sorted(missing)}")
    if args.behavior_limit_per_split_label is not None:
        limit = int(args.behavior_limit_per_split_label)
        parts = [
            group.sample(n=min(limit, len(group)), random_state=0)
            for _key, group in behavior.groupby(["split", "judge_label"], sort=True)
        ]
        behavior = pd.concat(parts, ignore_index=True)
    return behavior.reset_index(drop=True)


def _feature_paths(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: str,
) -> tuple[Path, Path, Path, Path]:
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(
        Path(args.feature_root).resolve()
        / dataset
        / model_slug
        / f"layer_{int(args.steering_layer):02d}_pca{int(args.pca_components)}"
    )
    return (
        output_root,
        output_root / f"{dataset}__clouds.joblib",
        output_root / f"{dataset}__features.parquet",
        output_root / f"{dataset}__reducer.joblib",
    )


def _build_or_load_layer_features(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: str,
    behavior: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    output_root, cloud_path, feature_path, reducer_path = _feature_paths(args, config, dataset)
    if cloud_path.exists() and feature_path.exists() and reducer_path.exists() and not args.force_features:
        cloud_df = pd.DataFrame(joblib.load(cloud_path)["cloud_df"])
        feature_df = pd.read_parquet(feature_path)
        reducer = joblib.load(reducer_path)
        expected_ids = set(behavior["example_id"].astype(str))
        cached_ids = set(cloud_df["example_id"].astype(str))
        if expected_ids == cached_ids:
            return cloud_df, feature_df, reducer
        print(
            f"[{dataset}] feature cache ID mismatch: expected={len(expected_ids)} cached={len(cached_ids)}; rebuilding",
            flush=True,
        )

    bundle = load_hf_model(config["model"], config["generation"])
    try:
        train_frame = behavior.loc[behavior["split"].eq("train")].reset_index(drop=True)
        if train_frame.empty:
            raise ValueError(f"{dataset} has no training rows for PCA fitting")
        train_rendered = topo3._rendered_prompts(bundle, config, train_frame["text"].astype(str).tolist())
        token_matrices = topo3._collect_pca_fit_tokens(
            bundle=bundle,
            frame=train_frame,
            rendered_prompts=train_rendered,
            layers=[int(args.steering_layer)],
            batch_size=int(args.feature_batch_size),
            max_length=int(args.max_length),
            token_cap=int(args.pca_fit_token_cap),
            seed=seed,
        )
        reducer = topo3._fit_reducers(
            token_matrices,
            n_components=int(args.pca_components),
            seed=seed,
        )[int(args.steering_layer)]
        rendered = topo3._rendered_prompts(bundle, config, behavior["text"].astype(str).tolist())
        cloud_df = topo3._extract_reduced_clouds(
            bundle=bundle,
            frame=behavior,
            rendered_prompts=rendered,
            layers=[int(args.steering_layer)],
            reducers={int(args.steering_layer): reducer},
            batch_size=int(args.feature_batch_size),
            max_length=int(args.max_length),
            topology_dim=int(args.pca_components),
        )
    finally:
        del bundle
        _release()

    feature_df = topo3._compute_feature_frame(cloud_df, parallel_jobs=int(args.feature_jobs))
    joblib.dump({"cloud_df": cloud_df}, cloud_path)
    joblib.dump(reducer, reducer_path)
    write_parquet(feature_df, feature_path)
    feature_df.to_csv(feature_path.with_suffix(".csv"), index=False)
    write_json(
        output_root / f"{dataset}__feature_metadata.json",
        {
            "dataset": dataset,
            "model": config["model"]["name"],
            "layer": int(args.steering_layer),
            "pca_components": int(args.pca_components),
            "pca_fit_split": "train",
            "n_behavior": int(len(behavior)),
            "n_train": int(behavior["split"].eq("train").sum()),
            "features": list(FEATURES),
        },
    )
    return cloud_df, feature_df, reducer


def _select_targets(
    *,
    behavior: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_mode: str,
    k: int,
    eval_n: int,
    limit: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    columns = ["example_id", *FEATURES]
    features = feature_df.loc[:, columns].drop_duplicates("example_id").copy()
    frame = behavior.merge(features, on="example_id", how="inner", validate="one_to_one")
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    train_plus = train.loc[train["behavior_label"].eq(1)].reset_index(drop=True)
    train_minus = train.loc[train["behavior_label"].eq(0)].reset_index(drop=True)
    test_minus = frame.loc[frame["split"].eq("test") & frame["behavior_label"].eq(0)].reset_index(drop=True)
    if train_plus.empty or train_minus.empty or test_minus.empty:
        raise ValueError(
            "Need train ACCEPTABLE, train UNACCEPTABLE, and test UNACCEPTABLE rows; "
            f"got {len(train_plus)}, {len(train_minus)}, and {len(test_minus)}"
        )

    requested_eval_n = int(eval_n)
    n_eval = len(test_minus) if requested_eval_n <= 0 else min(requested_eval_n, len(test_minus))
    eval_df = test_minus.sample(n=n_eval, random_state=seed).reset_index(drop=True)
    if limit is not None:
        eval_df = eval_df.head(int(limit)).reset_index(drop=True)

    scaler = StandardScaler().fit(train.loc[:, FEATURES].to_numpy(dtype=float))
    plus_scaled = scaler.transform(train_plus.loc[:, FEATURES].to_numpy(dtype=float))
    minus_scaled = scaler.transform(train_minus.loc[:, FEATURES].to_numpy(dtype=float))
    eval_scaled = scaler.transform(eval_df.loc[:, FEATURES].to_numpy(dtype=float))
    effective_k = min(int(k), len(train_plus), len(train_minus))
    plus_neighbors = NearestNeighbors(n_neighbors=effective_k, metric="euclidean").fit(plus_scaled)
    plus_distances, plus_indices = plus_neighbors.kneighbors(eval_scaled)
    minus_neighbors = NearestNeighbors(n_neighbors=effective_k, metric="euclidean").fit(minus_scaled)
    minus_distances, minus_indices = minus_neighbors.kneighbors(eval_scaled)
    plus_values = train_plus.loc[:, FEATURES].to_numpy(dtype=np.float32)
    minus_values = train_minus.loc[:, FEATURES].to_numpy(dtype=np.float32)
    plus_local = np.stack([plus_values[row_indices].mean(axis=0) for row_indices in plus_indices]).astype(np.float32)
    minus_local = np.stack([minus_values[row_indices].mean(axis=0) for row_indices in minus_indices]).astype(np.float32)
    current_values = eval_df.loc[:, FEATURES].to_numpy(dtype=np.float32)
    if target_mode == "nearest_abstention":
        target_values = plus_local
    elif target_mode == "local_contrast":
        target_values = current_values + (plus_local - minus_local)
    elif target_mode == "global_contrast":
        global_shift = plus_values.mean(axis=0) - minus_values.mean(axis=0)
        target_values = current_values + global_shift[None, :]
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    feature_std = np.asarray(scaler.scale_, dtype=np.float32)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0).astype(np.float32)

    neighbor_ids = train_plus["example_id"].astype(str).to_numpy()
    minus_neighbor_ids = train_minus["example_id"].astype(str).to_numpy()
    result = eval_df.copy()
    result["target_mode"] = str(target_mode)
    result["base_response_text"] = result["response_text"].astype(str)
    result["base_judge_label"] = result["judge_label"].astype(str)
    result["neighbor_k_requested"] = int(k)
    result["neighbor_k_effective"] = int(effective_k)
    result["neighbor_mean_distance"] = plus_distances.mean(axis=1)
    result["neighbor_min_distance"] = plus_distances.min(axis=1)
    result["direct_neighbor_mean_distance"] = minus_distances.mean(axis=1)
    result["target_example_ids"] = [
        "|".join(neighbor_ids[row_indices].tolist()) for row_indices in plus_indices
    ]
    result["direct_neighbor_example_ids"] = [
        "|".join(minus_neighbor_ids[row_indices].tolist()) for row_indices in minus_indices
    ]
    for feature_index, feature in enumerate(FEATURES):
        result[f"current__{feature}"] = current_values[:, feature_index]
        result[f"target__{feature}"] = target_values[:, feature_index]
    return result, current_values, target_values, feature_std


def _exact_features_tensor(cloud: torch.Tensor) -> torch.Tensor:
    edges = soft_h0._mst_edges_from_cloud(cloud)
    return soft_h0._hard_mst_features_one(cloud, edges, top_k=5)


def _gauss_newton_step(
    *,
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    delta: torch.Tensor,
    hidden_norm: float,
    lambda_value: float,
    damping: float,
) -> torch.Tensor:
    flat_delta = delta.reshape(-1)
    hidden_norm_sq = max(float(hidden_norm) ** 2, 1e-12)
    relative_reg = float(lambda_value) / hidden_norm_sq
    ridge = max(float(damping) + relative_reg, 1e-10)
    gradient = jacobian.T @ residual + relative_reg * flat_delta
    # Woodbury reduces the solve to the 3x3 feature space.
    feature_system = torch.eye(jacobian.shape[0], dtype=jacobian.dtype) + (jacobian @ jacobian.T) / ridge
    correction = jacobian.T @ torch.linalg.solve(feature_system, jacobian @ gradient)
    return (-gradient / ridge + correction / (ridge * ridge)).reshape_as(delta)


def _optimize_one(
    *,
    cloud_np: np.ndarray,
    target_np: np.ndarray,
    feature_std_np: np.ndarray,
    hidden_norm: float,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    gn_steps: int,
    line_search_steps: int,
    enforce_zero_mean: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    dtype = torch.float64
    base = torch.as_tensor(np.asarray(cloud_np, dtype=np.float64), dtype=dtype)
    target = torch.as_tensor(target_np, dtype=dtype)
    feature_std = torch.as_tensor(feature_std_np, dtype=dtype).clamp_min(1e-8)
    delta = torch.zeros_like(base)
    trust_radius = float(trust_ratio) * max(float(hidden_norm), 1e-12)

    def evaluate(candidate_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = _exact_features_tensor(base + candidate_delta)
        residual = (values - target) / feature_std
        relative_norm_sq = candidate_delta.pow(2).sum() / max(float(hidden_norm) ** 2, 1e-12)
        objective = residual.pow(2).sum() + float(lambda_value) * relative_norm_sq
        return values, residual, objective

    initial_values, initial_residual, initial_objective = evaluate(delta)
    objective = initial_objective
    accepted_steps = 0
    total_backtracks = 0
    status = "max_steps"

    for _iteration in range(int(gn_steps)):
        current = (base + delta).detach().requires_grad_(True)
        edges = soft_h0._mst_edges_from_cloud(current)

        def normalized_fixed_tree_features(value: torch.Tensor) -> torch.Tensor:
            raw = soft_h0._hard_mst_features_one(value, edges, top_k=5)
            return raw / feature_std

        values = soft_h0._hard_mst_features_one(current, edges, top_k=5)
        residual = values / feature_std - target / feature_std
        jacobian = torch.autograd.functional.jacobian(
            normalized_fixed_tree_features,
            current,
            vectorize=True,
        ).reshape(len(FEATURES), -1)
        if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
            status = "nonfinite_jacobian"
            break
        if float(torch.linalg.norm(residual).detach()) < 1e-5:
            status = "converged"
            break

        step = _gauss_newton_step(
            jacobian=jacobian,
            residual=residual.detach(),
            delta=delta,
            hidden_norm=hidden_norm,
            lambda_value=lambda_value,
            damping=damping,
        )
        if enforce_zero_mean:
            step = step - step.mean(dim=0, keepdim=True)
        accepted = False
        for backtrack in range(int(line_search_steps)):
            scale = 0.5**backtrack
            candidate = delta + scale * step
            if enforce_zero_mean:
                candidate = candidate - candidate.mean(dim=0, keepdim=True)
            candidate_norm = float(torch.linalg.norm(candidate))
            if trust_radius > 0.0 and candidate_norm > trust_radius:
                candidate = candidate * (trust_radius / max(candidate_norm, 1e-12))
            _candidate_values, _candidate_residual, candidate_objective = evaluate(candidate)
            if torch.isfinite(candidate_objective) and float(candidate_objective) < float(objective) - 1e-10:
                delta = candidate.detach()
                objective = candidate_objective.detach()
                accepted_steps += 1
                total_backtracks += backtrack
                accepted = True
                break
        if not accepted:
            status = "line_search_stopped"
            break
    else:
        status = "max_steps"

    final_values, final_residual, final_objective = evaluate(delta)
    delta_np = delta.numpy().astype(np.float32, copy=False)
    delta_token_mean_norm = float(np.linalg.norm(delta_np.mean(axis=0)))
    diagnostics: dict[str, Any] = {
        "optimization_status": status,
        "accepted_steps": int(accepted_steps),
        "line_search_backtracks": int(total_backtracks),
        "initial_normalized_target_error": float(torch.linalg.norm(initial_residual)),
        "final_normalized_target_error": float(torch.linalg.norm(final_residual)),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "pca_delta_norm": float(np.linalg.norm(delta_np)),
        "pca_cloud_norm": float(np.linalg.norm(cloud_np)),
        "relative_hidden_delta_norm": float(np.linalg.norm(delta_np) / max(float(hidden_norm), 1e-12)),
        "delta_token_mean_norm": delta_token_mean_norm,
        "zero_mean_constraint": bool(enforce_zero_mean),
        "trust_radius": float(trust_radius),
    }
    for feature_index, feature in enumerate(FEATURES):
        diagnostics[f"initial_exact__{feature}"] = float(initial_values[feature_index])
        diagnostics[f"final_exact__{feature}"] = float(final_values[feature_index])
        diagnostics[f"optimization_target__{feature}"] = float(target[feature_index])
    return delta_np, diagnostics


def _optimize_many(
    *,
    clouds: list[np.ndarray],
    target_features: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
    args: argparse.Namespace,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    tasks = (
        delayed(_optimize_one)(
            cloud_np=cloud,
            target_np=target,
            feature_std_np=feature_std,
            hidden_norm=float(hidden_norm),
            lambda_value=float(lambda_value),
            damping=float(damping),
            trust_ratio=float(trust_ratio),
            gn_steps=int(args.gn_steps),
            line_search_steps=int(args.line_search_steps),
            enforce_zero_mean=not bool(args.allow_mean_shift),
        )
        for cloud, target, hidden_norm in zip(clouds, target_features, hidden_norms, strict=True)
    )
    results = Parallel(n_jobs=max(1, int(args.optimization_jobs)), backend="loky")(
        tqdm(tasks, total=len(clouds), desc="exact_h0_gn", leave=False)
    )
    deltas = [result[0] for result in results]
    diagnostics = pd.DataFrame([{"row_index": index, **result[1]} for index, result in enumerate(results)])
    return deltas, diagnostics


def _run_dataset(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> None:
    seed = int(config["seed"])
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(args.artifact_root).resolve() / dataset / model_slug)
    behavior = _load_behavior(args, config, dataset)
    counts = behavior.groupby(["split", "judge_label"]).size().to_dict()
    print(f"[{dataset} {model_slug}] behavior={len(behavior)} counts={counts}", flush=True)

    cloud_df, feature_df, reducer = _build_or_load_layer_features(
        args=args,
        config=config,
        dataset=dataset,
        behavior=behavior,
        seed=seed,
    )
    cloud_by_id = cloud_df.drop_duplicates("example_id").set_index("example_id")
    components = reducer.components_[: int(args.pca_components)].astype(np.float32, copy=False)
    bundle = None
    try:
        for k in args.neighbor_ks:
            neighbor_path = output_root / f"{dataset}__neighbors_k{int(k)}.parquet"
            rows_df, current_features, neighbor_targets, feature_std = _select_targets(
                behavior=behavior,
                feature_df=feature_df,
                target_mode=str(args.target_mode),
                k=int(k),
                eval_n=int(args.eval_n),
                limit=args.limit,
                seed=seed,
            )
            write_parquet(rows_df, neighbor_path)
            rows_df.to_csv(neighbor_path.with_suffix(".csv"), index=False)
            example_ids = rows_df["example_id"].astype(str).tolist()
            clouds = [np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32) for example_id in example_ids]
            hidden_norms = np.asarray(
                [float(cloud_by_id.loc[example_id, "hidden_fro_norm"]) for example_id in example_ids],
                dtype=np.float64,
            )

            for alpha in args.alphas:
                z_target = current_features + float(alpha) * (neighbor_targets - current_features)
                for lambda_value in args.lambdas:
                    for damping in args.dampings:
                        for trust_ratio in args.trust_ratios:
                            run = _run_slug(
                                target_mode=str(args.target_mode),
                                k=int(k),
                                alpha=float(alpha),
                                lambda_value=float(lambda_value),
                                damping=float(damping),
                                trust_ratio=float(trust_ratio),
                                allow_mean_shift=bool(args.allow_mean_shift),
                            )
                            run_root = ensure_dir(output_root / run)
                            opt_path = run_root / f"{dataset}__exact_h0_gn__optimization.parquet"
                            raw_path = run_root / f"{dataset}__exact_h0_gn__raw.parquet"
                            if raw_path.exists() and not args.force_generate and not args.skip_generate:
                                print(f"[{dataset} {run}] raw exists", flush=True)
                                continue

                            if opt_path.exists() and not args.force_optimize:
                                opt_df = pd.read_parquet(opt_path)
                                delta_y = [np.asarray(value, dtype=np.float32) for value in opt_df["pca_delta"].tolist()]
                            else:
                                delta_y, diagnostics = _optimize_many(
                                    clouds=clouds,
                                    target_features=z_target,
                                    feature_std=feature_std,
                                    hidden_norms=hidden_norms,
                                    args=args,
                                    lambda_value=float(lambda_value),
                                    damping=float(damping),
                                    trust_ratio=float(trust_ratio),
                                )
                                opt_df = pd.concat(
                                    [rows_df.reset_index(drop=True), diagnostics.drop(columns="row_index")],
                                    axis=1,
                                )
                                opt_df["pca_delta"] = [delta.tolist() for delta in delta_y]
                                opt_df["neighbor_k"] = int(k)
                                opt_df["target_mode"] = str(args.target_mode)
                                opt_df["alpha"] = float(alpha)
                                opt_df["lambda"] = float(lambda_value)
                                opt_df["damping"] = float(damping)
                                opt_df["trust_ratio"] = float(trust_ratio)
                                for feature_index, feature in enumerate(FEATURES):
                                    opt_df[f"z_target__{feature}"] = z_target[:, feature_index]
                                write_parquet(opt_df, opt_path)
                                opt_df.drop(columns="pca_delta").to_csv(opt_path.with_suffix(".csv"), index=False)

                            if args.skip_generate:
                                continue
                            if bundle is None:
                                bundle = load_hf_model(config["model"], config["generation"])
                            delta_h = [(delta @ components).astype(np.float32, copy=False) for delta in delta_y]
                            responses = topo3._generate_with_token_deltas(
                                bundle=bundle,
                                config=config,
                                rows_df=rows_df,
                                token_deltas_h=delta_h,
                                layer=int(args.steering_layer),
                                max_length=int(args.max_length),
                                apply_on=str(args.apply_on),
                            )
                            run_df = opt_df.drop(columns="pca_delta").copy()
                            run_df["prompt_text"] = topo3._prompt_texts(config, run_df["text"].astype(str).tolist())
                            run_df["response_text"] = responses
                            run_df["strategy"] = "exact_h0_gauss_newton"
                            run_df["apply_on"] = str(args.apply_on)
                            run_df["layer"] = int(args.steering_layer)
                            write_parquet(run_df, raw_path)
                            run_df.to_csv(raw_path.with_suffix(".csv"), index=False)
                            write_json(
                                run_root / f"{dataset}__exact_h0_gn__summary.json",
                                {
                                    "dataset": dataset,
                                    "model": config["model"]["name"],
                                    "strategy": "exact_h0_gauss_newton",
                                    "n_eval": int(len(run_df)),
                                    "layer": int(args.steering_layer),
                                    "pca_components": int(args.pca_components),
                                    "neighbor_k": int(k),
                                    "target_mode": str(args.target_mode),
                                    "alpha": float(alpha),
                                    "lambda": float(lambda_value),
                                    "damping": float(damping),
                                    "trust_ratio": float(trust_ratio),
                                    "gn_steps": int(args.gn_steps),
                                    "zero_mean_constraint": not bool(args.allow_mean_shift),
                                    "allow_mean_shift": bool(args.allow_mean_shift),
                                    "apply_on": str(args.apply_on),
                                    "relative_hidden_delta_norm_mean": float(
                                        run_df["relative_hidden_delta_norm"].mean()
                                    ),
                                    "delta_token_mean_norm_mean": float(
                                        run_df["delta_token_mean_norm"].mean()
                                    ),
                                    "initial_normalized_target_error_mean": float(
                                        run_df["initial_normalized_target_error"].mean()
                                    ),
                                    "final_normalized_target_error_mean": float(
                                        run_df["final_normalized_target_error"].mean()
                                    ),
                                    "unique_responses": int(run_df["response_text"].nunique()),
                                    "empty_responses": int(run_df["response_text"].str.strip().eq("").sum()),
                                },
                            )
    finally:
        if bundle is not None:
            del bundle
        _release()


def main() -> None:
    args = _parse_args()
    set_global_seed(0)
    config = load_config(args.config)
    for dataset in args.datasets:
        _run_dataset(args, config, dataset)


if __name__ == "__main__":
    main()
