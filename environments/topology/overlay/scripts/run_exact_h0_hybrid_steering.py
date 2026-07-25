#!/usr/bin/env python3
"""Topology-changing, behavior-anchored token-specific steering.

The intervention decomposes into

    delta H_t = C_t + alpha_mean * v_local,

where C is a centered, token-specific exact-H0 Gauss-Newton deformation and
v_local is a raw mean-hidden-state difference. All-layer H0 mean persistence
selects local clarification-aware and direct-answer training neighbors. The
shared component is applied to prompt and decode tokens; the centered component
is applied only to the corresponding prompt-token cloud.

No API client is imported or called by this script.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_exact_h0_gauss_newton_steering as exact  # noqa: E402
import run_topology3_surrogate_steering as topo3  # noqa: E402

from aen_replication.config import load_config  # noqa: E402
from aen_replication.models.hf_model import load_hf_model  # noqa: E402
from aen_replication.train.steering import _extract_prompt_vectors, _prompt_texts  # noqa: E402
from aen_replication.utils.io_utils import ensure_dir, slugify, write_json, write_parquet  # noqa: E402
from aen_replication.utils.seed import set_global_seed  # noqa: E402


FEATURES = exact.FEATURES
LOCAL_POSITIVE = "GROUNDED_ACCEPTABLE"
LOCAL_NEGATIVE = "UNACCEPTABLE"
RULE_POSITIVE = "RULE_CLARIFICATION_AWARE"
RULE_NEGATIVE = "RULE_DIRECT"
TARGETED_RULE_POSITIVE = "RULE_TARGETED_CLARIFICATION"
DEFAULT_CONTRASTIVE_POSITIVE_INSTRUCTION = (
    "Treat the question as potentially ill-posed. Identify the specific missing condition, "
    "ambiguous entity, time, place, or interpretation, and ask a targeted clarification before "
    "giving any answer."
)
DEFAULT_CONTRASTIVE_NEGATIVE_INSTRUCTION = (
    "Answer the question directly with one answer. Do not ask for clarification and do not discuss "
    "whether the question is underspecified."
)
HIGH_PRECISION_BEHAVIOR_MARKERS = (
    "please clarify",
    "could you clarify",
    "can you clarify",
    "please specify",
    "could you specify",
    "can you specify",
    "do you mean",
    "what do you mean",
    "which one do you mean",
    "not enough information",
    "insufficient information",
    "need more information",
    "need more context",
    "depends on which",
    "depends on the year",
    "depends on the season",
    "depends on the country",
    "depends on the version",
    "multiple possible",
    "several possible",
    "real-time information",
    "current information",
    "latest information",
    "knowledge cutoff",
    "i'm not aware",
    "i am not aware",
    "i'm not familiar",
    "i am not familiar",
    "not provided",
    "not available",
    "not readily available",
    "not well-documented",
    "not well documented",
    "not known",
    "don't have the specific",
    "do not have the specific",
    "cannot determine",
    "can't determine",
    "impossible to determine",
    "without more specific",
    "for more precise information",
    "couldn't find any information",
    "could not find any information",
)
TARGETED_CLARIFICATION_MARKERS = (
    "please clarify",
    "could you clarify",
    "can you clarify",
    "please specify",
    "could you specify",
    "can you specify",
    "do you mean",
    "what do you mean",
    "which one do you mean",
    "provide more context",
    "provide more information",
    "provide more details",
    "not enough information",
    "insufficient information",
    "need more information",
    "need more context",
    "depends on which",
    "depends on the year",
    "depends on the season",
    "depends on the country",
    "depends on the version",
    "multiple possible",
    "several possible",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", default=["ambigqa", "situatedqa", "clamber"])
    parser.add_argument(
        "--source-root",
        default="artifacts/steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw",
    )
    parser.add_argument("--clamber-source-root", default="artifacts/reports/clamber_conditioned_steering")
    parser.add_argument("--feature-root", default="artifacts/steering_exact_h0_gn_features")
    parser.add_argument(
        "--direction-cache-root",
        default="artifacts/steering_exact_h0_direction_cache",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/steering_exact_h0_hybrid_grounded_alllayer",
    )
    parser.add_argument(
        "--local-label-path",
        default="artifacts/local_llm_rejudge_all_openai_outputs/local_judge_unique_pair_labels.parquet",
    )
    parser.add_argument(
        "--behavior-label-source",
        choices=(
            "legacy_local_fourway",
            "audited_local_fourway",
            "source_judge",
            "rule_high_precision",
            "rule_targeted_clarification",
        ),
        default="legacy_local_fourway",
        help=(
            "Construct D+/D- from the legacy local cache, a freshly audited rotating-choice local "
            "cache, the behavior table's existing ACCEPTABLE/UNACCEPTABLE labels, or deterministic "
            "high-precision clarification cues."
        ),
    )
    parser.add_argument(
        "--local-label-confidence-min",
        type=float,
        default=0.0,
        help="Minimum top-class probability for audited_local_fourway training labels.",
    )
    parser.add_argument(
        "--local-label-margin-min",
        type=float,
        default=0.0,
        help="Minimum top-two probability margin for audited_local_fourway training labels.",
    )
    parser.add_argument(
        "--require-positive-rule-marker",
        action="store_true",
        help=(
            "For audited_local_fourway, retain a positive D+ example only when its response also "
            "contains a high-precision clarification or insufficient-information cue."
        ),
    )
    parser.add_argument(
        "--positive-label-mode",
        choices=["grounded", "all_acceptable"],
        default="grounded",
        help=(
            "grounded uses only GROUNDED_ACCEPTABLE as D+; all_acceptable also includes "
            "GENERIC_ACCEPTABLE while retaining four-way evaluation."
        ),
    )
    parser.add_argument(
        "--ambigqa-all-layer-root",
        default="artifacts/token_cloud_h0_mean_persistence_all_layers_80_20",
    )
    parser.add_argument(
        "--situatedqa-all-layer-root",
        default="artifacts/token_cloud_h0_mean_persistence_all_layers_80_20_situatedqa",
    )
    parser.add_argument(
        "--clamber-all-layer-root",
        default="artifacts/token_cloud_h0_mean_persistence_all_layers_clamber",
    )
    parser.add_argument("--steering-layer", type=int, default=14)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--neighbor-ks", nargs="+", type=int, default=[20])
    parser.add_argument(
        "--retrieval-feature-mode",
        choices=("all_layer_mean", "all_layer_plus_exact3", "exact3"),
        default="all_layer_mean",
        help=(
            "Topology vector used only for local D+/D- retrieval. The controlled target remains "
            "the three exact layer-level H0 statistics."
        ),
    )
    parser.add_argument(
        "--retrieval-geometry",
        choices=("standard", "class_residual"),
        default="standard",
        help=(
            "Use standardized topology distance directly, or remove the global D+ minus D- "
            "topology axis before matching so neighbors align on residual question structure."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=[
            "nearest_grounded",
            "local_contrast",
            "global_contrast",
            "classifier_projection",
            "self_contrastive_topology",
        ],
        default="nearest_grounded",
    )
    parser.add_argument(
        "--classifier-target-quantile",
        type=float,
        default=0.5,
        help=(
            "For classifier_projection, project to this training D+ logit quantile along the "
            "minimum-norm standardized exact-H0 direction."
        ),
    )
    parser.add_argument("--topology-alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--mean-alphas", nargs="+", type=float, default=[0.0, 2.0, 4.0, 6.0, 8.0])
    parser.add_argument(
        "--shared-target-ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Replace raw mean-alphas with per-example scales chosen so the shared prompt "
            "perturbation has this Frobenius norm relative to the unsteered prompt state."
        ),
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--dampings", nargs="+", type=float, default=[0.01])
    parser.add_argument("--trust-ratios", nargs="+", type=float, default=[0.05])
    parser.add_argument("--gn-steps", type=int, default=8)
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument("--optimization-jobs", type=int, default=8)
    parser.add_argument(
        "--topology-controller",
        choices=[
            "free",
            "behavior_rank1",
            "behavior_lowrank",
            "behavior_tokenwise",
            "transport_prior",
        ],
        default="free",
        help=(
            "free optimizes an arbitrary token-specific PCA displacement; behavior_rank1 "
            "optimizes one centered scalar gate per token along the local clarification direction; "
            "behavior_lowrank uses a local subspace of grounded-minus-direct contrasts; "
            "behavior_tokenwise uses a query-specific clarification-minus-direct vector at "
            "each token; "
            "transport_prior initializes from token-wise nearest-point D+ minus D- cloud transport."
        ),
    )
    parser.add_argument("--behavior-rank", type=int, default=4)
    parser.add_argument(
        "--direction-readout",
        choices=("mean_pool", "last_token"),
        default="mean_pool",
        help=(
            "Readout used to construct the hidden-space behavior direction. The topology "
            "objective still acts on the full prompt-token cloud. last_token tests a more "
            "causally proximal inverse subspace without changing the controlled H0 features."
        ),
    )
    parser.add_argument(
        "--direction-source",
        choices=(
            "observed_groups",
            "observed_neighbor_tokenwise",
            "contrastive_prompt_pairs",
            "self_contrastive_prompt_pair",
            "self_contrastive_tokenwise",
        ),
        default="observed_groups",
        help=(
            "Build hidden-state directions from observed D+ minus D- groups or from paired "
            "clarification/direct instructions. contrastive_prompt_pairs averages paired "
            "training-question differences selected by topology; self_contrastive_prompt_pair "
            "uses the held-out question's own paired states to preserve its content; "
            "self_contrastive_tokenwise retains the full paired token sequence; "
            "observed_neighbor_tokenwise instead contrasts aligned token sequences from the "
            "topology-matched observed D+ and D- neighbors. Both tokenwise sources let different "
            "prompt tokens move in different hidden-space directions."
        ),
    )
    parser.add_argument(
        "--contrastive-positive-instruction",
        default=DEFAULT_CONTRASTIVE_POSITIVE_INSTRUCTION,
    )
    parser.add_argument(
        "--contrastive-negative-instruction",
        default=DEFAULT_CONTRASTIVE_NEGATIVE_INSTRUCTION,
    )
    parser.add_argument(
        "--causal-anchor-ratio",
        type=float,
        default=0.0,
        help=(
            "For behavior_lowrank, add a final-token shift along the positive local direction, "
            "projected into the exact-H0 Jacobian nullspace. Value is relative to ||H||_F."
        ),
    )
    parser.add_argument(
        "--causal-anchor-max-error-increase",
        type=float,
        default=0.1,
        help="Maximum increase in normalized exact-H0 target error accepted for the causal anchor.",
    )
    parser.add_argument(
        "--causal-anchor-suffix-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of prompt tokens receiving the positive causal anchor. Zero preserves the "
            "single-final-token anchor; values in (0, 1) distribute it over the prompt suffix."
        ),
    )
    parser.add_argument(
        "--causal-position-beta",
        type=float,
        default=0.0,
        help=(
            "Exponentially favor later prompt tokens during Gauss-Newton inversion. Zero is uniform; "
            "larger values penalize early-token coefficients more strongly."
        ),
    )
    parser.add_argument(
        "--transport-prior-ratio",
        type=float,
        default=0.025,
        help="Frobenius norm of the transport prior relative to the unsteered hidden-state norm.",
    )
    parser.add_argument(
        "--transport-match-mode",
        choices=("nearest", "position"),
        default="nearest",
        help=(
            "Construct the token-specific transport prior by nearest-point cloud matching or by "
            "interpolating neighbor token sequences onto the query's normalized token positions."
        ),
    )
    parser.add_argument("--eval-n", type=int, default=0, help="Use 0 for all held-out local-judge negatives.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--apply-on",
        choices=["prefill_only", "prompt_and_decode_shared"],
        default="prompt_and_decode_shared",
    )
    parser.add_argument(
        "--shared-intervention-site",
        choices=["layer_output", "layer_input"],
        default="layer_output",
        help=(
            "Apply the shared local direction at the selected layer output or input. "
            "The token-specific topology deformation is always applied at the layer output."
        ),
    )
    parser.add_argument(
        "--topology-decode-mode",
        choices=("none", "last_token", "suffix_mean"),
        default="none",
        help=(
            "Optionally carry a vector derived from the optimized token-specific topology "
            "deformation into every decode step. The prompt always receives the full per-token "
            "deformation."
        ),
    )
    parser.add_argument(
        "--topology-decode-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the topology-derived decode vector.",
    )
    parser.add_argument(
        "--topology-decode-suffix-fraction",
        type=float,
        default=0.25,
        help="Prompt suffix fraction averaged when --topology-decode-mode=suffix_mean.",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--force-optimize", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-directions", action="store_true")
    return parser.parse_args()


def _pair_hash(question: str, response: str) -> str:
    return hashlib.sha256(f"{question}\0{response}".encode("utf-8")).hexdigest()


def _slug_float(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _bounded_slug(value: str, max_length: int) -> str:
    """Keep filesystem components below NAME_MAX without losing uniqueness."""
    if len(value) <= int(max_length):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, int(max_length) - len(digest) - len("__h_"))
    return f"{value[:prefix_length]}__h_{digest}"


def _behavior_source_slug(args: argparse.Namespace) -> str:
    source = str(args.behavior_label_source)
    if source == "audited_local_fourway":
        source += (
            f"__conf_{_slug_float(args.local_label_confidence_min)}"
            f"__margin_{_slug_float(args.local_label_margin_min)}"
        )
        if args.require_positive_rule_marker:
            source += "__rulepositive"
    return source


def _as_float_matrix(value: Any) -> np.ndarray:
    """Normalize Arrow nested-list/object-array cells to a dense matrix."""
    array = np.asarray(value)
    if array.dtype == object:
        array = np.stack(array.tolist())
    return array.astype(np.float32, copy=False)


def _as_float_vector(value: Any) -> np.ndarray:
    """Normalize an Arrow list/object-array cell to a dense vector."""
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return array.astype(np.float32, copy=False).reshape(-1)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_high_precision_behavior_marker(response: str) -> bool:
    normalized = " ".join(str(response).lower().split())
    if any(marker in normalized for marker in HIGH_PRECISION_BEHAVIOR_MARKERS):
        return True
    targeted_question_patterns = (
        r"\bwhich\s+(?:country|year|season|version|episode|film|book|game|team|person|place|date)\b",
        r"\bwhat\s+(?:country|year|season|version|context|location|date)\b",
        r"\bwho\s+(?:exactly|specifically)\b",
    )
    return any(re.search(pattern, normalized) for pattern in targeted_question_patterns)


def _has_targeted_clarification_marker(response: str) -> bool:
    """Detect question-specific clarification rather than a generic refusal."""
    normalized = " ".join(str(response).lower().split())
    if any(marker in normalized for marker in TARGETED_CLARIFICATION_MARKERS):
        return True
    targeted_patterns = (
        r"\bwhich\s+(?:country|year|season|version|episode|film|book|game|team|person|place|date|one)\b",
        r"\bwhat\s+(?:country|year|season|version|context|location|date)\b",
        r"\bwho\s+(?:exactly|specifically)\b",
        r"\bwhich\s+.+\s+are you (?:asking|referring|talking) about\b",
        r"\bwhat\s+.+\s+are you (?:asking|referring|talking) about\b",
    )
    return any(re.search(pattern, normalized) for pattern in targeted_patterns)


def _is_degenerate_response(response: str) -> bool:
    """Match the conservative empty/repetition gate used by the local evaluator."""
    tokens = re.findall(r"[a-z0-9']+", str(response).lower())
    if not tokens:
        return True
    if len(tokens) < 24:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    counts: dict[tuple[str, str, str], int] = {}
    for trigram in trigrams:
        counts[trigram] = counts.get(trigram, 0) + 1
    distinct_ratio = len(counts) / max(len(trigrams), 1)
    max_trigram_count = max(counts.values(), default=0)
    token_counts: dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    max_token_ratio = max(token_counts.values(), default=0) / len(tokens)
    return (max_trigram_count >= 4 and distinct_ratio <= 0.6) or max_token_ratio >= 0.4


def _load_local_behavior(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: str,
) -> pd.DataFrame:
    path = exact._resolve_behavior_path(args, config, dataset)
    behavior = pd.read_parquet(path).copy()
    behavior["source_judge_label"] = behavior["judge_label"].astype(str)
    behavior["pair_hash"] = [
        _pair_hash(str(question), str(response))
        for question, response in zip(behavior["text"], behavior["response_text"], strict=True)
    ]
    if args.behavior_label_source in {"legacy_local_fourway", "audited_local_fourway"}:
        label_path = Path(args.local_label_path).resolve()
        labels = pd.read_parquet(label_path)
        required = {"pair_hash", "local_judge_label"}
        missing_columns = required.difference(labels.columns)
        if missing_columns:
            raise ValueError(f"Missing {sorted(missing_columns)} in local label table {label_path}")
        labels = labels.drop_duplicates("pair_hash").set_index("pair_hash")
        behavior["judge_label"] = behavior["pair_hash"].map(labels["local_judge_label"])
        if args.behavior_label_source == "audited_local_fourway":
            if "local_judge_confidence" not in labels or "local_judge_margin" not in labels:
                raise ValueError(
                    "audited_local_fourway requires local_judge_confidence and local_judge_margin "
                    f"in {label_path}"
                )
            behavior["local_judge_confidence"] = behavior["pair_hash"].map(
                labels["local_judge_confidence"]
            )
            behavior["local_judge_margin"] = behavior["pair_hash"].map(labels["local_judge_margin"])
            behavior = behavior.loc[
                behavior["local_judge_confidence"].ge(float(args.local_label_confidence_min))
                & behavior["local_judge_margin"].ge(float(args.local_label_margin_min))
            ].copy()
        missing = int(behavior["judge_label"].isna().sum())
        if missing:
            raise ValueError(
                f"Local four-way labels missing for {missing}/{len(behavior)} rows in {path}; "
                f"label table is {label_path}"
            )
        positive_labels = {LOCAL_POSITIVE}
        if args.positive_label_mode == "all_acceptable":
            positive_labels.add("GENERIC_ACCEPTABLE")
        behavior = behavior.loc[
            behavior["judge_label"].isin([*positive_labels, LOCAL_NEGATIVE])
        ].copy()
        if (
            args.behavior_label_source == "audited_local_fourway"
            and args.require_positive_rule_marker
        ):
            positive = behavior["judge_label"].isin(positive_labels)
            positive_has_marker = behavior["response_text"].fillna("").astype(str).map(
                _has_high_precision_behavior_marker
            )
            positive_is_valid = ~behavior["response_text"].fillna("").astype(str).map(
                _is_degenerate_response
            )
            behavior = behavior.loc[~positive | (positive_has_marker & positive_is_valid)].copy()
        behavior["behavior_label"] = behavior["judge_label"].isin(positive_labels).astype(int)
    elif args.behavior_label_source == "source_judge":
        behavior = behavior.loc[
            behavior["source_judge_label"].isin(["ACCEPTABLE", "UNACCEPTABLE"])
        ].copy()
        behavior["behavior_label"] = behavior["source_judge_label"].eq("ACCEPTABLE").astype(int)
        behavior["judge_label"] = np.where(
            behavior["behavior_label"].eq(1),
            LOCAL_POSITIVE,
            LOCAL_NEGATIVE,
        )
    else:
        marker = (
            _has_targeted_clarification_marker
            if args.behavior_label_source == "rule_targeted_clarification"
            else _has_high_precision_behavior_marker
        )
        behavior["behavior_label"] = behavior["response_text"].fillna("").astype(str).map(
            marker
        ).astype(int)
        behavior["judge_label"] = np.where(
            behavior["behavior_label"].eq(1),
            (
                TARGETED_RULE_POSITIVE
                if args.behavior_label_source == "rule_targeted_clarification"
                else RULE_POSITIVE
            ),
            RULE_NEGATIVE,
        )
    behavior["behavior_label_source"] = str(args.behavior_label_source)
    behavior["example_id"] = behavior["example_id"].astype(str)
    if "pair_id" not in behavior:
        behavior["pair_id"] = behavior["example_id"]
    if "dataset" not in behavior:
        behavior["dataset"] = dataset
    if "label_ambiguous" not in behavior:
        behavior["label_ambiguous"] = 1
    return behavior.reset_index(drop=True)


def _load_exact_cache(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    _root, cloud_path, feature_path, reducer_path = exact._feature_paths(args, config, dataset)
    for path in (cloud_path, feature_path, reducer_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Hybrid steering reuses the exact-H0 feature cache, but {path} is missing. "
                "Run the exact-H0 feature builder first."
            )
    cloud_df = pd.DataFrame(joblib.load(cloud_path)["cloud_df"])
    feature_df = pd.read_parquet(feature_path)
    reducer = joblib.load(reducer_path)
    cloud_df["example_id"] = cloud_df["example_id"].astype(str)
    feature_df["example_id"] = feature_df["example_id"].astype(str)
    return cloud_df, feature_df, reducer


def _all_layer_path(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> Path:
    roots = {
        "ambigqa": args.ambigqa_all_layer_root,
        "situatedqa": args.situatedqa_all_layer_root,
        "clamber": args.clamber_all_layer_root,
    }
    model_slug = slugify(config["model"]["name"])
    return (
        Path(roots[dataset]).resolve()
        / model_slug
        / "token_cloud_h0_mean_persistence_all_layers.parquet"
    )


def _load_all_layer_features(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: str,
) -> tuple[pd.DataFrame, list[str]]:
    path = _all_layer_path(args, config, dataset)
    frame = pd.read_parquet(path)
    frame = frame.loc[
        frame["dataset"].astype(str).eq(dataset)
        & frame["feature_variant"].astype(str).eq("h0_mean_persistence_all_layers")
    ].copy()
    columns = sorted(
        column
        for column in frame.columns
        if str(column).startswith("h0_mean_persistence__l")
    )
    if not columns:
        raise ValueError(f"No all-layer H0 mean-persistence columns in {path}")
    frame["example_id"] = frame["example_id"].astype(str)
    return frame[["example_id", *columns]].drop_duplicates("example_id"), columns


def _prepare_frame(
    behavior: pd.DataFrame,
    feature_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
) -> pd.DataFrame:
    layer_features = feature_df[["example_id", *FEATURES]].drop_duplicates("example_id")
    frame = behavior.merge(layer_features, on="example_id", how="inner", validate="one_to_one")
    frame = frame.merge(retrieval_df, on="example_id", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No rows remain after joining behavior, exact-H0, and all-layer retrieval features")
    return frame


def _select_neighbors_and_targets(
    *,
    frame: pd.DataFrame,
    retrieval_columns: list[str],
    retrieval_geometry: str,
    target_mode: str,
    k: int,
    eval_n: int,
    limit: int | None,
    seed: int,
    classifier_target_quantile: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    plus = train.loc[train["behavior_label"].eq(1)].reset_index(drop=True)
    minus = train.loc[train["behavior_label"].eq(0)].reset_index(drop=True)
    test_minus = frame.loc[
        frame["split"].eq("test") & frame["behavior_label"].eq(0)
    ].reset_index(drop=True)
    if plus.empty or minus.empty or test_minus.empty:
        raise ValueError(f"Need local D+, local D-, and held-out D-; got {len(plus)}, {len(minus)}, {len(test_minus)}")
    n_eval = len(test_minus) if int(eval_n) <= 0 else min(int(eval_n), len(test_minus))
    eval_df = test_minus.sample(n=n_eval, random_state=seed).reset_index(drop=True)
    if limit is not None:
        eval_df = eval_df.head(int(limit)).reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_retrieval = imputer.fit_transform(train[retrieval_columns])
    scaler.fit(train_retrieval)
    plus_scaled = scaler.transform(imputer.transform(plus[retrieval_columns]))
    minus_scaled = scaler.transform(imputer.transform(minus[retrieval_columns]))
    eval_scaled = scaler.transform(imputer.transform(eval_df[retrieval_columns]))
    if retrieval_geometry == "class_residual":
        class_direction = plus_scaled.mean(axis=0) - minus_scaled.mean(axis=0)
        class_direction_norm = float(np.linalg.norm(class_direction))
        if class_direction_norm > 1e-12:
            class_direction /= class_direction_norm

            def residualize(values: np.ndarray) -> np.ndarray:
                return values - (values @ class_direction)[:, None] * class_direction[None, :]

            plus_scaled = residualize(plus_scaled)
            minus_scaled = residualize(minus_scaled)
            eval_scaled = residualize(eval_scaled)
    effective_k = min(int(k), len(plus), len(minus))
    plus_nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean").fit(plus_scaled)
    minus_nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean").fit(minus_scaled)
    plus_dist, plus_idx = plus_nn.kneighbors(eval_scaled)
    minus_dist, minus_idx = minus_nn.kneighbors(eval_scaled)

    plus_values = plus[list(FEATURES)].to_numpy(dtype=np.float32)
    minus_values = minus[list(FEATURES)].to_numpy(dtype=np.float32)
    current = eval_df[list(FEATURES)].to_numpy(dtype=np.float32)
    plus_local = np.stack([plus_values[index].mean(axis=0) for index in plus_idx]).astype(np.float32)
    minus_local = np.stack([minus_values[index].mean(axis=0) for index in minus_idx]).astype(np.float32)
    classifier_current_score: np.ndarray | None = None
    classifier_target_score: np.ndarray | None = None
    classifier_raw_weight: np.ndarray | None = None
    classifier_raw_intercept: float | None = None
    if target_mode == "nearest_grounded":
        neighbor_target = plus_local
    elif target_mode == "local_contrast":
        neighbor_target = current + plus_local - minus_local
    elif target_mode == "global_contrast":
        neighbor_target = current + plus_values.mean(axis=0) - minus_values.mean(axis=0)
    elif target_mode == "classifier_projection":
        exact_imputer = SimpleImputer(strategy="median")
        exact_scaler = StandardScaler()
        train_exact = exact_imputer.fit_transform(train[list(FEATURES)])
        train_exact_scaled = exact_scaler.fit_transform(train_exact)
        eval_exact_scaled = exact_scaler.transform(exact_imputer.transform(eval_df[list(FEATURES)]))
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ).fit(train_exact_scaled, train["behavior_label"].to_numpy(dtype=int))
        weight = classifier.coef_.reshape(-1)
        weight_norm_sq = max(float(weight @ weight), 1e-12)
        classifier_current_score = classifier.decision_function(eval_exact_scaled).reshape(-1)
        plus_exact_scaled = exact_scaler.transform(exact_imputer.transform(plus[list(FEATURES)]))
        plus_scores = classifier.decision_function(plus_exact_scaled).reshape(-1)
        quantile = min(max(float(classifier_target_quantile), 0.0), 1.0)
        target_score_value = float(np.quantile(plus_scores, quantile))
        classifier_target_score = np.maximum(classifier_current_score, target_score_value)
        score_delta = classifier_target_score - classifier_current_score
        target_scaled = eval_exact_scaled + score_delta[:, None] * weight[None, :] / weight_norm_sq
        neighbor_target = exact_scaler.inverse_transform(target_scaled).astype(np.float32)
        classifier_raw_weight = weight / exact_scaler.scale_
        classifier_raw_intercept = float(
            classifier.intercept_[0] - classifier_raw_weight @ exact_scaler.mean_
        )
    else:
        # The query-specific counterfactual topology shift is extracted after the model is loaded.
        neighbor_target = current.copy()

    feature_std = train[list(FEATURES)].to_numpy(dtype=np.float32).std(axis=0, ddof=0)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0).astype(np.float32)
    plus_ids = plus["example_id"].astype(str).to_numpy()
    minus_ids = minus["example_id"].astype(str).to_numpy()
    rows = eval_df.copy()
    rows["base_response_text"] = rows["response_text"].astype(str)
    rows["base_judge_label"] = rows["judge_label"].astype(str)
    rows["target_mode"] = target_mode
    rows["retrieval_geometry"] = str(retrieval_geometry)
    rows["neighbor_k"] = int(k)
    rows["neighbor_k_effective"] = int(effective_k)
    rows["plus_neighbor_ids"] = ["|".join(plus_ids[index].tolist()) for index in plus_idx]
    rows["minus_neighbor_ids"] = ["|".join(minus_ids[index].tolist()) for index in minus_idx]
    rows["plus_topology_distance_mean"] = plus_dist.mean(axis=1)
    rows["minus_topology_distance_mean"] = minus_dist.mean(axis=1)
    if classifier_current_score is not None and classifier_target_score is not None:
        rows["classifier_current_score"] = classifier_current_score
        rows["classifier_target_score"] = classifier_target_score
        rows["classifier_target_quantile"] = float(classifier_target_quantile)
        rows["classifier_raw_intercept"] = float(classifier_raw_intercept)
        for feature_index, feature in enumerate(FEATURES):
            rows[f"classifier_raw_weight__{feature}"] = float(
                classifier_raw_weight[feature_index]
            )
    for feature_index, feature in enumerate(FEATURES):
        rows[f"current__{feature}"] = current[:, feature_index]
        rows[f"neighbor_target__{feature}"] = neighbor_target[:, feature_index]
    return rows, current, neighbor_target, feature_std, plus, minus


def _load_or_extract_pool_vectors(
    *,
    cache_path: Path,
    bundle: Any,
    config: dict[str, Any],
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    layer: int,
    readout: str,
    force: bool,
) -> dict[str, np.ndarray]:
    expected_ids = plus["example_id"].astype(str).tolist() + minus["example_id"].astype(str).tolist()
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    ensure_dir(cache_path.parent)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            if payload.get("example_ids") == expected_ids:
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
                return {example_id: vectors[index] for index, example_id in enumerate(expected_ids)}

        pool = pd.concat([plus, minus], ignore_index=True)
        prompt_suffix = str(config["steering"].get("prompt_suffix", ""))
        prompts = _prompt_texts(
            bundle,
            pool["text"].astype(str).tolist(),
            config["generation"],
            prompt_suffix,
        )
        vectors = _extract_prompt_vectors(
            bundle,
            prompts,
            config["extraction"],
            int(layer),
            readout=str(readout),
        ).astype(np.float32)
        joblib.dump(
            {"example_ids": expected_ids, "vectors": vectors, "readout": str(readout)},
            cache_path,
        )
    return {example_id: vectors[index] for index, example_id in enumerate(expected_ids)}


def _load_or_extract_contrastive_differences(
    *,
    cache_path: Path,
    bundle: Any,
    config: dict[str, Any],
    pool: pd.DataFrame,
    layer: int,
    positive_instruction: str,
    negative_instruction: str,
    readout: str,
    force: bool,
) -> dict[str, np.ndarray]:
    """Extract paired clarification-minus-direct prompt vectors for identical questions."""
    pool = pool.drop_duplicates("example_id").reset_index(drop=True)
    expected_ids = pool["example_id"].astype(str).tolist()
    instruction_digest = hashlib.sha256(
        f"{positive_instruction}\0{negative_instruction}".encode("utf-8")
    ).hexdigest()[:12]
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    ensure_dir(cache_path.parent)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            if (
                payload.get("example_ids") == expected_ids
                and payload.get("instruction_digest") == instruction_digest
            ):
                differences = np.asarray(payload["differences"], dtype=np.float32)
                return {
                    example_id: differences[index]
                    for index, example_id in enumerate(expected_ids)
                }

        prompt_suffix = str(config["steering"].get("prompt_suffix", ""))
        texts = pool["text"].astype(str).tolist()
        positive_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{positive_instruction}",
        )
        negative_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{negative_instruction}",
        )
        positive_vectors = _extract_prompt_vectors(
            bundle,
            positive_prompts,
            config["extraction"],
            int(layer),
            readout=str(readout),
        ).astype(np.float32)
        negative_vectors = _extract_prompt_vectors(
            bundle,
            negative_prompts,
            config["extraction"],
            int(layer),
            readout=str(readout),
        ).astype(np.float32)
        differences = positive_vectors - negative_vectors
        joblib.dump(
            {
                "example_ids": expected_ids,
                "instruction_digest": instruction_digest,
                "positive_instruction": positive_instruction,
                "negative_instruction": negative_instruction,
                "readout": str(readout),
                "differences": differences,
            },
            cache_path,
        )
    return {example_id: differences[index] for index, example_id in enumerate(expected_ids)}


def _load_or_extract_self_contrastive_state(
    *,
    cache_path: Path,
    bundle: Any,
    config: dict[str, Any],
    rows: pd.DataFrame,
    layer: int,
    readout: str,
    reducer: Any,
    pca_components: int,
    max_length: int,
    positive_instruction: str,
    negative_instruction: str,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract paired hidden directions and exact H0 statistics for each held-out query."""
    pool = rows.drop_duplicates("example_id").reset_index(drop=True)
    expected_ids = pool["example_id"].astype(str).tolist()
    instruction_digest = hashlib.sha256(
        f"{positive_instruction}\0{negative_instruction}".encode("utf-8")
    ).hexdigest()[:12]
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    ensure_dir(cache_path.parent)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            if (
                payload.get("example_ids") == expected_ids
                and payload.get("instruction_digest") == instruction_digest
                and payload.get("readout") == str(readout)
            ):
                differences = np.asarray(payload["differences"], dtype=np.float32)
                positive_features = np.asarray(payload["positive_features"], dtype=np.float32)
                negative_features = np.asarray(payload["negative_features"], dtype=np.float32)
                return (
                    {example_id: differences[index] for index, example_id in enumerate(expected_ids)},
                    {
                        example_id: positive_features[index]
                        for index, example_id in enumerate(expected_ids)
                    },
                    {
                        example_id: negative_features[index]
                        for index, example_id in enumerate(expected_ids)
                    },
                )

        prompt_suffix = str(config["steering"].get("prompt_suffix", ""))
        texts = pool["text"].astype(str).tolist()
        positive_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{positive_instruction}",
        )
        negative_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{negative_instruction}",
        )
        positive_vectors = _extract_prompt_vectors(
            bundle,
            positive_prompts,
            config["extraction"],
            int(layer),
            readout=str(readout),
        ).astype(np.float32)
        negative_vectors = _extract_prompt_vectors(
            bundle,
            negative_prompts,
            config["extraction"],
            int(layer),
            readout=str(readout),
        ).astype(np.float32)
        differences = positive_vectors - negative_vectors

        cloud_frame = pool.copy()
        cloud_frame["judge_label"] = "COUNTERFACTUAL"
        cloud_frame["behavior_label"] = 0
        reducers = {int(layer): reducer}
        positive_clouds = topo3._extract_reduced_clouds(
            bundle=bundle,
            frame=cloud_frame,
            rendered_prompts=positive_prompts,
            layers=[int(layer)],
            reducers=reducers,
            batch_size=int(config["extraction"]["batch_size"]),
            max_length=int(max_length),
            topology_dim=int(pca_components),
        )
        negative_clouds = topo3._extract_reduced_clouds(
            bundle=bundle,
            frame=cloud_frame,
            rendered_prompts=negative_prompts,
            layers=[int(layer)],
            reducers=reducers,
            batch_size=int(config["extraction"]["batch_size"]),
            max_length=int(max_length),
            topology_dim=int(pca_components),
        )

        def exact_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
            return np.stack(
                [
                    exact._exact_features_tensor(
                        torch.as_tensor(np.asarray(cloud), dtype=torch.float64)
                    ).numpy()
                    for cloud in frame["cloud"]
                ]
            ).astype(np.float32, copy=False)

        positive_features = exact_feature_matrix(positive_clouds)
        negative_features = exact_feature_matrix(negative_clouds)
        joblib.dump(
            {
                "example_ids": expected_ids,
                "instruction_digest": instruction_digest,
                "readout": str(readout),
                "positive_instruction": positive_instruction,
                "negative_instruction": negative_instruction,
                "differences": differences,
                "positive_features": positive_features,
                "negative_features": negative_features,
            },
            cache_path,
        )
    return (
        {example_id: differences[index] for index, example_id in enumerate(expected_ids)},
        {example_id: positive_features[index] for index, example_id in enumerate(expected_ids)},
        {example_id: negative_features[index] for index, example_id in enumerate(expected_ids)},
    )


def _extract_hidden_token_sequences(
    *,
    bundle: Any,
    prompts: list[str],
    layer: int,
    batch_size: int,
    max_length: int,
) -> list[np.ndarray]:
    """Extract non-special prompt-token states from one decoder layer."""
    tokenizer = bundle.tokenizer
    special_ids = {
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if token_id is not None
    }
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    sequences: list[np.ndarray] = []
    try:
        for start in range(0, len(prompts), max(1, int(batch_size))):
            encoded = tokenizer(
                prompts[start : start + int(batch_size)],
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            input_ids_cpu = encoded["input_ids"].detach().cpu()
            attention_mask_cpu = encoded["attention_mask"].detach().cpu()
            model_inputs = {key: value.to(bundle.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = bundle.model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            layer_output = hidden_states[int(layer) + 1].detach().float().cpu()
            for row_index in range(layer_output.shape[0]):
                valid = topo3._valid_token_mask(
                    input_ids_cpu[row_index],
                    attention_mask_cpu[row_index],
                    special_ids=special_ids,
                )
                sequences.append(
                    layer_output[row_index][valid].numpy().astype(np.float32, copy=False)
                )
            del outputs, hidden_states, layer_output, model_inputs
            if bundle.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return sequences


def _load_or_extract_self_contrastive_token_sequences(
    *,
    cache_path: Path,
    bundle: Any,
    config: dict[str, Any],
    rows: pd.DataFrame,
    layer: int,
    max_length: int,
    positive_instruction: str,
    negative_instruction: str,
    force: bool,
) -> dict[str, np.ndarray]:
    """Extract a clarification-minus-direct hidden vector for every prompt token."""
    pool = rows.drop_duplicates("example_id").reset_index(drop=True)
    expected_ids = pool["example_id"].astype(str).tolist()
    instruction_digest = hashlib.sha256(
        f"{positive_instruction}\0{negative_instruction}".encode("utf-8")
    ).hexdigest()[:12]
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    ensure_dir(cache_path.parent)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            if (
                payload.get("example_ids") == expected_ids
                and payload.get("instruction_digest") == instruction_digest
            ):
                values = [
                    np.asarray(value, dtype=np.float32)
                    for value in payload["token_differences"]
                ]
                return dict(zip(expected_ids, values, strict=True))

        prompt_suffix = str(config["steering"].get("prompt_suffix", ""))
        texts = pool["text"].astype(str).tolist()
        positive_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{positive_instruction}",
        )
        negative_prompts = _prompt_texts(
            bundle,
            texts,
            config["generation"],
            f"{prompt_suffix}\n\n{negative_instruction}",
        )
        extraction_batch_size = int(config["extraction"]["batch_size"])
        positive_sequences = _extract_hidden_token_sequences(
            bundle=bundle,
            prompts=positive_prompts,
            layer=int(layer),
            batch_size=extraction_batch_size,
            max_length=int(max_length),
        )
        negative_sequences = _extract_hidden_token_sequences(
            bundle=bundle,
            prompts=negative_prompts,
            layer=int(layer),
            batch_size=extraction_batch_size,
            max_length=int(max_length),
        )
        token_differences: list[np.ndarray] = []
        for positive, negative in zip(positive_sequences, negative_sequences, strict=True):
            common_count = max(len(positive), len(negative))
            shape_reference = np.zeros((common_count, 1), dtype=np.float32)
            aligned_positive = _position_cloud_match(shape_reference, positive)
            aligned_negative = _position_cloud_match(shape_reference, negative)
            token_differences.append(
                (aligned_positive - aligned_negative).astype(np.float32, copy=False)
            )
        joblib.dump(
            {
                "example_ids": expected_ids,
                "instruction_digest": instruction_digest,
                "positive_instruction": positive_instruction,
                "negative_instruction": negative_instruction,
                "token_differences": token_differences,
            },
            cache_path,
        )
    return dict(zip(expected_ids, token_differences, strict=True))


def _load_or_extract_pool_token_sequences(
    *,
    cache_path: Path,
    bundle: Any,
    config: dict[str, Any],
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    layer: int,
    max_length: int,
    force: bool,
) -> dict[str, np.ndarray]:
    """Extract full layer-output token sequences for the observed D+ and D- pools."""
    pool = pd.concat([plus, minus], ignore_index=True).drop_duplicates("example_id")
    pool = pool.reset_index(drop=True)
    expected_ids = pool["example_id"].astype(str).tolist()
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    ensure_dir(cache_path.parent)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            if payload.get("example_ids") == expected_ids:
                sequences = [
                    np.asarray(value, dtype=np.float32)
                    for value in payload["token_sequences"]
                ]
                return dict(zip(expected_ids, sequences, strict=True))

        prompt_suffix = str(config["steering"].get("prompt_suffix", ""))
        prompts = _prompt_texts(
            bundle,
            pool["text"].astype(str).tolist(),
            config["generation"],
            prompt_suffix,
        )
        sequences = _extract_hidden_token_sequences(
            bundle=bundle,
            prompts=prompts,
            layer=int(layer),
            batch_size=int(config["extraction"]["batch_size"]),
            max_length=int(max_length),
        )
        joblib.dump(
            {
                "example_ids": expected_ids,
                "layer": int(layer),
                "token_sequences": sequences,
            },
            cache_path,
        )
    return dict(zip(expected_ids, sequences, strict=True))


def _local_directions(rows: pd.DataFrame, vector_by_id: dict[str, np.ndarray]) -> np.ndarray:
    directions: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        plus_ids = str(row.plus_neighbor_ids).split("|")
        minus_ids = str(row.minus_neighbor_ids).split("|")
        plus = np.stack([vector_by_id[example_id] for example_id in plus_ids])
        minus = np.stack([vector_by_id[example_id] for example_id in minus_ids])
        directions.append(plus.mean(axis=0) - minus.mean(axis=0))
    return np.stack(directions).astype(np.float32, copy=False)


def _local_direction_bases(
    rows: pd.DataFrame,
    vector_by_id: dict[str, np.ndarray],
    rank: int,
) -> list[np.ndarray]:
    """Build per-query orthonormal bases from local grounded-direct contrasts."""
    bases: list[np.ndarray] = []
    requested_rank = max(1, int(rank))
    for row in rows.itertuples(index=False):
        plus_ids = str(row.plus_neighbor_ids).split("|")
        minus_ids = str(row.minus_neighbor_ids).split("|")
        plus = np.stack([vector_by_id[example_id] for example_id in plus_ids]).astype(np.float64)
        minus = np.stack([vector_by_id[example_id] for example_id in minus_ids]).astype(np.float64)
        contrasts = plus - minus
        mean_direction = contrasts.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_direction))
        basis_rows: list[np.ndarray] = []
        if mean_norm > 1e-12:
            first = mean_direction / mean_norm
            basis_rows.append(first)
            residuals = contrasts - (contrasts @ first)[:, None] * first[None, :]
        else:
            residuals = contrasts
        if len(basis_rows) < requested_rank and np.linalg.norm(residuals) > 1e-12:
            _u, singular_values, vh = np.linalg.svd(residuals, full_matrices=False)
            valid = int(np.sum(singular_values > 1e-10))
            for candidate in vh[:valid]:
                if len(basis_rows) >= requested_rank:
                    break
                basis_rows.append(candidate)
        if not basis_rows:
            basis_rows = [np.zeros(contrasts.shape[1], dtype=np.float64)]
        bases.append(np.stack(basis_rows).astype(np.float32, copy=False))
    return bases


def _selected_neighbor_ids(row: Any) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *str(row.plus_neighbor_ids).split("|"),
                *str(row.minus_neighbor_ids).split("|"),
            ]
        )
    )


def _local_paired_directions(
    rows: pd.DataFrame,
    difference_by_id: dict[str, np.ndarray],
) -> np.ndarray:
    directions = []
    for row in rows.itertuples(index=False):
        local = np.stack(
            [difference_by_id[example_id] for example_id in _selected_neighbor_ids(row)]
        )
        directions.append(local.mean(axis=0))
    return np.stack(directions).astype(np.float32, copy=False)


def _local_paired_direction_bases(
    rows: pd.DataFrame,
    difference_by_id: dict[str, np.ndarray],
    rank: int,
) -> list[np.ndarray]:
    """Build local behavior bases from paired instruction-state differences."""
    bases: list[np.ndarray] = []
    requested_rank = max(1, int(rank))
    for row in rows.itertuples(index=False):
        differences = np.stack(
            [difference_by_id[example_id] for example_id in _selected_neighbor_ids(row)]
        ).astype(np.float64)
        mean_direction = differences.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_direction))
        basis_rows: list[np.ndarray] = []
        if mean_norm > 1e-12:
            first = mean_direction / mean_norm
            basis_rows.append(first)
            residuals = differences - (differences @ first)[:, None] * first[None, :]
        else:
            residuals = differences
        if len(basis_rows) < requested_rank and np.linalg.norm(residuals) > 1e-12:
            _u, singular_values, vh = np.linalg.svd(residuals, full_matrices=False)
            valid = int(np.sum(singular_values > 1e-10))
            basis_rows.extend(vh[: min(valid, requested_rank - len(basis_rows))])
        if not basis_rows:
            basis_rows = [np.zeros(differences.shape[1], dtype=np.float64)]
        bases.append(np.stack(basis_rows).astype(np.float32, copy=False))
    return bases


def _self_paired_directions_and_bases(
    rows: pd.DataFrame,
    difference_by_id: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Use each held-out question's paired clarification-direct state difference."""
    directions = np.stack(
        [difference_by_id[str(example_id)] for example_id in rows["example_id"]]
    ).astype(np.float32, copy=False)
    bases: list[np.ndarray] = []
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            bases.append(np.zeros((1, direction.shape[0]), dtype=np.float32))
        else:
            bases.append((direction / norm)[None, :].astype(np.float32, copy=False))
    return directions, bases


def _self_tokenwise_directions_and_templates(
    rows: pd.DataFrame,
    difference_by_id: dict[str, np.ndarray],
    token_counts: list[int],
    *,
    readout: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Resample paired token differences onto each unmodified prompt cloud."""
    directions: list[np.ndarray] = []
    templates: list[np.ndarray] = []
    for example_id, token_count in zip(rows["example_id"], token_counts, strict=True):
        sequence = np.asarray(difference_by_id[str(example_id)], dtype=np.float32)
        shape_reference = np.zeros((int(token_count), 1), dtype=np.float32)
        template = _position_cloud_match(shape_reference, sequence)
        template_rms = float(np.linalg.norm(template) / np.sqrt(max(len(template), 1)))
        if template_rms > 1e-12:
            template = template / template_rms
        if str(readout) == "last_token":
            direction = sequence[-1]
        elif str(readout) == "mean_pool":
            direction = sequence.mean(axis=0)
        else:
            raise ValueError(f"Unsupported tokenwise direction readout: {readout}")
        directions.append(np.asarray(direction, dtype=np.float32))
        templates.append(template.astype(np.float32, copy=False))
    return np.stack(directions), templates


def _local_neighbor_tokenwise_directions_and_templates(
    rows: pd.DataFrame,
    sequence_by_id: dict[str, np.ndarray],
    token_counts: list[int],
    *,
    readout: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build aligned D+ minus D- token templates from topology-matched neighbors."""
    directions: list[np.ndarray] = []
    templates: list[np.ndarray] = []
    for row, token_count in zip(rows.itertuples(index=False), token_counts, strict=True):
        shape_reference = np.zeros((int(token_count), 1), dtype=np.float32)
        plus = np.stack(
            [
                _position_cloud_match(shape_reference, sequence_by_id[example_id])
                for example_id in str(row.plus_neighbor_ids).split("|")
            ]
        )
        minus = np.stack(
            [
                _position_cloud_match(shape_reference, sequence_by_id[example_id])
                for example_id in str(row.minus_neighbor_ids).split("|")
            ]
        )
        template = plus.mean(axis=0) - minus.mean(axis=0)
        if str(readout) == "last_token":
            direction = template[-1]
        elif str(readout) == "mean_pool":
            direction = template.mean(axis=0)
        else:
            raise ValueError(f"Unsupported tokenwise direction readout: {readout}")
        template_rms = float(np.linalg.norm(template) / np.sqrt(max(len(template), 1)))
        if template_rms > 1e-12:
            template = template / template_rms
        directions.append(np.asarray(direction, dtype=np.float32))
        templates.append(template.astype(np.float32, copy=False))
    return np.stack(directions), templates


def _nearest_cloud_match(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if len(query) == 0 or len(reference) == 0:
        return np.zeros_like(query)
    squared_distances = (
        np.sum(query**2, axis=1, keepdims=True)
        + np.sum(reference**2, axis=1)[None, :]
        - 2.0 * query @ reference.T
    )
    nearest = np.argmin(squared_distances, axis=1)
    return reference[nearest]


def _position_cloud_match(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Interpolate an ordered reference token cloud to the query token count."""
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if len(query) == 0 or len(reference) == 0:
        return np.zeros_like(query)
    if len(reference) == 1:
        return np.repeat(reference, len(query), axis=0)
    source_positions = np.linspace(0.0, 1.0, len(reference))
    target_positions = np.linspace(0.0, 1.0, len(query))
    return np.stack(
        [
            np.interp(target_positions, source_positions, reference[:, dimension])
            for dimension in range(reference.shape[1])
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _topology_decode_vector(
    token_deltas: np.ndarray,
    *,
    mode: str,
    scale: float,
    suffix_fraction: float,
) -> np.ndarray:
    """Reduce a token-specific prompt deformation to one decode-time vector."""
    values = np.asarray(token_deltas, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden] topology deltas, got {values.shape}")
    if mode == "none" or len(values) == 0:
        return np.zeros(values.shape[-1], dtype=np.float32)
    if mode == "last_token":
        vector = values[-1]
    elif mode == "suffix_mean":
        fraction = min(max(float(suffix_fraction), 0.0), 1.0)
        suffix_count = max(1, int(np.ceil(max(fraction, 1.0 / len(values)) * len(values))))
        vector = values[-suffix_count:].mean(axis=0)
    else:
        raise ValueError(f"Unknown topology decode mode: {mode}")
    return (float(scale) * vector).astype(np.float32, copy=False)


def _scaled_shared_vectors(
    directions: np.ndarray,
    hidden_norms: np.ndarray,
    token_counts: np.ndarray,
    *,
    mean_alpha: float | None,
    target_ratio: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale local directions by a raw alpha or an exact prompt-relative norm target."""
    values = np.asarray(directions, dtype=np.float32)
    direction_norms = np.linalg.norm(values, axis=1)
    if target_ratio is None:
        scales = np.full(len(values), float(mean_alpha or 0.0), dtype=np.float32)
    else:
        denominators = np.sqrt(np.maximum(token_counts, 1)) * np.maximum(direction_norms, 1e-12)
        scales = (
            float(target_ratio) * np.asarray(hidden_norms, dtype=np.float64) / denominators
        ).astype(np.float32)
    return (scales[:, None] * values).astype(np.float32, copy=False), scales


def _local_transport_templates(
    rows: pd.DataFrame,
    cloud_by_id: pd.DataFrame,
    match_mode: str = "nearest",
) -> list[np.ndarray]:
    """Build centered token-wise D+ minus D- cloud transport templates."""
    matcher = _nearest_cloud_match if match_mode == "nearest" else _position_cloud_match
    templates: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        query = np.asarray(cloud_by_id.loc[str(row.example_id), "cloud"], dtype=np.float32)
        plus_matches = np.stack(
            [
                matcher(
                    query,
                    np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32),
                )
                for example_id in str(row.plus_neighbor_ids).split("|")
            ]
        )
        minus_matches = np.stack(
            [
                matcher(
                    query,
                    np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32),
                )
                for example_id in str(row.minus_neighbor_ids).split("|")
            ]
        )
        template = plus_matches.mean(axis=0) - minus_matches.mean(axis=0)
        template -= template.mean(axis=0, keepdims=True)
        templates.append(template.astype(np.float32, copy=False))
    return templates


def _generate_with_hybrid_deltas(
    *,
    bundle: Any,
    config: dict[str, Any],
    rows: pd.DataFrame,
    topology_deltas: list[np.ndarray],
    shared_vectors: np.ndarray,
    layer: int,
    max_length: int,
    apply_on: str,
    shared_intervention_site: str,
    topology_decode_mode: str,
    topology_decode_scale: float,
    topology_decode_suffix_fraction: float,
    reducer: Any,
) -> tuple[list[str], np.ndarray]:
    tokenizer = bundle.tokenizer
    device = bundle.device
    target_layer = topo3._decoder_layers(bundle.model)[int(layer)]
    generation_cfg = dict(config["generation"])
    batch_size = int(generation_cfg.get("batch_size", 8))
    special_ids = set(
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if token_id is not None
    )
    responses: list[str] = []
    verified_features: list[np.ndarray] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="exact_h0_hybrid_generate", leave=False):
        batch_rows = rows.iloc[start : start + batch_size].reset_index(drop=True)
        batch_topology = topology_deltas[start : start + len(batch_rows)]
        batch_shared = shared_vectors[start : start + len(batch_rows)]
        rendered = topo3._rendered_prompts(bundle, config, batch_rows["text"].astype(str).tolist())
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenizer.padding_side = original_padding_side
        input_ids_cpu = encoded["input_ids"].detach().cpu()
        attention_mask_cpu = encoded["attention_mask"].detach().cpu()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        dtype = next(bundle.model.parameters()).dtype
        shared = torch.as_tensor(batch_shared, device=device, dtype=dtype)[:, None, :]
        topology_decode = torch.as_tensor(
            np.stack(
                [
                    _topology_decode_vector(
                        value,
                        mode=topology_decode_mode,
                        scale=topology_decode_scale,
                        suffix_fraction=topology_decode_suffix_fraction,
                    )
                    for value in batch_topology
                ]
            ),
            device=device,
            dtype=dtype,
        )[:, None, :]
        shared_prompt_delta = encoded["attention_mask"].unsqueeze(-1).to(dtype=dtype) * shared
        topology_prompt_delta = torch.zeros_like(shared_prompt_delta)
        valid_indices_by_row: list[list[int]] = []
        for row_index, centered_delta in enumerate(batch_topology):
            valid = topo3._valid_token_mask(
                input_ids_cpu[row_index],
                attention_mask_cpu[row_index],
                special_ids=special_ids,
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
            if len(valid_indices) != len(centered_delta):
                if len(valid_indices) < len(centered_delta):
                    centered_delta = centered_delta[-len(valid_indices) :]
                else:
                    valid_indices = valid_indices[-len(centered_delta) :]
            valid_indices_by_row.append(valid_indices)
            topology_prompt_delta[row_index, valid_indices, :] += torch.as_tensor(
                centered_delta,
                device=device,
                dtype=dtype,
            )
        output_prefill_applied = False
        input_prefill_applied = False
        output_prefill_captured = False

        def apply_output_delta(hidden_states: torch.Tensor) -> torch.Tensor:
            nonlocal output_prefill_applied
            if hidden_states.shape[:2] == topology_prompt_delta.shape[:2] and not output_prefill_applied:
                output_prefill_applied = True
                active = topology_prompt_delta
                if shared_intervention_site == "layer_output":
                    active = active + shared_prompt_delta
                return hidden_states + active.to(dtype=hidden_states.dtype)
            if apply_on == "prompt_and_decode_shared" and output_prefill_applied:
                active = topology_decode
                if shared_intervention_site == "layer_output":
                    active = active + shared
                return hidden_states + active.to(dtype=hidden_states.dtype).expand(
                    hidden_states.shape[0], hidden_states.shape[1], -1
                )
            return hidden_states

        def apply_input_delta(hidden_states: torch.Tensor) -> torch.Tensor:
            nonlocal input_prefill_applied
            if hidden_states.shape[:2] == shared_prompt_delta.shape[:2] and not input_prefill_applied:
                input_prefill_applied = True
                return hidden_states + shared_prompt_delta.to(dtype=hidden_states.dtype)
            if apply_on == "prompt_and_decode_shared" and input_prefill_applied:
                return hidden_states + shared.to(dtype=hidden_states.dtype).expand(
                    hidden_states.shape[0], hidden_states.shape[1], -1
                )
            return hidden_states

        def output_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            nonlocal output_prefill_captured
            if isinstance(output, tuple):
                patched = apply_output_delta(output[0].clone())
                result: Any = (patched,) + output[1:]
            else:
                patched = apply_output_delta(output.clone())
                result = patched
            if patched.shape[:2] == topology_prompt_delta.shape[:2] and not output_prefill_captured:
                output_prefill_captured = True
                patched_cpu = patched.detach().float().cpu()
                for row_index, valid_indices in enumerate(valid_indices_by_row):
                    tokens = patched_cpu[row_index, valid_indices, :].numpy()
                    cloud = reducer.transform(tokens)[:, : int(reducer.n_components_)]
                    values = exact._exact_features_tensor(
                        torch.as_tensor(cloud, dtype=torch.float64)
                    )
                    verified_features.append(values.numpy().astype(np.float32, copy=False))
            return result

        def input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs or not torch.is_tensor(inputs[0]):
                return inputs
            return (apply_input_delta(inputs[0].clone()),) + inputs[1:]

        handles = [target_layer.register_forward_hook(output_hook)]
        if shared_intervention_site == "layer_input":
            handles.append(target_layer.register_forward_pre_hook(input_hook))
        try:
            kwargs = topo3._build_generate_kwargs(generation_cfg, return_entropy=False)
            kwargs["pad_token_id"] = tokenizer.pad_token_id
            kwargs["eos_token_id"] = tokenizer.eos_token_id
            with torch.no_grad():
                output = bundle.model.generate(**encoded, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        prompt_length = encoded["input_ids"].shape[1]
        for row_index in range(len(rendered)):
            generated_ids = output.sequences[row_index, prompt_length:]
            responses.append(tokenizer.decode(generated_ids.detach().cpu(), skip_special_tokens=True).strip())
        del encoded, output, shared_prompt_delta, topology_prompt_delta, topology_decode, shared
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return responses, np.stack(verified_features).astype(np.float32, copy=False)


def _zero_diagnostics(
    clouds: list[np.ndarray],
    current: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    rows = []
    deltas = []
    for index, (cloud, values, hidden_norm) in enumerate(zip(clouds, current, hidden_norms, strict=True)):
        deltas.append(np.zeros_like(cloud, dtype=np.float32))
        rows.append(
            {
                "row_index": index,
                "optimization_status": "zero_topology_alpha",
                "accepted_steps": 0,
                "line_search_backtracks": 0,
                "initial_normalized_target_error": 0.0,
                "final_normalized_target_error": 0.0,
                "initial_objective": 0.0,
                "final_objective": 0.0,
                "pca_delta_norm": 0.0,
                "pca_cloud_norm": float(np.linalg.norm(cloud)),
                "relative_hidden_delta_norm": 0.0,
                "delta_token_mean_norm": 0.0,
                "zero_mean_constraint": True,
                "trust_radius": 0.0,
                **{f"initial_exact__{feature}": float(values[i]) for i, feature in enumerate(FEATURES)},
                **{f"final_exact__{feature}": float(values[i]) for i, feature in enumerate(FEATURES)},
                **{f"optimization_target__{feature}": float(values[i]) for i, feature in enumerate(FEATURES)},
            }
        )
    return deltas, pd.DataFrame(rows)


def _optimize_behavior_rank1_one(
    *,
    cloud_np: np.ndarray,
    target_np: np.ndarray,
    feature_std_np: np.ndarray,
    hidden_norm: float,
    direction_np: np.ndarray,
    components_np: np.ndarray,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    gn_steps: int,
    line_search_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimize centered per-token gates along one local behavior direction."""
    dtype = torch.float64
    base = torch.as_tensor(np.asarray(cloud_np, dtype=np.float64), dtype=dtype)
    target = torch.as_tensor(np.asarray(target_np, dtype=np.float64), dtype=dtype)
    feature_std = torch.as_tensor(np.asarray(feature_std_np, dtype=np.float64), dtype=dtype).clamp_min(1e-8)
    direction = np.asarray(direction_np, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 1e-12:
        direction_unit = direction / direction_norm
    else:
        direction_unit = np.zeros_like(direction)
    projected = direction_unit @ np.asarray(components_np, dtype=np.float64).T
    projection_ratio = float(np.linalg.norm(projected))
    projected_t = torch.as_tensor(projected, dtype=dtype)
    gates = torch.zeros(base.shape[0], dtype=dtype)
    trust_radius = float(trust_ratio) * max(float(hidden_norm), 1e-12)

    def cloud_from_gates(candidate: torch.Tensor) -> torch.Tensor:
        return base + candidate[:, None] * projected_t[None, :]

    def evaluate(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = exact._exact_features_tensor(cloud_from_gates(candidate))
        residual = (values - target) / feature_std
        relative_norm_sq = candidate.pow(2).sum() / max(float(hidden_norm) ** 2, 1e-12)
        objective = residual.pow(2).sum() + float(lambda_value) * relative_norm_sq
        return values, residual, objective

    initial_values, initial_residual, initial_objective = evaluate(gates)
    objective = initial_objective
    accepted_steps = 0
    total_backtracks = 0
    status = "max_steps"
    if direction_norm <= 1e-12 or projection_ratio <= 1e-8:
        status = "degenerate_behavior_direction"
    else:
        for _iteration in range(int(gn_steps)):
            current_gates = gates.detach().requires_grad_(True)
            current_cloud = cloud_from_gates(current_gates)
            edges = exact.soft_h0._mst_edges_from_cloud(current_cloud)

            def normalized_fixed_tree_features(candidate: torch.Tensor) -> torch.Tensor:
                cloud = cloud_from_gates(candidate)
                values = exact.soft_h0._hard_mst_features_one(cloud, edges, top_k=5)
                return values / feature_std

            values = exact.soft_h0._hard_mst_features_one(current_cloud, edges, top_k=5)
            residual = values / feature_std - target / feature_std
            jacobian = torch.autograd.functional.jacobian(
                normalized_fixed_tree_features,
                current_gates,
                vectorize=True,
            ).reshape(len(FEATURES), -1)
            if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
                status = "nonfinite_jacobian"
                break
            if float(torch.linalg.norm(residual).detach()) < 1e-5:
                status = "converged"
                break

            step = exact._gauss_newton_step(
                jacobian=jacobian,
                residual=residual.detach(),
                delta=gates,
                hidden_norm=hidden_norm,
                lambda_value=lambda_value,
                damping=damping,
            )
            step = step - step.mean()
            accepted = False
            for backtrack in range(int(line_search_steps)):
                candidate = gates + (0.5**backtrack) * step
                candidate = candidate - candidate.mean()
                candidate_norm = float(torch.linalg.norm(candidate))
                if trust_radius > 0.0 and candidate_norm > trust_radius:
                    candidate = candidate * (trust_radius / max(candidate_norm, 1e-12))
                _candidate_values, _candidate_residual, candidate_objective = evaluate(candidate)
                if torch.isfinite(candidate_objective) and float(candidate_objective) < float(objective) - 1e-10:
                    gates = candidate.detach()
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

    final_values, final_residual, final_objective = evaluate(gates)
    gates_np = gates.numpy().astype(np.float32, copy=False)
    delta_y = (gates_np[:, None] * projected.astype(np.float32)[None, :]).astype(np.float32, copy=False)
    diagnostics: dict[str, Any] = {
        "optimization_status": status,
        "accepted_steps": int(accepted_steps),
        "line_search_backtracks": int(total_backtracks),
        "initial_normalized_target_error": float(torch.linalg.norm(initial_residual)),
        "final_normalized_target_error": float(torch.linalg.norm(final_residual)),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "pca_delta_norm": float(np.linalg.norm(delta_y)),
        "pca_cloud_norm": float(np.linalg.norm(cloud_np)),
        "relative_hidden_delta_norm": float(np.linalg.norm(gates_np) / max(float(hidden_norm), 1e-12)),
        "delta_token_mean_norm": float(abs(gates_np.mean())),
        "zero_mean_constraint": True,
        "trust_radius": float(trust_radius),
        "behavior_direction_norm": direction_norm,
        "behavior_direction_pca_projection_ratio": projection_ratio,
        "token_gate_std": float(gates_np.std()),
        "token_gate_abs_max": float(np.abs(gates_np).max(initial=0.0)),
    }
    for feature_index, feature in enumerate(FEATURES):
        diagnostics[f"initial_exact__{feature}"] = float(initial_values[feature_index])
        diagnostics[f"final_exact__{feature}"] = float(final_values[feature_index])
        diagnostics[f"optimization_target__{feature}"] = float(target[feature_index])
    return gates_np, delta_y, diagnostics


def _optimize_behavior_rank1_many(
    *,
    clouds: list[np.ndarray],
    target_features: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
    directions: np.ndarray,
    components: np.ndarray,
    args: argparse.Namespace,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    tasks = (
        delayed(_optimize_behavior_rank1_one)(
            cloud_np=cloud,
            target_np=target,
            feature_std_np=feature_std,
            hidden_norm=float(hidden_norm),
            direction_np=direction,
            components_np=components,
            lambda_value=float(lambda_value),
            damping=float(damping),
            trust_ratio=float(trust_ratio),
            gn_steps=int(args.gn_steps),
            line_search_steps=int(args.line_search_steps),
        )
        for cloud, target, hidden_norm, direction in zip(
            clouds,
            target_features,
            hidden_norms,
            directions,
            strict=True,
        )
    )
    results = Parallel(n_jobs=max(1, int(args.optimization_jobs)), backend="loky")(
        tqdm(tasks, total=len(clouds), desc="exact_h0_behavior_rank1", leave=False)
    )
    gates = [result[0] for result in results]
    delta_y = [result[1] for result in results]
    diagnostics = pd.DataFrame(
        [{"row_index": index, **result[2]} for index, result in enumerate(results)]
    )
    return gates, delta_y, diagnostics


def _optimize_behavior_tokenwise_one(
    *,
    cloud_np: np.ndarray,
    target_np: np.ndarray,
    feature_std_np: np.ndarray,
    hidden_norm: float,
    template_np: np.ndarray,
    components_np: np.ndarray,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    gn_steps: int,
    line_search_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimize per-token gates along distinct query-specific hidden vectors."""
    dtype = torch.float64
    base = torch.as_tensor(np.asarray(cloud_np, dtype=np.float64), dtype=dtype)
    target = torch.as_tensor(np.asarray(target_np, dtype=np.float64), dtype=dtype)
    feature_std = torch.as_tensor(
        np.asarray(feature_std_np, dtype=np.float64), dtype=dtype
    ).clamp_min(1e-8)
    template = torch.as_tensor(np.asarray(template_np, dtype=np.float64), dtype=dtype)
    projected = template @ torch.as_tensor(
        np.asarray(components_np, dtype=np.float64).T,
        dtype=dtype,
    )
    gates = torch.zeros(base.shape[0], dtype=dtype)
    trust_radius = float(trust_ratio) * max(float(hidden_norm), 1e-12)

    def centered_hidden_delta(candidate: torch.Tensor) -> torch.Tensor:
        raw = candidate[:, None] * template
        return raw - raw.mean(dim=0, keepdim=True)

    def centered_projected_delta(candidate: torch.Tensor) -> torch.Tensor:
        raw = candidate[:, None] * projected
        return raw - raw.mean(dim=0, keepdim=True)

    def cloud_from_gates(candidate: torch.Tensor) -> torch.Tensor:
        return base + centered_projected_delta(candidate)

    def evaluate(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = exact._exact_features_tensor(cloud_from_gates(candidate))
        residual = (values - target) / feature_std
        relative_norm_sq = centered_hidden_delta(candidate).pow(2).sum() / max(
            float(hidden_norm) ** 2,
            1e-12,
        )
        objective = residual.pow(2).sum() + float(lambda_value) * relative_norm_sq
        return values, residual, objective

    initial_values, initial_residual, initial_objective = evaluate(gates)
    objective = initial_objective
    accepted_steps = 0
    total_backtracks = 0
    status = "max_steps"
    template_norm = float(torch.linalg.norm(template))
    projected_norm = float(torch.linalg.norm(projected))
    if template_norm <= 1e-8 or projected_norm <= 1e-8:
        status = "degenerate_tokenwise_template"
    else:
        for _iteration in range(int(gn_steps)):
            current_gates = gates.detach().requires_grad_(True)
            current_cloud = cloud_from_gates(current_gates)
            edges = exact.soft_h0._mst_edges_from_cloud(current_cloud)

            def normalized_fixed_tree_features(candidate: torch.Tensor) -> torch.Tensor:
                cloud = cloud_from_gates(candidate)
                values = exact.soft_h0._hard_mst_features_one(cloud, edges, top_k=5)
                return values / feature_std

            values = exact.soft_h0._hard_mst_features_one(current_cloud, edges, top_k=5)
            residual = values / feature_std - target / feature_std
            jacobian = torch.autograd.functional.jacobian(
                normalized_fixed_tree_features,
                current_gates,
                vectorize=True,
            ).reshape(len(FEATURES), -1)
            if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
                status = "nonfinite_jacobian"
                break
            if float(torch.linalg.norm(residual).detach()) < 1e-5:
                status = "converged"
                break
            step = exact._gauss_newton_step(
                jacobian=jacobian,
                residual=residual.detach(),
                delta=gates,
                hidden_norm=hidden_norm,
                lambda_value=lambda_value,
                damping=damping,
            )
            accepted = False
            for backtrack in range(int(line_search_steps)):
                candidate = gates + (0.5**backtrack) * step
                candidate_norm = float(torch.linalg.norm(centered_hidden_delta(candidate)))
                if trust_radius > 0.0 and candidate_norm > trust_radius:
                    candidate = candidate * (trust_radius / max(candidate_norm, 1e-12))
                _candidate_values, _candidate_residual, candidate_objective = evaluate(candidate)
                if (
                    torch.isfinite(candidate_objective)
                    and float(candidate_objective) < float(objective) - 1e-10
                ):
                    gates = candidate.detach()
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

    final_values, final_residual, final_objective = evaluate(gates)
    gates_np = gates.numpy().astype(np.float32, copy=False)
    hidden_delta_np = centered_hidden_delta(gates).numpy().astype(np.float32, copy=False)
    delta_y = centered_projected_delta(gates).numpy().astype(np.float32, copy=False)
    token_norms = np.linalg.norm(hidden_delta_np, axis=1)
    diagnostics: dict[str, Any] = {
        "optimization_status": status,
        "accepted_steps": int(accepted_steps),
        "line_search_backtracks": int(total_backtracks),
        "initial_normalized_target_error": float(torch.linalg.norm(initial_residual)),
        "final_normalized_target_error": float(torch.linalg.norm(final_residual)),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "pca_delta_norm": float(np.linalg.norm(delta_y)),
        "pca_cloud_norm": float(np.linalg.norm(cloud_np)),
        "relative_hidden_delta_norm": float(
            np.linalg.norm(hidden_delta_np) / max(float(hidden_norm), 1e-12)
        ),
        "delta_token_mean_norm": float(np.linalg.norm(hidden_delta_np.mean(axis=0))),
        "zero_mean_constraint": True,
        "trust_radius": float(trust_radius),
        "tokenwise_template_frobenius": template_norm,
        "tokenwise_template_pca_frobenius": projected_norm,
        "token_gate_rms": float(np.sqrt(np.mean(gates_np**2))),
        "token_gate_abs_max": float(np.abs(gates_np).max(initial=0.0)),
        "token_delta_norm_std": float(token_norms.std()),
    }
    for feature_index, feature in enumerate(FEATURES):
        diagnostics[f"initial_exact__{feature}"] = float(initial_values[feature_index])
        diagnostics[f"final_exact__{feature}"] = float(final_values[feature_index])
        diagnostics[f"optimization_target__{feature}"] = float(target[feature_index])
    return gates_np, delta_y, diagnostics


def _optimize_behavior_tokenwise_many(
    *,
    clouds: list[np.ndarray],
    target_features: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
    templates: list[np.ndarray],
    components: np.ndarray,
    args: argparse.Namespace,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    tasks = (
        delayed(_optimize_behavior_tokenwise_one)(
            cloud_np=cloud,
            target_np=target,
            feature_std_np=feature_std,
            hidden_norm=float(hidden_norm),
            template_np=template,
            components_np=components,
            lambda_value=float(lambda_value),
            damping=float(damping),
            trust_ratio=float(trust_ratio),
            gn_steps=int(args.gn_steps),
            line_search_steps=int(args.line_search_steps),
        )
        for cloud, target, hidden_norm, template in zip(
            clouds,
            target_features,
            hidden_norms,
            templates,
            strict=True,
        )
    )
    results = Parallel(n_jobs=max(1, int(args.optimization_jobs)), backend="loky")(
        tqdm(tasks, total=len(clouds), desc="exact_h0_behavior_tokenwise", leave=False)
    )
    gates = [result[0] for result in results]
    delta_y = [result[1] for result in results]
    diagnostics = pd.DataFrame(
        [{"row_index": index, **result[2]} for index, result in enumerate(results)]
    )
    return gates, delta_y, diagnostics


def _optimize_behavior_lowrank_one(
    *,
    cloud_np: np.ndarray,
    target_np: np.ndarray,
    feature_std_np: np.ndarray,
    hidden_norm: float,
    basis_np: np.ndarray,
    components_np: np.ndarray,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    gn_steps: int,
    line_search_steps: int,
    causal_anchor_ratio: float = 0.0,
    causal_anchor_max_error_increase: float = 0.1,
    causal_anchor_suffix_fraction: float = 0.0,
    causal_position_beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimize centered per-token coefficients in a local behavior subspace."""
    dtype = torch.float64
    base = torch.as_tensor(np.asarray(cloud_np, dtype=np.float64), dtype=dtype)
    target = torch.as_tensor(np.asarray(target_np, dtype=np.float64), dtype=dtype)
    feature_std = torch.as_tensor(np.asarray(feature_std_np, dtype=np.float64), dtype=dtype).clamp_min(1e-8)
    basis = np.asarray(basis_np, dtype=np.float64)
    projected = basis @ np.asarray(components_np, dtype=np.float64).T
    projected_t = torch.as_tensor(projected, dtype=dtype)
    coefficients = torch.zeros((base.shape[0], basis.shape[0]), dtype=dtype)
    normalized_positions = torch.linspace(0.0, 1.0, base.shape[0], dtype=dtype)
    position_weights = torch.exp(
        max(float(causal_position_beta), 0.0) * (normalized_positions - 1.0)
    ).clamp_min(1e-3)
    trust_radius = float(trust_ratio) * max(float(hidden_norm), 1e-12)
    projected_singular_values = np.linalg.svd(projected, compute_uv=False)

    def cloud_from_coefficients(candidate: torch.Tensor) -> torch.Tensor:
        return base + candidate @ projected_t

    def evaluate(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = exact._exact_features_tensor(cloud_from_coefficients(candidate))
        residual = (values - target) / feature_std
        weighted_candidate = candidate / position_weights[:, None]
        relative_norm_sq = weighted_candidate.pow(2).sum() / max(float(hidden_norm) ** 2, 1e-12)
        objective = residual.pow(2).sum() + float(lambda_value) * relative_norm_sq
        return values, residual, objective

    initial_values, initial_residual, initial_objective = evaluate(coefficients)
    objective = initial_objective
    accepted_steps = 0
    total_backtracks = 0
    status = "max_steps"
    if not np.isfinite(projected).all() or float(np.linalg.norm(projected)) <= 1e-8:
        status = "degenerate_behavior_subspace"
    else:
        for _iteration in range(int(gn_steps)):
            current_coefficients = coefficients.detach().requires_grad_(True)
            current_cloud = cloud_from_coefficients(current_coefficients)
            edges = exact.soft_h0._mst_edges_from_cloud(current_cloud)

            def normalized_fixed_tree_features(candidate: torch.Tensor) -> torch.Tensor:
                cloud = cloud_from_coefficients(candidate)
                values = exact.soft_h0._hard_mst_features_one(cloud, edges, top_k=5)
                return values / feature_std

            values = exact.soft_h0._hard_mst_features_one(current_cloud, edges, top_k=5)
            residual = values / feature_std - target / feature_std
            jacobian = torch.autograd.functional.jacobian(
                normalized_fixed_tree_features,
                current_coefficients,
                vectorize=True,
            ).reshape(len(FEATURES), -1)
            if not torch.isfinite(jacobian).all() or not torch.isfinite(residual).all():
                status = "nonfinite_jacobian"
                break
            if float(torch.linalg.norm(residual).detach()) < 1e-5:
                status = "converged"
                break
            flat_coefficients = coefficients.reshape(-1)
            flat_weights_sq = position_weights[:, None].expand_as(coefficients).reshape(-1).pow(2)
            hidden_norm_sq = max(float(hidden_norm) ** 2, 1e-12)
            relative_reg = float(lambda_value) / hidden_norm_sq
            ridge = max(float(damping) + relative_reg, 1e-10)
            inverse_weighted_coefficients = flat_coefficients / flat_weights_sq
            gradient = jacobian.T @ residual.detach() + relative_reg * inverse_weighted_coefficients
            inverse_ridge_gradient = flat_weights_sq * gradient / ridge
            inverse_ridge_jacobian_t = flat_weights_sq[:, None] * jacobian.T / ridge
            feature_system = torch.eye(len(FEATURES), dtype=dtype) + (
                jacobian @ inverse_ridge_jacobian_t
            )
            correction = inverse_ridge_jacobian_t @ torch.linalg.solve(
                feature_system,
                jacobian @ inverse_ridge_gradient,
            )
            step = (-inverse_ridge_gradient + correction).reshape_as(coefficients)
            step = step - step.mean(dim=0, keepdim=True)
            accepted = False
            for backtrack in range(int(line_search_steps)):
                candidate = coefficients + (0.5**backtrack) * step
                candidate = candidate - candidate.mean(dim=0, keepdim=True)
                candidate_norm = float(torch.linalg.norm(candidate))
                if trust_radius > 0.0 and candidate_norm > trust_radius:
                    candidate = candidate * (trust_radius / max(candidate_norm, 1e-12))
                _candidate_values, _candidate_residual, candidate_objective = evaluate(candidate)
                if torch.isfinite(candidate_objective) and float(candidate_objective) < float(objective) - 1e-10:
                    coefficients = candidate.detach()
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

    pre_anchor_values, pre_anchor_residual, pre_anchor_objective = evaluate(coefficients)
    anchor_requested_norm = max(float(causal_anchor_ratio), 0.0) * max(float(hidden_norm), 1e-12)
    anchor_applied_norm = 0.0
    anchor_scale = 0.0
    anchor_error_increase = 0.0
    if anchor_requested_norm > 0.0 and coefficients.shape[0] >= 2 and coefficients.shape[1] >= 1:
        current_coefficients = coefficients.detach().requires_grad_(True)
        current_cloud = cloud_from_coefficients(current_coefficients)
        edges = exact.soft_h0._mst_edges_from_cloud(current_cloud)

        def normalized_anchor_features(candidate: torch.Tensor) -> torch.Tensor:
            cloud = cloud_from_coefficients(candidate)
            values = exact.soft_h0._hard_mst_features_one(cloud, edges, top_k=5)
            return values / feature_std

        jacobian = torch.autograd.functional.jacobian(
            normalized_anchor_features,
            current_coefficients,
            vectorize=True,
        ).reshape(len(FEATURES), -1)
        template = torch.zeros_like(coefficients)
        requested_fraction = max(float(causal_anchor_suffix_fraction), 0.0)
        suffix_count = 1 if requested_fraction <= 0.0 else max(
            1,
            min(coefficients.shape[0] - 1, int(np.ceil(requested_fraction * coefficients.shape[0]))),
        )
        prefix_count = coefficients.shape[0] - suffix_count
        template[:prefix_count, 0] = -1.0 / float(prefix_count)
        template[prefix_count:, 0] = 1.0 / float(suffix_count)
        template_flat = template.reshape(-1)
        gram = jacobian @ jacobian.T
        ridge = 1e-8 * torch.eye(gram.shape[0], dtype=gram.dtype)
        projected_flat = template_flat - jacobian.T @ torch.linalg.solve(
            gram + ridge,
            jacobian @ template_flat,
        )
        projected_anchor = projected_flat.reshape_as(coefficients)
        projected_anchor = projected_anchor - projected_anchor.mean(dim=0, keepdim=True)
        if float(projected_anchor[-1, 0]) < 0.0:
            projected_anchor = -projected_anchor
        projected_norm = float(torch.linalg.norm(projected_anchor))
        if projected_norm > 1e-12 and torch.isfinite(projected_anchor).all():
            full_anchor = projected_anchor * (anchor_requested_norm / projected_norm)
            baseline_error = float(torch.linalg.norm(pre_anchor_residual))
            allowed_error = baseline_error + max(float(causal_anchor_max_error_increase), 0.0)
            for backtrack in range(12):
                scale = 0.5**backtrack
                candidate = (coefficients + scale * full_anchor).detach()
                candidate = candidate - candidate.mean(dim=0, keepdim=True)
                _values, candidate_residual, _objective = evaluate(candidate)
                candidate_error = float(torch.linalg.norm(candidate_residual))
                if np.isfinite(candidate_error) and candidate_error <= allowed_error + 1e-12:
                    coefficients = candidate
                    anchor_scale = scale
                    anchor_applied_norm = float(torch.linalg.norm(scale * full_anchor))
                    anchor_error_increase = candidate_error - baseline_error
                    status = f"{status}+causal_anchor"
                    break

    final_values, final_residual, final_objective = evaluate(coefficients)
    coefficients_np = coefficients.numpy().astype(np.float32, copy=False)
    delta_y = (coefficients_np @ projected.astype(np.float32)).astype(np.float32, copy=False)
    diagnostics: dict[str, Any] = {
        "optimization_status": status,
        "accepted_steps": int(accepted_steps),
        "line_search_backtracks": int(total_backtracks),
        "initial_normalized_target_error": float(torch.linalg.norm(initial_residual)),
        "final_normalized_target_error": float(torch.linalg.norm(final_residual)),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "pca_delta_norm": float(np.linalg.norm(delta_y)),
        "pca_cloud_norm": float(np.linalg.norm(cloud_np)),
        "relative_hidden_delta_norm": float(
            np.linalg.norm(coefficients_np) / max(float(hidden_norm), 1e-12)
        ),
        "delta_token_mean_norm": float(np.linalg.norm(coefficients_np.mean(axis=0))),
        "zero_mean_constraint": True,
        "trust_radius": float(trust_radius),
        "behavior_subspace_rank": int(basis.shape[0]),
        "behavior_subspace_pca_frobenius": float(np.linalg.norm(projected)),
        "behavior_subspace_pca_min_singular": float(
            projected_singular_values[-1] if len(projected_singular_values) else 0.0
        ),
        "token_coefficient_rms": float(np.sqrt(np.mean(coefficients_np**2))),
        "token_coefficient_abs_max": float(np.abs(coefficients_np).max(initial=0.0)),
        "pre_anchor_normalized_target_error": float(torch.linalg.norm(pre_anchor_residual)),
        "pre_anchor_objective": float(pre_anchor_objective),
        "causal_anchor_ratio": float(causal_anchor_ratio),
        "causal_anchor_requested_norm": float(anchor_requested_norm),
        "causal_anchor_applied_norm": float(anchor_applied_norm),
        "causal_anchor_scale": float(anchor_scale),
        "causal_anchor_error_increase": float(anchor_error_increase),
        "causal_anchor_suffix_fraction": float(causal_anchor_suffix_fraction),
        "causal_anchor_suffix_tokens": int(
            1
            if float(causal_anchor_suffix_fraction) <= 0.0
            else max(
                1,
                min(
                    coefficients.shape[0] - 1,
                    int(np.ceil(float(causal_anchor_suffix_fraction) * coefficients.shape[0])),
                ),
            )
        ),
        "causal_position_beta": float(causal_position_beta),
        "position_weight_min": float(position_weights.min()),
        "position_weight_max": float(position_weights.max()),
        "suffix_quarter_coefficient_norm_fraction": float(
            np.linalg.norm(coefficients_np[max(0, 3 * len(coefficients_np) // 4) :])
            / max(np.linalg.norm(coefficients_np), 1e-12)
        ),
        "final_token_behavior_coefficient": float(coefficients_np[-1, 0]),
    }
    for feature_index, feature in enumerate(FEATURES):
        diagnostics[f"initial_exact__{feature}"] = float(initial_values[feature_index])
        diagnostics[f"final_exact__{feature}"] = float(final_values[feature_index])
        diagnostics[f"optimization_target__{feature}"] = float(target[feature_index])
    return coefficients_np, delta_y, diagnostics


def _optimize_behavior_lowrank_many(
    *,
    clouds: list[np.ndarray],
    target_features: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
    bases: list[np.ndarray],
    components: np.ndarray,
    args: argparse.Namespace,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    tasks = (
        delayed(_optimize_behavior_lowrank_one)(
            cloud_np=cloud,
            target_np=target,
            feature_std_np=feature_std,
            hidden_norm=float(hidden_norm),
            basis_np=basis,
            components_np=components,
            lambda_value=float(lambda_value),
            damping=float(damping),
            trust_ratio=float(trust_ratio),
            gn_steps=int(args.gn_steps),
            line_search_steps=int(args.line_search_steps),
            causal_anchor_ratio=float(args.causal_anchor_ratio),
            causal_anchor_max_error_increase=float(args.causal_anchor_max_error_increase),
            causal_anchor_suffix_fraction=float(args.causal_anchor_suffix_fraction),
            causal_position_beta=float(args.causal_position_beta),
        )
        for cloud, target, hidden_norm, basis in zip(
            clouds,
            target_features,
            hidden_norms,
            bases,
            strict=True,
        )
    )
    results = Parallel(n_jobs=max(1, int(args.optimization_jobs)), backend="loky")(
        tqdm(tasks, total=len(clouds), desc="exact_h0_behavior_lowrank", leave=False)
    )
    coefficients = [result[0] for result in results]
    delta_y = [result[1] for result in results]
    diagnostics = pd.DataFrame(
        [{"row_index": index, **result[2]} for index, result in enumerate(results)]
    )
    return coefficients, delta_y, diagnostics


def _optimize_transport_one(
    *,
    cloud_np: np.ndarray,
    target_np: np.ndarray,
    feature_std_np: np.ndarray,
    hidden_norm: float,
    template_np: np.ndarray,
    prior_ratio: float,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
    gn_steps: int,
    line_search_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    template = np.asarray(template_np, dtype=np.float32).copy()
    template -= template.mean(axis=0, keepdims=True)
    template_norm = float(np.linalg.norm(template))
    requested_prior_norm = max(float(prior_ratio), 0.0) * max(float(hidden_norm), 1e-12)
    if template_norm > 1e-12 and requested_prior_norm > 0.0:
        prior = template * (requested_prior_norm / template_norm)
    else:
        prior = np.zeros_like(template)

    correction, correction_diagnostics = exact._optimize_one(
        cloud_np=np.asarray(cloud_np, dtype=np.float32) + prior,
        target_np=np.asarray(target_np, dtype=np.float32),
        feature_std_np=np.asarray(feature_std_np, dtype=np.float32),
        hidden_norm=float(hidden_norm),
        lambda_value=float(lambda_value),
        damping=float(damping),
        trust_ratio=float(trust_ratio),
        gn_steps=int(gn_steps),
        line_search_steps=int(line_search_steps),
        enforce_zero_mean=True,
    )
    total = prior + np.asarray(correction, dtype=np.float32)
    total -= total.mean(axis=0, keepdims=True)
    initial_values = exact._exact_features_tensor(
        torch.as_tensor(np.asarray(cloud_np, dtype=np.float64), dtype=torch.float64)
    ).numpy()
    prior_values = exact._exact_features_tensor(
        torch.as_tensor(np.asarray(cloud_np, dtype=np.float64) + prior, dtype=torch.float64)
    ).numpy()
    final_values = exact._exact_features_tensor(
        torch.as_tensor(np.asarray(cloud_np, dtype=np.float64) + total, dtype=torch.float64)
    ).numpy()
    feature_std = np.maximum(np.asarray(feature_std_np, dtype=np.float64), 1e-8)
    target = np.asarray(target_np, dtype=np.float64)
    initial_error = float(np.linalg.norm((initial_values - target) / feature_std))
    prior_error = float(np.linalg.norm((prior_values - target) / feature_std))
    final_error = float(np.linalg.norm((final_values - target) / feature_std))
    diagnostics = dict(correction_diagnostics)
    diagnostics.update(
        {
            "optimization_status": f"transport_prior+{correction_diagnostics['optimization_status']}",
            "initial_normalized_target_error": initial_error,
            "prior_normalized_target_error": prior_error,
            "final_normalized_target_error": final_error,
            "pca_delta_norm": float(np.linalg.norm(total)),
            "pca_cloud_norm": float(np.linalg.norm(cloud_np)),
            "relative_hidden_delta_norm": float(
                np.linalg.norm(total) / max(float(hidden_norm), 1e-12)
            ),
            "delta_token_mean_norm": float(np.linalg.norm(total.mean(axis=0))),
            "zero_mean_constraint": True,
            "transport_template_norm": template_norm,
            "transport_prior_requested_norm": requested_prior_norm,
            "transport_prior_applied_norm": float(np.linalg.norm(prior)),
            "transport_correction_norm": float(np.linalg.norm(correction)),
            "transport_prior_ratio": float(prior_ratio),
        }
    )
    for feature_index, feature in enumerate(FEATURES):
        diagnostics[f"initial_exact__{feature}"] = float(initial_values[feature_index])
        diagnostics[f"prior_exact__{feature}"] = float(prior_values[feature_index])
        diagnostics[f"final_exact__{feature}"] = float(final_values[feature_index])
        diagnostics[f"optimization_target__{feature}"] = float(target[feature_index])
    return total.astype(np.float32, copy=False), diagnostics


def _optimize_transport_many(
    *,
    clouds: list[np.ndarray],
    target_features: np.ndarray,
    feature_std: np.ndarray,
    hidden_norms: np.ndarray,
    templates: list[np.ndarray],
    args: argparse.Namespace,
    lambda_value: float,
    damping: float,
    trust_ratio: float,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    tasks = (
        delayed(_optimize_transport_one)(
            cloud_np=cloud,
            target_np=target,
            feature_std_np=feature_std,
            hidden_norm=float(hidden_norm),
            template_np=template,
            prior_ratio=float(args.transport_prior_ratio),
            lambda_value=float(lambda_value),
            damping=float(damping),
            trust_ratio=float(trust_ratio),
            gn_steps=int(args.gn_steps),
            line_search_steps=int(args.line_search_steps),
        )
        for cloud, target, hidden_norm, template in zip(
            clouds,
            target_features,
            hidden_norms,
            templates,
            strict=True,
        )
    )
    results = Parallel(n_jobs=max(1, int(args.optimization_jobs)), backend="loky")(
        tqdm(tasks, total=len(clouds), desc="exact_h0_transport_prior", leave=False)
    )
    deltas = [result[0] for result in results]
    diagnostics = pd.DataFrame(
        [{"row_index": index, **result[1]} for index, result in enumerate(results)]
    )
    return deltas, diagnostics


def _run_dataset(args: argparse.Namespace, config: dict[str, Any], dataset: str) -> None:
    if (
        args.target_mode == "self_contrastive_topology"
        and args.direction_source != "self_contrastive_prompt_pair"
    ):
        raise ValueError(
            "self_contrastive_topology requires --direction-source "
            "self_contrastive_prompt_pair"
        )
    tokenwise_direction_sources = {
        "self_contrastive_tokenwise",
        "observed_neighbor_tokenwise",
    }
    if (args.topology_controller == "behavior_tokenwise") != (
        args.direction_source in tokenwise_direction_sources
    ):
        raise ValueError(
            "behavior_tokenwise requires a tokenwise direction source, and tokenwise direction "
            "sources require behavior_tokenwise"
        )
    seed = int(config["seed"])
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(args.artifact_root).resolve() / dataset / model_slug)
    behavior = _load_local_behavior(args, config, dataset)
    cloud_df, feature_df, reducer = _load_exact_cache(args, config, dataset)
    cached_ids = set(cloud_df["example_id"].astype(str))
    behavior = behavior.loc[behavior["example_id"].isin(cached_ids)].reset_index(drop=True)
    retrieval_df, retrieval_columns = _load_all_layer_features(args, config, dataset)
    frame = _prepare_frame(behavior, feature_df, retrieval_df)
    if args.retrieval_feature_mode == "all_layer_plus_exact3":
        active_retrieval_columns = [*retrieval_columns, *FEATURES]
    elif args.retrieval_feature_mode == "exact3":
        active_retrieval_columns = list(FEATURES)
    else:
        active_retrieval_columns = retrieval_columns
    cloud_by_id = cloud_df.drop_duplicates("example_id").set_index("example_id")
    components = reducer.components_[: int(args.pca_components)].astype(np.float32, copy=False)
    behavior_source_slug = _behavior_source_slug(args)
    group_counts = frame.groupby(["split", "behavior_label"]).size().to_dict()
    print(
        f"[{dataset} {model_slug}] local-grounded frame={len(frame)} "
        f"groups={group_counts} source={args.behavior_label_source} "
        f"retrieval_mode={args.retrieval_feature_mode} "
        f"retrieval_geometry={args.retrieval_geometry} "
        f"retrieval_features={len(active_retrieval_columns)} "
        f"direction_source={args.direction_source} direction_readout={args.direction_readout} "
        f"layer={args.steering_layer}",
        flush=True,
    )

    bundle = load_hf_model(config["model"], config["generation"])
    try:
        for k in args.neighbor_ks:
            rows, current, neighbor_target, feature_std, plus, minus = _select_neighbors_and_targets(
                frame=frame,
                retrieval_columns=active_retrieval_columns,
                retrieval_geometry=str(args.retrieval_geometry),
                target_mode=str(args.target_mode),
                k=int(k),
                eval_n=int(args.eval_n),
                limit=args.limit,
                seed=seed,
                classifier_target_quantile=float(args.classifier_target_quantile),
            )
            neighbor_path = output_root / (
                f"{dataset}__hybrid_neighbors_{args.retrieval_feature_mode}_"
                f"{args.retrieval_geometry}_k{int(k)}.parquet"
            )
            write_parquet(rows, neighbor_path)
            rows.to_csv(neighbor_path.with_suffix(".csv"), index=False)
            example_ids = rows["example_id"].astype(str).tolist()
            clouds = [
                np.asarray(cloud_by_id.loc[example_id, "cloud"], dtype=np.float32)
                for example_id in example_ids
            ]
            hidden_norms = np.asarray(
                [
                    float(cloud_by_id.loc[example_id, "hidden_fro_norm"])
                    for example_id in example_ids
                ],
                dtype=np.float64,
            )
            pool_cache_root = ensure_dir(
                Path(args.direction_cache_root).resolve()
                / dataset
                / model_slug
                / f"source_{behavior_source_slug}"
                / f"labels_{args.positive_label_mode}"
            )
            tokenwise_templates: list[np.ndarray] | None = None
            if args.direction_source == "self_contrastive_tokenwise":
                contrastive_cache = (
                    pool_cache_root
                    / f"{dataset}__self_contrastive_token_sequences_"
                    f"layer{int(args.steering_layer):02d}.joblib"
                )
                token_difference_by_id = _load_or_extract_self_contrastive_token_sequences(
                    cache_path=contrastive_cache,
                    bundle=bundle,
                    config=config,
                    rows=rows,
                    layer=int(args.steering_layer),
                    max_length=int(args.max_length),
                    positive_instruction=str(args.contrastive_positive_instruction),
                    negative_instruction=str(args.contrastive_negative_instruction),
                    force=bool(args.force_directions),
                )
                directions, tokenwise_templates = _self_tokenwise_directions_and_templates(
                    rows,
                    token_difference_by_id,
                    [len(cloud) for cloud in clouds],
                    readout=str(args.direction_readout),
                )
                direction_bases = None
            elif args.direction_source == "observed_neighbor_tokenwise":
                token_pool_cache = (
                    pool_cache_root
                    / f"{dataset}__local_behavior_pool_token_sequences_"
                    f"layer{int(args.steering_layer):02d}.joblib"
                )
                token_sequence_by_id = _load_or_extract_pool_token_sequences(
                    cache_path=token_pool_cache,
                    bundle=bundle,
                    config=config,
                    plus=plus,
                    minus=minus,
                    layer=int(args.steering_layer),
                    max_length=int(args.max_length),
                    force=bool(args.force_directions),
                )
                directions, tokenwise_templates = (
                    _local_neighbor_tokenwise_directions_and_templates(
                        rows,
                        token_sequence_by_id,
                        [len(cloud) for cloud in clouds],
                        readout=str(args.direction_readout),
                    )
                )
                direction_bases = None
            elif args.direction_source == "self_contrastive_prompt_pair":
                contrastive_cache = (
                    pool_cache_root
                    / f"{dataset}__self_contrastive_prompt_state_"
                    f"layer{int(args.steering_layer):02d}__{args.direction_readout}.joblib"
                )
                (
                    differences_by_id,
                    positive_features_by_id,
                    negative_features_by_id,
                ) = _load_or_extract_self_contrastive_state(
                    cache_path=contrastive_cache,
                    bundle=bundle,
                    config=config,
                    rows=rows,
                    layer=int(args.steering_layer),
                    readout=str(args.direction_readout),
                    reducer=reducer,
                    pca_components=int(args.pca_components),
                    max_length=int(args.max_length),
                    positive_instruction=str(args.contrastive_positive_instruction),
                    negative_instruction=str(args.contrastive_negative_instruction),
                    force=bool(args.force_directions),
                )
                directions, self_bases = _self_paired_directions_and_bases(
                    rows,
                    differences_by_id,
                )
                direction_bases = (
                    self_bases if args.topology_controller == "behavior_lowrank" else None
                )
                if args.target_mode == "self_contrastive_topology":
                    positive_feature_matrix = np.stack(
                        [positive_features_by_id[str(value)] for value in rows["example_id"]]
                    ).astype(np.float32, copy=False)
                    negative_feature_matrix = np.stack(
                        [negative_features_by_id[str(value)] for value in rows["example_id"]]
                    ).astype(np.float32, copy=False)
                    neighbor_target = current + positive_feature_matrix - negative_feature_matrix
                    for feature_index, feature in enumerate(FEATURES):
                        rows[f"self_positive__{feature}"] = positive_feature_matrix[:, feature_index]
                        rows[f"self_negative__{feature}"] = negative_feature_matrix[:, feature_index]
                        rows[f"neighbor_target__{feature}"] = neighbor_target[:, feature_index]
            elif args.direction_source == "contrastive_prompt_pairs":
                contrastive_cache = (
                    pool_cache_root
                    / f"{dataset}__contrastive_prompt_differences_"
                    f"layer{int(args.steering_layer):02d}__{args.direction_readout}.joblib"
                )
                differences_by_id = _load_or_extract_contrastive_differences(
                    cache_path=contrastive_cache,
                    bundle=bundle,
                    config=config,
                    pool=pd.concat([plus, minus], ignore_index=True),
                    layer=int(args.steering_layer),
                    positive_instruction=str(args.contrastive_positive_instruction),
                    negative_instruction=str(args.contrastive_negative_instruction),
                    readout=str(args.direction_readout),
                    force=bool(args.force_directions),
                )
                directions = _local_paired_directions(rows, differences_by_id)
                direction_bases = (
                    _local_paired_direction_bases(
                        rows,
                        differences_by_id,
                        rank=int(args.behavior_rank),
                    )
                    if args.topology_controller == "behavior_lowrank"
                    else None
                )
            else:
                pool_cache = (
                    pool_cache_root
                    / f"{dataset}__local_behavior_pool_vectors_"
                    f"layer{int(args.steering_layer):02d}__{args.direction_readout}.joblib"
                )
                vector_by_id = _load_or_extract_pool_vectors(
                    cache_path=pool_cache,
                    bundle=bundle,
                    config=config,
                    plus=plus,
                    minus=minus,
                    layer=int(args.steering_layer),
                    readout=str(args.direction_readout),
                    force=bool(args.force_directions),
                )
                directions = _local_directions(rows, vector_by_id)
                direction_bases = (
                    _local_direction_bases(rows, vector_by_id, rank=int(args.behavior_rank))
                    if args.topology_controller == "behavior_lowrank"
                    else None
                )
            transport_templates = (
                _local_transport_templates(
                    rows,
                    cloud_by_id,
                    match_mode=str(args.transport_match_mode),
                )
                if args.topology_controller == "transport_prior"
                else None
            )

            for topology_alpha in args.topology_alphas:
                z_target = current + float(topology_alpha) * (neighbor_target - current)
                for lambda_value in args.lambdas:
                    for damping in args.dampings:
                        for trust_ratio in args.trust_ratios:
                            opt_slug = (
                                f"source_{behavior_source_slug}__labels_{args.positive_label_mode}__"
                                f"controller_{args.topology_controller}__"
                                f"direction_{args.direction_source}__"
                                f"readout_{args.direction_readout}__"
                                f"retrieval_{args.retrieval_feature_mode}__"
                                f"target_{args.target_mode}__k_{int(k)}__"
                                f"topoa_{_slug_float(topology_alpha)}__"
                                f"lambda_{_slug_float(lambda_value)}__damping_{_slug_float(damping)}__"
                                f"trust_{_slug_float(trust_ratio)}"
                            )
                            if args.retrieval_geometry != "standard":
                                opt_slug += f"__geometry_{args.retrieval_geometry}"
                            if args.target_mode == "classifier_projection":
                                opt_slug += (
                                    f"__targetq_{_slug_float(args.classifier_target_quantile)}"
                                )
                            if args.topology_controller == "behavior_lowrank":
                                opt_slug += (
                                    f"__rank_{int(args.behavior_rank)}__anchor_"
                                    f"{_slug_float(args.causal_anchor_ratio)}"
                                )
                                if float(args.causal_anchor_suffix_fraction) > 0.0:
                                    opt_slug += (
                                        f"__anchor_suffix_{_slug_float(args.causal_anchor_suffix_fraction)}"
                                    )
                                if float(args.causal_position_beta) > 0.0:
                                    opt_slug += f"__posbeta_{_slug_float(args.causal_position_beta)}"
                            elif args.topology_controller == "transport_prior":
                                opt_slug += f"__priorratio_{_slug_float(args.transport_prior_ratio)}"
                                if args.transport_match_mode != "nearest":
                                    opt_slug += f"__match_{args.transport_match_mode}"
                            opt_slug = _bounded_slug(opt_slug, max_length=175)
                            opt_root = ensure_dir(output_root / "_optimization" / opt_slug)
                            opt_path = opt_root / f"{dataset}__exact_h0_hybrid__optimization.parquet"
                            if opt_path.exists() and not args.force_optimize:
                                opt = pd.read_parquet(opt_path)
                                delta_y = [_as_float_matrix(value) for value in opt["pca_delta"].tolist()]
                                token_gates = (
                                    [_as_float_vector(value) for value in opt["token_gates"].tolist()]
                                    if "token_gates" in opt
                                    else None
                                )
                                token_coefficients = (
                                    [_as_float_matrix(value) for value in opt["token_coefficients"].tolist()]
                                    if "token_coefficients" in opt
                                    else None
                                )
                            else:
                                anchor_only = (
                                    args.topology_controller == "behavior_lowrank"
                                    and float(args.causal_anchor_ratio) > 0.0
                                )
                                if float(topology_alpha) == 0.0 and not anchor_only:
                                    delta_y, diagnostics = _zero_diagnostics(clouds, current, feature_std, hidden_norms)
                                    token_gates = (
                                        [np.zeros(len(cloud), dtype=np.float32) for cloud in clouds]
                                        if args.topology_controller
                                        in {"behavior_rank1", "behavior_tokenwise"}
                                        else None
                                    )
                                    token_coefficients = (
                                        [
                                            np.zeros((len(cloud), len(basis)), dtype=np.float32)
                                            for cloud, basis in zip(clouds, direction_bases or [], strict=True)
                                        ]
                                        if args.topology_controller == "behavior_lowrank"
                                        else None
                                    )
                                elif args.topology_controller == "behavior_rank1":
                                    token_coefficients = None
                                    token_gates, delta_y, diagnostics = _optimize_behavior_rank1_many(
                                        clouds=clouds,
                                        target_features=z_target,
                                        feature_std=feature_std,
                                        hidden_norms=hidden_norms,
                                        directions=directions,
                                        components=components,
                                        args=args,
                                        lambda_value=float(lambda_value),
                                        damping=float(damping),
                                        trust_ratio=float(trust_ratio),
                                    )
                                elif args.topology_controller == "behavior_tokenwise":
                                    if tokenwise_templates is None:
                                        raise ValueError(
                                            "behavior_tokenwise is missing query token templates"
                                        )
                                    token_coefficients = None
                                    token_gates, delta_y, diagnostics = (
                                        _optimize_behavior_tokenwise_many(
                                            clouds=clouds,
                                            target_features=z_target,
                                            feature_std=feature_std,
                                            hidden_norms=hidden_norms,
                                            templates=tokenwise_templates,
                                            components=components,
                                            args=args,
                                            lambda_value=float(lambda_value),
                                            damping=float(damping),
                                            trust_ratio=float(trust_ratio),
                                        )
                                    )
                                elif args.topology_controller == "behavior_lowrank":
                                    if direction_bases is None:
                                        raise ValueError("behavior_lowrank is missing local direction bases")
                                    token_gates = None
                                    token_coefficients, delta_y, diagnostics = _optimize_behavior_lowrank_many(
                                        clouds=clouds,
                                        target_features=z_target,
                                        feature_std=feature_std,
                                        hidden_norms=hidden_norms,
                                        bases=direction_bases,
                                        components=components,
                                        args=args,
                                        lambda_value=float(lambda_value),
                                        damping=float(damping),
                                        trust_ratio=float(trust_ratio),
                                    )
                                elif args.topology_controller == "transport_prior":
                                    if transport_templates is None:
                                        raise ValueError("transport_prior is missing local cloud templates")
                                    token_gates = None
                                    token_coefficients = None
                                    delta_y, diagnostics = _optimize_transport_many(
                                        clouds=clouds,
                                        target_features=z_target,
                                        feature_std=feature_std,
                                        hidden_norms=hidden_norms,
                                        templates=transport_templates,
                                        args=args,
                                        lambda_value=float(lambda_value),
                                        damping=float(damping),
                                        trust_ratio=float(trust_ratio),
                                    )
                                else:
                                    token_gates = None
                                    token_coefficients = None
                                    delta_y, diagnostics = exact._optimize_many(
                                        clouds=clouds,
                                        target_features=z_target,
                                        feature_std=feature_std,
                                        hidden_norms=hidden_norms,
                                        args=args,
                                        lambda_value=float(lambda_value),
                                        damping=float(damping),
                                        trust_ratio=float(trust_ratio),
                                    )
                                opt = pd.concat([rows.reset_index(drop=True), diagnostics.drop(columns="row_index")], axis=1)
                                opt["pca_delta"] = [value.tolist() for value in delta_y]
                                if token_gates is not None:
                                    opt["token_gates"] = [value.tolist() for value in token_gates]
                                if token_coefficients is not None:
                                    opt["token_coefficients"] = [value.tolist() for value in token_coefficients]
                                opt["topology_controller"] = str(args.topology_controller)
                                opt["direction_source"] = str(args.direction_source)
                                opt["retrieval_feature_mode"] = str(args.retrieval_feature_mode)
                                opt["retrieval_geometry"] = str(args.retrieval_geometry)
                                opt["behavior_rank"] = int(args.behavior_rank)
                                opt["topology_alpha"] = float(topology_alpha)
                                opt["alpha"] = float(topology_alpha)
                                opt["lambda"] = float(lambda_value)
                                opt["damping"] = float(damping)
                                opt["trust_ratio"] = float(trust_ratio)
                                for feature_index, feature in enumerate(FEATURES):
                                    opt[f"z_target__{feature}"] = z_target[:, feature_index]
                                if args.target_mode == "classifier_projection":
                                    classifier_weights = rows[
                                        [f"classifier_raw_weight__{feature}" for feature in FEATURES]
                                    ].to_numpy(dtype=np.float64)
                                    classifier_intercepts = rows["classifier_raw_intercept"].to_numpy(
                                        dtype=np.float64
                                    )
                                    opt["classifier_z_target_score"] = (
                                        z_target.astype(np.float64) * classifier_weights
                                    ).sum(axis=1) + classifier_intercepts
                                write_parquet(opt, opt_path)
                            if args.topology_controller == "behavior_rank1":
                                if token_gates is None:
                                    raise ValueError("behavior_rank1 optimization is missing token_gates")
                                direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
                                direction_units = directions / np.maximum(direction_norms, 1e-12)
                                centered_h = [
                                    (gates[:, None] * direction_unit[None, :]).astype(np.float32, copy=False)
                                    for gates, direction_unit in zip(token_gates, direction_units, strict=True)
                                ]
                            elif args.topology_controller == "behavior_tokenwise":
                                if token_gates is None or tokenwise_templates is None:
                                    raise ValueError(
                                        "behavior_tokenwise optimization is missing gates or templates"
                                    )
                                centered_h = []
                                for gates, template in zip(
                                    token_gates,
                                    tokenwise_templates,
                                    strict=True,
                                ):
                                    value = gates[:, None] * template
                                    value = value - value.mean(axis=0, keepdims=True)
                                    centered_h.append(value.astype(np.float32, copy=False))
                            elif args.topology_controller == "behavior_lowrank":
                                if token_coefficients is None or direction_bases is None:
                                    raise ValueError("behavior_lowrank optimization is missing coefficients or bases")
                                centered_h = [
                                    (coefficients @ basis).astype(np.float32, copy=False)
                                    for coefficients, basis in zip(
                                        token_coefficients,
                                        direction_bases,
                                        strict=True,
                                    )
                                ]
                            else:
                                centered_h = [(value @ components).astype(np.float32, copy=False) for value in delta_y]

                            if args.skip_generate:
                                continue
                            shared_settings = (
                                [(float(value), None) for value in args.mean_alphas]
                                if args.shared_target_ratios is None
                                else [(None, float(value)) for value in args.shared_target_ratios]
                            )
                            for mean_alpha, shared_target_ratio in shared_settings:
                                if shared_target_ratio is None:
                                    run_slug = f"{opt_slug}__meana_{_slug_float(mean_alpha)}"
                                else:
                                    run_slug = (
                                        f"{opt_slug}__sharedratio_"
                                        f"{_slug_float(shared_target_ratio)}"
                                    )
                                run_slug += f"__shared_{args.shared_intervention_site}"
                                if args.topology_decode_mode != "none":
                                    run_slug += (
                                        f"__topodecode_{args.topology_decode_mode}_"
                                        f"{_slug_float(args.topology_decode_scale)}"
                                    )
                                    if args.topology_decode_mode == "suffix_mean":
                                        run_slug += (
                                            f"__suffix_{_slug_float(args.topology_decode_suffix_fraction)}"
                                        )
                                run_slug = _bounded_slug(run_slug, max_length=230)
                                run_root = ensure_dir(output_root / run_slug)
                                raw_path = run_root / f"{dataset}__exact_h0_gn__raw.parquet"
                                if raw_path.exists() and not args.force_generate:
                                    print(f"[{dataset} {run_slug}] raw exists", flush=True)
                                    continue
                                shared, shared_scales = _scaled_shared_vectors(
                                    directions,
                                    hidden_norms,
                                    np.asarray([len(value) for value in centered_h]),
                                    mean_alpha=mean_alpha,
                                    target_ratio=shared_target_ratio,
                                )
                                # Temperature stays at the configured value (0.1), but common
                                # random numbers isolate intervention effects across settings.
                                set_global_seed(seed)
                                responses, verified_features = _generate_with_hybrid_deltas(
                                    bundle=bundle,
                                    config=config,
                                    rows=rows,
                                    topology_deltas=centered_h,
                                    shared_vectors=shared,
                                    layer=int(args.steering_layer),
                                    max_length=int(args.max_length),
                                    apply_on=str(args.apply_on),
                                    shared_intervention_site=str(args.shared_intervention_site),
                                    topology_decode_mode=str(args.topology_decode_mode),
                                    topology_decode_scale=float(args.topology_decode_scale),
                                    topology_decode_suffix_fraction=float(
                                        args.topology_decode_suffix_fraction
                                    ),
                                    reducer=reducer,
                                )
                                run = opt.drop(
                                    columns=["pca_delta", "token_gates", "token_coefficients"],
                                    errors="ignore",
                                ).copy()
                                centered_norms = np.asarray([np.linalg.norm(value) for value in centered_h])
                                shared_prompt_norms = np.asarray(
                                    [np.sqrt(len(value)) * np.linalg.norm(direction) for value, direction in zip(centered_h, shared, strict=True)]
                                )
                                total_norms = np.asarray(
                                    [
                                        np.linalg.norm(value + direction[None, :])
                                        for value, direction in zip(centered_h, shared, strict=True)
                                    ]
                                )
                                run["response_text"] = responses
                                method = (
                                    f"exact_h0_hybrid_alllayer_{behavior_source_slug}_"
                                    f"{args.topology_controller}_{args.direction_source}"
                                )
                                run["method"] = method
                                run["strategy"] = method
                                run["mean_alpha"] = (
                                    float(mean_alpha) if mean_alpha is not None else float("nan")
                                )
                                run["shared_target_ratio"] = (
                                    float(shared_target_ratio)
                                    if shared_target_ratio is not None
                                    else float("nan")
                                )
                                run["shared_scale_mean"] = float(shared_scales.mean())
                                run["shared_scale_std"] = float(shared_scales.std())
                                run["topology_alpha"] = float(topology_alpha)
                                run["retrieval_feature_mode"] = str(args.retrieval_feature_mode)
                                run["retrieval_geometry"] = str(args.retrieval_geometry)
                                run["transport_match_mode"] = str(args.transport_match_mode)
                                run["transport_prior_ratio"] = float(args.transport_prior_ratio)
                                run["behavior_label_source"] = str(args.behavior_label_source)
                                run["direction_source"] = str(args.direction_source)
                                run["direction_readout"] = str(args.direction_readout)
                                run["classifier_target_quantile"] = float(
                                    args.classifier_target_quantile
                                )
                                run["apply_on"] = str(args.apply_on)
                                run["shared_intervention_site"] = str(args.shared_intervention_site)
                                run["topology_decode_mode"] = str(args.topology_decode_mode)
                                run["topology_decode_scale"] = float(args.topology_decode_scale)
                                run["topology_decode_suffix_fraction"] = float(
                                    args.topology_decode_suffix_fraction
                                )
                                run["layer"] = int(args.steering_layer)
                                run["direction_norm"] = np.linalg.norm(directions, axis=1)
                                run["centered_hidden_delta_norm"] = centered_norms
                                run["shared_hidden_delta_norm"] = shared_prompt_norms
                                run["total_hidden_delta_norm"] = total_norms
                                run["relative_centered_hidden_delta_norm"] = centered_norms / np.maximum(hidden_norms, 1e-12)
                                run["relative_shared_hidden_delta_norm"] = shared_prompt_norms / np.maximum(hidden_norms, 1e-12)
                                run["relative_hidden_delta_norm"] = total_norms / np.maximum(hidden_norms, 1e-12)
                                run["delta_token_mean_norm"] = np.linalg.norm(shared, axis=1)
                                for feature_index, feature in enumerate(FEATURES):
                                    run[f"post_intervention_exact__{feature}"] = verified_features[:, feature_index]
                                target_matrix = run[
                                    [f"z_target__{feature}" for feature in FEATURES]
                                ].to_numpy(dtype=np.float64)
                                normalized_post_error = (
                                    verified_features.astype(np.float64) - target_matrix
                                ) / feature_std.astype(np.float64)[None, :]
                                run["post_intervention_normalized_target_error"] = np.linalg.norm(
                                    normalized_post_error,
                                    axis=1,
                                )
                                if args.target_mode == "classifier_projection":
                                    classifier_weights = run[
                                        [f"classifier_raw_weight__{feature}" for feature in FEATURES]
                                    ].to_numpy(dtype=np.float64)
                                    classifier_intercepts = run["classifier_raw_intercept"].to_numpy(
                                        dtype=np.float64
                                    )
                                    run["post_intervention_classifier_score"] = (
                                        verified_features.astype(np.float64) * classifier_weights
                                    ).sum(axis=1) + classifier_intercepts
                                write_parquet(run, raw_path)
                                run.to_csv(raw_path.with_suffix(".csv"), index=False)
                                write_json(
                                    run_root / f"{dataset}__exact_h0_hybrid__summary.json",
                                    {
                                        "dataset": dataset,
                                        "model": config["model"]["name"],
                                        "method": method,
                                        "n_eval": len(run),
                                        "neighbor_k": int(k),
                                        "retrieval_feature_mode": str(args.retrieval_feature_mode),
                                        "retrieval_geometry": str(args.retrieval_geometry),
                                        "retrieval_feature_count": len(active_retrieval_columns),
                                        "positive_label": (
                                            (
                                                TARGETED_RULE_POSITIVE
                                                if args.behavior_label_source
                                                == "rule_targeted_clarification"
                                                else RULE_POSITIVE
                                            )
                                            if args.behavior_label_source
                                            in {"rule_high_precision", "rule_targeted_clarification"}
                                            else (
                                                "ACCEPTABLE"
                                                if args.behavior_label_source == "source_judge"
                                                else (
                                                    LOCAL_POSITIVE
                                                    if args.positive_label_mode == "grounded"
                                                    else "GROUNDED_ACCEPTABLE|GENERIC_ACCEPTABLE"
                                                )
                                            )
                                        ),
                                        "positive_label_mode": str(args.positive_label_mode),
                                        "behavior_label_source": str(args.behavior_label_source),
                                        "direction_source": str(args.direction_source),
                                        "contrastive_positive_instruction": (
                                            str(args.contrastive_positive_instruction)
                                            if args.direction_source == "contrastive_prompt_pairs"
                                            else None
                                        ),
                                        "contrastive_negative_instruction": (
                                            str(args.contrastive_negative_instruction)
                                            if args.direction_source == "contrastive_prompt_pairs"
                                            else None
                                        ),
                                        "classifier_target_quantile": float(
                                            args.classifier_target_quantile
                                        ),
                                        "local_label_confidence_min": float(
                                            args.local_label_confidence_min
                                        ),
                                        "local_label_margin_min": float(args.local_label_margin_min),
                                        "require_positive_rule_marker": bool(
                                            args.require_positive_rule_marker
                                        ),
                                        "negative_label": (
                                            RULE_NEGATIVE
                                            if args.behavior_label_source
                                            in {"rule_high_precision", "rule_targeted_clarification"}
                                            else (
                                                "UNACCEPTABLE"
                                                if args.behavior_label_source == "source_judge"
                                                else LOCAL_NEGATIVE
                                            )
                                        ),
                                        "topology_controller": str(args.topology_controller),
                                        "behavior_rank": int(args.behavior_rank),
                                        "transport_match_mode": str(args.transport_match_mode),
                                        "transport_prior_ratio": float(args.transport_prior_ratio),
                                        "causal_anchor_ratio": float(args.causal_anchor_ratio),
                                        "causal_anchor_max_error_increase": float(
                                            args.causal_anchor_max_error_increase
                                        ),
                                        "topology_alpha": float(topology_alpha),
                                        "mean_alpha": (
                                            float(mean_alpha) if mean_alpha is not None else None
                                        ),
                                        "shared_target_ratio": (
                                            float(shared_target_ratio)
                                            if shared_target_ratio is not None
                                            else None
                                        ),
                                        "shared_scale_mean": float(shared_scales.mean()),
                                        "shared_scale_std": float(shared_scales.std()),
                                        "lambda": float(lambda_value),
                                        "damping": float(damping),
                                        "trust_ratio": float(trust_ratio),
                                        "apply_on": str(args.apply_on),
                                        "shared_intervention_site": str(args.shared_intervention_site),
                                        "topology_decode_mode": str(args.topology_decode_mode),
                                        "topology_decode_scale": float(args.topology_decode_scale),
                                        "topology_decode_suffix_fraction": float(
                                            args.topology_decode_suffix_fraction
                                        ),
                                        "relative_hidden_delta_norm_mean": float(run["relative_hidden_delta_norm"].mean()),
                                        "relative_centered_hidden_delta_norm_mean": float(run["relative_centered_hidden_delta_norm"].mean()),
                                        "relative_shared_hidden_delta_norm_mean": float(run["relative_shared_hidden_delta_norm"].mean()),
                                        "final_normalized_target_error_mean": float(run["final_normalized_target_error"].mean()),
                                        "post_intervention_normalized_target_error_mean": float(
                                            run["post_intervention_normalized_target_error"].mean()
                                        ),
                                        "unique_responses": int(run["response_text"].nunique()),
                                        "empty_responses": int(run["response_text"].fillna("").str.strip().eq("").sum()),
                                    },
                                )
    finally:
        del bundle
        _release()


def main() -> None:
    args = _parse_args()
    # The shared exact-H0 optimizer expects this ablation flag. Hybrid steering
    # deliberately keeps its topology-changing component centered.
    args.allow_mean_shift = False
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    for dataset in args.datasets:
        _run_dataset(args, config, dataset)


if __name__ == "__main__":
    main()
