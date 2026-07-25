"""ReDeEP representative statistics and low-dimensional inverse utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ReDeEPStatisticConfig:
    """Published Llama-2-7B Dolly statistic and AARF configuration."""

    copy_heads: tuple[tuple[int, int], ...]
    knowledge_layers: tuple[int, ...]
    external_min: float
    external_max: float
    parameter_min: float
    parameter_max: float
    detector_min: float
    detector_max: float
    detector_weight: float = 0.2
    intervention_threshold: float = 0.6
    aarf_attention_scale: float = 1.2
    aarf_ffn_scale: float = 0.8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


LLAMA2_7B_DOLLY_CONFIG = ReDeEPStatisticConfig(
    copy_heads=((31, 28), (31, 19), (31, 24), (28, 18)),
    knowledge_layers=(31, 0, 1),
    external_min=0.010007858276367188,
    external_max=2.060791015625,
    parameter_min=0.0,
    parameter_max=216.06683731079102,
    detector_min=-0.10041143162393507,
    detector_max=-0.028838051457178614,
    detector_weight=0.2,
    intervention_threshold=0.6,
    aarf_attention_scale=1.2,
    aarf_ffn_scale=0.8,
)


@dataclass(frozen=True)
class LinearizedInverseDiagnostics:
    initial_error: float
    predicted_error: float
    control_norm: float
    control_rms: float
    capped: bool


def minmax_normalize(
    value: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Apply the min-max transform used by ReDeEP without clipping."""

    width = float(maximum) - float(minimum)
    if width <= 0:
        raise ValueError("maximum must be greater than minimum")
    return (value.float() - float(minimum)) / width


def official_redeep_divergence(
    post_ffn_logits: torch.Tensor,
    pre_ffn_logits: torch.Tensor,
    *,
    scale: float = 1e6,
) -> torch.Tensor:
    """Compute the symmetric divergence used by the released ReDeEP code.

    The paper calls this quantity Jensen-Shannon divergence. The release passes
    the mixture as the target to ``torch.kl_div`` and averages over vocabulary,
    yielding ``0.5 * (KL(M || P) + KL(M || Q))``. We preserve that definition
    so targets and published normalization ranges remain compatible.
    """

    if post_ffn_logits.shape != pre_ffn_logits.shape:
        raise ValueError("pre- and post-FFN logits must have identical shapes")
    post_log_probs = F.log_softmax(post_ffn_logits.float(), dim=-1)
    pre_log_probs = F.log_softmax(pre_ffn_logits.float(), dim=-1)
    mixture = 0.5 * (post_log_probs.exp() + pre_log_probs.exp())
    post_term = F.kl_div(post_log_probs, mixture, reduction="none").mean(dim=-1)
    pre_term = F.kl_div(pre_log_probs, mixture, reduction="none").mean(dim=-1)
    return 0.5 * (post_term + pre_term) * float(scale)


def normalized_redeep_statistics(
    parameter_score: torch.Tensor,
    external_score: torch.Tensor,
    config: ReDeEPStatisticConfig,
) -> torch.Tensor:
    """Return ``[normalized PKS, normalized ECS]`` along the final axis."""

    parameter = minmax_normalize(
        parameter_score, config.parameter_min, config.parameter_max
    )
    external = minmax_normalize(
        external_score, config.external_min, config.external_max
    )
    return torch.stack((parameter, external), dim=-1)


def redeep_raw_detector_score(
    statistics: torch.Tensor,
    detector_weight: float,
) -> torch.Tensor:
    """Combine normalized PKS and ECS; larger values predict hallucination."""

    if statistics.shape[-1] != 2:
        raise ValueError("ReDeEP statistics must end in [PKS, ECS]")
    return statistics[..., 0] - float(detector_weight) * statistics[..., 1]


def redeep_detector_score(
    statistics: torch.Tensor,
    config: ReDeEPStatisticConfig,
) -> torch.Tensor:
    raw = redeep_raw_detector_score(statistics, config.detector_weight)
    return minmax_normalize(raw, config.detector_min, config.detector_max)


def project_to_redeep_target(
    statistics: torch.Tensor,
    config: ReDeEPStatisticConfig,
    *,
    target_score: float | None = None,
    score_shift: float | None = None,
    one_sided: bool = True,
) -> torch.Tensor:
    """Project ``[PKS, ECS]`` onto a lower ReDeEP detector-score level.

    Projection is Euclidean in normalized statistic space. Therefore PKS is
    reduced while ECS is increased, with their ratio fixed by the detector's
    normal vector ``[1, -weight]``.
    """

    if (target_score is None) == (score_shift is None):
        raise ValueError("provide exactly one of target_score or score_shift")
    if score_shift is not None and score_shift < 0:
        raise ValueError("score_shift must be non-negative")
    current_raw = redeep_raw_detector_score(statistics, config.detector_weight)
    detector_width = config.detector_max - config.detector_min
    if target_score is not None:
        desired_raw = config.detector_min + float(target_score) * detector_width
        desired_raw = torch.full_like(current_raw, desired_raw)
    else:
        desired_raw = current_raw - float(score_shift) * detector_width
    if one_sided:
        desired_raw = torch.minimum(current_raw, desired_raw)

    normal = statistics.new_tensor([1.0, -float(config.detector_weight)])
    distance = (current_raw - desired_raw) / normal.square().sum()
    return statistics - distance.unsqueeze(-1) * normal


def solve_linearized_statistic_inverse(
    jacobian: torch.Tensor,
    current_control: torch.Tensor,
    current_statistic: torch.Tensor,
    target_statistic: torch.Tensor,
    *,
    ridge: float,
    damping: float = 1.0,
    maximum_control_rms: float | None = None,
) -> tuple[torch.Tensor, LinearizedInverseDiagnostics]:
    """Solve the accumulated regularized minimum-norm local inverse.

    For ``z(c) ~= z(c0) + J(c-c0)``, this solves

    ``argmin_c ||Jc - (z_target-z(c0)+Jc0)||^2 + ridge ||c||^2``.
    """

    if jacobian.ndim != 2:
        raise ValueError("jacobian must have shape [statistic, control]")
    statistic_count, control_count = jacobian.shape
    if current_control.shape != (control_count,):
        raise ValueError("current_control has the wrong shape")
    if current_statistic.shape != (statistic_count,):
        raise ValueError("current_statistic has the wrong shape")
    if target_statistic.shape != (statistic_count,):
        raise ValueError("target_statistic has the wrong shape")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")

    matrix = jacobian.float()
    current = current_control.float()
    error = target_statistic.float() - current_statistic.float()
    right_hand_side = error + matrix @ current
    gram = matrix @ matrix.T
    regularized = gram + float(ridge) * torch.eye(
        statistic_count, device=matrix.device, dtype=matrix.dtype
    )
    if ridge == 0:
        dual = torch.linalg.pinv(regularized) @ right_hand_side
    else:
        dual = torch.linalg.solve(regularized, right_hand_side)
    solution = matrix.T @ dual
    proposed = current + float(damping) * (solution - current)

    capped = False
    if maximum_control_rms is not None:
        if maximum_control_rms < 0:
            raise ValueError("maximum_control_rms must be non-negative")
        rms = proposed.square().mean().sqrt()
        if float(rms) > maximum_control_rms:
            proposed = proposed * (float(maximum_control_rms) / rms.clamp_min(1e-12))
            capped = True

    predicted = current_statistic.float() + matrix @ (proposed - current)
    diagnostics = LinearizedInverseDiagnostics(
        initial_error=float(error.square().mean().sqrt()),
        predicted_error=float(
            (target_statistic.float() - predicted).square().mean().sqrt()
        ),
        control_norm=float(proposed.norm()),
        control_rms=float(proposed.square().mean().sqrt()),
        capped=capped,
    )
    return proposed.to(current_control.dtype), diagnostics


def load_official_chunk_features(path: str | Path) -> pd.DataFrame:
    """Load the released Llama-2-7B RAGTruth chunk-level signals."""

    with Path(path).open("r", encoding="utf-8") as handle:
        responses = json.load(handle)
    records: list[dict[str, object]] = []
    for response_index, response in enumerate(responses):
        for chunk_index, score in enumerate(response["scores"]):
            record: dict[str, object] = {
                "response_index": response_index,
                "source_id": str(response["source_id"]),
                "chunk_index": chunk_index,
                "hallucination_label": int(score["hallucination_label"]),
            }
            for name, value in score["prompt_attention_score"].items():
                record[f"ecs::{name}"] = float(value)
            for name, value in score["parameter_knowledge_scores"].items():
                record[f"pks::{name}"] = float(value)
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"no ReDeEP records found in {path}")
    return frame


def _safe_auc(labels: pd.Series, values: pd.Series) -> float:
    if labels.nunique() != 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _select_features(
    frame: pd.DataFrame,
    *,
    external_count: int,
    parameter_count: int,
) -> tuple[list[str], list[str]]:
    labels = frame["hallucination_label"]
    external = [column for column in frame if column.startswith("ecs::")]
    parameter = [column for column in frame if column.startswith("pks::")]
    external_rank = sorted(
        ((_safe_auc(1 - labels, frame[column]), column) for column in external),
        reverse=True,
    )
    parameter_rank = sorted(
        ((_safe_auc(labels, frame[column]), column) for column in parameter),
        reverse=True,
    )
    return (
        [column for _, column in external_rank[:external_count]],
        [column for _, column in parameter_rank[:parameter_count]],
    )


def _fit_range(values: pd.Series) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("cannot fit a non-degenerate min-max range")
    return minimum, maximum


def _score_feature_frame(
    frame: pd.DataFrame,
    *,
    external_features: Sequence[str],
    parameter_features: Sequence[str],
    external_range: tuple[float, float],
    parameter_range: tuple[float, float],
    detector_weight: float,
) -> pd.DataFrame:
    result = frame.copy()
    external = result[list(external_features)].sum(axis=1)
    parameter = result[list(parameter_features)].sum(axis=1)
    result["ecs_sum"] = external
    result["pks_sum"] = parameter
    result["ecs_normalized"] = (external - external_range[0]) / (
        external_range[1] - external_range[0]
    )
    result["pks_normalized"] = (parameter - parameter_range[0]) / (
        parameter_range[1] - parameter_range[0]
    )
    result["redeep_score"] = (
        result["pks_normalized"]
        - float(detector_weight) * result["ecs_normalized"]
    )
    return result


def evaluate_redeep_detector(
    frame: pd.DataFrame,
    *,
    external_count: int = 3,
    parameter_count: int = 4,
    detector_weight: float = 0.6,
    test_size: float = 0.2,
    seed: int = 42,
    leaky_official_protocol: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate ReDeEP with either the released or a grouped held-out protocol."""

    response_labels = frame.groupby("response_index")["hallucination_label"].max()
    if leaky_official_protocol:
        train_ids = response_labels.index
        test_ids = response_labels.index
    else:
        train_ids, test_ids = train_test_split(
            response_labels.index.to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=response_labels.to_numpy(),
        )
    train = frame[frame["response_index"].isin(train_ids)].copy()
    test = frame[frame["response_index"].isin(test_ids)].copy()
    external_features, parameter_features = _select_features(
        train,
        external_count=external_count,
        parameter_count=parameter_count,
    )
    train_external = train[external_features].sum(axis=1)
    train_parameter = train[parameter_features].sum(axis=1)
    external_range = _fit_range(train_external)
    parameter_range = _fit_range(train_parameter)
    scored = _score_feature_frame(
        test,
        external_features=external_features,
        parameter_features=parameter_features,
        external_range=external_range,
        parameter_range=parameter_range,
        detector_weight=detector_weight,
    )
    response_scores = scored.groupby("response_index").agg(
        redeep_score=("redeep_score", "mean"),
        hallucination_label=("hallucination_label", "max"),
        source_id=("source_id", "first"),
    )
    result: dict[str, object] = {
        "protocol": "official_leaky" if leaky_official_protocol else "grouped_heldout",
        "seed": seed,
        "train_responses": int(len(train_ids)),
        "test_responses": int(len(test_ids)),
        "test_chunks": int(len(scored)),
        "external_features": external_features,
        "parameter_features": parameter_features,
        "external_range": list(external_range),
        "parameter_range": list(parameter_range),
        "chunk_auc": _safe_auc(
            scored["hallucination_label"], scored["redeep_score"]
        ),
        "response_auc": _safe_auc(
            response_scores["hallucination_label"], response_scores["redeep_score"]
        ),
        "positive_response_rate": float(response_scores["hallucination_label"].mean()),
    }
    return result, scored
