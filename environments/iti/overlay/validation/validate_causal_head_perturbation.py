#!/usr/bin/env python3
"""Compare fixed ITI with causal, target-conditioned head perturbations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mc_scoring import find_answer_token_span, score_answer_logits
from utils import (
    get_com_directions,
    get_separated_activations,
    get_top_heads,
    layer_head_to_flattened_idx,
    load_truthfulqa_frame,
)
from truthfulqa import utilities
from truthfulqa.configs import ANSWER_COL, BEST_COL, INCORRECT_COL
from truthfulqa.models import MC_calcs, set_columns
from validate_margin_perturbation import DEFAULT_PREFIX, fold_train_val_indices


warnings.simplefilter("ignore", PerformanceWarning)


Method = Literal[
    "fixed_com",
    "adaptive_com",
    "adaptive_probe",
    "aggregate_com",
    "aggregate_probe",
    "targeted_iti",
    "targeted_probe_iti",
    "bounded_targeted_probe_iti",
    "headwise_probe_iti",
    "headwise_probe_min_norm",
    "group_direction_probe_iti",
    "group_direction_probe_min_norm",
]
ProbeScoreNormalization = Literal["standardized", "raw"]

DIAGNOSTIC_FIELDS = (
    "relative_action_norm",
    "intervention_rate",
    "pre_target_error",
    "post_target_error",
    "active_signed_target_error",
    "active_absolute_target_error",
    "active_target_overshoot",
    "clip_rate",
)


@dataclass(frozen=True)
class Setting:
    method: Method
    num_heads: int
    alpha: float
    target_quantile: float | None = None
    ridge_ratio: float | None = None
    relative_cap: float | None = None
    coefficient_cap: float | None = None
    probe_score_normalization: ProbeScoreNormalization = "standardized"

    @property
    def tag(self) -> str:
        pieces = [self.method, f"k{self.num_heads}", f"a{slug(self.alpha)}"]
        if self.target_quantile is not None:
            pieces.append(f"q{slug(self.target_quantile)}")
        if self.ridge_ratio is not None:
            pieces.append(f"r{slug(self.ridge_ratio)}")
        if self.relative_cap is not None:
            pieces.append(f"c{slug(self.relative_cap)}")
        if self.coefficient_cap is not None:
            pieces.append(f"b{slug(self.coefficient_cap)}")
        if self.probe_score_normalization == "raw":
            pieces.append("nraw")
        return "_".join(pieces)


@dataclass
class StatisticBank:
    top_heads: list[tuple[int, int]]
    com_directions: np.ndarray
    fixed_scales: np.ndarray
    com_means: np.ndarray
    com_stds: np.ndarray
    com_truthful: np.ndarray
    probe_directions: np.ndarray
    probe_intercepts: np.ndarray
    probe_means: np.ndarray
    probe_stds: np.ndarray
    probe_truthful: np.ndarray
    probe_group_direction: np.ndarray
    probe_group_positive_projection: np.ndarray
    probe_group_negative_projection: np.ndarray
    validation_accuracies: np.ndarray
    probe_raw_group_direction: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    probe_raw_group_positive_projection: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    probe_raw_group_negative_projection: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-prefix", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument(
        "--settings-file",
        type=Path,
        help="optional JSON list of exact Setting fields, overriding sweep grids",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "fixed_com",
            "adaptive_com",
            "adaptive_probe",
            "aggregate_com",
            "aggregate_probe",
            "targeted_iti",
            "targeted_probe_iti",
            "bounded_targeted_probe_iti",
            "headwise_probe_iti",
            "headwise_probe_min_norm",
            "group_direction_probe_iti",
            "group_direction_probe_min_norm",
        ),
        default=("fixed_com", "aggregate_com", "aggregate_probe", "targeted_iti"),
    )
    parser.add_argument("--fixed-alphas", type=float, nargs="+", default=(5.0, 10.0, 15.0, 20.0))
    parser.add_argument("--target-quantiles", type=float, nargs="+", default=(0.5, 0.75))
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--ridge-ratios", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--relative-caps", type=float, nargs="+", default=(0.1, 0.25))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--statistics-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class AttentionHeadController:
    """Forward pre-hook that edits selected slices of an attention-head tensor."""

    def __init__(self, head_dim: int, hidden_size: int) -> None:
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.mode = "off"
        self.start = 0
        self.stop = 0
        self.fixed_action: torch.Tensor | None = None
        self.heads: list[int] = []
        self.directions: torch.Tensor | None = None
        self.means: torch.Tensor | None = None
        self.stds: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.alpha = 0.0
        self.ridge_ratio = 0.0
        self.relative_cap: float | None = None
        self.last_state_sq: torch.Tensor | None = None
        self.last_action_sq: torch.Tensor | None = None
        self.last_active: torch.Tensor | None = None
        self.last_count = 0
        self.last_pre_error: torch.Tensor | None = None
        self.last_post_error: torch.Tensor | None = None
        self.last_collected: torch.Tensor | None = None

    def off(self) -> None:
        self.mode = "off"

    def collect(self, heads: list[int], start: int, stop: int) -> None:
        self.mode = "collect"
        self.heads = heads
        self.start = start
        self.stop = stop
        self.last_collected = None

    def fixed(self, action: torch.Tensor, heads: list[int], start: int, stop: int) -> None:
        self.mode = "fixed"
        self.fixed_action = action
        self.heads = heads
        self.start = start
        self.stop = stop

    def adaptive(
        self,
        *,
        heads: list[int],
        directions: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        targets: np.ndarray,
        alpha: float,
        ridge_ratio: float,
        relative_cap: float | None,
        start: int,
        stop: int,
    ) -> None:
        self.mode = "adaptive"
        self.heads = heads
        self.directions = torch.from_numpy(directions).float()
        self.means = torch.from_numpy(means).float()
        self.stds = torch.from_numpy(stds).float()
        self.targets = torch.from_numpy(targets).float()
        self.alpha = alpha
        self.ridge_ratio = ridge_ratio
        self.relative_cap = relative_cap
        self.start = start
        self.stop = stop

    def precomputed(self, action: torch.Tensor, heads: list[int], start: int, stop: int) -> None:
        self.mode = "precomputed"
        self.fixed_action = action
        self.heads = heads
        self.start = start
        self.stop = stop

    def __call__(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        if self.mode == "off":
            return None
        source = inputs[0]
        start_value = self.start if self.start >= 0 else source.shape[1] + self.start
        stop_value = self.stop if self.stop >= 0 else source.shape[1] + self.stop
        start = max(0, min(start_value, source.shape[1]))
        stop = max(start, min(stop_value, source.shape[1]))
        if start == stop:
            return None
        output = source.clone()

        if self.mode == "collect":
            self.last_collected = torch.stack(
                [
                    source[:, start:stop, head * self.head_dim : (head + 1) * self.head_dim]
                    for head in self.heads
                ],
                dim=2,
            ).detach().float()
            return None

        if self.mode in ("fixed", "precomputed"):
            if self.fixed_action is None:
                raise RuntimeError("Fixed mode has no action")
            action = self.fixed_action.to(device=source.device, dtype=source.dtype)
            if action.ndim == 1:
                expanded_action = action.view(1, 1, -1).expand(1, stop - start, -1)
            else:
                expanded_action = action.view(1, stop - start, -1)
            output[:, start:stop] += expanded_action
            state_sq = torch.zeros((), device=source.device)
            action_sq = torch.zeros((), device=source.device)
            for head in self.heads:
                left = head * self.head_dim
                right = (head + 1) * self.head_dim
                state_sq += source[:, start:stop, left:right].float().square().sum()
                action_sq += expanded_action[:, :, left:right].float().square().sum()
            self.last_state_sq = state_sq
            self.last_action_sq = action_sq
            count = (stop - start) * len(self.heads)
            self.last_active = torch.tensor(float(count), device=source.device)
            self.last_count = count
            self.last_pre_error = None
            self.last_post_error = None
            return (output, *inputs[1:])

        if any(value is None for value in (self.directions, self.means, self.stds, self.targets)):
            raise RuntimeError("Adaptive mode is missing statistic parameters")
        directions = self.directions.to(device=source.device)
        means = self.means.to(device=source.device)
        stds = self.stds.to(device=source.device)
        targets = self.targets.to(device=source.device)
        state_sq = torch.zeros((), device=source.device)
        action_sq = torch.zeros((), device=source.device)
        active = torch.zeros((), device=source.device)
        pre_error = torch.zeros((), device=source.device)
        post_error = torch.zeros((), device=source.device)
        count = 0

        for index, head in enumerate(self.heads):
            left = head * self.head_dim
            right = (head + 1) * self.head_dim
            state = source[:, start:stop, left:right].float()
            direction = directions[index]
            gradient = direction / stds[index]
            score = (torch.einsum("btd,d->bt", state, direction) - means[index]) / stds[index]
            gap = torch.relu(targets[index] - score)
            requested = self.alpha * gap
            gradient_norm_sq = gradient.square().sum().clamp_min(1e-12)
            delta = requested.unsqueeze(-1) * gradient / (
                gradient_norm_sq * (1.0 + self.ridge_ratio)
            )
            if self.relative_cap is not None:
                cap = self.relative_cap * state.norm(dim=-1, keepdim=True)
                scale = torch.clamp(cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
                delta = delta * scale
            output[:, start:stop, left:right] += delta.to(dtype=source.dtype)
            realized = torch.einsum("btd,d->bt", delta, gradient)
            post_score = score + realized
            state_sq += state.square().sum()
            action_sq += delta.square().sum()
            active += (requested > 0).float().sum()
            pre_error += torch.relu(targets[index] - score).sum()
            post_error += torch.relu(targets[index] - post_score).sum()
            count += score.numel()

        self.last_state_sq = state_sq
        self.last_action_sq = action_sq
        self.last_active = active
        self.last_count = count
        self.last_pre_error = pre_error
        self.last_post_error = post_error
        return (output, *inputs[1:])


def build_settings(args: argparse.Namespace) -> list[Setting]:
    if args.settings_file is not None:
        payload = json.loads(args.settings_file.read_text())
        settings = [Setting(**item) for item in payload]
        if any(setting.num_heads != args.num_heads for setting in settings):
            raise ValueError("Every explicit setting must match --num-heads")
        if any(
            setting.probe_score_normalization == "raw"
            and setting.method
            not in ("group_direction_probe_iti", "group_direction_probe_min_norm")
            for setting in settings
        ):
            raise ValueError("Raw probe scores are supported only for group-direction methods")
        return settings
    settings: list[Setting] = []
    if "fixed_com" in args.methods:
        settings.extend(
            Setting(method="fixed_com", num_heads=args.num_heads, alpha=alpha)
            for alpha in args.fixed_alphas
        )
    for method in (
        "adaptive_com",
        "adaptive_probe",
        "aggregate_com",
        "aggregate_probe",
        "targeted_iti",
        "targeted_probe_iti",
        "bounded_targeted_probe_iti",
        "headwise_probe_iti",
        "headwise_probe_min_norm",
        "group_direction_probe_iti",
        "group_direction_probe_min_norm",
    ):
        if method not in args.methods:
            continue
        settings.extend(
            Setting(
                method=method,
                num_heads=args.num_heads,
                alpha=alpha,
                target_quantile=quantile,
                ridge_ratio=ridge,
                relative_cap=cap,
            )
            for quantile in args.target_quantiles
            for alpha in args.strengths
            for ridge in args.ridge_ratios
            for cap in args.relative_caps
        )
    return settings


def prepare_statistics(
    *,
    head_activations: np.ndarray,
    labels: np.ndarray,
    tuning_activations: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    development_indices: np.ndarray,
    num_layers: int,
    num_heads: int,
    num_selected_heads: int,
    seed: int,
) -> StatisticBank:
    separated_x, separated_y, _ = get_separated_activations(labels, head_activations)
    top_heads, probes = get_top_heads(
        train_indices,
        validation_indices,
        separated_x,
        separated_y,
        num_layers,
        num_heads,
        seed,
        num_selected_heads,
        False,
    )
    top_heads = [(int(layer), int(head)) for layer, head in top_heads]
    com_all = get_com_directions(
        num_layers,
        num_heads,
        train_indices,
        validation_indices,
        separated_x,
        separated_y,
    )
    development_x = np.concatenate([separated_x[index] for index in development_indices])
    development_y = np.concatenate([separated_y[index] for index in development_indices])
    training_x = np.concatenate([separated_x[index] for index in train_indices])
    training_y = np.concatenate([separated_y[index] for index in train_indices])

    com_directions = []
    fixed_scales = []
    com_raw = []
    probe_directions = []
    probe_intercepts = []
    probe_raw = []
    probe_training_raw = []
    validation_accuracies = []
    validation_x = np.concatenate([separated_x[index] for index in validation_indices])
    validation_y = np.concatenate([separated_y[index] for index in validation_indices])
    for layer, head in top_heads:
        flat = layer_head_to_flattened_idx(layer, head, num_heads)
        com = np.asarray(com_all[flat], dtype=np.float32)
        com /= max(float(np.linalg.norm(com)), 1e-12)
        probe = probes[flat]
        probe_direction = np.asarray(probe.coef_, dtype=np.float32).reshape(-1)
        intercept = float(probe.intercept_.item())
        selected = np.asarray(development_x[:, layer, head, :], dtype=np.float32)
        selected_training = np.asarray(training_x[:, layer, head, :], dtype=np.float32)
        com_directions.append(com)
        fixed_scales.append(float(np.std(tuning_activations[:, layer, head, :] @ com)))
        com_raw.append(selected @ com)
        probe_directions.append(probe_direction)
        probe_intercepts.append(intercept)
        probe_raw.append(selected @ probe_direction + intercept)
        probe_training_raw.append(selected_training @ probe_direction + intercept)
        validation_accuracies.append(
            float(probe.score(validation_x[:, layer, head, :], validation_y))
        )

    com_raw_array = np.stack(com_raw, axis=1)
    probe_raw_array = np.stack(probe_raw, axis=1)
    com_means = com_raw_array.mean(axis=0).astype(np.float32)
    com_stds = np.maximum(com_raw_array.std(axis=0), 1e-6).astype(np.float32)
    probe_means = probe_raw_array.mean(axis=0).astype(np.float32)
    probe_stds = np.maximum(probe_raw_array.std(axis=0), 1e-6).astype(np.float32)
    com_standardized = (com_raw_array - com_means) / com_stds
    probe_standardized = (probe_raw_array - probe_means) / probe_stds
    probe_training_array = np.stack(probe_training_raw, axis=1)
    probe_training_standardized = (probe_training_array - probe_means) / probe_stds
    positive_mask = training_y == 1
    negative_mask = training_y == 0
    probe_positive = probe_training_standardized[positive_mask]
    probe_negative = probe_training_standardized[negative_mask]
    probe_raw_positive = probe_training_array[positive_mask]
    probe_raw_negative = probe_training_array[negative_mask]
    if not len(probe_positive) or not len(probe_negative):
        raise ValueError("Both positive and negative training candidates are required")
    probe_group_direction = probe_positive.mean(axis=0) - probe_negative.mean(axis=0)
    group_norm = max(float(np.linalg.norm(probe_group_direction)), 1e-12)
    probe_group_unit = probe_group_direction / group_norm
    probe_raw_group_direction = (
        probe_raw_positive.mean(axis=0) - probe_raw_negative.mean(axis=0)
    )
    raw_group_norm = max(float(np.linalg.norm(probe_raw_group_direction)), 1e-12)
    probe_raw_group_unit = probe_raw_group_direction / raw_group_norm
    return StatisticBank(
        top_heads=top_heads,
        com_directions=np.stack(com_directions),
        fixed_scales=np.asarray(fixed_scales, dtype=np.float32),
        com_means=com_means,
        com_stds=com_stds,
        com_truthful=com_standardized[development_y == 1],
        probe_directions=np.stack(probe_directions),
        probe_intercepts=np.asarray(probe_intercepts, dtype=np.float32),
        probe_means=probe_means,
        probe_stds=probe_stds,
        probe_truthful=probe_standardized[development_y == 1],
        probe_group_direction=probe_group_direction.astype(np.float32),
        probe_group_positive_projection=(probe_positive @ probe_group_unit).astype(np.float32),
        probe_group_negative_projection=(probe_negative @ probe_group_unit).astype(np.float32),
        validation_accuracies=np.asarray(validation_accuracies, dtype=np.float32),
        probe_raw_group_direction=probe_raw_group_direction.astype(np.float32),
        probe_raw_group_positive_projection=(
            probe_raw_positive @ probe_raw_group_unit
        ).astype(np.float32),
        probe_raw_group_negative_projection=(
            probe_raw_negative @ probe_raw_group_unit
        ).astype(np.float32),
    )


def save_statistics(path: Path, bank: StatisticBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        top_heads=np.asarray(bank.top_heads, dtype=np.int16),
        com_directions=bank.com_directions,
        fixed_scales=bank.fixed_scales,
        com_means=bank.com_means,
        com_stds=bank.com_stds,
        com_truthful=bank.com_truthful,
        probe_directions=bank.probe_directions,
        probe_intercepts=bank.probe_intercepts,
        probe_means=bank.probe_means,
        probe_stds=bank.probe_stds,
        probe_truthful=bank.probe_truthful,
        probe_group_direction=bank.probe_group_direction,
        probe_group_positive_projection=bank.probe_group_positive_projection,
        probe_group_negative_projection=bank.probe_group_negative_projection,
        validation_accuracies=bank.validation_accuracies,
        probe_raw_group_direction=bank.probe_raw_group_direction,
        probe_raw_group_positive_projection=bank.probe_raw_group_positive_projection,
        probe_raw_group_negative_projection=bank.probe_raw_group_negative_projection,
    )


def load_statistics(path: Path) -> StatisticBank:
    cached = np.load(path)
    return StatisticBank(
        top_heads=[tuple(map(int, row)) for row in cached["top_heads"]],
        com_directions=cached["com_directions"],
        fixed_scales=cached["fixed_scales"],
        com_means=cached["com_means"],
        com_stds=cached["com_stds"],
        com_truthful=cached["com_truthful"],
        probe_directions=cached["probe_directions"],
        probe_intercepts=cached["probe_intercepts"],
        probe_means=cached["probe_means"],
        probe_stds=cached["probe_stds"],
        probe_truthful=cached["probe_truthful"],
        probe_group_direction=(
            cached["probe_group_direction"]
            if "probe_group_direction" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
        probe_group_positive_projection=(
            cached["probe_group_positive_projection"]
            if "probe_group_positive_projection" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
        probe_group_negative_projection=(
            cached["probe_group_negative_projection"]
            if "probe_group_negative_projection" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
        validation_accuracies=cached["validation_accuracies"],
        probe_raw_group_direction=(
            cached["probe_raw_group_direction"]
            if "probe_raw_group_direction" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
        probe_raw_group_positive_projection=(
            cached["probe_raw_group_positive_projection"]
            if "probe_raw_group_positive_projection" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
        probe_raw_group_negative_projection=(
            cached["probe_raw_group_negative_projection"]
            if "probe_raw_group_negative_projection" in cached.files
            else np.empty(0, dtype=np.float32)
        ),
    )


def layer_indices(bank: StatisticBank) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, (layer, _head) in enumerate(bank.top_heads):
        result.setdefault(layer, []).append(index)
    return result


def collect_baseline_states(
    controllers: dict[int, AttentionHeadController],
    bank: StatisticBank,
) -> torch.Tensor:
    """Return baseline states as [causal positions, selected heads, head dim]."""

    grouped = layer_indices(bank)
    states = []
    for index, (layer, _head) in enumerate(bank.top_heads):
        collected = controllers[layer].last_collected
        if collected is None:
            raise RuntimeError("Baseline state collection did not run")
        local_index = grouped[layer].index(index)
        states.append(collected[0, :, local_index])
    return torch.stack(states, dim=1)


def active_target_diagnostics(
    *,
    gaps: torch.Tensor,
    post_scores: torch.Tensor,
    target: torch.Tensor,
    clipped: torch.Tensor,
) -> dict[str, float]:
    """Measure signed target error only where a correction was requested."""

    active = gaps > 0
    if not bool(active.any()):
        return {
            "active_signed_target_error": 0.0,
            "active_absolute_target_error": 0.0,
            "active_target_overshoot": 0.0,
            "clip_rate": 0.0,
        }
    signed = (post_scores - target).expand_as(gaps)
    return {
        "active_signed_target_error": float(signed[active].mean().item()),
        "active_absolute_target_error": float(signed[active].abs().mean().item()),
        "active_target_overshoot": float(torch.relu(signed[active]).mean().item()),
        "clip_rate": float(clipped[active].float().mean().item()),
    }


def build_aggregate_actions(
    *,
    controllers: dict[int, AttentionHeadController],
    bank: StatisticBank,
    setting: Setting,
    hidden_size: int,
    head_dim: int,
) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
    """Solve a global minimum-norm statistic correction at each causal token."""

    if setting.target_quantile is None or setting.ridge_ratio is None:
        raise ValueError("Aggregate settings require target and ridge parameters")
    states = collect_baseline_states(controllers, bank).float()
    if setting.method in ("aggregate_com", "targeted_iti"):
        directions = torch.from_numpy(bank.com_directions).to(states.device).float()
        means = torch.from_numpy(bank.com_means).to(states.device).float()
        stds = torch.from_numpy(bank.com_stds).to(states.device).float()
        truthful = bank.com_truthful
    elif setting.method in (
        "aggregate_probe",
        "targeted_probe_iti",
        "bounded_targeted_probe_iti",
        "headwise_probe_iti",
        "headwise_probe_min_norm",
        "group_direction_probe_iti",
        "group_direction_probe_min_norm",
    ):
        directions = torch.from_numpy(bank.probe_directions).to(states.device).float()
        means = torch.from_numpy(bank.probe_means - bank.probe_intercepts).to(states.device).float()
        stds = torch.from_numpy(bank.probe_stds).to(states.device).float()
        truthful = bank.probe_truthful
    else:
        raise ValueError(f"Unsupported aggregate method: {setting.method}")

    if setting.method in (
        "group_direction_probe_iti",
        "group_direction_probe_min_norm",
    ):
        if setting.probe_score_normalization == "raw":
            group_direction_array = bank.probe_raw_group_direction
            positive_projection = bank.probe_raw_group_positive_projection
            intercepts = torch.from_numpy(bank.probe_intercepts).to(states.device).float()
            per_head_scores = (
                torch.einsum("tkd,kd->tk", states, directions) + intercepts[None, :]
            )
            gradients = directions
        else:
            group_direction_array = bank.probe_group_direction
            positive_projection = bank.probe_group_positive_projection
            per_head_scores = (
                torch.einsum("tkd,kd->tk", states, directions) - means
            ) / stds
            gradients = directions / stds[:, None]
        if group_direction_array.shape != (len(bank.top_heads),):
            raise ValueError(
                "The statistics cache lacks the requested probe group direction; "
                "regenerate it with the current code"
            )
        if not len(positive_projection):
            raise ValueError("The statistics cache lacks the requested positive projections")
        group_direction = torch.from_numpy(group_direction_array).to(states.device).float()
        group_unit = group_direction / group_direction.norm().clamp_min(1e-12)
        projections = torch.einsum("tk,k->t", per_head_scores, group_unit)
        target = float(np.quantile(positive_projection, setting.target_quantile))
        target_tensor = torch.tensor(target, device=states.device)
        gaps = torch.relu(target_tensor - projections)
        requested = setting.alpha * gaps
        desired_score_change = requested[:, None] * group_unit[None, :]
        if setting.method == "group_direction_probe_iti":
            basis = torch.from_numpy(
                bank.fixed_scales[:, None] * bank.com_directions
            ).to(states.device).float()
            responses = torch.einsum("kd,kd->k", gradients, basis).clamp_min(1e-12)
            deltas = desired_score_change[:, :, None] * basis[None, :, :] / (
                responses[None, :, None] * (1.0 + setting.ridge_ratio)
            )
        else:
            gradient_norm_sq = gradients.square().sum(dim=1).clamp_min(1e-12)
            deltas = desired_score_change[:, :, None] * gradients[None, :, :] / (
                gradient_norm_sq[None, :, None] * (1.0 + setting.ridge_ratio)
            )
        clipped = torch.zeros_like(gaps, dtype=torch.bool)
        if setting.relative_cap is not None:
            cap = setting.relative_cap * states.flatten(1).norm(dim=1, keepdim=True)
            delta_norm = deltas.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(cap / delta_norm, max=1.0)
            deltas = deltas * scale[:, :, None]
            clipped = (scale.squeeze(1) < 1.0 - 1e-6) & (gaps > 0)

        realized_score_change = torch.einsum("tkd,kd->tk", deltas, gradients)
        post_scores = per_head_scores + realized_score_change
        post_projections = torch.einsum("tk,k->t", post_scores, group_unit)
        actions = {
            layer: torch.zeros(
                states.shape[0],
                hidden_size,
                dtype=torch.float32,
                device=states.device,
            )
            for layer in controllers
        }
        for index, (layer, head) in enumerate(bank.top_heads):
            left = head * head_dim
            right = (head + 1) * head_dim
            actions[layer][:, left:right] = deltas[:, index]
        metrics = {
            "relative_action_norm": float(
                deltas.norm().item() / max(states.norm().item(), 1e-12)
            ),
            "intervention_rate": float((requested > 0).float().mean().item()),
            "pre_target_error": float(gaps.mean().item()),
            "post_target_error": float(
                torch.relu(target_tensor - post_projections).mean().item()
            ),
            **active_target_diagnostics(
                gaps=gaps,
                post_scores=post_projections,
                target=target_tensor,
                clipped=clipped,
            ),
        }
        return actions, metrics

    per_head_scores = (torch.einsum("tkd,kd->tk", states, directions) - means) / stds
    if setting.method in ("headwise_probe_iti", "headwise_probe_min_norm"):
        targets = torch.from_numpy(
            np.quantile(truthful, setting.target_quantile, axis=0).astype(np.float32)
        ).to(states.device)
        gaps = torch.relu(targets[None, :] - per_head_scores)
        requested = setting.alpha * gaps
        gradients = directions / stds[:, None]
        if setting.method == "headwise_probe_iti":
            basis = torch.from_numpy(
                bank.fixed_scales[:, None] * bank.com_directions
            ).to(states.device).float()
            responses = torch.einsum("kd,kd->k", gradients, basis).clamp_min(1e-12)
            deltas = requested[:, :, None] * basis[None, :, :] / (
                responses[None, :, None] * (1.0 + setting.ridge_ratio)
            )
        else:
            gradient_norm_sq = gradients.square().sum(dim=1).clamp_min(1e-12)
            deltas = requested[:, :, None] * gradients[None, :, :] / (
                gradient_norm_sq[None, :, None] * (1.0 + setting.ridge_ratio)
            )
        clipped = torch.zeros_like(gaps, dtype=torch.bool)
        if setting.relative_cap is not None:
            cap = setting.relative_cap * states.flatten(1).norm(dim=1, keepdim=True)
            delta_norm = deltas.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(cap / delta_norm, max=1.0)
            deltas = deltas * scale[:, :, None]
            clipped = (scale < 1.0 - 1e-6).expand_as(gaps) & (gaps > 0)

        realized = torch.einsum("tkd,kd->tk", deltas, gradients)
        post_scores = per_head_scores + realized
        actions = {
            layer: torch.zeros(
                states.shape[0],
                hidden_size,
                dtype=torch.float32,
                device=states.device,
            )
            for layer in controllers
        }
        for index, (layer, head) in enumerate(bank.top_heads):
            left = head * head_dim
            right = (head + 1) * head_dim
            actions[layer][:, left:right] = deltas[:, index]
        metrics = {
            "relative_action_norm": float(
                deltas.norm().item() / max(states.norm().item(), 1e-12)
            ),
            "intervention_rate": float((requested > 0).float().mean().item()),
            "pre_target_error": float(gaps.mean().item()),
            "post_target_error": float(
                torch.relu(targets[None, :] - post_scores).mean().item()
            ),
            **active_target_diagnostics(
                gaps=gaps,
                post_scores=post_scores,
                target=targets[None, :],
                clipped=clipped,
            ),
        }
        return actions, metrics

    scores = per_head_scores.mean(dim=1)
    target = float(
        np.quantile(truthful.mean(axis=1), setting.target_quantile)
    )
    target_tensor = torch.tensor(target, device=states.device)
    gaps = torch.relu(target_tensor - scores)
    requested = setting.alpha * gaps
    gradients = directions / (len(bank.top_heads) * stds[:, None])
    gradient_norm_sq = gradients.square().sum().clamp_min(1e-12)
    if setting.method in (
        "targeted_iti",
        "targeted_probe_iti",
        "bounded_targeted_probe_iti",
    ):
        basis = torch.from_numpy(
            bank.fixed_scales[:, None] * bank.com_directions
        ).to(states.device).float()
        response = torch.einsum("kd,kd->", gradients, basis).clamp_min(1e-12)
        coefficients = requested / (response * (1.0 + setting.ridge_ratio))
        clipped = torch.zeros_like(gaps, dtype=torch.bool)
        if setting.coefficient_cap is not None:
            clipped = coefficients > setting.coefficient_cap
            coefficients = coefficients.clamp(max=setting.coefficient_cap)
        deltas = coefficients[:, None, None] * basis[None, :, :]
    else:
        clipped = torch.zeros_like(gaps, dtype=torch.bool)
        deltas = requested[:, None, None] * gradients[None, :, :] / (
            gradient_norm_sq * (1.0 + setting.ridge_ratio)
        )
    if setting.relative_cap is not None:
        cap = setting.relative_cap * states.flatten(1).norm(dim=1, keepdim=True)
        delta_norm = deltas.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(cap / delta_norm, max=1.0)
        deltas = deltas * scale[:, :, None]
        clipped = clipped | (scale.squeeze(1) < 1.0 - 1e-6)

    realized = torch.einsum("tkd,kd->t", deltas, gradients)
    post_scores = scores + realized
    actions = {
        layer: torch.zeros(
            states.shape[0],
            hidden_size,
            dtype=torch.float32,
            device=states.device,
        )
        for layer in controllers
    }
    for index, (layer, head) in enumerate(bank.top_heads):
        left = head * head_dim
        right = (head + 1) * head_dim
        actions[layer][:, left:right] = deltas[:, index]
    metrics = {
        "relative_action_norm": float(
            deltas.norm().item() / max(states.norm().item(), 1e-12)
        ),
        "intervention_rate": float((requested > 0).float().mean().item()),
        "pre_target_error": float(torch.relu(target_tensor - scores).mean().item()),
        "post_target_error": float(
            torch.relu(target_tensor - post_scores).mean().item()
        ),
        **active_target_diagnostics(
            gaps=gaps,
            post_scores=post_scores,
            target=target_tensor,
            clipped=clipped,
        ),
    }
    return actions, metrics


def configure_controllers(
    controllers: dict[int, AttentionHeadController],
    bank: StatisticBank,
    setting: Setting,
    start: int,
    stop: int,
) -> None:
    grouped = layer_indices(bank)
    if setting.method == "fixed_com":
        for layer, controller in controllers.items():
            action = torch.zeros(controller.hidden_size, dtype=torch.float32)
            for index in grouped[layer]:
                _layer, head = bank.top_heads[index]
                left = head * controller.head_dim
                right = (head + 1) * controller.head_dim
                action[left:right] = torch.from_numpy(
                    setting.alpha * bank.fixed_scales[index] * bank.com_directions[index]
                )
            controller.fixed(
                action,
                [bank.top_heads[index][1] for index in grouped[layer]],
                start,
                stop,
            )
        return

    if setting.target_quantile is None or setting.ridge_ratio is None:
        raise ValueError("Adaptive settings require target and ridge parameters")
    if setting.method not in ("adaptive_com", "adaptive_probe"):
        raise ValueError(f"Unsupported per-head adaptive method: {setting.method}")
    if setting.method == "adaptive_com":
        directions = bank.com_directions
        means = bank.com_means
        stds = bank.com_stds
        truthful = bank.com_truthful
    else:
        directions = bank.probe_directions
        means = bank.probe_means - bank.probe_intercepts
        stds = bank.probe_stds
        truthful = bank.probe_truthful
    targets = np.quantile(truthful, setting.target_quantile, axis=0).astype(np.float32)
    for layer, controller in controllers.items():
        indices = grouped[layer]
        controller.adaptive(
            heads=[bank.top_heads[index][1] for index in indices],
            directions=directions[indices],
            means=means[indices],
            stds=stds[indices],
            targets=targets[indices],
            alpha=setting.alpha,
            ridge_ratio=setting.ridge_ratio,
            relative_cap=setting.relative_cap,
            start=start,
            stop=stop,
        )


def diagnostics(controllers: dict[int, AttentionHeadController]) -> dict[str, float]:
    state_sq = 0.0
    action_sq = 0.0
    active = 0.0
    count = 0
    pre_error = 0.0
    post_error = 0.0
    for controller in controllers.values():
        if controller.last_state_sq is None or controller.last_action_sq is None:
            continue
        state_sq += float(controller.last_state_sq.item())
        action_sq += float(controller.last_action_sq.item())
        if controller.last_active is not None:
            active += float(controller.last_active.item())
        count += controller.last_count
        if controller.last_pre_error is not None:
            pre_error += float(controller.last_pre_error.item())
        if controller.last_post_error is not None:
            post_error += float(controller.last_post_error.item())
    return {
        "relative_action_norm": float(np.sqrt(action_sq) / max(np.sqrt(state_sq), 1e-12)),
        "intervention_rate": active / max(count, 1),
        "pre_target_error": pre_error / max(count, 1),
        "post_target_error": post_error / max(count, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    settings = build_settings(args)

    frame = load_truthfulqa_frame()
    folds = list(np.array_split(np.arange(len(frame)), 2))
    train_indices, validation_indices, development_indices = fold_train_val_indices(
        args.fold, folds, args.seed, args.val_ratio
    )
    eval_indices = validation_indices if args.eval_split == "validation" else folds[args.fold]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    if args.statistics_cache is not None and args.statistics_cache.exists():
        bank = load_statistics(args.statistics_cache)
        if len(bank.top_heads) != args.num_heads:
            raise ValueError("Statistics cache has a different selected-head count")
    else:
        head_activations = rearrange(
            np.load(f"{args.feature_prefix}_tqa_mc2_head_wise.npy", mmap_mode="r"),
            "b l (h d) -> b l h d",
            h=num_heads,
        )
        labels = np.load(f"{args.feature_prefix}_tqa_mc2_labels.npy")
        tuning_activations = rearrange(
            np.load(f"{args.feature_prefix}_tqa_gen_end_q_head_wise.npy", mmap_mode="r"),
            "b l (h d) -> b l h d",
            h=num_heads,
        )
        bank = prepare_statistics(
            head_activations=head_activations,
            labels=labels,
            tuning_activations=tuning_activations,
            train_indices=train_indices,
            validation_indices=validation_indices,
            development_indices=development_indices,
            num_layers=num_layers,
            num_heads=num_heads,
            num_selected_heads=args.num_heads,
            seed=args.seed,
        )
        if args.statistics_cache is not None:
            save_statistics(args.statistics_cache, bank)

    controllers = {
        layer: AttentionHeadController(head_dim, hidden_size)
        for layer, _head in bank.top_heads
    }
    handles = [
        model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(controller)
        for layer, controller in controllers.items()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fold_{args.fold}_{args.eval_split}_results.csv"
    summary_path = args.output_dir / f"fold_{args.fold}_{args.eval_split}_summary.csv"
    metadata_path = args.output_dir / f"fold_{args.fold}_{args.eval_split}_metadata.json"
    output = frame.iloc[eval_indices].copy().reset_index().rename(columns={"index": "dataset_index"})
    if args.question_offset:
        output = output.iloc[args.question_offset :].copy()
    if args.max_questions is not None:
        output = output.iloc[: args.max_questions].copy()
    if args.resume and output_path.exists():
        prior = pd.read_csv(output_path)
        if list(prior["dataset_index"]) != list(output["dataset_index"]):
            raise ValueError("Resume artifact does not match the requested evaluation rows")
        output = prior

    tags = ["baseline", *(setting.tag for setting in settings)]
    for tag in tags:
        set_columns(tag, output)
        if tag != "baseline":
            for suffix in DIAGNOSTIC_FIELDS:
                column = f"{tag} {suffix}"
                if column not in output:
                    output[column] = np.nan

    metadata = {
        "model_path": str(args.model_path),
        "feature_prefix": str(args.feature_prefix),
        "statistics_cache": str(args.statistics_cache) if args.statistics_cache else None,
        "fold": args.fold,
        "eval_split": args.eval_split,
        "question_offset": args.question_offset,
        "seed": args.seed,
        "settings": [asdict(setting) | {"tag": setting.tag} for setting in settings],
        "top_heads": bank.top_heads,
        "top_head_validation_accuracies": bank.validation_accuracies.tolist(),
        "scoring": "tokenizer-aware exact answer span",
        "application": "causal source positions predicting answer tokens",
        "api_used": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    try:
        for row_index in tqdm(output.index, desc=f"causal head fold {args.fold} {args.eval_split}"):
            if all(pd.notna(output.loc[row_index, f"{tag} MC2"]) for tag in tags):
                continue
            row = output.loc[row_index]
            true_answers = utilities.split_multi_answer(row[ANSWER_COL])
            false_answers = utilities.split_multi_answer(row[INCORRECT_COL])
            best_answer = utilities.format_best(row[BEST_COL])
            scores = {tag: [] for tag in tags}
            setting_diagnostics = {
                setting.tag: {
                    suffix: [] for suffix in DIAGNOSTIC_FIELDS
                }
                for setting in settings
            }

            for answer in [*true_answers, *false_answers]:
                prompt = DEFAULT_PREFIX + utilities.format_prompt_with_answer_strings(
                    str(row["Question"]), answer, "qa", format="general"
                )
                span = find_answer_token_span(tokenizer, prompt, answer)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                grouped = layer_indices(bank)
                for layer, controller in controllers.items():
                    controller.collect(
                        [bank.top_heads[index][1] for index in grouped[layer]],
                        span.causal_start,
                        span.answer_end - 1,
                    )
                with torch.inference_mode():
                    baseline_logits = model(input_ids=prompt_ids).logits
                scores["baseline"].append(score_answer_logits(baseline_logits, prompt_ids, span))

                for setting in settings:
                    aggregate_metrics = None
                    if setting.method in (
                        "aggregate_com",
                        "aggregate_probe",
                        "targeted_iti",
                        "targeted_probe_iti",
                        "bounded_targeted_probe_iti",
                        "headwise_probe_iti",
                        "headwise_probe_min_norm",
                        "group_direction_probe_iti",
                        "group_direction_probe_min_norm",
                    ):
                        actions, aggregate_metrics = build_aggregate_actions(
                            controllers=controllers,
                            bank=bank,
                            setting=setting,
                            hidden_size=hidden_size,
                            head_dim=head_dim,
                        )
                        for layer, controller in controllers.items():
                            controller.precomputed(
                                actions[layer],
                                [bank.top_heads[index][1] for index in grouped[layer]],
                                span.causal_start,
                                span.answer_end - 1,
                            )
                    else:
                        configure_controllers(
                            controllers,
                            bank,
                            setting,
                            span.causal_start,
                            span.answer_end - 1,
                        )
                    with torch.inference_mode():
                        steered_logits = model(input_ids=prompt_ids).logits
                    scores[setting.tag].append(score_answer_logits(steered_logits, prompt_ids, span))
                    current = aggregate_metrics or diagnostics(controllers)
                    for key, value in current.items():
                        setting_diagnostics[setting.tag][key].append(value)

            split = len(true_answers)
            for tag in tags:
                MC_calcs(
                    tag,
                    output,
                    row_index,
                    scores[tag][:split],
                    scores[tag][split:],
                    true_answers,
                    best_answer,
                )
            for setting in settings:
                for key, values in setting_diagnostics[setting.tag].items():
                    output.loc[row_index, f"{setting.tag} {key}"] = (
                        float(np.mean(values)) if values else np.nan
                    )
            if (row_index + 1) % args.checkpoint_every == 0:
                output.to_csv(output_path, index=False)
    finally:
        for handle in handles:
            handle.remove()

    output.to_csv(output_path, index=False)
    summaries = []
    for tag in tags:
        row = {
            "fold": args.fold,
            "eval_split": args.eval_split,
            "setting": tag,
            "n": len(output),
            "mc1": float(output[f"{tag} MC1"].mean()),
            "mc2": float(output[f"{tag} MC2"].mean()),
        }
        if tag != "baseline":
            for suffix in DIAGNOSTIC_FIELDS:
                row[suffix] = float(output[f"{tag} {suffix}"].mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values(["mc1", "mc2"], ascending=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
