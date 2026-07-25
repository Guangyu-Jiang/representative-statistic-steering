"""ReDeEP statistics, AARF, and target-conditioned mechanism control."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from .redeep_control import (
    LLAMA2_7B_DOLLY_CONFIG,
    LinearizedInverseDiagnostics,
    ReDeEPStatisticConfig,
    normalized_redeep_statistics,
    official_redeep_divergence,
    project_to_redeep_target,
    redeep_detector_score,
    solve_linearized_statistic_inverse,
)


@dataclass(frozen=True)
class ReDeEPGenerationConfig:
    method: str
    max_new_tokens: int = 64
    target_mode: str = "absolute"
    target_score: float = 0.4
    target_score_shift: float = 0.1
    ridge: float = 0.1
    finite_difference_epsilon: float = 0.05
    solver_damping: float = 1.0
    maximum_control_rms: float | None = 0.30
    maximum_control_abs: float | None = 0.60
    jacobian_refresh_interval: int = 0
    trigger_threshold: float | None = None
    intervention_positions: str = "last"

    def __post_init__(self) -> None:
        if self.method not in {"baseline", "fixed_aarf", "minimum_norm"}:
            raise ValueError(f"unsupported ReDeEP method: {self.method}")
        if self.intervention_positions not in {"last", "all"}:
            raise ValueError("intervention_positions must be 'last' or 'all'")
        if self.target_mode not in {"absolute", "relative"}:
            raise ValueError("target_mode must be 'absolute' or 'relative'")
        if self.target_score_shift < 0:
            raise ValueError("target_score_shift must be non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.finite_difference_epsilon <= 0:
            raise ValueError("finite_difference_epsilon must be positive")


@dataclass(frozen=True)
class ReDeEPStatistics:
    parameter_score: float
    external_score: float
    parameter_normalized: float
    external_normalized: float
    detector_score: float

    @property
    def normalized_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.parameter_normalized, self.external_normalized],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class ReDeEPStepDiagnostics:
    token_index: int
    triggered: bool
    baseline_statistics: ReDeEPStatistics
    controlled_statistics: ReDeEPStatistics
    target_parameter_normalized: float | None
    target_external_normalized: float | None
    target_error: float | None
    control: tuple[float, ...]
    control_rms: float
    final_hidden_relative_change: float
    jacobian_frobenius_norm: float | None
    jacobian_singular_values: tuple[float, ...] | None
    target_shift_norm: float | None
    achieved_target_fraction: float | None
    inverse: LinearizedInverseDiagnostics | None


@dataclass(frozen=True)
class DollyExample:
    source_id: int
    question: str
    passage: str
    reference_answer: str
    prompt: str
    hallucination_label: int


class ReDeEPMechanismController:
    """Expose ReDeEP's copying-head and Knowledge-FFN mechanisms as controls.

    The first coordinates scale individual copying-head outputs immediately
    before ``o_proj``. This is algebraically equivalent to the released AARF
    multiplication of those heads' post-softmax attention weights. Remaining
    coordinates scale the selected FFN outputs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        statistic_config: ReDeEPStatisticConfig = LLAMA2_7B_DOLLY_CONFIG,
        *,
        intervention_positions: str = "last",
    ) -> None:
        if intervention_positions not in {"last", "all"}:
            raise ValueError("intervention_positions must be 'last' or 'all'")
        self.model = model
        self.statistic_config = statistic_config
        self.intervention_positions = intervention_positions
        self.copy_heads = tuple(statistic_config.copy_heads)
        self.knowledge_layers = tuple(statistic_config.knowledge_layers)
        self.control_count = len(self.copy_heads) + len(self.knowledge_layers)
        self.control = torch.zeros(self.control_count, dtype=torch.float32)
        self.enabled = False
        self.prefix_hidden_state: torch.Tensor | None = None
        self.handles: list[Any] = []
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._attention_outputs: dict[int, torch.Tensor] = {}
        self._layer_outputs: dict[int, torch.Tensor] = {}
        self._attention_weights: dict[int, torch.Tensor] = {}
        self._final_hidden_state: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        layers = self.model.model.layers
        required_layers = set(self.knowledge_layers)
        required_layers.update(layer for layer, _ in self.copy_heads)
        if required_layers and max(required_layers) >= len(layers):
            raise ValueError(
                f"configuration requires layer {max(required_layers)}, "
                f"but model has {len(layers)} layers"
            )

        heads_by_layer: dict[int, list[tuple[int, int]]] = {}
        for control_index, (layer_index, head_index) in enumerate(self.copy_heads):
            heads_by_layer.setdefault(layer_index, []).append(
                (control_index, head_index)
            )
        ffn_control = {
            layer_index: len(self.copy_heads) + offset
            for offset, layer_index in enumerate(self.knowledge_layers)
        }

        for layer_index in required_layers:
            layer = layers[layer_index]
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._make_layer_pre_hook(layer_index)
                )
            )
            self.handles.append(
                layer.register_forward_hook(self._make_layer_hook(layer_index))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._make_attention_hook(layer_index)
                )
            )
            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._make_attention_output_hook(layer_index)
                )
            )
            if layer_index in heads_by_layer:
                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        self._make_head_control_hook(heads_by_layer[layer_index])
                    )
                )
            if layer_index in ffn_control:
                self.handles.append(
                    layer.mlp.register_forward_hook(
                        self._make_ffn_control_hook(ffn_control[layer_index])
                    )
                )
        self.handles.append(
            self.model.model.norm.register_forward_hook(self._capture_final_hidden)
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_capture(self) -> None:
        self._layer_inputs.clear()
        self._attention_outputs.clear()
        self._layer_outputs.clear()
        self._attention_weights.clear()
        self._final_hidden_state = None

    @property
    def final_hidden_state(self) -> torch.Tensor:
        if self._final_hidden_state is None:
            raise RuntimeError("final hidden state was not captured")
        return self._final_hidden_state

    def _capture_final_hidden(self, _module, _inputs, output):
        if torch.is_tensor(output):
            self._final_hidden_state = output
        return None

    def set_control(self, control: torch.Tensor | Sequence[float] | None) -> None:
        if control is None:
            self.control = torch.zeros(self.control_count, dtype=torch.float32)
            self.enabled = False
            return
        value = torch.as_tensor(control, dtype=torch.float32).reshape(-1)
        if value.numel() != self.control_count:
            raise ValueError(
                f"expected {self.control_count} controls, got {value.numel()}"
            )
        self.control = value
        self.enabled = bool(torch.any(value != 0))

    def set_prefix_hidden_state(
        self, hidden_state: torch.Tensor, prefix_length: int
    ) -> None:
        if hidden_state.ndim != 3 or hidden_state.shape[0] != 1:
            raise ValueError("ReDeEP currently supports batch size one")
        self.prefix_hidden_state = hidden_state[0, :prefix_length].detach().float()

    def _selected_positions(self, hidden: torch.Tensor) -> slice:
        return slice(None) if self.intervention_positions == "all" else slice(-1, None)

    def _make_layer_pre_hook(self, layer_index: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                self._layer_inputs[layer_index] = inputs[0]
            return None

        return hook

    def _make_layer_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self._layer_outputs[layer_index] = output
            return None

        return hook

    def _make_attention_output_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self._attention_outputs[layer_index] = output
            return None

        return hook

    def _make_attention_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                self._attention_weights[layer_index] = output[1]
            return None

        return hook

    def _make_head_control_hook(self, controls: list[tuple[int, int]]):
        def hook(module, inputs):
            if not self.enabled or not inputs or not torch.is_tensor(inputs[0]):
                return None
            hidden = inputs[0]
            if hidden.ndim != 3:
                return None
            num_heads = int(self.model.config.num_attention_heads)
            head_dim = hidden.shape[-1] // num_heads
            edited = hidden.clone()
            positions = self._selected_positions(hidden)
            for control_index, head_index in controls:
                start = head_index * head_dim
                end = start + head_dim
                scale = 1.0 + self.control[control_index].to(
                    device=hidden.device, dtype=hidden.dtype
                )
                edited[:, positions, start:end] = (
                    edited[:, positions, start:end] * scale
                )
            return (edited,) + inputs[1:]

        return hook

    def _make_ffn_control_hook(self, control_index: int):
        def hook(_module, _inputs, output):
            if not self.enabled or not torch.is_tensor(output) or output.ndim != 3:
                return None
            edited = output.clone()
            positions = self._selected_positions(output)
            scale = 1.0 + self.control[control_index].to(
                device=output.device, dtype=output.dtype
            )
            edited[:, positions, :] = edited[:, positions, :] * scale
            return edited

        return hook

    def statistics(self, final_hidden_state: torch.Tensor) -> tuple[torch.Tensor, ReDeEPStatistics]:
        """Compute normalized ``[PKS, ECS]`` and the detector score."""

        if final_hidden_state.ndim != 3 or final_hidden_state.shape[0] != 1:
            raise ValueError("ReDeEP currently supports batch size one")
        if self.prefix_hidden_state is None:
            raise RuntimeError("prefix hidden state has not been initialized")

        layer_scores: list[torch.Tensor] = []
        for layer_index in self.knowledge_layers:
            try:
                pre_ffn = (
                    self._layer_inputs[layer_index]
                    + self._attention_outputs[layer_index]
                )[:, -1, :]
                post_ffn = self._layer_outputs[layer_index][:, -1, :]
            except KeyError as error:
                raise RuntimeError(
                    f"missing capture for Knowledge FFN layer {layer_index}"
                ) from error
            post_logits = self.model.lm_head(self.model.model.norm(post_ffn)).float()
            pre_logits = self.model.lm_head(self.model.model.norm(pre_ffn)).float()
            layer_scores.append(official_redeep_divergence(post_logits, pre_logits))
        parameter_score = torch.stack(layer_scores, dim=0).mean(dim=0).reshape(())

        current_hidden = final_hidden_state[0, -1].float()
        prefix_hidden = self.prefix_hidden_state.to(current_hidden.device)
        similarities: list[torch.Tensor] = []
        for layer_index, head_index in self.copy_heads:
            if layer_index not in self._attention_weights:
                raise RuntimeError(
                    "attention weights were not returned; load Llama with "
                    "attn_implementation='eager'"
                )
            attention = self._attention_weights[layer_index]
            available = min(prefix_hidden.shape[0], attention.shape[-1])
            top_count = max(1, int(available * 0.1))
            weights = attention[0, head_index, -1, :available].float()
            indices = torch.topk(weights, k=top_count, largest=True).indices
            attended_hidden = prefix_hidden[indices].mean(dim=0)
            similarities.append(
                F.cosine_similarity(
                    attended_hidden.reshape(1, -1),
                    current_hidden.reshape(1, -1),
                    dim=-1,
                ).reshape(())
            )
        external_score = torch.stack(similarities).mean()
        normalized = normalized_redeep_statistics(
            parameter_score, external_score, self.statistic_config
        )
        detector = redeep_detector_score(normalized, self.statistic_config)
        record = ReDeEPStatistics(
            parameter_score=float(parameter_score.detach()),
            external_score=float(external_score.detach()),
            parameter_normalized=float(normalized[0].detach()),
            external_normalized=float(normalized[1].detach()),
            detector_score=float(detector.detach()),
        )
        return normalized, record


def clone_cache(cache: Any) -> Any:
    """Clone a Transformers cache because model forward mutates it in place."""

    return copy.deepcopy(cache)


def finite_difference_jacobian(
    evaluate: Callable[
        [torch.Tensor],
        tuple[Any, torch.Tensor, torch.Tensor, ReDeEPStatistics],
    ],
    base_control: torch.Tensor,
    base_statistic: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for index in range(base_control.numel()):
        perturbed = base_control.clone()
        perturbed[index] += float(epsilon)
        _, _, statistic, _ = evaluate(perturbed)
        columns.append((statistic - base_statistic) / float(epsilon))
    return torch.stack(columns, dim=-1)


def fixed_aarf_control(config: ReDeEPStatisticConfig) -> torch.Tensor:
    return torch.tensor(
        [config.aarf_attention_scale - 1.0] * len(config.copy_heads)
        + [config.aarf_ffn_scale - 1.0] * len(config.knowledge_layers),
        dtype=torch.float32,
    )


def _clip_control(
    control: torch.Tensor, maximum_absolute_value: float | None
) -> torch.Tensor:
    if maximum_absolute_value is None:
        return control
    if maximum_absolute_value < 0:
        raise ValueError("maximum_control_abs must be non-negative")
    return control.clamp(-maximum_absolute_value, maximum_absolute_value)


def generate_with_redeep(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    generation_config: ReDeEPGenerationConfig,
    statistic_config: ReDeEPStatisticConfig = LLAMA2_7B_DOLLY_CONFIG,
) -> tuple[str, dict[str, Any]]:
    """Greedily generate one response with online ReDeEP mechanism control.

    Prefill is intentionally left unchanged for every method. This makes the
    comparison isolate decode-time control and avoids changing the context
    stored in the KV cache before the first generated token.
    """

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    prefix_length = int(input_ids.shape[1])
    controller = ReDeEPMechanismController(
        model,
        statistic_config,
        intervention_positions=generation_config.intervention_positions,
    )
    trigger_threshold = (
        statistic_config.intervention_threshold
        if generation_config.trigger_threshold is None
        else generation_config.trigger_threshold
    )
    steps: list[ReDeEPStepDiagnostics] = []
    cached_jacobian: torch.Tensor | None = None
    generated: list[int] = []
    try:
        controller.reset_capture()
        controller.set_control(None)
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, use_cache=True, return_dict=True)
        controller.set_prefix_hidden_state(
            controller.final_hidden_state, prefix_length
        )
        next_token = int(prefill.logits[0, -1].argmax())
        generated.append(next_token)
        past = prefill.past_key_values

        eos_ids = tokenizer.eos_token_id
        eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else set(eos_ids or [])
        for token_index in range(1, generation_config.max_new_tokens):
            if generated[-1] in eos_set:
                break
            current_ids = torch.tensor(
                [[generated[-1]]], device=device, dtype=torch.long
            )

            def evaluate(control: torch.Tensor):
                candidate_cache = clone_cache(past)
                controller.reset_capture()
                controller.set_control(control)
                with torch.inference_mode():
                    output = model(
                        input_ids=current_ids,
                        past_key_values=candidate_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    final_hidden = controller.final_hidden_state.detach().clone()
                    statistic, record = controller.statistics(
                        final_hidden
                    )
                return output, final_hidden, statistic.detach().cpu(), record

            zero = torch.zeros(controller.control_count, dtype=torch.float32)
            (
                baseline_output,
                baseline_hidden_state,
                baseline_statistic,
                baseline_record,
            ) = evaluate(zero)
            triggered = baseline_record.detector_score > trigger_threshold
            control = zero
            target: torch.Tensor | None = None
            inverse: LinearizedInverseDiagnostics | None = None
            jacobian_frobenius_norm: float | None = None
            jacobian_singular_values: tuple[float, ...] | None = None
            if generation_config.method == "fixed_aarf" and triggered:
                control = fixed_aarf_control(statistic_config)
            elif generation_config.method == "minimum_norm" and triggered:
                target_arguments = (
                    {"target_score": generation_config.target_score}
                    if generation_config.target_mode == "absolute"
                    else {"score_shift": generation_config.target_score_shift}
                )
                target = project_to_redeep_target(
                    baseline_statistic, statistic_config, **target_arguments
                )
                refresh = cached_jacobian is None or (
                    generation_config.jacobian_refresh_interval > 0
                    and token_index % generation_config.jacobian_refresh_interval == 0
                )
                if refresh:
                    cached_jacobian = finite_difference_jacobian(
                        evaluate,
                        zero,
                        baseline_statistic,
                        epsilon=generation_config.finite_difference_epsilon,
                    )
                singular_values = torch.linalg.svdvals(cached_jacobian)
                jacobian_frobenius_norm = float(cached_jacobian.norm())
                jacobian_singular_values = tuple(
                    float(value) for value in singular_values
                )
                control, inverse = solve_linearized_statistic_inverse(
                    cached_jacobian,
                    zero,
                    baseline_statistic,
                    target,
                    ridge=generation_config.ridge,
                    damping=generation_config.solver_damping,
                    maximum_control_rms=generation_config.maximum_control_rms,
                )
                control = _clip_control(
                    control, generation_config.maximum_control_abs
                )

            if bool(torch.any(control != 0)):
                (
                    output,
                    controlled_hidden_state,
                    controlled_statistic,
                    controlled_record,
                ) = evaluate(control)
            else:
                output = baseline_output
                controlled_hidden_state = baseline_hidden_state
                controlled_statistic = baseline_statistic
                controlled_record = baseline_record

            baseline_hidden = baseline_hidden_state[0, -1].float()
            controlled_hidden = controlled_hidden_state[0, -1].float()
            relative_change = float(
                (controlled_hidden - baseline_hidden).norm()
                / baseline_hidden.norm().clamp_min(1e-12)
            )
            target_error = None
            target_shift_norm = None
            achieved_target_fraction = None
            if target is not None:
                target_error = float((controlled_statistic - target).norm())
                target_shift_norm = float((baseline_statistic - target).norm())
                achieved_target_fraction = 1.0 - target_error / max(
                    target_shift_norm, 1e-12
                )
            steps.append(
                ReDeEPStepDiagnostics(
                    token_index=token_index,
                    triggered=triggered,
                    baseline_statistics=baseline_record,
                    controlled_statistics=controlled_record,
                    target_parameter_normalized=(
                        None if target is None else float(target[0])
                    ),
                    target_external_normalized=(
                        None if target is None else float(target[1])
                    ),
                    target_error=target_error,
                    control=tuple(float(value) for value in control),
                    control_rms=float(control.square().mean().sqrt()),
                    final_hidden_relative_change=relative_change,
                    jacobian_frobenius_norm=jacobian_frobenius_norm,
                    jacobian_singular_values=jacobian_singular_values,
                    target_shift_norm=target_shift_norm,
                    achieved_target_fraction=achieved_target_fraction,
                    inverse=inverse,
                )
            )
            next_token = int(output.logits[0, -1].argmax())
            generated.append(next_token)
            past = output.past_key_values
    finally:
        controller.close()

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    triggered_steps = sum(step.triggered for step in steps)
    changed_steps = sum(step.control_rms > 0 for step in steps)
    diagnostics = {
        "method": generation_config.method,
        "generation_config": asdict(generation_config),
        "statistic_config": statistic_config.to_dict(),
        "prompt_tokens": prefix_length,
        "generated_tokens": len(generated),
        "evaluated_decode_steps": len(steps),
        "triggered_steps": triggered_steps,
        "changed_steps": changed_steps,
        "trigger_rate": triggered_steps / max(len(steps), 1),
        "mean_control_rms": float(np.mean([step.control_rms for step in steps]))
        if steps
        else 0.0,
        "mean_final_hidden_relative_change": float(
            np.mean([step.final_hidden_relative_change for step in steps])
        )
        if steps
        else 0.0,
        "mean_baseline_detector_score": float(
            np.mean([step.baseline_statistics.detector_score for step in steps])
        )
        if steps
        else float("nan"),
        "mean_controlled_detector_score": float(
            np.mean([step.controlled_statistics.detector_score for step in steps])
        )
        if steps
        else float("nan"),
        "mean_target_error": float(
            np.mean([step.target_error for step in steps if step.target_error is not None])
        )
        if any(step.target_error is not None for step in steps)
        else None,
        "mean_achieved_target_fraction": float(
            np.mean(
                [
                    step.achieved_target_fraction
                    for step in steps
                    if step.achieved_target_fraction is not None
                ]
            )
        )
        if any(step.achieved_target_fraction is not None for step in steps)
        else None,
        "steps": [asdict(step) for step in steps],
    }
    return response, diagnostics


def load_dolly_examples(
    source_path: str | Path,
    response_path: str | Path,
    *,
    response_model: str = "llama-2-7b-chat",
) -> list[DollyExample]:
    sources: dict[int, dict[str, Any]] = {}
    with Path(source_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            sources[int(item["source_id"])] = item
    labels: dict[int, int] = {}
    with Path(response_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("model") == response_model and item.get("split") == "test":
                labels[int(item["source_id"])] = int(bool(item.get("labels")))

    examples: list[DollyExample] = []
    for source_id in sorted(set(sources) & set(labels)):
        item = sources[source_id]
        info = item["source_info"]
        examples.append(
            DollyExample(
                source_id=source_id,
                question=str(info["question"]),
                passage=str(info["passages"]),
                reference_answer=str(item["human_response"]),
                prompt=str(item["prompt"]),
                hallucination_label=labels[source_id],
            )
        )
    if not examples:
        raise ValueError("no matching Dolly examples found")
    return examples


def split_dolly_examples(
    examples: Sequence[DollyExample],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[DollyExample], list[DollyExample]]:
    indices = np.arange(len(examples))
    labels = np.asarray([example.hallucination_label for example in examples])
    development, heldout = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return (
        [examples[int(index)] for index in sorted(development)],
        [examples[int(index)] for index in sorted(heldout)],
    )


def split_dolly_steering_evaluation(
    examples: Sequence[DollyExample],
    *,
    tuning_size: int = 6,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[DollyExample], list[DollyExample]]:
    """Return the frozen steering-tuning subset and all remaining examples."""

    development, _ = split_dolly_examples(
        examples, test_size=test_size, seed=seed
    )
    if tuning_size <= 0 or tuning_size >= len(examples):
        raise ValueError("tuning_size must be between 1 and len(examples)-1")
    tuning = development[:tuning_size]
    tuning_ids = {example.source_id for example in tuning}
    evaluation = [
        example for example in examples if example.source_id not in tuning_ids
    ]
    return tuning, evaluation


def format_llama2_chat_prompt(tokenizer: Any, prompt: str) -> str:
    prompt = prompt[:8000]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n"
        f"{prompt} [/INST]"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_examples(
    examples: Sequence[DollyExample],
    *,
    offset: int = 0,
    limit: int = 0,
    source_ids: Iterable[int] | None = None,
) -> list[DollyExample]:
    selected_ids = set(source_ids) if source_ids is not None else None
    selected = [
        example
        for example in examples
        if selected_ids is None or example.source_id in selected_ids
    ]
    selected = selected[offset:]
    return selected[:limit] if limit > 0 else selected
