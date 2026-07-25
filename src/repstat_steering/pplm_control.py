"""Modern PPLM and scalar minimum-norm control using the published SST head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PPLM_SENTIMENT_LABELS = {"positive": 2, "negative": 3}


class ClassificationHead(nn.Module):
    def __init__(self, class_size: int = 5, embed_size: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Linear(embed_size, class_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


@dataclass(frozen=True)
class PPLMGenerationConfig:
    method: str
    solver_version: str = "accumulated_v2"
    max_new_tokens: int = 24
    temperature: float = 1.0
    top_k: int = 10
    sample: bool = True
    horizon_length: int = 1
    gm_scale: float = 0.95
    kl_scale: float = 0.01
    pplm_steps: int = 5
    pplm_step_size: float = 0.04
    pplm_gamma: float = 1.0
    target_probability: float = 0.8
    target_margin_shift: float | None = None
    target_margin: float | None = None
    minimum_target_probability: float | None = None
    minimum_norm_steps: int = 3
    minimum_norm_damping: float = 1.0
    ridge: float = 0.1
    maximum_relative_norm: float | None = 0.10
    maximum_token_kl: float | None = None
    difficult_margin_threshold: float | None = None
    difficult_maximum_token_kl: float | None = None
    cache_component: str = "all"
    cache_last_n_layers: int | None = None
    statistic_mode: str = "margin"
    gradient_block_normalization: float = 0.0
    preserve_top_log_probs: int = 0
    log_probability_preservation_weight: float = 1.0
    persistent_cache: bool = False


def _legacy_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(tuple(item for item in layer) for layer in cache)


def _cache_add(
    cache: Sequence[Sequence[torch.Tensor]],
    delta: Sequence[Sequence[torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        tuple(base_item + delta_item for base_item, delta_item in zip(base_layer, delta_layer))
        for base_layer, delta_layer in zip(cache, delta)
    )


def _cache_norm(cache: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([item.float().square().sum() for layer in cache for item in layer]).sum().sqrt()


def _cache_zeros(
    cache: Sequence[Sequence[torch.Tensor]], requires_grad: bool = False
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        tuple(torch.zeros_like(item, requires_grad=requires_grad) for item in layer)
        for layer in cache
    )


def _detach_cache(
    cache: Sequence[Sequence[torch.Tensor]], requires_grad: bool = False
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        tuple(item.detach().requires_grad_(requires_grad) for item in layer)
        for layer in cache
    )


def _cache_map(function, cache):
    return tuple(tuple(function(item) for item in layer) for layer in cache)


def _cache_total_relative_norm(delta, source) -> float:
    return float(_cache_norm(delta) / _cache_norm(source).clamp_min(1e-12))


def _cap_cache_relative_norm(delta, source, maximum: float | None):
    if maximum is None:
        return delta
    ratio = _cache_norm(delta) / _cache_norm(source).clamp_min(1e-12)
    scale = min(1.0, maximum / max(float(ratio), 1e-12))
    return _cache_map(lambda item: item * scale, delta)


def _mask_cache_gradients(
    gradients: Sequence[torch.Tensor],
    cache: Sequence[Sequence[torch.Tensor]],
    component: str,
    last_n_layers: int | None,
) -> tuple[torch.Tensor, ...]:
    if component not in {"all", "key", "value"}:
        raise ValueError(f"Unknown cache component: {component}")
    first_layer = 0
    if last_n_layers is not None:
        first_layer = max(len(cache) - max(last_n_layers, 0), 0)

    selected = []
    gradient_index = 0
    for layer_index, layer in enumerate(cache):
        for item_index, _ in enumerate(layer):
            gradient = gradients[gradient_index]
            gradient_index += 1
            component_selected = (
                component == "all"
                or (component == "key" and item_index == 0)
                or (component == "value" and item_index == 1)
            )
            if layer_index < first_layer or not component_selected:
                gradient = torch.zeros_like(gradient)
            selected.append(gradient)
    return tuple(selected)


def _cache_gradient_dot(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> torch.Tensor:
    return torch.stack(
        [(left_item.float() * right_item.float()).sum() for left_item, right_item in zip(left, right)]
    ).sum()


def _top_k_probs(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return probs
    values, indices = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
    filtered = torch.zeros_like(probs).scatter(-1, indices, values)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-15)


def _categorical_kl(probabilities: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-15)
    reference = reference.clamp_min(1e-15)
    return (probabilities * (probabilities.log() - reference.log())).sum(dim=-1)


def _geometric_probability_mix(
    perturbed_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    temperature: float,
    maximum_scale: float,
    maximum_kl: float | None,
    search_steps: int = 12,
) -> tuple[torch.Tensor, float, float]:
    """Use the largest geometric-mixture scale inside a token-level KL ball."""
    perturbed_log_probs = (perturbed_logits / temperature).log_softmax(dim=-1)
    reference_log_probs = (reference_logits / temperature).log_softmax(dim=-1)
    reference_probabilities = reference_log_probs.exp()

    def mix(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        log_probabilities = (
            scale * perturbed_log_probs + (1.0 - scale) * reference_log_probs
        )
        probabilities = log_probabilities.softmax(dim=-1)
        return probabilities, _categorical_kl(probabilities, reference_probabilities)

    maximum_scale = min(max(float(maximum_scale), 0.0), 1.0)
    probabilities, divergence = mix(maximum_scale)
    if maximum_kl is None or float(divergence.max()) <= maximum_kl:
        return probabilities, maximum_scale, float(divergence.mean())

    low, high = 0.0, maximum_scale
    for _ in range(search_steps):
        midpoint = (low + high) / 2.0
        _, midpoint_divergence = mix(midpoint)
        if float(midpoint_divergence.max()) <= maximum_kl:
            low = midpoint
        else:
            high = midpoint
    probabilities, divergence = mix(low)
    return probabilities, low, float(divergence.mean())


def _select_token_kl_budget(
    margin: torch.Tensor | float,
    standard_budget: float | None,
    difficult_margin_threshold: float | None,
    difficult_budget: float | None,
) -> float | None:
    """Allocate more output-distribution change only for difficult states."""
    if difficult_margin_threshold is None and difficult_budget is None:
        return standard_budget
    if difficult_margin_threshold is None or difficult_budget is None:
        raise ValueError(
            "difficult_margin_threshold and difficult_maximum_token_kl must be set together"
        )
    if standard_budget is None:
        raise ValueError("maximum_token_kl is required for adaptive KL allocation")
    if standard_budget < 0 or difficult_budget < 0:
        raise ValueError("token KL budgets must be non-negative")
    return (
        difficult_budget
        if float(margin) < difficult_margin_threshold
        else standard_budget
    )


def resolve_margin_target(
    current_margin: torch.Tensor,
    *,
    target_probability: float,
    target_margin_shift: float | None = None,
    target_margin: float | None = None,
    minimum_target_probability: float | None = None,
) -> torch.Tensor:
    """Resolve an absolute, relative, or probability-derived margin target."""
    if target_margin is not None and target_margin_shift is not None:
        raise ValueError("target_margin and target_margin_shift are mutually exclusive")
    if target_margin is not None:
        target = torch.as_tensor(
            target_margin, device=current_margin.device, dtype=current_margin.dtype
        )
    elif target_margin_shift is not None:
        target = current_margin.detach() + target_margin_shift
    else:
        probability = min(max(target_probability, 1e-5), 1 - 1e-5)
        target = torch.tensor(
            probability / (1 - probability),
            device=current_margin.device,
            dtype=current_margin.dtype,
        ).log()
    if minimum_target_probability is not None:
        floor_probability = min(
            max(minimum_target_probability, 1e-5), 1 - 1e-5
        )
        floor_margin = torch.tensor(
            floor_probability / (1 - floor_probability),
            device=current_margin.device,
            dtype=current_margin.dtype,
        ).log()
        target = torch.maximum(target, floor_margin)
    return target


class PPLMSentimentExperiment:
    """Generate with the official PPLM statistic under matched interventions."""

    def __init__(
        self,
        model_path: str | Path,
        classifier_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.float32
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.classifier = ClassificationHead().to(self.device)
        state = torch.load(classifier_path, map_location=self.device, weights_only=True)
        self.classifier.load_state_dict(state)
        self.classifier.eval().requires_grad_(False)

    @staticmethod
    def _margin(classifier_logits: torch.Tensor, target_class: int) -> torch.Tensor:
        mask = torch.ones(
            classifier_logits.shape[-1], dtype=torch.bool, device=classifier_logits.device
        )
        mask[target_class] = False
        return classifier_logits[:, target_class] - torch.logsumexp(
            classifier_logits[:, mask], dim=-1
        )

    def _statistic_forward(
        self,
        base_cache,
        delta,
        last_token: torch.Tensor,
        accumulated_hidden: torch.Tensor,
        total_input_length: int,
        target_class: int,
        horizon_length: int,
        horizon_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        perturbed_cache = _cache_add(base_cache, delta)
        output = self.model(
            last_token,
            past_key_values=perturbed_cache,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[:, -1, :]
        hidden_sum = accumulated_hidden + output.hidden_states[-1][:, -1, :]
        denominator = total_input_length
        current_cache = (
            output.past_key_values if horizon_cache is None else horizon_cache
        )

        for _ in range(horizon_length):
            probabilities = logits.softmax(dim=-1)
            expected_embedding = probabilities @ self.model.get_input_embeddings().weight
            horizon_output = self.model(
                inputs_embeds=expected_embedding.unsqueeze(1),
                past_key_values=current_cache,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            logits = horizon_output.logits[:, -1, :]
            hidden_sum = hidden_sum + horizon_output.hidden_states[-1][:, -1, :]
            denominator += 1
            current_cache = horizon_output.past_key_values

        classifier_logits = self.classifier(hidden_sum / denominator)
        margin = self._margin(classifier_logits, target_class)
        return (
            margin,
            output.logits[:, -1, :],
            classifier_logits,
            _legacy_cache(output.past_key_values),
        )

    def _prepare_step(self, input_ids: torch.Tensor):
        if input_ids.shape[1] < 2:
            raise ValueError("PPLM requires a prefix containing at least two tokens")
        with torch.no_grad():
            prefix_output = self.model(
                input_ids[:, :-1],
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        cache = _detach_cache(_legacy_cache(prefix_output.past_key_values))
        accumulated = prefix_output.hidden_states[-1].sum(dim=1).detach()
        return cache, accumulated, input_ids[:, -1:]

    def _unperturbed_logits(
        self, cache, accumulated, last_token, total_length, target_class, horizon_length
    ):
        zero = _cache_zeros(cache)
        with torch.no_grad():
            return self._statistic_forward(
                cache,
                zero,
                last_token,
                accumulated,
                total_length,
                target_class,
                horizon_length,
            )

    def _pplm_delta(
        self,
        cache,
        accumulated,
        last_token,
        total_length,
        target_class,
        config: PPLMGenerationConfig,
        unperturbed_logits: torch.Tensor,
        horizon_cache=None,
    ):
        delta = _cache_zeros(cache)
        for _ in range(config.pplm_steps):
            variables = _detach_cache(delta, requires_grad=True)
            _, logits, classifier_logits, _ = self._statistic_forward(
                cache,
                variables,
                last_token,
                accumulated,
                total_length,
                target_class,
                config.horizon_length,
                horizon_cache=horizon_cache,
            )
            label = torch.full(
                (classifier_logits.shape[0],),
                target_class,
                device=self.device,
                dtype=torch.long,
            )
            loss = F.cross_entropy(classifier_logits, label)
            if config.kl_scale > 0:
                probabilities = logits.softmax(dim=-1).clamp_min(1e-15)
                base_probabilities = unperturbed_logits.softmax(dim=-1).clamp_min(1e-15)
                loss = loss + config.kl_scale * (
                    probabilities * (probabilities.log() - base_probabilities.log())
                ).sum()
            flat_variables = [item for layer in variables for item in layer]
            gradients = torch.autograd.grad(loss, flat_variables)
            updated = []
            gradient_index = 0
            for layer in variables:
                updated_layer = []
                for item in layer:
                    gradient = gradients[gradient_index]
                    gradient_index += 1
                    norm = gradient.norm().clamp_min(1e-15)
                    updated_layer.append(
                        item.detach()
                        - config.pplm_step_size
                        * gradient.detach()
                        / norm.pow(config.pplm_gamma)
                    )
                updated.append(tuple(updated_layer))
            delta = _cap_cache_relative_norm(tuple(updated), cache, config.maximum_relative_norm)
        return _detach_cache(delta)

    def _minimum_norm_delta(
        self,
        cache,
        accumulated,
        last_token,
        total_length,
        target_class,
        config: PPLMGenerationConfig,
        horizon_cache=None,
    ):
        if config.statistic_mode not in {"margin", "distribution"}:
            raise ValueError(f"Unknown statistic mode: {config.statistic_mode}")
        if config.statistic_mode != "margin" and config.target_margin is not None:
            raise ValueError("target_margin is supported only in margin statistic mode")
        if not 0 < config.minimum_norm_damping <= 1:
            raise ValueError("minimum_norm_damping must be in (0, 1]")
        delta = _cache_zeros(cache)
        target_statistic = None
        preserved_token_indices = None
        preserved_log_probabilities = None
        if config.preserve_top_log_probs < 0:
            raise ValueError("preserve_top_log_probs must be non-negative")
        if config.log_probability_preservation_weight < 0:
            raise ValueError("log_probability_preservation_weight must be non-negative")
        reference_class = next(
            index for index in range(self.classifier.mlp.out_features) if index != target_class
        )
        statistic_indices = [
            index
            for index in range(self.classifier.mlp.out_features)
            if index != reference_class
        ]
        for _ in range(config.minimum_norm_steps):
            variables = _detach_cache(delta, requires_grad=True)
            margin, logits, classifier_logits, _ = self._statistic_forward(
                cache,
                variables,
                last_token,
                accumulated,
                total_length,
                target_class,
                config.horizon_length,
                horizon_cache=horizon_cache,
            )
            if config.statistic_mode == "margin":
                statistic = margin
                if target_statistic is None:
                    target_statistic = resolve_margin_target(
                        margin,
                        target_probability=config.target_probability,
                        target_margin_shift=config.target_margin_shift,
                        target_margin=config.target_margin,
                        minimum_target_probability=config.minimum_target_probability,
                    )
                error = (target_statistic - statistic).clamp_min(0)
            else:
                logits = classifier_logits[0]
                statistic = logits[statistic_indices] - logits[reference_class]
                if target_statistic is None:
                    probabilities = logits.detach().softmax(dim=-1).clamp_min(1e-6)
                    current_probability = probabilities[target_class]
                    if config.target_margin_shift is not None:
                        current_odds = current_probability / (1 - current_probability)
                        target_probability = torch.sigmoid(
                            current_odds.log() + config.target_margin_shift
                        )
                    else:
                        target_probability = torch.tensor(
                            min(max(config.target_probability, 1e-5), 1 - 1e-5),
                            device=self.device,
                        )
                    target_probability = torch.maximum(
                        target_probability, current_probability
                    )
                    if config.minimum_target_probability is not None:
                        target_probability = torch.maximum(
                            target_probability,
                            torch.tensor(
                                min(
                                    max(config.minimum_target_probability, 1e-5),
                                    1 - 1e-5,
                                ),
                                device=self.device,
                            ),
                        )
                    target_probabilities = probabilities * (
                        (1 - target_probability) / (1 - current_probability)
                    )
                    target_probabilities[target_class] = target_probability
                    target_log_probabilities = target_probabilities.clamp_min(1e-6).log()
                    target_statistic = (
                        target_log_probabilities[statistic_indices]
                        - target_log_probabilities[reference_class]
                    )
                error = target_statistic - statistic

            if config.preserve_top_log_probs > 0:
                log_probabilities = logits[0].log_softmax(dim=-1)
                if preserved_token_indices is None:
                    preserved_token_indices = log_probabilities.topk(
                        min(config.preserve_top_log_probs, log_probabilities.numel())
                    ).indices.detach()
                    preserved_log_probabilities = log_probabilities[
                        preserved_token_indices
                    ].detach()
                preservation_scale = config.log_probability_preservation_weight**0.5
                preserved_statistic = (
                    preservation_scale * log_probabilities[preserved_token_indices]
                )
                preserved_target = (
                    preservation_scale * preserved_log_probabilities
                )
                statistic = torch.cat(
                    [statistic.reshape(-1), preserved_statistic.reshape(-1)]
                )
                error = torch.cat(
                    [error.reshape(-1), (preserved_target - preserved_statistic).reshape(-1)]
                )

            if float(error.detach().abs().max()) < 1e-4:
                delta = _detach_cache(variables)
                break
            flat_variables = [item for layer in variables for item in layer]
            gradient_sets = []
            for statistic_index in range(statistic.numel()):
                gradients = torch.autograd.grad(
                    statistic.reshape(-1)[statistic_index],
                    flat_variables,
                    retain_graph=statistic_index + 1 < statistic.numel(),
                )
                gradient_sets.append(
                    _mask_cache_gradients(
                        gradients,
                        cache,
                        config.cache_component,
                        config.cache_last_n_layers,
                    )
                )
            if config.gradient_block_normalization > 0:
                block_scales = []
                for gradient_index in range(len(flat_variables)):
                    aggregate_norm = torch.stack(
                        [
                            gradients[gradient_index].float().square().sum()
                            for gradients in gradient_sets
                        ]
                    ).sum().sqrt()
                    block_scales.append(
                        aggregate_norm.clamp_min(1e-12).pow(
                            -config.gradient_block_normalization
                        )
                    )
                weighted_gradient_sets = [
                    tuple(
                        scale * gradient
                        for scale, gradient in zip(block_scales, gradients)
                    )
                    for gradients in gradient_sets
                ]
            else:
                weighted_gradient_sets = gradient_sets
            gram = torch.stack(
                [
                    torch.stack(
                        [
                            _cache_gradient_dot(left, right)
                            for right in weighted_gradient_sets
                        ]
                    )
                    for left in gradient_sets
                ]
            )
            gram = gram + config.ridge * torch.eye(
                gram.shape[0], device=gram.device, dtype=gram.dtype
            )
            # Solve for the accumulated final perturbation, not another
            # independently regularized increment. Under the current
            # linearization, J(delta_new - delta) = error, hence the final
            # target displacement is error + J delta.
            flat_delta = [item for layer in variables for item in layer]
            projected_delta = torch.stack(
                [
                    _cache_gradient_dot(gradients, flat_delta)
                    for gradients in gradient_sets
                ]
            )
            right_hand_side = (
                error.detach().reshape(-1).float() + projected_delta
            )
            coefficients = torch.linalg.solve(gram, right_hand_side)
            updated = []
            gradient_index = 0
            for layer in variables:
                updated_layer = []
                for item in layer:
                    update = sum(
                        coefficient * gradients[gradient_index]
                        for coefficient, gradients in zip(
                            coefficients, weighted_gradient_sets
                        )
                    )
                    updated_layer.append(
                        item.detach()
                        + config.minimum_norm_damping
                        * (update.detach() - item.detach())
                    )
                    gradient_index += 1
                updated.append(tuple(updated_layer))
            delta = _cap_cache_relative_norm(tuple(updated), cache, config.maximum_relative_norm)
        return _detach_cache(delta)

    def score_text(self, input_ids: torch.Tensor) -> dict[str, object]:
        with torch.no_grad():
            output = self.model(
                input_ids, output_hidden_states=True, use_cache=False, return_dict=True
            )
            pooled = output.hidden_states[-1].mean(dim=1)
            probabilities = self.classifier(pooled).softmax(dim=-1)[0]
        return {
            "classifier_probabilities": [float(value) for value in probabilities],
            "positive_probability": float(probabilities[PPLM_SENTIMENT_LABELS["positive"]]),
            "negative_probability": float(probabilities[PPLM_SENTIMENT_LABELS["negative"]]),
        }

    def classifier_logits_for_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Score complete texts with the same mean-pooled SST head used in control."""
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                mask = encoded.attention_mask.unsqueeze(-1).to(
                    output.hidden_states[-1].dtype
                )
                pooled = (output.hidden_states[-1] * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1)
                batches.append(self.classifier(pooled).cpu())
        if not batches:
            return torch.empty((0, self.classifier.mlp.out_features))
        return torch.cat(batches, dim=0)

    def generate(
        self,
        prefix: str,
        target_label: str,
        config: PPLMGenerationConfig,
        seed: int,
    ) -> dict[str, object]:
        if config.method not in {"baseline", "pplm", "minimum_norm"}:
            raise ValueError(f"Unknown PPLM method: {config.method}")
        target_class = PPLM_SENTIMENT_LABELS[target_label]
        input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        step_diagnostics = []
        persistent_cache = None

        for _ in range(config.max_new_tokens):
            fresh_cache, accumulated, last_token = self._prepare_step(input_ids)
            base_margin, base_logits, _, base_next_cache = self._unperturbed_logits(
                fresh_cache,
                accumulated,
                last_token,
                input_ids.shape[1],
                target_class,
                config.horizon_length,
            )
            cache = (
                persistent_cache
                if config.persistent_cache and persistent_cache is not None
                else fresh_cache
            )
            horizon_cache = base_next_cache if config.persistent_cache else None
            if config.method == "baseline":
                delta = _cache_zeros(cache)
                logits = base_logits
                post_margin = base_margin
                next_cache = base_next_cache
            else:
                if config.method == "pplm":
                    delta = self._pplm_delta(
                        cache,
                        accumulated,
                        last_token,
                        input_ids.shape[1],
                        target_class,
                        config,
                        base_logits,
                        horizon_cache=horizon_cache,
                    )
                else:
                    delta = self._minimum_norm_delta(
                        cache,
                        accumulated,
                        last_token,
                        input_ids.shape[1],
                        target_class,
                        config,
                        horizon_cache=horizon_cache,
                    )
                with torch.no_grad():
                    post_margin, logits, _, next_cache = self._statistic_forward(
                        cache,
                        delta,
                        last_token,
                        accumulated,
                        input_ids.shape[1],
                        target_class,
                        config.horizon_length,
                        horizon_cache=horizon_cache,
                    )

            if config.persistent_cache:
                persistent_cache = _detach_cache(next_cache)

            if config.method == "baseline":
                probabilities = (logits / config.temperature).softmax(dim=-1)
                mix_scale = 0.0
                token_kl = 0.0
                raw_token_kl = 0.0
                token_kl_budget = None
            else:
                raw_probabilities = (logits / config.temperature).softmax(dim=-1)
                base_probabilities = (base_logits / config.temperature).softmax(dim=-1)
                raw_token_kl = float(
                    _categorical_kl(raw_probabilities, base_probabilities).mean()
                )
                token_kl_budget = _select_token_kl_budget(
                    base_margin,
                    config.maximum_token_kl,
                    config.difficult_margin_threshold,
                    config.difficult_maximum_token_kl,
                )
                probabilities, mix_scale, token_kl = _geometric_probability_mix(
                    logits,
                    base_logits,
                    config.temperature,
                    config.gm_scale,
                    token_kl_budget,
                )
            probabilities = _top_k_probs(probabilities, config.top_k)
            if config.sample:
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            else:
                next_token = probabilities.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            step_diagnostics.append(
                {
                    "pre_margin": float(base_margin),
                    "post_margin": float(post_margin),
                    "relative_cache_change": _cache_total_relative_norm(delta, cache),
                    "mix_scale": mix_scale,
                    "token_kl": token_kl,
                    "raw_token_kl": raw_token_kl,
                    "token_kl_budget": token_kl_budget,
                }
            )
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        continuation_ids = input_ids[:, prompt_length:]
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        continuation = self.tokenizer.decode(continuation_ids[0], skip_special_tokens=True)
        final_scores = self.score_text(input_ids)
        result = {
            "prefix": prefix,
            "target_label": target_label,
            "seed": seed,
            "method": config.method,
            "configuration": asdict(config),
            "text": text,
            "continuation": continuation,
            "tokens_generated": int(continuation_ids.shape[1]),
            **final_scores,
            "mean_pre_margin": sum(x["pre_margin"] for x in step_diagnostics)
            / len(step_diagnostics),
            "mean_post_margin": sum(x["post_margin"] for x in step_diagnostics)
            / len(step_diagnostics),
            "mean_relative_cache_change": sum(
                x["relative_cache_change"] for x in step_diagnostics
            )
            / len(step_diagnostics),
            "max_relative_cache_change": max(
                x["relative_cache_change"] for x in step_diagnostics
            ),
            "mean_mix_scale": sum(x["mix_scale"] for x in step_diagnostics)
            / len(step_diagnostics),
            "mean_token_kl": sum(x["token_kl"] for x in step_diagnostics)
            / len(step_diagnostics),
            "mean_raw_token_kl": sum(
                x["raw_token_kl"] for x in step_diagnostics
            )
            / len(step_diagnostics),
            "mean_token_kl_budget": sum(
                x["token_kl_budget"]
                for x in step_diagnostics
                if x["token_kl_budget"] is not None
            )
            / max(
                sum(x["token_kl_budget"] is not None for x in step_diagnostics),
                1,
            ),
        }
        return result


def distinct_ngram_fraction(texts: Iterable[str], n: int) -> float:
    ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = text.split()
        ngrams.extend(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return len(set(ngrams)) / max(len(ngrams), 1)
