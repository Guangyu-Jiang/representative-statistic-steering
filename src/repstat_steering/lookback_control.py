"""Lookback Lens scoring and minimum-norm attention-logit control."""

from __future__ import annotations

import copy
import gzip
import json
import math
import pickle
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama


@dataclass(frozen=True)
class LookbackGenerationConfig:
    method: str
    solver_version: str = "accumulated_v2"
    max_new_tokens: int = 64
    window_size: int = 8
    target_mode: str = "relative"
    target_probability: float = 0.95
    target_logit_shift: float = 1.0
    control_trigger_probability: float = 1.0
    high_confidence_logit_shift: float = 0.0
    ridge: float = 0.1
    solver_steps: int = 8
    solver_damping: float = 1.0
    maximum_bias_rms: float | None = 0.5
    context_bias_mode: str = "uniform"
    context_top_fraction: float = 0.25
    context_overlap_radius: int = 8
    active_control_count: int = 0
    bias_constraint: str = "unrestricted"
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    do_sample: bool = False
    chunk_size: int = 8
    num_candidates: int = 8
    seed: int = 42


@dataclass(frozen=True)
class LookbackBiasDiagnostics:
    initial_probability: float
    predicted_probability: float
    target_probability: float
    target_logit_error: float
    target_logit_shortfall: float
    control_objective: float
    bias_norm: float
    bias_rms: float
    bias_max_abs: float
    active_bias_count: int
    negative_bias_fraction: float
    iterations: int


@dataclass(frozen=True)
class NQExample:
    dataset_index: int
    question: str
    answers: tuple[str, ...]
    prompt: str


LOOKBACK_QUERY_STOP_WORDS = {
    "a", "an", "the", "and", "are", "as", "at", "be", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "into", "is", "it", "of", "on", "or", "that", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}


def _lexical_terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 2 and term not in LOOKBACK_QUERY_STOP_WORDS
    ]


def bm25_relevance_scores(question: str, units: Sequence[str]) -> list[float]:
    """Score document units against a question without answer supervision."""

    if not units:
        return []
    query_terms = set(_lexical_terms(question))
    tokenized = [_lexical_terms(unit) for unit in units]
    lengths = [len(terms) for terms in tokenized]
    average_length = max(sum(lengths) / len(lengths), 1.0)
    document_frequency = {
        term: sum(term in set(terms) for terms in tokenized)
        for term in query_terms
    }
    k1 = 1.2
    b = 0.75
    scores: list[float] = []
    for terms, length in zip(tokenized, lengths, strict=True):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            # This positive BM25 IDF is stable for the small three-passage corpus.
            inverse_frequency = math.log(
                1.0
                + (len(units) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * length / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _open_text(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8"
    )


def load_nq_examples(
    path: str | Path,
    *,
    offset: int = 0,
    limit: int = 0,
    indices: Sequence[int] | None = None,
) -> list[NQExample]:
    """Load the paper's Natural Questions context construction."""

    selected = set(indices) if indices is not None else None
    if selected is not None and not selected:
        return []
    remaining = set(selected) if selected is not None else None
    examples: list[NQExample] = []
    with _open_text(path) as handle:
        for dataset_index, line in enumerate(handle):
            if remaining is not None:
                if dataset_index not in remaining:
                    continue
            elif dataset_index < offset:
                continue

            item = json.loads(line)
            question = item["question"]
            question = question[0].upper() + question[1:]
            if not question.endswith("?"):
                question += "?"
            positive = [context for context in item["ctxs"] if context["hasanswer"]]
            negative = [context for context in item["ctxs"] if not context["hasanswer"]]
            if not positive or len(negative) < 2:
                continue
            document = "\n".join(
                [negative[0]["text"], positive[0]["text"], negative[1]["text"]]
            )
            prompt = (
                "Answer the question based on the information in the document. "
                "Explain your reasoning in the document step-by-step before providing "
                "the final answer.\n\n"
                f"#Document#: {document}\n#Question#: {question}\n#Answer#:"
            )
            examples.append(
                NQExample(
                    dataset_index=dataset_index,
                    question=question,
                    answers=tuple(item["answers"]),
                    prompt=prompt,
                )
            )
            if remaining is not None:
                remaining.remove(dataset_index)
                if not remaining:
                    break
            if selected is None and limit > 0 and len(examples) >= limit:
                break
    return examples


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def best_subspan_exact_match(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalize_answer(answer) in normalized_prediction for answer in answers)
    )


class LookbackClassifier:
    """Torch representation of the published scikit-learn classifier."""

    def __init__(self, weight: torch.Tensor, intercept: torch.Tensor, threshold: float):
        self.weight = weight.detach().float().reshape(-1)
        self.intercept = intercept.detach().float().reshape(())
        self.threshold = float(threshold)

    @classmethod
    def from_pickle(cls, path: str | Path) -> "LookbackClassifier":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        classifier = payload["clf"]
        return cls(
            torch.from_numpy(np.asarray(classifier.coef_[0])),
            torch.tensor(float(classifier.intercept_[0])),
            float(payload["best_threshold"]),
        )

    @property
    def feature_count(self) -> int:
        return int(self.weight.numel())

    def to(self, device: torch.device | str) -> "LookbackClassifier":
        return LookbackClassifier(
            self.weight.to(device), self.intercept.to(device), self.threshold
        )

    def logit(self, features: torch.Tensor) -> torch.Tensor:
        return features.float() @ self.weight.to(features.device) + self.intercept.to(
            features.device
        )

    def probability(self, features: torch.Tensor) -> torch.Tensor:
        return self.logit(features).sigmoid()


def biased_lookback_ratios(
    base_ratios: torch.Tensor,
    bias: torch.Tensor,
    context_focus_fraction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a uniform or focused context-logit bias in closed form."""

    logits = torch.logit(base_ratios.float().clamp(1e-6, 1 - 1e-6))
    if context_focus_fraction is None:
        log_context_scale = bias.float()
    else:
        fraction = context_focus_fraction.float().clamp(0, 1)
        # Only the selected fraction q of context attention receives the bias:
        # C' / C = (1 - q) + q exp(b).
        log_context_scale = torch.logaddexp(
            torch.log1p(-fraction), torch.log(fraction) + bias.float()
        )
    return torch.sigmoid(logits + log_context_scale)


def rolling_feature(
    current: torch.Tensor, history: Sequence[torch.Tensor], window_size: int
) -> torch.Tensor:
    retained = list(history[-max(window_size - 1, 0) :]) + [current]
    return torch.stack(retained).mean(dim=0)


def matched_random_mask(
    reference_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Select a deterministic random support with the reference cardinality."""

    reference_mask = reference_mask.detach().bool().reshape(-1)
    eligible_mask = eligible_mask.detach().bool().reshape(-1)
    if reference_mask.numel() != eligible_mask.numel():
        raise ValueError("reference_mask and eligible_mask must have equal length")
    eligible = eligible_mask.nonzero(as_tuple=False).reshape(-1)
    count = min(int(reference_mask.sum()), int(eligible.numel()))
    if count == 0:
        return reference_mask.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = eligible[torch.randperm(eligible.numel(), generator=generator)[:count]]
    result = torch.zeros_like(reference_mask)
    result[selected] = True
    return result


def solve_lookback_bias(
    base_ratios: torch.Tensor,
    history: Sequence[torch.Tensor],
    classifier: LookbackClassifier,
    *,
    window_size: int,
    ridge: float,
    steps: int,
    damping: float,
    maximum_bias_rms: float | None,
    target_probability: float | None = None,
    target_logit: float | torch.Tensor | None = None,
    context_focus_fraction: torch.Tensor | None = None,
    active_control_count: int = 0,
    bias_constraint: str = "unrestricted",
    tolerance: float = 1e-4,
) -> tuple[torch.Tensor, LookbackBiasDiagnostics]:
    """Iteratively solve the local minimum-norm classifier-logit inverse."""

    if (target_probability is None) == (target_logit is None):
        raise ValueError("provide exactly one of target_probability or target_logit")
    if target_probability is not None and not 0 < target_probability < 1:
        raise ValueError("target_probability must be in (0, 1)")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if active_control_count < 0:
        raise ValueError("active_control_count must be non-negative")
    if bias_constraint not in {"unrestricted", "nonnegative"}:
        raise ValueError(f"unknown bias_constraint: {bias_constraint}")
    if base_ratios.numel() != classifier.feature_count:
        raise ValueError(
            f"expected {classifier.feature_count} ratios, got {base_ratios.numel()}"
        )

    base_ratios = base_ratios.detach().float().reshape(-1)
    history = [item.detach().float().reshape(-1) for item in history]
    if context_focus_fraction is not None:
        context_focus_fraction = context_focus_fraction.detach().float().reshape(-1)
        if context_focus_fraction.numel() != base_ratios.numel():
            raise ValueError("context_focus_fraction must match base_ratios")
    if target_logit is None:
        target_logit_tensor = torch.logit(
            torch.tensor(
                target_probability, device=base_ratios.device, dtype=torch.float32
            )
        )
    else:
        target_logit_tensor = torch.as_tensor(
            target_logit, device=base_ratios.device, dtype=torch.float32
        ).reshape(())
    diagnostic_target_probability = float(target_logit_tensor.sigmoid())
    initial_feature = rolling_feature(base_ratios, history, window_size)
    initial_logit = classifier.logit(initial_feature)
    bias = torch.zeros_like(base_ratios)
    active_mask: torch.Tensor | None = None
    completed_steps = 0

    # One-sided control does not perturb an already sufficiently factual state.
    if float(initial_logit) < float(target_logit_tensor):
        for step in range(steps):
            point = bias.detach().requires_grad_(True)
            feature = rolling_feature(
                biased_lookback_ratios(
                    base_ratios, point, context_focus_fraction
                ),
                history,
                window_size,
            )
            statistic = classifier.logit(feature)
            error = target_logit_tensor - statistic.detach()
            gradient = torch.autograd.grad(statistic, point)[0].detach()
            if active_mask is None and 0 < active_control_count < gradient.numel():
                if bias_constraint == "nonnegative":
                    eligible = (gradient > 0).nonzero(as_tuple=False).reshape(-1)
                    count = min(active_control_count, int(eligible.numel()))
                    indices = (
                        eligible[gradient[eligible].topk(count).indices]
                        if count > 0
                        else eligible
                    )
                else:
                    indices = gradient.abs().topk(active_control_count).indices
                active_mask = torch.zeros_like(gradient, dtype=torch.bool)
                active_mask[indices] = True
            if active_mask is not None:
                gradient = gradient * active_mask
            gradient_norm_squared = gradient.square().sum()
            current_objective = error.square() + ridge * bias.square().sum()

            # Minimize the linearized *accumulated* objective
            #   (g^T u - error)^2 + ridge ||bias + u||^2.
            # The rank-one normal equation has this closed-form solution.
            raw_update = -bias + (
                (error + torch.dot(gradient, bias))
                * gradient
                / (gradient_norm_squared + ridge).clamp_min(1e-12)
            )
            step_scale = damping
            accepted_bias = bias
            # Near saturated ratios the Jacobian can be tiny, so the raw
            # Gauss-Newton step can be large. A deep backtrack still costs only
            # operations on the 1024-dimensional statistic, not model forwards.
            for _ in range(32):
                candidate = bias + step_scale * raw_update
                if active_mask is not None:
                    candidate = candidate * active_mask
                if bias_constraint == "nonnegative":
                    candidate = candidate.clamp_min(0)
                if maximum_bias_rms is not None:
                    candidate_rms = candidate.square().mean().sqrt()
                    if float(candidate_rms) > maximum_bias_rms:
                        candidate = candidate * (maximum_bias_rms / candidate_rms)
                candidate_feature = rolling_feature(
                    biased_lookback_ratios(
                        base_ratios, candidate, context_focus_fraction
                    ),
                    history,
                    window_size,
                )
                candidate_error = target_logit_tensor - classifier.logit(
                    candidate_feature
                )
                candidate_objective = (
                    candidate_error.square() + ridge * candidate.square().sum()
                )
                if float(candidate_objective) <= float(current_objective) + 1e-10:
                    accepted_bias = candidate.detach()
                    break
                step_scale *= 0.5
            update = accepted_bias - bias
            bias = accepted_bias
            completed_steps = step + 1
            if float(update.square().mean().sqrt()) <= tolerance:
                break

    final_feature = rolling_feature(
        biased_lookback_ratios(base_ratios, bias, context_focus_fraction),
        history,
        window_size,
    )
    predicted_logit = classifier.logit(final_feature).detach()
    signed_error = target_logit_tensor - predicted_logit
    control_objective = signed_error.square() + ridge * bias.square().sum()
    diagnostics = LookbackBiasDiagnostics(
        initial_probability=float(initial_logit.sigmoid()),
        predicted_probability=float(predicted_logit.sigmoid()),
        target_probability=diagnostic_target_probability,
        target_logit_error=abs(float(signed_error)),
        target_logit_shortfall=max(float(signed_error), 0.0),
        control_objective=float(control_objective),
        bias_norm=float(bias.norm()),
        bias_rms=float(bias.square().mean().sqrt()),
        bias_max_abs=float(bias.abs().max()),
        active_bias_count=int((bias.abs() > 1e-8).sum()),
        negative_bias_fraction=(
            float((bias < -1e-8).sum() / (bias.abs() > 1e-8).sum())
            if bool((bias.abs() > 1e-8).any())
            else 0.0
        ),
        iterations=completed_steps,
    )
    return bias.detach(), diagnostics


class LookbackAttentionController:
    """Capture and perturb Llama attention using the modern attention interface."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.original_implementation = model.config._attn_implementation
        self.original_eager_attention = modeling_llama.eager_attention_forward
        self.context_length: int | None = None
        self.bias: torch.Tensor | None = None
        self.enabled = False
        self._ratios: dict[int, torch.Tensor] = {}
        self._attention_kl: dict[int, torch.Tensor] = {}
        self._captured_focus_masks: dict[int, torch.Tensor] = {}
        self._captured_focus_fractions: dict[int, torch.Tensor] = {}
        self.focus_masks_to_apply: dict[int, torch.Tensor] | None = None
        self.capture_top_fraction: float | None = None
        self.capture_focus_mask: torch.Tensor | None = None
        # Keep the implementation name as eager so Transformers constructs the
        # standard 4D causal mask. A novel implementation name can trigger the
        # mask fast path and silently make a decoder bidirectional.
        modeling_llama.eager_attention_forward = self._forward
        model.config._attn_implementation = "eager"

    def begin(
        self,
        context_length: int,
        bias: torch.Tensor | None = None,
        *,
        focus_masks: dict[int, torch.Tensor] | None = None,
        capture_top_fraction: float | None = None,
        capture_focus_mask: torch.Tensor | None = None,
    ) -> None:
        self.context_length = int(context_length)
        self.bias = bias
        self.focus_masks_to_apply = focus_masks
        self.capture_top_fraction = capture_top_fraction
        self.capture_focus_mask = capture_focus_mask
        self.enabled = True
        self._ratios = {}
        self._attention_kl = {}
        self._captured_focus_masks = {}
        self._captured_focus_fractions = {}

    def end(self) -> None:
        self.enabled = False
        self.bias = None
        self.focus_masks_to_apply = None
        self.capture_top_fraction = None
        self.capture_focus_mask = None

    def close(self) -> None:
        self.end()
        modeling_llama.eager_attention_forward = self.original_eager_attention
        self.model.config._attn_implementation = self.original_implementation

    def features(self) -> torch.Tensor:
        expected = self.model.config.num_hidden_layers
        if len(self._ratios) != expected:
            raise RuntimeError(f"captured {len(self._ratios)} of {expected} attention layers")
        return torch.cat([self._ratios[index] for index in range(expected)])

    def mean_attention_kl(self) -> float:
        if not self._attention_kl:
            return 0.0
        return float(torch.stack(list(self._attention_kl.values())).mean())

    def captured_focus_masks(self) -> dict[int, torch.Tensor]:
        return {index: mask.clone() for index, mask in self._captured_focus_masks.items()}

    def captured_focus_fractions(self) -> torch.Tensor:
        expected = self.model.config.num_hidden_layers
        if len(self._captured_focus_fractions) != expected:
            raise RuntimeError(
                f"captured focus fractions for {len(self._captured_focus_fractions)} "
                f"of {expected} attention layers"
            )
        return torch.cat(
            [self._captured_focus_fractions[index] for index in range(expected)]
        )

    def _forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = modeling_llama.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(value, module.num_key_value_groups)
        attention_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attention_logits = attention_logits + attention_mask[:, :, :, : key_states.shape[-2]]

        base_attention = F.softmax(attention_logits, dim=-1, dtype=torch.float32)
        controlled_logits = attention_logits
        key_length = key_states.shape[-2]
        can_control = (
            self.enabled
            and self.context_length is not None
            and 0 < self.context_length < key_length
        )
        if can_control and self.bias is not None:
            layer_bias = self.bias[module.layer_idx].to(
                device=attention_logits.device, dtype=attention_logits.dtype
            )
            controlled_logits = attention_logits.clone()
            if self.focus_masks_to_apply is None:
                controlled_logits[:, :, -1:, : self.context_length] += (
                    layer_bias.reshape(1, -1, 1, 1)
                )
            else:
                focus_mask = self.focus_masks_to_apply[module.layer_idx].to(
                    device=attention_logits.device, dtype=attention_logits.dtype
                )
                controlled_logits[:, :, -1:, : self.context_length] += (
                    layer_bias.reshape(1, -1, 1, 1)
                    * focus_mask.reshape(1, focus_mask.shape[0], 1, -1)
                )

        attention = F.softmax(controlled_logits, dim=-1, dtype=torch.float32)
        if can_control:
            current = attention[0, :, -1, :]
            on_context = current[:, : self.context_length].mean(dim=-1)
            on_generated = current[:, self.context_length :].mean(dim=-1)
            ratio = on_context / (on_context + on_generated).clamp_min(1e-12)
            self._ratios[module.layer_idx] = ratio.detach()
            if (
                self.capture_top_fraction is not None
                or self.capture_focus_mask is not None
            ):
                context_attention = base_attention[
                    0, :, -1, : self.context_length
                ]
                if self.capture_focus_mask is None:
                    count = max(
                        1,
                        min(
                            self.context_length,
                            round(
                                self.context_length * self.capture_top_fraction
                            ),
                        ),
                    )
                    indices = context_attention.topk(count, dim=-1).indices
                    focus_mask = torch.zeros_like(
                        context_attention, dtype=torch.bool
                    )
                    focus_mask.scatter_(1, indices, True)
                else:
                    fixed_mask = self.capture_focus_mask.to(
                        device=context_attention.device, dtype=torch.bool
                    )
                    if fixed_mask.numel() != self.context_length:
                        raise ValueError(
                            "capture_focus_mask must match context_length"
                        )
                    focus_mask = fixed_mask.reshape(1, -1).expand(
                        context_attention.shape[0], -1
                    ).clone()
                    if self.capture_top_fraction is not None:
                        count = max(
                            1,
                            min(
                                self.context_length,
                                round(
                                    self.context_length
                                    * self.capture_top_fraction
                                ),
                            ),
                        )
                        indices = context_attention.topk(count, dim=-1).indices
                        focus_mask.scatter_(1, indices, True)
                selected_mass = (context_attention * focus_mask).sum(dim=-1)
                context_mass = context_attention.sum(dim=-1).clamp_min(1e-12)
                self._captured_focus_masks[module.layer_idx] = focus_mask.detach()
                self._captured_focus_fractions[module.layer_idx] = (
                    selected_mass / context_mass
                ).detach()
            if self.bias is not None:
                base = base_attention[0, :, -1, :].clamp_min(1e-12)
                steered = current.clamp_min(1e-12)
                self._attention_kl[module.layer_idx] = (
                    steered * (steered.log() - base.log())
                ).sum(dim=-1).mean().detach()

        attention = attention.to(query.dtype)
        attention = F.dropout(attention, p=dropout, training=module.training)
        output = torch.matmul(attention, value_states)
        output = output.transpose(1, 2).contiguous()
        return output, attention


def _cache_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def _crop_cache(cache: Any, length: int) -> Any:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    return tuple((key[..., :length, :], value[..., :length, :]) for key, value in cache)


def _clone_cache(cache: Any) -> Any:
    return copy.deepcopy(cache)


def _distribution_kl(logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    log_probabilities = logits.float().log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    reference_log_probabilities = reference_logits.float().log_softmax(dim=-1)
    return float(
        (probabilities * (log_probabilities - reference_log_probabilities)).sum()
    )


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        cutoff = filtered.topk(min(top_k, filtered.numel())).values[-1]
        filtered[filtered < cutoff] = -torch.inf
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = -torch.inf
    return filtered


class LookbackNQExperiment:
    stop_markers = (
        "### User:",
        "\nQ:",
        "#Document#:",
        "#Pondering#:",
        "#Question#:",
        "#Dialogue History#:",
    )

    def __init__(
        self,
        model_name: str,
        classifier_path: str | Path,
        *,
        device: str = "cuda",
        model_dtype: str = "float16",
    ):
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[model_dtype]
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.classifier = LookbackClassifier.from_pickle(classifier_path).to(self.device)
        expected = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        if self.classifier.feature_count != expected:
            raise ValueError(
                f"classifier has {self.classifier.feature_count} features but model has {expected} layer-heads"
            )
        self.controller = LookbackAttentionController(self.model)

    def close(self) -> None:
        self.controller.close()

    def _prepare(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, Any, int]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if input_ids.shape[-1] < 2:
            raise ValueError("prompt must contain at least two tokens")
        response_tokens = self.tokenizer("\n#Answer#:").input_ids
        extra_prompt_length = len(response_tokens) - 1
        context_length = input_ids.shape[-1] - extra_prompt_length
        with torch.inference_mode():
            output = self.model(
                input_ids[:, :-1], use_cache=True, return_dict=True
            )
        return input_ids[:, -1:], input_ids, output.past_key_values, context_length

    def _forward_token(
        self,
        token: torch.Tensor,
        cache: Any,
        context_length: int,
        bias: torch.Tensor | None,
        *,
        focus_masks: dict[int, torch.Tensor] | None = None,
        capture_top_fraction: float | None = None,
        capture_focus_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, Any]:
        shaped_bias = None
        if bias is not None:
            shaped_bias = bias.reshape(
                self.model.config.num_hidden_layers,
                self.model.config.num_attention_heads,
            )
        self.controller.begin(
            context_length,
            shaped_bias,
            focus_masks=focus_masks,
            capture_top_fraction=capture_top_fraction,
            capture_focus_mask=capture_focus_mask,
        )
        try:
            with torch.inference_mode():
                output = self.model(
                    token,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            features = self.controller.features()
            attention_kl = self.controller.mean_attention_kl()
        finally:
            self.controller.end()
        return output.logits[0, -1], features, attention_kl, output.past_key_values

    def _sample(
        self,
        logits: torch.Tensor,
        config: LookbackGenerationConfig,
        generator: torch.Generator,
        *,
        force_sampling: bool | None = None,
    ) -> torch.Tensor:
        do_sample = config.do_sample if force_sampling is None else force_sampling
        if not do_sample:
            return logits.argmax().reshape(1, 1)
        scaled = logits.float() / max(config.temperature, 1e-6)
        scaled = _filter_logits(scaled, config.top_k, config.top_p)
        token = torch.multinomial(scaled.softmax(dim=-1), 1, generator=generator)
        return token.reshape(1, 1)

    def _should_stop(self, generated: list[int]) -> bool:
        if generated and generated[-1] == self.tokenizer.eos_token_id:
            return True
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return any(marker in text for marker in self.stop_markers)

    def _trim_response(self, generated: list[int]) -> str:
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        positions = [response.find(marker) for marker in self.stop_markers]
        positions = [position for position in positions if position >= 0]
        if positions:
            response = response[: min(positions)]
        return response.strip()

    def _base_result(
        self,
        example: NQExample,
        config: LookbackGenerationConfig,
        response: str,
        generated: list[int],
        probabilities: list[float],
        diagnostics: dict[str, list[float]],
    ) -> dict[str, Any]:
        def mean(name: str) -> float:
            values = diagnostics.get(name, [])
            return float(np.mean(values)) if values else 0.0

        return {
            "dataset_index": example.dataset_index,
            "question": example.question,
            "answers": list(example.answers),
            "method": config.method,
            "solver_version": config.solver_version,
            "response": response,
            "exact_match": best_subspan_exact_match(response, example.answers),
            "generated_tokens": len(generated),
            "generated_token_ids": list(generated),
            "mean_factual_probability": float(np.mean(probabilities)) if probabilities else 0.0,
            "final_factual_probability": probabilities[-1] if probabilities else 0.0,
            "mean_initial_factual_probability": mean("initial_probability"),
            "mean_predicted_factual_probability": mean("predicted_probability"),
            "mean_actual_target_probability": mean("actual_target_probability"),
            "mean_bias_norm": mean("bias_norm"),
            "mean_bias_rms": mean("bias_rms"),
            "maximum_bias_abs": max(diagnostics.get("bias_max_abs", [0.0])),
            "mean_active_bias_count": mean("active_bias_count"),
            "mean_negative_bias_fraction": mean("negative_bias_fraction"),
            "mean_attention_kl": mean("attention_kl"),
            "mean_output_kl": mean("output_kl"),
            "mean_target_logit_error": mean("target_logit_error"),
            "mean_target_logit_shortfall": mean("target_logit_shortfall"),
            "mean_actual_target_logit_error": mean(
                "actual_target_logit_error"
            ),
            "mean_actual_target_logit_shortfall": mean(
                "actual_target_logit_shortfall"
            ),
            "mean_actual_logit_gain": mean("actual_logit_gain"),
            "mean_focus_attention_fraction": mean(
                "focus_attention_fraction"
            ),
            "mean_focus_token_fraction": mean("focus_token_fraction"),
            "mean_control_objective": mean("control_objective"),
            "classifier_threshold": self.classifier.threshold,
            "target_probability": config.target_probability,
            "target_mode": config.target_mode,
            "target_logit_shift": config.target_logit_shift,
            "control_trigger_probability": config.control_trigger_probability,
            "high_confidence_logit_shift": config.high_confidence_logit_shift,
            "ridge": config.ridge,
            "maximum_bias_rms": config.maximum_bias_rms,
            "context_bias_mode": config.context_bias_mode,
            "context_top_fraction": config.context_top_fraction,
            "context_overlap_radius": config.context_overlap_radius,
            "active_control_count": config.active_control_count,
            "bias_constraint": config.bias_constraint,
            "seed": config.seed,
        }

    def generate_baseline(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        token, _, cache, context_length = self._prepare(example.prompt)
        generator = torch.Generator(device=self.device).manual_seed(
            config.seed + example.dataset_index
        )
        generated: list[int] = []
        history: list[torch.Tensor] = []
        probabilities: list[float] = []
        for _ in range(config.max_new_tokens):
            logits, feature, _, cache = self._forward_token(
                token, cache, context_length, None
            )
            history.append(feature)
            pooled = torch.stack(history[-config.window_size :]).mean(dim=0)
            probabilities.append(float(self.classifier.probability(pooled)))
            token = self._sample(logits, config, generator)
            generated.append(int(token.item()))
            if self._should_stop(generated):
                break
        response = self._trim_response(generated)
        return self._base_result(
            example, config, response, generated, probabilities, {}
        )

    def generate_minimum_norm(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        token, _, cache, context_length = self._prepare(example.prompt)
        generator = torch.Generator(device=self.device).manual_seed(
            config.seed + example.dataset_index
        )
        generated: list[int] = []
        history: list[torch.Tensor] = []
        probabilities: list[float] = []
        diagnostics: dict[str, list[float]] = {
            name: []
            for name in (
                "initial_probability",
                "predicted_probability",
                "actual_target_probability",
                "bias_norm",
                "bias_rms",
                "bias_max_abs",
                "active_bias_count",
                "negative_bias_fraction",
                "attention_kl",
                "output_kl",
                "target_logit_error",
                "target_logit_shortfall",
                "actual_target_logit_error",
                "actual_target_logit_shortfall",
                "actual_logit_gain",
                "focus_attention_fraction",
                "focus_token_fraction",
                "control_objective",
            )
        }
        overlap_focus_mask = None
        if config.context_bias_mode in {
            "question_overlap",
            "question_top_union",
            "random_overlap",
            "retrieved_passage",
            "retrieved_sentence",
        }:
            if config.context_bias_mode in {
                "retrieved_passage",
                "retrieved_sentence",
            }:
                overlap_focus_mask = self._retrieval_focus_mask(
                    example.prompt,
                    example.question,
                    context_length,
                    unit_mode=(
                        "passage"
                        if config.context_bias_mode == "retrieved_passage"
                        else "sentence"
                    ),
                    padding_tokens=config.context_overlap_radius,
                )
            else:
                overlap_focus_mask = self._question_overlap_mask(
                    example.prompt,
                    example.question,
                    context_length,
                    config.context_overlap_radius,
                )
            if config.context_bias_mode == "random_overlap":
                overlap_focus_mask = matched_random_mask(
                    overlap_focus_mask,
                    self._document_token_mask(example.prompt, context_length),
                    seed=config.seed + example.dataset_index,
                )
        for _ in range(config.max_new_tokens):
            cache_length = _cache_length(cache)
            capture_top_fraction = (
                config.context_top_fraction
                if config.context_bias_mode in {
                    "top_attention",
                    "question_top_union",
                }
                else None
            )
            base_logits, base_feature, _, cache = self._forward_token(
                token,
                cache,
                context_length,
                None,
                capture_top_fraction=capture_top_fraction,
                capture_focus_mask=overlap_focus_mask,
            )
            if config.context_bias_mode in {
                "top_attention",
                "question_overlap",
                "question_top_union",
                "random_overlap",
                "retrieved_passage",
                "retrieved_sentence",
            }:
                focus_fractions = self.controller.captured_focus_fractions()
                focus_masks = self.controller.captured_focus_masks()
                focus_token_fraction = float(
                    torch.cat(
                        [
                            focus_masks[layer].float().reshape(-1)
                            for layer in sorted(focus_masks)
                        ]
                    ).mean()
                )
            elif config.context_bias_mode == "uniform":
                focus_fractions = None
                focus_masks = None
                focus_token_fraction = 1.0
            else:
                raise ValueError(
                    f"unknown context bias mode: {config.context_bias_mode}"
                )
            initial_feature = rolling_feature(
                base_feature, history, config.window_size
            )
            initial_logit = self.classifier.logit(initial_feature)
            if config.target_mode == "relative":
                if not 0 < config.control_trigger_probability <= 1:
                    raise ValueError(
                        "control_trigger_probability must be in (0, 1]"
                    )
                should_control = float(initial_logit.sigmoid()) < (
                    config.control_trigger_probability
                )
                solver_target_logit = initial_logit.detach() + (
                    config.target_logit_shift
                    if should_control
                    else config.high_confidence_logit_shift
                )
                target_probability = None
            elif config.target_mode == "absolute":
                solver_target_logit = None
                target_probability = config.target_probability
            else:
                raise ValueError(f"unknown target mode: {config.target_mode}")
            target_logit_for_diagnostics = (
                solver_target_logit
                if solver_target_logit is not None
                else torch.logit(
                    torch.tensor(
                        target_probability,
                        device=base_feature.device,
                        dtype=torch.float32,
                    )
                )
            )
            bias, solver = solve_lookback_bias(
                base_feature,
                history,
                self.classifier,
                target_probability=target_probability,
                target_logit=solver_target_logit,
                context_focus_fraction=focus_fractions,
                active_control_count=config.active_control_count,
                bias_constraint=config.bias_constraint,
                window_size=config.window_size,
                ridge=config.ridge,
                steps=config.solver_steps,
                damping=config.solver_damping,
                maximum_bias_rms=config.maximum_bias_rms,
            )
            if float(bias.norm()) > 0:
                cache = _crop_cache(cache, cache_length)
                logits, actual_feature, attention_kl, cache = self._forward_token(
                    token,
                    cache,
                    context_length,
                    bias,
                    focus_masks=focus_masks,
                )
                output_kl = _distribution_kl(logits, base_logits)
            else:
                logits = base_logits
                actual_feature = base_feature
                attention_kl = 0.0
                output_kl = 0.0

            pooled = rolling_feature(actual_feature, history, config.window_size)
            actual_logit = self.classifier.logit(pooled)
            actual_error = target_logit_for_diagnostics - actual_logit
            history.append(actual_feature)
            probabilities.append(float(actual_logit.sigmoid()))
            diagnostics["initial_probability"].append(solver.initial_probability)
            diagnostics["predicted_probability"].append(solver.predicted_probability)
            diagnostics["actual_target_probability"].append(solver.target_probability)
            diagnostics["bias_norm"].append(solver.bias_norm)
            diagnostics["bias_rms"].append(solver.bias_rms)
            diagnostics["bias_max_abs"].append(solver.bias_max_abs)
            diagnostics["active_bias_count"].append(solver.active_bias_count)
            diagnostics["negative_bias_fraction"].append(
                solver.negative_bias_fraction
            )
            diagnostics["attention_kl"].append(attention_kl)
            diagnostics["output_kl"].append(output_kl)
            diagnostics["target_logit_error"].append(solver.target_logit_error)
            diagnostics["target_logit_shortfall"].append(
                solver.target_logit_shortfall
            )
            diagnostics["actual_target_logit_error"].append(
                abs(float(actual_error))
            )
            diagnostics["actual_target_logit_shortfall"].append(
                max(float(actual_error), 0.0)
            )
            diagnostics["actual_logit_gain"].append(
                float(actual_logit - initial_logit)
            )
            diagnostics["focus_attention_fraction"].append(
                float(focus_fractions.mean())
                if focus_fractions is not None
                else 1.0
            )
            diagnostics["focus_token_fraction"].append(focus_token_fraction)
            diagnostics["control_objective"].append(solver.control_objective)

            token = self._sample(logits, config, generator)
            generated.append(int(token.item()))
            if self._should_stop(generated):
                break
        response = self._trim_response(generated)
        return self._base_result(
            example, config, response, generated, probabilities, diagnostics
        )

    def _question_overlap_mask(
        self,
        prompt: str,
        question: str,
        context_length: int,
        radius: int,
    ) -> torch.Tensor:
        if radius < 0:
            raise ValueError("context_overlap_radius must be non-negative")
        document_marker = "#Document#:"
        question_marker = "\n#Question#:"
        document_start = prompt.index(document_marker) + len(document_marker)
        document_end = prompt.index(question_marker, document_start)
        stop_words = {
            "the", "and", "are", "was", "were", "who", "what", "when",
            "where", "why", "how", "does", "did", "has", "have", "had",
            "for", "from", "with", "that", "this", "into", "about", "which",
        }
        terms = {
            word
            for word in re.findall(r"[a-z0-9]+", question.lower())
            if len(word) >= 3 and word not in stop_words
        }
        spans: list[tuple[int, int]] = []
        document = prompt[document_start:document_end]
        for term in terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", document.lower()):
                spans.append(
                    (
                        document_start + match.start(),
                        document_start + match.end(),
                    )
                )
        encoded = self.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=True
        )
        offsets = encoded["offset_mapping"]
        mask = torch.zeros(context_length, dtype=torch.bool)
        for token_index, (start, end) in enumerate(offsets[:context_length]):
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                left = max(0, token_index - radius)
                right = min(context_length, token_index + radius + 1)
                mask[left:right] = True
        if not bool(mask.any()):
            # Keep the mode defined even for questions whose terms do not occur
            # verbatim in the retrieved document.
            mask[:] = True
        return mask

    def _document_token_mask(
        self, prompt: str, context_length: int
    ) -> torch.Tensor:
        document_marker = "#Document#:"
        question_marker = "\n#Question#:"
        document_start = prompt.index(document_marker) + len(document_marker)
        document_end = prompt.index(question_marker, document_start)
        offsets = self.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=True
        )["offset_mapping"]
        mask = torch.zeros(context_length, dtype=torch.bool)
        for token_index, (start, end) in enumerate(offsets[:context_length]):
            if start < document_end and end > document_start:
                mask[token_index] = True
        return mask

    def _retrieval_focus_mask(
        self,
        prompt: str,
        question: str,
        context_length: int,
        *,
        unit_mode: str,
        padding_tokens: int,
    ) -> torch.Tensor:
        """Select the most question-relevant passage or sentence in the document."""

        if padding_tokens < 0:
            raise ValueError("context_overlap_radius must be non-negative")
        document_marker = "#Document#:"
        question_marker = "\n#Question#:"
        document_start = prompt.index(document_marker) + len(document_marker)
        document_end = prompt.index(question_marker, document_start)
        document = prompt[document_start:document_end]
        if unit_mode == "passage":
            matches = list(re.finditer(r"[^\n]+", document))
        elif unit_mode == "sentence":
            matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", document))
        else:
            raise ValueError(f"unknown retrieval unit mode: {unit_mode}")
        matches = [match for match in matches if match.group().strip()]
        if not matches:
            return self._document_token_mask(prompt, context_length)
        scores = bm25_relevance_scores(
            question, [match.group() for match in matches]
        )
        if not scores or max(scores) <= 0:
            return self._document_token_mask(prompt, context_length)
        selected = matches[int(np.argmax(scores))]
        span_start = document_start + selected.start()
        span_end = document_start + selected.end()
        offsets = self.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=True
        )["offset_mapping"]
        mask = torch.zeros(context_length, dtype=torch.bool)
        selected_indices = [
            token_index
            for token_index, (start, end) in enumerate(offsets[:context_length])
            if end > start and start < span_end and end > span_start
        ]
        if not selected_indices:
            return self._document_token_mask(prompt, context_length)
        left = max(0, min(selected_indices) - padding_tokens)
        right = min(context_length, max(selected_indices) + padding_tokens + 1)
        mask[left:right] = True
        return mask

    def _generate_candidate(
        self,
        token: torch.Tensor,
        cache: Any,
        context_length: int,
        config: LookbackGenerationConfig,
        generator: torch.Generator,
    ) -> tuple[list[int], torch.Tensor, Any, torch.Tensor]:
        candidate_cache = _clone_cache(cache)
        candidate_token = token.clone()
        generated: list[int] = []
        features: list[torch.Tensor] = []
        for _ in range(config.chunk_size):
            logits, feature, _, candidate_cache = self._forward_token(
                candidate_token, candidate_cache, context_length, None
            )
            features.append(feature)
            candidate_token = self._sample(
                logits, config, generator, force_sampling=True
            )
            generated.append(int(candidate_token.item()))
            if self._should_stop(generated):
                break
        pooled = torch.stack(features).mean(dim=0)
        return generated, pooled, candidate_cache, candidate_token

    def generate_guided(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        token, _, cache, context_length = self._prepare(example.prompt)
        generator = torch.Generator(device=self.device).manual_seed(
            config.seed + example.dataset_index
        )
        generated: list[int] = []
        probabilities: list[float] = []
        while len(generated) < config.max_new_tokens:
            candidates = [
                self._generate_candidate(
                    token, cache, context_length, config, generator
                )
                for _ in range(config.num_candidates)
            ]
            scores = [float(self.classifier.probability(item[1])) for item in candidates]
            best = int(np.argmax(scores))
            candidate_tokens, _, cache, token = candidates[best]
            remaining = config.max_new_tokens - len(generated)
            generated.extend(candidate_tokens[:remaining])
            probabilities.extend([scores[best]] * min(len(candidate_tokens), remaining))
            if self._should_stop(generated) or len(candidate_tokens) < config.chunk_size:
                break
        response = self._trim_response(generated)
        return self._base_result(
            example, config, response, generated, probabilities, {}
        )

    def _generate_replay_rerank(
        self,
        example: NQExample,
        config: LookbackGenerationConfig,
        *,
        candidate_method: str,
    ) -> dict[str, Any]:
        """Rerank complete sampled generations with an unsteered replay score."""
        if config.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if candidate_method == "baseline":
            generator = self.generate_baseline
        elif candidate_method == "minimum_norm":
            generator = self.generate_minimum_norm
        else:
            raise ValueError(f"unsupported rerank candidate method: {candidate_method}")
        candidates: list[dict[str, Any]] = []
        replay_probabilities: list[list[float]] = []
        for candidate_index in range(config.num_candidates):
            candidate_seed = config.seed + candidate_index * 1_000_003
            candidate_config = replace(
                config,
                method=candidate_method,
                do_sample=True,
                seed=candidate_seed,
            )
            candidate = generator(example, candidate_config)
            candidate["candidate_seed"] = candidate_seed
            candidates.append(candidate)
            replay_probabilities.append(
                self._score_generated_tokens(
                    example,
                    candidate["generated_token_ids"],
                    config.window_size,
                )
            )

        controlled_scores = [
            float(row["mean_factual_probability"]) for row in candidates
        ]
        replay_scores = [
            float(np.mean(values)) if values else 0.0
            for values in replay_probabilities
        ]
        selected_index = int(np.argmax(replay_scores))
        selected = dict(candidates[selected_index])
        selected_replay = replay_probabilities[selected_index]
        selected.update(
            {
                "method": "minimum_norm_rerank",
                "candidate_generation_method": candidate_method,
                "seed": config.seed,
                "controlled_mean_factual_probability": selected[
                    "mean_factual_probability"
                ],
                "controlled_final_factual_probability": selected[
                    "final_factual_probability"
                ],
                "mean_factual_probability": replay_scores[selected_index],
                "final_factual_probability": (
                    selected_replay[-1] if selected_replay else 0.0
                ),
                "selected_candidate_index": selected_index,
                "selected_candidate_seed": candidates[selected_index]["candidate_seed"],
                "candidate_controlled_factual_probabilities": controlled_scores,
                "candidate_online_factual_probabilities": controlled_scores,
                "candidate_replay_factual_probabilities": replay_scores,
                "candidate_responses": [row["response"] for row in candidates],
                "candidate_generated_token_ids": [
                    row["generated_token_ids"] for row in candidates
                ],
                "candidate_exact_matches": [
                    float(row["exact_match"]) for row in candidates
                ],
                "candidate_generated_tokens": [
                    int(row["generated_tokens"]) for row in candidates
                ],
                "candidate_mean_bias_rms": [
                    float(row["mean_bias_rms"]) for row in candidates
                ],
                "candidate_mean_output_kl": [
                    float(row["mean_output_kl"]) for row in candidates
                ],
                "candidate_mean_target_logit_error": [
                    float(row["mean_target_logit_error"]) for row in candidates
                ],
                "candidate_mean_actual_target_logit_error": [
                    float(row["mean_actual_target_logit_error"])
                    for row in candidates
                ],
            }
        )
        return selected

    def generate_baseline_rerank(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        result = self._generate_replay_rerank(
            example, config, candidate_method="baseline"
        )
        result["method"] = "baseline_rerank"
        return result

    def generate_minimum_norm_rerank(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        return self._generate_replay_rerank(
            example, config, candidate_method="minimum_norm"
        )

    def _score_generated_tokens(
        self,
        example: NQExample,
        generated: Sequence[int],
        window_size: int,
    ) -> list[float]:
        """Replay a continuation without control and return rolling scores."""
        features = self.replay_generated_features(example, generated, window_size)
        return [float(self.classifier.probability(feature)) for feature in features]

    def replay_generated_features(
        self,
        example: NQExample,
        generated: Sequence[int],
        window_size: int = 8,
    ) -> list[torch.Tensor]:
        """Replay a continuation without control and return rolling features."""
        token, _, cache, context_length = self._prepare(example.prompt)
        history: list[torch.Tensor] = []
        rolling_features: list[torch.Tensor] = []
        for generated_token in generated:
            _, feature, _, cache = self._forward_token(
                token, cache, context_length, None
            )
            history.append(feature)
            pooled = torch.stack(history[-window_size:]).mean(dim=0)
            rolling_features.append(pooled.detach().cpu())
            token = torch.tensor(
                [[generated_token]], device=self.device, dtype=torch.long
            )
        return rolling_features

    def replay_response_features(
        self,
        example: NQExample,
        response: str,
        window_size: int = 8,
    ) -> list[torch.Tensor]:
        """Tokenize and replay a saved response without adding special tokens."""
        generated = self.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].tolist()
        return self.replay_generated_features(example, generated, window_size)

    def evaluate(
        self, example: NQExample, config: LookbackGenerationConfig
    ) -> dict[str, Any]:
        random.seed(config.seed + example.dataset_index)
        np.random.seed(config.seed + example.dataset_index)
        torch.manual_seed(config.seed + example.dataset_index)
        if config.method == "baseline":
            return self.generate_baseline(example, config)
        if config.method == "baseline_rerank":
            return self.generate_baseline_rerank(example, config)
        if config.method == "minimum_norm":
            return self.generate_minimum_norm(example, config)
        if config.method == "guided":
            return self.generate_guided(example, config)
        if config.method == "minimum_norm_rerank":
            return self.generate_minimum_norm_rerank(example, config)
        raise ValueError(f"unknown method: {config.method}")


def summarize_lookback_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summaries = []
    metrics = (
        "exact_match",
        "generated_tokens",
        "generation_seconds",
        "mean_factual_probability",
        "final_factual_probability",
        "mean_initial_factual_probability",
        "mean_predicted_factual_probability",
        "mean_actual_target_probability",
        "mean_bias_norm",
        "mean_bias_rms",
        "maximum_bias_abs",
        "mean_active_bias_count",
        "mean_negative_bias_fraction",
        "mean_attention_kl",
        "mean_output_kl",
        "mean_target_logit_error",
        "mean_target_logit_shortfall",
        "mean_actual_target_logit_error",
        "mean_actual_target_logit_shortfall",
        "mean_actual_logit_gain",
        "mean_focus_attention_fraction",
        "mean_focus_token_fraction",
        "mean_control_objective",
    )
    for method, method_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"method": method, "n": len(method_rows)}
        for metric in metrics:
            # Older resumable shards may predate newly added diagnostics.
            values = [row[metric] for row in method_rows if metric in row]
            summary[metric] = float(np.mean(values)) if values else float("nan")
        summaries.append(summary)
    return summaries
