"""Utilities for text generation and generation-time entropy measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from aen_replication.models.hf_model import HFModelBundle


@dataclass(slots=True)
class GenerationResult:
    """Decoded output and optional entropy statistics for one prompt."""

    prompt_text: str
    response_text: str
    generated_token_count: int
    average_entropy: float | None


def render_prompts(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    use_chat_template: bool,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> list[str]:
    """Render prompts for an instruction-tuned model."""

    return [
        render_prompt(
            bundle=bundle,
            prompt_text=prompt_text,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        for prompt_text in prompt_texts
    ]


def render_prompt(
    bundle: HFModelBundle,
    prompt_text: str,
    use_chat_template: bool,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render a prompt for an instruction-tuned model."""

    tokenizer = bundle.tokenizer
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if system_prompt:
        return f"{system_prompt}\n\n{prompt_text}"
    return prompt_text


def _entropy_from_logits(logits: Tensor) -> Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


def _build_generate_kwargs(generation_config: dict[str, Any], *, return_entropy: bool) -> dict[str, Any]:
    temperature = float(generation_config.get("temperature", 0.1))
    do_sample = bool(generation_config.get("do_sample", temperature > 0.0))
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 64)),
        "return_dict_in_generate": True,
        "output_scores": return_entropy,
    }
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        if generation_config.get("top_p") is not None:
            generate_kwargs["top_p"] = float(generation_config["top_p"])
    else:
        generate_kwargs["do_sample"] = False
    return generate_kwargs


def generate_batch(
    bundle: HFModelBundle,
    prompt_texts: list[str],
    generation_config: dict[str, Any],
    *,
    return_entropy: bool = False,
) -> list[GenerationResult]:
    """Generate batched responses and optionally compute average token entropy."""

    if not prompt_texts:
        return []

    use_chat_template = bool(generation_config.get("use_chat_template", True))
    system_prompt = generation_config.get("system_prompt")
    rendered_prompts = render_prompts(
        bundle=bundle,
        prompt_texts=prompt_texts,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        add_generation_prompt=True,
    )
    original_padding_side = bundle.tokenizer.padding_side
    bundle.tokenizer.padding_side = "left"
    encoded = bundle.tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=generation_config.get("prompt_max_length"),
    )
    bundle.tokenizer.padding_side = original_padding_side
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    generate_kwargs = _build_generate_kwargs(generation_config, return_entropy=return_entropy)
    generate_kwargs["pad_token_id"] = bundle.tokenizer.pad_token_id
    generate_kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id

    with torch.no_grad():
        generation_output = bundle.model.generate(**encoded, **generate_kwargs)

    prompt_length = encoded["input_ids"].shape[1]
    results: list[GenerationResult] = []
    pad_token_id = bundle.tokenizer.pad_token_id
    eos_token_id = bundle.tokenizer.eos_token_id

    for index, rendered_prompt in enumerate(rendered_prompts):
        generated_ids = generation_output.sequences[index, prompt_length:]
        generated_ids_cpu = generated_ids.detach().cpu()
        token_ids = generated_ids_cpu.tolist()
        effective_token_count = len(token_ids)
        if pad_token_id is not None:
            effective_token_count = sum(token_id != pad_token_id for token_id in token_ids)
        if eos_token_id is not None and eos_token_id in token_ids:
            effective_token_count = min(effective_token_count, token_ids.index(eos_token_id) + 1)
        response_text = bundle.tokenizer.decode(generated_ids_cpu, skip_special_tokens=True).strip()

        average_entropy = None
        if return_entropy and generation_output.scores:
            entropies: list[float] = []
            for step_index, step_logits in enumerate(generation_output.scores[:effective_token_count]):
                entropies.append(_entropy_from_logits(step_logits[index]).item())
            if entropies:
                average_entropy = float(sum(entropies) / len(entropies))

        results.append(
            GenerationResult(
                prompt_text=rendered_prompt,
                response_text=response_text,
                generated_token_count=int(effective_token_count),
                average_entropy=average_entropy,
            )
        )
    return results


def generate_one(
    bundle: HFModelBundle,
    prompt_text: str,
    generation_config: dict[str, Any],
    *,
    return_entropy: bool = False,
) -> GenerationResult:
    """Generate one response and optionally compute average token entropy."""

    return generate_batch(
        bundle=bundle,
        prompt_texts=[prompt_text],
        generation_config=generation_config,
        return_entropy=return_entropy,
    )[0]
