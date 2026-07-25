"""Hugging Face model-loading helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class HFModelBundle:
    """Loaded tokenizer, model, and resolved device metadata."""

    model_name: str
    tokenizer_name: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _select_dtype(dtype_name: str | None, device: torch.device) -> torch.dtype | None:
    if dtype_name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def _resolve_local_snapshot_path(name_or_path: str, local_files_only: bool) -> str:
    if not local_files_only:
        return name_or_path
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    return snapshot_download(repo_id=name_or_path, local_files_only=True)


def load_hf_model(model_config: dict[str, Any], extraction_config: dict[str, Any]) -> HFModelBundle:
    """Load a tokenizer and a model for forward-pass hidden-state extraction."""

    model_name = model_config["name"]
    tokenizer_name = model_config.get("tokenizer_name", model_name)
    device = _select_device(model_config.get("device", "auto"))
    local_files_only = bool(
        extraction_config.get("local_files_only", model_config.get("local_files_only", False))
    )
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    torch_dtype = _select_dtype(model_config.get("torch_dtype"), device)
    device_map = model_config.get("device_map", extraction_config.get("device_map"))
    use_fast = bool(model_config.get("use_fast", True))

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    resolved_tokenizer_name = _resolve_local_snapshot_path(tokenizer_name, local_files_only=local_files_only)
    resolved_model_name = _resolve_local_snapshot_path(model_name, local_files_only=local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=use_fast,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_class = model_config.get("model_class", "causal_lm")
    loader = AutoModelForCausalLM if model_class == "causal_lm" else AutoModel
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "dtype": torch_dtype,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = loader.from_pretrained(resolved_model_name, **load_kwargs)
    if device_map is None:
        model.to(device)
    model.eval()

    resolved_device = next(model.parameters()).device if device_map is not None else device

    return HFModelBundle(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
    )
