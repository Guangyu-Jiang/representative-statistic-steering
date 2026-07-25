"""Readout strategies for token-level hidden states."""

from __future__ import annotations

import torch


def last_token_readout(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the final non-padding token representation for each sequence."""

    last_indices = attention_mask.sum(dim=1) - 1
    last_indices = last_indices.clamp(min=0)
    batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
    return hidden_state[batch_indices, last_indices]


def mean_pool_readout(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean over valid tokens only."""

    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def unmasked_mean_pool_readout(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean over all sequence positions, including padding."""

    del attention_mask
    return hidden_state.mean(dim=1)


def apply_readout(strategy: str, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply a configured readout strategy."""

    if strategy == "last_token":
        return last_token_readout(hidden_state=hidden_state, attention_mask=attention_mask)
    if strategy == "mean_pool":
        return mean_pool_readout(hidden_state=hidden_state, attention_mask=attention_mask)
    if strategy == "unmasked_mean_pool":
        return unmasked_mean_pool_readout(hidden_state=hidden_state, attention_mask=attention_mask)
    raise ValueError(f"Unsupported readout strategy: {strategy}")
