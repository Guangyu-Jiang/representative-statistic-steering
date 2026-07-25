from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch
from transformers.cache_utils import DynamicCache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from generate_causal_head_perturbation import fork_dynamic_cache, reuse_generation_columns


def test_cache_fork_does_not_advance_authoritative_cache() -> None:
    cache = DynamicCache()
    key = torch.zeros((1, 1, 2, 4))
    value = torch.zeros((1, 1, 2, 4))
    cache.update(key, value, layer_idx=0)
    fork = fork_dynamic_cache(cache)
    assert fork is not None
    fork.update(torch.ones_like(key), torch.ones_like(value), layer_idx=0)
    assert cache.get_seq_length() == 2
    assert fork.get_seq_length() == 4
    assert cache.key_cache[0].data_ptr() != fork.key_cache[0].data_ptr()


def test_reuse_generation_columns_aligns_by_dataset_index() -> None:
    output = pd.DataFrame({"dataset_index": [20, 10], "Question": ["b", "a"]})
    prior = pd.DataFrame(
        {
            "dataset_index": [10, 20],
            "baseline_answer": ["answer-a", "answer-b"],
            "other_answer": ["ignore-a", "ignore-b"],
        }
    )

    reused = reuse_generation_columns(output, prior, ["baseline"])

    assert reused["baseline_answer"].tolist() == ["answer-b", "answer-a"]
    assert "other_answer" not in reused
