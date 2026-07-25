#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
output="artifacts/lookback_nq/validation_offset160_n100_baseline_rerank_replay/candidates4"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${output}" \
  --methods baseline_rerank \
  --offset 160 --limit 100 --max-new-tokens 64 \
  --num-candidates 4 \
  --target-mode relative --target-logit-shift 4 \
  --temperature 0.9 --top-p 0.95
