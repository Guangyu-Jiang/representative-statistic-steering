#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
common=(
  --methods baseline
  --target-mode relative
  --ridge 0
  --device cuda
)

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/development_n60/baseline_sampled \
  --offset 0 --limit 60 --max-new-tokens 64 --do-sample \
  "${common[@]}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/heldout_offset60_n100_max256/baseline_greedy \
  --offset 60 --limit 100 --max-new-tokens 256 \
  "${common[@]}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/heldout_offset60_n100_max256/baseline_sampled \
  --offset 60 --limit 100 --max-new-tokens 256 --do-sample \
  "${common[@]}"
