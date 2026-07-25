#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
root="artifacts/lookback_nq/heldout_offset60_n100_max256"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${root}/minimum_norm_sparse128_shift4_cap0.5" \
  --methods minimum_norm \
  --offset 60 --limit 100 --max-new-tokens 256 \
  --target-mode relative --target-logit-shift 4 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 0.5 \
  --context-bias-mode question_overlap --context-overlap-radius 8 \
  --active-control-count 128 --bias-constraint unrestricted

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${root}/minimum_norm_gated0.93_shift4_cap1.0" \
  --methods minimum_norm \
  --offset 60 --limit 100 --max-new-tokens 256 \
  --target-mode relative --target-logit-shift 4 \
  --control-trigger-probability 0.93 --high-confidence-logit-shift 0 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 1.0 \
  --context-bias-mode question_overlap --context-overlap-radius 8

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${root}/guided_official" \
  --methods guided \
  --offset 60 --limit 100 --max-new-tokens 256 \
  --do-sample --temperature 0.9 --top-p 0.95 \
  --chunk-size 8 --num-candidates 8
