#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
output="artifacts/lookback_nq/development_n60_matched_rerank_diagnostics/candidates4"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${output}" \
  --methods baseline_rerank minimum_norm_rerank \
  --offset 0 --limit 60 --max-new-tokens 64 \
  --num-candidates 4 \
  --target-mode relative --target-logit-shift 4 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 0.5 \
  --context-bias-mode question_overlap --context-overlap-radius 8 \
  --active-control-count 128 --bias-constraint unrestricted \
  --temperature 0.9 --top-p 0.95
