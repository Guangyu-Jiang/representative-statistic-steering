#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pid="${WAIT_PID:-}"
output="artifacts/lookback_nq/development_n30_minimum_norm_rerank_replay/candidates4_sparse128_shift4_cap0.5"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${output}" \
  --methods minimum_norm_rerank \
  --offset 0 --limit 30 --max-new-tokens 64 \
  --num-candidates 4 \
  --target-mode relative --target-logit-shift 4 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 0.5 \
  --context-bias-mode question_overlap --context-overlap-radius 8 \
  --active-control-count 128 --bias-constraint unrestricted
