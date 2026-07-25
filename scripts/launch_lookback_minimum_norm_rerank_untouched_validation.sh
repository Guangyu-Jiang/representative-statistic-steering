#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pids="${WAIT_PIDS:-}"
output="artifacts/lookback_nq/validation_offset160_n100_minimum_norm_rerank_replay/candidates4_sparse128_shift4_cap0.5"

for wait_pid in ${wait_pids}; do
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
done

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${output}" \
  --methods baseline minimum_norm_rerank \
  --offset 160 --limit 100 --max-new-tokens 64 \
  --num-candidates 4 \
  --target-mode relative --target-logit-shift 4 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 0.5 \
  --context-bias-mode question_overlap --context-overlap-radius 8 \
  --active-control-count 128 --bias-constraint unrestricted \
  --do-sample --temperature 0.9 --top-p 0.95
