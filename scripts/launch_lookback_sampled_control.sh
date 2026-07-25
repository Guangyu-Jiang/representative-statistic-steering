#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pid="${WAIT_PID:-}"
if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 10
  done
fi

common=(
  --methods minimum_norm
  --target-mode relative --target-logit-shift 4
  --ridge 0 --solver-steps 16 --solver-damping 0.5
  --maximum-bias-rms 0.5
  --context-bias-mode question_overlap --context-overlap-radius 8
  --active-control-count 128 --bias-constraint unrestricted
  --do-sample --temperature 0.9 --top-p 0.95
)

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/development_n60_sampled/minimum_norm_sparse128_shift4_cap0.5 \
  --offset 0 --limit 60 --max-new-tokens 64 \
  "${common[@]}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/heldout_offset60_n100_max256/minimum_norm_sampled_sparse128_shift4_cap0.5 \
  --offset 60 --limit 100 --max-new-tokens 256 \
  "${common[@]}"
