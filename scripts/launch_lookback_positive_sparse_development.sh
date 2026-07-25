#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pid="${WAIT_PID:-}"
if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

for count in 64 128 256; do
  for shift in 4 6; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
      --output-dir "artifacts/lookback_nq/development_n60_positive_sparse/heads${count}_shift${shift}_cap0.5" \
      --methods minimum_norm \
      --offset 0 --limit 60 --max-new-tokens 64 \
      --target-mode relative --target-logit-shift "${shift}" \
      --ridge 0 --solver-steps 16 --solver-damping 0.5 \
      --maximum-bias-rms 0.5 \
      --context-bias-mode question_overlap --context-overlap-radius 8 \
      --active-control-count "${count}" --bias-constraint nonnegative
  done
done
