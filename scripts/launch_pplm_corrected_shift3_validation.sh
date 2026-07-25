#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-0}"
wait_pid="${WAIT_PID:-}"
output="artifacts/pplm_sentiment/corrected_accumulated_shift3_validation_seeds22_33"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "${output}" \
  --methods minimum_norm \
  --targets positive negative \
  --max-new-tokens 24 \
  --seeds 22 33 \
  --target-margin-shift 3.0 \
  --minimum-norm-steps 5 \
  --minimum-norm-damping 0.5 \
  --ridge 0.01 \
  --maximum-relative-norm 0.03 \
  --maximum-token-kl 0.75 \
  --difficult-margin-threshold -4.0 \
  --difficult-maximum-token-kl 1.5 \
  --statistic-mode margin \
  --gradient-block-normalization 0.5 \
  --gm-scale 0.85 \
  --preserve-top-log-probs 2 \
  --log-probability-preservation-weight 0.01 \
  --persistent-cache \
  --device cuda:0

CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_pplm_sentiment.py \
  --input "${output}/generations.csv" \
  --output-dir "${output}/external_eval" \
  --device cuda:0
