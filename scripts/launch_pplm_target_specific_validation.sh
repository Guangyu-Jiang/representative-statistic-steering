#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-0}"
wait_pid="${WAIT_PID:-}"
root="artifacts/pplm_sentiment/corrected_accumulated_target_specific_validation_seeds22_33"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

run_target() {
  local target="$1"
  local shift="$2"
  local tag="${shift/./p}"
  local output_dir="${root}/${target}_shift${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "${output_dir}" \
    --methods minimum_norm \
    --targets "${target}" \
    --max-new-tokens 24 \
    --seeds 22 33 \
    --target-margin-shift "${shift}" \
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
    --input "${output_dir}/generations.csv" \
    --output-dir "${output_dir}/external_eval" \
    --device cuda:0
}

# Development selected different smallest effective shifts for the two targets.
run_target negative 1.0
run_target positive 0.5
