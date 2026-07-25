#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-1}"
wait_pid="${WAIT_PID:-}"
root="artifacts/pplm_sentiment/corrected_accumulated_mechanism_development"
prefixes=("The library" "The stadium" "The factory" "The island" "The bakery")

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

run_target() {
  local name="$1"
  local target="$2"
  local shift="$3"
  shift 3
  local output_dir="${root}/${name}_${target}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "${output_dir}" \
    --methods minimum_norm \
    --targets "${target}" \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 --seeds 11 \
    --target-margin-shift "${shift}" \
    --minimum-norm-steps 5 --minimum-norm-damping 0.5 \
    --ridge 0.01 --maximum-relative-norm 0.03 \
    --maximum-token-kl 0.75 \
    --difficult-margin-threshold -4.0 --difficult-maximum-token-kl 1.5 \
    --statistic-mode margin --gm-scale 0.85 \
    --preserve-top-log-probs 2 --log-probability-preservation-weight 0.01 \
    --persistent-cache --device cuda:0 \
    "$@"

  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_pplm_sentiment.py \
    --input "${output_dir}/generations.csv" \
    --output-dir "${output_dir}/external_eval" --device cuda:0
}

run_variant() {
  local name="$1"
  shift
  run_target "${name}" negative 1.0 "$@"
  run_target "${name}" positive 0.5 "$@"
}

run_variant all_g0 --cache-component all --gradient-block-normalization 0
run_variant key_g0 --cache-component key --gradient-block-normalization 0
run_variant key_g0p5 --cache-component key --gradient-block-normalization 0.5
run_variant value_g0 --cache-component value --gradient-block-normalization 0
run_variant value_g0p5 --cache-component value --gradient-block-normalization 0.5
run_variant last4_g0p5 --cache-component all --cache-last-n-layers 4 --gradient-block-normalization 0.5
run_variant last8_g0p5 --cache-component all --cache-last-n-layers 8 --gradient-block-normalization 0.5
run_variant last12_g0p5 --cache-component all --cache-last-n-layers 12 --gradient-block-normalization 0.5
