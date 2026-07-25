#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-0}"
root="artifacts/pplm_sentiment/corrected_accumulated_output_preservation_development"
prefixes=("The library" "The stadium" "The factory" "The island" "The bakery")

run_variant() {
  local name="$1"
  local top_count="$2"
  local weight="$3"
  local output="${root}/${name}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "${output}" \
    --methods minimum_norm --targets positive negative \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 --seeds 11 \
    --target-margin-shift 3.0 \
    --minimum-norm-steps 5 --minimum-norm-damping 0.5 \
    --ridge 0.01 --maximum-relative-norm 0.03 \
    --maximum-token-kl 0.75 \
    --difficult-margin-threshold -4.0 --difficult-maximum-token-kl 1.5 \
    --statistic-mode margin --gradient-block-normalization 0.5 \
    --gm-scale 0.85 --preserve-top-log-probs "${top_count}" \
    --log-probability-preservation-weight "${weight}" \
    --persistent-cache --device cuda:0

  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_pplm_sentiment.py \
    --input "${output}/generations.csv" \
    --output-dir "${output}/external_eval" --device cuda:0
}

run_variant top2_w0p05 2 0.05
run_variant top2_w0p1 2 0.1
run_variant top5_w0p1 5 0.1
run_variant top5_w0p5 5 0.5
