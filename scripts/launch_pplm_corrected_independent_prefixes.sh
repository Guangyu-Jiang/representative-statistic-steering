#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-1}"
output_dir="${PPLM_OUTPUT_DIR:-artifacts/pplm_sentiment/corrected_accumulated_independent_prefixes_seed44}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "${output_dir}" \
  --methods baseline pplm minimum_norm \
  --targets positive negative \
  --prefixes \
    "The computer" "The ocean" "The school" "The house" "The phone" \
    "The garden" "The hospital" "The train" "The music" "The meeting" \
  --max-new-tokens 24 \
  --seeds 44 \
  --minimum-norm-steps 5 \
  --minimum-norm-damping 0.5 \
  --ridge 0.01 \
  --maximum-relative-norm 0.03 \
  --maximum-token-kl 0.75 \
  --difficult-margin-threshold -4.0 \
  --difficult-maximum-token-kl 1.5 \
  --statistic-mode margin \
  --target-probability 0.95 \
  --gradient-block-normalization 0.5 \
  --gm-scale 0.85 \
  --preserve-top-log-probs 2 \
  --log-probability-preservation-weight 0.01 \
  --persistent-cache \
  --pplm-steps 5 \
  --pplm-step-size 0.04 \
  --device cuda:0

CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_pplm_sentiment.py \
  --input "${output_dir}/generations.csv" \
  --output-dir "${output_dir}/external_eval" \
  --device cuda:0
