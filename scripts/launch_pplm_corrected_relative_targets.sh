#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-1}"
root="artifacts/pplm_sentiment"

for shift in 0.5 1.0 1.5 2.0 3.0; do
  tag="${shift/./p}"
  output_dir="${root}/corrected_accumulated_relative_shift${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "${output_dir}" \
    --methods minimum_norm \
    --targets positive negative \
    --prefixes "The library" "The stadium" "The factory" "The island" "The bakery" \
    --max-new-tokens 24 --seeds 11 \
    --target-margin-shift "${shift}" \
    --minimum-norm-steps 5 --minimum-norm-damping 0.5 \
    --ridge 0.01 --maximum-relative-norm 0.03 \
    --maximum-token-kl 0.75 \
    --difficult-margin-threshold -4.0 --difficult-maximum-token-kl 1.5 \
    --statistic-mode margin --gradient-block-normalization 0.5 \
    --gm-scale 0.85 --preserve-top-log-probs 2 \
    --log-probability-preservation-weight 0.01 \
    --persistent-cache --device cuda:0
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_pplm_sentiment.py \
    --input "${output_dir}/generations.csv" \
    --output-dir "${output_dir}/external_eval" --device cuda:0
done
