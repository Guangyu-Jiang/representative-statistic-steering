#!/usr/bin/env bash
set -euo pipefail

gpu="${PPLM_GPU:-2}"
calibration="artifacts/pplm_sentiment/quantile_calibration_sst5/quantile_targets.json"
root="artifacts/pplm_sentiment/quantile_target_development"
prefixes=("The library" "The stadium" "The factory" "The island" "The bakery")

if [[ ! -f "${calibration}" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python \
    scripts/calibrate_pplm_quantile_targets.py \
    --output-dir "$(dirname "${calibration}")" \
    --quantiles 0.5 0.7 0.75 0.8 0.9 \
    --device cuda:0
fi

for quantile in 0.5 0.7 0.8 0.9; do
  tag="${quantile/./p}"
  output="${root}/q${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "${output}" \
    --methods minimum_norm \
    --targets positive negative \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 \
    --seeds 11 \
    --quantile-targets "${calibration}" \
    --target-quantile "${quantile}" \
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
done
