#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
common=(
  --methods minimum_norm
  --targets positive negative
  --prefixes "${prefixes[@]}"
  --max-new-tokens 24
  --seeds 11 22 33
  --minimum-norm-steps 3
  --ridge 0.1
  --maximum-relative-norm 0.03
  --statistic-mode margin
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
  --gm-scale 0.95
  --persistent-cache
  --device cuda:0
)

names=(
  shift05_kl005
  shift1_kl003 shift1_kl005 shift1_kl01 shift1_kl015
  shift2_kl003 shift2_kl005 shift2_kl01
)
shifts=(0.5 1 1 1 1 2 2 2)
budgets=(0.05 0.03 0.05 0.1 0.15 0.03 0.05 0.1)

mkdir -p artifacts/logs artifacts/pplm_sentiment/persistent_kl_tuning
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=$((index % 3))
  output="artifacts/pplm_sentiment/persistent_kl_tuning/$name"
  log="artifacts/logs/pplm_persistent_kl_${name}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" \
      --target-margin-shift "${shifts[$index]}" \
      --maximum-token-kl "${budgets[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
