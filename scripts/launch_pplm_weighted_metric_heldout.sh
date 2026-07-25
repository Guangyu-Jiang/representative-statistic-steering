#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
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
  --target-margin-shift 3
  --minimum-target-probability 0.5
  --device cuda:0
)

names=(gamma025_gm05 gamma05_gm045 gamma075_gm05 gamma1_gm05 gamma05_gm035)
gammas=(0.25 0.5 0.75 1 0.5)
mixes=(0.5 0.45 0.5 0.5 0.35)
gpus=(0 2 0 2 0)

mkdir -p artifacts/logs artifacts/pplm_sentiment/weighted_metric_heldout
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=${gpus[$index]}
  output="artifacts/pplm_sentiment/weighted_metric_heldout/${name}"
  log="artifacts/logs/pplm_weighted_heldout_${name}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" \
      --gradient-block-normalization "${gammas[$index]}" \
      --gm-scale "${mixes[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
