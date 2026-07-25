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
  --target-margin-shift 3
  --minimum-target-probability 0.5
  --device cuda:0
)

names=(
  margin_g025 margin_g05 margin_g075 margin_g1
  dist_g025 dist_g05 dist_g075 dist_g1
  margin_g05_gm035 margin_g05_gm045 dist_g05_gm035 dist_g05_gm05
)
arguments=(
  "--statistic-mode margin --gradient-block-normalization 0.25 --gm-scale 0.5"
  "--statistic-mode margin --gradient-block-normalization 0.5 --gm-scale 0.5"
  "--statistic-mode margin --gradient-block-normalization 0.75 --gm-scale 0.5"
  "--statistic-mode margin --gradient-block-normalization 1 --gm-scale 0.5"
  "--statistic-mode distribution --gradient-block-normalization 0.25 --gm-scale 0.45"
  "--statistic-mode distribution --gradient-block-normalization 0.5 --gm-scale 0.45"
  "--statistic-mode distribution --gradient-block-normalization 0.75 --gm-scale 0.45"
  "--statistic-mode distribution --gradient-block-normalization 1 --gm-scale 0.45"
  "--statistic-mode margin --gradient-block-normalization 0.5 --gm-scale 0.35"
  "--statistic-mode margin --gradient-block-normalization 0.5 --gm-scale 0.45"
  "--statistic-mode distribution --gradient-block-normalization 0.5 --gm-scale 0.35"
  "--statistic-mode distribution --gradient-block-normalization 0.5 --gm-scale 0.5"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/weighted_metric_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/weighted_metric_tuning/${name}"
  log="artifacts/logs/pplm_weighted_${name}.log"
  read -r -a extra <<< "${arguments[$index]}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" "${extra[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
