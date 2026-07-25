#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
common=(
  --methods minimum_norm
  --targets positive negative
  --prefixes "${prefixes[@]}"
  --max-new-tokens 16
  --seeds 11 22
  --minimum-norm-steps 3
  --ridge 0.1
  --maximum-relative-norm 0.03
  --target-margin-shift 3
  --device cuda:0
)

names=(
  dist_floor02_gm05 dist_floor03_gm05 dist_floor04_gm05
  dist_floor05_gm05 dist_floor06_gm05 dist_floor07_gm05
  dist_floor03_gm04 dist_floor04_gm04 dist_floor05_gm04 dist_floor06_gm04
  dist_floor05_gm035 dist_floor05_gm06
  margin_floor05_gm05 margin_floor06_gm05
)
arguments=(
  "--statistic-mode distribution --minimum-target-probability 0.2 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.3 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.4 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.5 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.6 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.7 --gm-scale 0.5"
  "--statistic-mode distribution --minimum-target-probability 0.3 --gm-scale 0.4"
  "--statistic-mode distribution --minimum-target-probability 0.4 --gm-scale 0.4"
  "--statistic-mode distribution --minimum-target-probability 0.5 --gm-scale 0.4"
  "--statistic-mode distribution --minimum-target-probability 0.6 --gm-scale 0.4"
  "--statistic-mode distribution --minimum-target-probability 0.5 --gm-scale 0.35"
  "--statistic-mode distribution --minimum-target-probability 0.5 --gm-scale 0.6"
  "--statistic-mode margin --minimum-target-probability 0.5 --gm-scale 0.5"
  "--statistic-mode margin --minimum-target-probability 0.6 --gm-scale 0.5"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/hybrid_target_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/hybrid_target_tuning/${name}"
  log="artifacts/logs/pplm_hybrid_${name}.log"
  read -r -a extra <<< "${arguments[$index]}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" "${extra[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" \
      --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done

wait
