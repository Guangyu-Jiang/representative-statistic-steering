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
  --device cuda:0
)

names=(
  margin_abs06_gm035 margin_abs07_gm035 margin_abs08_gm035 margin_abs09_gm035
  margin_abs09_gm05 margin_abs08_gm065
  margin_shift3_gm05 margin_shift4_gm05 margin_shift5_gm05
  margin_abs08_kl02 margin_abs08_kl04 margin_abs08_kl08
  dist_abs07_gm05 dist_abs08_gm05 dist_abs09_gm05
  dist_shift3_gm05 dist_shift4_gm05 dist_shift5_gm05
  dist_abs08_gm035 dist_abs09_gm035
  margin_abs08_gm05_temp085 margin_abs09_gm04_temp085
  dist_shift4_gm04_temp085 margin_abs08_gm05_topk5
)
arguments=(
  "--target-probability 0.6 --gm-scale 0.35"
  "--target-probability 0.7 --gm-scale 0.35"
  "--target-probability 0.8 --gm-scale 0.35"
  "--target-probability 0.9 --gm-scale 0.35"
  "--target-probability 0.9 --gm-scale 0.5"
  "--target-probability 0.8 --gm-scale 0.65"
  "--target-margin-shift 3 --gm-scale 0.5"
  "--target-margin-shift 4 --gm-scale 0.5"
  "--target-margin-shift 5 --gm-scale 0.5"
  "--target-probability 0.8 --gm-scale 0.95 --maximum-token-kl 0.02"
  "--target-probability 0.8 --gm-scale 0.95 --maximum-token-kl 0.04"
  "--target-probability 0.8 --gm-scale 0.95 --maximum-token-kl 0.08"
  "--statistic-mode distribution --target-probability 0.7 --gm-scale 0.5"
  "--statistic-mode distribution --target-probability 0.8 --gm-scale 0.5"
  "--statistic-mode distribution --target-probability 0.9 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 3 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 4 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 5 --gm-scale 0.5"
  "--statistic-mode distribution --target-probability 0.8 --gm-scale 0.35"
  "--statistic-mode distribution --target-probability 0.9 --gm-scale 0.35"
  "--target-probability 0.8 --gm-scale 0.5 --temperature 0.85"
  "--target-probability 0.9 --gm-scale 0.4 --temperature 0.85"
  "--statistic-mode distribution --target-margin-shift 4 --gm-scale 0.4 --temperature 0.85"
  "--target-probability 0.8 --gm-scale 0.5 --top-k 5"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/improvement_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/improvement_tuning/${name}"
  log="artifacts/logs/pplm_improvement_${name}.log"
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
