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
  --device cuda:0
)

names=(
  dist_shift3_gm045 dist_shift3_gm05 dist_shift3_gm055 dist_shift3_topk5
  margin_shift3_gm05 margin_floor05_gm05 margin_floor06_gm05
  dist_floor02_gm05 dist_floor05_gm045 dist_floor05_gm05
  margin_abs06_gm05 margin_floor05_topk5
)
arguments=(
  "--statistic-mode distribution --target-margin-shift 3 --gm-scale 0.45"
  "--statistic-mode distribution --target-margin-shift 3 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 3 --gm-scale 0.55"
  "--statistic-mode distribution --target-margin-shift 3 --gm-scale 0.5 --top-k 5"
  "--statistic-mode margin --target-margin-shift 3 --gm-scale 0.5"
  "--statistic-mode margin --target-margin-shift 3 --minimum-target-probability 0.5 --gm-scale 0.5"
  "--statistic-mode margin --target-margin-shift 3 --minimum-target-probability 0.6 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 3 --minimum-target-probability 0.2 --gm-scale 0.5"
  "--statistic-mode distribution --target-margin-shift 3 --minimum-target-probability 0.5 --gm-scale 0.45"
  "--statistic-mode distribution --target-margin-shift 3 --minimum-target-probability 0.5 --gm-scale 0.5"
  "--statistic-mode margin --target-probability 0.6 --gm-scale 0.5"
  "--statistic-mode margin --target-margin-shift 3 --minimum-target-probability 0.5 --gm-scale 0.5 --top-k 5"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/matched_protocol_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/matched_protocol_tuning/${name}"
  log="artifacts/logs/pplm_matched_protocol_${name}.log"
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
