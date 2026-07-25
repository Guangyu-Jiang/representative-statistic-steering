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
  --statistic-mode distribution
  --device cuda:0
)

names=(
  dist_shift2_gm05 dist_shift2p5_gm05
  dist_shift3_gm04 dist_shift3_gm045 dist_shift3_gm055 dist_shift3_gm06
  dist_shift3p5_gm045 dist_shift3p5_gm05
  dist_shift3_gm05_topk5 dist_shift3_gm05_temp09
  dist_shift3_gm05_value dist_shift3_gm05_value_last12
)
arguments=(
  "--target-margin-shift 2 --gm-scale 0.5"
  "--target-margin-shift 2.5 --gm-scale 0.5"
  "--target-margin-shift 3 --gm-scale 0.4"
  "--target-margin-shift 3 --gm-scale 0.45"
  "--target-margin-shift 3 --gm-scale 0.55"
  "--target-margin-shift 3 --gm-scale 0.6"
  "--target-margin-shift 3.5 --gm-scale 0.45"
  "--target-margin-shift 3.5 --gm-scale 0.5"
  "--target-margin-shift 3 --gm-scale 0.5 --top-k 5"
  "--target-margin-shift 3 --gm-scale 0.5 --temperature 0.9"
  "--target-margin-shift 3 --gm-scale 0.5 --cache-component value"
  "--target-margin-shift 3 --gm-scale 0.5 --cache-component value --cache-last-n-layers 12"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/improvement_refinement
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/improvement_refinement/${name}"
  log="artifacts/logs/pplm_refinement_${name}.log"
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
