#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
base=(
  --targets positive negative
  --prefixes "${prefixes[@]}"
  --max-new-tokens 24
  --seeds 11 22 33
  --persistent-cache
  --device cuda:0
)
minimum_norm=(
  --methods minimum_norm
  --minimum-norm-steps 3
  --ridge 0.1
  --maximum-relative-norm 0.03
  --statistic-mode margin
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
)

names=(
  shift05_gm03 shift1_gm03 shift1_gm04
  shift2_gm03 shift2_gm04 shift3_gm03 shift3_gm04
  preserve_shift1_gm04 preserve_shift2_gm04 preserve_shift3_gm04
  gamma025_shift2_gm04 cap003_shift2_gm04
)
arguments=(
  "--target-margin-shift 0.5 --gm-scale 0.3"
  "--target-margin-shift 1 --gm-scale 0.3"
  "--target-margin-shift 1 --gm-scale 0.4"
  "--target-margin-shift 2 --gm-scale 0.3"
  "--target-margin-shift 2 --gm-scale 0.4"
  "--target-margin-shift 3 --gm-scale 0.3"
  "--target-margin-shift 3 --gm-scale 0.4"
  "--target-margin-shift 1 --gm-scale 0.4 --preserve-top-log-probs 1 --log-probability-preservation-weight 0.001"
  "--target-margin-shift 2 --gm-scale 0.4 --preserve-top-log-probs 1 --log-probability-preservation-weight 0.001"
  "--target-margin-shift 3 --gm-scale 0.4 --preserve-top-log-probs 1 --log-probability-preservation-weight 0.001"
  "--target-margin-shift 2 --gm-scale 0.4 --gradient-block-normalization 0.25"
  "--target-margin-shift 2 --gm-scale 0.4 --maximum-relative-norm 0.003"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/persistent_tuning
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=$((index % 3))
  output="artifacts/pplm_sentiment/persistent_tuning/$name"
  log="artifacts/logs/pplm_persistent_${name}.log"
  read -r -a extra <<< "${arguments[$index]}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${base[@]}" "${minimum_norm[@]}" "${extra[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done

pplm_output="artifacts/pplm_sentiment/persistent_tuning/pplm10"
(
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$pplm_output" "${base[@]}" \
    --methods pplm --pplm-steps 10 --pplm-step-size 0.04 \
    --maximum-relative-norm 0.10 --gm-scale 0.95
  CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
    --input "$pplm_output/generations.csv" \
    --output-dir "$pplm_output/external_eval" --device cuda:0
) > artifacts/logs/pplm_persistent_pplm10.log 2>&1 < /dev/null &

wait
