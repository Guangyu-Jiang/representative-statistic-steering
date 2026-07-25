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
  --statistic-mode margin
  --target-margin-shift 3
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
  --gm-scale 0.45
  --device cuda:0
)

names=(
  cap003 cap004 cap005 cap006 cap004_gm05
  shift4_cap004 shift4_cap005 floor06_cap004 gamma025_cap004 gamma025_cap005
)
arguments=(
  "--maximum-relative-norm 0.003"
  "--maximum-relative-norm 0.004"
  "--maximum-relative-norm 0.005"
  "--maximum-relative-norm 0.006"
  "--maximum-relative-norm 0.004 --gm-scale 0.5"
  "--maximum-relative-norm 0.004 --target-margin-shift 4"
  "--maximum-relative-norm 0.005 --target-margin-shift 4"
  "--maximum-relative-norm 0.004 --minimum-target-probability 0.6"
  "--maximum-relative-norm 0.004 --gradient-block-normalization 0.25"
  "--maximum-relative-norm 0.005 --gradient-block-normalization 0.25"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/norm_cap_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/norm_cap_tuning/${name}"
  log="artifacts/logs/pplm_norm_cap_${name}.log"
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
