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
  --target-margin-shift 3
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
  --gm-scale 0.45
  --device cuda:0
)

names=(
  temp095 temp09 temp085 topk8 topk7 topk5
  temp095_topk8 temp09_topk8 gm0425 gm04 gamma025_gm045 gm0425_topk8
)
arguments=(
  "--temperature 0.95"
  "--temperature 0.9"
  "--temperature 0.85"
  "--top-k 8"
  "--top-k 7"
  "--top-k 5"
  "--temperature 0.95 --top-k 8"
  "--temperature 0.9 --top-k 8"
  "--gm-scale 0.425"
  "--gm-scale 0.4"
  "--gradient-block-normalization 0.25"
  "--gm-scale 0.425 --top-k 8"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/weighted_quality_tuning
for index in "${!names[@]}"; do
  gpu=$(((index % 2) * 2))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/weighted_quality_tuning/${name}"
  log="artifacts/logs/pplm_weighted_quality_${name}.log"
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
