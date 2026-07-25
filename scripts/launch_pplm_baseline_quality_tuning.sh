#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
common=(
  --methods pplm
  --targets positive negative
  --prefixes "${prefixes[@]}"
  --max-new-tokens 24
  --seeds 11 22 33
  --pplm-steps 10
  --pplm-step-size 0.04
  --maximum-relative-norm 0.10
  --device cuda:0
)

names=(gm05 gm07 gm08 gm09 gm07_topk5 gm07_topk8 gm095_temp095 gm095_topk5 gm095_topk8)
arguments=(
  "--gm-scale 0.5"
  "--gm-scale 0.7"
  "--gm-scale 0.8"
  "--gm-scale 0.9"
  "--gm-scale 0.7 --top-k 5"
  "--gm-scale 0.7 --top-k 8"
  "--gm-scale 0.95 --temperature 0.95"
  "--gm-scale 0.95 --top-k 5"
  "--gm-scale 0.95 --top-k 8"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/pplm_baseline_quality_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/pplm_baseline_quality_tuning/${name}"
  log="artifacts/logs/pplm_baseline_quality_${name}.log"
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
