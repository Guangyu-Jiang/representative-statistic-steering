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
  --maximum-relative-norm 0.03
  --target-margin-shift 3
  --minimum-target-probability 0.5
  --gm-scale 0.45
  --device cuda:0
)

names=(
  dist_s5_r01 dist_s8_r01 dist_s5_r001 dist_s8_r001 dist_s5_r0001 dist_s8_r0001
  margin_s5_r01 margin_s8_r01 margin_s5_r001 margin_s8_r001
  dist_s5_r001_gm035 dist_s5_r001_gm05
)
arguments=(
  "--statistic-mode distribution --minimum-norm-steps 5 --ridge 0.1"
  "--statistic-mode distribution --minimum-norm-steps 8 --ridge 0.1"
  "--statistic-mode distribution --minimum-norm-steps 5 --ridge 0.01"
  "--statistic-mode distribution --minimum-norm-steps 8 --ridge 0.01"
  "--statistic-mode distribution --minimum-norm-steps 5 --ridge 0.001"
  "--statistic-mode distribution --minimum-norm-steps 8 --ridge 0.001"
  "--statistic-mode margin --minimum-norm-steps 5 --ridge 0.1"
  "--statistic-mode margin --minimum-norm-steps 8 --ridge 0.1"
  "--statistic-mode margin --minimum-norm-steps 5 --ridge 0.01"
  "--statistic-mode margin --minimum-norm-steps 8 --ridge 0.01"
  "--statistic-mode distribution --minimum-norm-steps 5 --ridge 0.01 --gm-scale 0.35"
  "--statistic-mode distribution --minimum-norm-steps 5 --ridge 0.01 --gm-scale 0.5"
)

mkdir -p artifacts/logs artifacts/pplm_sentiment/solver_tuning
for index in "${!names[@]}"; do
  gpu=$((index % 3))
  name=${names[$index]}
  output="artifacts/pplm_sentiment/solver_tuning/${name}"
  log="artifacts/logs/pplm_solver_${name}.log"
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
