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
  top1_w0001 top1_w0003 top1_w001 top1_w003
  top2_w0001 top2_w0003 top2_w001 top2_w003
  top4_w0003 top4_w001
)
top_counts=(1 1 1 1 2 2 2 2 4 4)
weights=(0.001 0.003 0.01 0.03 0.001 0.003 0.01 0.03 0.003 0.01)

mkdir -p artifacts/logs artifacts/pplm_sentiment/output_preservation_low_weight
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=$((index % 3))
  output="artifacts/pplm_sentiment/output_preservation_low_weight/$name"
  log="artifacts/logs/pplm_output_preservation_low_${name}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" \
      --preserve-top-log-probs "${top_counts[$index]}" \
      --log-probability-preservation-weight "${weights[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
