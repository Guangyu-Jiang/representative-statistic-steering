#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
tags=(chicken country painting pizza potato president company game weather conversation)
root="artifacts/pplm_sentiment/hybrid_heldout"
mkdir -p "$root" artifacts/logs

shards=()
for index in "${!prefixes[@]}"; do
  output="$root/margin_floor05_${tags[$index]}"
  shards+=("$output")
  CUDA_VISIBLE_DEVICES=$((index % 3)) PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$output" --methods minimum_norm --targets positive negative \
    --prefixes "${prefixes[$index]}" --max-new-tokens 24 --seeds 11 22 33 \
    --minimum-norm-steps 3 --ridge 0.1 --maximum-relative-norm 0.03 \
    --statistic-mode margin --target-margin-shift 3 \
    --minimum-target-probability 0.5 --gm-scale 0.5 --device cuda:0 \
    > "artifacts/logs/pplm_hybrid_heldout_${tags[$index]}.log" 2>&1 &
done
wait

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${shards[@]}" --output-dir "$root/margin_floor05_merged" \
  --expected-count 60
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/margin_floor05_merged/generations.csv" \
  --output-dir "$root/margin_floor05_merged/external_eval" --device cuda:0
