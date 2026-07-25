#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
tags=(chicken country painting pizza potato president company game weather conversation)
root="artifacts/pplm_sentiment/persistent_candidate_v2_heldout"
mkdir -p "$root" artifacts/logs

shard_dirs=()
for index in "${!prefixes[@]}"; do
  output="$root/${tags[$index]}"
  shard_dirs+=("$output")
  gpu=$((index % 3))
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$output" \
    --methods minimum_norm \
    --targets positive negative \
    --prefixes "${prefixes[$index]}" \
    --max-new-tokens 24 \
    --seeds 11 22 33 \
    --minimum-norm-steps 3 \
    --ridge 0.1 \
    --maximum-relative-norm 0.03 \
    --statistic-mode margin \
    --target-margin-shift 1.5 \
    --minimum-target-probability 0.5 \
    --gradient-block-normalization 0.5 \
    --gm-scale 0.35 \
    --preserve-top-log-probs 2 \
    --log-probability-preservation-weight 0.01 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_persistent_candidate_v2_${tags[$index]}.log" 2>&1 &
done
wait

merged="$root/merged"
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${shard_dirs[@]}" \
  --output-dir "$merged" \
  --expected-count 60
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$merged/generations.csv" \
  --output-dir "$merged/external_eval" --device cuda:0
