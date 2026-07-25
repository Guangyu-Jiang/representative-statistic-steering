#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
tags=(chicken country painting pizza potato president company game weather conversation)
settings=(gm095_topk8 gm07_topk8 gm095_topk5)
arguments=("--gm-scale 0.95 --top-k 8" "--gm-scale 0.7 --top-k 8" "--gm-scale 0.95 --top-k 5")
root="artifacts/pplm_sentiment/pplm_baseline_quality_heldout_sharded"
mkdir -p "$root" artifacts/logs

for setting_index in "${!settings[@]}"; do
  setting=${settings[$setting_index]}
  read -r -a extra <<< "${arguments[$setting_index]}"
  shard_dirs=()
  for prefix_index in "${!prefixes[@]}"; do
    tag=${tags[$prefix_index]}
    output="$root/${setting}_${tag}"
    shard_dirs+=("$output")
    CUDA_VISIBLE_DEVICES="$setting_index" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" --methods pplm --targets negative \
      --prefixes "${prefixes[$prefix_index]}" --max-new-tokens 24 --seeds 11 22 33 \
      --pplm-steps 10 --pplm-step-size 0.04 --maximum-relative-norm 0.10 \
      "${extra[@]}" --device cuda:0 \
      > "artifacts/logs/pplm_baseline_shard_${setting}_${tag}.log" 2>&1 &
  done
  printf '%s\n' "${shard_dirs[@]}" > "$root/${setting}_shards.txt"
done
wait

for setting_index in "${!settings[@]}"; do
  setting=${settings[$setting_index]}
  mapfile -t shard_dirs < "$root/${setting}_shards.txt"
  PYTHONPATH=src python scripts/merge_pplm_shards.py \
    --shard-dir "${shard_dirs[@]}" --output-dir "$root/${setting}_negative_merged" \
    --expected-count 30
  CUDA_VISIBLE_DEVICES="$setting_index" python scripts/evaluate_pplm_sentiment.py \
    --input "$root/${setting}_negative_merged/generations.csv" \
    --output-dir "$root/${setting}_negative_merged/external_eval" --device cuda:0 &
done
wait
