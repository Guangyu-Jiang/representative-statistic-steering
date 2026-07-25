#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The river" "The mountain" "The doctor" "The teacher" "The airport"
  "The bicycle" "The window" "The newspaper" "The camera" "The market"
  "The museum" "The bridge" "The forest" "The clock" "The letter"
  "The kitchen" "The office" "The festival" "The airplane" "The village"
)
root="artifacts/pplm_sentiment/persistent_absolute_target_validation_v2"
gpu=${PPLM_GPU:-2}
mkdir -p "$root" artifacts/logs

run_target() {
  local target=$1
  local output_dir="$root/$target"
  if [[ -f "$output_dir/generations.csv" ]]; then
    return
  fi
  rm -f "$output_dir/generations.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$output_dir" \
    --methods minimum_norm \
    --targets "$target" \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 \
    --seeds 11 22 33 \
    --minimum-norm-steps 3 \
    --ridge 0.1 \
    --maximum-relative-norm 0.03 \
    --statistic-mode margin \
    --target-probability 0.95 \
    --gradient-block-normalization 0.5 \
    --gm-scale 0.40 \
    --preserve-top-log-probs 2 \
    --log-probability-preservation-weight 0.01 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_absolute_target_validation_v2_${target}.log" 2>&1
}

run_target positive &
positive_pid=$!
run_target negative &
negative_pid=$!
wait "$positive_pid" "$negative_pid"

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "$root/positive" "$root/negative" \
  --output-dir "$root/merged" \
  --expected-count 120

CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
  --input "$root/merged/generations.csv" \
  --output-dir "$root/merged/external_eval" \
  --device cuda:0
