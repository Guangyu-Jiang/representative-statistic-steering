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
kl_caps=(1.0 1.5 2.0)
root="artifacts/pplm_sentiment/persistent_adaptive_kl_tuning_v2"
gpu=${PPLM_GPU:-2}
mkdir -p "$root" artifacts/logs

run_pids=()
run_dirs=()
for kl_cap in "${kl_caps[@]}"; do
  tag="kl${kl_cap/./p}"
  output_dir="$root/$tag"
  run_dirs+=("$output_dir")
  if [[ -f "$output_dir/generations.csv" ]]; then
    continue
  fi
  rm -f "$output_dir/generations.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$output_dir" \
    --methods minimum_norm \
    --targets positive negative \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 \
    --seeds 11 22 33 \
    --minimum-norm-steps 3 \
    --ridge 0.1 \
    --maximum-relative-norm 0.03 \
    --maximum-token-kl "$kl_cap" \
    --statistic-mode margin \
    --target-probability 0.95 \
    --gradient-block-normalization 0.5 \
    --gm-scale 0.95 \
    --preserve-top-log-probs 2 \
    --log-probability-preservation-weight 0.01 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_adaptive_kl_v2_${tag}.log" 2>&1 &
  run_pids+=("$!")
done
if ((${#run_pids[@]} > 0)); then
  wait "${run_pids[@]}"
fi

eval_pids=()
for output_dir in "${run_dirs[@]}"; do
  if [[ -f "$output_dir/external_eval/summary.csv" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
    --input "$output_dir/generations.csv" \
    --output-dir "$output_dir/external_eval" \
    --device cuda:0 \
    > "artifacts/logs/pplm_adaptive_kl_v2_eval_$(basename "$output_dir").log" 2>&1 &
  eval_pids+=("$!")
done
if ((${#eval_pids[@]} > 0)); then
  wait "${eval_pids[@]}"
fi
