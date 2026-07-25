#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
target_probabilities=(0.90 0.95 0.98 0.99)
mix_scales=(0.30 0.35 0.40 0.45)
root="artifacts/pplm_sentiment/persistent_absolute_target_tuning"
mkdir -p "$root" artifacts/logs
gpu=${PPLM_GPU:-2}
maximum_jobs=${PPLM_MAX_JOBS:-5}

run_dirs=()
job_index=0
active_jobs=0
for target_probability in "${target_probabilities[@]}"; do
  probability_tag=${target_probability/./p}
  for mix_scale in "${mix_scales[@]}"; do
    mix_tag=${mix_scale/./p}
    tag="prob${probability_tag}_gm${mix_tag}"
    output_dir="$root/$tag"
    run_dirs+=("$output_dir")
    if [[ -f "$output_dir/generations.csv" ]]; then
      job_index=$((job_index + 1))
      continue
    fi
    rm -f "$output_dir/generations.jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
      python scripts/run_pplm_sentiment.py \
      --output-dir "$output_dir" \
      --methods minimum_norm \
      --targets positive negative \
      --prefixes "${prefixes[@]}" \
      --max-new-tokens 24 \
      --seeds 11 22 33 \
      --minimum-norm-steps 3 \
      --ridge 0.1 \
      --maximum-relative-norm 0.03 \
      --statistic-mode margin \
      --target-probability "$target_probability" \
      --gradient-block-normalization 0.5 \
      --gm-scale "$mix_scale" \
      --preserve-top-log-probs 2 \
      --log-probability-preservation-weight 0.01 \
      --persistent-cache \
      --device cuda:0 \
      > "artifacts/logs/pplm_absolute_target_${tag}.log" 2>&1 &
    active_jobs=$((active_jobs + 1))
    if ((active_jobs >= maximum_jobs)); then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
    job_index=$((job_index + 1))
  done
done
wait

active_jobs=0
for index in "${!run_dirs[@]}"; do
  output_dir=${run_dirs[$index]}
  if [[ -f "$output_dir/external_eval/summary.csv" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/evaluate_pplm_sentiment.py \
    --input "$output_dir/generations.csv" \
    --output-dir "$output_dir/external_eval" \
    --device cuda:0 \
    > "artifacts/logs/pplm_absolute_target_eval_$(basename "$output_dir").log" 2>&1 &
  active_jobs=$((active_jobs + 1))
  if ((active_jobs >= maximum_jobs)); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done
wait
