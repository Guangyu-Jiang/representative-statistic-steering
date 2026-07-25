#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Fixed before generation and disjoint from every earlier PPLM split.
prefixes=(
  "The library" "The stadium" "The factory" "The island" "The bakery"
  "The hotel" "The park" "The radio" "The spacecraft" "The university"
  "The courtroom" "The laboratory" "The theater" "The warehouse" "The harbor"
  "The highway" "The farm" "The concert" "The conference" "The satellite"
)
root="artifacts/pplm_sentiment/persistent_adaptive_policy_validation_v3"
gpu=${PPLM_GPU:-2}
mkdir -p "$root" artifacts/logs

run_reference() {
  local target=$1
  local output_dir="$root/reference_$target"
  if [[ -f "$output_dir/generations.csv" ]]; then
    return
  fi
  rm -f "$output_dir/generations.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$output_dir" \
    --methods pplm \
    --targets "$target" \
    --prefixes "${prefixes[@]}" \
    --max-new-tokens 24 \
    --seeds 11 22 33 \
    --pplm-steps 10 \
    --pplm-step-size 0.04 \
    --maximum-relative-norm 0.10 \
    --gm-scale 0.95 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_adaptive_policy_v3_reference_${target}.log" 2>&1
}

run_candidate() {
  local target=$1
  local output_dir="$root/candidate_$target"
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
    --maximum-token-kl 1.0 \
    --difficult-margin-threshold -4.0 \
    --difficult-maximum-token-kl 2.0 \
    --statistic-mode margin \
    --target-probability 0.95 \
    --gradient-block-normalization 0.5 \
    --gm-scale 0.95 \
    --preserve-top-log-probs 2 \
    --log-probability-preservation-weight 0.01 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_adaptive_policy_v3_candidate_${target}.log" 2>&1
}

run_reference positive &
reference_positive_pid=$!
run_reference negative &
reference_negative_pid=$!
run_candidate positive &
candidate_positive_pid=$!
run_candidate negative &
candidate_negative_pid=$!
wait "$reference_positive_pid" "$reference_negative_pid" \
  "$candidate_positive_pid" "$candidate_negative_pid"

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "$root/reference_positive" "$root/reference_negative" \
  --output-dir "$root/reference_merged" \
  --expected-count 120
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "$root/candidate_positive" "$root/candidate_negative" \
  --output-dir "$root/candidate_merged" \
  --expected-count 120

CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
  --input "$root/reference_merged/generations.csv" \
  --output-dir "$root/reference_merged/external_eval" \
  --device cuda:0 &
reference_eval_pid=$!
CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
  --input "$root/candidate_merged/generations.csv" \
  --output-dir "$root/candidate_merged/external_eval" \
  --device cuda:0 &
candidate_eval_pid=$!
wait "$reference_eval_pid" "$candidate_eval_pid"
