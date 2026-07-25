#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Fixed before generation and disjoint from tuning, held-out, and validation-v1.
prefixes=(
  "The river" "The mountain" "The doctor" "The teacher" "The airport"
  "The bicycle" "The window" "The newspaper" "The camera" "The market"
  "The museum" "The bridge" "The forest" "The clock" "The letter"
  "The kitchen" "The office" "The festival" "The airplane" "The village"
)
root="artifacts/pplm_sentiment/persistent_independent_validation_v2"
mkdir -p "$root" artifacts/logs

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$root/reference_positive" \
  --methods pplm \
  --targets positive \
  --prefixes "${prefixes[@]}" \
  --max-new-tokens 24 \
  --seeds 11 22 33 \
  --pplm-steps 10 \
  --pplm-step-size 0.04 \
  --maximum-relative-norm 0.10 \
  --gm-scale 0.95 \
  --persistent-cache \
  --device cuda:0 \
  > artifacts/logs/pplm_independent_v2_reference_positive.log 2>&1 &
reference_positive_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$root/reference_negative" \
  --methods pplm \
  --targets negative \
  --prefixes "${prefixes[@]}" \
  --max-new-tokens 24 \
  --seeds 11 22 33 \
  --pplm-steps 10 \
  --pplm-step-size 0.04 \
  --maximum-relative-norm 0.10 \
  --gm-scale 0.95 \
  --persistent-cache \
  --device cuda:0 \
  > artifacts/logs/pplm_independent_v2_reference_negative.log 2>&1 &
reference_negative_pid=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$root/candidate_positive" \
  --methods minimum_norm \
  --targets positive \
  --prefixes "${prefixes[@]}" \
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
  > artifacts/logs/pplm_independent_v2_candidate_positive.log 2>&1 &
candidate_positive_pid=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$root/candidate_negative" \
  --methods minimum_norm \
  --targets negative \
  --prefixes "${prefixes[@]}" \
  --max-new-tokens 24 \
  --seeds 11 22 33 \
  --minimum-norm-steps 3 \
  --ridge 0.1 \
  --maximum-relative-norm 0.03 \
  --statistic-mode margin \
  --target-margin-shift 1.5 \
  --minimum-target-probability 0.5 \
  --gradient-block-normalization 0.5 \
  --gm-scale 0.45 \
  --preserve-top-log-probs 2 \
  --log-probability-preservation-weight 0.01 \
  --persistent-cache \
  --device cuda:0 \
  > artifacts/logs/pplm_independent_v2_candidate_negative.log 2>&1 &
candidate_negative_pid=$!

wait "$reference_positive_pid" "$reference_negative_pid" \
  "$candidate_positive_pid" "$candidate_negative_pid"

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "$root/reference_positive" "$root/reference_negative" \
  --output-dir "$root/reference_merged" \
  --expected-count 120

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/reference_merged/generations.csv" \
  --output-dir "$root/reference_merged/external_eval" \
  --device cuda:0 &
reference_eval_pid=$!
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/candidate_positive/generations.csv" \
  --output-dir "$root/candidate_positive/external_eval" \
  --device cuda:0 &
candidate_positive_eval_pid=$!
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/candidate_negative/generations.csv" \
  --output-dir "$root/candidate_negative/external_eval" \
  --device cuda:0 &
candidate_negative_eval_pid=$!
wait "$reference_eval_pid" "$candidate_positive_eval_pid" \
  "$candidate_negative_eval_pid"
