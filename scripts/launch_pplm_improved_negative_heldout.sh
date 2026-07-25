#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
common=(
  --methods minimum_norm
  --targets negative
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
  --device cuda:0
)

preservation_output="artifacts/pplm_sentiment/output_preservation_heldout/top1_w0001_negative"
kl_output="artifacts/pplm_sentiment/weighted_kl_heldout/kl015_negative"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$preservation_output" "${common[@]}" \
  --gm-scale 0.45 \
  --preserve-top-log-probs 1 \
  --log-probability-preservation-weight 0.001 \
  > artifacts/logs/pplm_output_preservation_heldout_negative.log 2>&1 &
preservation_pid=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$kl_output" "${common[@]}" \
  --gm-scale 0.95 \
  --maximum-token-kl 0.15 \
  > artifacts/logs/pplm_weighted_kl_heldout_negative.log 2>&1 &
kl_pid=$!

wait "$preservation_pid" "$kl_pid"

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
  --input "$preservation_output/generations.csv" \
  --output-dir "$preservation_output/external_eval" --device cuda:0
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
  --input "$kl_output/generations.csv" \
  --output-dir "$kl_output/external_eval" --device cuda:0
