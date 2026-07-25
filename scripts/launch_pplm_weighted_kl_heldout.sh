#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
output="artifacts/pplm_sentiment/weighted_kl_heldout/kl015"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir "$output" \
  --methods minimum_norm \
  --targets positive negative \
  --prefixes "${prefixes[@]}" \
  --max-new-tokens 24 \
  --seeds 11 22 33 \
  --minimum-norm-steps 3 \
  --ridge 0.1 \
  --maximum-relative-norm 0.03 \
  --statistic-mode margin \
  --target-margin-shift 3 \
  --minimum-target-probability 0.5 \
  --gradient-block-normalization 0.5 \
  --gm-scale 0.95 \
  --maximum-token-kl 0.15 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$output/generations.csv" \
  --output-dir "$output/external_eval" --device cuda:0
