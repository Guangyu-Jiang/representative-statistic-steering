#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The lake"
  "The chicken"
  "The country"
  "The painting"
  "The movie"
  "The book"
  "The pizza"
  "The potato"
  "The city"
  "The president"
  "The company"
  "The game"
  "The restaurant"
  "The weather"
  "The conversation"
)
tags=(
  lake chicken country painting movie book pizza potato city president
  company game restaurant weather conversation
)

mkdir -p artifacts/pplm_sentiment/eval_main artifacts/logs
for index in "${!prefixes[@]}"; do
  gpu=$((index % 3))
  tag=${tags[$index]}
  output="artifacts/pplm_sentiment/eval_main/mnn_prob0p8_gm0p5_prefix_${tag}"
  log="artifacts/logs/pplm_eval_main_mnn_prob0p8_gm0p5_prefix_${tag}.log"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    python scripts/run_pplm_sentiment.py \
      --output-dir "$output" \
      --methods minimum_norm \
      --targets positive negative \
      --prefixes "${prefixes[$index]}" \
      --max-new-tokens 24 \
      --seeds 11 22 33 \
      --target-probability 0.8 \
      --ridge 0.1 \
      --minimum-norm-steps 3 \
      --maximum-relative-norm 0.03 \
      --gm-scale 0.5 \
      --device cuda:0 \
    > "$log" 2>&1 < /dev/null &
done

