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
  output="artifacts/pplm_sentiment/eval_main/pplm_steps10_prefix_${tag}"
  log="artifacts/logs/pplm_eval_main_pplm_steps10_prefix_${tag}.log"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    python scripts/run_pplm_sentiment.py \
      --output-dir "$output" \
      --methods pplm \
      --targets positive negative \
      --prefixes "${prefixes[$index]}" \
      --max-new-tokens 24 \
      --seeds 11 22 33 \
      --pplm-steps 10 \
      --pplm-step-size 0.04 \
      --maximum-relative-norm 0.10 \
      --device cuda:0 \
    > "$log" 2>&1 < /dev/null &
done

