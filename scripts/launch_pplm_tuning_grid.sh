#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/logs artifacts/pplm_sentiment/tuning

probabilities=(0.6 0.7 0.8)
caps=(0.01 0.03 0.10)
gpu=0

for probability in "${probabilities[@]}"; do
  for cap in "${caps[@]}"; do
    probability_tag=${probability//./p}
    cap_tag=${cap//./p}
    name="mnn_prob${probability_tag}_cap${cap_tag}"
    output="artifacts/pplm_sentiment/tuning/${name}"
    log="artifacts/logs/pplm_${name}.log"
    setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
      python scripts/run_pplm_sentiment.py \
        --output-dir "$output" \
        --methods minimum_norm \
        --targets positive negative \
        --prefixes "The lake" "The movie" "The book" "The city" "The restaurant" \
        --max-new-tokens 16 \
        --seeds 11 22 \
        --target-probability "$probability" \
        --ridge 0.1 \
        --minimum-norm-steps 3 \
        --maximum-relative-norm "$cap" \
        --device cuda:0 \
      > "$log" 2>&1 < /dev/null &
    gpu=$(((gpu + 1) % 3))
  done
done

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python scripts/run_pplm_sentiment.py \
    --output-dir artifacts/pplm_sentiment/tuning/matched_baselines \
    --methods baseline pplm \
    --targets positive negative \
    --prefixes "The lake" "The movie" "The book" "The city" "The restaurant" \
    --max-new-tokens 16 \
    --seeds 11 22 \
    --pplm-steps 5 \
    --pplm-step-size 0.04 \
    --maximum-relative-norm 0.10 \
    --device cuda:0 \
  > artifacts/logs/pplm_matched_baselines.log 2>&1 < /dev/null &

