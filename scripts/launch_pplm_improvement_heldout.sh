#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The chicken" "The country" "The painting" "The pizza" "The potato"
  "The president" "The company" "The game" "The weather" "The conversation"
)
tags=(chicken country painting pizza potato president company game weather conversation)
root="artifacts/pplm_sentiment/improvement_heldout"
mkdir -p "$root" artifacts/logs

unified_dirs=()
negative_dirs=()
positive_dirs=()
for index in "${!prefixes[@]}"; do
  prefix=${prefixes[$index]}
  tag=${tags[$index]}
  unified="$root/unified_${tag}"
  negative="$root/negative_topk5_${tag}"
  positive="$root/positive_gm055_${tag}"
  unified_dirs+=("$unified")
  negative_dirs+=("$negative")
  positive_dirs+=("$positive")

  CUDA_VISIBLE_DEVICES=$((index % 3)) PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$unified" --methods minimum_norm --targets positive negative \
    --prefixes "$prefix" --max-new-tokens 24 --seeds 11 22 33 \
    --minimum-norm-steps 3 --ridge 0.1 --maximum-relative-norm 0.03 \
    --statistic-mode distribution --target-margin-shift 3 --gm-scale 0.5 \
    --device cuda:0 > "artifacts/logs/pplm_heldout_unified_${tag}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=$(((index + 1) % 3)) PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$negative" --methods minimum_norm --targets negative \
    --prefixes "$prefix" --max-new-tokens 24 --seeds 11 22 33 \
    --minimum-norm-steps 3 --ridge 0.1 --maximum-relative-norm 0.03 \
    --statistic-mode distribution --target-margin-shift 3 --gm-scale 0.5 --top-k 5 \
    --device cuda:0 > "artifacts/logs/pplm_heldout_negative_${tag}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=$(((index + 2) % 3)) PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$positive" --methods minimum_norm --targets positive \
    --prefixes "$prefix" --max-new-tokens 24 --seeds 11 22 33 \
    --minimum-norm-steps 3 --ridge 0.1 --maximum-relative-norm 0.03 \
    --statistic-mode distribution --target-margin-shift 3 --gm-scale 0.55 \
    --device cuda:0 > "artifacts/logs/pplm_heldout_positive_${tag}.log" 2>&1 &
done
wait

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${unified_dirs[@]}" --output-dir "$root/unified_merged" --expected-count 60
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${negative_dirs[@]}" --output-dir "$root/negative_topk5_merged" --expected-count 30
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${positive_dirs[@]}" --output-dir "$root/positive_gm055_merged" --expected-count 30

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/unified_merged/generations.csv" \
  --output-dir "$root/unified_merged/external_eval" --device cuda:0 \
  > artifacts/logs/pplm_heldout_eval_unified.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/negative_topk5_merged/generations.csv" \
  --output-dir "$root/negative_topk5_merged/external_eval" --device cuda:0 \
  > artifacts/logs/pplm_heldout_eval_negative.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/positive_gm055_merged/generations.csv" \
  --output-dir "$root/positive_gm055_merged/external_eval" --device cuda:0 \
  > artifacts/logs/pplm_heldout_eval_positive.log 2>&1 &
wait
