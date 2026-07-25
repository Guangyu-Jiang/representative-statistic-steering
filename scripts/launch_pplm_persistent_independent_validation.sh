#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=(
  "The computer" "The ocean" "The school" "The house" "The phone"
  "The garden" "The hospital" "The train" "The music" "The meeting"
)
tags=(computer ocean school house phone garden hospital train music meeting)
root="artifacts/pplm_sentiment/persistent_independent_validation"
mkdir -p "$root" artifacts/logs

reference_dirs=()
positive_dirs=()
negative_dirs=()
for index in "${!prefixes[@]}"; do
  prefix=${prefixes[$index]}
  tag=${tags[$index]}
  reference="$root/reference_$tag"
  positive="$root/candidate_positive_$tag"
  negative="$root/candidate_negative_$tag"
  reference_dirs+=("$reference")
  positive_dirs+=("$positive")
  negative_dirs+=("$negative")

  CUDA_VISIBLE_DEVICES="$((index % 3))" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$reference" \
    --methods pplm \
    --targets positive negative \
    --prefixes "$prefix" \
    --max-new-tokens 24 \
    --seeds 11 22 33 \
    --pplm-steps 10 \
    --pplm-step-size 0.04 \
    --maximum-relative-norm 0.10 \
    --gm-scale 0.95 \
    --persistent-cache \
    --device cuda:0 \
    > "artifacts/logs/pplm_independent_reference_${tag}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES="$(((index + 1) % 3))" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$positive" \
    --methods minimum_norm \
    --targets positive \
    --prefixes "$prefix" \
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
    > "artifacts/logs/pplm_independent_candidate_positive_${tag}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES="$(((index + 2) % 3))" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
    --output-dir "$negative" \
    --methods minimum_norm \
    --targets negative \
    --prefixes "$prefix" \
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
    > "artifacts/logs/pplm_independent_candidate_negative_${tag}.log" 2>&1 &
done
wait

PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${reference_dirs[@]}" \
  --output-dir "$root/reference_merged" \
  --expected-count 60
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${positive_dirs[@]}" \
  --output-dir "$root/candidate_positive_merged" \
  --expected-count 30
PYTHONPATH=src python scripts/merge_pplm_shards.py \
  --shard-dir "${negative_dirs[@]}" \
  --output-dir "$root/candidate_negative_merged" \
  --expected-count 30

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/reference_merged/generations.csv" \
  --output-dir "$root/reference_merged/external_eval" --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/candidate_positive_merged/generations.csv" \
  --output-dir "$root/candidate_positive_merged/external_eval" --device cuda:0 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_pplm_sentiment.py \
  --input "$root/candidate_negative_merged/generations.csv" \
  --output-dir "$root/candidate_negative_merged/external_eval" --device cuda:0 &
wait
