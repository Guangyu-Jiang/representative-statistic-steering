#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefixes=("The lake" "The movie" "The book" "The city" "The restaurant")
common=(
  --methods minimum_norm
  --targets positive negative
  --prefixes "${prefixes[@]}"
  --max-new-tokens 24
  --seeds 11 22 33
  --minimum-norm-steps 3
  --ridge 0.1
  --maximum-relative-norm 0.03
  --statistic-mode margin
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
  --preserve-top-log-probs 1
  --log-probability-preservation-weight 0.001
  --persistent-cache
  --device cuda:0
)

names=(
  shift125_gm035 shift125_gm0375 shift125_gm04
  shift15_gm035 shift15_gm0375 shift15_gm04
  shift175_gm035 shift175_gm0375 shift175_gm04
)
shifts=(1.25 1.25 1.25 1.5 1.5 1.5 1.75 1.75 1.75)
mixes=(0.35 0.375 0.4 0.35 0.375 0.4 0.35 0.375 0.4)

mkdir -p artifacts/logs artifacts/pplm_sentiment/persistent_frontier_tuning
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=$((index % 3))
  output="artifacts/pplm_sentiment/persistent_frontier_tuning/$name"
  log="artifacts/logs/pplm_persistent_frontier_${name}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" \
      --target-margin-shift "${shifts[$index]}" \
      --gm-scale "${mixes[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
