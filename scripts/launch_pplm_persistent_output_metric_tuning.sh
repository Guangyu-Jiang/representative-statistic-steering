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
  --target-margin-shift 1.5
  --minimum-target-probability 0.5
  --gradient-block-normalization 0.5
  --persistent-cache
  --device cuda:0
)

names=(
  gm035_top1_w001 gm035_top1_w01
  gm035_top2_w0001 gm035_top2_w001 gm035_top2_w01
  gm035_top4_w0001 gm035_top4_w001 gm035_top4_w01
  gm04_top2_w001 gm04_top2_w01
  gm04_top4_w001 gm04_top4_w01
)
mixes=(0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.4 0.4 0.4 0.4)
top_counts=(1 1 2 2 2 4 4 4 2 2 4 4)
weights=(0.01 0.1 0.001 0.01 0.1 0.001 0.01 0.1 0.01 0.1 0.01 0.1)

mkdir -p artifacts/logs artifacts/pplm_sentiment/persistent_output_metric_tuning
for index in "${!names[@]}"; do
  name=${names[$index]}
  gpu=$((index % 3))
  output="artifacts/pplm_sentiment/persistent_output_metric_tuning/$name"
  log="artifacts/logs/pplm_persistent_output_metric_${name}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_pplm_sentiment.py \
      --output-dir "$output" "${common[@]}" \
      --gm-scale "${mixes[$index]}" \
      --preserve-top-log-probs "${top_counts[$index]}" \
      --log-probability-preservation-weight "${weights[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_pplm_sentiment.py \
      --input "$output/generations.csv" \
      --output-dir "$output/external_eval" --device cuda:0
  ) > "$log" 2>&1 < /dev/null &
done
wait
