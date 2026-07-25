#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

gpu=${TRUTHX_GPU:-2}
root="artifacts/truthx_mc/corrected_accumulated_development"
mkdir -p "$root" artifacts/logs

run_setting() {
  local name=$1
  shift
  local output="$root/$name"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/run_truthx_mc.py \
    --output-dir "$output" \
    --method minimum_norm \
    --offset 0 \
    --limit 64 \
    --optimization-steps 10 \
    --model-dtype bfloat16 \
    --device cuda:0 \
    "$@"
}

run_setting unrestricted_t0p25_r0p01_d0p5_cap0p1 \
  --target-mode cosine_margin --target-strength 0.25 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 0.1
run_setting unrestricted_t0p25_r0p01_d0p5_cap0p25 \
  --target-mode cosine_margin --target-strength 0.25 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 0.25
run_setting decoder_t0p25_r0p01_d0p5_cap0p5 \
  --target-mode cosine_margin_decoder --target-strength 0.25 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 0.5
run_setting decoder_t0p25_r0p01_d0p5_cap1 \
  --target-mode cosine_margin_decoder --target-strength 0.25 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 1.0
run_setting decoder_t0p25_r0p01_d0p5_cap2 \
  --target-mode cosine_margin_decoder --target-strength 0.25 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 2.0
run_setting decoder_t0p5_r0p01_d0p5_cap1 \
  --target-mode cosine_margin_decoder --target-strength 0.5 --ridge 0.01 \
  --learning-rate 0.5 --maximum-relative-norm 1.0
