#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

gpu="${TRUTHX_GPU:-2}"
root="artifacts/truthx_mc/corrected_accumulated_development"

run_setting() {
  local name=$1
  shift
  local output="${root}/${name}"
  if [[ -f "${output}/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
    --output-dir "${output}" \
    --method minimum_norm \
    --offset 0 --limit 64 \
    --target-mode cosine_margin_decoder \
    --ridge 0 --optimization-steps 10 --learning-rate 0.5 \
    --model-dtype bfloat16 --device cuda:0 \
    "$@"
}

run_setting decoder_t0p25_r0_d0p5_cap0p5 \
  --target-strength 0.25 --maximum-relative-norm 0.5
run_setting decoder_t0p25_r0_d0p5_cap1 \
  --target-strength 0.25 --maximum-relative-norm 1.0
run_setting decoder_t0p5_r0_d0p5_cap1 \
  --target-strength 0.5 --maximum-relative-norm 1.0
run_setting decoder_t0p5_r0_d0p5_cap2 \
  --target-strength 0.5 --maximum-relative-norm 2.0
