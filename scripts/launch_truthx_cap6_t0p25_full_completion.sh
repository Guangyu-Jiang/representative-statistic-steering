#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
output="artifacts/truthx_mc/corrected_accumulated_full_completion/decoder_t0p25_r0_d1_cap6p0_offset64_n640"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
  --output-dir "${output}" --method minimum_norm \
  --offset 64 --limit 640 \
  --target-mode cosine_margin_decoder \
  --target-strength 0.25 --ridge 0 \
  --optimization-steps 10 --learning-rate 1.0 \
  --maximum-relative-norm 6.0 \
  --model-dtype bfloat16 --device cuda:0
