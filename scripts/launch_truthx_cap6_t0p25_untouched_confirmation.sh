#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pids="${WAIT_PIDS:-}"
output="artifacts/truthx_mc/corrected_accumulated_confirmation/decoder_t0p25_r0_d1_cap6p0_offset704_n113"

for wait_pid in ${wait_pids}; do
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
done

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
  --output-dir "${output}" --method minimum_norm \
  --offset 704 --limit 113 \
  --target-mode cosine_margin_decoder \
  --target-strength 0.25 --ridge 0 \
  --optimization-steps 10 --learning-rate 1.0 \
  --maximum-relative-norm 6.0 \
  --model-dtype bfloat16 --device cuda:0
