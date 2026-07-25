#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
output="artifacts/truthx_mc/corrected_accumulated_validation/decoder_t0p1_r0_d1_cap4p0_offset448_n256"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
  --output-dir "${output}" --method minimum_norm \
  --offset 448 --limit 256 \
  --target-mode cosine_margin_decoder \
  --target-strength 0.1 --ridge 0 \
  --optimization-steps 10 --learning-rate 1.0 \
  --maximum-relative-norm 4.0 \
  --model-dtype bfloat16 --device cuda:0
