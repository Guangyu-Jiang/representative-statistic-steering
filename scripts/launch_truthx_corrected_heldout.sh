#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
output="artifacts/truthx_mc/corrected_accumulated_heldout/decoder_t0p1_r0p01_d0p5_cap0p5_offset64_n128"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
  --output-dir "${output}" \
  --method minimum_norm \
  --offset 64 --limit 128 \
  --target-mode cosine_margin_decoder \
  --target-strength 0.1 \
  --ridge 0.01 \
  --optimization-steps 10 \
  --learning-rate 0.5 \
  --maximum-relative-norm 0.5 \
  --model-dtype bfloat16 \
  --device cuda:0
