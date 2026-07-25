#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
root="artifacts/truthx_mc/corrected_accumulated_strength_extension"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

for cap in 6.0 8.0; do
  for target in 0.1 0.25; do
    cap_tag="${cap/./p}"
    target_tag="${target/./p}"
    output="${root}/decoder_t${target_tag}_r0_d1_cap${cap_tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
      --output-dir "${output}" --method minimum_norm \
      --offset 0 --limit 64 \
      --target-mode cosine_margin_decoder \
      --target-strength "${target}" --ridge 0 \
      --optimization-steps 10 --learning-rate 1.0 \
      --maximum-relative-norm "${cap}" \
      --model-dtype bfloat16 --device cuda:0
  done
done
