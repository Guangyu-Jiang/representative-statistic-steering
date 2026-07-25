#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
root="artifacts/truthx_mc/corrected_direction_constrained_development"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

for backtracking in 0 4; do
  for target in 0.25 0.5 1.0; do
    target_tag="${target/./p}"
    output="${root}/decoder_t${target_tag}_r0_d1_cap4p0_bt${backtracking}_ray"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
      --output-dir "${output}" --method minimum_norm \
      --offset 0 --limit 32 \
      --target-mode cosine_margin_decoder \
      --target-strength "${target}" --ridge 0 \
      --optimization-steps 10 --learning-rate 1.0 \
      --maximum-relative-norm 4.0 \
      --directional-backtracking-steps "${backtracking}" \
      --directional-nonnegative \
      --model-dtype bfloat16 --device cuda:0
  done
done
