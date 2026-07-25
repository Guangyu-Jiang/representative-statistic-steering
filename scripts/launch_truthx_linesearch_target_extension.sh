#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
root="artifacts/truthx_mc/corrected_linesearch_target_extension"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

for cap in 4.0 6.0; do
  for target in 0.5 1.0 1.5; do
    cap_tag="${cap/./p}"
    target_tag="${target/./p}"
    output="${root}/decoder_t${target_tag}_r0_d1_cap${cap_tag}_bt4"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
      --output-dir "${output}" --method minimum_norm \
      --offset 0 --limit 32 \
      --target-mode cosine_margin_decoder \
      --target-strength "${target}" --ridge 0 \
      --optimization-steps 10 --learning-rate 1.0 \
      --maximum-relative-norm "${cap}" \
      --directional-backtracking-steps 4 \
      --model-dtype bfloat16 --device cuda:0
  done
done
