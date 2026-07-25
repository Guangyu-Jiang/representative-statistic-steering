#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
root="artifacts/truthx_mc/corrected_linesearch_development"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

run_setting() {
  local mode="$1"
  local target="$2"
  local cap="$3"
  local mode_tag="${mode//cosine_margin_/}"
  local target_tag="${target/./p}"
  local cap_tag="${cap/./p}"
  local output="${root}/${mode_tag}_t${target_tag}_r0_d1_cap${cap_tag}_bt4"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
    --output-dir "${output}" --method minimum_norm \
    --offset 0 --limit 32 \
    --target-mode "${mode}" --target-strength "${target}" --ridge 0 \
    --optimization-steps 10 --learning-rate 1.0 \
    --maximum-relative-norm "${cap}" \
    --directional-backtracking-steps 4 \
    --model-dtype bfloat16 --device cuda:0
}

run_setting cosine_margin_decoder 0.1 4.0
run_setting cosine_margin_decoder 0.25 4.0

for cap in 2.0 4.0; do
  for shift in 0.1 0.25 0.5; do
    run_setting cosine_margin_shift_decoder "${shift}" "${cap}"
  done
done
