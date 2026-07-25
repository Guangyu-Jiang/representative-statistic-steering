#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
wait_pid="${WAIT_PID:-}"
root="artifacts/truthx_mc/corrected_accumulated_strong_development"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

run_config() {
  local target="$1"
  local cap="$2"
  local target_tag="${target/./p}"
  local cap_tag="${cap/./p}"
  local output="${root}/decoder_t${target_tag}_r0_d1_cap${cap_tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
    --output-dir "${output}" --method minimum_norm \
    --offset 0 --limit 64 \
    --target-mode cosine_margin_decoder \
    --target-strength "${target}" --ridge 0 \
    --optimization-steps 10 --learning-rate 1.0 \
    --maximum-relative-norm "${cap}" \
    --model-dtype bfloat16 --device cuda:0
}

run_config 0.25 2.0
run_config 0.25 4.0
run_config 0.1 4.0
