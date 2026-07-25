#!/usr/bin/env bash
set -euo pipefail

gpu="${EXPERIMENT_GPU:-2}"
wait_pids="${WAIT_PIDS:-}"

for wait_pid in ${wait_pids}; do
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
done

PYTHONPATH=src python scripts/merge_truthx_shards.py \
  --shard-dir \
    artifacts/truthx_mc/corrected_accumulated_strength_extension/decoder_t0p25_r0_d1_cap6p0 \
    artifacts/truthx_mc/corrected_accumulated_full_completion/decoder_t0p25_r0_d1_cap6p0_offset64_n640 \
    artifacts/truthx_mc/corrected_accumulated_confirmation/decoder_t0p25_r0_d1_cap6p0_offset704_n113 \
  --output-dir artifacts/truthx_mc/corrected_accumulated_full/decoder_t0p25_r0_d1_cap6p0_n817 \
  --expected-count 817

LOOKBACK_GPU="${gpu}" bash scripts/launch_lookback_baseline_rerank_untouched_validation.sh
LOOKBACK_GPU="${gpu}" bash scripts/launch_lookback_minimum_norm_rerank_development_diagnostics.sh
TRUTHX_GPU="${gpu}" bash scripts/launch_truthx_published_gate_development.sh
