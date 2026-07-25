#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
root="${LOOKBACK_OUTPUT_ROOT:-artifacts/lookback_nq/development_n60_retrieval}"

for mode in retrieved_passage retrieved_sentence; do
  for spec in shift4_cap0.65 shift6_cap0.5; do
    if [[ "${spec}" == "shift4_cap0.65" ]]; then
      shift=4
      cap=0.65
    else
      shift=6
      cap=0.5
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
      --output-dir "${root}/${mode}_${spec}" \
      --methods minimum_norm \
      --offset 0 \
      --limit 60 \
      --max-new-tokens 64 \
      --target-mode relative \
      --target-logit-shift "${shift}" \
      --ridge 0 \
      --solver-steps 16 \
      --solver-damping 0.5 \
      --maximum-bias-rms "${cap}" \
      --context-bias-mode "${mode}" \
      --context-overlap-radius 2
  done
done
