#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pid="${WAIT_PID:-}"

if [[ -n "${wait_pid}" ]]; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src python scripts/extract_lookback_replay_features.py --device cuda:0

python scripts/train_nq_lookback_classifier.py
