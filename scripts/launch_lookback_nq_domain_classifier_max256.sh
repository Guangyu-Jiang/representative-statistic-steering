#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
baseline="artifacts/lookback_nq/development_n60_max256/baseline_greedy"
features="artifacts/lookback_nq/nq_replay_features_max256"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${baseline}" \
  --methods baseline --offset 0 --limit 60 --max-new-tokens 256

CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src python scripts/extract_lookback_replay_features.py \
  --results \
    "${baseline}/results.jsonl" \
    artifacts/lookback_nq/heldout_offset60_n100_max256/baseline_greedy/results.jsonl \
  --output-dir "${features}" --device cuda:0

python scripts/train_nq_lookback_classifier.py \
  --feature-dir "${features}" \
  --output checkpoints/lookback_nq_domain_classifier_max256.pkl
