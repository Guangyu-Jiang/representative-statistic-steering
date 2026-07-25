#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/steering_exact_h0_hybrid_grounded_alllayer_pilot}"
GPU_INDEX="${GPU_INDEX:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH=src

python scripts/run_exact_h0_hybrid_steering.py \
  --config configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml \
  --datasets ambigqa \
  --artifact-root "$ARTIFACT_ROOT" \
  --neighbor-ks 20 \
  --target-mode nearest_grounded \
  --topology-alphas 0 0.5 1 \
  --mean-alphas 0 2 4 6 8 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.05 \
  --gn-steps 8 \
  --optimization-jobs 8 \
  --eval-n 0 \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --batch-size 32
