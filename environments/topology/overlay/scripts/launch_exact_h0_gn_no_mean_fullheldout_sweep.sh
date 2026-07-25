#!/usr/bin/env bash
# Run the exact-H0 full-held-out grid without the zero-mean token constraint.
# Usage: scripts/launch_exact_h0_gn_no_mean_fullheldout_sweep.sh {llama|gemma|mistral} GPU_ID
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {llama|gemma|mistral} GPU_ID" >&2
  exit 2
fi

model_key="$1"
gpu_id="$2"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

case "$model_key" in
  llama)
    config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
    ;;
  gemma)
    config="configs/runs/gemma_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
    ;;
  mistral)
    config="configs/runs/mistral_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
    ;;
  *)
    echo "Unknown model: $model_key" >&2
    exit 2
    ;;
esac

common_args=(
  --config "$config"
  --datasets ambigqa situatedqa clamber
  --feature-root artifacts/steering_exact_h0_gn_features
  --artifact-root artifacts/steering_exact_h0_gauss_newton_fullheldout_no_mean_constraint
  --lambdas 0.1
  --dampings 0.01
  --trust-ratios 0.02 0.05 0.1
  --gn-steps 8
  --eval-n 0
  --optimization-jobs 8
  --feature-jobs 8
  --feature-batch-size 8
  --allow-mean-shift
)

CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src python scripts/run_exact_h0_gauss_newton_steering.py \
  "${common_args[@]}" \
  --target-mode nearest_abstention \
  --neighbor-ks 1 5 20 \
  --alphas 0.5 1 2

CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src python scripts/run_exact_h0_gauss_newton_steering.py \
  "${common_args[@]}" \
  --target-mode local_contrast \
  --neighbor-ks 5 20 \
  --alphas 1 2 4 8
