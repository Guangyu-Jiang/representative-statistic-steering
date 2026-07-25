#!/usr/bin/env bash
# Run additional exact-H0 full-test sweeps outside the primary development grid.
# Usage: scripts/launch_exact_h0_gn_expanded_fullheldout_sweep.sh {llama|gemma|mistral} GPU_ID {regularization|nearest_regularization|global}
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {llama|gemma|mistral} GPU_ID {regularization|nearest_regularization|global}" >&2
  exit 2
fi

model_key="$1"
gpu_id="$2"
sweep_kind="$3"
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
  --artifact-root artifacts/steering_exact_h0_gauss_newton_fullheldout_expanded
  --gn-steps 8
  --eval-n 0
  --optimization-jobs 8
  --feature-jobs 8
  --feature-batch-size 8
)

case "$sweep_kind" in
  regularization)
    # Explore the two regularization components while retaining the strongest local target family.
    CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src python scripts/run_exact_h0_gauss_newton_steering.py \
      "${common_args[@]}" \
      --target-mode local_contrast \
      --neighbor-ks 5 20 \
      --alphas 2 4 8 \
      --lambdas 0.01 1 \
      --dampings 0.001 0.1 \
      --trust-ratios 0.05 0.1
    ;;
  nearest_regularization)
    # Mirror the local regularization sweep for the direct nearest-D+ topology target.
    CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src python scripts/run_exact_h0_gauss_newton_steering.py \
      "${common_args[@]}" \
      --target-mode nearest_abstention \
      --neighbor-ks 1 5 20 \
      --alphas 0.5 1 2 \
      --lambdas 0.01 1 \
      --dampings 0.001 0.1 \
      --trust-ratios 0.05 0.1
    ;;
  global)
    # k is immaterial for the global target; k=1 avoids duplicated runs.
    CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src python scripts/run_exact_h0_gauss_newton_steering.py \
      "${common_args[@]}" \
      --target-mode global_contrast \
      --neighbor-ks 1 \
      --alphas 0.5 1 2 4 8 \
      --lambdas 0.01 0.1 1 \
      --dampings 0.01 \
      --trust-ratios 0.02 0.05 0.1
    ;;
  *)
    echo "Unknown sweep kind: $sweep_kind" >&2
    exit 2
    ;;
esac
