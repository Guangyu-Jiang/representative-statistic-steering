#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-2}"
root="artifacts/steering_exact_h0_lowrank_causal_anchor_grounded_pilot"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

for anchor_ratio in 0.01 0.025 0.05; do
  python scripts/run_exact_h0_hybrid_steering.py \
    --config "$config" \
    --datasets ambigqa \
    --artifact-root "$root" \
    --positive-label-mode grounded \
    --topology-controller behavior_lowrank \
    --behavior-rank 4 \
    --neighbor-ks 20 \
    --target-mode local_contrast \
    --topology-alphas 0.5 1 2 4 \
    --mean-alphas 0 \
    --lambdas 0.1 \
    --dampings 0.01 \
    --trust-ratios 0.1 \
    --gn-steps 12 \
    --optimization-jobs 8 \
    --eval-n 64 \
    --causal-anchor-ratio "$anchor_ratio" \
    --causal-anchor-max-error-increase 0.1 \
    --shared-intervention-site layer_output \
    --apply-on prompt_and_decode_shared
done

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
