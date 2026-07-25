#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
root="artifacts/steering_exact_h0_combined_fullheldout_llama_ambigqa"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets ambigqa \
  --artifact-root "$root" \
  --behavior-label-source rule_high_precision \
  --positive-label-mode grounded \
  --topology-controller behavior_lowrank \
  --behavior-rank 4 \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --neighbor-ks 20 \
  --target-mode local_contrast \
  --topology-alphas 0 1 2 \
  --mean-alphas 6 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 0 \
  --causal-position-beta 2 \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
