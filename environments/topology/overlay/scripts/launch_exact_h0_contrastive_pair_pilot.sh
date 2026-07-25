#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
root="artifacts/steering_exact_h0_contrastive_pair_pilot"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

common=(
  --config "$config"
  --datasets ambigqa
  --artifact-root "$root"
  --behavior-label-source rule_high_precision
  --positive-label-mode grounded
  --direction-source contrastive_prompt_pairs
  --topology-controller behavior_lowrank
  --behavior-rank 4
  --retrieval-feature-mode all_layer_plus_exact3
  --retrieval-geometry class_residual
  --neighbor-ks 20
  --topology-alphas 0 1 2
  --shared-target-ratios 0.05 0.1 0.2 0.3
  --lambdas 0.1
  --dampings 0.01
  --trust-ratios 0.1
  --gn-steps 12
  --optimization-jobs 8
  --eval-n 64
  --causal-position-beta 2
  --topology-decode-mode none
  --shared-intervention-site layer_output
  --apply-on prompt_and_decode_shared
)

python scripts/run_exact_h0_hybrid_steering.py \
  "${common[@]}" \
  --target-mode local_contrast

python scripts/run_exact_h0_hybrid_steering.py \
  "${common[@]}" \
  --target-mode classifier_projection \
  --classifier-target-quantile 0.5
