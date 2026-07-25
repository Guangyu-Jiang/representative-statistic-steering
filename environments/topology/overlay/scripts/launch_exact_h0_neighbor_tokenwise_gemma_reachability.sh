#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
root="artifacts/steering_exact_h0_neighbor_tokenwise_gemma_reachability"
config="configs/runs/gemma_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets ambigqa situatedqa \
  --artifact-root "$root" \
  --behavior-label-source rule_targeted_clarification \
  --positive-label-mode grounded \
  --topology-controller behavior_tokenwise \
  --direction-source observed_neighbor_tokenwise \
  --direction-readout mean_pool \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --retrieval-geometry class_residual \
  --neighbor-ks 20 \
  --target-mode classifier_projection \
  --classifier-target-quantile 0.25 \
  --topology-alphas 0 1 2 \
  --shared-target-ratios 0.4 \
  --lambdas 0.01 \
  --dampings 0.001 \
  --trust-ratios 0.2 \
  --gn-steps 20 \
  --optimization-jobs 8 \
  --eval-n 128 \
  --topology-decode-mode none \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 24
