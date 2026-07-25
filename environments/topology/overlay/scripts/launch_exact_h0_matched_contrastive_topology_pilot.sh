#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
root="artifacts/steering_exact_h0_matched_contrastive_topology_pilot"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
positive_instruction="Identify the specific missing detail or unclear interpretation, then ask a targeted clarification before answering."
negative_instruction="Assume a specific interpretation is intended, then provide one direct answer without asking for clarification."

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The two counterfactual instructions have the same LLaMA token count. This
# removes the dominant token-count shift observed with the original pair.
python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets ambigqa \
  --artifact-root "$root" \
  --behavior-label-source rule_high_precision \
  --positive-label-mode grounded \
  --topology-controller behavior_lowrank \
  --behavior-rank 1 \
  --direction-source self_contrastive_prompt_pair \
  --direction-readout last_token \
  --contrastive-positive-instruction "$positive_instruction" \
  --contrastive-negative-instruction "$negative_instruction" \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --retrieval-geometry standard \
  --neighbor-ks 20 \
  --target-mode self_contrastive_topology \
  --topology-alphas 0 0.5 1 2 \
  --shared-target-ratios 0.05 0.1 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 64 \
  --causal-position-beta 2 \
  --topology-decode-mode none \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
