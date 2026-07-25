#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="${ROOT:-artifacts/steering_exact_h0_last_token_readout_pilot}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/launch_exact_h0_last_token_readout_pilot}"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-src}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMON=(
  --config configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml
  --datasets ambigqa
  --artifact-root "$ROOT"
  --behavior-label-source rule_high_precision
  --positive-label-mode grounded
  --topology-controller behavior_lowrank
  --behavior-rank 4
  --direction-source observed_groups
  --direction-readout last_token
  --retrieval-feature-mode all_layer_plus_exact3
  --retrieval-geometry standard
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
  --shared-intervention-site layer_output
  --apply-on prompt_and_decode_shared
)

python scripts/run_exact_h0_hybrid_steering.py \
  "${COMMON[@]}" \
  --target-mode local_contrast

python scripts/run_exact_h0_hybrid_steering.py \
  "${COMMON[@]}" \
  --target-mode classifier_projection \
  --classifier-target-quantile 0.5

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$ROOT" \
  --batch-size 32

python scripts/report_exact_h0_perturbation_campaign.py
