#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="${ROOT:-artifacts/steering_exact_h0_self_contrastive_pilot}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-src}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$ROOT" \
  --batch-size 32

python scripts/report_exact_h0_perturbation_campaign.py
