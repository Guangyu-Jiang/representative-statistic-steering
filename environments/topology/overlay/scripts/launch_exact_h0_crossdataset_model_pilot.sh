#!/usr/bin/env bash
set -euo pipefail

model_key="${1:?usage: $0 MODEL_KEY GPU}"
gpu="${2:?usage: $0 MODEL_KEY GPU}"

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
    printf 'Unknown model key: %s\n' "$model_key" >&2
    exit 2
    ;;
esac

root="artifacts/steering_exact_h0_crossdataset_rule_pilot_${model_key}"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets ambigqa situatedqa \
  --artifact-root "$root" \
  --behavior-label-source rule_high_precision \
  --positive-label-mode grounded \
  --topology-controller behavior_lowrank \
  --behavior-rank 4 \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --retrieval-geometry class_residual \
  --neighbor-ks 20 \
  --target-mode local_contrast \
  --topology-alphas 0 2 \
  --mean-alphas 4 6 8 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 64 \
  --causal-position-beta 2 \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets clamber \
  --artifact-root "$root" \
  --behavior-label-source rule_high_precision \
  --positive-label-mode grounded \
  --topology-controller behavior_lowrank \
  --behavior-rank 4 \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --retrieval-geometry class_residual \
  --neighbor-ks 20 \
  --target-mode local_contrast \
  --topology-alphas 0 2 \
  --mean-alphas 8 12 16 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 64 \
  --causal-position-beta 2 \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
