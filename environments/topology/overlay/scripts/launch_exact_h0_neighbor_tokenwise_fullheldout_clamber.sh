#!/usr/bin/env bash
set -euo pipefail

model_key="${1:?usage: $0 MODEL_KEY GPU}"
gpu="${2:?usage: $0 MODEL_KEY GPU}"

case "$model_key" in
  llama)
    config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
    shared_ratio="0.3"
    ;;
  gemma)
    config="configs/runs/gemma_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
    shared_ratio="0.4"
    ;;
  *)
    printf 'Unsupported model key: %s\n' "$model_key" >&2
    exit 2
    ;;
esac

root="artifacts/steering_exact_h0_neighbor_tokenwise_fullheldout_${model_key}_clamber"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets clamber \
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
  --classifier-target-quantile 0.5 \
  --topology-alphas 0 1 2 \
  --shared-target-ratios "$shared_ratio" \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 0 \
  --topology-decode-mode none \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 24
