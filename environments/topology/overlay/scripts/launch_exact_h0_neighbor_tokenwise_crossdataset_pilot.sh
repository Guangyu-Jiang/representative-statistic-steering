#!/usr/bin/env bash
set -euo pipefail

model_key="${1:?usage: $0 MODEL_KEY GPU [DATASET ...]}"
gpu="${2:?usage: $0 MODEL_KEY GPU [DATASET ...]}"
shift 2
datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
  datasets=(ambigqa situatedqa clamber)
fi

case "$model_key" in
  llama) config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml" ;;
  gemma) config="configs/runs/gemma_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml" ;;
  mistral) config="configs/runs/mistral_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml" ;;
  *) printf 'Unknown model key: %s\n' "$model_key" >&2; exit 2 ;;
esac

root="artifacts/steering_exact_h0_neighbor_tokenwise_crossdataset_${model_key}"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/run_exact_h0_hybrid_steering.py \
  --config "$config" \
  --datasets "${datasets[@]}" \
  --artifact-root "$root" \
  --behavior-label-source rule_targeted_clarification \
  --positive-label-mode grounded \
  --topology-controller behavior_tokenwise \
  --direction-source observed_neighbor_tokenwise \
  --direction-readout mean_pool \
  --retrieval-feature-mode all_layer_plus_exact3 \
  --retrieval-geometry class_residual \
  --neighbor-ks 5 20 \
  --target-mode classifier_projection \
  --classifier-target-quantile 0.5 \
  --topology-alphas 0 1 2 \
  --shared-target-ratios 0.2 0.3 0.4 \
  --lambdas 0.1 \
  --dampings 0.01 \
  --trust-ratios 0.1 \
  --gn-steps 12 \
  --optimization-jobs 8 \
  --eval-n 64 \
  --topology-decode-mode none \
  --shared-intervention-site layer_output \
  --apply-on prompt_and_decode_shared

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
