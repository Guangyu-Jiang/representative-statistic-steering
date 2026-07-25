#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-2}"
root="artifacts/steering_exact_h0_neighbor_tokenwise_target_ablation_llama"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"

run_dataset() {
  local dataset="$1"
  local shared_ratio="$2"
  local target_mode="$3"
  python scripts/run_exact_h0_hybrid_steering.py \
    --config "$config" \
    --datasets "$dataset" \
    --artifact-root "$root" \
    --behavior-label-source rule_targeted_clarification \
    --positive-label-mode grounded \
    --topology-controller behavior_tokenwise \
    --direction-source observed_neighbor_tokenwise \
    --direction-readout mean_pool \
    --retrieval-feature-mode all_layer_plus_exact3 \
    --retrieval-geometry class_residual \
    --neighbor-ks 20 \
    --target-mode "$target_mode" \
    --classifier-target-quantile 0.5 \
    --topology-alphas 0 0.5 1 2 \
    --shared-target-ratios "$shared_ratio" \
    --lambdas 0.1 \
    --dampings 0.01 \
    --trust-ratios 0.1 \
    --gn-steps 12 \
    --optimization-jobs 8 \
    --eval-n 128 \
    --topology-decode-mode none \
    --shared-intervention-site layer_output \
    --apply-on prompt_and_decode_shared
}

for target_mode in local_contrast nearest_grounded; do
  run_dataset ambigqa 0.3 "$target_mode"
  run_dataset situatedqa 0.4 "$target_mode"
  run_dataset clamber 0.3 "$target_mode"
done

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 24
