#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
root="artifacts/steering_exact_h0_audited_local_pilot"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
label_path="artifacts/local_rotating_fourway_base_behavior/qwen_qwen2_5_7b_instruct_rotating_choice_bdbf18432ffc/base_behavior_local_labels.parquet"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

for positive_mode in grounded all_acceptable; do
  python scripts/run_exact_h0_hybrid_steering.py \
    --config "$config" \
    --datasets ambigqa \
    --artifact-root "$root" \
    --behavior-label-source audited_local_fourway \
    --local-label-path "$label_path" \
    --local-label-confidence-min 0.6 \
    --local-label-margin-min 0.2 \
    --require-positive-rule-marker \
    --positive-label-mode "$positive_mode" \
    --topology-controller behavior_lowrank \
    --behavior-rank 4 \
    --retrieval-feature-mode all_layer_plus_exact3 \
    --retrieval-geometry class_residual \
    --neighbor-ks 20 \
    --target-mode local_contrast \
    --topology-alphas 0 1 2 \
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
done

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
