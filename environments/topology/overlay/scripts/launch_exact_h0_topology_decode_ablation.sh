#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
root="artifacts/steering_exact_h0_topology_decode_ablation_cleanlabel_pilot"
config="configs/runs/llama_steering_openai_ibm_80_20_clamber_aligned_mean_diff_raw.yaml"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"

common=(
  --config "$config"
  --datasets ambigqa
  --artifact-root "$root"
  --behavior-label-source rule_high_precision
  --positive-label-mode grounded
  --topology-controller behavior_lowrank
  --behavior-rank 4
  --retrieval-feature-mode all_layer_plus_exact3
  --retrieval-geometry standard
  --neighbor-ks 20
  --target-mode local_contrast
  --mean-alphas 0 4 6
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
  "${common[@]}" \
  --topology-alphas 0 1 2 \
  --topology-decode-mode none

for scale in 0.25 0.5 1.0; do
  python scripts/run_exact_h0_hybrid_steering.py \
    "${common[@]}" \
    --topology-alphas 1 2 \
    --topology-decode-mode last_token \
    --topology-decode-scale "$scale"
done

for scale in 0.5 1.0; do
  python scripts/run_exact_h0_hybrid_steering.py \
    "${common[@]}" \
    --topology-alphas 1 2 \
    --topology-decode-mode suffix_mean \
    --topology-decode-scale "$scale" \
    --topology-decode-suffix-fraction 0.25
done

python scripts/judge_exact_h0_gn_local.py \
  --artifact-root "$root" \
  --batch-size 32
