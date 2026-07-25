#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_alpha_q_joint_heldout_settings.json"
OUT="artifacts/iti_attention_head/heldout_general_min_norm_alpha_q_joint_r0p1_c3_k48"
ALPHA_REFERENCE="artifacts/iti_attention_head/heldout_general_min_norm_alpha_q0p7_r0p1_c3_k48"
ZERO_RIDGE_REFERENCE="artifacts/iti_attention_head/heldout_general_min_norm_norm_matched_k48"
LOGS="artifacts/iti_attention_head/logs"
KNEE="aggregate_com_k48_a22_q0p7_r0p1_c3"
PERFORMANCE_REFERENCE="aggregate_com_k48_a26_q0p7_r0p1_c3"
ZERO_RIDGE="aggregate_com_k48_a20_q0p75_r0_c3"
CANDIDATES=(
  aggregate_com_k48_a30_q0p55_r0p1_c3
  aggregate_com_k48_a26_q0p6_r0p1_c3
  aggregate_com_k48_a24_q0p65_r0p1_c3
  aggregate_com_k48_a28_q0p55_r0p1_c3
)
mkdir -p "$OUT/fold0" "$OUT/fold1" "$OUT/merged" "$LOGS"

run_fold() {
  local fold="$1"
  local count="$2"

  env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split test \
      --num-heads 48 \
      --settings-file "$SETTINGS" \
      --question-offset 0 \
      --max-questions "$count" \
      --checkpoint-every 5 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/fold${fold}" \
      >"$LOGS/general_min_norm_alpha_q_joint_heldout_fold${fold}.log" 2>&1
}

run_fold 0 409 & fold0_pid=$!
run_fold 1 408 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold*/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUT/combined_test_summary.csv" \
  >"$LOGS/general_min_norm_alpha_q_joint_heldout_summary.log" 2>&1

cp "$OUT/fold0/fold_0_test_results.csv" "$OUT/merged/fold_0_test_results.csv"
cp "$OUT/fold1/fold_1_test_results.csv" "$OUT/merged/fold_1_test_results.csv"

for candidate in "${CANDIDATES[@]}"; do
  for reference in "$KNEE" "$PERFORMANCE_REFERENCE"; do
    "$PYTHON_BIN" validation/compare_paired_mc.py \
      "$ALPHA_REFERENCE/merged/fold_0_test_results.csv" \
      "$ALPHA_REFERENCE/merged/fold_1_test_results.csv" \
      --candidate-results \
        "$OUT/merged/fold_0_test_results.csv" \
        "$OUT/merged/fold_1_test_results.csv" \
      --reference "$reference" --settings "$candidate" \
      --samples 20000 \
      --output "$OUT/paired_${candidate}_vs_${reference}.csv" \
      >"$LOGS/general_min_norm_alpha_q_joint_${candidate}_vs_${reference}.log" 2>&1
  done

  "$PYTHON_BIN" validation/compare_paired_mc.py \
    "$ZERO_RIDGE_REFERENCE/merged/fold_0_test_results.csv" \
    "$ZERO_RIDGE_REFERENCE/merged/fold_1_test_results.csv" \
    --candidate-results \
      "$OUT/merged/fold_0_test_results.csv" \
      "$OUT/merged/fold_1_test_results.csv" \
    --reference "$ZERO_RIDGE" --settings "$candidate" \
    --samples 20000 \
    --output "$OUT/paired_${candidate}_vs_${ZERO_RIDGE}.csv" \
    >"$LOGS/general_min_norm_alpha_q_joint_${candidate}_vs_${ZERO_RIDGE}.log" 2>&1
done

echo "Joint alpha/quantile held-out evaluation complete: $OUT/combined_test_summary.csv"
