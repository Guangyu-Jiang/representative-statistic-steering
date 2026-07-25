#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_ridge_confirm_settings.json"
REFERENCE="artifacts/iti_attention_head/heldout_general_min_norm_norm_matched_k48"
OUT="artifacts/iti_attention_head/heldout_general_min_norm_ridge_confirm_k48"
LOGS="artifacts/iti_attention_head/logs"
CANDIDATE="aggregate_com_k48_a20_q0p75_r0p1_c3"
BASELINE="aggregate_com_k48_a20_q0p75_r0_c3"
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
      >"$LOGS/general_min_norm_ridge_confirm_fold${fold}.log" 2>&1
}

run_fold 0 409 & fold0_pid=$!
run_fold 1 408 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold*/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUT/combined_test_summary.csv" \
  >"$LOGS/general_min_norm_ridge_confirm_summary.log" 2>&1

cp "$OUT/fold0/fold_0_test_results.csv" "$OUT/merged/fold_0_test_results.csv"
cp "$OUT/fold1/fold_1_test_results.csv" "$OUT/merged/fold_1_test_results.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$REFERENCE/merged/fold_0_test_results.csv" \
  "$REFERENCE/merged/fold_1_test_results.csv" \
  --candidate-results \
    "$OUT/merged/fold_0_test_results.csv" \
    "$OUT/merged/fold_1_test_results.csv" \
  --reference "$BASELINE" --settings "$CANDIDATE" \
  --samples 20000 \
  --output "$OUT/paired_${CANDIDATE}_vs_${BASELINE}.csv" \
  >"$LOGS/general_min_norm_ridge_confirm_paired.log" 2>&1
