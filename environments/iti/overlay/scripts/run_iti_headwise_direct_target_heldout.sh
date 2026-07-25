#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_FOLD0="${ITI_GPU_FOLD0:-1}"
GPU_FOLD1="${ITI_GPU_FOLD1:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_headwise_direct_target_heldout_settings.json"
OUTPUT="artifacts/iti_attention_head/heldout_headwise_direct_target_k48"
FIXED_REFERENCE="artifacts/iti_attention_head/heldout_k48"
SCALAR_REFERENCE="artifacts/iti_attention_head/heldout_direct_probe_q0p99_k48/merged"
LOGS="artifacts/iti_attention_head/logs"
CANDIDATE="headwise_probe_iti_k48_a1_q1_r0"
SCALAR="targeted_probe_iti_k48_a1_q0p99_r0"
mkdir -p "$OUTPUT/fold0" "$OUTPUT/fold1" "$OUTPUT/merged" "$LOGS"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local count="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" \
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
      --resume \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUTPUT/fold${fold}" \
      >"$LOGS/headwise_direct_target_heldout_fold${fold}.log" 2>&1
}

run_fold 0 "$GPU_FOLD0" 409 & fold0_pid=$!
run_fold 1 "$GPU_FOLD1" 408 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUTPUT"/fold*/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUTPUT/combined_test_summary.csv" \
  >"$LOGS/headwise_direct_target_heldout_summary.log" 2>&1

cp "$OUTPUT/fold0/fold_0_test_results.csv" "$OUTPUT/merged/fold_0_test_results.csv"
cp "$OUTPUT/fold1/fold_1_test_results.csv" "$OUTPUT/merged/fold_1_test_results.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$OUTPUT/merged/fold_0_test_results.csv" \
  "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference baseline --settings "$CANDIDATE" \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_baseline.csv" \
  >"$LOGS/headwise_direct_target_paired_vs_baseline.log" 2>&1

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$SCALAR_REFERENCE/fold_0_test_results.csv" \
  "$SCALAR_REFERENCE/fold_1_test_results.csv" \
  --candidate-results \
    "$OUTPUT/merged/fold_0_test_results.csv" \
    "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference "$SCALAR" --settings "$CANDIDATE" \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_scalar_direct.csv" \
  >"$LOGS/headwise_direct_target_paired_vs_scalar.log" 2>&1

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$FIXED_REFERENCE/fold_0_test_results.csv" \
  "$FIXED_REFERENCE/fold_1_test_results.csv" \
  --candidate-results \
    "$OUTPUT/merged/fold_0_test_results.csv" \
    "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference fixed_com_k48_a15 --settings "$CANDIDATE" \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_fixed_a15.csv" \
  >"$LOGS/headwise_direct_target_paired_vs_fixed_a15.log" 2>&1

cat "$OUTPUT/combined_test_summary.csv"
