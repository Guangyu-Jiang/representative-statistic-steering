#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_generation_final_validation_selected_settings.json"
OUT="artifacts/iti_attention_head/heldout_final_bounded_k48"
REFERENCE="artifacts/iti_attention_head/heldout_k48"
LOGS="artifacts/iti_attention_head/logs"
JUDGE_DONE="artifacts/iti_attention_head/generation_final_validation_selected_k48/local_qwen72b_judge/heldout_method_comparison.csv"
GPU_FOLD0="${GPU_FOLD0:-0}"
GPU_FOLD1="${GPU_FOLD1:-2}"
mkdir -p "$OUT" "$LOGS"

while [[ ! -s "$JUDGE_DONE" ]]; do
  sleep 30
done

run_fold() {
  local gpu="$1"
  local fold="$2"
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
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --checkpoint-every 2 \
      --output-dir "$OUT" \
      >"$LOGS/heldout_final_bounded_mc_fold${fold}.log" 2>&1
}

run_fold "$GPU_FOLD0" 0 & pid0=$!
run_fold "$GPU_FOLD1" 1 & pid1=$!
wait "$pid0" "$pid1"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUT/combined_test_summary.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$OUT/fold_0_test_results.csv" "$OUT/fold_1_test_results.csv" \
  --reference fixed_com_k48_a8 \
  --settings bounded_targeted_probe_iti_k48_a12_q0p75_r0_c2_b10 \
  --samples 20000 --seed 42 \
  --output "$OUT/paired_vs_fixed_a8.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$REFERENCE/fold_0_test_results.csv" "$REFERENCE/fold_1_test_results.csv" \
  --candidate-results "$OUT/fold_0_test_results.csv" "$OUT/fold_1_test_results.csv" \
  --reference fixed_com_k48_a15 \
  --settings bounded_targeted_probe_iti_k48_a12_q0p75_r0_c2_b10 \
  --samples 20000 --seed 42 \
  --output "$OUT/paired_vs_fixed_a15.csv"
