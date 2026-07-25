#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_FOLD0="${ITI_GPU_FOLD0:-1}"
GPU_FOLD1="${ITI_GPU_FOLD1:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
GRID="validation/iti_group_direction_validation_settings.json"
SETTINGS="validation/iti_group_direction_heldout_settings.json"
VALIDATION="artifacts/iti_attention_head/validation_group_direction_k48"
OUTPUT="artifacts/iti_attention_head/heldout_group_direction_k48"
SCALAR_REFERENCE="artifacts/iti_attention_head/heldout_direct_probe_q0p99_k48/merged"
HEADWISE_REFERENCE="artifacts/iti_attention_head/heldout_headwise_direct_target_k48/merged"
FIXED_REFERENCE="artifacts/iti_attention_head/heldout_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUTPUT/fold0" "$OUTPUT/fold1" "$OUTPUT/merged" "$LOGS"

"$PYTHON_BIN" validation/select_iti_group_direction_candidates.py \
  --summary "$VALIDATION/combined_validation_summary.csv" \
  --grid "$GRID" \
  --output-settings "$SETTINGS" \
  --output-report "$VALIDATION/selected_candidates.csv"

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
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48_group_direction.npz" \
      --output-dir "$OUTPUT/fold${fold}" \
      >"$LOGS/group_direction_heldout_fold${fold}.log" 2>&1
}

run_fold 0 "$GPU_FOLD0" 409 & fold0_pid=$!
run_fold 1 "$GPU_FOLD1" 408 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUTPUT"/fold*/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUTPUT/combined_test_summary.csv" \
  >"$LOGS/group_direction_heldout_summary.log" 2>&1

cp "$OUTPUT/fold0/fold_0_test_results.csv" "$OUTPUT/merged/fold_0_test_results.csv"
cp "$OUTPUT/fold1/fold_1_test_results.csv" "$OUTPUT/merged/fold_1_test_results.csv"

CANDIDATES=$(
  "$PYTHON_BIN" -c \
    "import json; from validation.select_iti_group_direction_candidates import setting_tag; print(' '.join(setting_tag(x) for x in json.load(open('$SETTINGS'))))"
)

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$OUTPUT/merged/fold_0_test_results.csv" \
  "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference baseline --settings $CANDIDATES \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_baseline.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$SCALAR_REFERENCE/fold_0_test_results.csv" \
  "$SCALAR_REFERENCE/fold_1_test_results.csv" \
  --candidate-results \
    "$OUTPUT/merged/fold_0_test_results.csv" \
    "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference targeted_probe_iti_k48_a1_q0p99_r0 --settings $CANDIDATES \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_scalar_direct.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$HEADWISE_REFERENCE/fold_0_test_results.csv" \
  "$HEADWISE_REFERENCE/fold_1_test_results.csv" \
  --candidate-results \
    "$OUTPUT/merged/fold_0_test_results.csv" \
    "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference headwise_probe_iti_k48_a1_q1_r0 --settings $CANDIDATES \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_headwise_direct.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  "$FIXED_REFERENCE/fold_0_test_results.csv" \
  "$FIXED_REFERENCE/fold_1_test_results.csv" \
  --candidate-results \
    "$OUTPUT/merged/fold_0_test_results.csv" \
    "$OUTPUT/merged/fold_1_test_results.csv" \
  --reference fixed_com_k48_a15 --settings $CANDIDATES \
  --samples 20000 --seed 42 \
  --output "$OUTPUT/paired_vs_fixed_a15.csv"

cat "$OUTPUT/combined_test_summary.csv"
