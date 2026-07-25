#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUT="artifacts/iti_attention_head/generation_fixed_norm_a7p1_k48"
GEN="$OUT/merged"
TARGET="artifacts/iti_attention_head/generation_selected_probe_a12_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$LOGS"

while pgrep -f '^bash scripts/run_iti_fixed_norm_generation_after_checks\.sh$' >/dev/null; do
  sleep 30
done
sleep 15

run_judge() {
  local gpu="$1"
  local model="$2"
  local batch_size="$3"
  local max_tokens="$4"
  local judge_dir="$5"
  local log="$6"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
      "$GEN/fold_0_test_generations.csv" \
      "$GEN/fold_1_test_generations.csv" \
      --judge-model "$model" \
      --batch-size "$batch_size" --max-new-tokens "$max_tokens" \
      --checkpoint-every 5 --output-dir "$judge_dir" \
      --answer-columns fixed_com_k48_a7p1_answer >"$log" 2>&1
}

QWEN_OUT="$OUT/local_qwen7b_judge"
MISTRAL_OUT="$OUT/local_mistral7b_judge_64"
mkdir -p "$QWEN_OUT" "$MISTRAL_OUT"

run_judge 0 Qwen/Qwen2.5-7B-Instruct 24 24 "$QWEN_OUT" \
  "$LOGS/generation_fixed_norm_a7p1_qwen7b_judge.log" & qwen_pid=$!
run_judge 3 mistralai/Mistral-7B-Instruct-v0.3 24 64 "$MISTRAL_OUT" \
  "$LOGS/generation_fixed_norm_a7p1_mistral7b_judge.log" & mistral_pid=$!
wait "$qwen_pid" "$mistral_pid"

summarize_and_compare() {
  local model="$1"
  local slug="$2"
  local judge_dir="$3"
  local target_dir="$4"

  "$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
    "$judge_dir"/*__"$slug"_judged.csv \
    --judge-model "$model" \
    --output "$judge_dir/local_qwen_judge_summary.csv"

  "$PYTHON_BIN" validation/compare_local_judge.py \
    --reference \
      "$judge_dir/fold_0_test_generations__fixed_com_k48_a7p1_answer__${slug}_judged.csv" \
      "$judge_dir/fold_1_test_generations__fixed_com_k48_a7p1_answer__${slug}_judged.csv" \
    --candidate \
      "$target_dir/fold_0_test_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__${slug}_judged.csv" \
      "$target_dir/fold_1_test_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__${slug}_judged.csv" \
    --samples 20000 --seed 42 \
    --output "$judge_dir/targeted_a12_vs_fixed_a7p1_paired.csv"
}

summarize_and_compare \
  Qwen/Qwen2.5-7B-Instruct qwen_qwen2_5_7b_instruct \
  "$QWEN_OUT" "$TARGET/local_qwen7b_judge"
summarize_and_compare \
  mistralai/Mistral-7B-Instruct-v0.3 mistralai_mistral_7b_instruct_v0_3 \
  "$MISTRAL_OUT" "$TARGET/local_mistral7b_judge_64"
