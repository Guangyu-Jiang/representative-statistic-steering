#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

JUDGE_GPUS="${ITI_JUDGE_GPUS:-2}"
read -r -a GPU_MEMORY <<<"${ITI_JUDGE_MEMORY_GIB:-12}"
BATCH_SIZE="${ITI_JUDGE_BATCH_SIZE:-512}"
GEN="artifacts/iti_attention_head/generation_aggregate_com_a30_q0p55_k48/merged"
OUT="artifacts/iti_attention_head/generation_aggregate_com_a30_q0p55_k48/local_qwen72b_judge"
REF15="artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge"
REF8="artifacts/iti_attention_head/generation_final_validation_selected_k48/local_qwen72b_judge"
LOG="artifacts/iti_attention_head/logs/generation_aggregate_com_a30_q0p55_qwen72b_judge.log"
JUDGE_TAG="qwen_qwen2_5_72b_instruct"
NEW="aggregate_com_k48_a30_q0p55_r0p1_c3_answer"
mkdir -p "$OUT"

while [[ ! -s "$GEN/fold_0_test_generations.csv" || ! -s "$GEN/fold_1_test_generations.csv" ]]; do
  sleep 30
done

env CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$GEN/fold_0_test_generations.csv" \
    "$GEN/fold_1_test_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size "$BATCH_SIZE" --max-new-tokens 24 --checkpoint-every 2 \
    --gpu-memory-gib "${GPU_MEMORY[@]}" --cpu-memory-gib 256 \
    --output-dir "$OUT" --answer-columns "$NEW" \
    >"$LOG" 2>&1

candidate=(
  "$OUT/fold_0_test_generations__${NEW}__${JUDGE_TAG}_judged.csv"
  "$OUT/fold_1_test_generations__${NEW}__${JUDGE_TAG}_judged.csv"
)

compare() {
  local reference_dir="$1"
  local reference="$2"
  local output="$3"
  "$PYTHON_BIN" validation/compare_local_judge.py \
    --reference \
      "$reference_dir/fold_0_test_generations__${reference}__${JUDGE_TAG}_judged.csv" \
      "$reference_dir/fold_1_test_generations__${reference}__${JUDGE_TAG}_judged.csv" \
    --candidate "${candidate[@]}" \
    --samples 20000 --seed 42 --output "$OUT/$output"
}

compare "$REF15" baseline_answer aggregate_com_vs_baseline_paired.csv
compare "$REF15" fixed_com_k48_a15_answer aggregate_com_vs_fixed_a15_paired.csv
compare "$REF8" fixed_com_k48_a8_answer aggregate_com_vs_fixed_a8_paired.csv

echo "Qwen-72B COM judge complete: $OUT/local_qwen_judge_summary.csv"
