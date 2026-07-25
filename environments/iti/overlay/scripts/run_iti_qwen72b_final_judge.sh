#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT="artifacts/iti_attention_head/generation_selected_probe_a12_k48/merged"
OUTPUT="artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge"
LOG="artifacts/iti_attention_head/logs/generation_selected_probe_a12_qwen72b_judge.log"

while [[ ! -s "$INPUT/fold_0_test_generations.csv" || ! -s "$INPUT/fold_1_test_generations.csv" ]]; do
  sleep 30
done
sleep 15
mkdir -p "$OUTPUT"

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$INPUT/fold_0_test_generations.csv" \
    "$INPUT/fold_1_test_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size 8 \
    --max-new-tokens 24 \
    --checkpoint-every 2 \
    --gpu-memory-gib 44 30 44 44 \
    --output-dir "$OUTPUT" \
    --answer-columns baseline_answer fixed_com_k48_a15_answer \
      targeted_probe_iti_k48_a12_q0p75_r0_c2_answer >"$LOG" 2>&1
