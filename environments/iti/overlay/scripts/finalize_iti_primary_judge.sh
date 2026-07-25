#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GEN="artifacts/iti_attention_head/generation_selected_probe_a12_k48/merged"
JUDGE="artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge"
TAG="qwen_qwen2_5_72b_instruct"
BASE="baseline_answer"
FIXED="fixed_com_k48_a15_answer"
TARGET="targeted_probe_iti_k48_a12_q0p75_r0_c2_answer"

"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE"/fold_*__"$TAG"_judged.csv \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output "$JUDGE/local_qwen_judge_summary.csv"

compare() {
  local reference="$1"
  local candidate="$2"
  local output="$3"
  "$PYTHON_BIN" validation/compare_local_judge.py \
    --reference \
      "$JUDGE/fold_0_test_generations__${reference}__${TAG}_judged.csv" \
      "$JUDGE/fold_1_test_generations__${reference}__${TAG}_judged.csv" \
    --candidate \
      "$JUDGE/fold_0_test_generations__${candidate}__${TAG}_judged.csv" \
      "$JUDGE/fold_1_test_generations__${candidate}__${TAG}_judged.csv" \
    --samples 20000 --seed 42 --output "$JUDGE/$output"
}

compare "$FIXED" "$TARGET" targeted_a12_vs_fixed_a15_paired.csv
compare "$BASE" "$TARGET" targeted_a12_vs_baseline_paired.csv
compare "$BASE" "$FIXED" fixed_a15_vs_baseline_paired.csv

"$PYTHON_BIN" validation/audit_generation_outputs.py \
  "$GEN/fold_0_test_generations.csv" "$GEN/fold_1_test_generations.csv" \
  --answer-columns "$BASE" "$FIXED" "$TARGET" \
  --output artifacts/iti_attention_head/generation_selected_probe_a12_k48/generation_audit.csv
