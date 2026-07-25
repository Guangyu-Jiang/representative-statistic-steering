#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

SETTINGS="validation/iti_generation_fixed_norm_a7p1_settings.json"
OUT="artifacts/iti_attention_head/generation_fixed_norm_a7p1_k48"
TARGET="artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge"
JUDGE="$OUT/local_qwen72b_judge"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$LOGS"

while pgrep -f 'scripts/run_iti_post_qwen_checks.sh' >/dev/null; do
  sleep 60
done
sleep 30

bash scripts/run_iti_generation_candidate_heldout.sh "$SETTINGS" "$OUT" \
  >"$LOGS/generation_fixed_norm_a7p1.log" 2>&1

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$OUT/merged/fold_0_test_generations.csv" \
    "$OUT/merged/fold_1_test_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size 8 --max-new-tokens 24 --checkpoint-every 2 \
    --gpu-memory-gib 44 34 44 44 \
    --output-dir "$JUDGE" \
    --answer-columns fixed_com_k48_a7p1_answer \
    >"$LOGS/generation_fixed_norm_a7p1_qwen72b_judge.log" 2>&1

"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE"/*__qwen_qwen2_5_72b_instruct_judged.csv \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output "$JUDGE/local_qwen_judge_summary.csv"

"$PYTHON_BIN" validation/compare_local_judge.py \
  --reference \
    "$JUDGE/fold_0_test_generations__fixed_com_k48_a7p1_answer__qwen_qwen2_5_72b_instruct_judged.csv" \
    "$JUDGE/fold_1_test_generations__fixed_com_k48_a7p1_answer__qwen_qwen2_5_72b_instruct_judged.csv" \
  --candidate \
    "$TARGET/fold_0_test_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__qwen_qwen2_5_72b_instruct_judged.csv" \
    "$TARGET/fold_1_test_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__qwen_qwen2_5_72b_instruct_judged.csv" \
  --samples 20000 --seed 42 \
  --output "$JUDGE/targeted_a12_vs_fixed_a7p1_paired.csv"

"$PYTHON_BIN" validation/audit_generation_outputs.py \
  "$OUT/merged/fold_0_test_generations.csv" \
  "$OUT/merged/fold_1_test_generations.csv" \
  --answer-columns baseline_answer fixed_com_k48_a7p1_answer \
  --output "$OUT/generation_audit.csv"
