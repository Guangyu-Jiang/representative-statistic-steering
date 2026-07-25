#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

BOUNDED="artifacts/iti_attention_head/generation_validation_bounded_probe_k48"
SUMMARY="$BOUNDED/local_qwen72b_judge/validation_controller_comparison.csv"
SETTINGS="validation/iti_generation_final_validation_selected_settings.json"
RANKING="$BOUNDED/local_qwen72b_judge/bounded_validation_ranking.csv"
TAG_FILE="$BOUNDED/local_qwen72b_judge/selected_bounded_tag.txt"
OUT="artifacts/iti_attention_head/generation_final_validation_selected_k48"
JUDGE="$OUT/local_qwen72b_judge"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT" "$JUDGE" "$LOGS"

while pgrep -f '^bash scripts/run_iti_bounded_probe_validation_after_fixed\.sh$' >/dev/null; do
  sleep 30
done
if [[ ! -s "$SUMMARY" ]]; then
  echo "Missing bounded validation summary: $SUMMARY" >&2
  exit 1
fi

"$PYTHON_BIN" validation/select_bounded_generation_setting.py \
  --summary "$SUMMARY" \
  --settings-file validation/iti_generation_bounded_probe_validation_settings.json \
  --fixed-alpha 8 \
  --output-settings "$SETTINGS" \
  --output-ranking "$RANKING" \
  --output-tag "$TAG_FILE" \
  >"$LOGS/generation_final_selection.log" 2>&1

bash scripts/run_iti_generation_candidate_heldout.sh "$SETTINGS" "$OUT" \
  >"$LOGS/generation_final_validation_selected.log" 2>&1

selected_tag="$(<"$TAG_FILE")"
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$OUT/merged/fold_0_test_generations.csv" \
    "$OUT/merged/fold_1_test_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size 8 --max-new-tokens 24 --checkpoint-every 2 \
    --gpu-memory-gib 44 34 44 44 --output-dir "$JUDGE" \
    --answer-columns fixed_com_k48_a8_answer "${selected_tag}_answer" \
    >"$LOGS/generation_final_validation_selected_qwen72b_judge.log" 2>&1

"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE"/*__qwen_qwen2_5_72b_instruct_judged.csv \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output "$JUDGE/heldout_method_comparison.csv"

reference=("$JUDGE"/*__fixed_com_k48_a8_answer__qwen_qwen2_5_72b_instruct_judged.csv)
candidate=("$JUDGE"/*__"${selected_tag}_answer"__qwen_qwen2_5_72b_instruct_judged.csv)
"$PYTHON_BIN" validation/compare_local_judge.py \
  --reference "${reference[@]}" \
  --candidate "${candidate[@]}" \
  --samples 20000 \
  --output "$JUDGE/bounded_vs_fixed_a8_paired.csv"
