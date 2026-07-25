#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_q_cap_sweep_settings.json"
SCREEN="artifacts/iti_attention_head/screen_general_min_norm_q_cap_r0p1_k48"
OUT="artifacts/iti_attention_head/confirm_general_min_norm_q_cap_r0p1_k48_constrained"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT/fold0" "$OUT/fold1" "$LOGS"

"$PYTHON_BIN" validation/select_iti_general_min_norm_candidates.py \
  --summary "$SCREEN/combined_validation_summary.csv" \
  --settings "$SETTINGS" \
  --ranking-output "$SCREEN/q_cap_constrained_ranking.csv" \
  --selected-settings-output "$OUT/selected_settings.json" \
  --top-mc1 4 --top-mc2 4 --top-mean 5 --top-efficiency 6 \
  --efficiency-max-mean-gap 0.06 \
  --include-setting-tags aggregate_com_k48_a20_q0p75_r0p1_c3 \
  >"$LOGS/general_min_norm_q_cap_constrained_selection.log" 2>&1

run_fold() {
  local fold="$1"

  env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads 48 \
      --settings-file "$OUT/selected_settings.json" \
      --question-offset 0 \
      --max-questions 82 \
      --checkpoint-every 2 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/fold${fold}" \
      >"$LOGS/general_min_norm_q_cap_constrained_fold${fold}.log" 2>&1
}

run_fold 0 & fold0_pid=$!
run_fold 1 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUT/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_q_cap_constrained_summary.log" 2>&1

echo "Constrained quantile/cap validation complete: $OUT/combined_validation_summary.csv"
