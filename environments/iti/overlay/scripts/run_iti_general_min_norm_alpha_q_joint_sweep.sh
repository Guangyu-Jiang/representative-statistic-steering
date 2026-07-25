#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_alpha_q_joint_settings.json"
SCREEN="artifacts/iti_attention_head/screen_general_min_norm_alpha_q_joint_k48"
CONFIRM="artifacts/iti_attention_head/confirm_general_min_norm_alpha_q_joint_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$SCREEN/fold0" "$SCREEN/fold1" "$CONFIRM/fold0" "$CONFIRM/fold1" "$LOGS"

"$PYTHON_BIN" validation/build_iti_general_min_norm_sweep.py \
  --output "$SETTINGS" \
  --methods aggregate_com \
  --alphas 18 20 22 24 26 28 30 \
  --target-quantiles 0.55 0.60 0.65 0.70 0.75 0.80 0.85 \
  --relative-caps 3.0 \
  --ridge-ratios 0.1 \
  --fixed-alphas 8 15 \
  --no-legacy-best

run_fold() {
  local phase="$1"
  local fold="$2"
  local count="$3"
  local settings="$4"
  local output_root="$5"

  env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads 48 \
      --settings-file "$settings" \
      --question-offset 0 \
      --max-questions "$count" \
      --checkpoint-every 2 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$output_root/fold${fold}" \
      >"$LOGS/general_min_norm_alpha_q_${phase}_fold${fold}.log" 2>&1
}

# Broad 49-setting screen on 48 validation questions.
run_fold screen 0 24 "$SETTINGS" "$SCREEN" & screen0_pid=$!
run_fold screen 1 24 "$SETTINGS" "$SCREEN" & screen1_pid=$!
wait "$screen0_pid" "$screen1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$SCREEN"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$SCREEN/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_alpha_q_screen_summary.log" 2>&1

"$PYTHON_BIN" validation/select_iti_general_min_norm_candidates.py \
  --summary "$SCREEN/combined_validation_summary.csv" \
  --settings "$SETTINGS" \
  --ranking-output "$SCREEN/alpha_q_ranking.csv" \
  --selected-settings-output "$CONFIRM/selected_settings.json" \
  --top-mc1 5 --top-mc2 5 --top-mean 6 --top-efficiency 8 \
  --efficiency-max-mean-gap 0.04 \
  --include-setting-tags \
    aggregate_com_k48_a22_q0p7_r0p1_c3 \
    aggregate_com_k48_a26_q0p7_r0p1_c3 \
  >"$LOGS/general_min_norm_alpha_q_selection.log" 2>&1

# Confirm the selected frontier on all 164 validation questions.
run_fold confirm 0 82 "$CONFIRM/selected_settings.json" "$CONFIRM" & confirm0_pid=$!
run_fold confirm 1 82 "$CONFIRM/selected_settings.json" "$CONFIRM" & confirm1_pid=$!
wait "$confirm0_pid" "$confirm1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$CONFIRM"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$CONFIRM/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_alpha_q_confirm_summary.log" 2>&1

echo "Joint alpha/quantile validation complete: $CONFIRM/combined_validation_summary.csv"
