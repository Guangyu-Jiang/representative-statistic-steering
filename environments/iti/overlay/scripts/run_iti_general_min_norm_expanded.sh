#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_expanded_settings.json"
SCREEN="artifacts/iti_attention_head/screen_general_min_norm_expanded_k48"
CONFIRM="artifacts/iti_attention_head/confirm_general_min_norm_expanded_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$SCREEN" "$CONFIRM" "$LOGS"

"$PYTHON_BIN" validation/build_iti_general_min_norm_sweep.py \
  --output "$SETTINGS"

run_shard() {
  local gpu="$1"
  local phase="$2"
  local fold="$3"
  local offset="$4"
  local count="$5"
  local settings="$6"
  local output_root="$7"
  local tag="fold${fold}_offset${offset}"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads 48 \
      --settings-file "$settings" \
      --question-offset "$offset" \
      --max-questions "$count" \
      --checkpoint-every 1 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$output_root/$tag" \
      >"$LOGS/general_min_norm_${phase}_${tag}.log" 2>&1
}

# Stage 1: 24 validation questions per fold, split over GPUs 0, 2, and 3.
run_shard 0 screen 0  0 8 "$SETTINGS" "$SCREEN" & s00=$!
run_shard 2 screen 0  8 8 "$SETTINGS" "$SCREEN" & s01=$!
run_shard 3 screen 0 16 8 "$SETTINGS" "$SCREEN" & s02=$!
run_shard 0 screen 1  0 8 "$SETTINGS" "$SCREEN" & s10=$!
run_shard 2 screen 1  8 8 "$SETTINGS" "$SCREEN" & s11=$!
run_shard 3 screen 1 16 8 "$SETTINGS" "$SCREEN" & s12=$!
wait "$s00" "$s01" "$s02" "$s10" "$s11" "$s12"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$SCREEN"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 6 \
  --output "$SCREEN/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_screen_summary.log" 2>&1

"$PYTHON_BIN" validation/select_iti_general_min_norm_candidates.py \
  --summary "$SCREEN/combined_validation_summary.csv" \
  --settings "$SETTINGS" \
  --ranking-output "$SCREEN/general_min_norm_ranking.csv" \
  --selected-settings-output "$CONFIRM/selected_settings.json" \
  >"$LOGS/general_min_norm_candidate_selection.log" 2>&1

# Stage 2: confirm the candidate union on all 82 validation questions per fold.
run_shard 0 confirm 0  0 28 "$CONFIRM/selected_settings.json" "$CONFIRM" & c00=$!
run_shard 2 confirm 0 28 27 "$CONFIRM/selected_settings.json" "$CONFIRM" & c01=$!
run_shard 3 confirm 0 55 27 "$CONFIRM/selected_settings.json" "$CONFIRM" & c02=$!
run_shard 0 confirm 1  0 28 "$CONFIRM/selected_settings.json" "$CONFIRM" & c10=$!
run_shard 2 confirm 1 28 27 "$CONFIRM/selected_settings.json" "$CONFIRM" & c11=$!
run_shard 3 confirm 1 55 27 "$CONFIRM/selected_settings.json" "$CONFIRM" & c12=$!
wait "$c00" "$c01" "$c02" "$c10" "$c11" "$c12"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$CONFIRM"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 6 \
  --output "$CONFIRM/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_confirm_summary.log" 2>&1
