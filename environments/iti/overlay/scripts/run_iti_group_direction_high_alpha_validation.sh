#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_group_direction_high_alpha_settings.json"
OUTPUT="artifacts/iti_attention_head/validation_group_direction_high_alpha_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUTPUT" "$OUTPUT/merged" "$LOGS"

"$PYTHON_BIN" validation/build_iti_group_direction_high_alpha_settings.py

run_shard() {
  local fold="$1"
  local gpu="$2"
  local shard="$3"
  local offset="$4"
  local count="$5"
  local shard_dir="$OUTPUT/fold${fold}_shard${shard}"
  mkdir -p "$shard_dir"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads 48 \
      --settings-file "$SETTINGS" \
      --question-offset "$offset" \
      --max-questions "$count" \
      --checkpoint-every 1 \
      --resume \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48_group_direction_raw.npz" \
      --output-dir "$shard_dir" \
      >"$LOGS/group_direction_high_alpha_fold${fold}_shard${shard}.log" 2>&1
}

run_queue() {
  local gpu="$1"
  local shard="$2"
  local offset="$3"
  local count="$4"
  run_shard 0 "$gpu" "$shard" "$offset" "$count"
  run_shard 1 "$gpu" "$shard" "$offset" "$count"
}

run_queue 0 0 0 28 & queue0_pid=$!
run_queue 1 1 28 27 & queue1_pid=$!
run_queue 2 2 55 27 & queue2_pid=$!
wait "$queue0_pid" "$queue1_pid" "$queue2_pid"

for fold in 0 1; do
  "$PYTHON_BIN" validation/merge_causal_head_result_shards.py \
    "$OUTPUT"/fold${fold}_shard*/fold_${fold}_validation_results.csv \
    --expected-rows 82 \
    --output "$OUTPUT/merged/fold_${fold}_validation_results.csv"
done

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUTPUT"/fold*_shard*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 6 \
  --output "$OUTPUT/combined_validation_summary.csv" \
  >"$LOGS/group_direction_high_alpha_summary.log" 2>&1

cat "$OUTPUT/combined_validation_summary.csv"
