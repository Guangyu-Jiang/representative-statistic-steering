#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_FOLD0="${ITI_GPU_FOLD0:-1}"
GPU_FOLD1="${ITI_GPU_FOLD1:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
OUTPUT="artifacts/iti_attention_head/validation_headwise_direct_target_k_sweep"
STATISTICS="artifacts/iti_attention_head/statistics"
LOGS="artifacts/iti_attention_head/logs"
K_VALUES=(8 16 32 48)
mkdir -p "$OUTPUT/settings" "$LOGS"

for k in "${K_VALUES[@]}"; do
  "$PYTHON_BIN" validation/build_iti_headwise_direct_target_settings.py \
    --num-heads "$k" --output "$OUTPUT/settings/k${k}.json"
done

run_fold() {
  local k="$1"
  local fold="$2"
  local gpu="$3"
  local out="$OUTPUT/k${k}/fold${fold}"
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads "$k" \
      --settings-file "$OUTPUT/settings/k${k}.json" \
      --max-questions 82 \
      --checkpoint-every 2 \
      --resume \
      --statistics-cache "$STATISTICS/llama2_fold${fold}_k${k}.npz" \
      --output-dir "$out" \
      >"$LOGS/headwise_direct_target_k${k}_fold${fold}.log" 2>&1
}

for k in "${K_VALUES[@]}"; do
  run_fold "$k" 0 "$GPU_FOLD0" & fold0_pid=$!
  run_fold "$k" 1 "$GPU_FOLD1" & fold1_pid=$!
  wait "$fold0_pid" "$fold1_pid"

  "$PYTHON_BIN" validation/summarize_causal_head_results.py \
    "$OUTPUT/k${k}"/fold*/fold_*_validation_summary.csv \
    --min-folds 2 --min-sources 2 \
    --output "$OUTPUT/k${k}/combined_validation_summary.csv" \
    >"$LOGS/headwise_direct_target_k${k}_summary.log" 2>&1
done

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUTPUT"/k*/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUTPUT/combined_validation_summary.csv" \
  >"$LOGS/headwise_direct_target_k_sweep_summary.log" 2>&1

cat "$OUTPUT/combined_validation_summary.csv"
