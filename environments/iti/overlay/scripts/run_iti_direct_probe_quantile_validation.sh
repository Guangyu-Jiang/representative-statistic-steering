#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_FOLD0="${ITI_GPU_FOLD0:-1}"
GPU_FOLD1="${ITI_GPU_FOLD1:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_direct_probe_quantile_settings.json"
OUTPUT="artifacts/iti_attention_head/validation_direct_probe_quantiles_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUTPUT/fold0" "$OUTPUT/fold1" "$LOGS"

run_fold() {
  local fold="$1"
  local gpu="$2"
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
      --max-questions 82 \
      --checkpoint-every 2 \
      --resume \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUTPUT/fold${fold}" \
      >"$LOGS/direct_probe_quantiles_fold${fold}.log" 2>&1
}

run_fold 0 "$GPU_FOLD0" & fold0_pid=$!
run_fold 1 "$GPU_FOLD1" & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUTPUT"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUTPUT/combined_validation_summary.csv" \
  >"$LOGS/direct_probe_quantiles_summary.log" 2>&1

cat "$OUTPUT/combined_validation_summary.csv"
