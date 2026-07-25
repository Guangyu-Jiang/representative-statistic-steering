#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_general_min_norm_alpha_sweep_settings.json"
OUT="artifacts/iti_attention_head/validation_general_min_norm_alpha_q0p7_r0p1_c3_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT/fold0" "$OUT/fold1" "$LOGS"

"$PYTHON_BIN" validation/build_iti_general_min_norm_sweep.py \
  --output "$SETTINGS" \
  --methods aggregate_com \
  --alphas 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 \
  --target-quantiles 0.70 \
  --relative-caps 3.0 \
  --ridge-ratios 0.1 \
  --fixed-alphas 8 15 \
  --no-legacy-best

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
      --settings-file "$SETTINGS" \
      --question-offset 0 \
      --max-questions 82 \
      --checkpoint-every 2 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/fold${fold}" \
      >"$LOGS/general_min_norm_alpha_sweep_fold${fold}.log" 2>&1
}

run_fold 0 & fold0_pid=$!
run_fold 1 & fold1_pid=$!
wait "$fold0_pid" "$fold1_pid"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold*/fold_*_validation_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$OUT/combined_validation_summary.csv" \
  >"$LOGS/general_min_norm_alpha_sweep_summary.log" 2>&1

echo "Alpha validation complete: $OUT/combined_validation_summary.csv"
