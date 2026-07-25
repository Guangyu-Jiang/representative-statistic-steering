#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
OUT="artifacts/iti_attention_head/screen_headwise_probe_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT" "$LOGS"

# Do not contend with the four-GPU 72B judge that selects the generation setting.
while pgrep -f 'local_truthfulqa_judge.py.*Qwen/Qwen2.5-72B-Instruct' >/dev/null; do
  sleep 30
done
sleep 15

run_screen() {
  local gpu="$1"
  local fold="$2"
  local quantile="$3"
  local quantile_tag="$4"
  local output="$OUT/q${quantile_tag}_fold${fold}"
  local log="$LOGS/headwise_probe_q${quantile_tag}_fold${fold}.log"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split validation \
      --num-heads 48 \
      --methods headwise_probe_iti \
      --target-quantiles "$quantile" \
      --strengths 0.5 1 2 4 \
      --ridge-ratios 0 \
      --relative-caps 0.25 0.5 1 2 \
      --max-questions 24 \
      --checkpoint-every 1 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$output" >"$log" 2>&1
}

run_screen 0 0 0.50 050 &
pid0=$!
run_screen 1 1 0.50 050 &
pid1=$!
run_screen 2 0 0.75 075 &
pid2=$!
run_screen 3 1 0.75 075 &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/*/fold_*_validation_summary.csv \
  --min-folds 2 \
  --min-sources 2 \
  --output "$OUT/combined_validation_summary.csv"
