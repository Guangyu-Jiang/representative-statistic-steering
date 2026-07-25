#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SETTINGS_JSON OUTPUT_DIRECTORY" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
SETTINGS="$1"
OUT="$2"
ORIGINAL="artifacts/iti_attention_head/generation_heldout_k48/merged"
ALPHA8="artifacts/iti_attention_head/generation_selected_a8_k48/merged"
LOGS="artifacts/iti_attention_head/logs"
LOG_TAG="$(basename "$OUT")"
mkdir -p "$OUT" "$LOGS"

run_shard() {
  local gpu="$1"
  local fold="$2"
  local offset="$3"
  local count="$4"
  local tag="fold${fold}_offset${offset}"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/generate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --settings-file "$SETTINGS" \
      --eval-split test \
      --fold "$fold" \
      --question-offset "$offset" \
      --max-questions "$count" \
      --max-new-tokens 50 \
      --checkpoint-every 2 \
      --reuse-from "$ORIGINAL/fold_${fold}_test_generations.csv" \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/$tag" >"$LOGS/${LOG_TAG}_${tag}.log" 2>&1
}

run_pair() {
  local gpu="$1"
  local fold0_offset="$2"
  local fold0_count="$3"
  local fold1_offset="$4"
  local fold1_count="$5"
  run_shard "$gpu" 0 "$fold0_offset" "$fold0_count"
  run_shard "$gpu" 1 "$fold1_offset" "$fold1_count"
}

run_pair 0   0 103   0 102 & pid0=$!
run_pair 1 103 102 102 102 & pid1=$!
run_pair 2 205 102 204 102 & pid2=$!
run_pair 3 307 102 306 102 & pid3=$!
wait "$pid0" "$pid1" "$pid2" "$pid3"

mkdir -p "$OUT/merged"
"$PYTHON_BIN" validation/merge_generation_shards.py \
  "$OUT"/fold0_offset*/fold_0_test_generations.csv \
  --expected-rows 409 \
  --enrich-from "$ORIGINAL/fold_0_test_generations.csv" \
    "$ALPHA8/fold_0_test_generations.csv" \
  --output "$OUT/merged/fold_0_test_generations.csv"
"$PYTHON_BIN" validation/merge_generation_shards.py \
  "$OUT"/fold1_offset*/fold_1_test_generations.csv \
  --expected-rows 408 \
  --enrich-from "$ORIGINAL/fold_1_test_generations.csv" \
    "$ALPHA8/fold_1_test_generations.csv" \
  --output "$OUT/merged/fold_1_test_generations.csv"
