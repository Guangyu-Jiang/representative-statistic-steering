#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${ITI_GPU:-2}"
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
SETTINGS="validation/iti_generation_aggregate_com_a30_q0p55_settings.json"
OUT="artifacts/iti_attention_head/generation_aggregate_com_a30_q0p55_k48"
ORIGINAL="artifacts/iti_attention_head/generation_heldout_k48/merged"
ALPHA8="artifacts/iti_attention_head/generation_final_validation_selected_k48/merged"
LOGS="artifacts/iti_attention_head/logs"
TAG="aggregate_com_k48_a30_q0p55_r0p1_c3"
mkdir -p "$OUT" "$LOGS"

run_shard() {
  local fold="$1"
  local offset="$2"
  local count="$3"
  local shard="fold${fold}_offset${offset}"

  env CUDA_VISIBLE_DEVICES="$GPU" \
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
      --resume \
      --reuse-from "$ORIGINAL/fold_${fold}_test_generations.csv" \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/$shard" \
      >"$LOGS/generation_aggregate_com_a30_q0p55_${shard}.log" 2>&1
}

run_worker() {
  local fold0_offset="$1"
  local fold0_count="$2"
  local fold1_offset="$3"
  local fold1_count="$4"
  run_shard 0 "$fold0_offset" "$fold0_count"
  run_shard 1 "$fold1_offset" "$fold1_count"
}

# Three Llama-2 workers occupy about 41 GiB together on one 48 GiB GPU.
run_worker   0 137   0 136 & worker0=$!
run_worker 137 136 136 136 & worker1=$!
run_worker 273 136 272 136 & worker2=$!
wait "$worker0" "$worker1" "$worker2"

mkdir -p "$OUT/merged"
"$PYTHON_BIN" validation/merge_generation_shards.py \
  "$OUT"/fold0_offset*/fold_0_test_generations.csv \
  --expected-rows 409 \
  --allow-empty-answers \
  --enrich-from \
    "$ORIGINAL/fold_0_test_generations.csv" \
    "$ALPHA8/fold_0_test_generations.csv" \
  --output "$OUT/merged/fold_0_test_generations.csv"
"$PYTHON_BIN" validation/merge_generation_shards.py \
  "$OUT"/fold1_offset*/fold_1_test_generations.csv \
  --expected-rows 408 \
  --allow-empty-answers \
  --enrich-from \
    "$ORIGINAL/fold_1_test_generations.csv" \
    "$ALPHA8/fold_1_test_generations.csv" \
  --output "$OUT/merged/fold_1_test_generations.csv"

"$PYTHON_BIN" validation/audit_generation_outputs.py \
  "$OUT/merged/fold_0_test_generations.csv" \
  "$OUT/merged/fold_1_test_generations.csv" \
  --answer-columns \
    baseline_answer \
    fixed_com_k48_a15_answer \
    fixed_com_k48_a8_answer \
    "${TAG}_answer" \
  --output "$OUT/generation_audit.csv"

echo "COM open-ended generation complete: $OUT/merged"
