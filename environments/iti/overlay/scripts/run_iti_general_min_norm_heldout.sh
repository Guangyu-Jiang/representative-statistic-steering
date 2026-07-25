#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
CONFIRM="artifacts/iti_attention_head/confirm_general_min_norm_expanded_k48"
SETTINGS="$CONFIRM/heldout_selected_settings.json"
OUT="artifacts/iti_attention_head/heldout_general_min_norm_expanded_k48"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT" "$OUT/merged" "$LOGS"

"$PYTHON_BIN" validation/select_iti_general_min_norm_final.py \
  --summary "$CONFIRM/combined_validation_summary.csv" \
  --candidate-settings "$CONFIRM/selected_settings.json" \
  --output "$SETTINGS" \
  --ranking-output "$CONFIRM/final_validation_ranking.csv" \
  >"$LOGS/general_min_norm_final_selection.log" 2>&1

run_shard() {
  local gpu="$1"
  local fold="$2"
  local offset="$3"
  local count="$4"
  local tag="fold${fold}_offset${offset}"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" \
      --feature-prefix "$FEATURES" \
      --fold "$fold" \
      --eval-split test \
      --num-heads 48 \
      --settings-file "$SETTINGS" \
      --question-offset "$offset" \
      --max-questions "$count" \
      --checkpoint-every 2 \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/$tag" \
      >"$LOGS/general_min_norm_heldout_${tag}.log" 2>&1
}

run_shard 0 0   0 137 & p00=$!
run_shard 2 0 137 136 & p01=$!
run_shard 3 0 273 136 & p02=$!
run_shard 0 1   0 136 & p10=$!
run_shard 2 1 136 136 & p11=$!
run_shard 3 1 272 136 & p12=$!
wait "$p00" "$p01" "$p02" "$p10" "$p11" "$p12"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$OUT"/fold*/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 6 \
  --output "$OUT/combined_test_summary.csv" \
  >"$LOGS/general_min_norm_heldout_summary.log" 2>&1

"$PYTHON_BIN" validation/merge_causal_head_result_shards.py \
  "$OUT"/fold0_*/fold_0_test_results.csv \
  --expected-rows 409 --output "$OUT/merged/fold_0_test_results.csv"
"$PYTHON_BIN" validation/merge_causal_head_result_shards.py \
  "$OUT"/fold1_*/fold_1_test_results.csv \
  --expected-rows 408 --output "$OUT/merged/fold_1_test_results.csv"

GENERAL_TAG="aggregate_com_k48_a24_q0p75_r0_c3"
for reference in fixed_com_k48_a8 fixed_com_k48_a15; do
  "$PYTHON_BIN" validation/compare_paired_mc.py \
    "$OUT/merged/fold_0_test_results.csv" \
    "$OUT/merged/fold_1_test_results.csv" \
    --reference "$reference" --settings "$GENERAL_TAG" \
    --samples 20000 \
    --output "$OUT/paired_${GENERAL_TAG}_vs_${reference}.csv" \
    >"$LOGS/general_min_norm_paired_vs_${reference}.log" 2>&1
done
