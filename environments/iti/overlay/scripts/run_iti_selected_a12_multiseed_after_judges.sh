#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_generation_probe_a12_candidate_settings.json"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$LOGS"

while pgrep -f '^bash scripts/run_iti_fixed_norm_sensitivity_after_primary\.sh$' >/dev/null; do
  sleep 30
done
sleep 15

run_fold() {
  local gpu="$1"
  local seed="$2"
  local fold="$3"
  local out="artifacts/iti_attention_head/robustness_selected_a12_seed${seed}_k48"
  mkdir -p "$out"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/validate_causal_head_perturbation.py \
      --model-path "$MODEL" --feature-prefix "$FEATURES" \
      --seed "$seed" --fold "$fold" --eval-split test --num-heads 48 \
      --settings-file "$SETTINGS" \
      --statistics-cache \
        "artifacts/iti_attention_head/statistics/llama2_seed${seed}_fold${fold}_k48.npz" \
      --checkpoint-every 2 --output-dir "$out" \
      >"$LOGS/robustness_selected_a12_seed${seed}_fold${fold}.log" 2>&1
}

run_fold 0 1 0 & p10=$!
run_fold 1 1 1 & p11=$!
run_fold 2 2 0 & p20=$!
run_fold 3 2 1 & p21=$!
wait "$p10" "$p11" "$p20" "$p21"

run_fold 0 3 0 & p30=$!
run_fold 1 3 1 & p31=$!
wait "$p30" "$p31"

for seed in 1 2 3; do
  out="artifacts/iti_attention_head/robustness_selected_a12_seed${seed}_k48"
  reference="artifacts/iti_attention_head/robustness_seed${seed}_k48"
  "$PYTHON_BIN" validation/summarize_causal_head_results.py \
    "$out"/fold_*_test_summary.csv --min-folds 2 --min-sources 2 \
    --output "$out/combined_test_summary.csv"
  "$PYTHON_BIN" validation/compare_paired_mc.py \
    "$reference/fold_0_test_results.csv" "$reference/fold_1_test_results.csv" \
    --candidate-results "$out/fold_0_test_results.csv" "$out/fold_1_test_results.csv" \
    --reference fixed_com_k48_a15 \
    --settings targeted_probe_iti_k48_a12_q0p75_r0_c2 \
    --samples 20000 --seed 42 \
    --output "$out/paired_targeted_vs_fixed.csv"
done
