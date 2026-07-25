#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
FEATURES="features/llama2_chat_7B"
SETTINGS="validation/iti_generation_probe_a12_candidate_settings.json"
MC_OUT="artifacts/iti_attention_head/heldout_targeted_probe_a12_k48"
GEN="artifacts/iti_attention_head/generation_selected_probe_a12_k48/merged"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$MC_OUT" "$LOGS"

while pgrep -f 'local_truthfulqa_judge.py.*Qwen/Qwen2.5-72B-Instruct' >/dev/null; do
  sleep 30
done
sleep 15

run_mc() {
  local gpu="$1"
  local fold="$2"
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
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --checkpoint-every 2 \
      --output-dir "$MC_OUT" >"$LOGS/heldout_probe_a12_mc_fold${fold}.log" 2>&1
}

run_mc 0 0 & pid0=$!

env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
  TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$GEN/fold_0_test_generations.csv" "$GEN/fold_1_test_generations.csv" \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 24 --max-new-tokens 24 --checkpoint-every 5 \
    --output-dir artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen7b_judge \
    --answer-columns baseline_answer fixed_com_k48_a15_answer \
      targeted_probe_iti_k48_a12_q0p75_r0_c2_answer \
    >"$LOGS/generation_selected_probe_a12_qwen7b_judge.log" 2>&1 & pid1=$!

run_mc 2 1 & pid2=$!

env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
  TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$GEN/fold_0_test_generations.csv" "$GEN/fold_1_test_generations.csv" \
    --judge-model mistralai/Mistral-7B-Instruct-v0.3 \
    --batch-size 24 --max-new-tokens 64 --checkpoint-every 5 \
    --output-dir artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_mistral7b_judge_64 \
    --answer-columns baseline_answer fixed_com_k48_a15_answer \
      targeted_probe_iti_k48_a12_q0p75_r0_c2_answer \
    >"$LOGS/generation_selected_probe_a12_mistral7b_judge.log" 2>&1 & pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"

"$PYTHON_BIN" validation/summarize_causal_head_results.py \
  "$MC_OUT"/fold_*_test_summary.csv \
  --min-folds 2 --min-sources 2 \
  --output "$MC_OUT/combined_test_summary.csv"

"$PYTHON_BIN" validation/compare_paired_mc.py \
  artifacts/iti_attention_head/heldout_k48/fold_0_test_results.csv \
  artifacts/iti_attention_head/heldout_k48/fold_1_test_results.csv \
  --candidate-results "$MC_OUT/fold_0_test_results.csv" "$MC_OUT/fold_1_test_results.csv" \
  --reference fixed_com_k48_a15 \
  --settings targeted_probe_iti_k48_a12_q0p75_r0_c2 \
  --samples 20000 --seed 42 \
  --output "$MC_OUT/paired_vs_fixed_a15.csv"
