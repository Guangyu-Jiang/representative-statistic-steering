#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
SETTINGS="validation/iti_generation_bounded_probe_validation_settings.json"
SOURCE="artifacts/iti_attention_head/generation_validation_sweep_k48"
FIXED="artifacts/iti_attention_head/generation_validation_fixed_sweep_k48"
OUT="artifacts/iti_attention_head/generation_validation_bounded_probe_k48"
JUDGE="$OUT/local_qwen72b_judge"
LOGS="artifacts/iti_attention_head/logs"
mkdir -p "$OUT" "$JUDGE" "$LOGS"

while pgrep -f '^bash scripts/run_iti_fixed_validation_sweep_after_multiseed\.sh$' >/dev/null; do
  sleep 30
done
sleep 15

run_shard() {
  local gpu="$1"
  local fold="$2"
  local offset="$3"
  local count="$4"
  local source="$SOURCE/merged/fold_${fold}_validation_generations.csv"
  local tag="fold${fold}_offset${offset}"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/generate_causal_head_perturbation.py \
      --model-path "$MODEL" --settings-file "$SETTINGS" \
      --eval-split validation --fold "$fold" \
      --question-offset "$offset" --max-questions "$count" \
      --max-new-tokens 50 --checkpoint-every 2 --reuse-from "$source" \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/$tag" \
      >"$LOGS/generation_validation_bounded_probe_${tag}.log" 2>&1
}

run_shard 0 0  0 41 & p00=$!
run_shard 1 0 41 41 & p01=$!
run_shard 2 1  0 41 & p10=$!
run_shard 3 1 41 41 & p11=$!
wait "$p00" "$p01" "$p10" "$p11"

mkdir -p "$OUT/merged"
for fold in 0 1; do
  "$PYTHON_BIN" validation/merge_generation_shards.py \
    "$OUT"/fold${fold}_offset*/fold_${fold}_validation_generations.csv \
    --expected-rows 82 \
    --enrich-from "$SOURCE/merged/fold_${fold}_validation_generations.csv" \
    --output "$OUT/merged/fold_${fold}_validation_generations.csv"
done

answer_columns=()
for cap in 4 6 8 10 15; do
  answer_columns+=("bounded_targeted_probe_iti_k48_a12_q0p75_r0_c2_b${cap}_answer")
done

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$OUT/merged/fold_0_validation_generations.csv" \
    "$OUT/merged/fold_1_validation_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size 8 --max-new-tokens 24 --checkpoint-every 2 \
    --gpu-memory-gib 44 34 44 44 --output-dir "$JUDGE" \
    --answer-columns "${answer_columns[@]}" \
    >"$LOGS/generation_validation_bounded_probe_qwen72b_judge.log" 2>&1

PRIMARY="$SOURCE/local_qwen72b_judge"
FIXED_JUDGE="$FIXED/local_qwen72b_judge"
"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE"/*__qwen_qwen2_5_72b_instruct_judged.csv \
  "$FIXED_JUDGE"/*__qwen_qwen2_5_72b_instruct_judged.csv \
  "$PRIMARY"/fold_0_validation_generations__fixed_com_k48_a15_answer__qwen_qwen2_5_72b_instruct_judged.csv \
  "$PRIMARY"/fold_1_validation_generations__fixed_com_k48_a15_answer__qwen_qwen2_5_72b_instruct_judged.csv \
  "$PRIMARY"/fold_0_validation_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__qwen_qwen2_5_72b_instruct_judged.csv \
  "$PRIMARY"/fold_1_validation_generations__targeted_probe_iti_k48_a12_q0p75_r0_c2_answer__qwen_qwen2_5_72b_instruct_judged.csv \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output "$JUDGE/validation_controller_comparison.csv"
