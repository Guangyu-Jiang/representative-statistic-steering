#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
SETTINGS="validation/iti_generation_alpha_rho_sweep_settings.json"
OUT="artifacts/iti_attention_head/generation_validation_alpha_rho_sweep_k48"
SOURCE="artifacts/iti_attention_head/generation_validation_bounded_probe_k48"
LOGS="artifacts/iti_attention_head/logs"
ALL_COLUMNS="$OUT/all_answer_columns.txt"
JUDGE7="$OUT/local_qwen7b_judge"
JUDGE72="$OUT/local_qwen72b_judge"
mkdir -p "$OUT" "$LOGS" "$JUDGE7" "$JUDGE72"

"$PYTHON_BIN" validation/build_iti_alpha_rho_sweep.py \
  --output "$SETTINGS" --columns-output "$ALL_COLUMNS"

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
      --model-path "$MODEL" --settings-file "$SETTINGS" \
      --eval-split validation --fold "$fold" \
      --question-offset "$offset" --max-questions "$count" \
      --max-new-tokens 50 --checkpoint-every 1 \
      --reuse-from "$SOURCE/merged/fold_${fold}_validation_generations.csv" \
      --statistics-cache "artifacts/iti_attention_head/statistics/llama2_fold${fold}_k48.npz" \
      --output-dir "$OUT/$tag" \
      >"$LOGS/generation_alpha_rho_${tag}.log" 2>&1
}

run_shard 0 0  0 21 & p00=$!
run_shard 1 0 21 21 & p01=$!
run_shard 2 0 42 20 & p02=$!
run_shard 3 0 62 20 & p03=$!
run_shard 0 1  0 21 & p10=$!
run_shard 1 1 21 21 & p11=$!
run_shard 2 1 42 20 & p12=$!
run_shard 3 1 62 20 & p13=$!
wait "$p00" "$p01" "$p02" "$p03" "$p10" "$p11" "$p12" "$p13"

mkdir -p "$OUT/merged"
for fold in 0 1; do
  "$PYTHON_BIN" validation/merge_generation_shards.py \
    "$OUT"/fold${fold}_offset*/fold_${fold}_validation_generations.csv \
    --expected-rows 82 \
    --enrich-from "$SOURCE/merged/fold_${fold}_validation_generations.csv" \
    --output "$OUT/merged/fold_${fold}_validation_generations.csv"
done

mapfile -t all_columns < "$ALL_COLUMNS"
declare -a columns0=() columns1=() columns2=() columns3=()
for index in "${!all_columns[@]}"; do
  case $((index % 4)) in
    0) columns0+=("${all_columns[index]}") ;;
    1) columns1+=("${all_columns[index]}") ;;
    2) columns2+=("${all_columns[index]}") ;;
    3) columns3+=("${all_columns[index]}") ;;
  esac
done

run_qwen7() {
  local gpu="$1"
  shift
  env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
      "$OUT/merged/fold_0_validation_generations.csv" \
      "$OUT/merged/fold_1_validation_generations.csv" \
      --judge-model Qwen/Qwen2.5-7B-Instruct \
      --batch-size 24 --max-new-tokens 24 --checkpoint-every 2 \
      --output-dir "$JUDGE7" --answer-columns "$@" \
      >"$LOGS/alpha_rho_qwen7_gpu${gpu}.log" 2>&1
}

run_qwen7 0 "${columns0[@]}" & j0=$!
run_qwen7 1 "${columns1[@]}" & j1=$!
run_qwen7 2 "${columns2[@]}" & j2=$!
run_qwen7 3 "${columns3[@]}" & j3=$!
wait "$j0" "$j1" "$j2" "$j3"

"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE7"/*__qwen_qwen2_5_7b_instruct_judged.csv \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --output "$JUDGE7/alpha_rho_summary.csv"

CURRENT_TAG="bounded_targeted_probe_iti_k48_a12_q0p75_r0_c2_b10"
"$PYTHON_BIN" validation/select_iti_alpha_rho_candidates.py \
  --summary "$JUDGE7/alpha_rho_summary.csv" --settings "$SETTINGS" \
  --top-product 8 --top-joint 8 --best-per-alpha \
  --force-tag "$CURRENT_TAG" \
  --ranking-output "$JUDGE7/alpha_rho_ranking.csv" \
  --columns-output "$OUT/qwen72_answer_columns.txt" \
  --selected-settings-output "$OUT/qwen72_selected_settings.json" \
  >"$LOGS/alpha_rho_candidate_selection.log" 2>&1

mapfile -t qwen72_columns < "$OUT/qwen72_answer_columns.txt"
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" validation/local_truthfulqa_judge.py \
    "$OUT/merged/fold_0_validation_generations.csv" \
    "$OUT/merged/fold_1_validation_generations.csv" \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --batch-size 8 --max-new-tokens 24 --checkpoint-every 2 \
    --gpu-memory-gib 44 34 44 44 --output-dir "$JUDGE72" \
    --answer-columns "${qwen72_columns[@]}" \
    >"$LOGS/alpha_rho_qwen72.log" 2>&1

fixed=(
  artifacts/iti_attention_head/generation_validation_fixed_sweep_k48/local_qwen72b_judge/*__fixed_com_k48_a8_answer__qwen_qwen2_5_72b_instruct_judged.csv
)
"$PYTHON_BIN" validation/summarize_local_judge_outputs.py \
  "$JUDGE72"/*__qwen_qwen2_5_72b_instruct_judged.csv "${fixed[@]}" \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output "$JUDGE72/alpha_rho_with_fixed_a8_summary.csv"

"$PYTHON_BIN" validation/select_iti_alpha_rho_candidates.py \
  --summary "$JUDGE72/alpha_rho_with_fixed_a8_summary.csv" --settings "$SETTINGS" \
  --allow-subset --top-product 8 --top-joint 8 --best-per-alpha \
  --force-tag "$CURRENT_TAG" \
  --ranking-output "$JUDGE72/alpha_rho_ranking.csv" \
  --columns-output "$JUDGE72/final_candidate_columns.txt" \
  --selected-settings-output "$JUDGE72/final_candidate_settings.json" \
  >"$LOGS/alpha_rho_final_ranking.log" 2>&1
