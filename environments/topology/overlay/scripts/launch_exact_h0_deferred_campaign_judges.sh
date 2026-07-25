#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
memory_limit_mb="${MEMORY_LIMIT_MB:-20000}"
log_dir="artifacts/logs/launch_exact_h0_deferred_campaign_judges"
mkdir -p "$log_dir"
exec >>"$log_dir/gpu${gpu}.log" 2>&1
printf '%s\n' "$$" >"$log_dir/gpu${gpu}.pid"

pid_files=(
  artifacts/logs/launch_exact_h0_classifier_target_ablation/gpu0.pid
  artifacts/logs/launch_exact_h0_contrastive_pair_pilot/gpu3.pid
  artifacts/logs/launch_exact_h0_normalized_crossdataset_llama/gpu3.pid
  artifacts/logs/launch_exact_h0_normalized_crossdataset_gemma/gpu2.pid
  artifacts/logs/launch_exact_h0_normalized_crossdataset_mistral/gpu2.pid
)
for pid_file in "${pid_files[@]}"; do
  pid="$(cat "$pid_file")"
  printf '%s waiting pid=%s source=%s\n' "$(date -Is)" "$pid" "$pid_file"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
done

while true; do
  memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( memory_used <= memory_limit_mb )); then
    break
  fi
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="${PYTHONPATH:-}:src"
roots=(
  artifacts/steering_exact_h0_contrastive_pair_pilot
  artifacts/steering_exact_h0_normalized_crossdataset_llama
  artifacts/steering_exact_h0_normalized_crossdataset_gemma
  artifacts/steering_exact_h0_normalized_crossdataset_mistral
  artifacts/steering_exact_h0_crossdataset_rule_pilot_llama
  artifacts/steering_exact_h0_crossdataset_rule_pilot_gemma
  artifacts/steering_exact_h0_crossdataset_rule_pilot_mistral
)
for root in "${roots[@]}"; do
  printf '%s judging root=%s\n' "$(date -Is)" "$root"
  python scripts/judge_exact_h0_gn_local.py --artifact-root "$root" --batch-size 32
done

python scripts/report_exact_h0_perturbation_campaign.py
