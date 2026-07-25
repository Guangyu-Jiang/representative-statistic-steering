#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU WAIT_PID}"
wait_pid="${2:?usage: $0 GPU WAIT_PID}"
memory_limit_mb="${MEMORY_LIMIT_MB:-20000}"
log_dir="artifacts/logs/exact_h0_lowrank_position_weighted"
mkdir -p "$log_dir"
exec >>"$log_dir/pilot_gpu${gpu}.log" 2>&1
printf '%s\n' "$$" >"$log_dir/pilot_gpu${gpu}.pid"
printf '%s queued gpu=%s wait_pid=%s memory_limit_mb=%s\n' \
  "$(date -Is)" "$gpu" "$wait_pid" "$memory_limit_mb"

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done

while true; do
  memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( memory_used <= memory_limit_mb )); then
    break
  fi
  sleep 30
done

exec bash scripts/launch_exact_h0_lowrank_position_weighted_pilot.sh "$gpu"
