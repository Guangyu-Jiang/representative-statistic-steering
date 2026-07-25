#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
output="artifacts/lookback_nq/development_n60_rerank_refinement/shift3_cap0p5"
results="${output}/results.jsonl"
if [[ -f "${results}" ]] && [[ "$(wc -l < "${results}")" -ge 60 ]]; then
  echo "Skipping complete Lookback refinement setting: ${output}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir "${output}" \
  --methods minimum_norm_rerank \
  --offset 0 --limit 60 --max-new-tokens 64 \
  --num-candidates 4 \
  --target-mode relative --target-logit-shift 3 \
  --ridge 0 --solver-steps 16 --solver-damping 0.5 \
  --maximum-bias-rms 0.5 \
  --context-bias-mode question_overlap --context-overlap-radius 8 \
  --active-control-count 128 --bias-constraint unrestricted \
  --temperature 0.9 --top-p 0.95
