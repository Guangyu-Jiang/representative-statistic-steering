#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
root="artifacts/lookback_nq/development_n60_rerank_refinement"
expected_count=60
settings=(
  "1 0.25"
  "2 0.25"
  "2 0.5"
  "3 0.5"
)

for setting in "${settings[@]}"; do
  read -r shift cap <<< "${setting}"
  shift_tag="${shift/./p}"
  cap_tag="${cap/./p}"
  output="${root}/shift${shift_tag}_cap${cap_tag}"
  results="${output}/results.jsonl"
  if [[ -f "${results}" ]] && [[ "$(wc -l < "${results}")" -ge "${expected_count}" ]]; then
    echo "Skipping complete Lookback refinement setting: ${output}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
    --output-dir "${output}" \
    --methods minimum_norm_rerank \
    --offset 0 --limit 60 --max-new-tokens 64 \
    --num-candidates 4 \
    --target-mode relative --target-logit-shift "${shift}" \
    --ridge 0 --solver-steps 16 --solver-damping 0.5 \
    --maximum-bias-rms "${cap}" \
    --context-bias-mode question_overlap --context-overlap-radius 8 \
    --active-control-count 128 --bias-constraint unrestricted \
    --temperature 0.9 --top-p 0.95
done
