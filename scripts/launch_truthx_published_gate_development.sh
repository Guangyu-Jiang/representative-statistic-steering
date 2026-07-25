#!/usr/bin/env bash
set -euo pipefail

gpu="${TRUTHX_GPU:-2}"
root="artifacts/truthx_mc/corrected_published_gate_development"
expected_count=64

is_complete() {
  local output="$1"
  local results="${output}/results.jsonl"
  [[ -f "${results}" ]] && [[ "$(wc -l < "${results}")" -ge "${expected_count}" ]]
}

for cap in 6.0 8.0; do
  cap_tag="${cap/./p}"
  output="${root}/decoder_t0p25_gate0_r0_d1_cap${cap_tag}"
  if is_complete "${output}"; then
    echo "Skipping complete TruthX gate setting: ${output}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
    --output-dir "${output}" --method minimum_norm \
    --offset 0 --limit 64 \
    --target-mode cosine_margin_decoder \
    --target-strength 0.25 --intervention-margin-threshold 0 \
    --ridge 0 --optimization-steps 10 --learning-rate 1.0 \
    --maximum-relative-norm "${cap}" \
    --model-dtype bfloat16 --device cuda:0
done

output="${root}/decoder_t0p25_gatem0p25_r0_d1_cap8p0"
if is_complete "${output}"; then
  echo "Skipping complete TruthX gate setting: ${output}"
  exit 0
fi
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_truthx_mc.py \
  --output-dir "${output}" --method minimum_norm \
  --offset 0 --limit 64 \
  --target-mode cosine_margin_decoder \
  --target-strength 0.25 --intervention-margin-threshold -0.25 \
  --ridge 0 --optimization-steps 10 --learning-rate 1.0 \
  --maximum-relative-norm 8.0 \
  --model-dtype bfloat16 --device cuda:0
