#!/usr/bin/env bash
set -euo pipefail

wait_pid="${WAIT_PID:?WAIT_PID is required}"
while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 15
done

selection="artifacts/reports/lookback_rerank_refinement_selection.json"
shift="$(jq -r '.selected.target_logit_shift' "${selection}")"
cap="$(jq -r '.selected.maximum_bias_rms_config' "${selection}")"
shift_tag="${shift/./p}"
cap_tag="${cap/./p}"
output="artifacts/lookback_nq/refined_confirmation_offset260_n100/shift${shift_tag}_cap${cap_tag}"
results="${output}/results.jsonl"

if [[ ! -f "${results}" ]] || [[ "$(wc -l < "${results}")" -ne 300 ]]; then
  echo "Expected 300 confirmation rows in ${results}" >&2
  exit 1
fi

selected_development="$(jq -r '.selected.run' "${selection}")/results.jsonl"
PYTHONPATH=src python scripts/train_evaluate_lookback_candidate_ranker.py \
  --development-controlled "${selected_development}" \
  --validation "${results}" \
  --validation-rerank-baseline "${results}" \
  --output-dir artifacts/lookback_nq/candidate_ranker_refined_confirmation \
  --require-complete

PYTHONPATH=src python scripts/build_lookback_refinement_confirmation.py \
  --require-complete
PYTHONPATH=src python scripts/audit_final_splits.py --require-complete
pytest -q
