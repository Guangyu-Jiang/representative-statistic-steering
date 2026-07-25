#!/usr/bin/env bash
set -euo pipefail

gpu="${LOOKBACK_GPU:-2}"
wait_pids="${WAIT_PIDS:-}"

for wait_pid in ${wait_pids}; do
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 15
  done
done

PYTHONPATH=src python scripts/select_lookback_rerank_refinement.py \
  --require-complete

selection="artifacts/reports/lookback_rerank_refinement_selection.json"
shift="$(jq -r '.selected.target_logit_shift' "${selection}")"
cap="$(jq -r '.selected.maximum_bias_rms_config' "${selection}")"
shift_tag="${shift/./p}"
cap_tag="${cap/./p}"
root="artifacts/lookback_nq/refined_confirmation_offset260_n100"
output="${root}/shift${shift_tag}_cap${cap_tag}"
shard_a="${root}/shards/shift${shift_tag}_cap${cap_tag}_offset260_n50"
shard_b="${root}/shards/shift${shift_tag}_cap${cap_tag}_offset310_n50"

run_shard() {
  local offset="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python scripts/run_lookback_nq.py \
    --output-dir "${shard}" \
    --methods baseline baseline_rerank minimum_norm_rerank \
    --offset "${offset}" --limit 50 --max-new-tokens 64 \
    --num-candidates 4 \
    --target-mode relative --target-logit-shift "${shift}" \
    --ridge 0 --solver-steps 16 --solver-damping 0.5 \
    --maximum-bias-rms "${cap}" \
    --context-bias-mode question_overlap --context-overlap-radius 8 \
    --active-control-count 128 --bias-constraint unrestricted \
    --temperature 0.9 --top-p 0.95
}

run_shard 260 "${shard_a}" &
pid_a=$!
run_shard 310 "${shard_b}" &
pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

PYTHONPATH=src python scripts/merge_lookback_shards.py \
  --shard-dir "${shard_a}" "${shard_b}" \
  --output-dir "${output}" \
  --expected-questions 100

selected_development="$(jq -r '.selected.run' "${selection}")/results.jsonl"
PYTHONPATH=src python scripts/train_evaluate_lookback_candidate_ranker.py \
  --development-controlled "${selected_development}" \
  --validation "${output}/results.jsonl" \
  --validation-rerank-baseline "${output}/results.jsonl" \
  --output-dir artifacts/lookback_nq/candidate_ranker_refined_confirmation \
  --require-complete

PYTHONPATH=src python scripts/build_lookback_refinement_confirmation.py \
  --require-complete
PYTHONPATH=src python scripts/audit_final_splits.py --require-complete
pytest -q
