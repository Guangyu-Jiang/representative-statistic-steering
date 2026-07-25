# API-free ITI and adaptive-margin reproduction

This checkout contains local, API-free reproduction paths for Llama-2-chat and
Llama-3-Instruct. The original repository's `judge` and `info` metrics call
fine-tuned OpenAI completion models. Those metric names are disabled here.
MC1/MC2 use the official local TruthfulQA scorer, and generated answers can be
evaluated with local Qwen-2.5-7B, Qwen-2.5-72B, or Mistral-7B judges. The 72B
judge is used for primary open-ended model selection and reporting; the 7B
judges are sensitivity checks.

## Feature extraction

Run the two official activation banks once. The resulting arrays are cached in
`features/`.

```bash
cd get_activations
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python get_activations.py \
  --model_name llama3_8B_instruct --dataset_name tqa_mc2
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python get_activations.py \
  --model_name llama3_8B_instruct --dataset_name tqa_gen_end_q
```

## Official fixed ITI

```bash
cd validation
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python validate_2fold.py \
  --model_name llama3_8B_instruct --num_heads 48 --alpha 15 \
  --num_fold 2 --use_center_of_mass --instruction_prompt default \
  --metrics mc --skip_distribution_metrics
```

`validate_fixed_iti_positions.py` additionally audits the modern last-token
application site against the legacy answer-position rule and the causal
positions that predict answer tokens.

## Adaptive aggregate-margin intervention

The two folds are independent and can run on separate GPUs.

```bash
cd validation
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python validate_margin_perturbation.py \
  --fold 0 --alphas 0.25 0.5 0.75 1 2 --ridge-ratios 0 0.1 1 \
  --output-dir ../artifacts/perturbation_margin_full
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python validate_margin_perturbation.py \
  --fold 1 --alphas 0.25 0.5 0.75 1 2 --ridge-ratios 0 0.1 1 \
  --output-dir ../artifacts/perturbation_margin_full
```

For open-ended evaluation, run `generate_margin_perturbation.py` with a chosen
`--alpha` and `--ridge-ratio`, then evaluate its answer column with
`local_truthfulqa_judge.py`. All runners use cached local checkpoints and do
not make API calls.

The target statistic is controlled with `--target-quantile`. For example, the
best quality-preserving open-ended setting from the local sweep was:

```bash
cd validation
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python generate_margin_perturbation.py \
  --fold 0 --alpha 2 --ridge-ratio 0 --target-quantile 0.75 \
  --output-dir ../artifacts/perturbation_margin_generation_a2_r0_q075
CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python generate_margin_perturbation.py \
  --fold 1 --alpha 2 --ridge-ratio 0 --target-quantile 0.75 \
  --output-dir ../artifacts/perturbation_margin_generation_a2_r0_q075
```

The local judge can evaluate several matched answer columns while loading the
judge model only once:

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python local_truthfulqa_judge.py \
  ../artifacts/perturbation_margin_generation_a2_r0_q075/fold_0_margin_a2_r0.csv \
  ../artifacts/perturbation_margin_generation_a2_r0_q075/fold_1_margin_a2_r0.csv \
  --answer-columns baseline_answer margin_a2_r0_answer \
  --batch-size 16 \
  --output-dir ../artifacts/perturbation_margin_generation_a2_r0_q075/local_judge
```

See `validation/iti_perturbation_results.md` for the completed two-fold
results and the scoring-position compatibility finding.

## Causal attention-head reproduction

`validation/validate_causal_head_perturbation.py` is the paper-aligned MC
runner. It fixes two issues in the older compatibility scripts:

1. It identifies the candidate answer span from tokenizer offset mappings.
   A fixed `log_probs[3:]` happens to work for Llama-2, but removes the first
   semantic answer token for the cached Llama-3 tokenizer.
2. It edits exactly the hidden-state positions whose next-token logits score
   the candidate answer. This is the teacher-forced equivalent of applying ITI
   to the last prompt token and every subsequent decode token.

The runner implements the following intervention families on the same
probe-ranked attention heads:

- `fixed_com`: the original ITI center-of-mass direction with a constant
  `alpha * projection_std` displacement.
- `adaptive_com` and `adaptive_probe`: independent one-sided minimum-norm
  corrections for each selected head.
- `aggregate_com` and `aggregate_probe`: a two-pass causal correction. The
  first pass measures one aggregate statistic across selected heads at every
  causal token; the second pass applies the minimum-norm action needed to move
  that token toward a truthful training quantile.
- `targeted_iti`: uses the same aggregate target as `aggregate_com`, but
  constrains the correction to the original ITI COM/variance-scaled basis.
- `targeted_probe_iti`: uses the standardized aggregate probe margin as the
  target statistic while retaining that same original ITI basis. This isolates
  target-conditioned magnitude control from direction and head selection.
- `bounded_targeted_probe_iti`: adds an optional scalar trust region to the
  probe-targeted solution. Its `coefficient_cap` is expressed in fixed-ITI
  alpha units, so each token-specific action is no larger than a directly
  comparable fixed intervention while still solving toward the probe target.

No OpenAI or other external API is used. The local Llama-2 mirror can be
extracted once with:

```bash
cd get_activations
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  ../.venv/bin/python get_activations.py \
  --model_name llama2_chat_7B --model_path "$MODEL" \
  --dataset_name tqa_mc2 --local_files_only --head_wise_only
CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  ../.venv/bin/python get_activations.py \
  --model_name llama2_chat_7B --model_path "$MODEL" \
  --dataset_name tqa_gen_end_q --local_files_only --head_wise_only
```

Use validation rows for hyperparameter selection and reserve `--eval-split
test` for the final report. A compact validation run is:

```bash
cd validation
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  ../.venv/bin/python validate_causal_head_perturbation.py \
  --model-path "$MODEL" --feature-prefix ../features/llama2_chat_7B \
  --fold 0 --eval-split validation --num-heads 48 \
  --methods fixed_com aggregate_com aggregate_probe targeted_iti \
  --fixed-alphas 10 15 20 \
  --target-quantiles 0.75 0.9 --strengths 2 4 8 16 \
  --ridge-ratios 0 --relative-caps 0.25 0.5 1 \
  --statistics-cache ../artifacts/iti_attention_head/statistics/fold0_k48.npz \
  --output-dir ../artifacts/iti_attention_head/validation_fold0
```

The statistics cache contains the exact head ranking, directions, scales, and
truthful target distribution for a fold. Reuse it for the held-out run to
prevent accidental retraining drift.

The locked fixed-ITI and probe-targeted settings can be evaluated together on
one held-out fold with:

```bash
MODEL="${MODEL_PATH:-NousResearch/Llama-2-7b-chat-hf}"
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python validation/validate_causal_head_perturbation.py \
  --model-path "$MODEL" --feature-prefix features/llama2_chat_7B \
  --fold 0 --eval-split test --num-heads 48 \
  --settings-file validation/iti_generation_settings.json \
  --statistics-cache artifacts/iti_attention_head/statistics/llama2_fold0_k48.npz \
  --output-dir artifacts/iti_attention_head/heldout_fixed_and_probe_fold0
```

Run fold 1 on a second GPU by changing both occurrences of `fold0` to `fold1`
and setting `--fold 1`. The fold-stratified paired interval is then computed
with `validation/compare_paired_mc.py`.

## Online generation and local judging

`validation/generate_causal_head_perturbation.py` applies the same locked
settings during autoregressive generation. It measures the current probe
margin, computes a token-specific scalar in the original ITI basis, applies
the action, and advances only the steered KV cache. `--question-offset` and
`--max-questions` support non-overlapping GPU shards. Merge them only after
all answer columns are complete:

```bash
.venv/bin/python validation/merge_generation_shards.py \
  artifacts/iti_attention_head/generation_heldout_k48/fold_0_test_generations.csv \
  artifacts/iti_attention_head/generation_heldout_k48/fold0_tail/fold_0_test_generations.csv \
  --expected-rows 409 \
  --output artifacts/iti_attention_head/generation_heldout_k48/merged/fold_0_test_generations.csv
```

Judge baseline, fixed ITI, and probe-targeted answers in one local-model load:

```bash
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python validation/local_truthfulqa_judge.py \
  artifacts/iti_attention_head/generation_heldout_k48/merged/fold_0_test_generations.csv \
  artifacts/iti_attention_head/generation_heldout_k48/merged/fold_1_test_generations.csv \
  --answer-columns baseline_answer fixed_com_k48_a15_answer \
    targeted_probe_iti_k48_a16_q0p75_r0_c4_answer \
  --batch-size 16 \
  --checkpoint-every 10 \
  --output-dir artifacts/iti_attention_head/generation_heldout_k48/local_qwen_judge
```

Judge outputs are checkpointed and the identical command resumes only rows
whose raw judge output is still empty. For a model sharded over several GPUs,
`--gpu-memory-gib` accepts one Accelerate memory ceiling per visible GPU; this
is useful when another process occupies part of one card and also reserves
space for generation activations.

The local judge is an API-free proxy and is not numerically interchangeable
with the fine-tuned GPT judges used by the original ITI report.

The summary reports two deliberately different composite metrics:

- `truth_x_info` is the paper-compatible score. It averages Truth and Info
  across folds separately and then multiplies those two marginal rates, as in
  `legacy/llama_validate_2fold.py`.
- `joint_truth_info` is the stricter fraction of individual answers receiving
  both positive labels. It is useful diagnostically but is not the score named
  `True*Info` in the released ITI evaluator.

Existing checkpoint files can be summarized after a metric-code change without
loading the judge model again:

```bash
.venv/bin/python validation/summarize_local_judge_outputs.py \
  artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge/*_judged.csv \
  --judge-model Qwen/Qwen2.5-72B-Instruct \
  --output artifacts/iti_attention_head/generation_selected_probe_a12_k48/local_qwen72b_judge/local_qwen_judge_summary.csv
```

Audit response validity, diversity, exact answer changes, and intervention
diagnostics independently of the judge:

```bash
.venv/bin/python validation/audit_generation_outputs.py \
  artifacts/iti_attention_head/generation_selected_probe_a12_k48/merged/fold_0_test_generations.csv \
  artifacts/iti_attention_head/generation_selected_probe_a12_k48/merged/fold_1_test_generations.csv \
  --answer-columns baseline_answer fixed_com_k48_a15_answer \
    targeted_probe_iti_k48_a12_q0p75_r0_c2_answer \
  --output artifacts/iti_attention_head/generation_selected_probe_a12_k48/generation_audit.csv
```

## Validation-selected bounded comparison

The final tuned comparison keeps head selection and direction construction
identical between methods. It first evaluates fixed ITI strengths on the 164
inner-validation questions, then evaluates coefficient bounds for the
probe-targeted controller, selects the bound using validation `True*Info`, and
only then generates and judges the 817 outer-fold test questions:

```bash
bash scripts/run_iti_fixed_validation_sweep_after_multiseed.sh
bash scripts/run_iti_bounded_probe_validation_after_fixed.sh
bash scripts/run_iti_final_heldout_after_bounded.sh
bash scripts/run_iti_final_bounded_mc_after_judge.sh
```

The final script writes the selected setting to
`validation/iti_generation_final_validation_selected_settings.json`. It uses
four non-overlapping generation shards per fold, requires exactly 409 and 408
complete unique rows during merging, and judges both methods in one local
Qwen-2.5-72B model load. No external API is called. The authoritative outputs
are:

- `artifacts/iti_attention_head/generation_validation_fixed_sweep_k48/local_qwen72b_judge/validation_method_comparison.csv`
- `artifacts/iti_attention_head/generation_validation_bounded_probe_k48/local_qwen72b_judge/validation_controller_comparison.csv`
- `artifacts/iti_attention_head/generation_validation_bounded_probe_k48/local_qwen72b_judge/bounded_validation_ranking.csv`
- `artifacts/iti_attention_head/generation_final_validation_selected_k48/generation_audit.csv`
- `artifacts/iti_attention_head/generation_final_validation_selected_k48/local_qwen72b_judge/heldout_method_comparison.csv`
- `artifacts/iti_attention_head/generation_final_validation_selected_k48/local_qwen72b_judge/bounded_vs_fixed_a8_paired.csv`
- `artifacts/iti_attention_head/heldout_final_bounded_k48/combined_test_summary.csv`
- `artifacts/iti_attention_head/heldout_final_bounded_k48/paired_vs_fixed_a8.csv`
- `artifacts/iti_attention_head/heldout_final_bounded_k48/paired_vs_fixed_a15.csv`

`bounded_vs_fixed_a8_paired.csv` contains fold-balanced paired bootstrap
intervals over the same 817 questions. In particular, it compares adaptive
magnitude selection against a validation-tuned fixed intervention rather than
only against the paper-prescribed alpha-15 setting.
