# Representative-Statistic Minimum-Norm Steering

This workspace tests target-conditioned minimum-norm activation steering on four
published representative statistics with differentiable or locally estimable
control maps:

- PPLM's differentiable SST sentiment-classifier score over mean-pooled GPT-2
  hidden states.
- TruthX's truthful-latent cosine margin produced by its published autoencoder.
- Lookback Lens's rolling context-attention ratios and published factuality
  classifier.
- ReDeEP's external-context similarity (ECS) and parametric-knowledge shift
  (PKS) detector state.

The official repositories are kept unchanged under `external/`. Experiments do
not call an external API.

The evaluated snapshot uses:

- `external/PPLM` at commit `e236b898`.
- `external/TruthX` at commit `a41093a6`.
- `external/Lookback-Lens` at commit `e0a1fa3`.
- `external/ReDEeP-ICLR` at commit `4d081915`.
- Local GPT-2-medium, Llama-2-7B-chat, DistilBERT-SST2, PPLM SST-head, and
two-fold TruthX checkpoints under `checkpoints/` or the Hugging Face cache.
Lookback Lens uses the local Llama-2-7B-chat checkpoint and the official
Natural Questions file and classifier distributed by its repository.
ReDeEP uses the local Llama-2-7B-chat checkpoint and its released Dolly and
RAGTruth artifacts.

No OpenAI or other hosted-model API is used for generation or evaluation.

## Compared methods

PPLM compares an uncontrolled model, the original normalized-gradient update,
and an iterative scalar Gauss-Newton minimum-norm cache perturbation. Its scalar
statistic is the target SST class logit minus the log-sum-exp of all other class
logits. In the paper-aligned protocol, the perturbed key/value cache is carried
through later decode steps instead of being reconstructed from the sampled text.

The quality-aware inverse augments the behavioral Jacobian with selected
next-token log-probability Jacobians:

```text
[J_z; sqrt(beta) J_logp,top-K] delta ~= [z_target - z; 0].
```

This moves the sentiment statistic while locally preserving the unperturbed
model's highest-probability output coordinates. Cache blocks use gradient-norm
weighting before the regularized Gauss-Newton solve.

All current inverse runs use solver version `accumulated_v2`. At each
re-linearization, the ridge term penalizes the final accumulated perturbation,
not a newly added step. Artifacts that predate this correction are retained for
auditability and are explicitly treated as legacy results.

The robust adaptive variant targets an absolute classifier probability of
`0.95` and allocates its output-distribution budget from the current
unperturbed target margin `m_t`:

```text
epsilon_t = 2.0 if m_t < -4.0 else 1.0.
```

The geometric mixture uses the largest scale up to `0.95` whose token-level KL
does not exceed `epsilon_t`. Thus difficult decode states receive stronger
control while ordinary states retain the lower-KL continuation distribution.

TruthX compares an uncontrolled model, the published decoder-mapped latent edit,
and three inverse mappings: unrestricted hidden-state minimum norm, nonlinear
full-latent matching, and scalar minimum norm constrained to TruthX's decoded
edit direction. The scalar statistic is

```text
cos(z_truth(H), positive_center) - cos(z_truth(H), negative_center).
```

Attention interventions are applied to the input of `self_attn.o_proj`, exactly
matching the official pre-output-projection site. MLP interventions are applied
to the MLP output before the residual addition. TruthX uses the official top-10
module ranking and two-fold checkpoints.

Lookback Lens compares greedy decoding, the paper's candidate-reranking
controller, and direct minimum-norm control of its representative statistic.
For each layer and attention head, a scalar bias is added to the logits of
context-key positions for the current decode query. If the unmodified lookback
ratio is `r_lh` and the bias is `b_lh`, the controlled ratio obeys the exact
identity

```text
logit(r'_lh) = logit(r_lh) + b_lh.
```

Consequently, no learned surrogate is required. An iterative regularized
Gauss-Newton solve finds the smallest 32-by-32 bias tensor that reaches either
an absolute classifier probability or a relative classifier-logit shift. The
modified attention output and its key/value cache are passed to subsequent
layers and decode steps. Evaluation uses the official local Lookback Lens
classifier and Natural Questions exact match; no hosted judge or API is called.
Writing the published logistic-classifier logit as `s(.)`, the per-token inverse
is

```text
b* = argmin_b (s(rolling_mean(sigmoid(logit(r) + b))) - s_target)^2
                 + lambda ||b||_2^2.
```

Each local rank-one update solves the accumulated-bias normal equation, and a
backtracking line search guarantees that this objective does not increase.

ReDeEP represents each decode state with two normalized coordinates,
`z=[PKS,ECS]`. PKS is the released symmetric vocabulary-distribution divergence
between the residual stream immediately before and after selected Knowledge
FFNs. ECS is the mean cosine similarity between the current final hidden state
and prompt-token states selected by published Copying Heads. The detector is
linear in this state:

```text
s(z) = normalize(PKS - weight * ECS).
```

The published AARF baseline scales four Copying Heads by `1.2` and three
Knowledge FFNs by `0.8`. The target-conditioned variant exposes these same
seven mechanism coordinates, projects `z` toward a lower detector score, and
solves the regularized local inverse

```text
c* = J^T (J J^T + lambda I)^-1 (z_target - z),
```

where `J` is a finite-difference Jacobian of `[PKS,ECS]` with respect to the
seven controls. The initial comparison refreshes `J` at every decode token,
changes only the current decode position, and leaves prefill unchanged for all
methods. This isolates online control from prompt-cache modification.

## Setup and tests

```bash
python -m pip install -e .
pytest -q
```

## PPLM sentiment pilot

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/run_pplm_sentiment.py \
  --output-dir artifacts/pplm_sentiment/pilot \
  --methods baseline pplm minimum_norm \
  --targets positive negative
```

Run the reproducible tuning grid with:

```bash
scripts/launch_pplm_tuning_grid.sh
```

Run the paper-aligned persistent-cache comparison and build its report with:

```bash
scripts/launch_pplm_persistent_tuning.sh
scripts/launch_pplm_persistent_output_metric_tuning.sh
scripts/launch_pplm_persistent_reference_heldout_shards.sh
scripts/launch_pplm_persistent_candidate_v2_heldout_shards.sh
scripts/launch_pplm_persistent_negative_gm045_heldout_shards.sh
scripts/launch_pplm_persistent_independent_validation.sh
scripts/launch_pplm_persistent_absolute_target_tuning.sh
scripts/launch_pplm_persistent_adaptive_kl_tuning_v2.sh
scripts/launch_pplm_persistent_adaptive_policy_validation_v2.sh
scripts/launch_pplm_persistent_adaptive_policy_validation_v3.sh
python scripts/build_pplm_improvement_report.py
```

Run the corrected accumulated-action development and disjoint validations with:

```bash
scripts/launch_pplm_corrected_refinement.sh
scripts/launch_pplm_corrected_validation.sh
scripts/launch_pplm_corrected_independent_prefixes.sh
scripts/launch_pplm_corrected_relative_targets.sh
scripts/launch_pplm_corrected_shift3_mechanism_development.sh
scripts/launch_pplm_corrected_shift3_validation.sh
scripts/launch_pplm_corrected_shift_sensitivity_validation.sh
scripts/launch_pplm_corrected_output_preservation_development.sh
scripts/launch_pplm_corrected_output_preservation_validation.sh
python scripts/build_pplm_corrected_report.py
```

Calibrate absolute sentiment-margin targets from the SST-5 training split,
select a quantile on the five-prefix development split, and evaluate the frozen
choice on the disjoint 60-case validation with:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python \
  scripts/calibrate_pplm_quantile_targets.py \
  --output-dir artifacts/pplm_sentiment/quantile_calibration_sst5 \
  --quantiles 0.5 0.7 0.75 0.8 0.9 \
  --device cuda:0
PPLM_GPU=2 scripts/launch_pplm_quantile_development.sh
PPLM_GPU=2 TARGET_QUANTILE=0.5 scripts/launch_pplm_quantile_validation.sh
PYTHONPATH=src python scripts/build_pplm_quantile_report.py \
  --candidate artifacts/pplm_sentiment/quantile_target_validation_q0p5_seeds22_33/external_eval/evaluated_generations.csv
```

For target class `y`, calibration scores complete SST-5 training sentences with
the same mean-pooled GPT-2/SST classifier used by the controller and stores

```text
z_target(y, q) = Quantile_q({margin_y(x_i) : label(x_i) = y}).
```

Only the very-positive and very-negative SST-5 classes are used, giving 1,288
positive and 1,092 negative calibration sentences. During decoding, the
one-sided inverse acts only when the current margin is below this absolute
target. The calibration split contains no generation prefixes or held-out
validation outputs.

The final persistent minimum-norm setting uses a relative target-margin shift of
`1.5`, cache-norm cap `0.03`, ridge `0.1`, three local Gauss-Newton steps,
gradient-block exponent `0.5`, top-2 log-probability preservation with weight
`0.01`, and geometric-mixture weight `0.35`. A separately reported calibrated
variant uses mixture weight `0.45` for the harder negative target while retaining
`0.35` for the positive target.

The later adaptive setting uses target probability `0.95`, ordinary/hard token
KL budgets `1.0/2.0`, difficulty threshold `-4.0`, maximum mixture scale `0.95`,
and otherwise retains the same three-step inverse, ridge, cache cap, block
weighting, and top-2 preservation configuration.

The selection split contains five prefixes and the held-out split contains ten.
After target calibration, a second independent validation used ten previously
unused prefixes: `The computer`, `The ocean`, `The school`, `The house`,
`The phone`, `The garden`, `The hospital`, `The train`, `The music`, and
`The meeting`. Every split uses three fixed seeds, two targets, and 24 generated
tokens. The adaptive policy was developed on a separate 20-prefix set and then
frozen before evaluation on a third disjoint 20-prefix set. Generation and
evaluation are fully local.

## TruthX TruthfulQA multiple choice

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_truthx_mc.py \
  --method minimum_norm \
  --target-mode cosine_margin_decoder \
  --target-strength 0.25 \
  --ridge 0 \
  --optimization-steps 10 \
  --learning-rate 1 \
  --maximum-relative-norm 4 \
  --limit 100 \
  --output-dir artifacts/truthx_mc/minimum_norm_decoder
```

Each run stores raw per-generation or per-question records and a compact
summary, including statistic-target error and relative activation change.
TruthX uses BF16 by default to avoid FP16 overflow at the paper's large editing
strength; `valid_scores` records whether every candidate score remained finite.

Pilot settings and evaluation settings are stored in separate artifact
directories. Directories prefixed with `preproj_` use the aligned official
attention site. Earlier exploratory directories without that prefix are retained
for auditability but must not be used in final TruthX comparisons.

The corrected accumulated-action protocol and its disjoint validation can be
reproduced with:

```bash
scripts/launch_truthx_corrected_refinement.sh
scripts/launch_truthx_corrected_heldout.sh
scripts/launch_truthx_corrected_strong_development.sh
scripts/launch_truthx_corrected_cap4_validation.sh
scripts/launch_truthx_corrected_cap4_t0p1_validation.sh
scripts/launch_truthx_linesearch_development.sh
scripts/launch_truthx_linesearch_target_extension.sh
scripts/launch_truthx_direction_constrained_development.sh
scripts/launch_truthx_corrected_strength_extension.sh
scripts/launch_truthx_cap6_t0p25_untouched_confirmation.sh
python scripts/build_truthx_corrected_report.py
```

## Lookback Lens Natural Questions

Run a paper-style local comparison with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/run_lookback_nq.py \
  --output-dir artifacts/lookback_nq/paper_comparison \
  --methods baseline guided minimum_norm \
  --limit 100 \
  --target-mode relative \
  --target-logit-shift 2 \
  --ridge 0.1 \
  --maximum-bias-rms 0.5
```

Aggregate all completed Lookback Lens runs with:

```bash
PYTHONPATH=src python scripts/build_lookback_report.py
```

The frozen held-out comparison and answer-blind replay reranker are launched by
`scripts/launch_lookback_heldout_methods.sh`,
`scripts/launch_lookback_sampled_control.sh`, and
`scripts/launch_lookback_minimum_norm_rerank_development.sh`. The frozen replay
reranker is confirmed on new indices 160--259 with
`scripts/launch_lookback_minimum_norm_rerank_untouched_validation.sh`; its saved
candidate-ranking statistics are audited with
`scripts/analyze_lookback_candidate_ranking.py`.

The focused-control launchers include a matched random-support control,
question-overlap support, sparse head control, answer-blind BM25 passage
retrieval, and disjoint 256-token held-out runs. The report pairs greedy direct
control with the greedy baseline and the paper's sampled candidate reranker
with a single-candidate sampled baseline.

Run `python scripts/audit_final_splits.py --require-complete` after all final
jobs. It verifies that the Lookback, PPLM, and TruthX development and final
confirmation units are complete and disjoint, and writes the machine-readable
audit to `artifacts/reports/final_split_audit.json`.

Each run stores the question, gold short answers, generated response, exact
match, classifier probabilities, target error, attention-bias norm, attention
KL, and output-token KL. Runs are resumable at the method/example level.

## ReDeEP RAG hallucination

Reproduce the released Llama-2-7B RAGTruth detector AUC and a leakage-free
response-grouped audit from the official precomputed ECS/PKS features:

```bash
PYTHONPATH=src python scripts/reproduce_redeep_detector.py
```

Run the expanded Dolly evaluation locally on GPU 2. Six examples are reserved
for controller tuning and all remaining 93 examples form the evaluation split:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/run_redeep_dolly.py \
  --device cuda:0 \
  --split evaluation \
  --max-new-tokens 128 \
  --methods baseline fixed_aarf minimum_norm \
  --target-mode relative \
  --target-score-shift 0.10 \
  --ridge 0 \
  --jacobian-refresh-interval 1 \
  --output artifacts/redeep/evaluation_shift0p10/dolly_generations.jsonl
```

Evaluate every saved answer with local Qwen-2.5-7B-Instruct; this command makes
no hosted-model or API call:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python \
  scripts/evaluate_redeep_dolly_local.py \
  --device cuda:0 \
  --input artifacts/redeep/evaluation_shift0p10/dolly_generations.jsonl
```

## Current corrected results

The maintained ReDeEP detector exactly reproduces the released Llama-2-7B
RAGTruth response AUC (`0.747254`). When feature selection and normalization are
fit only on training responses, the response-grouped held-out AUC is
`0.7175 +/- 0.0182` over five seeds. The expanded frozen-setting Dolly
evaluation contains the original 20 examples plus 73 newly generated examples;
only six tuning examples are excluded. Local Qwen-2.5-7B judging on these 93
examples gives fully supported rates of `64.5%`, `67.7%`, and `66.7%` for
unsteered, fixed AARF, and target-conditioned minimum norm. The minimum-norm
controller reduces the ReDeEP detector by `0.0684` versus `0.0289` for AARF
while changing the final hidden state by `9.02%` versus `12.48%`. Its paired
judge-score difference from baseline is `-0.0054` with a bootstrap interval of
`[-0.0484, +0.0323]`; AARF's difference is `+0.0215` with an interval of
`[-0.0108, +0.0591]`. A two-answer-order local pairwise judge gives minimum norm
`4.3%` wins, `5.4%` losses, and `90.3%` ties, while AARF gives `4.3%` wins,
`3.2%` losses, and `92.5%` ties. ReDeEP statistic control is therefore
validated, but neither intervention establishes an answer-quality improvement.

On the 60-question Lookback development split, greedy and sampled baselines
score `0.650` and `0.617` exact match. The official eight-candidate guided
decoder scores `0.717`. The frozen direct inverse (128 active statistic-gradient
heads, relative logit shift 4, RMS cap 0.5) scores `0.700`; a cardinality-matched
random support scores `0.633`. A high-attention support reaches the classifier
target but remains at `0.650`, documenting that detector movement alone can be
causally ineffective. On held-out indices 60--159, the matched sampled baseline
and official guided decoder score `0.710` and `0.700`; the paired interval for
the `-0.010` difference is `[-0.090, +0.070]`. The greedy baseline and frozen
direct inverse score `0.630` and `0.660`; that `+0.030` interval is
`[-0.040, +0.100]`. Neither intervention is a conclusive held-out improvement.

On the newer indices 160--259 matched four-candidate validation, fixed replay
reranking selects correct answers for `0.660` of controlled questions versus
`0.640` unsteered (paired difference `+0.020`, bootstrap interval
`[0.000, +0.050]`). This does not establish better controlled generation:
mean candidate exact match is `0.645` controlled versus `0.665` unsteered
(difference `-0.020`, interval `[-0.045, +0.0075]`). Smaller edits are therefore
selected on development candidate means and confirmed on a new untouched split;
the fixed replay selector and a shared answer-blind learned selector are both
reported. On this first validation, the shared learned selector reaches `0.690`
for controlled candidates and `0.680` for unsteered candidates; the paired
`+0.010` interval `[-0.050, +0.070]` still does not establish a perturbation
benefit.

The development-selected smaller edit (relative shift `1`, RMS cap `0.25`)
also fails to confirm on untouched indices 260--359. Controlled versus
unsteered candidate-mean exact match is `0.6125` versus `0.6200` (paired
`-0.0075`, 95% interval `[-0.0275, +0.0125]`). Fixed replay reranking reaches
`0.620` versus `0.630` (paired `-0.010`, interval `[-0.040, +0.020]`), and the
shared learned selector reaches `0.630` versus `0.650` (paired `-0.020`,
interval `[-0.060, +0.020]`). The selected edit therefore neither improves
average controlled generation nor outperforms its matched unsteered control.

On the 60-case PPLM disjoint split, the development-selected median (`q=0.50`)
absolute margin target reaches external target probability `0.954` and success
`0.983`, versus `0.852/0.850` for original PPLM and `0.856/0.867` for the
relative-shift-3 inverse. Its mean perplexity is `14.43`, compared with `14.02`
for original PPLM and `18.91` for relative shift 3; mean relative cache change
is `0.00102`, compared with `0.00179` and `0.00137`. Against original PPLM, the
paired target-probability difference is `+0.102` (95% bootstrap interval
`[+0.020,+0.191]`) and success improves by `+0.133`
(`[+0.033,+0.233]`), while the perplexity interval includes zero. Against
relative shift 3, target probability and success are also significantly higher,
and perplexity is lower by `4.48` (`[-7.20,-1.90]`). The earlier
development-selected output-preserving candidate remains a negative result:
target probability fell to `0.817` and perplexity rose to `19.73` on disjoint
validation.

On two non-overlapping 256-question TruthfulQA splits, cap-4 decoder-subspace
minimum norm significantly improves MC2 over the matched unsteered model by
`+0.118` (target 0.25) and `+0.065` (target 0.1). The stronger target reaches
MC1/MC2 `0.363/0.589` at mean relative action `2.38`, versus `0.402/0.659` at
`6.22` for the published edit. All-position normalized action is similar
(`0.0446` versus `0.0462`), while the published edit produces the larger MC2
gain. The minimum-norm setting is therefore not a Pareto or action-efficiency
win and remains significantly below published MC2.

A subsequent pre-specified strength extension selected target margin `0.25`
with cap `6` on development data. On the final untouched TruthfulQA indices
704--816 (`n=113`), it reaches MC1/MC2 `0.522/0.623`, versus `0.434/0.547` for
the paired unsteered model and `0.504/0.636` for published TruthX. Its paired
MC2 difference from published TruthX is `-0.013` with 95% interval
`[-0.090, +0.065]`, while mean relative action is `3.54` rather than `6.36`.
Those norms average changed positions only. Minimum norm changes `2.01%` of
positions versus `0.995%` for published TruthX, yielding all-position normalized
action `0.0724` versus `0.0638`. The methods are statistically indistinguishable
on the untouched split, but this is not a total-action-efficiency win. The
paired improvement over baseline also remains inconclusive at this sample size.
An exploratory development-only intervention-gate ablation did not improve the
frontier: gate-zero cap-6 and cap-8 reach MC2 `0.679` and `0.686`, and gate
`-0.25` cap-8 reaches `0.660`, versus `0.698` for the ungated cap-6 setting and
`0.702` for published TruthX on the same 64 questions.

## Historical legacy results

The tables below were produced before the accumulated-action solver audit.
They remain useful for provenance, but they must not be cited as results of the
corrected `accumulated_v2` inverse. Corrected tables are generated under
`artifacts/reports/`.

Sentiment is scored by a local DistilBERT-SST2 model, and continuation
perplexity by the unmodified local GPT-2-medium model. The following are macro
averages over balanced positive and negative targets (`n=60` per row).

| Split | Method | Target probability | Success | Mean PPL | Median PPL | P90 PPL | Relative cache change |
|---|---|---:|---:|---:|---:|---:|---:|
| Held-out | Persistent PPLM, 10 steps | 0.898 | 0.900 | 22.73 | 14.79 | 40.54 | 0.00260 |
| Held-out | Minimum norm, global mix | 0.868 | 0.867 | 17.85 | 16.56 | 29.43 | 0.00241 |
| Held-out | Minimum norm, calibrated | 0.941 | 0.950 | 21.60 | 17.67 | 36.97 | 0.00242 |
| Independent | Persistent PPLM, 10 steps | 0.912 | 0.933 | 36.90 | 18.50 | 44.30 | 0.00261 |
| Independent | Minimum norm, calibrated | 0.919 | 0.933 | 22.11 | 19.06 | 37.21 | 0.00232 |
| Adaptive development v2 | Persistent PPLM, 10 steps | 0.887 | 0.892 | 29.80 | 18.76 | 65.69 | 0.00267 |
| Adaptive development v2 | Minimum norm, adaptive KL | 0.898 | 0.900 | 26.24 | 20.51 | 40.99 | 0.00244 |
| Adaptive independent v3 | Persistent PPLM, 10 steps | 0.857 | 0.867 | 28.44 | 21.27 | 45.28 | 0.00264 |
| Adaptive independent v3 | Minimum norm, adaptive KL | 0.926 | 0.925 | 23.63 | 20.35 | 38.91 | 0.00234 |

The calibrated minimum-norm method has better point estimates for mean target
probability and mean/tail perplexity on the independent prefixes, with equal
success. Its median perplexity is slightly worse. Paired bootstrap 95% intervals
at `n=60` include zero for target probability, success, and mean perplexity, so
the current evidence establishes a promising Pareto improvement rather than a
statistically conclusive win.

The frozen adaptive policy subsequently gives a stronger independent result at
`n=120`: candidate-minus-PPLM target probability is `+0.0686` with paired
bootstrap 95% interval `[+0.0081, +0.1321]`. Success improves by `+0.0583` and
mean PPL decreases by `4.80`, although their separate intervals still include
zero. Median, 10%-trimmed, P90, and maximum perplexity all improve on this split.

The full TruthX evaluation uses all 817 TruthfulQA multiple-choice questions.

| Method | MC1 | MC2 | MC3 | Mean relative action on changed positions |
|---|---:|---:|---:|---:|
| Unsteered | 0.345 | 0.517 | 0.253 | 0.000 |
| Unrestricted minimum norm | 0.345 | 0.515 | 0.253 | 0.069 |
| Decoder-subspace minimum norm | 0.421 | 0.608 | 0.326 | 2.470 |
| Published TruthX | 0.490 | 0.688 | 0.410 | 6.318 |

The unrestricted TruthX result is a useful negative result: reducing the
autoencoder margin error does not by itself change multiple-choice behavior.
Constraining the inverse map to TruthX's decoded edit direction recovers about
half of the published method's gain with about 39% of its per-changed-position
relative action. Intervention coverage was not included in this legacy table,
so it cannot support a total-action-efficiency claim.
Machine-readable tables are in `artifacts/reports/`.
