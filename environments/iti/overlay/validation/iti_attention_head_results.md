# ITI attention-head perturbation results

## Protocol

These experiments use the cached `NousResearch/Llama-2-7b-chat-hf` mirror,
TruthfulQA's 817 questions, seed 42, and the official two-fold ITI protocol.
For each test fold, the other half is split 80/20 to train per-head logistic
probes and rank 48 attention heads. Center-of-mass directions, activation
scales, normalization statistics, and truthful target quantiles are fitted
only on that development half. Candidate-answer likelihoods are computed from
tokenizer-derived answer spans. Both fixed and target-conditioned methods edit
the causal source state that predicts each answer token.

No external API is used. The repository's published Llama-2-chat-7B reference
is baseline MC1/MC2 `0.34/0.51` and fixed ITI (`K=48`, `alpha=15`)
`0.40/0.58`.

| Reproduction check | Baseline MC1 | Baseline MC2 | Fixed ITI MC1 | Fixed ITI MC2 |
| --- | ---: | ---: | ---: | ---: |
| Released repository report | 34.00 | 51.00 | 40.00 | 58.00 |
| This implementation, seed 42 | 33.54 | 50.82 | 41.25 | 59.99 |
| This implementation, seeds 1-3 mean | 33.54 | 50.82 | 42.47 | 60.66 |

Thus the unsteered scores reproduce the report within 0.5 points and the fixed
intervention reproduces or modestly exceeds the reported gains. The local
open-ended judge is intentionally not compared numerically with the report's
fine-tuned GPT judges.

## Held-out multiple choice

| Method | Target | Alpha | Cap | MC1 | MC2 | Relative action/state norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | none | 0 | - | 33.54 | 50.82 | 0.000 |
| Fixed ITI, validation-tuned | none | 8 | - | 40.88 | 58.13 | 1.096 |
| Fixed ITI | none | 15 | - | 41.25 | 59.99 | 2.117 |
| Bounded targeted probe ITI, validation-selected | Probe q=0.75 | 12 | coefficient 10 | 42.11 | 60.41 | **1.052** |
| Aggregate minimum norm | COM q=0.75 | 16 | 1 | 38.56 | 57.84 | 0.946 |
| Targeted ITI, norm matched | COM q=0.75 | 16 | 2 | 43.82 | 61.90 | 1.812 |
| Targeted ITI, accuracy | COM q=0.75 | 16 | 4 | **46.51** | **64.01** | 2.451 |
| Targeted ITI | COM q=0.90 | 16 | 4 | 45.29 | 63.38 | 3.088 |
| Targeted probe ITI, generation-selected | Probe q=0.75 | 12 | 2 | 43.70 | 61.79 | **1.179** |
| Targeted probe ITI | Probe q=0.75 | 16 | 4 | 44.55 | 63.10 | **1.612** |

Values for MC1 and MC2 are percentages; the relative norm is a ratio. The
fixed baseline closely reproduces the repository's result. The probe-targeted
variant changes only the scalar magnitude in the original ITI basis and uses
the same 48 heads, COM directions, and variance scales. It improves fixed ITI
while using 23.8% less relative activation change.

An additional fixed-ITI alpha-12 control nearly matches the probe-targeted
activation budget: it obtains 41.25 MC1 and 59.84 MC2 at relative norm 1.673.
The probe-targeted result remains higher at 44.55/63.10 while using the smaller
norm 1.612. Its paired gains over this control are +3.30 MC1 (95% CI
[+1.71, +5.02]) and +3.25 MC2 ([+2.01, +4.53]).

In the final validation-tuned comparison, bounded probe targeting improves
fixed ITI alpha 8 by +1.22 MC1 (paired 95% CI [0.00, +2.45]) and +2.28 MC2
([+1.51, +3.08]) while using 4.0% less selected-head action. Relative to the
paper-default alpha 15, its +0.86 MC1 and +0.42 MC2 differences are
statistically tied. This separates a gain over a similarly tuned fixed control
from the larger gains of unconstrained variants selected for MC accuracy.

## Paired uncertainty

Fold-stratified paired bootstrap intervals use 20,000 resamples of the same
817 questions.

| Candidate minus fixed ITI | MC1 difference (95% CI) | MC2 difference (95% CI) |
| --- | ---: | ---: |
| Targeted ITI, COM q=0.75, cap=2 | +2.57 [+0.73, +4.41] | +1.91 [+0.82, +3.03] |
| Targeted ITI, COM q=0.75, cap=4 | +5.26 [+2.82, +7.71] | +4.02 [+2.15, +5.92] |
| Targeted probe ITI, q=0.75, alpha=12, cap=2 | +2.45 [+0.49, +4.53] | +1.80 [+0.27, +3.33] |
| Targeted probe ITI, q=0.75, cap=4 | +3.30 [+1.47, +5.14] | +3.10 [+1.67, +4.55] |
| Bounded probe target, coefficient cap=10, versus fixed alpha 8 | +1.22 [0.00, +2.45] | +2.28 [+1.51, +3.08] |
| Bounded probe target, coefficient cap=10, versus fixed alpha 15 | +0.86 [-1.10, +2.82] | +0.42 [-1.16, +2.01] |

The lower bounds are positive for both metrics in the first four comparisons;
the bounded controller has a nonnegative MC1 bound and positive MC2 bound
against tuned fixed alpha 8, while both intervals include zero against fixed
alpha 15. Raw per-question scores, fold summaries, settings manifests, and
paired bootstrap tables are under `artifacts/iti_attention_head/heldout_k48`,
`artifacts/iti_attention_head/heldout_targeted_probe_k48`, and
`artifacts/iti_attention_head/heldout_final_bounded_k48`.

For the probe-targeted comparison, MC1 improves on 43 questions, worsens on
16, and is unchanged on 758. MC2 improves on 537 questions, worsens on 239,
and is unchanged on 41. Thus the aggregate gain is not explained by one or
two unusually large examples.

## Multi-seed robustness

The fixed setting and an accuracy-focused alpha-16 probe-targeted setting were
rerun for seeds 1, 2, and 3, matching the seed convention used by the
repository's Llama-2 replication table. Each seed contains both folds and all
817 held-out questions.

| Seed | Fixed MC1 | Targeted MC1 | Fixed MC2 | Targeted MC2 | Fixed norm | Targeted norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.96 | 47.86 | 61.07 | 65.74 | 2.101 | 1.739 |
| 2 | 41.74 | 44.92 | 60.33 | 63.35 | 2.043 | 1.711 |
| 3 | 42.72 | 46.51 | 60.57 | 63.57 | 2.061 | 1.672 |
| Mean | 42.47 | **46.43** | 60.66 | **64.22** | 2.068 | **1.707** |

The perturbation method wins on both MC metrics for every seed and uses 17.4%
less relative action norm on average. Baseline remains 33.54/50.82 because the
model and held-out questions do not change with probe-training seed. The
aggregate artifact is
`artifacts/iti_attention_head/robustness_seeds1_3_k48_combined_summary.csv`.

The validation-selected alpha-12, cap-2 setting was then confirmed under the
same three seeds:

| Seed | Fixed MC1 | Selected target MC1 | Fixed MC2 | Selected target MC2 | Fixed norm | Selected target norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.96 | **44.80** | 61.07 | **63.48** | 2.101 | **1.250** |
| 2 | 41.74 | **44.43** | 60.33 | **62.11** | 2.043 | **1.225** |
| 3 | 42.72 | **44.55** | 60.57 | **61.89** | 2.061 | **1.217** |
| Mean | 42.47 | **44.59** | 60.66 | **62.49** | 2.068 | **1.230** |

The selected setting wins both MC metrics for every seed, averaging +2.12 MC1
and +1.83 MC2 points with 40.5% less action norm. Fold-stratified paired CIs
are positive for both metrics in seed 2 and for MC2 in seed 1; the remaining
intervals narrowly include zero. Per-seed summaries and paired tables are in
`artifacts/iti_attention_head/robustness_selected_a12_seed{1,2,3}_k48`.

## Open-ended validation selection

Open-ended hyperparameters were selected using 164 validation questions (82
per fold) and a local Qwen-2.5-72B reference-grounded judge. Both folds
independently select probe-targeted alpha 12, truthful quantile 0.75, and cap
2. The score called `True*Info` below exactly follows the released evaluator:
the fold-averaged Truth rate multiplied by the fold-averaged Info rate. `Joint`
is the stricter per-answer conjunction and is reported separately.

| Method | Truth | Info | True*Info | Joint |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 57.32 | 75.00 | 42.99 | 32.93 |
| Fixed ITI, alpha 6 | 60.98 | 85.98 | 52.42 | 48.17 |
| Fixed ITI, alpha 8 | **62.80** | 84.15 | 52.85 | 49.39 |
| Fixed ITI, alpha 10 | 60.98 | 85.37 | 52.05 | 48.17 |
| Fixed ITI, alpha 15 | 57.32 | 73.78 | 42.29 | 42.68 |
| Probe target, alpha 8, cap 2 | 62.20 | 82.32 | 51.20 | 47.56 |
| Probe target, alpha 12, cap 2 | 60.98 | **87.20** | 53.17 | 49.39 |
| Probe target, alpha 16, cap 2 | 59.76 | 81.10 | 48.46 | 45.73 |
| Probe target, alpha 16, cap 4 | 60.98 | 78.05 | 47.59 | 43.90 |
| COM target, alpha 16, cap 2 | 55.49 | 79.88 | 44.32 | 46.34 |
| Bounded probe target, coefficient cap 4 | 58.54 | 82.32 | 48.19 | 41.46 |
| Bounded probe target, coefficient cap 6 | 60.37 | 82.93 | 50.06 | 46.34 |
| Bounded probe target, coefficient cap 8 | 62.20 | 84.15 | 52.33 | 50.00 |
| Bounded probe target, coefficient cap 10 | 62.20 | 86.59 | **53.85** | **50.61** |
| Bounded probe target, coefficient cap 15 | 61.59 | 86.59 | 53.32 | 49.39 |

Against fixed ITI, the selected setting improves validation True*Info by
10.88 points (paired fold-stratified bootstrap 95% CI [+4.83, +16.82]) and
Info by 13.41 points ([+6.71, +20.12]). The corresponding joint gain is 6.71
points ([0.00, +13.43]). These values come from
`generation_validation_sweep_k48/local_qwen72b_judge`; no test-fold judge
labels were used to choose the setting.

The additional fixed-alpha sweep makes the baseline comparison stricter.
Unbounded probe targeting is statistically tied with validation-selected
fixed alpha 8: its True*Info difference is +0.32 points (paired 95% CI
[-4.29, +4.93]) and its joint difference is 0.00 points ([-5.49, +5.49]).
It nevertheless uses a smaller mean action/state ratio, 1.03 versus 1.13.
Adding an upper bound of 10 to the token-specific ITI coefficient gives the
best validation result, 53.85 True*Info and 50.61 joint, at a mean ratio of
0.96. This cap and fixed alpha 8 are therefore carried forward together to a
single paired held-out evaluation.

## Generation integrity audit

All 817 held-out questions were generated before judging. The decoder measures
the current statistic and makes a steered pass at each token, advancing only
the steered KV cache.

| Method | Nonempty | Unique | Changed from baseline | Mean characters | Mean action/state ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 817/817 | 641 | 0.0% | 84.9 | - |
| Fixed ITI, alpha 15 | 817/817 | 764 | 94.2% | 152.0 | 2.140 |
| Fixed ITI, alpha 7.1, norm control | 817/817 | 708 | 79.9% | 114.2 | **1.002** |
| Probe target, alpha 12, cap 2 | 817/817 | 713 | 78.5% | 106.9 | **1.015** |
| Validation-tuned fixed ITI, alpha 8 | 817/817 | 716 | 82.5% | 117.2 | 1.132 |
| Bounded probe target, alpha 12, coefficient cap 10 | 817/817 | 710 | 76.9% | 105.2 | **0.958** |

The ratio is \(\lVert\delta A_{\mathcal H}\rVert_2 /
\lVert A_{\mathcal H}\rVert_2\), averaged over causal generation steps and
restricted to the 48 selected head slices; it is not a ratio to the entire
residual stream. The selected perturbation uses 52.5% less action than fixed
ITI alpha 15 and intervenes on 87.4% of measured positions. Fixed alpha 7.1
was derived only to match that budget: its 1.002 ratio differs from the
target-conditioned ratio by 1.3%. The reproducible audits are
`generation_selected_probe_a12_k48/generation_audit.csv` and
`generation_fixed_norm_a7p1_k48/generation_audit.csv`. In the stricter
validation-tuned comparison, bounded targeting uses 15.4% less action than
fixed alpha 8 and intervenes on 87.7% of decode source positions. Its complete
audit is `generation_final_validation_selected_k48/generation_audit.csv`.

## Open-ended held-out evaluation

The validation-selected setting was then evaluated once on all 817 outer-fold
test questions with the local Qwen-2.5-72B judge. All parse rates are 100%.

| Method | Truth | Info | True*Info | Joint |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 57.16 | 75.88 | 43.38 | 34.88 |
| Fixed ITI, alpha 15 | 58.63 | 77.35 | 45.35 | 43.57 |
| Probe target, alpha 12, cap 2 | **61.81** | **80.29** | **49.63** | **44.92** |

Relative to fixed ITI, target-conditioned perturbation improves the paper's
True*Info metric by 4.28 points (paired fold-stratified bootstrap 95% CI
[+1.27, +7.26], one-sided bootstrap mass at or below zero 0.0023). Truth rises
by 3.18 points ([-0.12, +6.61]) and Info by 2.94 points ([-0.12, +5.99]). The
stricter joint rate rises by 1.35 points but is statistically tied
([-2.08, +4.78]); the method therefore outperforms fixed ITI on the released
paper metric, not on every auxiliary aggregation.

Relative to the unsteered baseline, the perturbation gains 6.25 True*Info
points ([+3.60, +8.94]) and 10.03 joint points ([+6.73, +13.34]). Fixed ITI's
True*Info gain over baseline is 1.97 points ([-1.26, +5.26]) under this local
judge. Primary summaries and paired tables are in
`generation_selected_probe_a12_k48/local_qwen72b_judge`.

### Validation-tuned bounded comparison

The bounded controller selected on the 164-question validation split was also
compared once against validation-tuned fixed ITI alpha 8 on the same 817
held-out questions. No test labels were used to choose either setting.

| Method | Truth | Info | True*Info | Joint | Action/state ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed ITI, alpha 8 | **63.89** | 81.15 | **51.85** | **47.36** | 1.132 |
| Bounded probe target, alpha 12, coefficient cap 10 | 61.57 | **82.00** | 50.49 | 45.28 | **0.958** |

Bounded targeting raises Info by 0.86 points (paired 95% CI
[-0.86, +2.57]) while using 15.4% less selected-head action, but lowers Truth
by 2.32 points ([-4.40, -0.24]). Its True*Info difference is -1.36 points
([-3.23, +0.52]) and its joint difference is -2.08 points
([-4.28, +0.13]). Thus it is statistically tied with tuned fixed ITI on the
paper metric and joint rate, but does not outperform that stronger control;
the significant Truth reduction is a substantive negative result. Complete
scores and paired bootstrap output are in
`generation_final_validation_selected_k48/local_qwen72b_judge`.

## Local-judge sensitivity

The same 817 questions and frozen generated strings were also scored by two
independent 7B local judges. No answers were regenerated and no setting was
retuned for these checks.

| Judge | Method | Truth | Info | True*Info | Joint |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen-2.5-72B | Baseline | 57.16 | 75.88 | 43.38 | 34.88 |
| Qwen-2.5-72B | Fixed ITI, alpha 15 | 58.63 | 77.35 | 45.35 | 43.57 |
| Qwen-2.5-72B | Fixed ITI, alpha 7.1 | **63.28** | **80.78** | **51.12** | **45.89** |
| Qwen-2.5-72B | Probe target, alpha 12 | 61.81 | 80.29 | 49.63 | 44.92 |
| Qwen-2.5-7B | Baseline | 41.74 | 74.54 | 31.11 | 40.51 |
| Qwen-2.5-7B | Fixed ITI, alpha 15 | 37.94 | **83.23** | 31.58 | 36.84 |
| Qwen-2.5-7B | Fixed ITI, alpha 7.1 | **47.61** | 82.12 | **39.10** | **46.38** |
| Qwen-2.5-7B | Probe target, alpha 12 | 44.91 | 81.88 | 36.78 | 43.81 |
| Mistral-7B-v0.3 | Baseline | 54.84 | 51.16 | 28.06 | 47.25 |
| Mistral-7B-v0.3 | Fixed ITI, alpha 15 | 64.87 | **69.52** | **45.10** | **61.93** |
| Mistral-7B-v0.3 | Fixed ITI, alpha 7.1 | **65.24** | 66.33 | 43.27 | 61.07 |
| Mistral-7B-v0.3 | Probe target, alpha 12 | 64.50 | 65.97 | 42.55 | 60.34 |

The target-conditioned method's True*Info gain over fixed ITI is +4.28 points
with Qwen-72B (95% CI [+1.27, +7.26]) and +5.20 points with Qwen-7B
([+2.01, +8.40]). Mistral instead assigns it a nonsignificant -2.54 point
difference ([-6.47, +1.44]) and significantly lower Info. Therefore the
multiple-choice improvement is judge-independent, whereas the open-ended
advantage is supported by both Qwen judges but is sensitive to the choice of
local evaluator. The primary judge remains Qwen-72B because it was fixed
before held-out evaluation and also used for validation-only model selection.

At the matched generation-action budget, target minus fixed alpha 7.1 is
-1.49 True*Info points with Qwen-72B (95% CI [-3.57, +0.53]), -2.32 with
Qwen-7B ([-4.68, +0.04]), and -0.72 with Mistral ([-3.40, +1.96]). Thus the
target method is statistically tied with the equal-norm fixed control on the
primary and Mistral judges and is not better on Qwen-7B. The evidence supports
an advantage over the paper-prescribed alpha-15 baseline, but not a general
advantage over a lower-strength, norm-matched fixed intervention.

## K-dimensional head-margin control

The vector controller retains the separate probe margin of each selected ITI
head instead of averaging the margins into one scalar. For token position
\(t\), its representative statistic is
\(z_t=(s_{t,1},\ldots,s_{t,K})\). Each coordinate is assigned a training-only
truthful target \(\tau_{q,h}\), giving the one-sided residual
\(r_{t,h}=[\tau_{q,h}-s_{t,h}]_+\). The selected inverse maps each residual
back through the corresponding ITI head direction \(b_h\):

\[
\delta_{t,h}=\frac{r_{t,h}}{J_h b_h}b_h,
\]

where \(J_h\) is the linear probe Jacobian for head \(h\). Selection over
\(K\in\{8,16,32,48\}\) and
\(q\in\{0.95,0.975,0.99,1.0\}\) on the two validation folds chose
\(K=48\), \(q=1.0\), and \(\alpha=1\). The held-out evaluation contains all
817 TruthfulQA questions.

| Method | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Scalar direct probe target, \(K=48,q=0.99\) | 35.74 | 53.21 | 0.214 |
| K-dimensional head target, \(K=48,q=1.0\) | **37.33** | **54.23** | 1.031 |
| Fixed ITI, \(K=48,\alpha=15\) | 41.25 | 59.99 | 2.117 |

Relative to the unsteered baseline, the vector controller gains 3.79 MC1
points (paired 95% CI [2.20, 5.51]) and 3.41 MC2 points ([2.29, 4.57]). It
also improves over scalar direct targeting by 1.59 MC1 points ([0.24, 2.94])
and 1.02 MC2 points ([0.17, 1.90]). It remains below fixed ITI by 3.92 MC1
points ([-6.49, -1.35]) and 5.76 MC2 points ([-7.90, -3.60]), while using
48.7% of fixed ITI's relative action norm. Thus preserving the per-head
statistic recovers information lost by scalar averaging, but does not yet
match the stronger fixed intervention.

The reported near-zero post-target error is the error under the cached local
linear probe model, not a remeasurement after propagating every intervention
through the sequential transformer layers. Interventions at earlier layers
can change the later-layer activations on which subsequent cached corrections
were computed, so exact nonlinear target attainment remains to be measured.

## Positive-negative direction in K-dimensional statistic space

The group-directed controller estimates a joint direction from truthful and
false training candidates in the standardized 48-head probe-margin space:

\[
d_z=\mu_+^z-\mu_-^z,
\qquad
u_z=d_z/\lVert d_z\rVert_2.
\]

For each causal token, it moves the current statistic toward a training-only
positive projection quantile,

\[
r_t=\alpha[\tau_q-u_z^\top z_t]_+,
\qquad
z_t^\star=z_t+r_tu_z.
\]

An 84-setting validation sweep covered two inverse maps,
\(q\in\{0.5,0.75,0.9,0.95,0.99,1.0\}\), and
\(\alpha\in\{0.25,0.5,1,1.5,2,3,4\}\). Validation mean MC1/MC2 selected
\(q=1,\alpha=4\) independently for both inverse maps. All 817 held-out
TruthfulQA questions were then evaluated once with those settings.

| Method | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Scalar direct probe target | 35.74 | 53.21 | 0.214 |
| Independent K-head target | 37.33 | 54.23 | 1.031 |
| Group direction, minimum-norm inverse | 35.50 | 53.83 | 2.019 |
| Group direction, ITI-basis inverse | **37.58** | **56.29** | 3.309 |
| Fixed ITI, alpha 15 | 41.25 | 59.99 | 2.117 |

Relative to baseline, the group ITI-basis controller gains 4.04 MC1 points
(paired 95% CI [1.96, 6.12]) and 5.46 MC2 points ([3.74, 7.25]). Relative to
the independent K-head target, its MC1 difference is +0.24 points
([-1.59, 2.08]) and its MC2 gain is +2.05 points ([0.72, 3.40]). It remains
below fixed ITI by 3.67 MC1 points ([-5.88, -1.59]) and 3.71 MC2 points
([-5.57, -1.89]), while using a larger activation change. The minimum-norm
inverse is less effective: it gains 1.96 MC1 and 3.01 MC2 points over baseline
but remains below both the independent K-head controller and fixed ITI.

The result supports the causal usefulness of the positive-negative statistic
axis, especially for MC2, but it does not establish an improvement over the
fixed intervention. Two limitations are visible. First, the selected
\(\alpha=4\) deliberately overshoots the positive projection threshold and
has high action cost. Second, after per-head standardization the learned group
direction is almost uniform across heads (fold cosines 0.996 and 0.995 with an
equal-weight vector), so a uniform-direction control is required before gains
can be attributed to the learned coordinate weights themselves.

## Ridge and activation-cap ablation

We next regularized the group-direction inverse while holding the
validation-selected statistic target fixed at \(K=48\), \(q=1\), and
\(\alpha=4\). A ridge ratio \(\gamma\) changes the inverse denominator from
\(J J^\top\) to \((1+\gamma)J J^\top\). A relative activation cap \(c\) is
applied independently at each causal token:

\[
\lVert\delta_t\rVert_2
\leq c\lVert H_{t,\mathcal H}\rVert_2,
\]

where \(H_{t,\mathcal H}\) contains the 48 selected head activations. The
broad screen used ridge ratios
\(\{0,0.1,0.25,0.5,1,2,4,8\}\) without a cap and caps
\(\{0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3\}\) at zero ridge for both inverse
maps. The focused minimum-norm refinement crossed
\(\gamma\in\{0,0.05,0.1,0.25\}\) with
\(c\in\{1.5,1.75,2,2.25,2.5,3\}\). Selection again used only the two
validation folds, maximizing mean MC1/MC2 and breaking ties by lower action
norm.

| Validation setting | MC1 | MC2 | Relative action norm | Clip rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 31.10 | 45.40 | 0.000 | -- |
| Minimum norm, no regularization | 35.37 | 51.76 | 2.012 | 0.0% |
| Minimum norm, ridge 0.10 | 35.37 | 51.62 | 1.829 | 0.0% |
| Minimum norm, cap 2.00 | 35.37 | 51.86 | 1.762 | 56.0% |
| Minimum norm, cap 2.25 | **35.37** | **51.93** | 1.863 | 40.9% |
| ITI basis, no regularization | 35.37 | 52.78 | 3.298 | 0.0% |

No ridge or cap improved the ITI-basis inverse. For the minimum-norm inverse,
the validation-selected setting was zero ridge and cap 2.25. It was then
evaluated once on all 817 held-out questions:

| Held-out setting | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Minimum norm, no regularization | **35.50** | **53.83** | 2.019 |
| Minimum norm, cap 2.25 | **35.50** | 53.67 | **1.869** |
| ITI basis, no regularization | 37.58 | 56.29 | 3.309 |
| Fixed ITI, alpha 15 | 41.25 | 59.99 | 2.117 |

The capped controller gains 1.96 MC1 points (paired 95% CI [0.37, 3.55]) and
2.84 MC2 points ([1.70, 4.03]) over baseline. Relative to the unregularized
minimum-norm inverse, MC1 is unchanged and MC2 decreases by 0.17 points
([-0.32, -0.03]), while relative action norm decreases by 7.4%. Thus the cap
provides a modest efficiency tradeoff rather than a behavioral improvement.
Ridge is largely redundant with reducing \(\alpha\) when the cap is inactive;
the token-adaptive cap is the more meaningful regularizer in this ablation.

## Ablation without probe-margin standardization

The standard controller represents head \(h\) by its standardized probe
margin

\[
z_{t,h}=\frac{w_h^\top a_{t,h}+b_h-\mu_h}{s_h}.
\]

We ablated this normalization while preserving the fitted logistic probes,
the validation-ranked 48 heads, data folds, target construction, and inverse
maps. The raw variant instead uses

\[
z^{\mathrm{raw}}_{t,h}=w_h^\top a_{t,h}+b_h,
\]

recomputes the positive-minus-negative direction and positive target
quantiles in raw-margin space, and uses \(w_h\), rather than \(w_h/s_h\), as
the inverse Jacobian. Fresh statistics caches prevent reuse of standardized
targets. The same 84-setting validation budget was used for each
normalization: two inverse maps, six target quantiles, and seven alpha values.
Validation selected \(q=0.99,\alpha=4\) for the raw ITI-basis inverse and
\(q=1,\alpha=4\) for the raw minimum-norm inverse. These settings were then
evaluated once on all 817 held-out questions.

| Held-out setting | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Raw margins, minimum-norm inverse | 35.13 | 53.36 | **1.722** |
| Standardized margins, minimum-norm inverse | 35.50 | 53.83 | 2.019 |
| Raw margins, ITI-basis inverse | 36.23 | 53.96 | 2.119 |
| Standardized margins, ITI-basis inverse | **37.58** | **56.29** | 3.309 |
| Fixed ITI, alpha 15 | 41.25 | 59.99 | 2.117 |

The raw ITI-basis controller gains 2.69 MC1 points (paired 95% CI
[0.98, 4.41]) and 3.14 MC2 points ([1.85, 4.45]) over baseline. Relative to
the standardized ITI-basis controller, it loses 1.35 MC1 points
([-2.94, 0.24]) and 2.32 MC2 points ([-3.36, -1.32]), while reducing relative
action norm by 36.0%. The raw minimum-norm controller loses 0.37 MC1 points
([-0.98, 0.24]) and 0.47 MC2 points ([-0.74, -0.22]) relative to its
standardized counterpart, while reducing action norm by 14.7%.

Raw and standardized group directions have cosine similarities of 0.930 and
0.936 in the two folds, but the five largest raw coordinates account for
36--38% of squared direction mass, compared with 14% after standardization.
Thus removing standardization does not destroy the signal, but lets
high-variance probe margins dominate. It yields a lower-cost controller at the
expense of behavior, especially MC2; standardized margins remain preferable
when steering performance is the primary objective.

## Group-direction alpha sweep through 20

The initial group-direction search stopped at \(\alpha=4\). We extended both
inverse maps with

\[
\alpha\in\{5,6,8,10,12,15,20\}
\]

while retaining all six target quantiles and testing both standardized and
raw probe margins. This added 168 validation settings. The two folds were
sharded across three GPUs, but candidate selection still aggregated all 164
validation questions. No ridge or activation cap was used in this sweep.
One winner was selected by mean validation MC1/MC2 for each inverse and
normalization before held-out evaluation.

| Validation winner | Alpha | Quantile | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standardized, ITI basis | 20 | 0.90 | **53.66** | **69.27** | 7.692 |
| Raw, ITI basis | 20 | 0.90 | 51.83 | 67.94 | 7.116 |
| Standardized, minimum norm | 20 | 1.00 | 46.34 | 67.12 | 10.060 |
| Raw, minimum norm | 20 | 1.00 | 47.56 | 66.06 | 8.597 |

All four validation winners occurred at the upper alpha boundary. Their
held-out evaluation contains all 817 TruthfulQA questions:

| Held-out setting | MC1 | MC2 | Relative action norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Fixed ITI, alpha 15 | 41.25 | 59.99 | **2.117** |
| Raw, minimum norm, alpha 20 | 48.59 | 68.44 | 8.611 |
| Standardized, minimum norm, alpha 20 | 48.84 | 68.89 | 10.095 |
| Raw, ITI basis, alpha 20 | 54.71 | 71.42 | 7.129 |
| Standardized, ITI basis, alpha 20 | **56.92** | **72.48** | 7.728 |

The standardized ITI-basis winner improves over baseline by 23.38 MC1 points
(paired 95% CI [19.46, 27.30]) and 21.66 MC2 points
([18.21, 25.19]). It also exceeds fixed ITI by 15.67 MC1 points
([11.87, 19.34]) and 12.49 MC2 points ([9.38, 15.68]). Relative to the
previous standardized group-direction setting at \(\alpha=4\), the gains are
19.34 MC1 points ([15.54, 23.01]) and 16.20 MC2 points
([13.01, 19.49]). These improvements require a relative action norm of 7.728,
which is 3.65 times fixed ITI and 2.34 times the alpha-4 group controller.

The high-alpha result therefore establishes that the statistic-conditioned
controller can outperform fixed ITI on TruthfulQA multiple-choice likelihood
metrics, but not at a matched intervention cost. Because every selected
setting lies at \(\alpha=20\), the behavioral optimum may be beyond the tested
range; stronger interventions should not be explored without caps and
open-ended quality or distribution-shift checks. MC1/MC2 measure candidate
likelihood ranking and do not by themselves establish preservation of
open-ended generation quality.
