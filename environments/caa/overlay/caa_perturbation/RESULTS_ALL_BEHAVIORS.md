# All-behavior CAA perturbation results

## Scope

This run compares the original fixed contrastive activation addition (CAA)
with two state-dependent, closed-form perturbation controllers on all seven CAA
behaviors. It uses the locally cached `meta-llama/Llama-3.1-8B-Instruct` model;
this is a portability experiment rather than an exact Llama-2-7B-Chat
replication. No external API was called.

For each behavior, the intervention layer is selected using 80/20
cross-validated AUC on the contrastive training activation pairs. The 50
official held-out choice examples are then split, stratified by the matching
answer option, into a tuning half and an evaluation half. Hyperparameters are
selected on the tuning half, and the table below reports only the evaluation
half. Cells are `matching probability / matching accuracy / mean relative
action norm`.

| Behavior | Layer | Unsteered | Fixed CAA | Scalar target | PCA target |
|---|---:|---:|---:|---:|---:|
| coordinate-other-ais | 12 | 0.191 / 0.160 / 0.000 | 0.206 / 0.160 / 0.241 | 0.202 / 0.160 / 0.141 | 0.205 / 0.160 / 0.188 |
| corrigible-neutral-HHH | 13 | 0.801 / 0.840 / 0.000 | 0.850 / 0.880 / 0.500 | 0.850 / 0.880 / 0.494 | 0.847 / 0.840 / 0.500 |
| hallucination | 9 | 0.292 / 0.308 / 0.000 | 0.317 / 0.308 / 0.500 | 0.317 / 0.308 / 0.500 | 0.317 / 0.308 / 0.498 |
| myopic-reward | 9 | 0.601 / 0.680 / 0.000 | 0.597 / 0.640 / 0.340 | 0.598 / 0.680 / 0.046 | 0.602 / 0.680 / 0.017 |
| refusal | 12 | 0.763 / 0.731 / 0.000 | 0.798 / 0.769 / 0.500 | 0.799 / 0.769 / 0.496 | 0.806 / 0.808 / 0.472 |
| survival-instinct | 13 | 0.568 / 0.577 / 0.000 | 0.648 / 0.654 / 0.498 | 0.646 / 0.654 / 0.492 | 0.636 / 0.615 / 0.460 |
| sycophancy | 13 | 0.789 / 0.840 / 0.000 | 0.799 / 0.840 / 0.281 | 0.782 / 0.800 / 0.288 | 0.750 / 0.720 / 0.281 |

Macro averages over the seven behaviors are:

| Method | Match probability | Accuracy | Relative action norm | Probability gain over unsteered |
|---|---:|---:|---:|---:|
| Unsteered | 0.5721 | 0.5908 | 0.0000 | 0.0000 |
| Fixed CAA | 0.6022 | 0.6073 | 0.4086 | +0.0300 |
| Scalar target | 0.5990 | 0.6073 | 0.3509 | +0.0269 |
| PCA target | 0.5949 | 0.5901 | 0.3452 | +0.0227 |

Scalar targeting matches fixed CAA's macro accuracy with 14.1% lower mean
relative action norm, but its macro matching probability is 0.0032 lower. The
PCA controller is 0.0073 below fixed CAA on macro matching probability. Both
perturbation methods beat fixed CAA on three of seven individual behaviors, so
the held-out results do not establish a consistent perturbation advantage. A
separate full-held-out oracle summary is retained for exploration but should
not be used as the primary result because it selects on the reported data.

## Open-ended check

The tuning-selected settings generate all official open-ended prompts: 50 for
six behaviors and 53 for sycophancy. Generation patches the final prompt state
and every cached decode state. `Qwen/Qwen2.5-7B-Instruct` assigns the official
behavior-oriented 0--10 scores locally. All 1,412 generated responses are
non-empty, and all 1,412 judgments parse successfully. Higher scores mean more
of the named target behavior; they do not mean generally better answers. In
particular, the positive survival target is willingness to accept shutdown.

| Behavior | Unsteered | Fixed CAA | Scalar target | PCA target |
|---|---:|---:|---:|---:|
| coordinate-other-ais | 2.06 | 2.98 | 3.18 | 2.90 |
| corrigible-neutral-HHH | 7.44 | 7.12 | 7.28 | 7.56 |
| hallucination | 7.16 | 7.78 | 7.30 | 7.10 |
| myopic-reward | 5.50 | 6.16 | 5.80 | 6.00 |
| refusal | 7.30 | 7.64 | 7.24 | 7.78 |
| survival-instinct | 7.04 | 7.14 | 7.20 | 7.56 |
| sycophancy | 3.98 | 4.00 | 3.92 | 4.13 |
| **Macro average** | **5.783** | **6.117** | **5.989** | **6.147** |

PCA targeting is 0.030 above fixed CAA in macro score and has a higher mean on
four of seven behaviors. Paired 95% bootstrap intervals show significant gains
over fixed CAA for corrigibility (`+0.44`, `[+0.06, +0.86]`) and shutdown
acceptance (`+0.42`, `[+0.02, +0.80]`), but a significant loss for
hallucination (`-0.68`, `[-1.32, -0.14]`). All other PCA-versus-CAA intervals
include zero. Scalar targeting is 0.128 below fixed CAA in macro score; its only
significant direct difference is worse hallucination steering (`-0.48`,
`[-1.00, -0.06]`).

## Methods and grid

- Fixed CAA adds the raw contrastive mean difference.
- Scalar targeting moves the current projection toward the 75th percentile of
  positive training projections using a ridge-regularized minimum-norm action.
- PCA targeting moves a pair-centered PCA statistic toward the positive
  centroid using a ridge-regularized minimum-norm action. Answer-option identity
  is removed as a nuisance subspace, including C--H options in the survival
  data.
- CAA strengths: `-2, -1, -0.5, 0.5, 1, 2`.
- Target strengths: `0.25, 0.5, 1, 1.5, 2`.
- PCA dimensions: `1, 2, 4, 8`; ridge values: `0.01, 0.1, 1`.
- Per-token relative action norm is capped at `0.5`.
- Each behavior has 82 configurations, for 574 held-out configurations total.
- Primary fixed-CAA selection considers only positive strengths, which point
  toward the matching behavior; negative-strength rows remain in the full grid
  as directionality controls.

## Limitations

- The answer model is Llama-3.1-8B-Instruct, not the paper's Llama-2 model.
- Each leakage-aware evaluation half contains only 25 or 26 examples, so many
  paired confidence intervals remain wide.
- The local Qwen judge is an API-free diagnostic, not numerically comparable to
  the paper's GPT-4 judge. For most behaviors, the open-ended prompts are
  reformulations of the same held-out questions used in the A/B set, so this
  secondary check is not an additional independent test set.
- The search reaches the `0.5` norm cap for several behaviors. Larger actions
  were intentionally excluded from this run to avoid conflating stronger
  control with severe activation corruption.

The complete grid, per-question predictions, paired bootstrap summaries,
norm-matched comparisons, generated responses, and local judgments are stored
under `artifacts/caa_perturbation_all_behaviors/`.
