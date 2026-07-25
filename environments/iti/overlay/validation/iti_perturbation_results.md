# Local ITI and adaptive-margin results

All results use `meta-llama/Meta-Llama-3-8B-Instruct`, TruthfulQA's 817
questions, two-fold evaluation, 48 selected heads, seed 42, and no external
API. Open-ended answers are evaluated with the cached local
`Qwen/Qwen2.5-7B-Instruct` checkpoint.

## Fixed ITI application-site audit

| Setting | MC1 | MC2 |
| --- | ---: | ---: |
| Baseline | 39.05 | 58.34 |
| Installed modern last-position hook | 39.05 | 58.34 |
| Intended legacy answer-position hook | 40.02 | 58.03 |
| Causal answer-prefix-and-token hook | 36.47 | 55.88 |

The installed modern hook edits only the final position of a full candidate
answer. That position cannot affect the likelihood of preceding answer tokens,
so its multiple-choice scores are exactly equal to baseline. The intended
legacy answer-position path partially reproduces the reported MC1 gain, but it
does not reproduce the reported MC2 gain in this environment.

## Adaptive aggregate-margin multiple choice

The adaptive action is the regularized minimum-norm change along the gradients
of standardized selected-head probe margins. The target is a quantile of the
truthful training-margin distribution.

| Target quantile | Alpha | Ridge ratio | MC1 | MC2 | Mean action/state norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | - | 39.05 | 58.34 | 0.00 |
| 0.25 | 2 | 0 | 39.90 | 59.04 | 4.44 |
| 0.25 | 8 | 0 | 41.00 | 60.69 | 17.77 |
| 0.75 | 1 | 0 | 40.02 | 59.20 | 8.08 |
| 0.75 | 2 | 0 | 40.39 | 59.93 | 16.16 |
| 0.75 | 4 | 0 | 41.00 | 61.06 | 32.32 |
| 0.75 | 8 | 0 | 43.70 | 63.35 | 64.64 |

Values in the final column are percentages. Higher margin targets improve MC1
and MC2, but require substantially larger activation changes.

## Local open-ended evaluation

| Setting | Answers changed | Mean action/state norm | Truthful | Informative | Truth x info |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.00 | 0.00 | 50.18 | 80.17 | 49.08 |
| Fixed ITI, alpha 15 | 86.29 | not recorded | 31.95 | 61.32 | 30.48 |
| Margin q=0.25, alpha 2 | 0.86 | 0.56 | 50.31 | 80.17 | 49.20 |
| Margin q=0.75, alpha 2 | 15.91 | 7.55 | 50.55 | 80.66 | 49.45 |
| Margin q=0.75, alpha 4 | 26.93 | 15.09 | 49.94 | 79.80 | 48.71 |
| Margin q=0.75, alpha 8 | 37.33 | 30.19 | 50.43 | 78.34 | 49.20 |

Values are percentages. No adaptive setting produced empty responses. The
moderate `q=0.75, alpha=2, ridge=0` setting is the best local open-ended result:
it improves both truthfulness and informativeness slightly. Stronger settings
produce larger MC gains but lose informativeness, showing that probe-margin
control and open-ended answer quality are not interchangeable objectives.
