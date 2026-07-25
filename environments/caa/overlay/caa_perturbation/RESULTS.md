# Current results

## Official released artifacts

The repository's released Llama-2-7B-Chat A/B outputs reproduce the expected
bidirectional CAA effect. The strongest layer by `p(+1) - p(-1)` is layer 12 for
sycophancy (`0.4650 / 0.6949 / 0.7039` for `-1 / 0 / +1`) and layer 13 for
refusal (`0.4233 / 0.7437 / 0.8873`). These are validations of released output
files, not fresh Llama-2 inference.

## Llama-3.1-8B-Instruct portability run

All rows use the official 50-example held-out A/B split and layer 13. Confidence
intervals are paired 95% bootstrap intervals for the gain over the unsteered
score.

| Behavior | Method | Match probability | Accuracy | `||delta h||/||h||` | Gain (95% CI) |
|---|---:|---:|---:|---:|---:|
| Sycophancy | Unsteered | 0.7803 | 0.76 | 0.000 | 0.0000 |
| Sycophancy | Fixed CAA | 0.7974 | 0.80 | 0.141 | +0.0171 `[+0.0059, +0.0317]` |
| Sycophancy | Scalar target | 0.7970 | 0.80 | 0.132 | +0.0167 `[+0.0051, +0.0310]` |
| Sycophancy | PCA target | 0.7894 | 0.80 | 0.111 | +0.0090 `[-0.0004, +0.0205]` |
| Refusal | Unsteered | 0.7871 | 0.78 | 0.000 | 0.0000 |
| Refusal | Fixed CAA | 0.8468 | 0.84 | 0.499 | +0.0597 `[+0.0184, +0.1115]` |
| Refusal | Scalar target | 0.8497 | 0.84 | 0.437 | +0.0626 `[+0.0220, +0.1154]` |
| Refusal | PCA target | 0.8500 | 0.86 | 0.383 | +0.0629 `[+0.0274, +0.1041]` |

The target-conditioned controllers match or slightly exceed fixed CAA on
refusal with smaller interventions. On sycophancy, scalar targeting matches
fixed CAA with a slightly smaller intervention; the PCA-target confidence
interval includes zero.

## Local open-ended evaluation

Qwen2.5-7B-Instruct scores all 53 sycophancy examples and all 50 refusal
examples. For refusal, higher means more refusal; for sycophancy, higher means
more sycophancy.

| Behavior | Method | Local score | Gain (95% CI) |
|---|---:|---:|---:|
| Sycophancy | Unsteered | 3.98 | 0.00 |
| Sycophancy | Fixed CAA | 3.79 | -0.19 `[-0.68, +0.30]` |
| Sycophancy | Scalar target | 3.72 | -0.26 `[-0.75, +0.23]` |
| Sycophancy | PCA target | 3.87 | -0.11 `[-0.62, +0.40]` |
| Refusal | Unsteered | 7.30 | 0.00 |
| Refusal | Fixed CAA | 7.66 | +0.36 `[-0.42, +1.18]` |
| Refusal | Scalar target | 8.20 | +0.90 `[+0.32, +1.60]` |
| Refusal | PCA target | 7.38 | +0.08 `[-0.70, +0.86]` |

Only the refusal scalar target has a local open-ended interval entirely above
zero. The sycophancy open-ended diagnostic is inconclusive and does not track
the held-out A/B metric. Local-Qwen scores are not numerically comparable to the
paper's GPT-4 scores.
