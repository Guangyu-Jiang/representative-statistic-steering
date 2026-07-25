# Lookback-Lens Minimum-Norm Control

For decode step `t`, layer `l`, and head `h`, let `C` denote the original
prompt/document keys and `G_t` the generated-token keys. The published
Lookback statistic is

```math
r_{l,h,t} =
\frac{\operatorname{mean}_{j\in C} a_{l,h,t,j}}
     {\operatorname{mean}_{j\in C} a_{l,h,t,j}+
      \operatorname{mean}_{j\in G_t} a_{l,h,t,j}}.
```

The 1024 ratios from Llama-2-7B-chat (32 layers by 32 heads) are averaged over
the latest eight decode steps and passed through the official logistic
factuality classifier,

```math
s_t = w^\top \bar r_t + c.
```

## Differentiable intervention map

The intervention variable `b_{l,h,t}` is an additive attention-logit bias. If
it is applied to every context key in one head, the representative statistic
has the exact conditional map

```math
\operatorname{logit}(r'_{l,h,t}) =
\operatorname{logit}(r_{l,h,t}) + b_{l,h,t}.
```

For a selected context support that originally receives fraction `q` of the
head's total context-attention mass, the map is

```math
\operatorname{logit}(r'_{l,h,t}) =
\operatorname{logit}(r_{l,h,t}) +
\log((1-q_{l,h,t}) + q_{l,h,t}\exp(b_{l,h,t})).
```

This removes the need for a learned inverse surrogate. The map is exact for a
fixed layer's pre-intervention query/key scores. Applying controls at all
layers changes downstream queries and keys, so every run also records the
actual post-forward statistic error rather than reporting only the conditional
prediction.

## Accumulated-action inverse

At a current accumulated bias `b`, let `g = d s / d b` and
`e = s_target - s(b)`. The local subproblem is

```math
\min_u (g^\top u-e)^2 + \lambda\lVert b+u\rVert_2^2.
```

Its rank-one solution is

```math
u^* = -b +
\frac{e+g^\top b}{\lVert g\rVert_2^2+\lambda}g.
```

The implementation damps this update, applies an optional RMS cap, and uses
backtracking to prevent an increase in the exact low-dimensional control
objective. Regularization is therefore applied to the final accumulated bias,
not independently to each iteration.

## Underdetermination and statistic gaming

Matching a low-dimensional statistic does not identify a unique intervention.
For a local scalar target, every action with the same projection onto `g`
produces the same first-order score change. The minimum-norm solution selects
the component parallel to `g`, but this can amplify keys that already receive
high attention without improving the generated answer. We therefore treat
statistic matching as a necessary control constraint, not as evidence of a
causal factuality improvement.

To test whether the support of the intervention matters, the solver can be
restricted to an answer-blind, example-dependent support `S(x)`. Lexical and
retrieval supports identify question-relevant document tokens before decoding;
the matched-cardinality `random_overlap` support is a causal control. A
high-attention support that attains the classifier target but does not improve
exact match is an explicit statistic-gaming control.

An optional `minimum_norm_rerank` hybrid draws independently seeded responses
under the same direct intervention, replays each completed response without an
intervention, and selects the response with the highest mean replayed Lookback
probability. The unsteered replay prevents a controller from grading its own
manipulated statistic. Selection is answer-blind, candidate zero uses the same
seed as the one-candidate sampled control, and every controlled/replayed score
and seed is retained in the raw artifact. This isolates whether the direct
inverse and the published selection mechanism are complementary.

## Context supports and controls

- `uniform` biases every prompt/document key.
- `question_overlap` biases document tokens within a fixed radius of
  non-stopword question-term matches.
- `question_top_union` unions that lexical support with each head's most
  attended context keys.
- `top_attention` uses only each head's most attended context keys.
- `random_overlap` is a deterministic causal control with the same support
  cardinality as `question_overlap`, sampled only from document tokens.
- `retrieved_passage` and `retrieved_sentence` use answer-blind BM25 relevance
  between the question and document passages or sentences.

Sparse variants restrict the solve to the largest-magnitude statistic
gradients. For a nonnegative intervention, sparse selection uses only positive
score gradients before applying the top-k rule; selecting large negative
gradients and clipping them afterward would waste control coordinates. Runs
record active-bias counts and the fraction of negative biases as signed-action
diagnostics. A gated relative target uses a strong logit shift below a
classifier-probability trigger and either zero or a smaller shift above it.

## Evaluation protocol

Development uses NQ examples 0--59. Final configurations are frozen before a
disjoint held-out run beginning at example 60. The final comparison uses the
repository's prompt construction, Llama-2-7B-chat, greedy decoding, eight-way
guided candidate reranking with eight-token chunks, and the repository's
substring exact-match metric. Paper-aligned final runs use 256 maximum new
tokens. Direct interventions are paired with a baseline using the same decode
policy and generation length. Guided reranking is paired with sampled
single-candidate decoding because both draw stochastic candidates; comparing it
only to a greedy baseline would confound control with decoding policy. All
generation and evaluation are local; no hosted judge or API is used.
