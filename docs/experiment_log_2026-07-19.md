# Corrected Perturbation-Steering Experiment Log: 2026-07-19

All experiments in this log use local checkpoints and local evaluators. No
hosted API or external judge is used. Runs marked `accumulated_v2` solve for the
final accumulated action under each new local linearization; older artifacts
that independently regularized every increment are retained only as legacy
provenance.

## Lookback Lens

- Development split: Natural Questions indices 0--59, at most 64 generated
  tokens.
- Held-out split: indices 60--159, at most 256 generated tokens.
- Frozen direct inverse: question-overlap support, 128 active score-gradient
  heads, relative classifier-logit shift 4, RMS cap 0.5, unrestricted signed
  attention-logit bias.
- Held-out greedy baseline: 0.630 exact match.
- Held-out frozen direct inverse: 0.660 exact match; paired delta +0.030,
  bootstrap 95% interval [-0.040, +0.100], 8 improvements and 5 regressions.
- Held-out gated inverse (trigger 0.93): 0.640 exact match; paired delta +0.010,
  interval [0.000, +0.030], one improvement and no regressions.
- Held-out sampled direct inverse: 0.650 exact match versus 0.710 for the
  matched sampled baseline; paired delta -0.060, interval [-0.130, +0.010],
  three improvements and nine regressions. The direct effect is therefore not
  robust to decoding policy.
- The official score is not a strong causal proxy for NQ exact match on this
  split: baseline score AUC is 0.596, controlled score AUC is 0.512, and the
  correlation between score change and exact-match change is -0.094.
- The official eight-candidate guided decoder scores 0.700 versus 0.710 for
  its matched sampled baseline: paired delta -0.010, interval
  [-0.090, +0.070], with seven improvements and eight regressions. Its
  development improvement therefore does not replicate on held-out NQ.
- An initial four-candidate inverse reranker was stopped because controlled
  scores saturated. Its replacement ranks candidates by an unsteered replay
  score and is running as an answer-blind complementarity test; the report
  separately records candidate-average, oracle, and selected exact match.
- On the frozen indices 160--259 matched four-candidate comparison, controlled
  replay reranking scores 0.660 versus 0.640 unsteered (paired +0.020,
  bootstrap interval [0.000, +0.050]). This does not reflect stronger average
  generations: controlled candidate-mean exact match is 0.645 versus 0.665
  unsteered (paired -0.020, interval [-0.045, +0.0075]).
- A subsequent development-only refinement compares relative shifts 1--4 and
  RMS caps 0.25--0.5 using identical candidate seeds. The predeclared
  lexicographic rule first maximizes candidate-mean exact match, then reranked
  exact match, then minimizes RMS. It selects shift 1 with cap 0.25: candidate
  mean 0.650 and reranked EM 0.733, versus 0.6125 and 0.700 for matched
  unsteered candidates. Mean bias RMS is 0.235 and output KL is 0.00257. This
  development-selected setting is evaluated once on new indices 260--359.
- A shared answer-blind logistic selector, trained with grouped out-of-fold
  development predictions and document-grounding features, reaches 0.690 on
  controlled candidates and 0.680 on matched unsteered candidates on indices
  160--259. The paired difference is +0.010 with interval [-0.050, +0.070], so
  improving candidate selection does not rescue a clear causal effect for the
  shift-4 edit.
- The frozen shift-1, RMS-cap-0.25 refinement is complete on untouched indices
  260--359. Controlled candidate-mean exact match is 0.6125 versus 0.6200 for
  matched unsteered candidates (paired -0.0075, interval [-0.0275, +0.0125]).
  Fixed replay reranking scores 0.620 versus 0.630 (paired -0.010, interval
  [-0.040, +0.020]); the shared learned selector scores 0.630 versus 0.650
  (paired -0.020, interval [-0.060, +0.020]). Mean controlled bias RMS is
  0.2358. The development gain does not generalize, and neither selector
  supports a causal Lookback improvement from this perturbation.

## PPLM sentiment control

- Corrected disjoint validation: 15 default prefixes, seeds 22 and 33, positive
  and negative targets; 60 rows per method.
- Original PPLM: target probability 0.8521, success 0.850, mean perplexity
  14.02, relative cache change 0.00179.
- Corrected absolute-target minimum norm: target probability 0.7800, success
  0.800, mean perplexity 18.81, relative cache change 0.00148. The paired target
  delta is -0.0721 with 95% interval [-0.1783, +0.0331], and perplexity is
  4.79 points worse.
- A five-prefix development sweep selected relative margin shift 3.0: target
  probability 0.9943, success 1.0, perplexity 16.98, relative cache change
  0.00132. On the 60-case disjoint validation it reaches 0.8556 target
  probability versus PPLM's 0.8521, a paired +0.0036 with interval
  [-0.0906, +0.0988], but perplexity is 18.91 versus 14.02, a significant
  +4.89 with interval [+2.57, +7.27]. The development Pareto win does not
  generalize.
- A shift-3 structural development ablation compares all-cache, key-only,
  value-only, and last-layer-block intervention spaces.
- A frozen top-2 output-log-probability preservation candidate (weight 0.05)
  is being evaluated on the disjoint 60-case split after reaching 0.8787
  target probability at perplexity 14.87 on development, versus 0.9020 and
  18.83 for PPLM. This candidate was selected before inspecting validation.

## TruthX multiple choice

- Development split: TruthfulQA indices 0--63.
- Held-out split: indices 64--191.
- Development selected decoder-direction scalar control with target margin 0.1,
  ridge 0.01, damping 0.5, and relative norm cap 0.5. It improved MC2 by 0.0423
  with paired interval [+0.0157, +0.0782].
- On held-out data the same setting changes MC1 by +0.0078 and MC2 by +0.0063;
  the MC2 interval [-0.0027, +0.0164] crosses zero. The development gain did not
  generalize.
- A corrected strong-action ablation confirms that the historical
  decoder-subspace gain came from a larger semantic displacement. With target
  margin 0.25, ridge 0, damping 1, and cap 4, development MC1/MC2 are
  0.391/0.646 versus 0.234/0.463 for the matched baseline. The MC2 gain is
  +0.183 with interval [+0.077, +0.288], at mean relative action 2.60. Legacy
  repeated-update overshooting is not reused.
- On untouched indices 192--447, target 0.25 with cap 4 reaches MC1/MC2
  0.363/0.589 versus 0.258/0.470 for the matched baseline. The paired gains
  are +0.105 MC1 (95% interval [+0.051, +0.160]) and +0.118 MC2
  ([+0.073, +0.165]). It remains below the published edit's 0.402/0.659;
  MC2 differs by -0.070 ([-0.115, -0.025]). Mean relative action is 2.38
  versus 6.22 for the published edit.
- On a separate untouched split, indices 448--703, target 0.1 with cap 4
  reaches MC1/MC2 0.445/0.586 versus 0.418/0.522. Its MC2 gain is +0.0647
  ([+0.0194, +0.1124]), but it remains 0.0819 below the published edit.
  Its MC2 gain per changed-position norm is 0.0255 versus 0.0232 for the
  published edit, but it intervenes on 1.77% versus 0.746% of positions.
  All-position MC2 gain per action is therefore 1.39 versus 3.08, favoring the
  published edit. The changed-position-only ratio must not be presented as a
  total-action efficiency result.
- A monotone-backtracking solver and relative-margin targets are now screened
  only on development data. On the first 32 development questions,
  backtracking target 0.1 reduces final target RMSE from 1.242 to 0.579 and
  relative action from 2.62 to 1.27, but MC2 falls from 0.494 to 0.333. This
  demonstrates that better local statistic matching need not produce a better
  downstream behavioral intervention.
- A pre-specified cap extension on development indices 0--63 found that target
  margin 0.25 with cap 6 reaches MC1/MC2 0.500/0.698 at mean relative action
  3.68. It nearly matches the published edit's 0.484/0.702 while using 42% less
  action. Raising the cap to 8 gives 0.484/0.699 at action 4.62 and is therefore
  dominated by cap 6. Nonnegative scalar-ray variants reach at most MC2 0.358
  on the first 32 questions, so they were rejected without held-out testing.
- The cap-6, target-0.25 setting was frozen and evaluated once on the remaining
  untouched indices 704--816 (n=113). It reaches MC1/MC2 0.522/0.623 versus
  0.434/0.547 for the paired baseline and 0.504/0.636 for published TruthX.
  Candidate-minus-baseline differences are +0.088 MC1 (95% interval
  [-0.009, +0.186]) and +0.076 MC2 ([-0.008, +0.160]); candidate-minus-published
  differences are +0.018 MC1 ([-0.062, +0.097]) and -0.013 MC2
  ([-0.090, +0.065]). Mean relative action is 3.54 versus 6.36 for published
  TruthX when averaged over changed positions. However, the controller changes
  2.01% of positions versus 0.995% for published TruthX; all-position normalized
  action is therefore 0.0724 versus 0.0638. Thus this confirmation is
  statistically indistinguishable from the published controller on both MC
  metrics, but it does not establish lower total perturbation. Its improvement
  over baseline is also not conclusive at n=113.
- A development-only activation-gate ablation does not improve this setting.
  Gate-zero cap 6 and cap 8 reach MC2 0.679 and 0.686, while gate -0.25 with
  cap 8 reaches 0.660. The ungated cap-6 setting reaches 0.698 and also has
  lower all-position normalized action (0.0831 versus 0.0856--0.0947).
