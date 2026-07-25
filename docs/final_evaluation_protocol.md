# Final evaluation protocol

All generation, scoring, and evaluation are local. No hosted model or judge API
is used. Hyperparameters are selected only on the development units listed
below; final comparisons retain raw per-example records and use paired bootstrap
intervals with 10,000 resamples.

## Lookback Lens

The representative statistic is the published logistic factuality score over
rolling attention-head lookback ratios. Minimum-norm control changes context-key
attention logits for 128 statistic-gradient-selected heads, using relative
classifier-logit target shift 4 and attention-bias RMS cap 0.5. Four complete
controlled responses are sampled per question and scored by replaying each
response without intervention. Selection never uses gold answers.

- Steering/reranking development: Natural Questions indices 0--59.
- Final validation: indices 160--259.
- Decoding: 64 new tokens, temperature 0.9, top-p 0.95, four candidates.
- Primary metric: short-answer subspan exact match.
- Controls: one sampled unsteered response; four sampled unsteered responses
  reranked by the same replay score; mean and oracle outcomes over each
  four-candidate set.

The one-candidate comparison measures the combined effect of perturbation and
candidate selection. The matched four-candidate comparison isolates the effect
of perturbation under equal generation and reranking compute. Candidate-mean
comparisons isolate generation from the replay selector.

The first frozen validation exposed a selector/generation mismatch: fixed
replay reranking scores `0.660` for controlled candidates and `0.640` for the
matched unsteered candidates, but their candidate means are `0.645` and
`0.665`. Therefore, a development-only refinement tests smaller logit shifts
and RMS caps on indices 0--59. Settings are ranked lexicographically by mean
candidate exact match, reranked exact match, and lower intervention RMS. The
selected setting is confirmed once on new indices 260--359. An answer-blind
logistic candidate ranker uses only replay/online statistic scores, response
length and repetition, question-term retention, and overlap with the supplied
document. It is trained on indices 0--59 and applied identically to controlled
and unsteered candidate pools on indices 260--359.

The confirmation is complete. The selected shift-1, RMS-cap-0.25 edit yields
candidate-mean exact match `0.6125` versus `0.6200` unsteered, replay-reranked
exact match `0.620` versus `0.630`, and learned-ranker exact match `0.630`
versus `0.650`. All paired 95% intervals include zero. These negative results
are retained as the final confirmatory outcome rather than selecting another
setting after inspecting validation.

## PPLM sentiment

The representative statistic is the local SST classifier target margin over the
GPT-2-medium cache. Development uses five prefixes with seed 11. Final validation
uses 15 disjoint prefixes and seeds 22 and 33, crossed with positive and negative
targets for 60 paired generations. Original PPLM and minimum-norm control use
matched prefixes, seeds, decoding, and local external sentiment/perplexity
evaluation. Primary outcomes are external target probability and success;
perplexity is the fluency-quality outcome.

## TruthX

The representative statistic at each selected module is the cosine difference
between the TruthX truthful latent and its positive versus negative centers.
Minimum-norm control is constrained to the autoencoder decoder direction and is
applied at the official pre-output-projection attention site or MLP residual
site. The frozen setting uses target margin 0.25, ridge 0, damping 1, and
per-position relative norm cap 6.

- Strength development: TruthfulQA indices 0--63.
- Final untouched confirmation: indices 704--816 (`n=113`).
- Descriptive full-corpus completion: indices 64--703, merged with the above
  shards only after the untouched comparison was reported.
- Controls: matched unsteered Llama-2-7B-chat and published TruthX, using the
  same questions, answer choices, top-10 modules, and two-fold checkpoints.
- Primary metrics: TruthfulQA MC1 and MC2; MC3 is also retained.

`mean_relative_action_norm` averages `||Delta h||/||h||` only over positions
where the action is nonzero. `intervention_rate` is the fraction of all eligible
module-position calls that change. The all-position normalized action is

```text
mean over questions of
    mean_relative_action_norm(question) * intervention_rate(question).
```

Both quantities must be reported. A lower changed-position norm does not imply
less total perturbation when intervention coverage differs.

## Reproduction and audit

```bash
bash scripts/launch_lookback_minimum_norm_rerank_untouched_validation.sh
bash scripts/launch_lookback_baseline_rerank_untouched_validation.sh
bash scripts/launch_lookback_minimum_norm_rerank_development_diagnostics.sh
bash scripts/launch_lookback_rerank_refinement_development.sh
python scripts/select_lookback_rerank_refinement.py --require-complete
bash scripts/launch_lookback_refinement_confirmation_queue.sh
python scripts/train_evaluate_lookback_candidate_ranker.py --require-complete

bash scripts/launch_truthx_cap6_t0p25_untouched_confirmation.sh
bash scripts/launch_truthx_cap6_t0p25_full_completion.sh

python scripts/build_pplm_corrected_report.py
python scripts/build_truthx_corrected_report.py
python scripts/build_lookback_report.py
python scripts/build_lookback_refinement_confirmation.py --require-complete
python scripts/build_final_heldout_report.py --require-complete
python scripts/audit_final_splits.py --require-complete
pytest -q
```
