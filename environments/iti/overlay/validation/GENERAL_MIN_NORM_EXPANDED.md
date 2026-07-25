# Expanded general minimum-norm ITI experiment

## Method

For the selected 48 attention heads, the aggregate COM statistic has exact
gradient (g). At causal token position (t), the intervention requests the
one-sided shift

\[
r_t=\alpha[\tau_q-z_t]_+
\]

and applies the regularized minimum-norm action

\[
\delta_t^\star=
\frac{r_t}{(1+\lambda)\lVert g\rVert_2^2}g.
\]

The action is not restricted to the original ITI basis (B). An
activation-relative trust region clips it to

\[
\lVert\delta_t^\star\rVert_2
\leq \rho\lVert A_{t,\mathcal H}\rVert_2.
\]

## Selection

The expanded screen tested 85 general settings at (K=48): COM and probe
statistics, alphas 4/8/12/16/20/24/32, truthful target quantiles 0.75/0.90,
and new relative caps 1.5/2/3, plus the previous COM alpha-16, quantile-0.75,
cap-1 setting. Ridge ratio was fixed at zero because, before clipping, it is
algebraically redundant with rescaling alpha. Fixed ITI alpha 8 and 15 were
evaluated as matched controls.

The grid was screened on 48 development questions. A union of the top MC1,
MC2, mean-MC, efficiency, and per-statistic settings was then evaluated on all
164 validation questions. The highest validation mean was COM, alpha 24,
quantile 0.75, cap 3. A second COM, alpha 20, quantile 0.75, cap 3 setting was
retained as the activation-norm-matched control because its validation action
norm was below fixed ITI alpha 15.

## Held-out multiple choice

All values below use the same 817 TruthfulQA questions and fold-specific head
selection. MC values are percentages.

| Method | MC1 | MC2 | Relative action/state norm |
| --- | ---: | ---: | ---: |
| Baseline | 33.54 | 50.82 | 0.000 |
| Fixed ITI, alpha 8 | 40.88 | 58.13 | 1.096 |
| Fixed ITI, alpha 15 | 41.25 | 59.99 | 2.117 |
| General minimum norm, alpha 24 | 43.33 | 62.24 | 2.331 |
| General minimum norm, alpha 20, norm-matched | **43.33** | **62.41** | **2.038** |

For the norm-matched setting versus fixed ITI alpha 15, the paired gains are
+2.08 MC1 points (95% CI [-0.24, +4.41]) and +2.41 MC2 points
([+0.21, +4.58]). Its action norm is 3.7% lower. Versus fixed ITI alpha 8,
the gains are +2.45 MC1 points ([0.00, +4.90]) and +4.27 MC2 points
([+2.02, +6.55]). Thus the general minimum-norm solution now exceeds both
fixed baselines in point estimates, with a statistically clear MC2 gain over
both; the MC1 advantage over alpha 15 remains uncertain.

## Artifacts

- Expanded screen: `artifacts/iti_attention_head/screen_general_min_norm_expanded_k48`
- Full validation confirmation: `artifacts/iti_attention_head/confirm_general_min_norm_expanded_k48`
- Validation-selected held-out run: `artifacts/iti_attention_head/heldout_general_min_norm_expanded_k48`
- Norm-matched held-out run: `artifacts/iti_attention_head/heldout_general_min_norm_norm_matched_k48`
- Reproduction runners: `scripts/run_iti_general_min_norm_expanded.sh`,
  `scripts/run_iti_general_min_norm_heldout.sh`, and
  `scripts/run_iti_general_min_norm_norm_matched_heldout.sh`

No external API was used.
