# Causal target-conditioned attention-head steering

## Control state

For selected attention head \(h=(\ell,j)\), let
\(a_{t,h}\in\mathbb{R}^{d_h}\) be the input slice to the attention output
projection at causal source position \(t\). Two linear representative
statistics are supported:

\[
s_{t,h}^{\mathrm{probe}}
=
\frac{w_h^\top a_{t,h}+b_h-\mu_h}{\sigma_h},
\qquad
s_{t,h}^{\mathrm{com}}
=
\frac{u_h^\top a_{t,h}-\mu_h}{\sigma_h},
\]

where \((w_h,b_h)\) are the official ITI logistic-probe parameters and \(u_h\)
is the unit center-of-mass direction from false to truthful candidate answers.
The means and standard deviations are estimated only from the development
half. The aggregate control state is

\[
z_t=\frac{1}{K}\sum_{h\in\mathcal H_K}s_{t,h}.
\]

The target \(\tau_q\) is quantile \(q\) of the corresponding aggregate
statistic on truthful development candidates. At inference, the requested
one-sided shift is

\[
r_t=\alpha[\tau_q-z_t]_+.
\]

Thus no intervention is made when the current state already lies on the
truthful side of the target.

## Minimum-norm perturbation

Because the statistic is linear in the selected head activations, its gradient
is available exactly. Writing the concatenated gradient as

\[
g=\left[\frac{v_h}{K\sigma_h}\right]_{h\in\mathcal H_K},
\]

where \(v_h=w_h\) for probe margins and \(v_h=u_h\) for COM projections, we
use a dimensionless relative ridge ratio \(\lambda\). The local objective is

\[
\min_{\delta_t}
\left(g^\top\delta_t-r_t\right)^2
+\lambda\lVert g\rVert_2^2\lVert\delta_t\rVert_2^2.
\]

Its regularized minimum-norm solution is

\[
\delta_t^\star
=
\frac{r_t}{(1+\lambda)\lVert g\rVert_2^2}g.
\]

An optional trust region clips the joint selected-head perturbation:

\[
\lVert\delta_t^\star\rVert_2
\leq
\rho\lVert a_{t,\mathcal H_K}\rVert_2.
\]

The implementation uses a causal two-pass procedure. The first pass measures
\(z_t\) from unmodified activations. The second pass applies the resulting
token-specific actions. Every action at position \(t\) depends only on a
causal hidden state whose receptive field ends at \(t\); it never uses future
answer tokens.

## ITI-basis perturbation

To isolate adaptive magnitude selection from direction selection, the
`targeted_iti` and `targeted_probe_iti` variants constrain the action to the
original ITI basis

\[
B=\left[\hat\sigma_h u_h\right]_{h\in\mathcal H_K},
\]

where \(\hat\sigma_h\) is the original tuning-set projection standard
deviation. It solves for one scalar coefficient under the one-dimensional
objective

\[
\min_{\beta_t}
\left(\beta_t g^\top B-r_t\right)^2
+\lambda\left(\beta_t g^\top B\right)^2,
\]

which gives

\[
\delta_t^\star=\beta_t B,
\qquad
\beta_t=\frac{r_t}{(1+\lambda)g^\top B}.
\]

The `bounded_targeted_probe_iti` variant adds an ITI-coefficient trust region

\[
0\leq\beta_t\leq\beta_{\max},
\qquad
\beta_t=\min\!\left\{
\frac{r_t}{(1+\lambda)g^\top B},\beta_{\max}
\right\}.
\]

Unlike the activation-relative trust region \(\rho\),
\(\beta_{\max}\) is directly comparable to the fixed ITI alpha: it limits a
token-specific correction to at most the displacement produced by fixed ITI
with strength \(\beta_{\max}\). This preserves exact target matching whenever
the required coefficient is inside the trust region and otherwise takes the
largest permitted step toward the target.

All three variants use exactly the same selected heads, COM directions, and
per-head scaling as fixed ITI. `targeted_iti` defines \(z_t\) from standardized
COM projections, whereas `targeted_probe_iti` defines \(z_t\) from the
standardized logistic-probe margins. The only change to the original action is
that its scalar magnitude is conditioned on the token's current statistic.

## Headwise target ablation

The aggregate statistic can conceal a deficient head behind another head with
an already-high margin. The optional `headwise_probe_iti` ablation instead
defines a truthful development target \(\tau_{q,h}\) for every selected head
and requests

\[
r_{t,h}=\alpha[\tau_{q,h}-s_{t,h}^{\mathrm{probe}}]_+.
\]

It still restricts each action to that head's original ITI direction
\(b_h=\hat\sigma_hu_h\). Because the probe statistic is linear, the exact
unregularized coefficient is

\[
\beta_{t,h}
=
\frac{r_{t,h}}
{(w_h/\sigma_h)^\top b_h},
\qquad
\delta_{t,h}=\beta_{t,h}b_h.
\]

A joint selected-head trust region is applied after solving the independent
coefficients. This ablation changes per-head magnitudes but neither the heads
nor the direction family used by fixed ITI.

The `headwise_probe_min_norm` variant keeps the same vector statistic and
targets but removes the ITI direction constraint. Writing

\[
g_h=\frac{w_h}{\sigma_h},
\]

its block-diagonal Jacobian gives the exact minimum-norm per-head correction

\[
\delta_{t,h}
=
\frac{r_{t,h}}{\lVert g_h\rVert_2^2}g_h.
\]

This directly maps every coordinate of the desired statistic displacement
back to the attention-head activation from which that coordinate was derived.

## Positive-negative group direction in statistic space

The group-directed variants replace independent per-head targets with a joint
direction estimated from labeled training candidates. Let

\[
z=(s_1^{\mathrm{probe}},\ldots,s_K^{\mathrm{probe}})
\]

be the standardized selected-head probe-margin vector. Using only the training
partition, estimate

\[
d_z=\mu_+^z-\mu_-^z,
\qquad
u_z=\frac{d_z}{\lVert d_z\rVert_2},
\]

where the two means correspond to truthful and false candidate sequences. A
positive-group projection target is then

\[
\tau_q=Q_q\bigl(u_z^\top z\mid y=+\bigr).
\]

For a causal token with statistic \(z_t\), the controller requests the joint
displacement

\[
r_t=\alpha[\tau_q-u_z^\top z_t]_+,
\qquad
z_t^\star=z_t+r_tu_z.
\]

This preserves the correlated margin-change pattern observed between the two
behavior groups instead of combining independently selected per-head maxima.
The `group_direction_probe_iti` inverse restricts each coordinate to the
original ITI basis,

\[
\delta_{t,h}
=
\frac{r_tu_{z,h}}{g_h^\top b_h}b_h,
\]

whereas `group_direction_probe_min_norm` uses the block-diagonal probe
Jacobian,

\[
\delta_{t,h}
=
\frac{r_tu_{z,h}}{\lVert g_h\rVert_2^2}g_h.
\]

Both variants therefore use the same statistic-space target and differ only
in how that target is mapped back to attention-head activations.

## Fixed ITI baseline

The matched fixed baseline applies

\[
\delta_{t,h}^{\mathrm{ITI}}
=
\alpha_{\mathrm{ITI}}\hat\sigma_h u_h
\]

to every selected head at every decode source position. In teacher-forced
TruthfulQA scoring, both methods edit only positions whose next-token logits
score an answer token. This is equivalent to intervening on the final prompt
token and each autoregressive decode token.

## Evaluation discipline

TruthfulQA's 817 questions are divided into two fixed folds. For each held-out
fold, the other half is split 80/20 to train probes and rank heads. Directions
and target distributions use only that development half. Hyperparameters are
selected from validation questions. The held-out fold is evaluated once after
selection. MC1 and MC2 use tokenizer-aware exact answer spans; no fixed number
of prefix tokens is removed. All metrics are local and API-free.
