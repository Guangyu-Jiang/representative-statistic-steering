# Target-conditioned perturbation steering on CAA

This extension keeps the official CAA contrastive datasets, held-out A/B
evaluation, block-output intervention site, and additive residual-stream patch.
It compares fixed contrastive activation addition against two controls whose
action depends on the current hidden state. No external inference or judging API
is used.

## Methods

For positive and negative answer activations at layer `l`, fixed CAA uses

```text
v = mean_i(h_i_positive - h_i_negative),   delta_h = alpha * v.
```

The scalar target controller projects the current state onto the unit CAA
direction. Its target is the 75th percentile of positive training projections,
and its minimum-norm ridge action has the analytic form

```text
s(h) = <h - center, unit(v)>
delta_h = alpha * (s_target - s(h)) / (1 + lambda) * unit(v).
```

The PCA target controller first pair-centers every contrastive example, removes
the A-versus-B answer-letter nuisance direction, and fits a standardized PCA
statistic `z(h) = A(h - center)`. It targets the positive cluster centroid:

```text
delta_h = A^T (A A^T + lambda I)^-1 alpha * (z_target - z(h)).
```

All methods patch the same decoder block output. The A/B evaluation patches the
assistant answer-prefix token. Open-ended generation patches the final prompt
state that predicts the first answer token and every subsequent cached decode
state. `--max-relative-norm` bounds `||delta_h||_2 / ||h||_2` per patched token.

## Reproduce

The commands below use the cached Llama-3.1-8B-Instruct model as a portability
replication because the original gated Llama-2-7B-Chat checkpoint is not
available in this environment.

```bash
python -m caa_perturbation.run_experiment extract \
  --behavior sycophancy --device cuda:0 --batch-size 8
python -m caa_perturbation.run_experiment scan \
  --behavior sycophancy --device cuda:0 --batch-size 8
python -m caa_perturbation.run_experiment evaluate \
  --behavior sycophancy --device cuda:0 --layer 13 --batch-size 8

python -m caa_perturbation.run_experiment extract \
  --behavior refusal --device cuda:0 --batch-size 8
python -m caa_perturbation.run_experiment scan \
  --behavior refusal --device cuda:0 --batch-size 8
python -m caa_perturbation.run_experiment evaluate \
  --behavior refusal --device cuda:0 --layer 13 --batch-size 8
```

Generate an open-ended setting and score it with a local judge:

```bash
python -m caa_perturbation.run_experiment generate \
  --behavior refusal --device cuda:0 --layer 13 \
  --method scalar_target --components 8 --strength 1.5 --ridge 0.1

python -m caa_perturbation.local_judge \
  --behavior refusal --device cuda:0 --batch-size 8 \
  artifacts/caa_perturbation/meta-llama__Llama-3.1-8B-Instruct/refusal/open_ended/scalar_target__r8__strength1.5__ridge0.1.json
```

Summarize the official released Llama-2 artifacts and the new held-out results,
including paired bootstrap confidence intervals for both A/B and local
open-ended evaluation:

```bash
python -m caa_perturbation.summarize_results
```

The completed seven-behavior comparison is documented in
[`RESULTS_ALL_BEHAVIORS.md`](RESULTS_ALL_BEHAVIORS.md). Its machine-readable
reports can be regenerated with:

```bash
python -m caa_perturbation.summarize_all_behaviors \
  --artifact-root artifacts/caa_perturbation_all_behaviors
python -m caa_perturbation.summarize_local_open_ended_all \
  --artifact-root artifacts/caa_perturbation_all_behaviors
```

## Fidelity and interpretation

- The files under `results/` are the official repository's released Llama-2
  outputs. The summary command validates those artifacts without model access.
- Fresh inference here uses Llama-3.1-8B-Instruct, so it is a portability
  replication rather than an exact numerical reproduction of the paper.
- The official GPT-4 open-ended scorer is replaced by
  `Qwen/Qwen2.5-7B-Instruct`. Local scores are secondary diagnostics and must not
  be presented as directly comparable to the paper's GPT-4 scores.
- Hyperparameters are selected from the contrastive training activations. The 50
  official A/B test examples remain held out for evaluation.

Run all extension and upstream helper tests with:

```bash
python -m pytest -q tests utils/test_helpers.py
```
