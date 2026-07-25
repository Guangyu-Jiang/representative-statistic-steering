# Representative-Statistic Perturbation Steering

This repository contains the current code for steering frozen language models
by targeting representative statistics of their internal states. The common
inverse-control problem is

\[
\Delta H^\star =
\arg\min_{\Delta H}
D\!\left(g_\eta(H+\Delta H), z^\star\right)
+ \lambda R(\Delta H),
\]

where \(z^\star\) is a target statistic, \(g_\eta\) is either the exact
differentiable statistic or a local surrogate, and \(R\) controls intervention
cost. The repository includes exact, local-linear, Gauss-Newton, and
surrogate-based inverse mappings.

All generation and evaluation paths packaged here are API-free. Hosted judge
outputs and credentials are intentionally excluded.

## Included environments

| Environment | Representative statistic | Intervention state |
|---|---|---|
| PPLM | SST classifier margin | GPT-2 key/value cache |
| TruthX | truthful-latent cosine margin | attention/MLP hidden state |
| Lookback Lens | context-attention ratios and factuality score | attention logits |
| ReDeEP | PKS/ECS detector state | copying-head and knowledge-FFN controls |
| ITI | per-head truthfulness probe margins | selected attention-head activations |
| Topology | exact or surrogate \(H_0\) persistence statistics | prompt-token hidden states |
| CAA | behavior-matching activation score | residual-stream activation |

CAA is retained as a behavioral-activation ablation. Unlike the other
environments, its published method does not define a separate low-dimensional
representative statistic.

## Clone and setup

Clone with the pinned baselines:

```bash
git clone --recurse-submodules git@github.com:Guangyu-Jiang/representative-statistic-steering.git
cd representative-statistic-steering
python -m pip install -e .
pytest -q
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The PPLM, TruthX, Lookback Lens, and ReDeEP runners operate from this repository
root. Detailed commands are in [docs/CORE_EXPERIMENTS.md](docs/CORE_EXPERIMENTS.md).

ITI, topology, and CAA use separate dependency stacks. Materialize one without
modifying its pinned submodule:

```bash
python tools/materialize_environment.py iti
python tools/materialize_environment.py topology
python tools/materialize_environment.py caa
```

This creates `workspaces/<name>/`, applies the tracked patch, and copies the
experiment overlay. Follow the environment-specific README:

- [ITI](environments/iti/README.md)
- [Topology](environments/topology/README.md)
- [CAA](environments/caa/README.md)

Minimal Conda specifications for the shared core, topology, and CAA stacks are
under `env/`. ITI retains the paper's `environment.yaml` in its materialized
workspace because its pyvene/Transformers stack is version-sensitive.

## Repository policy

- Model weights, checkpoints, cached activations, datasets, and generated
  outputs are not tracked.
- Raw hosted-API artifacts and API-calling launchers are not part of this
  package.
- Large published baselines are pinned by Git submodule commit; the compact CAA
  and topology source trees are vendored with their provenance recorded.
- Development-set sweeps and held-out evaluation scripts remain separate.
- Each environment retains the license from its upstream repository.

## Validation

Run the lightweight package checks:

```bash
python tools/verify_package.py
pytest -q
```

Environment-specific tests can be run after materialization as documented in
the corresponding README.
