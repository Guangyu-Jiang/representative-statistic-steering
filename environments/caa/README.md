# CAA environment

This environment reproduces Contrastive Activation Addition and evaluates a
minimum-norm activation perturbation on its behavior datasets. CAA directly
uses behavior-matching activations rather than a distinct representative
statistic, so this environment is an ablation of the general framework rather
than a primary statistic-inversion benchmark.

Materialize and install:

```bash
python tools/materialize_environment.py caa
cd workspaces/caa
python -m pip install -r requirements.txt
pytest -q tests
```

The relevant code is under:

- `caa_exact_replication/` for the paper-aligned local replication;
- `caa_perturbation/` for inverse-control experiments and summaries.

Generation and evaluation use local models. No hosted-model API is required.

