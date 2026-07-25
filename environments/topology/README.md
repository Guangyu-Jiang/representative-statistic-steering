# Topology environment

This environment controls exact or learned \(H_0\) persistence statistics of
prompt-token clouds for AmbigQA, SituatedQA, CLAMBER, and FalseQA. It contains:

- differentiable MST-based \(H_0\) statistics;
- surrogate-based topology control;
- exact Gauss-Newton and hybrid inverse solvers;
- tokenwise, low-rank, causal-anchor, matched-control, and neighbor-target
  ablations;
- local-model judging and campaign reports.

Materialize and install:

```bash
python tools/materialize_environment.py topology
cd workspaces/topology
python -m pip install -e .
pytest -q tests/test_exact_h0_gauss_newton_steering.py \
  tests/test_exact_h0_hybrid_steering.py \
  tests/test_falseqa_steering.py \
  tests/test_falseqa_topology.py
```

The main exact controller is `scripts/run_exact_h0_gauss_newton_steering.py`.
Launch scripts beginning with `launch_exact_h0_` record the evaluated
hyperparameter grids. `scripts/report_exact_h0_perturbation_campaign.py`
builds the aggregate report.

Large token-cloud caches and model outputs are deliberately not tracked. Paths
to those assets must be supplied on the target server.

