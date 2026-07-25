# FalseQA topology-local steering

This experiment evaluates false-premise rebuttal on the official FalseQA
question/correction pairs. It does not call an external API.

## Method

The pair-grouped random split uses 80% of FalseQA pairs for training and 20%
for evaluation. PCA reducers and topology standardization are fit on training
questions only. For each question, the neighbor representation concatenates
the following statistics from every decoder layer:

- `h0_mean_persistence`
- `h0_persistence_entropy`
- `h0_top5_persistence_fraction`

The topology output layer is selected using a validation subset of training
pairs. If H0 is measured on decoder output \(H^{(\ell)}\), that exact tensor is
captured and patched as the input to decoder layer \(\ell+1\). The paired
activation contrast for training pair \(i\) is computed from its
attention-mask mean-pooled prompt representation:

\[
v_i^{(\ell)} = h_{i,\mathrm{false}}^{(\ell)} -
h_{i,\mathrm{corrected}}^{(\ell)}.
\]

The global baseline averages all training contrasts. The topology-local
method finds the \(k\) nearest training false-premise questions in standardized
all-layer H0 space and averages only their paired contrasts. The selected
vector is multiplied by `alpha` and added to the target decoder-layer input at
every prompt token and every decode step. Generation uses temperature 0.1.

The local Qwen judge assigns `GROUNDED_REBUTTAL`, `GENERIC_REJECTION`,
`PREMISE_ACCEPTANCE`, or `NEITHER`, using the FalseQA reference rebuttal as
evidence. The `false_premise_reference_grounded_fourway_evidence_v4` prompt
requires an exact correction phrase, or `NONE`, before the label. Reports
retain that Qwen label and add a local DeBERTa reference-entailment score for
diagnosis. The NLI score is not a hard gate by default because several FalseQA
references are noisy conjunctions that reject valid atomic corrections. An
optional `--enforce-nli-gate` flag enables the conservative gate for ablation.
Reports include Qwen and NLI-supported rates, response validity, uniqueness,
relative activation change, conversion of base premise acceptances to grounded
rebuttals, and retention of already-grounded base responses.
The reported relative activation change is the prompt intervention Frobenius
norm divided by the unsteered prompt layer-input Frobenius norm; decode-time
steps are not included in that denominator.

Hyperparameters are selected on a fixed 64-question pilot subset of the 473
held-out false-premise questions. Pilot generations, judgments, and IDs are
frozen under each model's `pilot64` artifacts. The selected global and
topology-local settings are then evaluated on the disjoint remaining 409
questions. The report includes a paired bootstrap confidence interval and an
exact McNemar test for the grounded-rebuttal difference.

## Commands

The topology classification artifacts must already exist under
`artifacts/falseqa_topology_classification/<model_slug>`.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_falseqa_topology_steering.py \
  --config configs/runs/llama_steering_paper_style.yaml \
  --protocol random80 \
  --neighbor-ks 5 20 50 \
  --alphas 0.5 1 2 4 8 \
  --eval-n 0 \
  --temperature 0.1
```

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/judge_falseqa_topology_steering_local.py \
  --model-slug meta_llama_llama_3_1_8b_instruct \
  --protocol random80 \
  --batch-size 16
```

The frozen pilot outputs can be rejudged independently:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/judge_falseqa_topology_steering_local.py \
  --model-slug meta_llama_llama_3_1_8b_instruct \
  --protocol random80 \
  --raw-subdir pilot64/raw \
  --summary-stem pilot64_local_judge_summary \
  --batch-size 16 \
  --force
```

```bash
PYTHONPATH=src python scripts/report_falseqa_topology_steering.py \
  --protocol random80 \
  --expected-n 473
```

Raw generations and per-question judge outputs are retained under
`artifacts/falseqa_topology_steering_aligned/<model_slug>/random80/raw`, allowing the
same generations to be reevaluated with another local judge.
