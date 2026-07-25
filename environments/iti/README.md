# ITI environment

This environment reproduces the original ITI attention-head baseline and adds
target-conditioned perturbation steering over per-head truthfulness probe
statistics. It includes:

- fixed ITI;
- scalar probe-target control;
- headwise direct targets;
- general minimum-norm inverse control;
- group-direction targets with ITI-basis and unrestricted minimum-norm
  mappings;
- standardized and raw probe-score ablations;
- ridge, relative-cap, target-quantile, alpha, and rho sweeps;
- API-free TruthfulQA multiple-choice and local open-ended evaluation.

Materialize the environment:

```bash
python tools/materialize_environment.py iti
cd workspaces/iti
conda env create -f environment.yaml
conda activate iti
```

The upstream environment is retained for paper replication. On newer systems,
an existing PyTorch/Transformers environment can be used if it also provides
`datasets`, `einops`, `pyvene`, `scikit-learn`, and `pytest`.

Extract head activations, then run a sweep from `workspaces/iti`:

```bash
cd get_activations
python get_activations.py --model_name llama_7B --dataset_name tqa_mc2 \
  --head_wise_only
cd ..
bash scripts/run_iti_group_direction_validation.sh
bash scripts/run_iti_group_direction_high_alpha_validation.sh
```

Held-out runners and frozen candidate selectors are under `scripts/` and
`validation/`. The method and current result audit are documented in:

- `validation/CAUSAL_HEAD_PERTURBATION.md`
- `validation/iti_attention_head_results.md`
- `LOCAL_REPRODUCTION.md`

No OpenAI judge is required. Use the multiple-choice evaluator or the packaged
local judge for open-ended generations.

