# Protocol-aligned CAA replication

This directory isolates an API-free replication of the original CAA evaluation.
It deliberately leaves the original scripts and existing perturbation artifacts
unchanged.

## Aligned settings

- Llama-2-7B-Chat, decoder layer 13, in float32.
- Llama-2-13B-Chat, decoder layer 14, in float16.
- Original manual Llama-2 `[INST] ... [/INST]` prompt format.
- The vector assets behind the paper results: norm-corrected vectors for
  sycophancy and survival instinct, whose regenerated datasets had unusually
  small raw vector norms, and raw mean-difference vectors for the other five
  behaviors. `--vector-source raw` and `--vector-source normalized` retain the
  two internally consistent alternatives as explicit ablations.
- Vector addition to the decoder-block output starting at the final `[/INST]`
  token and continuing through every generated token.
- A/B multipliers: `-1 -0.5 0 0.5 1`.
- A/B system conditions: no system prompt, positive system prompt, and negative
  system prompt.
- Open-ended multipliers: `-2 -1.5 -1 0 1 1.5 2`.
- Every official test item: 50 per behavior, except the released sycophancy
  open-ended file, which contains 53.
- Greedy open-ended generation with `max_new_tokens=100` and `top_k=1`.
- Original open-ended scoring instructions and 0-10 scale.

The only intended semantic evaluation change is replacing GPT-4 with the local
`Qwen/Qwen2.5-7B-Instruct` judge. The canonical Meta checkpoints are gated for
the configured account, so the runner defaults to locally cached
`NousResearch/Llama-2-{7b,13b}-chat-hf` mirrors and records this provenance in
the manifest.

## Run

```bash
python -m caa_exact_replication.run \
  --model-size 7b --task all --device cuda:0

python -m caa_exact_replication.run \
  --model-size 13b --task all --device cuda:1

python -m caa_exact_replication.local_judge \
  --device cuda:2

python -m caa_exact_replication.summarize
```

Use `--behaviors` to split one model across GPUs. Runs are resumable: completed
JSON files are skipped unless `--overwrite` is supplied. All outputs are placed
under `artifacts/caa_exact_replication/`.

For a one-example smoke test, use a separate output directory:

```bash
python -m caa_exact_replication.run \
  --model-size 7b --task all --behaviors sycophancy \
  --system-prompts none --ab-multipliers 0 \
  --open-multipliers 0 --limit 1 --device cuda:0 \
  --output-root artifacts/caa_exact_replication_smoke
```
