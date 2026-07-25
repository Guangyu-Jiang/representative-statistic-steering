# Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs

This repository is a fresh replication project for the paper "Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs". I did not find a public implementation linked from the paper or ACL page, so this codebase rebuilds the experiment from the paper plus official public dataset sources.

## Sources Used

- Paper: https://aclanthology.org/2025.emnlp-main.813.pdf
- AmbigQA official repo: `external/AmbigQA`
- SituatedQA official repo: `external/SituatedQA`

## Implemented Scope

- Official-data contrastive pair construction for AmbigQA and SituatedQA.
- Mean-pooled prefill hidden-state extraction.
- Logistic ambiguity probes at the paper default layer (`l = 14`) and layerwise sweeps.
- AEN identification via top-k perturbation.
- AEN-only probes and cross-domain evaluation.
- All-layer ambiguity-feature export for later TDA, including multiple subspaces per layer (`aed_final`, `top_20`, `top_50`, `top_100`, `nonzero_support`), layerwise point clouds, and per-example trajectories.
- Local topology-based ambiguity classifier using per-question persistent-homology descriptors from kNN neighborhoods in the ambiguity subspace, with both single-layer and fused multi-layer variants.
- Independent raw-hidden-state topology classifier that fits unsupervised PCA per layer and builds geometry/topology features without using AEN or probe-derived coordinates.
- Prompt baselines from Appendix B: CLAM, CLAMBER, and INFOGAIN.
- Local steering pipeline with both a deterministic fallback judge and a 4-GPU local LLM judge path.
- Reproducible steering experiment commands for IBM-style, topology-conditioned, direct topology-surrogate, soft-H0, and CLAMBER prediction-conditioned runs in `docs/STEERING_EXPERIMENTS.md`.

## In Progress

- Full local-judge steering runs and prompt-baseline executions for all model/dataset combinations.

## Quickstart

```bash
cd /home/ubuntu/sparse_neurons_ambiguity_replication
PYTHONPATH=src python -m aen_replication.pipelines.prepare_data --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.extract_hidden_states --config configs/default.yaml --dataset ambigqa
PYTHONPATH=src python -m aen_replication.pipelines.extract_hidden_states --config configs/default.yaml --dataset situatedqa
PYTHONPATH=src python -m aen_replication.pipelines.train_detection --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.export_tda_features --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_prompt_baselines --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_steering --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.generate_paper_audit --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_persistent_homology --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_focused_persistent_homology --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_ph_visualizations --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_ph_descriptors --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_mapper_analysis --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_topology_classifier --config configs/default.yaml
PYTHONPATH=src python -m aen_replication.pipelines.run_independent_topology_classifier --config configs/default.yaml
```

## Steering Reruns

See `docs/STEERING_EXPERIMENTS.md` for the exact commands used to rerun the current steering experiments on another GPU server, including OpenAI judge commands and the required artifact inputs.

## Local Judge

- Default local judge model: `Qwen/Qwen2.5-72B-Instruct`
- Runtime mode: `device_map: auto` across 4 GPUs
- Fallback judge: deterministic rules-based classifier in `src/aen_replication/eval/judge.py`

## TDA Export Artifacts

- `artifacts/features/<model_slug>/layerwise_ambiguity_features.parquet`
- `artifacts/features/<model_slug>/ambiguity_trajectories.parquet`
- `artifacts/features/<model_slug>/ambiguity_trajectory_summary.parquet`
- `artifacts/features/<model_slug>/layerwise_aen_summary.parquet`
- `artifacts/features/<model_slug>/cross_dataset_layer_overlap.parquet`

The layerwise, trajectory, and overlap tables now include `subspace_name` so TDA can be run on the compact final AEDs or on larger ambiguity-related subspaces such as `top_50`, `top_100`, and `nonzero_support`.

## Persistent-Homology Comparison Visualizations

- `artifacts/topology/<model_slug>/persistent_homology_summary.parquet`
- `artifacts/topology_focused/<model_slug>/focused_persistent_homology_distances.parquet`
- `artifacts/topology_visualizations/<model_slug>/ph_label_comparison_aggregate.parquet`
- `artifacts/topology_visualizations/<model_slug>/ph_label_comparison_summary.md`
- `artifacts/topology_visualizations/<model_slug>/plots/`

The visualization stage renders ambiguous-vs-clear layer traces for `H0` and `H1` total persistence and feature counts, plus delta heatmaps and focused Wasserstein heatmaps derived from the saved PH artifacts.

## Extended TDA Outputs

- `artifacts/topology_descriptors/<model_slug>/ph_descriptor_summary.parquet`
- `artifacts/topology_descriptors/<model_slug>/ph_betti_curves.parquet`
- `artifacts/topology_descriptors/<model_slug>/ph_descriptor_summary.md`
- `artifacts/topology_classifier/<model_slug>/topology_classifier_candidate_metrics.parquet`
- `artifacts/topology_classifier/<model_slug>/topology_classifier_final_metrics.parquet`
- `artifacts/topology_classifier/<model_slug>/topology_classifier_summary.md`
- `artifacts/independent_topology_classifier/<model_slug>/independent_topology_classifier_candidate_metrics.parquet`
- `artifacts/independent_topology_classifier/<model_slug>/independent_topology_classifier_final_metrics.parquet`
- `artifacts/independent_topology_classifier/<model_slug>/independent_topology_classifier_summary.md`
- `artifacts/mapper/<model_slug>/mapper_stats.parquet`
- `artifacts/mapper/<model_slug>/mapper_summary.md`
- `artifacts/mapper/<model_slug>/plots/`

The PH descriptor stage adds persistence entropy, mean lifetime, normalized Betti-curve area, and representative Betti-curve visualizations. The topology-classifier stage turns local PH neighborhoods into per-question features, then compares topology-only, geometry-only, and hybrid ambiguity classifiers in both single-layer and fused multi-layer settings. The independent classifier stage repeats that exercise directly from raw hidden states using unsupervised PCA coordinates, so it can be compared against AEN-based detection without reusing the ambiguity probe. The Mapper stage summarizes graph complexity across layers using simple lenses (`signed_distance`, `z_0`) and renders layerwise graph-statistic plots plus representative graph views.

## Paper Audit

- Audit report: `artifacts/reports/paper_audit/replication_audit.md`
- Recreated tables: `artifacts/reports/paper_audit/tables/`
- Recreated figures: `artifacts/reports/paper_audit/figures/`

The audit checks the local paper PDF against the currently available artifacts, marks exact versus partial replications, and highlights missing items such as the unrun TriviaQA side-effect evaluation.
