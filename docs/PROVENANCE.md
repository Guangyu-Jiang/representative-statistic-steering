# Baseline provenance

The package was assembled on 2026-07-25 from the following revisions:

| Component | Repository | Revision | Packaging |
|---|---|---|---|
| PPLM | `uber-research/PPLM` | `e236b8989322128360182d29a79944627957ad47` | submodule |
| TruthX | `ictnlp/TruthX` | `a41093a6ae3bcbcb523759da782de0f329d03d91` | submodule |
| Lookback Lens | `voidism/Lookback-Lens` | `e0a1fa3a898fbf6512af7be5567dea8ffe7a6620` | submodule |
| ReDeEP | `Jeryi-Sun/ReDEeP-ICLR` | `4d081915b8fb4430fda65c411da61540cc73cc57` | submodule |
| ITI | `likenneth/honest_llama` | `2c6b2179be7b5aa8f0a171688cf9e01b812ca327` | submodule plus patch/overlay |
| TruthfulQA evaluator | `sylinrl/TruthfulQA` | `d71c110897f5d31c5d7f309e7bc316c152f6f031` | submodule copied into ITI workspace |
| CAA | `nrimsky/CAA` | `5dabbbd9a0bca5f25e174501e959de378806aa48` | compact vendored source plus patch/overlay |
| Topology | `Guangyu-Jiang/sparse-neurons-ambiguity-replication` | `8166e54b24fbe594ecddf45420fde280d79510ec` | compact vendored source plus patch/overlay |

The upstream licenses are retained in submodules or vendored base trees where
provided. Patches contain only project-specific modifications. Generated
artifacts and model checkpoints are not part of the source package.
