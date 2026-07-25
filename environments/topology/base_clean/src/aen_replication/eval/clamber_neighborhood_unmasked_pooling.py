"""Evaluate CLAMBER neighborhood-PH with unmasked mean pooling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aen_replication.config import load_config
from aen_replication.eval import clamber_topology_extensions as ext
from aen_replication.models.generation import render_prompts
from aen_replication.models.hidden_state_extractor import HiddenStateExtractor
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.utils.io_utils import ensure_dir, utc_now_iso, write_json, write_markdown, write_parquet

MODEL_SPECS = {
    "meta_llama_llama_3_1_8b_instruct": {
        "label": "LLaMA 3.1 8B",
        "config_path": "/home/ubuntu/sparse_neurons_ambiguity_replication/configs/runs/llama_clamber.yaml",
    },
}

MASKED_RESULTS = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_topology_extensions/clamber_topology_extensions_results.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-slugs",
        nargs="+",
        default=["meta_llama_llama_3_1_8b_instruct"],
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument("--label-spaces", nargs="+", default=["4way", "9way"], choices=["4way", "9way"])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 14, 31])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--topology-components", type=int, default=6)
    parser.add_argument("--neighborhood-k", type=int, default=24)
    parser.add_argument("--betti-grid-size", type=int, default=32)
    parser.add_argument("--persistence-image-grid-side", type=int, default=4)
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument("--coeff", type=int, default=2)
    parser.add_argument(
        "--cache-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states_unmasked_neighborhood",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_neighborhood_unmasked_pooling",
    )
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def _ensure_unmasked_hidden_root(*, model_slug: str, layers: list[int], force: bool) -> Path:
    spec = MODEL_SPECS[model_slug]
    config = load_config(spec["config_path"])
    extraction_cfg = dict(config["extraction"])
    output_dir = Path(args.cache_root)
    hidden_root = output_dir / model_slug
    expected = [hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet" for layer in layers]
    if not force and all(path.exists() for path in expected):
        return hidden_root

    bundle: HFModelBundle = load_hf_model(config["model"], extraction_cfg)
    pairs_path = Path(config["data"]["pair_output_dir"]) / "clamber_pairs.parquet"
    df = pd.read_parquet(pairs_path)
    extraction_df = df.copy()
    text_column = str(extraction_cfg.get("text_column", "text"))
    use_chat_template = bool(extraction_cfg.get("use_chat_template", False))
    system_prompt = extraction_cfg.get("system_prompt")
    if use_chat_template or system_prompt:
        extraction_df["_rendered_text"] = render_prompts(
            bundle=bundle,
            prompt_texts=df[text_column].astype(str).tolist(),
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            add_generation_prompt=False,
        )
        text_column = "_rendered_text"

    extractor = HiddenStateExtractor(
        bundle=bundle,
        batch_size=int(extraction_cfg.get("batch_size", 4)),
        max_length=int(extraction_cfg.get("max_length", 96)),
        use_mixed_precision=bool(extraction_cfg.get("use_mixed_precision", False)),
    )
    records = extractor.extract_and_save(
        df=extraction_df,
        layers=[int(layer) for layer in layers],
        readouts=["unmasked_mean_pool"],
        text_column=text_column,
        output_dir=output_dir,
        project_root=config["_meta"]["project_root"],
        file_prefix="clamber__",
        metadata_overrides={
            "input_text_column": extraction_cfg.get("text_column", "text"),
            "input_use_chat_template": use_chat_template,
            "input_system_prompt": system_prompt,
            "input_add_generation_prompt": False,
            "readout": "unmasked_mean_pool",
        },
    )
    # Reuse the exact filenames expected by the neighborhood-PH evaluator, but
    # keep them under a separate hidden-root to avoid touching masked caches.
    for record in records:
        source = Path(record.parquet_path)
        source_meta = Path(record.metadata_path)
        target = hidden_root / source.name.replace("__unmasked_mean_pool", "__mean_pool")
        target_meta = hidden_root / source_meta.name.replace("__unmasked_mean_pool", "__mean_pool")
        ensure_dir(target.parent)
        source.replace(target)
        source_meta.replace(target_meta)
    return hidden_root


def _label_order(hidden_root: Path, *, layer: int, label_space: str) -> list[str]:
    meta, matrix = ext.load_hidden_state_table(hidden_root / f"clamber__layer_{int(layer):02d}__mean_pool.parquet")
    meta, _matrix, labels = ext._prepare_label_space(meta, matrix, label_space=label_space)
    del meta, _matrix
    return labels


def _render_report(results_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# CLAMBER Neighborhood-PH Unmasked Pooling",
        "",
        "This reruns the neighborhood-PH CLAMBER experiment with unmasked mean pooling and compares it against the existing masked mean-pool run.",
        "",
    ]
    for (model_label, label_space), group_df in results_df.groupby(["model_label", "label_space"], dropna=False):
        lines.extend([f"## {model_label} / {label_space}", ""])
        for row in group_df.sort_values(["variant", "method"]).to_dict(orient="records"):
            lines.append(
                f"- `{row['variant']}` / `{row['method']}`: accuracy `{row['accuracy']:.4f}`, "
                f"macro-F1 `{row['macro_f1']:.4f}`, selection `{row['selection_signature']}`."
            )
        lines.append("")
    write_markdown(output_path, "\n".join(lines) + "\n")


def main() -> None:
    output_root = ensure_dir(args.output_root)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    masked_df = pd.read_parquet(MASKED_RESULTS)

    for model_slug in args.model_slugs:
        hidden_root = _ensure_unmasked_hidden_root(
            model_slug=model_slug,
            layers=args.layers,
            force=bool(args.force_reextract),
        )
        spec = MODEL_SPECS[model_slug]
        for label_space in args.label_spaces:
            labels = _label_order(hidden_root, layer=int(args.layers[0]), label_space=label_space)
            candidate_df, candidate_cache = ext._candidate_selection_rows(
                hidden_root=hidden_root,
                layers=[int(layer) for layer in args.layers],
                label_space=label_space,
                labels=labels,
                seed=int(args.seed),
                val_fraction=float(args.val_fraction),
                pca_components=int(args.pca_components),
                topology_components=int(args.topology_components),
                neighborhood_k=int(args.neighborhood_k),
                betti_grid_size=int(args.betti_grid_size),
                persistence_image_grid_side=int(args.persistence_image_grid_side),
                maxdim=int(args.maxdim),
                coeff=int(args.coeff),
            )
            candidate_df["model_slug"] = model_slug
            candidate_df["model_label"] = spec["label"]
            candidate_df["label_space"] = label_space
            candidate_df["variant"] = "unmasked"
            candidate_rows.extend(candidate_df.to_dict(orient="records"))

            single_row = candidate_df.loc[candidate_df["method"].eq("neighborhood_ph")].iloc[0]
            single_result = ext._evaluate_final_layer_method(
                hidden_root=hidden_root,
                layer=int(single_row["layer"]),
                label_space=label_space,
                labels=labels,
                method="neighborhood_ph",
                seed=int(args.seed),
                pca_components=int(args.pca_components),
                topology_components=int(args.topology_components),
                neighborhood_k=int(args.neighborhood_k),
                betti_grid_size=int(args.betti_grid_size),
                persistence_image_grid_side=int(args.persistence_image_grid_side),
                maxdim=int(args.maxdim),
                coeff=int(args.coeff),
            )
            multi_result = None
            multi_candidates = ext._multilayer_selection_rows(
                candidate_cache,
                labels=labels,
                seed=int(args.seed),
            )
            if not multi_candidates.empty:
                best_multi = multi_candidates.iloc[0]
                layers = [int(part.strip()) for part in str(best_multi["selection_signature"]).split("|")]
                multi_result = ext._evaluate_final_multilayer(
                    hidden_root=hidden_root,
                    layers=layers,
                    label_space=label_space,
                    labels=labels,
                    seed=int(args.seed),
                    pca_components=int(args.pca_components),
                    topology_components=int(args.topology_components),
                    neighborhood_k=int(args.neighborhood_k),
                    betti_grid_size=int(args.betti_grid_size),
                    persistence_image_grid_side=int(args.persistence_image_grid_side),
                    maxdim=int(args.maxdim),
                    coeff=int(args.coeff),
                )

            masked_subset = masked_df.loc[
                masked_df["model_slug"].eq(model_slug)
                & masked_df["label_space"].eq(label_space)
                & masked_df["method"].isin(["neighborhood_ph", "neighborhood_ph_multilayer"])
            ].copy()
            for row in masked_subset.to_dict(orient="records"):
                row["model_label"] = spec["label"]
                row["variant"] = "masked"
                rows.append(row)
            for row in [single_result] + ([multi_result] if multi_result is not None else []):
                row["model_slug"] = model_slug
                row["model_label"] = spec["label"]
                row["label_space"] = label_space
                row["variant"] = "unmasked"
                rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        ["model_label", "label_space", "method", "variant"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    results_path = output_root / "clamber_neighborhood_unmasked_pooling_results.parquet"
    candidates_path = output_root / "clamber_neighborhood_unmasked_pooling_candidates.parquet"
    report_path = output_root / "clamber_neighborhood_unmasked_pooling_report.md"
    metadata_path = output_root / "clamber_neighborhood_unmasked_pooling_metadata.json"
    write_parquet(results_df, results_path)
    write_parquet(candidates_df, candidates_path)
    _render_report(results_df, report_path)
    write_json(
        metadata_path,
        {
            "created_at": utc_now_iso(),
            "model_slugs": list(args.model_slugs),
            "label_spaces": list(args.label_spaces),
            "layers": [int(layer) for layer in args.layers],
            "seed": int(args.seed),
            "val_fraction": float(args.val_fraction),
            "pca_components": int(args.pca_components),
            "topology_components": int(args.topology_components),
            "neighborhood_k": int(args.neighborhood_k),
            "betti_grid_size": int(args.betti_grid_size),
            "persistence_image_grid_side": int(args.persistence_image_grid_side),
            "maxdim": int(args.maxdim),
            "coeff": int(args.coeff),
            "cache_root": str(args.cache_root),
            "results_path": str(results_path),
            "candidates_path": str(candidates_path),
            "report_path": str(report_path),
        },
    )


if __name__ == "__main__":
    args = parse_args()
    main()
