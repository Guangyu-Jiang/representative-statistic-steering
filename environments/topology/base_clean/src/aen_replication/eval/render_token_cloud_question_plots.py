"""Render single-question token-cloud topology comparisons (ambiguous vs clear)."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aen_replication.config import load_config
from aen_replication.models.generation import render_prompts
from aen_replication.models.hf_model import HFModelBundle, load_hf_model
from aen_replication.train.independent_topology_classifier import _compute_diagrams
from aen_replication.train.token_cloud_topology_classifier import _extract_train_token_matrices, _fit_layer_reducers
from aen_replication.utils.io_utils import ensure_dir, write_markdown


LABEL_COLORS = {"ambiguous": "#c44e52", "clear": "#4c72b0"}


def _slugify(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _prepare_prompt_frame(
    df: pd.DataFrame,
    *,
    bundle: HFModelBundle,
    text_column: str,
    use_chat_template: bool,
    system_prompt: str | None,
) -> tuple[pd.DataFrame, str]:
    if not use_chat_template and not system_prompt:
        return df.copy(), text_column
    prompt_df = df.copy()
    prompt_df["_rendered_text"] = render_prompts(
        bundle=bundle,
        prompt_texts=df[text_column].astype(str).tolist(),
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        add_generation_prompt=False,
    )
    return prompt_df, "_rendered_text"


def _valid_token_mask(
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    *,
    special_ids: set[int],
    drop_special_tokens: bool,
) -> torch.Tensor:
    valid = attention_mask_row.bool().clone()
    if drop_special_tokens and special_ids:
        special_mask = torch.zeros_like(valid)
        for special_id in special_ids:
            special_mask |= input_ids_row.eq(int(special_id))
        valid &= ~special_mask
    if int(valid.sum().item()) == 0:
        valid = attention_mask_row.bool()
    return valid


def _extract_single_cloud(
    *,
    bundle: HFModelBundle,
    text: str,
    layer: int,
    reducer: Any,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    encoder = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    max_length = int(config.get("max_length", 96))
    topology_dim = int(config.get("topology_components", config.get("pca_components", 8)))
    drop_special_tokens = bool(config.get("drop_special_tokens", True))
    special_ids = set(int(token_id) for token_id in getattr(encoder, "all_special_ids", []) if token_id is not None)

    encoded = encoder(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"].to(device)
    input_ids = encoded["input_ids"].to(device)
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states for token-cloud visualization.")
    layer_output = hidden_states[layer + 1].detach().float().cpu()
    input_ids_cpu = input_ids.detach().cpu()
    attention_mask_cpu = attention_mask.detach().cpu()
    valid = _valid_token_mask(
        input_ids_cpu[0],
        attention_mask_cpu[0],
        special_ids=special_ids,
        drop_special_tokens=drop_special_tokens,
    )
    token_vectors = layer_output[0][valid].numpy()
    reduced = reducer.transform(token_vectors).astype(np.float32, copy=False)
    reduced_dim = min(int(topology_dim), int(reducer.n_components_))
    cloud_topology = reduced[:, :reduced_dim]
    if reduced.shape[1] >= 2:
        cloud_2d = reduced[:, :2]
    else:
        cloud_2d = np.column_stack([reduced[:, 0], np.zeros(len(reduced), dtype=float)])
    return cloud_topology, cloud_2d


def _plot_point_cloud(ax: plt.Axes, cloud_2d: np.ndarray, *, title: str, color: str) -> None:
    ax.scatter(cloud_2d[:, 0], cloud_2d[:, 1], s=12, alpha=0.7, color=color)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return diagram
    finite_mask = np.isfinite(diagram[:, 1])
    return diagram[finite_mask]


def _plot_persistence_diagram(ax: plt.Axes, diagram: np.ndarray, *, title: str, color: str) -> float:
    diagram = _finite_diagram(diagram)
    if diagram.size:
        ax.scatter(diagram[:, 0], diagram[:, 1], s=16, alpha=0.75, color=color)
    max_value = float(np.max(diagram)) if diagram.size else 1.0
    if not np.isfinite(max_value):
        max_value = 1.0
    ax.plot([0, max_value], [0, max_value], color="#666666", linewidth=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.grid(True, alpha=0.2)
    return max_value


def _plot_barcode(
    ax: plt.Axes,
    diagram: np.ndarray,
    *,
    title: str,
    color: str,
    top_n: int,
    xlim: tuple[float, float],
) -> None:
    ax.set_title(title)
    diagram = _finite_diagram(diagram)
    if diagram.size == 0:
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Feature")
        ax.grid(True, alpha=0.2)
        return
    persistence = diagram[:, 1] - diagram[:, 0]
    order = np.argsort(-persistence)[:top_n]
    selected = diagram[order]
    for idx, (birth, death) in enumerate(selected):
        ax.plot([birth, death], [idx, idx], color=color, linewidth=2.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(-1, len(selected) + 1)
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Feature")
    ax.grid(True, alpha=0.2)


def _write_report(
    *,
    report_path: Path,
    model_name: str,
    dataset: str,
    layer: int,
    ambig_text: str,
    clear_text: str,
    outputs: dict[str, Path],
) -> None:
    lines = [
        "# Token-Cloud Question-Level Topology Plots",
        "",
        f"- Model: `{model_name}`",
        f"- Dataset: `{dataset}`",
        f"- Layer: `{layer}`",
        "",
        "## Selected Questions",
        "",
        "### Ambiguous",
        "",
        "```text",
        textwrap.fill(ambig_text, width=88),
        "```",
        "",
        "### Clear",
        "",
        "```text",
        textwrap.fill(clear_text, width=88),
        "```",
        "",
        "## Plots",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    write_markdown(report_path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="ambigqa")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--ambig-index", type=int, default=0)
    parser.add_argument("--clear-index", type=int, default=0)
    parser.add_argument("--ambig-example-id", default=None)
    parser.add_argument("--clear-example-id", default=None)
    parser.add_argument("--output-root", default="artifacts/token_cloud_question_plots")
    args = parser.parse_args()

    config = load_config(args.config)
    classifier_cfg = dict(config["token_cloud_topology_classifier"])
    model_name = str(config["model"]["name"])
    bundle = load_hf_model(config["model"], classifier_cfg)

    dataset_name = str(args.dataset)
    pair_output_dir = Path(config["data"]["pair_output_dir"])
    data_path = pair_output_dir / f"{dataset_name}_pairs.parquet"
    df = pd.read_parquet(data_path)

    prepared_df, text_column = _prepare_prompt_frame(
        df,
        bundle=bundle,
        text_column=str(classifier_cfg.get("text_column", "text")),
        use_chat_template=bool(classifier_cfg.get("use_chat_template", False)),
        system_prompt=classifier_cfg.get("system_prompt"),
    )
    prepared_df["_token_cloud_text"] = prepared_df[text_column]

    test_df = prepared_df.loc[prepared_df["split"].eq("test")].copy()
    ambig_df = test_df.loc[test_df["label_ambiguous"].eq(1)].sort_values("example_id")
    clear_df = test_df.loc[test_df["label_ambiguous"].eq(0)].sort_values("example_id")
    if ambig_df.empty or clear_df.empty:
        raise ValueError("Need both ambiguous and clear examples in the test split.")

    if args.ambig_example_id:
        ambig_matches = ambig_df.loc[ambig_df["example_id"].eq(str(args.ambig_example_id))]
        if ambig_matches.empty:
            raise ValueError(f"Ambiguous example_id not found in test split: {args.ambig_example_id}")
        ambig_row = ambig_matches.iloc[0]
    else:
        ambig_row = ambig_df.iloc[int(args.ambig_index) % len(ambig_df)]

    if args.clear_example_id:
        clear_matches = clear_df.loc[clear_df["example_id"].eq(str(args.clear_example_id))]
        if clear_matches.empty:
            raise ValueError(f"Clear example_id not found in test split: {args.clear_example_id}")
        clear_row = clear_matches.iloc[0]
    else:
        clear_row = clear_df.iloc[int(args.clear_index) % len(clear_df)]

    train_df = prepared_df.loc[prepared_df["split"].eq("train")].copy().reset_index(drop=True)
    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=[int(args.layer)],
        config={**classifier_cfg, "_seed": int(config.get("seed", 0))},
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_cfg, seed=int(config.get("seed", 0)))
    reducer = reducers[int(args.layer)]

    ambig_cloud, ambig_cloud_2d = _extract_single_cloud(
        bundle=bundle,
        text=str(ambig_row["_token_cloud_text"]),
        layer=int(args.layer),
        reducer=reducer,
        config=classifier_cfg,
    )
    clear_cloud, clear_cloud_2d = _extract_single_cloud(
        bundle=bundle,
        text=str(clear_row["_token_cloud_text"]),
        layer=int(args.layer),
        reducer=reducer,
        config=classifier_cfg,
    )

    maxdim = int(classifier_cfg.get("maxdim", 1))
    coeff = int(classifier_cfg.get("coeff", 2))
    distance_metric = str(classifier_cfg.get("distance_metric", "euclidean"))
    ambig_diagrams = _compute_diagrams(ambig_cloud, maxdim=maxdim, coeff=coeff, distance_metric=distance_metric)
    clear_diagrams = _compute_diagrams(clear_cloud, maxdim=maxdim, coeff=coeff, distance_metric=distance_metric)

    output_root = ensure_dir(Path(args.output_root) / _slugify(model_name) / dataset_name)
    point_cloud_path = output_root / f"layer_{int(args.layer):02d}__point_clouds.png"
    h0_path = output_root / f"layer_{int(args.layer):02d}__h0_diagram_barcode.png"
    h1_path = output_root / f"layer_{int(args.layer):02d}__h1_diagram_barcode.png"
    report_path = output_root / "token_cloud_question_plots.md"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    _plot_point_cloud(
        axes[0],
        ambig_cloud_2d,
        title="Ambiguous token cloud",
        color=LABEL_COLORS["ambiguous"],
    )
    _plot_point_cloud(
        axes[1],
        clear_cloud_2d,
        title="Clear token cloud",
        color=LABEL_COLORS["clear"],
    )
    fig.suptitle(f"{model_name} {dataset_name} layer {int(args.layer)} token clouds", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(point_cloud_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    def _diagram_barcode_figure(dim: int, output_path: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
        ambig_diagram = ambig_diagrams[dim] if dim < len(ambig_diagrams) else np.zeros((0, 2), dtype=float)
        clear_diagram = clear_diagrams[dim] if dim < len(clear_diagrams) else np.zeros((0, 2), dtype=float)
        ambig_max = _plot_persistence_diagram(
            axes[0, 0],
            ambig_diagram,
            title=f"Ambiguous H{dim} diagram",
            color=LABEL_COLORS["ambiguous"],
        )
        clear_max = _plot_persistence_diagram(
            axes[0, 1],
            clear_diagram,
            title=f"Clear H{dim} diagram",
            color=LABEL_COLORS["clear"],
        )
        max_value = max(ambig_max, clear_max, 1.0)
        xlim = (0.0, max_value * 1.05)
        _plot_barcode(
            axes[1, 0],
            ambig_diagram,
            title=f"Ambiguous H{dim} barcode",
            color=LABEL_COLORS["ambiguous"],
            top_n=20,
            xlim=xlim,
        )
        _plot_barcode(
            axes[1, 1],
            clear_diagram,
            title=f"Clear H{dim} barcode",
            color=LABEL_COLORS["clear"],
            top_n=20,
            xlim=xlim,
        )
        fig.suptitle(f"{model_name} {dataset_name} layer {int(args.layer)} H{dim} topology", y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    _diagram_barcode_figure(0, h0_path)
    _diagram_barcode_figure(1, h1_path)

    _write_report(
        report_path=report_path,
        model_name=model_name,
        dataset=dataset_name,
        layer=int(args.layer),
        ambig_text=str(ambig_row["_token_cloud_text"]),
        clear_text=str(clear_row["_token_cloud_text"]),
        outputs={
            "Point cloud comparison": point_cloud_path,
            "H0 diagram + barcode": h0_path,
            "H1 diagram + barcode": h1_path,
        },
    )


if __name__ == "__main__":
    main()
