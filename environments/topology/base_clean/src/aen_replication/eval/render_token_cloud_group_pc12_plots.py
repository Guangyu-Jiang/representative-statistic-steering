"""Render group-level PC1/PC2 token-cloud comparisons for ambiguous vs clear."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aen_replication.config import load_config
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.token_cloud_topology_classifier import (
    _extract_reduced_clouds,
    _extract_train_token_matrices,
    _fit_layer_reducers,
    _prepare_prompt_frame,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown


LABEL_NAME = {1: "Ambiguous", 0: "Clear"}
LABEL_COLOR = {1: "#c44e52", 0: "#4c72b0"}
LABEL_CMAP = {1: "Reds", 0: "Blues"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _load_best_single_layers(
    *,
    classifier_output_root: Path,
    model_name: str,
    datasets: list[str],
    final_metrics_filename: str,
) -> dict[str, int]:
    final_path = classifier_output_root / _model_slug(model_name) / final_metrics_filename
    final_df = pd.read_parquet(final_path)
    layers: dict[str, int] = {}
    for dataset in datasets:
        subset = final_df.loc[
            final_df["dataset"].eq(dataset) & final_df["feature_set"].eq("topology_only")
        ].copy()
        if subset.empty:
            raise ValueError(f"No topology_only row found for dataset {dataset} in {final_path}")
        layers[dataset] = int(subset.iloc[0]["layer"])
    return layers


def _prepare_full_frame(config: dict[str, Any], classifier_config: dict[str, Any], bundle: Any, datasets: list[str]) -> pd.DataFrame:
    pair_output_dir = _resolve_path(str(config["data"]["pair_output_dir"]))
    text_column = str(classifier_config.get("text_column", "text"))
    use_chat_template = bool(classifier_config.get("use_chat_template", False))
    system_prompt = classifier_config.get("system_prompt")
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        path = pair_output_dir / f"{dataset}_pairs.parquet"
        dataset_df = pd.read_parquet(path)
        prepared_df, prepared_text_column = _prepare_prompt_frame(
            dataset_df,
            bundle=bundle,
            text_column=text_column,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        prepared_df["_token_cloud_text"] = prepared_df[prepared_text_column]
        frames.append(prepared_df)
    return pd.concat(frames, ignore_index=True)


def _ensure_2d(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 2:
        return points[:, :2]
    return np.column_stack([points[:, 0], np.zeros(len(points), dtype=float)])


def _build_group_frames(
    subset: pd.DataFrame,
    *,
    max_questions_per_class: int,
    tokens_per_question: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    token_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []

    for label_value in [1, 0]:
        label_df = subset.loc[subset["label_ambiguous"].eq(label_value)].copy().sort_values("example_id")
        if max_questions_per_class > 0 and len(label_df) > max_questions_per_class:
            selected = np.sort(rng.choice(len(label_df), size=max_questions_per_class, replace=False))
            label_df = label_df.iloc[selected]
        for row in label_df.to_dict(orient="records"):
            cloud = _ensure_2d(np.asarray(row["cloud"], dtype=float))
            if len(cloud) == 0:
                continue
            sample_size = min(tokens_per_question, len(cloud))
            if len(cloud) > sample_size:
                sample_idx = np.sort(rng.choice(len(cloud), size=sample_size, replace=False))
                sampled = cloud[sample_idx]
            else:
                sampled = cloud
            for point in sampled:
                token_rows.append(
                    {
                        "example_id": str(row["example_id"]),
                        "label_ambiguous": int(label_value),
                        "pc1": float(point[0]),
                        "pc2": float(point[1]),
                    }
                )
            centroid = cloud.mean(axis=0)
            radius = float(np.sqrt(np.mean(np.sum((cloud - centroid) ** 2, axis=1))))
            centroid_rows.append(
                {
                    "example_id": str(row["example_id"]),
                    "label_ambiguous": int(label_value),
                    "pc1": float(centroid[0]),
                    "pc2": float(centroid[1]),
                    "radius": radius,
                    "token_count": int(len(cloud)),
                }
            )
    return pd.DataFrame(token_rows), pd.DataFrame(centroid_rows)


def _shared_limits(frames: list[pd.DataFrame]) -> tuple[tuple[float, float], tuple[float, float]]:
    combined = pd.concat(frames, ignore_index=True)
    x = combined["pc1"].to_numpy(dtype=float)
    y = combined["pc2"].to_numpy(dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = max((x_max - x_min) * 0.06, 1e-3)
    y_pad = max((y_max - y_min) * 0.06, 1e-3)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _plot_token_density(
    token_df: pd.DataFrame,
    *,
    dataset: str,
    layer: int,
    model_name: str,
    output_path: Path,
    tokens_per_question: int,
) -> None:
    ambig = token_df.loc[token_df["label_ambiguous"].eq(1)].copy()
    clear = token_df.loc[token_df["label_ambiguous"].eq(0)].copy()
    xlim, ylim = _shared_limits([ambig, clear])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
    for axis, frame, label_value in [(axes[0], ambig, 1), (axes[1], clear, 0)]:
        hb = axis.hexbin(
            frame["pc1"].to_numpy(dtype=float),
            frame["pc2"].to_numpy(dtype=float),
            gridsize=34,
            cmap=LABEL_CMAP[label_value],
            mincnt=1,
            bins="log",
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_title(f"{LABEL_NAME[label_value]} token density")
        axis.set_xlabel("PC1")
        axis.grid(True, alpha=0.18)
        fig.colorbar(hb, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("PC2")
    fig.suptitle(
        f"{model_name}: {dataset} token density in PC1/PC2 at layer {layer}",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        f"Balanced sampling: up to {tokens_per_question} tokens per question, shared axes across classes",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_question_centroids(
    centroid_df: pd.DataFrame,
    *,
    dataset: str,
    layer: int,
    model_name: str,
    output_path: Path,
) -> None:
    ambig = centroid_df.loc[centroid_df["label_ambiguous"].eq(1)].copy()
    clear = centroid_df.loc[centroid_df["label_ambiguous"].eq(0)].copy()
    xlim, ylim = _shared_limits([ambig, clear])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for frame, label_value in [(clear, 0), (ambig, 1)]:
        ax.scatter(
            frame["pc1"].to_numpy(dtype=float),
            frame["pc2"].to_numpy(dtype=float),
            s=16,
            alpha=0.28,
            color=LABEL_COLOR[label_value],
            label=LABEL_NAME[label_value],
        )
        mean_point = frame.loc[:, ["pc1", "pc2"]].mean().to_numpy(dtype=float)
        ax.scatter(
            [mean_point[0]],
            [mean_point[1]],
            s=110,
            color=LABEL_COLOR[label_value],
            edgecolor="white",
            linewidth=1.2,
            marker="X",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{model_name}: {dataset} question-cloud centroids at layer {layer}")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    output_path: Path,
    model_name: str,
    dataset_layers: dict[str, int],
    token_plot_paths: dict[str, Path],
    centroid_plot_paths: dict[str, Path],
    centroid_df_map: dict[str, pd.DataFrame],
    token_df_map: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Token-Cloud Group PC1/PC2 Plots",
        "",
        f"- Model: `{model_name}`",
        "",
    ]
    for dataset, layer in dataset_layers.items():
        token_df = token_df_map[dataset]
        centroid_df = centroid_df_map[dataset]
        ambig_q = int(centroid_df["label_ambiguous"].eq(1).sum())
        clear_q = int(centroid_df["label_ambiguous"].eq(0).sum())
        ambig_t = int(token_df["label_ambiguous"].eq(1).sum())
        clear_t = int(token_df["label_ambiguous"].eq(0).sum())
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Layer: `{layer}`",
                f"- Questions plotted: ambiguous `{ambig_q}`, clear `{clear_q}`",
                f"- Sampled tokens plotted: ambiguous `{ambig_t}`, clear `{clear_t}`",
                f"- Token density plot: `{token_plot_paths[dataset]}`",
                f"- Question centroid plot: `{centroid_plot_paths[dataset]}`",
                "",
            ]
        )
    write_markdown(output_path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--max-questions-per-class", type=int, default=1000)
    parser.add_argument("--tokens-per-question", type=int, default=12)
    parser.add_argument("--output-root", default="artifacts/token_cloud_group_plots")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_config(args.config)
    classifier_config = dict(config["token_cloud_topology_classifier"])
    model_name = str(config["model"]["name"])
    bundle = load_hf_model(config["model"], classifier_config)

    datasets = list(args.datasets) if args.datasets else list(classifier_config.get("datasets", ["ambigqa"]))
    classifier_output_root = _resolve_path(str(classifier_config.get("output_dir", "artifacts/token_cloud_topology_classifier_all_datasets")))
    final_metrics_filename = str(classifier_config.get("final_metrics_filename", "token_cloud_topology_final_metrics.parquet"))
    if str(args.layer).lower() == "auto":
        dataset_layers = _load_best_single_layers(
            classifier_output_root=classifier_output_root,
            model_name=model_name,
            datasets=datasets,
            final_metrics_filename=final_metrics_filename,
        )
    else:
        layer = int(args.layer)
        dataset_layers = {dataset: layer for dataset in datasets}

    full_df = _prepare_full_frame(config, classifier_config, bundle, datasets)
    train_df = full_df.loc[full_df["split"].eq("train")].copy().reset_index(drop=True)
    layers = sorted(set(dataset_layers.values()))
    token_matrices = _extract_train_token_matrices(
        bundle=bundle,
        train_df=train_df,
        text_column="_token_cloud_text",
        layers=layers,
        config={**classifier_config, "_seed": int(args.seed)},
    )
    reducers = _fit_layer_reducers(token_matrices, config=classifier_config, seed=int(args.seed))
    cloud_df = _extract_reduced_clouds(
        bundle=bundle,
        df=full_df,
        text_column="_token_cloud_text",
        layers=layers,
        reducers=reducers,
        config=classifier_config,
    )

    output_root = ensure_dir(_resolve_path(str(args.output_root)) / _model_slug(model_name))
    token_plot_paths: dict[str, Path] = {}
    centroid_plot_paths: dict[str, Path] = {}
    centroid_df_map: dict[str, pd.DataFrame] = {}
    token_df_map: dict[str, pd.DataFrame] = {}

    for dataset in datasets:
        layer = dataset_layers[dataset]
        subset = cloud_df.loc[
            cloud_df["dataset"].eq(dataset) & cloud_df["split"].eq("test") & cloud_df["layer"].eq(layer)
        ].copy()
        token_df, centroid_df = _build_group_frames(
            subset,
            max_questions_per_class=int(args.max_questions_per_class),
            tokens_per_question=int(args.tokens_per_question),
            seed=int(args.seed),
        )
        dataset_root = ensure_dir(output_root / dataset)
        token_plot_path = dataset_root / f"layer_{layer:02d}__token_density_pc12.png"
        centroid_plot_path = dataset_root / f"layer_{layer:02d}__question_centroids_pc12.png"
        _plot_token_density(
            token_df,
            dataset=dataset,
            layer=layer,
            model_name=model_name,
            output_path=token_plot_path,
            tokens_per_question=int(args.tokens_per_question),
        )
        _plot_question_centroids(
            centroid_df,
            dataset=dataset,
            layer=layer,
            model_name=model_name,
            output_path=centroid_plot_path,
        )
        token_plot_paths[dataset] = token_plot_path
        centroid_plot_paths[dataset] = centroid_plot_path
        centroid_df_map[dataset] = centroid_df
        token_df_map[dataset] = token_df

    report_path = output_root / "token_cloud_group_pc12_plots.md"
    _write_report(
        output_path=report_path,
        model_name=model_name,
        dataset_layers=dataset_layers,
        token_plot_paths=token_plot_paths,
        centroid_plot_paths=centroid_plot_paths,
        centroid_df_map=centroid_df_map,
        token_df_map=token_df_map,
    )


if __name__ == "__main__":
    main()
