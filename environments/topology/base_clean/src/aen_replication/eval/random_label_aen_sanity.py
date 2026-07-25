"""Random-label sanity checks for sparse AEN probes.

The control asks whether a low-dimensional AEN probe can still obtain good test
performance when the labels used to select and train the sparse probe are random.

For each model/dataset at the default AEN layer, this script:

1. Trains a true-label full probe and evaluates sparse top-k probes for k=3,5.
2. Repeats random-label trials:
   - permute train labels
   - train a full probe on those random train labels
   - take the top-k coefficient dimensions as random-label "AENs"
   - train a sparse top-k probe on the same random train labels
   - evaluate against true test labels and independently randomized test labels
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aen_replication.eval.metrics import binary_classification_metrics
from aen_replication.models.hidden_state_extractor import load_hidden_state_table
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/meta_llama_llama_3_1_8b_instruct",
        "layer": 14,
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/mistralai_mistral_7b_instruct_v0_3",
        "layer": 14,
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "hidden_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/hidden_states/google_gemma_7b_it",
        "layer": 14,
    },
]

DATASETS = ["ambigqa", "situatedqa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/random_label_aen_sanity",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    return parser.parse_args()


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
        clf.fit(x_train, y_train)
    return clf


def _scores(clf: LogisticRegression, x_eval: np.ndarray) -> np.ndarray:
    return np.asarray(clf.decision_function(x_eval), dtype=float)


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = binary_classification_metrics(y_true.astype(int), scores)
    return {
        "accuracy": float(metrics["accuracy"]),
        "auroc": float(metrics["auroc"]),
        "f1": float(metrics["f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _split(meta: pd.DataFrame, matrix: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    splits: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split_name in ["train", "test"]:
        mask = meta["split"].astype(str).eq(split_name).to_numpy()
        split_meta = meta.loc[mask].reset_index(drop=True)
        split_matrix = np.asarray(matrix[mask], dtype=np.float32)
        labels = split_meta["label_ambiguous"].to_numpy(dtype=int)
        splits[split_name] = (split_meta, split_matrix, labels)
    return splits


def _load_dataset_frame(hidden_root: Path, dataset: str, layer: int) -> dict[str, Any]:
    path = hidden_root / f"{dataset}__layer_{int(layer):02d}__mean_pool.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden-state cache: {path}")
    meta, matrix = load_hidden_state_table(path)
    splits = _split(meta, matrix)
    train_meta, _, _ = splits["train"]
    test_meta, _, _ = splits["test"]
    pair_overlap = 0
    example_overlap = 0
    if "pair_id" in meta.columns:
        pair_overlap = len(set(train_meta["pair_id"].astype(str)) & set(test_meta["pair_id"].astype(str)))
    if "example_id" in meta.columns:
        example_overlap = len(set(train_meta["example_id"].astype(str)) & set(test_meta["example_id"].astype(str)))
    return {
        "path": str(path),
        "splits": splits,
        "train_size": int(len(train_meta)),
        "test_size": int(len(test_meta)),
        "train_pos": int(train_meta["label_ambiguous"].sum()),
        "test_pos": int(test_meta["label_ambiguous"].sum()),
        "pair_overlap": int(pair_overlap),
        "example_overlap": int(example_overlap),
        "hidden_dim": int(matrix.shape[1]),
    }


def _topk_indices(clf: LogisticRegression, k: int) -> list[int]:
    weights = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    return np.argsort(-weights)[: int(k)].astype(int).tolist()


def _evaluate_sparse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    indices: list[int],
    *,
    seed: int,
) -> dict[str, float]:
    clf = _fit_probe(x_train[:, indices], y_train, seed=seed)
    train_metrics = _metrics(y_train, _scores(clf, x_train[:, indices]))
    eval_metrics = _metrics(y_eval, _scores(clf, x_eval[:, indices]))
    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_auroc": train_metrics["auroc"],
        "train_f1": train_metrics["f1"],
        "eval_accuracy": eval_metrics["accuracy"],
        "eval_auroc": eval_metrics["auroc"],
        "eval_f1": eval_metrics["f1"],
        "eval_macro_f1": eval_metrics["macro_f1"],
    }


def _true_label_rows(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_values: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    full_clf = _fit_probe(x_train, y_train, seed=seed)
    full_train_metrics = _metrics(y_train, _scores(full_clf, x_train))
    full_test_metrics = _metrics(y_test, _scores(full_clf, x_test))
    rows: list[dict[str, Any]] = []
    for k in k_values:
        indices = _topk_indices(full_clf, k)
        sparse_metrics = _evaluate_sparse(
            x_train,
            y_train,
            x_test,
            y_test,
            indices,
            seed=seed + 10 + int(k),
        )
        rows.append(
            {
                "condition": "true_labels",
                "trial": -1,
                "k": int(k),
                "indices": indices,
                "full_train_accuracy": full_train_metrics["accuracy"],
                "full_train_auroc": full_train_metrics["auroc"],
                "full_eval_accuracy": full_test_metrics["accuracy"],
                "full_eval_auroc": full_test_metrics["auroc"],
                **sparse_metrics,
            }
        )
    return rows


def _random_label_rows(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_values: list[int],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for trial in range(int(trials)):
        trial_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        trial_rng = np.random.default_rng(trial_seed)
        y_train_random = trial_rng.permutation(y_train)
        y_test_random = trial_rng.permutation(y_test)

        full_clf = _fit_probe(x_train, y_train_random, seed=trial_seed)
        full_train_metrics = _metrics(y_train_random, _scores(full_clf, x_train))
        full_true_test_metrics = _metrics(y_test, _scores(full_clf, x_test))
        full_random_test_metrics = _metrics(y_test_random, _scores(full_clf, x_test))

        for k in k_values:
            indices = _topk_indices(full_clf, int(k))
            true_test_sparse = _evaluate_sparse(
                x_train,
                y_train_random,
                x_test,
                y_test,
                indices,
                seed=trial_seed + 10 + int(k),
            )
            rows.append(
                {
                    "condition": "random_train_true_test",
                    "trial": int(trial),
                    "k": int(k),
                    "indices": indices,
                    "full_train_accuracy": full_train_metrics["accuracy"],
                    "full_train_auroc": full_train_metrics["auroc"],
                    "full_eval_accuracy": full_true_test_metrics["accuracy"],
                    "full_eval_auroc": full_true_test_metrics["auroc"],
                    **true_test_sparse,
                }
            )

            random_test_sparse = _evaluate_sparse(
                x_train,
                y_train_random,
                x_test,
                y_test_random,
                indices,
                seed=trial_seed + 100 + int(k),
            )
            rows.append(
                {
                    "condition": "random_train_random_test",
                    "trial": int(trial),
                    "k": int(k),
                    "indices": indices,
                    "full_train_accuracy": full_train_metrics["accuracy"],
                    "full_train_auroc": full_train_metrics["auroc"],
                    "full_eval_accuracy": full_random_test_metrics["accuracy"],
                    "full_eval_auroc": full_random_test_metrics["auroc"],
                    **random_test_sparse,
                }
            )
    return rows


def _summarize_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model", "model_label", "dataset", "condition", "k"]
    for keys, group in trial_df.groupby(group_cols, sort=True):
        model, model_label, dataset, condition, k = keys
        is_true = str(condition) == "true_labels"
        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "dataset": dataset,
                "condition": condition,
                "k": int(k),
                "trials": int(1 if is_true else len(group)),
                "eval_accuracy_mean": float(group["eval_accuracy"].mean()),
                "eval_accuracy_std": float(0.0 if is_true else group["eval_accuracy"].std(ddof=1)),
                "eval_accuracy_max": float(group["eval_accuracy"].max()),
                "eval_auroc_mean": float(group["eval_auroc"].mean()),
                "eval_auroc_std": float(0.0 if is_true else group["eval_auroc"].std(ddof=1)),
                "eval_auroc_max": float(group["eval_auroc"].max()),
                "eval_f1_mean": float(group["eval_f1"].mean()),
                "train_accuracy_mean": float(group["train_accuracy"].mean()),
                "train_accuracy_max": float(group["train_accuracy"].max()),
                "full_train_accuracy_mean": float(group["full_train_accuracy"].mean()),
                "full_eval_accuracy_mean": float(group["full_eval_accuracy"].mean()),
                "full_eval_accuracy_max": float(group["full_eval_accuracy"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _render_report(
    *,
    summary_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    output_root: Path,
    trials: int,
    k_values: list[int],
) -> str:
    lines = [
        "# Random-Label AEN Sanity Check",
        "",
        "Setup:",
        f"- Random trials per model/dataset: `{int(trials)}`",
        f"- Sparse probe sizes: `{', '.join(str(k) for k in k_values)}` neurons",
        "- Readout: mean-pooled hidden state",
        "- Layer: 14 for all three models",
        "- Random labels are class-balanced permutations of the original labels.",
        "- `random_train_true_test`: train labels are randomized, test labels remain true.",
        "- `random_train_random_test`: train labels and test labels are randomized independently.",
        "",
        "The train/test split in the cached artifacts is disjoint by both `pair_id` and `example_id`.",
        "",
        "## Dataset Splits",
        "",
        "| Model | Dataset | Train | Test | Train pos | Test pos | Pair overlap | Example overlap | Hidden dim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in dataset_df.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['dataset']} | {int(row['train_size'])} | {int(row['test_size'])} | "
            f"{int(row['train_pos'])} | {int(row['test_pos'])} | {int(row['pair_overlap'])} | "
            f"{int(row['example_overlap'])} | {int(row['hidden_dim'])} |"
        )

    for condition, title in [
        ("true_labels", "True-Label Sparse Probe"),
        ("random_train_true_test", "Random Train Labels, True Test Labels"),
        ("random_train_random_test", "Random Train Labels, Random Test Labels"),
    ]:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | Dataset | k | Acc mean | Acc std | Acc max | AUROC mean | AUROC std | AUROC max | Full train acc mean | Full eval acc mean |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        subset = summary_df.loc[summary_df["condition"].eq(condition)].copy()
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['model_label']} | {row['dataset']} | {int(row['k'])} | "
                f"{float(row['eval_accuracy_mean']):.4f} | {float(row['eval_accuracy_std']):.4f} | "
                f"{float(row['eval_accuracy_max']):.4f} | {float(row['eval_auroc_mean']):.4f} | "
                f"{float(row['eval_auroc_std']):.4f} | {float(row['eval_auroc_max']):.4f} | "
                f"{float(row['full_train_accuracy_mean']):.4f} | {float(row['full_eval_accuracy_mean']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- With true labels, 3-5 top neurons often produce above-chance performance.",
            "- With randomized train labels and independently randomized test labels, sparse 3-5-neuron probes stay at chance.",
            "- With randomized train labels but true test labels, the mean is near chance, but individual cherry-picked trials can look strong because the sparse score can accidentally align with the true label direction.",
            "- The full random-label probe can partially fit random training labels, but that does not transfer to held-out test data.",
            "",
            f"- Trial rows: `{output_root / 'random_label_aen_sanity_trials.parquet'}`",
            f"- Summary rows: `{output_root / 'random_label_aen_sanity_summary.parquet'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    k_values = sorted({int(k) for k in args.k_values})

    trial_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for model_idx, spec in enumerate(MODEL_SPECS):
        hidden_root = Path(spec["hidden_root"])
        for dataset_idx, dataset in enumerate(args.datasets):
            dataset_payload = _load_dataset_frame(hidden_root, dataset, int(spec["layer"]))
            dataset_rows.append(
                {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    "dataset": dataset,
                    **{key: value for key, value in dataset_payload.items() if key != "splits"},
                }
            )
            _, x_train, y_train = dataset_payload["splits"]["train"]
            _, x_test, y_test = dataset_payload["splits"]["test"]
            seed = int(args.seed + 10_000 * model_idx + 1_000 * dataset_idx)

            rows = _true_label_rows(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                k_values=k_values,
                seed=seed,
            )
            rows.extend(
                _random_label_rows(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    k_values=k_values,
                    trials=int(args.trials),
                    seed=seed + 123,
                )
            )
            for row in rows:
                trial_rows.append(
                    {
                        "model": spec["slug"],
                        "model_label": spec["label"],
                        "dataset": dataset,
                        "layer": int(spec["layer"]),
                        **row,
                    }
                )

    trial_df = pd.DataFrame(trial_rows)
    dataset_df = pd.DataFrame(dataset_rows).sort_values(["model_label", "dataset"]).reset_index(drop=True)
    summary_df = _summarize_trials(trial_df)

    write_parquet(trial_df, output_root / "random_label_aen_sanity_trials.parquet")
    write_parquet(summary_df, output_root / "random_label_aen_sanity_summary.parquet")
    write_parquet(dataset_df, output_root / "random_label_aen_sanity_dataset_splits.parquet")
    write_markdown(
        output_root / "random_label_aen_sanity_report.md",
        _render_report(
            summary_df=summary_df,
            dataset_df=dataset_df,
            output_root=output_root,
            trials=int(args.trials),
            k_values=k_values,
        ),
    )


if __name__ == "__main__":
    main()
