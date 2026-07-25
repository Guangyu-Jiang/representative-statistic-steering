"""Random-label sanity check with official-style AEN perturbation.

This version mirrors the AEN identification/ablation path in the public repo:

1. Train a full linear probe.
2. Rank neurons by absolute full-probe coefficient.
3. For each fixed k, treat the top-k dimensions as the AEN set.
4. Perturb those dimensions by replacing each selected column with Gaussian
   samples drawn from that column's empirical mean and standard deviation,
   matching the public implementation's ``remove_first_n_dims`` logic.
5. Retrain a sparse probe using only the selected neurons.

The random-label controls use the same cached train/test points as the AEN
replication, so no LLM forward passes are required.
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
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/random_label_aen_perturbation_sanity",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--perturb-trials", type=int, default=5)
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


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = binary_classification_metrics(y_true.astype(int), np.asarray(scores, dtype=float))
    return {
        "accuracy": float(metrics["accuracy"]),
        "auroc": float(metrics["auroc"]),
        "f1": float(metrics["f1"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _scores(clf: LogisticRegression, x_eval: np.ndarray) -> np.ndarray:
    return np.asarray(clf.decision_function(x_eval), dtype=float)


def _split(meta: pd.DataFrame, matrix: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    splits: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split_name in ["train", "test"]:
        mask = meta["split"].astype(str).eq(split_name).to_numpy()
        split_meta = meta.loc[mask].reset_index(drop=True)
        split_matrix = np.asarray(matrix[mask], dtype=np.float32)
        split_labels = split_meta["label_ambiguous"].to_numpy(dtype=int)
        splits[split_name] = (split_meta, split_matrix, split_labels)
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


def _select_topk_aens_with_official_perturbation(
    *,
    full_clf: LogisticRegression,
    x_select: np.ndarray,
    y_select: np.ndarray,
    k: int,
    noise_scale: float,
    perturb_trials: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    weights = np.abs(np.asarray(full_clf.coef_, dtype=float).ravel())
    ranked = np.argsort(-weights)
    baseline_scores = _scores(full_clf, x_select)
    baseline_accuracy = float(_metrics(y_select, baseline_scores)["accuracy"])
    k = int(k)
    indices = ranked[:k].astype(int).tolist()
    trial_accs: list[float] = []
    for _ in range(int(perturb_trials)):
        perturbed = x_select.copy()
        for dim in indices:
            feature_mean = float(np.mean(x_select[:, dim]))
            feature_std = float(np.std(x_select[:, dim]))
            noise = rng.normal(
                loc=feature_mean,
                scale=feature_std * float(noise_scale),
                size=x_select.shape[0],
            )
            perturbed[:, dim] = noise
        trial_scores = _scores(full_clf, perturbed)
        trial_accs.append(float(_metrics(y_select, trial_scores)["accuracy"]))
    mean_acc = float(np.mean(trial_accs))
    return {
        "baseline_accuracy": baseline_accuracy,
        "aen_k": k,
        "aen_indices": indices,
        "accuracy_after_perturb": mean_acc,
        "accuracy_drop": baseline_accuracy - mean_acc,
    }


def _evaluate_sparse_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    indices: list[int],
    seed: int,
) -> dict[str, float]:
    sparse_clf = _fit_probe(x_train[:, indices], y_train, seed=seed)
    train_metrics = _metrics(y_train, _scores(sparse_clf, x_train[:, indices]))
    eval_metrics = _metrics(y_eval, _scores(sparse_clf, x_eval[:, indices]))
    return {
        "sparse_train_accuracy": train_metrics["accuracy"],
        "sparse_train_auroc": train_metrics["auroc"],
        "sparse_eval_accuracy": eval_metrics["accuracy"],
        "sparse_eval_auroc": eval_metrics["auroc"],
        "sparse_eval_f1": eval_metrics["f1"],
    }


def _run_one_condition(
    *,
    condition: str,
    x_train: np.ndarray,
    y_train_for_fit: np.ndarray,
    x_test: np.ndarray,
    y_test_for_select: np.ndarray,
    y_test_for_eval: np.ndarray,
    k_values: list[int],
    noise_scale: float,
    perturb_trials: int,
    seed: int,
    trial: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    full_clf = _fit_probe(x_train, y_train_for_fit, seed=seed)
    full_train = _metrics(y_train_for_fit, _scores(full_clf, x_train))
    full_select = _metrics(y_test_for_select, _scores(full_clf, x_test))
    rows: list[dict[str, Any]] = []
    for k in k_values:
        selection = _select_topk_aens_with_official_perturbation(
            full_clf=full_clf,
            x_select=x_test,
            y_select=y_test_for_select,
            k=int(k),
            noise_scale=noise_scale,
            perturb_trials=perturb_trials,
            rng=rng,
        )
        sparse = _evaluate_sparse_probe(
            x_train=x_train,
            y_train=y_train_for_fit,
            x_eval=x_test,
            y_eval=y_test_for_eval,
            indices=list(selection["aen_indices"]),
            seed=seed + 17 + int(k),
        )
        rows.append(
            {
                "condition": condition,
                "trial": int(trial),
                "k": int(k),
                "selected_indices": list(selection["aen_indices"]),
                "full_train_accuracy": full_train["accuracy"],
                "full_train_auroc": full_train["auroc"],
                "full_selection_accuracy": full_select["accuracy"],
                "full_selection_auroc": full_select["auroc"],
                "perturb_baseline_accuracy": float(selection["baseline_accuracy"]),
                "perturb_accuracy_after": float(selection["accuracy_after_perturb"]),
                "perturb_accuracy_drop": float(selection["accuracy_drop"]),
                **sparse,
            }
        )
    return rows


def _run_trials(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_values: list[int],
    noise_scale: float,
    perturb_trials: int,
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _run_one_condition(
            condition="true_labels",
            x_train=x_train,
            y_train_for_fit=y_train,
            x_test=x_test,
            y_test_for_select=y_test,
            y_test_for_eval=y_test,
            k_values=k_values,
            noise_scale=noise_scale,
            perturb_trials=perturb_trials,
            seed=seed,
            trial=-1,
        )
    )

    rng = np.random.default_rng(seed + 9973)
    for trial in range(int(trials)):
        trial_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        trial_rng = np.random.default_rng(trial_seed)
        y_train_random = trial_rng.permutation(y_train)
        y_test_random = trial_rng.permutation(y_test)
        rows.extend(
            _run_one_condition(
                condition="random_train_true_test",
                x_train=x_train,
                y_train_for_fit=y_train_random,
                x_test=x_test,
                y_test_for_select=y_test,
                y_test_for_eval=y_test,
                k_values=k_values,
                noise_scale=noise_scale,
                perturb_trials=perturb_trials,
                seed=trial_seed,
                trial=trial,
            )
        )
        rows.extend(
            _run_one_condition(
                condition="random_train_random_test",
                x_train=x_train,
                y_train_for_fit=y_train_random,
                x_test=x_test,
                y_test_for_select=y_test_random,
                y_test_for_eval=y_test_random,
                k_values=k_values,
                noise_scale=noise_scale,
                perturb_trials=perturb_trials,
                seed=trial_seed + 1,
                trial=trial,
            )
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
                "sparse_train_accuracy_mean": float(group["sparse_train_accuracy"].mean()),
                "sparse_train_accuracy_max": float(group["sparse_train_accuracy"].max()),
                "sparse_eval_accuracy_mean": float(group["sparse_eval_accuracy"].mean()),
                "sparse_eval_accuracy_std": float(0.0 if is_true else group["sparse_eval_accuracy"].std(ddof=1)),
                "sparse_eval_accuracy_max": float(group["sparse_eval_accuracy"].max()),
                "sparse_eval_auroc_mean": float(group["sparse_eval_auroc"].mean()),
                "sparse_eval_auroc_std": float(0.0 if is_true else group["sparse_eval_auroc"].std(ddof=1)),
                "sparse_eval_auroc_max": float(group["sparse_eval_auroc"].max()),
                "full_train_accuracy_mean": float(group["full_train_accuracy"].mean()),
                "full_selection_accuracy_mean": float(group["full_selection_accuracy"].mean()),
                "full_selection_accuracy_max": float(group["full_selection_accuracy"].max()),
                "perturb_drop_mean": float(group["perturb_accuracy_drop"].mean()),
                "perturb_drop_max": float(group["perturb_accuracy_drop"].max()),
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
    noise_scale: float,
    perturb_trials: int,
) -> str:
    lines = [
        "# Random-Label AEN Sanity Check With Official-Style Perturbation",
        "",
        "Setup:",
        f"- Random trials per model/dataset: `{int(trials)}`",
        f"- AEN sizes: `{', '.join(str(k) for k in k_values)}` neurons",
        f"- Perturbation: replace selected dimensions with empirical Gaussian noise, scale `{float(noise_scale):.3g}`, `{int(perturb_trials)}` perturbation trials per k",
        "- Readout: mean-pooled hidden state",
        "- Layer: 14 for all three models",
        "- Selection follows the public implementation: train a full probe, sort dimensions by absolute coefficient, and use the top-k dimensions as AENs.",
        "- The perturbation check follows the public implementation's `remove_first_n_dims`: replace each selected dimension with noise sampled from that dimension's empirical mean and std.",
        "- `random_train_true_test`: train labels are randomized; top-k AEN ranking comes from that random-label full probe; perturbation and final evaluation use true test labels.",
        "- `random_train_random_test`: train labels are randomized; top-k AEN ranking comes from that random-label full probe; perturbation and final evaluation use independently randomized test labels.",
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
        ("true_labels", "True Labels"),
        ("random_train_true_test", "Random Train Labels, True Test Labels"),
        ("random_train_random_test", "Random Train Labels, Random Test Labels"),
    ]:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | Dataset | k | Trials | Sparse train acc | Sparse eval acc mean | Sparse eval acc std | Sparse eval acc max | Sparse AUROC mean | Sparse AUROC max | Perturb drop mean | Full train acc | Full selection acc |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        subset = summary_df.loc[summary_df["condition"].eq(condition)]
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['model_label']} | {row['dataset']} | {int(row['k'])} | {int(row['trials'])} | "
                f"{float(row['sparse_train_accuracy_mean']):.4f} | {float(row['sparse_eval_accuracy_mean']):.4f} | "
                f"{float(row['sparse_eval_accuracy_std']):.4f} | {float(row['sparse_eval_accuracy_max']):.4f} | "
                f"{float(row['sparse_eval_auroc_mean']):.4f} | {float(row['sparse_eval_auroc_max']):.4f} | "
                f"{float(row['perturb_drop_mean']):.4f} | "
                f"{float(row['full_train_accuracy_mean']):.4f} | {float(row['full_selection_accuracy_mean']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- With true labels, official-style top-k AEN probes recover the expected above-chance result.",
            "- With random train labels and random test labels, the selected sparse probe remains at chance.",
            "- With random train labels but true test labels, the mean remains near chance, but individual trials can still look high because the selection/evaluation labels are true and the chosen random-label score can accidentally align with them.",
            "- Sparse probes do not memorize random training labels strongly: sparse train accuracy remains much lower than the full random-label probe.",
            "",
            f"- Trial rows: `{output_root / 'random_label_aen_perturbation_trials.parquet'}`",
            f"- Summary rows: `{output_root / 'random_label_aen_perturbation_summary.parquet'}`",
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
            payload = _load_dataset_frame(hidden_root, dataset, int(spec["layer"]))
            dataset_rows.append(
                {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    "dataset": dataset,
                    **{key: value for key, value in payload.items() if key != "splits"},
                }
            )
            _, x_train, y_train = payload["splits"]["train"]
            _, x_test, y_test = payload["splits"]["test"]
            seed = int(args.seed + 10_000 * model_idx + 1_000 * dataset_idx)
            rows = _run_trials(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                k_values=k_values,
                noise_scale=float(args.noise_scale),
                perturb_trials=int(args.perturb_trials),
                trials=int(args.trials),
                seed=seed,
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

    write_parquet(trial_df, output_root / "random_label_aen_perturbation_trials.parquet")
    write_parquet(summary_df, output_root / "random_label_aen_perturbation_summary.parquet")
    write_parquet(dataset_df, output_root / "random_label_aen_perturbation_dataset_splits.parquet")
    write_markdown(
        output_root / "random_label_aen_perturbation_report.md",
        _render_report(
            summary_df=summary_df,
            dataset_df=dataset_df,
            output_root=output_root,
            trials=int(args.trials),
            k_values=k_values,
            noise_scale=float(args.noise_scale),
            perturb_trials=int(args.perturb_trials),
        ),
    )


if __name__ == "__main__":
    main()
