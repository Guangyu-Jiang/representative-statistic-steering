#!/usr/bin/env python3
"""Train an answer-blind Lookback candidate ranker on disjoint development data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from repstat_steering.lookback_control import (
    LOOKBACK_QUERY_STOP_WORDS,
    load_nq_examples,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FEATURE_NAMES = (
    "replay_logit",
    "controlled_logit",
    "replay_logit_z",
    "controlled_logit_z",
    "log_token_count",
    "question_term_coverage",
    "question_content_coverage",
    "document_content_precision",
    "max_passage_content_precision",
    "response_unique_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "external/Lookback-Lens/data/"
            "nq-open-10_total_documents_gold_at_4.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path(
            "artifacts/lookback_nq/development_n60_matched_rerank_diagnostics/"
            "candidates4/results.jsonl"
        ),
    )
    parser.add_argument(
        "--development-controlled",
        type=Path,
        help=(
            "Optional separate results.jsonl containing the selected "
            "minimum_norm_rerank development candidates. Baseline candidates "
            "are always read from --development."
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "artifacts/lookback_nq/validation_offset160_n100_minimum_norm_rerank_replay/"
            "candidates4_sparse128_shift4_cap0.5/results.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-rerank-baseline",
        type=Path,
        default=Path(
            "artifacts/lookback_nq/validation_offset160_n100_baseline_rerank_replay/"
            "candidates4/results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lookback_nq/candidate_ranker_final"),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_rows(path: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") == method:
            rows.append(row)
    return rows


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def terms(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [
        term
        for term in terms(text)
        if len(term) >= 2 and term not in LOOKBACK_QUERY_STOP_WORDS
    ]


def document_from_prompt(prompt: str) -> str:
    marker = "#Document#: "
    if marker not in prompt or "\n#Question#:" not in prompt:
        raise ValueError("unexpected Natural Questions prompt format")
    return prompt.split(marker, 1)[1].split("\n#Question#:", 1)[0]


def flatten(
    rows: list[dict[str, Any]], prompt_by_index: dict[int, str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        replay = logit(
            np.asarray(row["candidate_replay_factual_probabilities"], dtype=np.float64)
        )
        controlled = logit(
            np.asarray(
                row["candidate_controlled_factual_probabilities"], dtype=np.float64
            )
        )
        responses = row.get("candidate_responses")
        token_ids = row.get("candidate_generated_token_ids")
        labels = row["candidate_exact_matches"]
        if responses is None or token_ids is None:
            raise ValueError(
                f"dataset index {row['dataset_index']} lacks saved candidate text"
            )
        dataset_index = int(row["dataset_index"])
        if dataset_index not in prompt_by_index:
            raise ValueError(f"missing prompt for dataset index {dataset_index}")
        document = document_from_prompt(prompt_by_index[dataset_index])
        passages = document.splitlines()
        question_terms = set(terms(row["question"]))
        question_content = set(content_terms(row["question"]))
        document_content = set(content_terms(document))
        passage_content = [set(content_terms(passage)) for passage in passages]
        for candidate_index, response in enumerate(responses):
            response_terms = terms(response)
            response_set = set(response_terms)
            response_content = set(content_terms(response))
            coverage = (
                len(question_terms & response_set) / len(question_terms)
                if question_terms
                else 0.0
            )
            unique_fraction = (
                len(response_set) / len(response_terms) if response_terms else 0.0
            )
            question_content_coverage = (
                len(question_content & response_content) / len(question_content)
                if question_content
                else 0.0
            )
            document_content_precision = (
                len(document_content & response_content) / len(response_content)
                if response_content
                else 0.0
            )
            max_passage_content_precision = (
                max(
                    (
                        len(passage & response_content) / len(response_content)
                        for passage in passage_content
                    ),
                    default=0.0,
                )
                if response_content
                else 0.0
            )
            features = (
                replay[candidate_index],
                controlled[candidate_index],
                zscore(replay)[candidate_index],
                zscore(controlled)[candidate_index],
                np.log1p(len(token_ids[candidate_index])),
                coverage,
                question_content_coverage,
                document_content_precision,
                max_passage_content_precision,
                unique_fraction,
            )
            record: dict[str, Any] = {
                "dataset_index": dataset_index,
                "candidate_index": candidate_index,
                "candidate_generation_method": row.get(
                    "candidate_generation_method", "minimum_norm"
                ),
                "question": row["question"],
                "response": response,
                "exact_match": float(labels[candidate_index]),
                "replay_selected": int(candidate_index == row["selected_candidate_index"]),
            }
            record.update(dict(zip(FEATURE_NAMES, features, strict=True)))
            records.append(record)
    return pd.DataFrame(records)


def selection_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    selected = frame.loc[
        frame.groupby("dataset_index")[score_column].idxmax()
    ].copy()
    groups = frame.groupby("dataset_index").exact_match
    return {
        "n_questions": int(selected.dataset_index.nunique()),
        "selected_exact_match": float(selected.exact_match.mean()),
        "random_candidate_exact_match": float(groups.mean().mean()),
        "oracle_exact_match": float(groups.max().mean()),
        "selected": selected,
    }


def bootstrap_interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    means = values[
        generator.integers(0, values.size, size=(samples, values.size))
    ].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    development_controlled_path = (
        args.development_controlled or args.development
    )
    development_controlled_rows = read_rows(
        development_controlled_path, "minimum_norm_rerank"
    )
    development_baseline_rows = read_rows(args.development, "baseline_rerank")
    development_rows = development_controlled_rows + development_baseline_rows
    validation_rows = read_rows(args.validation, "minimum_norm_rerank")
    validation_baseline_rows = read_rows(args.validation, "baseline")
    validation_rerank_baseline_rows = read_rows(
        args.validation_rerank_baseline, "baseline_rerank"
    )
    development_ids = {int(row["dataset_index"]) for row in development_rows}
    development_controlled_ids = {
        int(row["dataset_index"]) for row in development_controlled_rows
    }
    development_baseline_ids = {
        int(row["dataset_index"]) for row in development_baseline_rows
    }
    validation_ids = {int(row["dataset_index"]) for row in validation_rows}
    if development_ids & validation_ids:
        raise ValueError("development and validation dataset indices overlap")
    if args.require_complete and (
        len(development_ids) != 60
        or len(development_controlled_ids) != 60
        or len(development_baseline_ids) != 60
        or len(validation_ids) != 100
        or len(validation_baseline_rows) != 100
        or len(validation_rerank_baseline_rows) != 100
    ):
        raise ValueError(
            f"incomplete inputs: development={len(development_ids)}, "
            f"validation={len(validation_ids)}, "
            f"rerank_baseline={len(validation_rerank_baseline_rows)}"
        )

    all_indices = sorted(development_ids | validation_ids)
    examples = load_nq_examples(args.data, indices=all_indices)
    prompt_by_index = {example.dataset_index: example.prompt for example in examples}
    if set(prompt_by_index) != set(all_indices):
        missing = sorted(set(all_indices) - set(prompt_by_index))
        raise ValueError(f"missing Natural Questions examples: {missing}")
    development = flatten(development_rows, prompt_by_index)
    validation = flatten(validation_rows, prompt_by_index)
    feature_columns = list(FEATURE_NAMES)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1_000, random_state=42
        ),
    )
    cross_validator = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    development["ranker_probability"] = np.nan
    for train_indices, test_indices in cross_validator.split(
        development[feature_columns],
        development.exact_match,
        groups=development.dataset_index,
    ):
        fold_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1_000, random_state=42
            ),
        )
        fold_model.fit(
            development.iloc[train_indices][feature_columns],
            development.iloc[train_indices].exact_match,
        )
        development.loc[
            development.index[test_indices], "ranker_probability"
        ] = fold_model.predict_proba(
            development.iloc[test_indices][feature_columns]
        )[:, 1]

    model.fit(development[feature_columns], development.exact_match)
    validation["ranker_probability"] = model.predict_proba(
        validation[feature_columns]
    )[:, 1]
    validation_rerank_baseline = flatten(
        validation_rerank_baseline_rows, prompt_by_index
    )
    validation_rerank_baseline["ranker_probability"] = model.predict_proba(
        validation_rerank_baseline[feature_columns]
    )[:, 1]
    development["replay_score"] = development.replay_logit
    validation["replay_score"] = validation.replay_logit
    development_controlled = development[
        development.candidate_generation_method.eq("minimum_norm")
    ]
    development_unsteered = development[
        development.candidate_generation_method.eq("baseline")
    ]
    development_controlled_ranker = selection_metrics(
        development_controlled, "ranker_probability"
    )
    development_controlled_replay = selection_metrics(
        development_controlled, "replay_score"
    )
    development_unsteered_ranker = selection_metrics(
        development_unsteered, "ranker_probability"
    )
    development_unsteered_replay = selection_metrics(
        development_unsteered, "replay_score"
    )
    validation_ranker = selection_metrics(validation, "ranker_probability")
    validation_replay = selection_metrics(validation, "replay_score")
    validation_rerank_baseline_ranker = selection_metrics(
        validation_rerank_baseline, "ranker_probability"
    )

    development_controlled_ranker.pop("selected")
    development_controlled_replay.pop("selected")
    development_unsteered_ranker.pop("selected")
    development_unsteered_replay.pop("selected")
    ranker_selected = validation_ranker.pop("selected")
    replay_selected = validation_replay.pop("selected")
    rerank_baseline_selected = validation_rerank_baseline_ranker.pop("selected")
    paired = ranker_selected[["dataset_index", "exact_match"]].merge(
        replay_selected[["dataset_index", "exact_match"]],
        on="dataset_index",
        suffixes=("_ranker", "_replay"),
        validate="one_to_one",
    )
    delta = (paired.exact_match_ranker - paired.exact_match_replay).to_numpy()
    lower, upper = bootstrap_interval(delta)
    validation_baseline = pd.DataFrame(validation_baseline_rows)[
        ["dataset_index", "exact_match"]
    ].drop_duplicates("dataset_index")
    baseline_paired = ranker_selected[["dataset_index", "exact_match"]].merge(
        validation_baseline,
        on="dataset_index",
        suffixes=("_ranker", "_baseline"),
        validate="one_to_one",
    )
    baseline_delta = (
        baseline_paired.exact_match_ranker
        - baseline_paired.exact_match_baseline
    ).to_numpy()
    baseline_lower, baseline_upper = bootstrap_interval(baseline_delta)
    rerank_baseline_paired = ranker_selected[
        ["dataset_index", "exact_match"]
    ].merge(
        rerank_baseline_selected[["dataset_index", "exact_match"]],
        on="dataset_index",
        suffixes=("_controlled", "_unsteered"),
        validate="one_to_one",
    )
    rerank_baseline_delta = (
        rerank_baseline_paired.exact_match_controlled
        - rerank_baseline_paired.exact_match_unsteered
    ).to_numpy()
    rerank_baseline_lower, rerank_baseline_upper = bootstrap_interval(
        rerank_baseline_delta
    )
    summary = {
        "configuration": {
            "features": feature_columns,
            "model": "standard_scaler_logistic_regression",
            "C": 1.0,
            "class_weight": "balanced",
            "development_protocol": "five_fold_stratified_group_oof",
            "selection_unit": "question",
            "development_baseline": str(args.development),
            "development_controlled": str(development_controlled_path),
            "validation_controlled": str(args.validation),
            "validation_unsteered": str(args.validation_rerank_baseline),
        },
        "development": {
            "combined_candidate_auc": float(
                roc_auc_score(
                    development.exact_match, development.ranker_probability
                )
            ),
            "controlled_ranker": development_controlled_ranker,
            "controlled_replay": development_controlled_replay,
            "unsteered_ranker": development_unsteered_ranker,
            "unsteered_replay": development_unsteered_replay,
        },
        "validation": {
            "candidate_auc": float(
                roc_auc_score(validation.exact_match, validation.ranker_probability)
            ),
            "ranker": validation_ranker,
            "replay": validation_replay,
            "ranker_minus_replay_exact_match": float(delta.mean()),
            "ranker_minus_replay_ci95": [lower, upper],
            "sampled_baseline_exact_match": float(
                validation_baseline.exact_match.mean()
            ),
            "ranker_minus_sampled_baseline_exact_match": float(
                baseline_delta.mean()
            ),
            "ranker_minus_sampled_baseline_ci95": [
                baseline_lower,
                baseline_upper,
            ],
            "ranked_unsteered_exact_match": float(
                rerank_baseline_selected.exact_match.mean()
            ),
            "ranked_controlled_minus_ranked_unsteered_exact_match": float(
                rerank_baseline_delta.mean()
            ),
            "ranked_controlled_minus_ranked_unsteered_ci95": [
                rerank_baseline_lower,
                rerank_baseline_upper,
            ],
            "unsteered_rerank": validation_rerank_baseline_ranker,
        },
        "split": {
            "development_questions": len(development_ids),
            "development_controlled_questions": len(development_controlled_ids),
            "development_unsteered_questions": len(development_baseline_ids),
            "validation_questions": len(validation_ids),
            "overlap": 0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "ranker.joblib")
    development.to_csv(args.output_dir / "development_candidates.csv", index=False)
    validation.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    validation_rerank_baseline.to_csv(
        args.output_dir / "validation_unsteered_candidates.csv", index=False
    )
    ranker_selected.to_csv(
        args.output_dir / "validation_ranker_selected.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
