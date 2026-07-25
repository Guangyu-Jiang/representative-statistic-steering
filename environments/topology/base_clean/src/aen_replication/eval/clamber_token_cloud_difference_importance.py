"""Analyze difference-feature importance for CLAMBER 4-way token-cloud topology models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aen_replication.train.token_cloud_topology_classifier import (
    _build_multilayer_feature_frames,
    _topology_feature_columns,
)
from aen_replication.utils.io_utils import ensure_dir, write_markdown, write_parquet

GROUP4_MAP = {
    "polysemy": "ambiguity",
    "co-reference": "ambiguity",
    "what": "missing_condition",
    "when": "missing_condition",
    "where": "missing_condition",
    "whom": "missing_condition",
    "ICL": "conflicting_condition",
    "none": "clear",
}

MODEL_SPECS = [
    {
        "slug": "meta_llama_llama_3_1_8b_instruct",
        "label": "LLaMA 3.1 8B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/meta_llama_llama_3_1_8b_instruct",
    },
    {
        "slug": "mistralai_mistral_7b_instruct_v0_3",
        "label": "Mistral 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/mistralai_mistral_7b_instruct_v0_3",
    },
    {
        "slug": "google_gemma_7b_it",
        "label": "Gemma 7B",
        "subclass_root": "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/clamber_subclass_classification/google_gemma_7b_it",
    },
]

FINAL_METRICS_PATH = Path(
    "/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_regrouped_4way_topology/clamber_regrouped_4way_topology_final_metrics.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/sparse_neurons_ambiguity_replication/artifacts/reports/clamber_token_cloud_difference_importance",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _fit_multiclass_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_fit, y_train)
    return clf, scaler


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _load_feature_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df = df.loc[df["subclass"].isin(GROUP4_MAP)].copy()
    df["group4"] = df["subclass"].map(GROUP4_MAP)
    return df.reset_index(drop=True)


def _parse_selection_signature(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(item.strip()) for item in str(value).split("|")]


def _make_multilayer_frame(feature_df: pd.DataFrame, layers: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selections = [{"layer": int(layer), "val_auroc": 0.0} for layer in layers]
    train_df, test_df, meta = _build_multilayer_feature_frames(
        feature_df=feature_df,
        dataset="clamber",
        selections=selections,
    )
    lookup = feature_df.loc[:, ["example_id", "group4"]].drop_duplicates()
    train_df = train_df.merge(lookup, on="example_id", how="left")
    test_df = test_df.merge(lookup, on="example_id", how="left")
    feature_cols = list(meta["topology_columns"]) + list(meta["topology_summary_columns"])
    return train_df, test_df, feature_cols


def _is_difference_feature(name: str) -> bool:
    return ("wasserstein" in name) or ("bottleneck" in name)


def _base_difference_name(name: str) -> str:
    if "__l" in name:
        return name.split("__l", 1)[0]
    return name


def _coef_importance_df(feature_columns: list[str], clf: LogisticRegression) -> pd.DataFrame:
    coef = np.asarray(clf.coef_, dtype=float)
    norms = np.linalg.norm(coef, axis=0)
    rows: list[dict[str, Any]] = []
    for feature, value in zip(feature_columns, norms.tolist(), strict=False):
        rows.append(
            {
                "feature": feature,
                "coef_norm": float(value),
                "is_difference_feature": bool(_is_difference_feature(feature)),
                "difference_base": _base_difference_name(feature) if _is_difference_feature(feature) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("coef_norm", ascending=False).reset_index(drop=True)


def _evaluate_feature_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
) -> dict[str, float]:
    x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_df["group4"].astype(str).to_numpy()
    x_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_test = test_df["group4"].astype(str).to_numpy()
    clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=seed)
    pred = clf.predict(scaler.transform(x_test))
    return _metrics(y_test, pred)


def _difference_ablation_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    full_metrics = _evaluate_feature_subset(train_df, test_df, feature_columns=feature_columns, seed=seed)
    diff_bases = sorted({_base_difference_name(column) for column in feature_columns if _is_difference_feature(column)})
    rows = [
        {
            "ablation": "none",
            "remaining_feature_count": int(len(feature_columns)),
            "accuracy": float(full_metrics["accuracy"]),
            "macro_f1": float(full_metrics["macro_f1"]),
            "macro_f1_drop": 0.0,
        }
    ]
    for base in diff_bases + ["all_difference_features"]:
        if base == "all_difference_features":
            kept = [column for column in feature_columns if not _is_difference_feature(column)]
        else:
            kept = [column for column in feature_columns if _base_difference_name(column) != base]
        metrics = _evaluate_feature_subset(train_df, test_df, feature_columns=kept, seed=seed)
        rows.append(
            {
                "ablation": base,
                "remaining_feature_count": int(len(kept)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_f1_drop": float(full_metrics["macro_f1"] - metrics["macro_f1"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1_drop", "ablation"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    final_metrics = pd.read_parquet(FINAL_METRICS_PATH)

    coef_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        feature_df = _load_feature_df(Path(spec["subclass_root"]) / "clamber_token_cloud_all_layer_features.parquet")
        model_final = final_metrics.loc[final_metrics["model"].eq(spec["slug"])].copy()
        for method in ["token_cloud_single", "token_cloud_multilayer"]:
            row = model_final.loc[model_final["method"].eq(method)].iloc[0].to_dict()
            if method == "token_cloud_single":
                layer = int(row["layer"])
                train_df = feature_df.loc[(feature_df["split"].eq("train")) & (feature_df["layer"].eq(layer))].copy()
                test_df = feature_df.loc[(feature_df["split"].eq("test")) & (feature_df["layer"].eq(layer))].copy()
                feature_columns = _topology_feature_columns(train_df)
                selection_signature = str(layer)
            else:
                layers = _parse_selection_signature(row["selection_signature"])
                train_df, test_df, feature_columns = _make_multilayer_frame(feature_df, layers)
                selection_signature = str(row["selection_signature"])

            x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
            y_train = train_df["group4"].astype(str).to_numpy()
            clf, scaler = _fit_multiclass_logistic(x_train, y_train, seed=args.seed)
            coef_df = _coef_importance_df(feature_columns, clf)
            coef_df["model"] = spec["slug"]
            coef_df["model_label"] = spec["label"]
            coef_df["method"] = method
            coef_df["selection_signature"] = selection_signature
            coef_rows.extend(coef_df.to_dict(orient="records"))

            ablation_df = _difference_ablation_df(
                train_df,
                test_df,
                feature_columns=feature_columns,
                seed=args.seed,
            )
            ablation_df["model"] = spec["slug"]
            ablation_df["model_label"] = spec["label"]
            ablation_df["method"] = method
            ablation_df["selection_signature"] = selection_signature
            ablation_rows.extend(ablation_df.to_dict(orient="records"))

            top_diff = coef_df.loc[coef_df["is_difference_feature"]].iloc[0].to_dict()
            top_base = (
                coef_df.loc[coef_df["is_difference_feature"]]
                .groupby("difference_base", as_index=False)["coef_norm"]
                .sum()
                .sort_values("coef_norm", ascending=False)
                .iloc[0]
                .to_dict()
            )
            biggest_drop = ablation_df.iloc[0].to_dict()
            summary_rows.append(
                {
                    "model": spec["slug"],
                    "model_label": spec["label"],
                    "method": method,
                    "selection_signature": selection_signature,
                    "top_difference_feature": str(top_diff["feature"]),
                    "top_difference_feature_coef_norm": float(top_diff["coef_norm"]),
                    "top_difference_base": str(top_base["difference_base"]),
                    "top_difference_base_coef_norm_sum": float(top_base["coef_norm"]),
                    "largest_ablation": str(biggest_drop["ablation"]),
                    "largest_macro_f1_drop": float(biggest_drop["macro_f1_drop"]),
                }
            )

    coef_df = pd.DataFrame(coef_rows).sort_values(
        ["model_label", "method", "coef_norm"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    ablation_df = pd.DataFrame(ablation_rows).sort_values(
        ["model_label", "method", "macro_f1_drop"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["model_label", "method"]).reset_index(drop=True)

    coef_path = output_root / "clamber_token_cloud_difference_feature_coefficients.parquet"
    ablation_path = output_root / "clamber_token_cloud_difference_feature_ablations.parquet"
    summary_path = output_root / "clamber_token_cloud_difference_feature_summary.parquet"
    report_path = output_root / "clamber_token_cloud_difference_feature_report.md"
    write_parquet(coef_df, coef_path)
    write_parquet(ablation_df, ablation_path)
    write_parquet(summary_df, summary_path)

    lines = [
        "# CLAMBER Token-Cloud Difference Feature Importance",
        "",
        "Difference features are the prototype-distance terms:",
        "- `h*_wasserstein_to_clear`",
        "- `h*_wasserstein_to_ambiguous`",
        "- `h*_bottleneck_to_clear`",
        "- `h*_bottleneck_to_ambiguous`",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['model_label']} / {row['method']}",
                "",
                f"- Selection: `{row['selection_signature']}`",
                f"- Highest-norm difference feature: `{row['top_difference_feature']}` "
                f"(`{row['top_difference_feature_coef_norm']:.4f}`)",
                f"- Highest aggregated difference family: `{row['top_difference_base']}` "
                f"(`{row['top_difference_base_coef_norm_sum']:.4f}`)",
                f"- Largest macro-F1 drop under ablation: `{row['largest_ablation']}` "
                f"(`{row['largest_macro_f1_drop']:.4f}`)",
                "",
            ]
        )
    write_markdown(report_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
