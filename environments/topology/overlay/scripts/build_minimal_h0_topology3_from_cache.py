#!/usr/bin/env python3
"""Build minimal H0 topology3 features from token-cloud forward caches.

For H0 persistent homology of a finite Euclidean point cloud, the finite
persistence lifetimes are the edge lengths of the Euclidean minimum spanning
tree. This script uses that equivalence to compute only the three features
needed by Topo local raw steering, and skips the expensive classifier-only
distance/prototype feature blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm


FEATURES = (
    "h0_mean_persistence",
    "h0_persistence_entropy",
    "h0_top5_persistence_fraction",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="token_cloud_forward_cache.joblib")
    parser.add_argument("--output", required=True, help="Output token_cloud_topology_features.parquet")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _normalized_entropy(lifetimes: np.ndarray) -> float:
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if lifetimes.size <= 1:
        return 0.0
    weights = lifetimes / lifetimes.sum()
    entropy = float(-(weights * np.log(weights + 1e-12)).sum())
    return float(entropy / np.log(len(weights)))


def _h0_lifetimes_from_mst(cloud: np.ndarray) -> np.ndarray:
    cloud = np.asarray(cloud, dtype=np.float64)
    if cloud.ndim != 2 or len(cloud) <= 1:
        return np.zeros(0, dtype=np.float64)
    distances = squareform(pdist(cloud, metric="euclidean"))
    tree = minimum_spanning_tree(distances)
    lifetimes = np.asarray(tree.data, dtype=np.float64)
    return lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]


def _feature_values(cloud: np.ndarray) -> dict[str, float]:
    lifetimes = _h0_lifetimes_from_mst(cloud)
    if lifetimes.size == 0:
        return {
            "h0_mean_persistence": 0.0,
            "h0_persistence_entropy": 0.0,
            "h0_top5_persistence_fraction": 0.0,
        }
    sorted_desc = np.sort(lifetimes)[::-1]
    total = float(sorted_desc.sum())
    top5 = float(sorted_desc[:5].sum()) if total > 0.0 else 0.0
    return {
        "h0_mean_persistence": float(lifetimes.mean()),
        "h0_persistence_entropy": _normalized_entropy(lifetimes),
        "h0_top5_persistence_fraction": float(top5 / total) if total > 0.0 else 0.0,
    }


def _row_features(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        "example_id": str(row["example_id"]),
        "pair_id": str(row["pair_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label_ambiguous": int(row["label_ambiguous"]),
        "layer": int(row["layer"]),
        "feature_variant": "single_layer",
        "token_count": int(row["token_count"]),
    }
    output.update(_feature_values(row["cloud"]))
    return output


def _load_cache(path: Path) -> pd.DataFrame:
    payload = joblib.load(path)
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict) and "cloud_df" in payload:
        return pd.DataFrame(payload["cloud_df"])
    raise ValueError(f"Unsupported forward cache payload: {path}")


def _stack_rows(layer_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_columns = ["example_id", "pair_id", "dataset", "split", "label_ambiguous"]
    for keys, group in layer_df.groupby(key_columns, sort=False, dropna=False):
        row = dict(zip(key_columns, keys, strict=True))
        row["layer"] = "stack"
        row["feature_variant"] = "multilayer"
        row["token_count"] = int(group["token_count"].max())
        for _, layer_row in group.sort_values("layer").iterrows():
            suffix = f"__l{int(layer_row['layer']):02d}"
            for feature in FEATURES:
                row[f"{feature}{suffix}"] = float(layer_row[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    cache_path = Path(args.cache).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        print(f"exists: {output_path}")
        return

    cloud_df = _load_cache(cache_path)
    records = cloud_df.to_dict(orient="records")
    layer_rows = Parallel(n_jobs=max(1, int(args.n_jobs)), backend="loky")(
        delayed(_row_features)(row)
        for row in tqdm(records, desc="minimal_h0_topology3")
    )
    layer_df = pd.DataFrame(layer_rows)
    stack_df = _stack_rows(layer_df)
    feature_df = pd.concat([layer_df, stack_df], ignore_index=True, sort=False)
    feature_df["layer"] = feature_df["layer"].astype(str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
    feature_df.to_csv(output_path.with_suffix(".csv"), index=False)
    metadata = {
        "source_cache": str(cache_path),
        "rows": int(len(feature_df)),
        "single_layer_rows": int(len(layer_df)),
        "stack_rows": int(len(stack_df)),
        "features": list(FEATURES),
        "method": "h0_lifetimes_from_euclidean_mst",
    }
    pd.Series(metadata).to_json(output_path.with_suffix(".metadata.json"), indent=2)
    print(f"wrote {output_path} rows={len(feature_df)} stack_rows={len(stack_df)}")


if __name__ == "__main__":
    main()
