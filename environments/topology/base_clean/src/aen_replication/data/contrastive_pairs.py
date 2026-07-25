"""Paper-style contrastive split construction."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from aen_replication.data.ambigqa import build_ambigqa_pairs
from aen_replication.data.schema import coerce_pairs
from aen_replication.data.situatedqa import build_situatedqa_pairs

LOGGER = logging.getLogger(__name__)


def _assign_train_test(df: pd.DataFrame, train_pairs: int, test_pairs: int, seed: int) -> pd.DataFrame:
    pair_ids = df["pair_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    needed = train_pairs + test_pairs
    if len(pair_ids) < needed:
        raise ValueError(f"Need {needed} pairs, found {len(pair_ids)}")
    train_set = set(pair_ids[:train_pairs])
    test_set = set(pair_ids[train_pairs : train_pairs + test_pairs])
    split_map = {pair_id: "train" for pair_id in train_set}
    split_map.update({pair_id: "test" for pair_id in test_set})
    out = df[df["pair_id"].isin(split_map)].copy()
    out["split"] = out["pair_id"].map(split_map)
    return out.reset_index(drop=True)


def prepare_all_datasets(config: dict) -> dict[str, pd.DataFrame]:
    """Build paper-style AmbigQA and SituatedQA train/test pair tables."""

    split_cfg = config["data"]["split"]
    train_pairs = int(split_cfg["train_pairs"])
    test_pairs = int(split_cfg["test_pairs"])
    seed = int(config["seed"])
    selection_strategy = str(config["data"].get("clear_selection", "random_seeded"))

    ambigqa_df = build_ambigqa_pairs(
        train_path=config["data"]["ambigqa"]["train_path"],
        dev_path=config["data"]["ambigqa"]["dev_path"],
        seed=seed,
        selection_strategy=selection_strategy,
    )
    ambigqa_df = _assign_train_test(ambigqa_df, train_pairs=train_pairs, test_pairs=test_pairs, seed=seed)

    situatedqa_df = build_situatedqa_pairs(
        temp_paths=list(config["data"]["situatedqa"]["temp_paths"]),
        geo_paths=list(config["data"]["situatedqa"]["geo_paths"]),
        seed=seed,
        selection_strategy=selection_strategy,
    )
    situatedqa_df = _assign_train_test(
        situatedqa_df,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        seed=seed,
    )

    return {
        "ambigqa": coerce_pairs(ambigqa_df),
        "situatedqa": coerce_pairs(situatedqa_df),
    }


def save_prepared_pairs(config: dict) -> dict[str, str]:
    """Prepare and save dataset-specific contrastive pair files."""

    output_dir = Path(config["data"]["pair_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for dataset_name, df in prepare_all_datasets(config).items():
        path = output_dir / f"{dataset_name}_pairs.parquet"
        df.to_parquet(path, index=False)
        outputs[dataset_name] = str(path)
        LOGGER.info("Saved %s contrastive pairs to %s (%d rows)", dataset_name, path, len(df))
    return outputs
