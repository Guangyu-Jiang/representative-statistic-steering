"""Prepare the released CLAMBER benchmark as a binary ambiguity dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.data.clamber import build_clamber_pairs
from aen_replication.data.schema import coerce_pairs
from aen_replication.utils.io_utils import append_command_history
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "prepare_clamber_data.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    clamber_cfg = config["data"]["clamber"]
    output_dir = Path(config["data"]["pair_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_clamber_pairs(
        source_path=clamber_cfg["source_path"],
        seed=int(config["seed"]),
        train_fraction=float(clamber_cfg.get("train_fraction", 0.8)),
    )
    df = coerce_pairs(df)
    output_path = output_dir / "clamber_pairs.parquet"
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
