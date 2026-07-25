"""Run AEN-style ambiguity detection on CLAMBER."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.train.clamber_detection import run_clamber_detection_experiment
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
    setup_logging(
        config["runtime"]["log_level"],
        Path(config["runtime"]["log_dir"]) / "train_clamber_detection.log",
    )
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)
    run_clamber_detection_experiment(config)


if __name__ == "__main__":
    main()
