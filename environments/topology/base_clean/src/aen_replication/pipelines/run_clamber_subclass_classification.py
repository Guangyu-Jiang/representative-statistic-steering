"""Run CLAMBER subclass classification experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.train.clamber_subclass_classification import run_clamber_subclass_classification
from aen_replication.utils.io_utils import append_command_history
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        config["runtime"]["log_level"],
        Path(config["runtime"]["log_dir"]) / "run_clamber_subclass_classification.log",
    )
    LOGGER.info("Starting CLAMBER subclass classification: config=%s", args.config)
    try:
        set_global_seed(int(config["seed"]))
        append_command_history(config["runtime"]["command_history_path"], sys.argv)
        outputs = run_clamber_subclass_classification(config=config, seed=int(config["seed"]))
        LOGGER.info("Completed CLAMBER subclass classification: outputs=%s", outputs)
    except Exception:
        LOGGER.exception("CLAMBER subclass classification failed: config=%s", args.config)
        raise


if __name__ == "__main__":
    main()
