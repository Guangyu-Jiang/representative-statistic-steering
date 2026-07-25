"""Run the independent topology classifier from raw hidden states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.train.independent_topology_classifier import run_independent_topology_classifier_analysis
from aen_replication.utils.io_utils import append_command_history, slugify
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
        Path(config["runtime"]["log_dir"]) / "run_independent_topology_classifier.log",
    )
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    model_name = str(config["model"]["name"])
    hidden_state_root = Path(config["extraction"]["cache_dir"]) / slugify(model_name)
    if not hidden_state_root.exists():
        raise FileNotFoundError(f"Hidden-state cache directory not found: {hidden_state_root}")

    run_independent_topology_classifier_analysis(
        model_name=model_name,
        hidden_state_root=hidden_state_root,
        classifier_config=config["independent_topology_classifier"],
        seed=int(config["seed"]),
    )


if __name__ == "__main__":
    main()
