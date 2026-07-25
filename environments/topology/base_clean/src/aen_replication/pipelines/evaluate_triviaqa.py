"""Evaluate exact-model AEN probes on TriviaQA clear questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.eval.triviaqa_eval import evaluate_triviaqa_false_positives
from aen_replication.models.hf_model import load_hf_model
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "evaluate_triviaqa.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    bundle = load_hf_model(config["model"], config["extraction"])
    evaluate_triviaqa_false_positives(bundle=bundle, config=config)


if __name__ == "__main__":
    main()
