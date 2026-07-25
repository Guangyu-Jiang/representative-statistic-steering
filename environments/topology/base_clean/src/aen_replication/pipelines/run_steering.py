"""Run local activation-steering experiments."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from aen_replication.config import load_config
from aen_replication.eval.judge import load_judge
from aen_replication.models.hf_model import load_hf_model
from aen_replication.train.steering import (
    generate_base_behavior_outputs,
    generate_steering_outputs,
    judge_base_behavior_outputs,
    judge_steering_outputs_and_summarize,
)
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_steering.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    official_annotations = config.get("steering", {}).get("base_behavior_source") == "official_annotations"

    if official_annotations:
        generate_base_behavior_outputs(bundle=None, config=config)
    else:
        bundle = load_hf_model(config["model"], config["generation"])
        generate_base_behavior_outputs(bundle=bundle, config=config)
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judge = load_judge(config)
    judge_base_behavior_outputs(config=config, judge=judge)
    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bundle = load_hf_model(config["model"], config["generation"])
    generate_steering_outputs(bundle=bundle, config=config)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    judge = load_judge(config)
    judge_steering_outputs_and_summarize(config=config, judge=judge)


if __name__ == "__main__":
    main()
