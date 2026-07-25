"""Run targeted steering diagnostics for the Table 3 mismatch."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

from aen_replication.config import load_config
from aen_replication.eval.judge import load_judge
from aen_replication.eval.steering_diagnostics import run_steering_diagnostics
from aen_replication.models.hf_model import load_hf_model
from aen_replication.utils.io_utils import append_command_history
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def main() -> None:
    if "--config" not in sys.argv:
        raise SystemExit("Usage: python -m aen_replication.pipelines.run_steering_diagnostics --config <path>")
    config_path = sys.argv[sys.argv.index("--config") + 1]
    config = load_config(config_path)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_steering_diagnostics.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    bundle = load_hf_model(config["model"], config["generation"])
    judge = load_judge(config)
    try:
        run_steering_diagnostics(bundle=bundle, config=config, judge=judge)
    finally:
        del judge
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
