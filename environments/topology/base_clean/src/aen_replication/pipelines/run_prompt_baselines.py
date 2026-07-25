"""Run prompting and entropy baselines from the paper appendix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from aen_replication.config import load_config
from aen_replication.eval.baselines import run_baselines
from aen_replication.models.hf_model import load_hf_model
from aen_replication.utils.io_utils import append_command_history, write_json
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_prompt_baselines.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    bundle = load_hf_model(config["model"], config["generation"])
    summaries: dict[str, dict[str, object]] = {}
    for dataset_name in config["baselines"].get("datasets", ["ambigqa", "situatedqa"]):
        pairs_path = Path(config["data"]["pair_output_dir"]) / f"{dataset_name}_pairs.parquet"
        dataset_df = pd.read_parquet(pairs_path)
        summaries[dataset_name] = run_baselines(
            bundle=bundle,
            dataset_df=dataset_df,
            config=config,
            dataset_name=dataset_name,
        )
    write_json(Path(config["baselines"]["artifact_dir"]) / "latest_summary.json", summaries)


if __name__ == "__main__":
    main()
