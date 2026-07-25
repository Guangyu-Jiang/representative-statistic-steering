"""Run Mapper summaries from saved layerwise ambiguity features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.mapper_analysis import run_mapper_analysis
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_mapper_analysis.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    mapper_config = config.get("mapper_analysis", {})
    if not bool(mapper_config.get("enabled", True)):
        return

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    features_path = Path(config["tda_export"]["output_dir"]) / model_slug / str(
        config["tda_export"]["layer_features_filename"]
    )
    output_dir = Path(mapper_config["output_dir"]) / model_slug

    run_mapper_analysis(
        model_name=model_name,
        layerwise_features_path=features_path,
        mapper_config={**mapper_config, "output_dir": str(output_dir)},
    )


if __name__ == "__main__":
    main()
