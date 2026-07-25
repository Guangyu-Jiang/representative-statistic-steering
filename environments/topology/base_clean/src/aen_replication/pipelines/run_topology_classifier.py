"""Run the local topology-based ambiguity classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.train.topology_classifier import run_topology_classifier_analysis
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_topology_classifier.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    layerwise_features_path = (
        Path(config["tda_export"]["output_dir"])
        / model_slug
        / str(config["tda_export"]["layer_features_filename"])
    )
    if not layerwise_features_path.exists():
        raise FileNotFoundError(
            f"Layerwise ambiguity features are required before topology classification: {layerwise_features_path}"
        )

    run_topology_classifier_analysis(
        model_name=model_name,
        layerwise_features_path=layerwise_features_path,
        classifier_config=config["topology_classifier"],
        seed=int(config["seed"]),
    )


if __name__ == "__main__":
    main()
