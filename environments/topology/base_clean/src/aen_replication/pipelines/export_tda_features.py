"""Export all-layer ambiguity features for later TDA analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.layerwise import export_layerwise_tda_features
from aen_replication.utils.io_utils import append_command_history, slugify
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "export_tda_features.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)
    if not bool(config.get("tda_export", {}).get("enabled", True)):
        LOGGER.info("tda_export.enabled=false; skipping all-layer export.")
        return

    model_slug = slugify(config["model"]["name"])
    manifest_paths = [
        Path(config["extraction"]["cache_dir"]) / model_slug / f"{dataset}_manifest.json"
        for dataset in config["tda_export"]["datasets"]
    ]
    export_layerwise_tda_features(
        manifest_paths=manifest_paths,
        probe_config=config["probe"],
        export_config=config["tda_export"],
        seed=int(config["seed"]),
    )


if __name__ == "__main__":
    main()
