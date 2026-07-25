"""Run persistent homology over exported layerwise ambiguity subspaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.persistent_homology import run_persistent_homology_analysis
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_persistent_homology.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    ph_config = config.get("persistent_homology", {})
    if not bool(ph_config.get("enabled", True)):
        return

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    feature_summary_path = (
        Path(config["tda_export"]["output_dir"]) / model_slug / str(config["tda_export"]["layer_summary_filename"])
    )

    run_persistent_homology_analysis(
        project_root=config["_meta"]["project_root"],
        model_name=model_name,
        feature_summary_path=feature_summary_path,
        cache_dir=config["extraction"]["cache_dir"],
        ph_config=ph_config,
        seed=int(config["seed"]),
    )


if __name__ == "__main__":
    main()
