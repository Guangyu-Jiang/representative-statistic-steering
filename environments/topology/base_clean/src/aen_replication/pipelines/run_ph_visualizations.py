"""Render ambiguous-vs-clear PH comparison plots from saved PH artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.ph_visualization import run_ph_visualization_analysis
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_ph_visualizations.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    visualization_config = config.get("ph_visualization", {})
    if not bool(visualization_config.get("enabled", True)):
        return

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    summary_path = Path(config["persistent_homology"]["output_dir"]) / model_slug / str(
        config["persistent_homology"]["summary_filename"]
    )
    focused_distance_path = Path(config["focused_persistent_homology"]["output_dir"]) / model_slug / str(
        config["focused_persistent_homology"]["distance_summary_filename"]
    )
    output_dir = Path(config["ph_visualization"]["output_dir"]) / model_slug

    run_ph_visualization_analysis(
        model_name=model_name,
        summary_path=summary_path,
        output_dir=output_dir,
        visualization_config=visualization_config,
        focused_distance_path=focused_distance_path if focused_distance_path.exists() else None,
    )


if __name__ == "__main__":
    main()
