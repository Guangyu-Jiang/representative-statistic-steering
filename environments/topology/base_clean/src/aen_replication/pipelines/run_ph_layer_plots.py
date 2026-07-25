"""Render per-layer persistence diagrams and barcodes from saved PH artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.ph_layer_plots import run_ph_layer_plot_analysis
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
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "run_ph_layer_plots.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    layer_plot_config = config.get("ph_layer_plots", {})
    if not bool(layer_plot_config.get("enabled", True)):
        return

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    diagrams_path = Path(config["persistent_homology"]["output_dir"]) / model_slug / str(
        config["persistent_homology"]["diagrams_filename"]
    )
    output_dir = Path(layer_plot_config["output_dir"]) / model_slug
    run_ph_layer_plot_analysis(
        model_name=model_name,
        diagrams_path=diagrams_path,
        layer_plot_config={**layer_plot_config, "output_dir": str(output_dir)},
    )


if __name__ == "__main__":
    main()
