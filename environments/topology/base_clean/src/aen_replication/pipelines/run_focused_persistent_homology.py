"""Run focused ambiguous-vs-clear persistent-homology comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.features.focused_persistent_homology import run_focused_persistent_homology_analysis
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
    setup_logging(
        config["runtime"]["log_level"],
        Path(config["runtime"]["log_dir"]) / "run_focused_persistent_homology.log",
    )
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    focus_config = config.get("focused_persistent_homology", {})
    if not bool(focus_config.get("enabled", True)):
        return

    model_name = str(config["model"]["name"])
    model_slug = slugify(model_name)
    topology_root = Path(config["persistent_homology"]["output_dir"]) / model_slug
    run_focused_persistent_homology_analysis(
        model_name=model_name,
        summary_path=topology_root / str(config["persistent_homology"]["summary_filename"]),
        diagrams_path=topology_root / str(config["persistent_homology"]["diagrams_filename"]),
        focus_config={
            **focus_config,
            "output_dir": str(Path(focus_config["output_dir"]) / model_slug),
        },
    )


if __name__ == "__main__":
    main()
