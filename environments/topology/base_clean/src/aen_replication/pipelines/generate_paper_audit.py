"""Generate a paper-to-artifact replication audit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aen_replication.config import load_config
from aen_replication.eval.paper_audit import generate_paper_replication_audit
from aen_replication.utils.io_utils import append_command_history
from aen_replication.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paper-pdf", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / "generate_paper_audit.log")
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    project_root = Path(config["_meta"]["project_root"])
    audit_cfg = config.get("paper_audit", {})
    paper_pdf_path = args.paper_pdf or audit_cfg.get("paper_pdf_path") or (project_root / "2025.emnlp-main.813.pdf")
    output_dir = args.output_dir or audit_cfg.get("output_dir") or (project_root / "artifacts" / "reports" / "paper_audit")
    outputs = generate_paper_replication_audit(
        project_root=project_root,
        paper_pdf_path=paper_pdf_path,
        output_dir=output_dir,
    )
    LOGGER.info("Paper audit report written to %s", outputs["report_path"])


if __name__ == "__main__":
    main()
