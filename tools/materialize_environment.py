#!/usr/bin/env python3
"""Create a runnable experiment workspace from a pinned baseline and overlay."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    "iti": {
        "base": ROOT / "external" / "honest_llama",
        "spec": ROOT / "environments" / "iti",
        "extras": [
            (ROOT / "external" / "TruthfulQA", Path("TruthfulQA")),
        ],
    },
    "topology": {
        "base": ROOT / "environments" / "topology" / "base_clean",
        "spec": ROOT / "environments" / "topology",
    },
    "caa": {
        "base": ROOT / "environments" / "caa" / "base",
        "spec": ROOT / "environments" / "caa",
    },
}

IGNORED_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "logs",
    "outputs",
    "results",
    "wandb",
}


def ignore_generated(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_NAMES or name.endswith(".pyc")}


def copy_overlay(overlay: Path, destination: Path) -> None:
    for source in overlay.rglob("*"):
        relative = source.relative_to(overlay)
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.name not in IGNORED_NAMES and not source.name.endswith(".pyc"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def materialize(name: str, output: Path, force: bool) -> None:
    environment = ENVIRONMENTS[name]
    base = environment["base"]
    spec = environment["spec"]
    patch = spec / "tracked.patch"
    overlay = spec / "overlay"

    if not base.exists():
        raise SystemExit(f"Missing packaged base tree: {base.relative_to(ROOT)}")
    if str(base).startswith(str(ROOT / "external")) and not (base / ".git").exists():
        raise SystemExit(
            f"Submodule {base.relative_to(ROOT)} is not initialized. "
            "Run: git submodule update --init --recursive"
        )
    if output.exists():
        if not force:
            raise SystemExit(f"{output} exists; pass --force to rebuild it")
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, output, ignore=ignore_generated)
    for extra, relative in environment.get("extras", []):
        if not (extra / ".git").exists():
            raise SystemExit(
                f"Submodule {extra.relative_to(ROOT)} is not initialized. "
                "Run: git submodule update --init --recursive"
            )
        shutil.copytree(extra, output / relative, ignore=ignore_generated)

    if patch.exists() and patch.stat().st_size:
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch)],
            cwd=output,
            check=True,
        )
    copy_overlay(overlay, output)
    print(f"Materialized {name} at {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=sorted(ENVIRONMENTS))
    parser.add_argument(
        "--output",
        type=Path,
        help="destination (default: workspaces/<environment>)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "workspaces" / args.environment
    materialize(args.environment, output.resolve(), args.force)


if __name__ == "__main__":
    main()
