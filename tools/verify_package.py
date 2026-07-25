#!/usr/bin/env python3
"""Check repository structure without downloading models or datasets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "external/PPLM": "e236b8989322128360182d29a79944627957ad47",
    "external/TruthX": "a41093a6ae3bcbcb523759da782de0f329d03d91",
    "external/Lookback-Lens": "e0a1fa3a898fbf6512af7be5567dea8ffe7a6620",
    "external/ReDEeP-ICLR": "4d081915b8fb4430fda65c411da61540cc73cc57",
    "external/honest_llama": "2c6b2179be7b5aa8f0a171688cf9e01b812ca327",
    "external/TruthfulQA": "d71c110897f5d31c5d7f309e7bc316c152f6f031",
}
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
)


def check_submodules() -> None:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not (path / ".git").exists():
            raise RuntimeError(f"missing submodule: {relative}")
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual != expected:
            raise RuntimeError(f"{relative}: expected {expected}, found {actual}")


def check_environment_specs() -> None:
    for name in ("iti", "topology", "caa"):
        spec = ROOT / "environments" / name
        for relative in ("README.md", "tracked.patch", "overlay"):
            if not (spec / relative).exists():
                raise RuntimeError(f"missing environments/{name}/{relative}")
    bases = {
        "topology": "base_clean",
        "caa": "base",
    }
    for name, base in bases.items():
        if not (ROOT / "environments" / name / base).is_dir():
            raise RuntimeError(f"missing environments/{name}/{base}")


def scan_text_secrets() -> None:
    excluded = {".git", "external", "workspaces"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise RuntimeError(f"possible credential in {path.relative_to(ROOT)}")


def main() -> None:
    check_submodules()
    check_environment_specs()
    scan_text_secrets()
    print("Package structure, pinned revisions, and credential scan: OK")


if __name__ == "__main__":
    main()
