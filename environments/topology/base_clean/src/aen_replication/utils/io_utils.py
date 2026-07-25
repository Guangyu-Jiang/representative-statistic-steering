"""I/O helpers and artifact utilities."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON to disk with stable formatting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_markdown(path: str | Path, content: str) -> None:
    """Write markdown text to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame as parquet."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_commit(project_root: str | Path) -> str | None:
    """Return the current git commit if available."""

    try:
        output = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return output.strip() or None


def slugify(text: str) -> str:
    """Convert a model or artifact name into a filesystem-safe slug."""

    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char.lower())
        else:
            allowed.append("_")
    slug = "".join(allowed)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def append_command_history(path: str | Path, argv: Iterable[str]) -> None:
    """Append a timestamped command line to the command log."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(arg) for arg in argv)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now_iso()}\t{command}\n")
