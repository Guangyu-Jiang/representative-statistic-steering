"""Configuration loading utilities for the research pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def _read_yaml(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(project_root: Path, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith(("hf://", "s3://")):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def _resolve_nested_paths(project_root: Path, node: Any) -> Any:
    if isinstance(node, dict):
        resolved: ConfigDict = {}
        for key, value in node.items():
            if key.endswith(("_path", "_dir", "_root")) and value is not None:
                resolved[key] = _resolve_path(project_root, value)
            elif key.endswith("_paths") and isinstance(value, list):
                resolved[key] = [_resolve_path(project_root, item) for item in value]
            else:
                resolved[key] = _resolve_nested_paths(project_root, value)
        return resolved
    if isinstance(node, list):
        return [_resolve_nested_paths(project_root, item) for item in node]
    return node


def _discover_project_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ValueError(f"Could not locate project root from {start_path}")


def _load_config_tree(config_path: Path) -> ConfigDict:
    config = _read_yaml(config_path)
    extends = config.pop("extends", None)
    if extends is None:
        return config
    extends_path = Path(extends)
    if not extends_path.is_absolute():
        extends_path = (config_path.parent / extends_path).resolve()
    base_config = _load_config_tree(extends_path)
    return _deep_merge(base_config, config)


def load_config(config_path: str | Path) -> ConfigDict:
    """Load and resolve a YAML configuration file."""

    config_path = Path(config_path).resolve()
    project_root = _discover_project_root(config_path.parent)
    config = _load_config_tree(config_path)

    model_profile_path = config.get("model", {}).get("profile_path")
    if model_profile_path:
        model_config = _read_yaml(Path(_resolve_path(project_root, model_profile_path)))
        config["model"] = _deep_merge(model_config, config.get("model", {}))

    config = _resolve_nested_paths(project_root, config)
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    return config
