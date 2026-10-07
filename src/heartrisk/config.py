"""Configuration loading: packaged defaults <- user YAML <- ``key.sub=value`` overrides."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_config() -> dict[str, Any]:
    text = resources.files("heartrisk").joinpath("configs/default.yaml").read_text()
    return yaml.safe_load(text)


def apply_override(cfg: dict, assignment: str) -> dict:
    """Apply ``a.b.c=value`` (value parsed as YAML, so lists/numbers/bools work)."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key.sub=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path:
        with open(path) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        cfg = apply_override(cfg, item)
    return cfg
