from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(_read_yaml(path.parent / base), cfg)
    return cfg


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving `base:` inheritance and `key.path=value` overrides."""
    cfg = _read_yaml(Path(path))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        _set_nested(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def apply_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of `cfg` for one training seed.

    The run name and the stage-1 checkpoint it is initialized from get a `_seed{N}` suffix,
    so `text_contrastive_seed43` initializes `semantic_anchors_seed43`.
    The data split seed is not affected.
    """
    cfg = copy.deepcopy(cfg)
    cfg["training"]["seed"] = int(seed)
    cfg["name"] = f"{cfg['name']}_seed{seed}"
    if cfg["training"].get("init_from"):
        cfg["training"]["init_from"] = f"{cfg['training']['init_from']}_seed{seed}"
    return cfg


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def run_dir(cfg: dict[str, Any], name: str | None = None) -> Path:
    return Path(cfg["paths"]["output_dir"]) / "runs" / (name or cfg["name"])


def manifest_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["output_dir"]) / "cache" / "manifest.csv"


def text_cache_dir(cfg: dict[str, Any]) -> Path:
    model_name = cfg["text_encoder"]["model_name"]
    return Path(cfg["paths"]["output_dir"]) / "cache" / "text_embeddings" / slugify(model_name)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
