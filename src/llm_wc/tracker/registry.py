"""Read ``tracker/registry.yaml`` and ``tracker/candidates.yaml``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from llm_wc.tracker.paths import CANDIDATES, REGISTRY


@lru_cache(maxsize=4)
def _load(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def registry(path: Path = REGISTRY) -> dict:
    return _load(Path(path))


def settings(path: Path = REGISTRY) -> dict[str, dict]:
    return registry(path)["settings"]


def setting(key: str, path: Path = REGISTRY) -> dict:
    return settings(path)[key]


def settings_for_source(source: str, path: Path = REGISTRY) -> list[str]:
    return [k for k, s in settings(path).items() if s["source"] == source]


def analysis_params(path: Path = REGISTRY) -> dict:
    return registry(path)["analysis"]


def candidates(path: Path = CANDIDATES) -> list[dict]:
    if not Path(path).is_file():
        return []
    return _load(Path(path)).get("candidates", [])
