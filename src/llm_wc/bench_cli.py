from __future__ import annotations

"""CLI helpers shared across benchmark runner scripts.

These utilities consolidate common argument parsing helpers, model availability
checks, and run metadata output used by multiple benchmark entry points.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml

from llm_wc.bench_utils import RunConfig, run_config_to_dict


def sanitize_name(name: str) -> str:
    """Sanitize a string for use as a filesystem-friendly directory name.

    Args:
        name: Raw model name or identifier.

    Returns:
        str: Sanitized name containing only alphanumeric, dot, dash, and underscore.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def parse_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI list into normalized strings.

    Args:
        value: Input string (comma-separated) or None.

    Returns:
        list[str] | None: List of items, or None when no entries are provided.
    """
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_model_filter(value: str | None) -> set[str] | None:
    """Parse the model filter argument into a set of names.

    Args:
        value: Comma-separated model list or None.

    Returns:
        set[str] | None: Set of model names, or None if no filter is applied.
    """
    items = parse_list(value)
    return set(items) if items else None


def list_available_models(
    api_base: str, api_key: str, logger: logging.Logger
) -> set[str] | None:
    """List model IDs available at the given OpenAI-compatible endpoint.

    Args:
        api_base: Base URL for the OpenAI-compatible API.
        api_key: API key or placeholder.
        logger: Logger for warning messages.

    Returns:
        set[str] | None: Set of available model IDs or None if listing fails.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.models.list()
    except Exception as exc:
        logger.warning("Unable to list models for api_base=%s: %s", api_base, exc)
        return None

    models = set()
    data = getattr(response, "data", response)
    for item in data:
        model_id = getattr(item, "id", None)
        if model_id is None and isinstance(item, dict):
            model_id = item.get("id")
        if model_id:
            models.add(model_id)

    if not models:
        logger.warning("Model listing returned empty for api_base=%s", api_base)
    return models


def filter_available_models(
    config: RunConfig,
    logger: logging.Logger,
    *,
    resolve_api_key: Callable[[Any], str],
) -> RunConfig:
    """Filter RunConfig models to those available at their endpoints.

    Args:
        config: RunConfig instance containing model definitions.
        logger: Logger for warning output.

    Returns:
        RunConfig: Updated config with unavailable models removed.
    """
    available_cache: dict[tuple[str, str], set[str] | None] = {}
    filtered_models = []
    for model_cfg in config.models:
        api_key = resolve_api_key(model_cfg)
        cache_key = (model_cfg.api_base, api_key)
        if cache_key not in available_cache:
            available_cache[cache_key] = list_available_models(
                model_cfg.api_base, api_key, logger
            )
        available = available_cache[cache_key]
        if available is not None and model_cfg.name not in available:
            logger.warning(
                "Model '%s' not available at %s, skipping",
                model_cfg.name,
                model_cfg.api_base,
            )
            continue
        filtered_models.append(model_cfg)

    return RunConfig(
        benchmark=config.benchmark,
        models=filtered_models,
    )


def build_output_dir(config_path: Path, output_dir: str | None) -> Path:
    """Construct the output directory for a run.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Optional override for benchmark.output_dir.

    Returns:
        Path: Output directory for the run.
    """
    if output_dir:
        return Path(output_dir)
    config_stem = config_path.stem
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return config_path.parent.parent / "results" / f"{config_stem}_{timestamp}"


def write_run_metadata(
    output_dir: Path,
    config: RunConfig,
    *,
    benchmark_overrides: dict[str, Any] | None = None,
) -> None:
    """Write run metadata and a config snapshot to the output directory.

    Args:
        output_dir: Directory to write run.json/config.yaml into.
        config: RunConfig used for the evaluation.
        benchmark_overrides: Optional benchmark-specific metadata to persist.
    """
    run_meta = run_config_to_dict(config)
    run_meta["benchmark"]["output_dir"] = str(output_dir)
    if benchmark_overrides:
        run_meta["benchmark"].update(benchmark_overrides)

    config_snapshot = dict(run_meta)
    run_meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps(run_meta, indent=2))
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False)
    )
