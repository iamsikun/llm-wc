from __future__ import annotations

"""Shared configuration and utility helpers for benchmark runners."""

import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration options from a run config."""

    name: str
    output_dir: str
    seed: int | None = None
    limit_subjects: list[str] | None = None
    limit_per_subject: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration options from a run config."""

    name: str
    api_base: str
    api_key: str | None = None
    api_key_env: str | None = None
    output_name: str | None = None
    request_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Full run configuration, including benchmark and model settings."""

    benchmark: BenchmarkConfig
    models: list[ModelConfig]


class ProgressBar:
    """Simple stderr progress bar for long-running evaluations."""

    def __init__(self, total: int, *, label: str = "Progress") -> None:
        self.total = max(total, 1)
        self.label = label
        self.count = 0
        self.start = time.time()
        self._render()

    def update(self, step: int = 1) -> None:
        self.count += step
        if self.count > self.total:
            self.count = self.total
        self._render()

    def close(self) -> None:
        self._render(final=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, final: bool = False) -> None:
        elapsed = max(time.time() - self.start, 0.0)
        pct = self.count / self.total if self.total else 1.0
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "#" * filled + "-" * (bar_width - filled)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        msg = (
            f"\r{self.label} [{bar}] {self.count}/{self.total} "
            f"({pct:.1%}) {rate:.2f} it/s"
        )
        if final:
            msg = msg.rstrip()
        sys.stderr.write(msg)
        sys.stderr.flush()


def build_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Build a logger configured for CLI benchmark output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def resolve_api_key(model: ModelConfig, env: dict[str, str] | None = None) -> str:
    """Resolve the API key for a model using explicit or env-based inputs."""
    env = env or os.environ
    if model.api_key_env:
        value = env.get(model.api_key_env)
        if value is None:
            raise ValueError(f"Environment variable '{model.api_key_env}' is not set")
        return value
    if model.api_key is not None:
        return model.api_key
    return "ollama"


def parse_run_config(data: dict[str, Any]) -> RunConfig:
    """Parse a raw config mapping into typed RunConfig structures."""
    if "benchmark" not in data:
        raise KeyError("Missing required key 'benchmark' in config")
    if "models" not in data:
        raise KeyError("Missing required key 'models' in config")

    benchmark_raw = data["benchmark"]
    if not isinstance(benchmark_raw, dict):
        raise ValueError("'benchmark' must be a mapping")
    name = benchmark_raw.get("name")
    output_dir = benchmark_raw.get("output_dir")
    if not name:
        raise ValueError("benchmark.name is required")
    if not output_dir:
        raise ValueError("benchmark.output_dir is required")

    limit_subjects = _normalize_subjects(benchmark_raw.get("limit_subjects"))

    benchmark = BenchmarkConfig(
        name=name,
        output_dir=output_dir,
        seed=benchmark_raw.get("seed"),
        limit_subjects=limit_subjects,
        limit_per_subject=benchmark_raw.get("limit_per_subject"),
    )

    models_raw = data["models"]
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("'models' must be a non-empty list")
    models: list[ModelConfig] = []
    for idx, model_raw in enumerate(models_raw):
        if not isinstance(model_raw, dict):
            raise ValueError(f"models[{idx}] must be a mapping")
        name = model_raw.get("name")
        api_base = model_raw.get("api_base")
        if not name:
            raise ValueError(f"models[{idx}].name is required")
        if not api_base:
            raise ValueError(f"models[{idx}].api_base is required")
        request_overrides = model_raw.get("request_overrides") or {}
        if not isinstance(request_overrides, dict):
            raise ValueError(f"models[{idx}].request_overrides must be a mapping")
        models.append(
            ModelConfig(
                name=name,
                api_base=api_base,
                api_key=model_raw.get("api_key"),
                api_key_env=model_raw.get("api_key_env"),
                output_name=model_raw.get("output_name"),
                request_overrides=request_overrides,
            )
        )

    return RunConfig(benchmark=benchmark, models=models)


def apply_overrides(
    config: RunConfig,
    *,
    output_dir: str | None = None,
    seed: int | None = None,
    limit_subjects: list[str] | None = None,
    limit_subjects_set: bool = False,
    limit_per_subject: int | None = None,
    model_names: set[str] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Apply CLI or programmatic overrides to a RunConfig."""
    benchmark = config.benchmark
    if output_dir is not None:
        benchmark = replace(benchmark, output_dir=output_dir)
    if seed is not None:
        benchmark = replace(benchmark, seed=seed)
    if limit_subjects_set:
        benchmark = replace(benchmark, limit_subjects=limit_subjects)
    if limit_per_subject is not None:
        benchmark = replace(benchmark, limit_per_subject=limit_per_subject)

    models = config.models
    if model_names:
        models = [model for model in models if model.name in model_names]

    if api_base or api_key or api_key_env or request_overrides:
        updated = []
        for model in models:
            updated.append(
                replace(
                    model,
                    api_base=api_base or model.api_base,
                    api_key=api_key if api_key is not None else model.api_key,
                    api_key_env=(
                        api_key_env if api_key_env is not None else model.api_key_env
                    ),
                    request_overrides=_merge_params(
                        request_overrides, model.request_overrides
                    ),
                )
            )
        models = updated

    return RunConfig(benchmark=benchmark, models=models)


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Serialize a RunConfig into a JSON-serializable mapping."""
    return {
        "benchmark": benchmark_to_dict(config.benchmark),
        "models": [model_to_dict(model) for model in config.models],
    }


def benchmark_to_dict(config: BenchmarkConfig) -> dict[str, Any]:
    """Serialize a BenchmarkConfig into a dict."""
    return {
        "name": config.name,
        "output_dir": config.output_dir,
        "seed": config.seed,
        "limit_subjects": config.limit_subjects,
        "limit_per_subject": config.limit_per_subject,
    }


def model_to_dict(config: ModelConfig) -> dict[str, Any]:
    """Serialize a ModelConfig into a dict."""
    return {
        "name": config.name,
        "api_base": config.api_base,
        "api_key": config.api_key,
        "api_key_env": config.api_key_env,
        "output_name": config.output_name,
        "request_overrides": dict(config.request_overrides),
    }


def _merge_params(
    overrides: dict[str, Any] | None, existing: dict[str, Any]
) -> dict[str, Any]:
    """Merge non-null overrides onto an existing mapping."""
    merged = dict(existing)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def _normalize_subjects(value: Any) -> list[str] | None:
    """Normalize subject filters from CLI or config input."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    raise ValueError("limit_subjects must be a list, comma-separated string, or null")
