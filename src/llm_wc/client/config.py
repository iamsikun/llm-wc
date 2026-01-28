from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientConfig:
    """Configuration for building an LLM client."""

    provider: str
    model: str
    api_base: str | None = None
    api_key: str | None = None
