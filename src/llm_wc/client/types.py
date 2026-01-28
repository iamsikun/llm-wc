from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response returned by an LLM provider."""

    content: str
    reasoning: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    model: str | None = None
    raw: Any | None = None
