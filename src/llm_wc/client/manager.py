from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_wc.client.config import ClientConfig
from llm_wc.client.providers.base import ProviderBase
from llm_wc.client.providers.openai import OpenAIProvider
from llm_wc.client.providers.anthropic import AnthropicProvider
from llm_wc.client.types import ChatMessage, ModelResponse


@dataclass
class ProviderRegistry:
    """Registry for LLM provider adapters."""

    _providers: dict[str, ProviderBase]

    def register(self, provider: ProviderBase) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> ProviderBase:
        if name not in self._providers:
            raise KeyError(f"Unknown provider '{name}'")
        return self._providers[name]


class LLMClient:
    """Provider-agnostic LLM client facade."""

    def __init__(self, config: ClientConfig, provider: ProviderBase) -> None:
        self.config = config
        self._provider = provider
        self._client = provider.build_client(config)

    def chat(
        self,
        messages: list[ChatMessage],
        **kwargs: object,
    ) -> ModelResponse:
        return self._provider.chat(
            self._client,
            messages,
            model=self.config.model,
            **kwargs,
        )


def default_registry() -> ProviderRegistry:
    """Create a provider registry with built-in providers."""
    registry = ProviderRegistry(_providers={})
    registry.register(OpenAIProvider())
    registry.register(AnthropicProvider())
    return registry


def build_client(
    config: ClientConfig, *, registry: ProviderRegistry | None = None
) -> LLMClient:
    """Build an LLMClient using the provided provider registry."""
    registry = registry or default_registry()
    provider = registry.get(config.provider)
    return LLMClient(config, provider)


__all__ = ["ClientConfig", "LLMClient", "ProviderRegistry", "build_client"]
