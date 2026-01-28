from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import functools
import random
import time
from typing import Any

from llm_wc.client.config import ClientConfig
from llm_wc.client.types import ChatMessage, ModelResponse


class ProviderBase(ABC):
    """Abstract base class for provider adapters."""

    name: str

    @abstractmethod
    def build_client(self, config: ClientConfig) -> Any:
        """Build and return the provider-specific client object."""

    @abstractmethod
    def chat(
        self,
        client: Any,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: object,
    ) -> ModelResponse:
        """Send a chat request and return a normalized response."""


def with_retry(
    *,
    retry_exceptions: tuple[type[BaseException], ...],
    max_attempts: int = 3,
    initial_backoff_s: float = 2.0,
    max_backoff_s: float = 30.0,
    retry_status_codes: set[int] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry decorator with exponential backoff and optional status-code checks."""
    retry_status_codes = retry_status_codes or set(range(500, 600))

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if max_attempts < 1:
                raise ValueError("max_attempts must be >= 1")
            attempt = 0
            backoff_s = max(0.0, float(initial_backoff_s))
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as exc:
                    if attempt >= max_attempts:
                        raise
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    if not (
                        isinstance(status, int)
                        and status in retry_status_codes
                        and attempt < max_attempts
                    ):
                        raise
                sleep_s = min(backoff_s, float(max_backoff_s))
                sleep_s = sleep_s + random.uniform(0.0, min(1.0, sleep_s / 4.0))
                time.sleep(sleep_s)
                backoff_s = min(max_backoff_s, max(backoff_s * 2.0, 0.1))

        return wrapper

    return decorator


__all__ = ["ProviderBase", "with_retry"]
