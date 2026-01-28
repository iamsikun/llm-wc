from __future__ import annotations

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from llm_wc.client.config import ClientConfig
from llm_wc.client.providers.base import ProviderBase, with_retry
from llm_wc.client.types import ChatMessage, ModelResponse, TokenUsage


OPENAI_DEFAULT_REQUEST_PARAMS: dict[str, object] = {
    "temperature": 0,
    "max_tokens": 32768,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


class OpenAIProvider(ProviderBase):
    """Adapter for OpenAI-compatible chat endpoints (including Ollama)."""

    name = "openai"

    def build_client(self, config: ClientConfig):
        if not config.api_base:
            raise ValueError("api_base is required for openai provider")
        if config.api_key is None:
            raise ValueError("api_key is required for openai provider")
        return OpenAI(api_key=config.api_key, base_url=config.api_base)

    def chat(
        self,
        client,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: object,
    ) -> ModelResponse:
        params: dict[str, object] = dict(OPENAI_DEFAULT_REQUEST_PARAMS)
        params.update({k: v for k, v in kwargs.items() if v is not None})

        @with_retry(
            retry_exceptions=(APITimeoutError, APIConnectionError, RateLimitError)
        )
        def _create_completion():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **params,
            )

        completion = _create_completion()
        if not completion.choices:
            raise ValueError("Chat completion returned no choices")
        message = completion.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )

        usage = getattr(completion, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        finish_reason = getattr(completion.choices[0], "finish_reason", None)
        response_model = getattr(completion, "model", None)

        return ModelResponse(
            content=content,
            reasoning=reasoning,
            usage=token_usage,
            finish_reason=finish_reason,
            model=response_model,
            raw=completion,
        )


__all__ = [
    "OPENAI_DEFAULT_REQUEST_PARAMS",
    "OpenAIProvider",
]
