from __future__ import annotations

from typing import Any

from llm_wc.client.config import ClientConfig
from llm_wc.client.providers.base import ProviderBase, with_retry
from llm_wc.client.types import ChatMessage, ModelResponse, TokenUsage


ANTHROPIC_DEFAULT_REQUEST_PARAMS: dict[str, object] = {
    "max_tokens": 1024,
}


class AnthropicProvider(ProviderBase):
    """Adapter for Anthropic's Messages API."""

    name = "anthropic"

    def build_client(self, config: ClientConfig) -> Any:
        if config.api_key is None:
            raise ValueError("api_key is required for anthropic provider")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic SDK is not installed. Install 'anthropic' to use this provider."
            ) from exc
        return Anthropic(api_key=config.api_key)

    def chat(
        self,
        client: Any,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: object,
    ) -> ModelResponse:
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_messages.append({"role": role, "content": content})

        params: dict[str, object] = dict(ANTHROPIC_DEFAULT_REQUEST_PARAMS)
        params.update({k: v for k, v in kwargs.items() if v is not None})

        request_kwargs = dict(params)
        if system_parts:
            request_kwargs["system"] = "\n\n".join(system_parts)
        @with_retry(retry_exceptions=(TimeoutError, OSError))
        def _create_message():
            return client.messages.create(
                model=model,
                messages=user_messages,
                **request_kwargs,
            )

        response = _create_message()

        text = ""
        reasoning_parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text += getattr(block, "text", "")
            elif block_type in {"thinking", "reasoning"}:
                reasoning_parts.append(getattr(block, "text", ""))

        usage = getattr(response, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "input_tokens", None),
                completion_tokens=getattr(usage, "output_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        return ModelResponse(
            content=text,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
            usage=token_usage,
            finish_reason=getattr(response, "stop_reason", None),
            model=getattr(response, "model", None),
            raw=response,
        )


__all__ = ["AnthropicProvider"]
