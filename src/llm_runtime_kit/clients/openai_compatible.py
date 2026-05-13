"""OpenAI-compatible chat completions client.

Supports Ollama, OpenAI, OpenRouter, and similar APIs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_runtime_kit.config import ModelProfile, ProviderConfig
from llm_runtime_kit.credentials import CredentialResolver
from llm_runtime_kit.types import LLMRequest, LLMResponse

Urlopen = Callable[..., object]


class OpenAICompatibleClient:
    """HTTP client for OpenAI-compatible `/chat/completions` endpoints."""

    def __init__(
        self,
        opener: Urlopen = urlopen,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        self._opener = opener
        self._credential_resolver = credential_resolver or CredentialResolver()

    def complete(
        self,
        request: LLMRequest,
        provider_name: str,
        provider: ProviderConfig,
        profile: ModelProfile,
    ) -> LLMResponse:
        """Execute one chat completion."""

        model = request.model or profile.model
        instruction_role = (
            "developer" if provider.kind in {"openai", "openrouter"} else "system"
        )
        output_token_key = (
            "max_completion_tokens"
            if provider.kind in {"openai", "openrouter"}
            else "max_tokens"
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": instruction_role, "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature
            if request.temperature is not None
            else profile.temperature,
            output_token_key: request.max_tokens
            if request.max_tokens is not None
            else profile.max_tokens,
            "stream": False,
        }
        if provider.kind == "ollama" and profile.think is not None:
            payload["think"] = profile.think
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = (
            self._credential_resolver.get(provider.api_key_name)
            if provider.api_key_name
            else ""
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider.api_key_name and not api_key and provider.kind != "ollama":
            return LLMResponse(
                False,
                "",
                provider_name,
                model,
                error=f"{provider.api_key_name} not set",
            )
        req = Request(
            f"{provider.base_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(req, timeout=provider.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            return LLMResponse(
                False,
                "",
                provider_name,
                model,
                error=str(error),
                status_code=error.code,
            )
        except URLError as error:
            return LLMResponse(False, "", provider_name, model, error=str(error.reason))
        except TimeoutError as error:
            return LLMResponse(False, "", provider_name, model, error=str(error))
        parsed = json.loads(raw)
        choices = parsed.get("choices", [])
        if not choices:
            return LLMResponse(
                False, "", provider_name, model, error="response missing choices"
            )
        message = choices[0].get("message", {}) or {}
        text = (
            message.get("content")
            or message.get("reasoning")
            or message.get("thinking")
            or ""
        )
        usage = parsed.get("usage", {}) or {}
        return LLMResponse(
            bool(str(text).strip()),
            str(text),
            provider_name,
            model,
            tokens_used=int(usage.get("total_tokens") or 0),
            error=None if str(text).strip() else "empty response",
        )
