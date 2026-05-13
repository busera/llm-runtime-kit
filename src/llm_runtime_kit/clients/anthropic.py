"""Anthropic Messages API client."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_runtime_kit.config import ModelProfile, ProviderConfig
from llm_runtime_kit.credentials import CredentialResolver
from llm_runtime_kit.types import LLMRequest, LLMResponse

Urlopen = Callable[..., object]


class AnthropicClient:
    """HTTP client for Anthropic's Messages API."""

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
        """Execute one Anthropic message request."""

        model = request.model or profile.model
        api_key = (
            self._credential_resolver.get(provider.api_key_name)
            if provider.api_key_name
            else ""
        )
        if not api_key:
            return LLMResponse(
                False,
                "",
                provider_name,
                model,
                error=f"{provider.api_key_name} not set",
            )
        payload = {
            "model": model,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature
            if request.temperature is not None
            else profile.temperature,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else profile.max_tokens,
        }
        req = Request(
            f"{provider.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
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
        chunks = parsed.get("content", [])
        text = "".join(
            str(chunk.get("text", ""))
            for chunk in chunks
            if chunk.get("type") == "text"
        )
        usage = parsed.get("usage", {}) or {}
        tokens = int(usage.get("input_tokens") or 0) + int(
            usage.get("output_tokens") or 0
        )
        return LLMResponse(
            bool(text.strip()),
            text,
            provider_name,
            model,
            tokens_used=tokens,
            error=None if text.strip() else "empty response",
        )
