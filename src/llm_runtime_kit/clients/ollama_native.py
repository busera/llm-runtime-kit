"""Native Ollama `/api/chat` client for local or direct cloud API use."""

from __future__ import annotations

import json
from collections.abc import Callable
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_runtime_kit.config import ModelProfile, ProviderConfig
from llm_runtime_kit.credentials import CredentialResolver
from llm_runtime_kit.types import LLMRequest, LLMResponse

Urlopen = Callable[..., object]


class OllamaNativeClient:
    """HTTP client for Ollama's native `/api/chat` endpoint."""

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
        """Execute one native Ollama chat request."""

        model = request.model or profile.model
        max_context_tokens = (
            request.max_context_tokens
            if request.max_context_tokens is not None
            else profile.max_context_tokens
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "options": {
                "temperature": request.temperature
                if request.temperature is not None
                else profile.temperature,
                "num_predict": request.max_tokens
                if request.max_tokens is not None
                else profile.max_tokens,
            },
            "stream": False,
        }
        if max_context_tokens is not None:
            options = payload["options"]
            if isinstance(options, dict):
                options["num_ctx"] = max_context_tokens
        if profile.think is not None:
            payload["think"] = profile.think
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = (
            self._credential_resolver.get(provider.api_key_name)
            if provider.api_key_name
            else ""
        )
        if provider.api_key_name and not api_key:
            return LLMResponse(
                False,
                "",
                provider_name,
                model,
                error=f"{provider.api_key_name} not set",
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(
            f"{provider.base_url}/chat",
            data=json.dumps(payload).encode("utf-8"),
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
        try:
            parsed = json.loads(raw)
        except JSONDecodeError as error:
            return LLMResponse(False, "", provider_name, model, error=str(error))
        message = parsed.get("message", {}) or {}
        text = message.get("content") or message.get("thinking") or ""
        prompt_tokens = int(parsed.get("prompt_eval_count") or 0)
        completion_tokens = int(parsed.get("eval_count") or 0)
        return LLMResponse(
            bool(str(text).strip()),
            str(text),
            provider_name,
            model,
            tokens_used=prompt_tokens + completion_tokens,
            error=None if str(text).strip() else "empty response",
        )
