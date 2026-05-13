"""Provider routing and fallback orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from llm_runtime_kit.clients.anthropic import AnthropicClient
from llm_runtime_kit.clients.base import ProviderClient
from llm_runtime_kit.clients.ollama_native import OllamaNativeClient
from llm_runtime_kit.clients.openai_compatible import OpenAICompatibleClient
from llm_runtime_kit.config import (
    ModelProfile,
    OutputConfig,
    ProviderConfig,
    RuntimeConfig,
    is_cloud_model_tag,
)
from llm_runtime_kit.credentials import CredentialResolver
from llm_runtime_kit.output import validate_output
from llm_runtime_kit.retry import calculate_retry_delay
from llm_runtime_kit.types import LLMRequest, LLMResponse


class LLMRouter:
    """Route LLM requests through configured profiles and fallbacks."""

    def __init__(
        self,
        config: RuntimeConfig,
        clients: dict[str, ProviderClient] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        credential_resolver: CredentialResolver | None = None,
        *,
        jitter: Callable[[float, float], float] | None = None,
    ) -> None:
        self.config = config
        self.credential_resolver = credential_resolver or CredentialResolver(
            config.credentials
        )
        self.clients = clients or {
            "openai_compatible": OpenAICompatibleClient(
                credential_resolver=self.credential_resolver
            ),
            "ollama_native": OllamaNativeClient(
                credential_resolver=self.credential_resolver
            ),
            "anthropic": AnthropicClient(credential_resolver=self.credential_resolver),
        }
        self._sleep = sleep
        self._jitter = jitter

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run the request through the profile's fallback chain."""

        profile_name = request.profile or self.config.default_profile
        chain = self.config.fallbacks.get(profile_name, [profile_name])
        if not chain:
            chain = [profile_name]
        last_response: LLMResponse | None = None
        for index, candidate in enumerate(chain):
            profile = self.config.profiles[candidate]
            provider = self.config.providers[profile.provider]
            blocks_cloud = not self.config.allow_cloud and self._uses_cloud(
                candidate, request.model
            )
            if blocks_cloud:
                last_response = LLMResponse(
                    False,
                    "",
                    profile.provider,
                    profile.model,
                    error="cloud profile blocked by allow_cloud=false",
                    fallback_used=index > 0,
                )
                continue
            client = self.clients[provider.client_key]
            response = self._complete_with_retry(
                client, request, profile.provider, provider, profile
            )
            response = LLMResponse(
                response.success,
                response.text,
                response.provider,
                response.model,
                tokens_used=response.tokens_used,
                error=response.error,
                status_code=response.status_code,
                attempts=response.attempts,
                fallback_used=index > 0,
                parsed=response.parsed,
                validation_error=response.validation_error,
                raw_text=response.raw_text,
            )
            if response.success:
                return response
            last_response = response
        if last_response is not None:
            return last_response
        return LLMResponse(False, "", "", "", error="no fallback candidates executed")

    def _uses_cloud(self, profile_name: str, model_override: str | None = None) -> bool:
        """Return whether a profile would send data to non-local infrastructure."""

        profile = self.config.profiles[profile_name]
        provider = self.config.providers[profile.provider]
        effective_model = model_override or profile.model
        if is_cloud_model_tag(effective_model):
            return True
        return not (provider.kind == "ollama" and provider.require_loopback)

    def _complete_with_retry(
        self,
        client: ProviderClient,
        request: LLMRequest,
        provider_name: str,
        provider: ProviderConfig,
        profile: ModelProfile,
    ) -> LLMResponse:
        """Execute one candidate profile with bounded transient-response retries."""

        attempt = 1
        active_request = request
        while True:
            response = client.complete(active_request, provider_name, provider, profile)
            if response.success:
                response = self._apply_output_contract(response, active_request)
            if response.success or not self._should_retry_response(
                response, active_request
            ):
                return LLMResponse(
                    response.success,
                    response.text,
                    response.provider,
                    response.model,
                    tokens_used=response.tokens_used,
                    error=response.error,
                    status_code=response.status_code,
                    attempts=attempt,
                    fallback_used=response.fallback_used,
                    parsed=response.parsed,
                    validation_error=response.validation_error,
                    raw_text=response.raw_text,
                )
            if attempt >= self.config.retry.max_attempts:
                return LLMResponse(
                    response.success,
                    response.text,
                    response.provider,
                    response.model,
                    tokens_used=response.tokens_used,
                    error=response.error,
                    status_code=response.status_code,
                    attempts=attempt,
                    fallback_used=response.fallback_used,
                    parsed=response.parsed,
                    validation_error=response.validation_error,
                    raw_text=response.raw_text,
                )
            if response.validation_error:
                active_request = self._request_for_validation_retry(
                    active_request,
                    response.validation_error,
                )
            delay = calculate_retry_delay(
                self.config.retry,
                attempt,
                self._jitter,
            )
            self._sleep(delay)
            attempt += 1

    def _should_retry_response(
        self, response: LLMResponse, request: LLMRequest
    ) -> bool:
        """Return whether a provider response should be retried."""

        if response.validation_error:
            return self._output_config_for(request).validation_failure_is_retryable
        if response.status_code in self.config.retry.retryable_statuses:
            return True
        error = (response.error or "").lower()
        return (
            "timeout" in error
            or "timed out" in error
            or "temporarily unavailable" in error
        )

    def _apply_output_contract(
        self,
        response: LLMResponse,
        request: LLMRequest,
    ) -> LLMResponse:
        """Apply optional generic output validation to a successful response."""

        output_config = self._output_config_for(request)
        if not self._has_output_contract(request, output_config):
            return response
        result = validate_output(
            response.text,
            output_config,
            output_model=request.output_model,
            validator=request.output_validator,
        )
        if result.success:
            return LLMResponse(
                True,
                result.normalized_text,
                response.provider,
                response.model,
                tokens_used=response.tokens_used,
                status_code=response.status_code,
                attempts=response.attempts,
                fallback_used=response.fallback_used,
                parsed=result.parsed,
                raw_text=response.text,
            )
        return LLMResponse(
            False,
            response.text,
            response.provider,
            response.model,
            tokens_used=response.tokens_used,
            error=result.validation_error,
            status_code=response.status_code,
            attempts=response.attempts,
            fallback_used=response.fallback_used,
            parsed=None,
            validation_error=result.validation_error,
            raw_text=response.text,
        )

    def _output_config_for(self, request: LLMRequest) -> OutputConfig:
        """Merge runtime output defaults with per-request overrides."""

        updates = {}
        mapping = {
            "output_mode": "mode",
            "require_valid_json": "require_valid_json",
            "repair_json": "repair_json",
            "max_repair_attempts": "max_repair_attempts",
            "validation_failure_is_retryable": "validation_failure_is_retryable",
            "include_validation_error_in_retry_prompt": (
                "include_validation_error_in_retry_prompt"
            ),
        }
        for request_field, config_field in mapping.items():
            value = getattr(request, request_field)
            if value is not None:
                updates[config_field] = value
        return OutputConfig.model_validate(self.config.output.model_dump() | updates)

    def _has_output_contract(
        self,
        request: LLMRequest,
        output_config: OutputConfig,
    ) -> bool:
        """Return whether this call should validate structured output."""

        return (
            output_config.mode == "json"
            or output_config.require_valid_json
            or request.output_model is not None
            or request.output_validator is not None
        )

    def _request_for_validation_retry(
        self,
        request: LLMRequest,
        validation_error: str,
    ) -> LLMRequest:
        """Optionally add validation feedback to the retry prompt."""

        output_config = self._output_config_for(request)
        if not output_config.include_validation_error_in_retry_prompt:
            return request
        feedback = (
            "\n\nThe previous response failed output validation. "
            "Return only output matching the requested contract. "
            f"Validation error: {validation_error}"
        )
        return replace(request, prompt=f"{request.prompt}{feedback}")
