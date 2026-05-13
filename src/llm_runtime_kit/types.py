"""Runtime request and response types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


def _validate_positive_int(name: str, value: int | None) -> None:
    """Validate optional positive integer request fields."""

    if value is not None and value < 1:
        raise ValueError(f"{name} must be greater than or equal to 1")


def _validate_temperature(value: float | None) -> None:
    """Validate optional request temperature."""

    if value is not None and not 0 <= value <= 2:
        raise ValueError("temperature must be between 0 and 2")


@dataclass(frozen=True)
class LLMRequest:
    """A provider-agnostic chat completion request."""

    prompt: str
    system_prompt: str = "You are a precise, evidence-led assistant."
    profile: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    max_context_tokens: int | None = None
    output_mode: str | None = None
    require_valid_json: bool | None = None
    repair_json: bool | None = None
    max_repair_attempts: int | None = None
    validation_failure_is_retryable: bool | None = None
    include_validation_error_in_retry_prompt: bool | None = None
    output_model: type[Any] | None = None
    output_validator: Callable[[object], object | None] | None = None

    def __post_init__(self) -> None:
        """Validate provider-agnostic request overrides."""

        _validate_temperature(self.temperature)
        _validate_positive_int("max_tokens", self.max_tokens)
        _validate_positive_int("max_context_tokens", self.max_context_tokens)
        _validate_positive_int("max_repair_attempts", self.max_repair_attempts)
        if self.output_mode is not None and self.output_mode not in {"text", "json"}:
            raise ValueError("output_mode must be text or json")


@dataclass(frozen=True)
class LLMResponse:
    """A provider-agnostic LLM response."""

    success: bool
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    error: str | None = None
    status_code: int | None = None
    attempts: int = 1
    fallback_used: bool = False
    parsed: object | None = None
    validation_error: str | None = None
    raw_text: str | None = None
