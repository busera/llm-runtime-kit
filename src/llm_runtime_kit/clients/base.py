"""Provider client protocol."""

from __future__ import annotations

from typing import Protocol

from llm_runtime_kit.config import ModelProfile, ProviderConfig
from llm_runtime_kit.types import LLMRequest, LLMResponse


class ProviderClient(Protocol):
    """Protocol implemented by LLM provider clients."""

    def complete(
        self,
        request: LLMRequest,
        provider_name: str,
        provider: ProviderConfig,
        profile: ModelProfile,
    ) -> LLMResponse:
        """Execute one completion request."""
