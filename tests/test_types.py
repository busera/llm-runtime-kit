import pytest

from llm_runtime_kit.types import LLMRequest


def test_llm_request_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LLMRequest(prompt="p", max_tokens=0)


def test_llm_request_rejects_invalid_max_context_tokens() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        LLMRequest(prompt="p", max_context_tokens=0)


def test_llm_request_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LLMRequest(prompt="p", temperature=2.1)


def test_llm_request_preserves_positional_metadata_compatibility() -> None:
    request = LLMRequest("p", "s", None, None, None, None, {"trace": "1"})

    assert request.metadata == {"trace": "1"}
    assert request.max_context_tokens is None
