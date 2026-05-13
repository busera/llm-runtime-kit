from pydantic import BaseModel

from llm_runtime_kit.config import RuntimeConfig
from llm_runtime_kit.router import LLMRouter
from llm_runtime_kit.types import LLMRequest, LLMResponse


class Questions(BaseModel):
    objective: str
    questions: list[str]


class SequenceClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls = 0
        self.requests = []

    def complete(self, request, provider_name, provider, profile):
        self.requests.append(request)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def base_config(allow_cloud: bool = False) -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "default_profile": "cloud",
            "allow_cloud": allow_cloud,
            "retry": {"max_attempts": 2, "base_delay_seconds": 0, "jitter_seconds": 0},
            "providers": {
                "ollama_cloud": {
                    "kind": "ollama",
                    "api_style": "openai_compatible",
                    "base_url": "http://localhost:11434",
                    "require_loopback": True,
                },
                "ollama_cloud_api": {
                    "kind": "ollama",
                    "api_style": "native",
                    "base_url": "https://ollama.com/api",
                    "api_key_name": "OLLAMA_API_KEY",
                    "require_loopback": False,
                },
                "ollama_local": {
                    "kind": "ollama",
                    "api_style": "openai_compatible",
                    "base_url": "http://localhost:11434",
                    "require_loopback": True,
                },
                "openai": {
                    "kind": "openai",
                    "api_style": "openai_compatible",
                    "base_url": "https://api.openai.com",
                    "api_key_name": "OPENAI_API_KEY",
                },
                "openrouter": {
                    "kind": "openrouter",
                    "api_style": "openai_compatible",
                    "base_url": "https://openrouter.ai/api",
                    "api_key_name": "OPENROUTER_API_KEY",
                },
                "anthropic": {
                    "kind": "anthropic",
                    "api_style": "native",
                    "base_url": "https://api.anthropic.com",
                    "api_key_name": "ANTHROPIC_API_KEY",
                },
            },
            "profiles": {
                "cloud": {
                    "provider": "ollama_cloud",
                    "model": "deepseek-v4-flash:cloud",
                },
                "cloud_direct": {
                    "provider": "ollama_cloud_api",
                    "model": "gpt-oss:120b",
                },
                "local": {
                    "provider": "ollama_local",
                    "model": "qwen3.6:35b-a3b-coding-mxfp8",
                },
                "openai": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                },
                "openrouter": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5.2",
                },
                "anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-latest",
                },
            },
            "fallbacks": {"cloud": ["cloud", "local"]},
        }
    )


def test_router_executes_default_route_alias_chain() -> None:
    raw = base_config(allow_cloud=True).model_dump()
    raw["default_profile"] = "route_alias"
    raw["fallbacks"]["route_alias"] = ["local"]
    config = RuntimeConfig.model_validate(raw)
    client = SequenceClient([LLMResponse(True, "alias ok", "ollama_local", "qwen")])
    router = LLMRouter(config, clients={"openai_compatible": client})

    response = router.complete(LLMRequest(prompt="hello"))

    assert response.success is True
    assert response.text == "alias ok"
    assert client.calls == 1


def test_router_blocks_cloud_then_uses_local_fallback() -> None:
    client = SequenceClient([LLMResponse(True, "local ok", "ollama_local", "qwen")])
    router = LLMRouter(
        base_config(allow_cloud=False), clients={"openai_compatible": client}
    )

    response = router.complete(LLMRequest(prompt="hello"))

    assert response.success is True
    assert response.text == "local ok"
    assert response.fallback_used is True
    assert client.calls == 1


def test_router_blocks_api_key_provider_when_cloud_disabled() -> None:
    client = SequenceClient([LLMResponse(True, "remote ok", "openai", "gpt")])
    router = LLMRouter(
        base_config(allow_cloud=False), clients={"openai_compatible": client}
    )

    response = router.complete(LLMRequest(prompt="hello", profile="openai"))

    assert response.success is False
    assert response.error == "cloud profile blocked by allow_cloud=false"
    assert client.calls == 0


def test_router_blocks_openrouter_provider_when_cloud_disabled() -> None:
    client = SequenceClient(
        [LLMResponse(True, "remote ok", "openrouter", "openai/gpt-5.2")]
    )
    router = LLMRouter(
        base_config(allow_cloud=False), clients={"openai_compatible": client}
    )

    response = router.complete(LLMRequest(prompt="hello", profile="openrouter"))

    assert response.success is False
    assert response.error == "cloud profile blocked by allow_cloud=false"
    assert client.calls == 0


def test_router_blocks_model_override_cloud_tag_when_cloud_disabled() -> None:
    client = SequenceClient([LLMResponse(True, "remote ok", "ollama_local", "gpt")])
    router = LLMRouter(
        base_config(allow_cloud=False), clients={"openai_compatible": client}
    )

    response = router.complete(
        LLMRequest(prompt="hello", profile="local", model="gpt-oss:120b-cloud")
    )

    assert response.success is False
    assert response.error == "cloud profile blocked by allow_cloud=false"
    assert client.calls == 0


def test_router_blocks_anthropic_provider_when_cloud_disabled() -> None:
    client = SequenceClient([LLMResponse(True, "remote ok", "anthropic", "claude")])
    router = LLMRouter(
        base_config(allow_cloud=False),
        clients={"openai_compatible": client, "anthropic": client},
    )

    response = router.complete(LLMRequest(prompt="hello", profile="anthropic"))

    assert response.success is False
    assert response.error == "cloud profile blocked by allow_cloud=false"
    assert client.calls == 0


def test_router_retries_transient_response_before_fallback_success() -> None:
    client = SequenceClient(
        [
            LLMResponse(
                False, "", "ollama_cloud", "deepseek", status_code=503, error="HTTP 503"
            ),
            LLMResponse(True, "cloud ok", "ollama_cloud", "deepseek"),
        ]
    )
    router = LLMRouter(
        base_config(allow_cloud=True),
        clients={"openai_compatible": client},
        sleep=lambda _delay: None,
    )

    response = router.complete(LLMRequest(prompt="hello"))

    assert response.success is True
    assert response.text == "cloud ok"
    assert response.attempts == 2
    assert response.fallback_used is False


def test_router_applies_configured_jitter_to_retry_sleep() -> None:
    raw = base_config(allow_cloud=True).model_dump()
    raw["retry"] = {
        "max_attempts": 2,
        "base_delay_seconds": 1.0,
        "max_delay_seconds": 20.0,
        "jitter_seconds": 0.25,
    }
    config = RuntimeConfig.model_validate(raw)
    sleeps: list[float] = []
    client = SequenceClient(
        [
            LLMResponse(
                False, "", "ollama_cloud", "deepseek", status_code=503, error="HTTP 503"
            ),
            LLMResponse(True, "cloud ok", "ollama_cloud", "deepseek"),
        ]
    )
    router = LLMRouter(
        config,
        clients={"openai_compatible": client},
        sleep=sleeps.append,
        jitter=lambda start, stop: stop,
    )

    response = router.complete(LLMRequest(prompt="hello"))

    assert response.success is True
    assert response.attempts == 2
    assert sleeps == [1.25]


def test_router_preserves_positional_credential_resolver_compatibility() -> None:
    class StaticResolver:
        def get(self, name: str) -> str | None:
            return f"resolved-{name}"

    resolver = StaticResolver()
    router = LLMRouter(
        base_config(allow_cloud=True),
        {"openai_compatible": SequenceClient([])},
        lambda _delay: None,
        resolver,
    )

    assert router.credential_resolver is resolver


def test_router_uses_native_ollama_client_for_direct_cloud_api() -> None:
    client = SequenceClient(
        [LLMResponse(True, "native ok", "ollama_cloud_api", "gpt-oss:120b")]
    )
    router = LLMRouter(
        base_config(allow_cloud=True),
        clients={"ollama_native": client},
    )

    response = router.complete(LLMRequest(prompt="hello", profile="cloud_direct"))

    assert response.success is True
    assert response.text == "native ok"
    assert client.calls == 1


def test_router_without_output_contract_preserves_plain_text_response() -> None:
    client = SequenceClient([LLMResponse(True, "plain", "ollama_local", "llama")])
    router = LLMRouter(
        base_config(allow_cloud=False),
        clients={"openai_compatible": client},
    )

    response = router.complete(LLMRequest(prompt="plain", profile="local"))

    assert response.success is True
    assert response.text == "plain"
    assert response.parsed is None
    assert response.validation_error is None
    assert response.raw_text is None


def test_router_validates_request_pydantic_output_model() -> None:
    client = SequenceClient(
        [
            LLMResponse(
                True,
                '```json\n{"objective":"o","questions":["q"]}\n```',
                "ollama_local",
                "llama",
            )
        ]
    )
    config = base_config(allow_cloud=False)
    router = LLMRouter(config, clients={"openai_compatible": client})

    response = router.complete(
        LLMRequest(
            prompt="return json",
            profile="local",
            output_model=Questions,
        )
    )

    assert response.success is True
    assert response.parsed == Questions(objective="o", questions=["q"])
    assert response.text == '{"objective":"o","questions":["q"]}'
    assert response.raw_text == '```json\n{"objective":"o","questions":["q"]}\n```'


def test_router_retries_validation_failure_then_falls_back() -> None:
    first = SequenceClient(
        [
            LLMResponse(True, "not json", "ollama_local", "llama"),
            LLMResponse(True, "still not json", "ollama_local", "llama"),
        ]
    )
    fallback = SequenceClient(
        [LLMResponse(True, '{"ok": true}', "ollama_local", "llama")]
    )
    raw = base_config(allow_cloud=False).model_dump()
    raw["retry"] = {
        "max_attempts": 2,
        "base_delay_seconds": 0,
        "max_delay_seconds": 0,
        "jitter_seconds": 0,
        "retryable_statuses": [429, 500],
    }
    raw["output"] = {
        "mode": "json",
        "require_valid_json": True,
        "repair_json": False,
        "validation_failure_is_retryable": True,
    }
    raw["fallbacks"] = {"local": ["local", "secondary"]}
    raw["profiles"]["secondary"] = raw["profiles"]["local"] | {"model": "backup"}
    config = RuntimeConfig.model_validate(raw)
    router = LLMRouter(
        config,
        clients={"openai_compatible": first},
        sleep=lambda delay: None,
    )

    def pick_client(request, provider_name, provider, profile):
        if profile.model == "backup":
            return fallback.complete(request, provider_name, provider, profile)
        return first.complete(request, provider_name, provider, profile)

    router.clients["openai_compatible"] = type(
        "DispatchClient",
        (),
        {"complete": staticmethod(pick_client)},
    )()

    response = router.complete(LLMRequest(prompt="return json", profile="local"))

    assert response.success is True
    assert response.parsed == {"ok": True}
    assert response.fallback_used is True
    assert response.attempts == 1
    assert first.calls == 2
    assert fallback.calls == 1


def test_router_does_not_include_validation_error_in_retry_prompt_by_default() -> None:
    client = SequenceClient(
        [
            LLMResponse(True, '{"objective":123,"questions":["q"]}', "local", "llama"),
            LLMResponse(True, '{"objective":"o","questions":["q"]}', "local", "llama"),
        ]
    )
    raw = base_config(allow_cloud=False).model_dump()
    raw["retry"] = {
        "max_attempts": 2,
        "base_delay_seconds": 0,
        "max_delay_seconds": 0,
        "jitter_seconds": 0,
    }
    config = RuntimeConfig.model_validate(raw)
    router = LLMRouter(config, clients={"openai_compatible": client})

    response = router.complete(
        LLMRequest(prompt="return json", profile="local", output_model=Questions)
    )

    assert response.success is True
    assert len(client.requests) == 2
    assert client.requests[1].prompt == "return json"


def test_router_retry_prompt_uses_sanitized_validation_error_when_enabled() -> None:
    client = SequenceClient(
        [
            LLMResponse(True, '{"objective":123,"questions":["q"]}', "local", "llama"),
            LLMResponse(True, '{"objective":"o","questions":["q"]}', "local", "llama"),
        ]
    )
    raw = base_config(allow_cloud=False).model_dump()
    raw["retry"] = {
        "max_attempts": 2,
        "base_delay_seconds": 0,
        "max_delay_seconds": 0,
        "jitter_seconds": 0,
    }
    raw["output"] = {"include_validation_error_in_retry_prompt": True}
    config = RuntimeConfig.model_validate(raw)
    router = LLMRouter(config, clients={"openai_compatible": client})

    response = router.complete(
        LLMRequest(prompt="return json", profile="local", output_model=Questions)
    )

    assert response.success is True
    retry_prompt = client.requests[1].prompt
    assert "Validation error:" in retry_prompt
    assert "objective" in retry_prompt
    assert "input_value" not in retry_prompt
    assert "123" not in retry_prompt


def test_router_returns_validation_failure_when_chain_exhausted() -> None:
    client = SequenceClient([LLMResponse(True, "not json", "ollama_local", "llama")])
    raw = base_config(allow_cloud=False).model_dump()
    raw["output"] = {
        "mode": "json",
        "require_valid_json": True,
        "repair_json": False,
        "validation_failure_is_retryable": False,
    }
    config = RuntimeConfig.model_validate(raw)
    router = LLMRouter(config, clients={"openai_compatible": client})

    response = router.complete(LLMRequest(prompt="return json", profile="local"))

    assert response.success is False
    assert response.text == "not json"
    assert response.raw_text == "not json"
    assert response.parsed is None
    assert "No JSON object or array found" in str(response.validation_error)
    assert response.error == response.validation_error


def test_router_output_request_override_does_not_change_old_positional_metadata() -> (
    None
):
    request = LLMRequest("prompt", "system", "local", None, None, None, {"id": "1"})

    assert request.metadata == {"id": "1"}
    assert request.output_mode is None


def test_router_uses_openai_compatible_client_for_openrouter() -> None:
    client = SequenceClient(
        [LLMResponse(True, "openrouter ok", "openrouter", "openai/gpt-5.2")]
    )
    router = LLMRouter(
        base_config(allow_cloud=True),
        clients={"openai_compatible": client},
    )

    response = router.complete(LLMRequest(prompt="hello", profile="openrouter"))

    assert response.success is True
    assert response.text == "openrouter ok"
    assert client.calls == 1
