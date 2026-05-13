import json
from urllib.error import HTTPError

from llm_runtime_kit.clients.anthropic import AnthropicClient
from llm_runtime_kit.clients.ollama_native import OllamaNativeClient
from llm_runtime_kit.clients.openai_compatible import OpenAICompatibleClient
from llm_runtime_kit.config import ModelProfile, ProviderConfig
from llm_runtime_kit.credentials import CredentialConfig, CredentialResolver
from llm_runtime_kit.types import LLMRequest


class FakeResponse:
    def __init__(self, payload: dict | str) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_uses_reasoning_fallback_for_empty_content() -> None:
    def opener(req, timeout):
        return FakeResponse(
            {"choices": [{"message": {"content": "", "reasoning": "answer"}}]}
        )

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_local",
        ProviderConfig(
            kind="ollama", base_url="http://localhost:11434", require_loopback=True
        ),
        ModelProfile(provider="ollama_local", model="qwen", think=False),
    )

    assert response.success is True
    assert response.text == "answer"


def test_openai_compatible_client_reports_http_status(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def opener(req, timeout):
        raise HTTPError(req.full_url, 429, "rate", {}, None)

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "openai",
        ProviderConfig(
            kind="openai",
            base_url="https://api.openai.com",
            api_key_name="OPENAI_API_KEY",
        ),
        ModelProfile(provider="openai", model="gpt-5.5"),
    )

    assert response.success is False
    assert response.status_code == 429


def test_openai_compatible_client_appends_chat_completions_path(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "openai",
        ProviderConfig(
            kind="openai",
            base_url="https://api.openai.com",
            api_key_name="OPENAI_API_KEY",
        ),
        ModelProfile(provider="openai", model="gpt-5.5"),
    )

    assert response.success is True
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["payload"]["messages"][0]["role"] == "developer"
    assert seen["payload"]["max_completion_tokens"] == 4096
    assert "max_tokens" not in seen["payload"]


def test_openai_compatible_client_reads_key_from_dotenv(tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=dkey\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["headers"] = dict(req.header_items())
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    resolver = CredentialResolver(
        CredentialConfig(sources=["dotenv"], dotenv_path=dotenv_path),
        environ={},
    )
    client = OpenAICompatibleClient(opener=opener, credential_resolver=resolver)
    response = client.complete(
        LLMRequest(prompt="p"),
        "openai",
        ProviderConfig(
            kind="openai",
            base_url="https://api.openai.com",
            api_key_name="OPENAI_API_KEY",
        ),
        ModelProfile(provider="openai", model="gpt-5.5"),
    )

    assert response.success is True
    assert seen["headers"]["Authorization"] == "Bearer dkey"


def test_openrouter_client_uses_openai_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "openrouter",
        ProviderConfig(
            kind="openrouter",
            base_url="https://openrouter.ai/api",
            api_key_name="OPENROUTER_API_KEY",
        ),
        ModelProfile(provider="openrouter", model="openai/gpt-5.2"),
    )

    assert response.success is True
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["payload"]["model"] == "openai/gpt-5.2"
    assert seen["payload"]["messages"][0]["role"] == "developer"
    assert seen["payload"]["max_completion_tokens"] == 4096
    assert "max_tokens" not in seen["payload"]


def test_openrouter_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def opener(req, timeout):
        raise AssertionError("OpenRouter request should fail before HTTP call")

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "openrouter",
        ProviderConfig(
            kind="openrouter",
            base_url="https://openrouter.ai/api",
            api_key_name="OPENROUTER_API_KEY",
        ),
        ModelProfile(provider="openrouter", model="openai/gpt-5.2"),
    )

    assert response.success is False
    assert response.error == "OPENROUTER_API_KEY not set"


def test_ollama_openai_compatible_client_keeps_system_and_max_tokens() -> None:
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_local",
        ProviderConfig(
            kind="ollama", base_url="http://localhost:11434", require_loopback=True
        ),
        ModelProfile(provider="ollama_local", model="qwen", think=False),
    )

    assert response.success is True
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert seen["payload"]["max_tokens"] == 4096
    assert "max_completion_tokens" not in seen["payload"]


def test_ollama_native_client_posts_to_chat_and_parses_thinking(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["headers"] = dict(req.header_items())
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse(
            {
                "message": {"content": "", "thinking": "native answer"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )

    client = OllamaNativeClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(provider="ollama_cloud_api", model="gpt-oss:120b", think=False),
    )

    assert response.success is True
    assert response.text == "native answer"
    assert response.tokens_used == 15
    assert seen["url"] == "https://ollama.com/api/chat"
    assert seen["timeout"] == 120
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["think"] is False
    assert seen["payload"]["options"]["num_predict"] == 4096
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_ollama_native_client_sends_max_context_tokens(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "ok"}})

    client = OllamaNativeClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p", max_context_tokens=64000),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(
            provider="ollama_cloud_api",
            model="gpt-oss:120b",
            max_context_tokens=128000,
        ),
    )

    assert response.success is True
    assert seen["payload"]["options"]["num_ctx"] == 64000


def test_ollama_native_client_uses_profile_max_context_tokens(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "ok"}})

    client = OllamaNativeClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(
            provider="ollama_cloud_api",
            model="gpt-oss:120b",
            max_context_tokens=128000,
        ),
    )

    assert response.success is True
    assert seen["payload"]["options"]["num_ctx"] == 128000


def test_ollama_native_client_omits_max_context_tokens_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    seen: dict[str, object] = {}

    def opener(req, timeout):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "ok"}})

    client = OllamaNativeClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(provider="ollama_cloud_api", model="gpt-oss:120b"),
    )

    assert response.success is True
    assert "num_ctx" not in seen["payload"]["options"]


def test_ollama_native_client_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = OllamaNativeClient()

    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(provider="ollama_cloud_api", model="gpt-oss:120b"),
    )

    assert response.success is False
    assert response.error == "OLLAMA_API_KEY not set"


def test_ollama_native_client_reports_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

    def opener(req, timeout):
        return FakeResponse("")

    client = OllamaNativeClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "ollama_cloud_api",
        ProviderConfig(
            kind="ollama",
            api_style="native",
            base_url="https://ollama.com/api",
            api_key_name="OLLAMA_API_KEY",
        ),
        ModelProfile(provider="ollama_cloud_api", model="gpt-oss:120b"),
    )

    assert response.success is False
    assert "Expecting value" in str(response.error)


def test_anthropic_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()

    response = client.complete(
        LLMRequest(prompt="p"),
        "anthropic",
        ProviderConfig(
            kind="anthropic",
            base_url="https://api.anthropic.com",
            api_key_name="ANTHROPIC_API_KEY",
        ),
        ModelProfile(provider="anthropic", model="claude-sonnet-4-5"),
    )

    assert response.success is False
    assert response.error == "ANTHROPIC_API_KEY not set"


def test_anthropic_client_appends_messages_path(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen: dict[str, str] = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        return FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    client = AnthropicClient(opener=opener)
    response = client.complete(
        LLMRequest(prompt="p"),
        "anthropic",
        ProviderConfig(
            kind="anthropic",
            base_url="https://api.anthropic.com",
            api_key_name="ANTHROPIC_API_KEY",
        ),
        ModelProfile(provider="anthropic", model="claude-sonnet-4-5"),
    )

    assert response.success is True
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
