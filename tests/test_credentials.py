from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_runtime_kit.config import ProviderConfig, RuntimeConfig, load_config
from llm_runtime_kit.credentials import CredentialConfig, CredentialResolver


def test_credential_resolver_prefers_process_env_over_dotenv_and_keyring(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=dkey\n", encoding="utf-8")
    resolver = CredentialResolver(
        CredentialConfig(dotenv_path=dotenv_path),
        environ={"OPENAI_API_KEY": "env-key"},
        keyring_getter=lambda _service, _name: "keyring-key",
    )

    assert resolver.get("OPENAI_API_KEY") == "env-key"


def test_credential_resolver_reads_project_dotenv_when_env_missing(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=dkey\n", encoding="utf-8")
    resolver = CredentialResolver(
        CredentialConfig(dotenv_path=dotenv_path),
        environ={},
        keyring_getter=lambda _service, _name: "keyring-key",
    )

    assert resolver.get("OPENAI_API_KEY") == "dkey"


def test_credential_resolver_uses_keyring_after_env_and_dotenv_miss() -> None:
    calls: list[tuple[str, str]] = []

    def keyring_getter(service: str, name: str) -> str | None:
        calls.append((service, name))
        return "keyring-key"

    resolver = CredentialResolver(
        CredentialConfig(dotenv_path=None, keyring_service="llm-runtime-kit"),
        environ={},
        keyring_getter=keyring_getter,
    )

    assert resolver.get("OPENAI_API_KEY") == "keyring-key"
    assert calls == [("llm-runtime-kit", "OPENAI_API_KEY")]


def test_credential_resolver_returns_none_without_optional_keyring_package() -> None:
    resolver = CredentialResolver(
        CredentialConfig(sources=["keyring"]),
        environ={},
        keyring_getter=lambda _service, _name: None,
    )

    assert resolver.get("OPENAI_API_KEY") is None


def test_credential_sources_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        CredentialConfig(sources=["env", "env"])


def test_provider_config_accepts_deprecated_api_key_env_alias() -> None:
    provider = ProviderConfig(
        kind="openai",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
    )

    assert provider.api_key_name == "OPENAI_API_KEY"
    assert provider.api_key_env == "OPENAI_API_KEY"


def test_remote_provider_requires_api_key_name() -> None:
    with pytest.raises(ValueError, match="requires api_key_name"):
        ProviderConfig(kind="openai", base_url="https://api.openai.com")


def test_load_config_resolves_relative_dotenv_path_from_config_file() -> None:
    config = load_config(Path(__file__).parents[1] / "config_llm.yaml")

    assert config.credentials.dotenv_path == Path(__file__).parents[1] / ".env"


def test_runtime_config_missing_cloud_key_does_not_break_local_config() -> None:
    raw = {
        "default_profile": "local",
        "allow_cloud": False,
        "credentials": {"sources": ["env"], "dotenv_path": None},
        "providers": {
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
        },
        "profiles": {
            "local": {"provider": "ollama_local", "model": "qwen"},
            "openai": {"provider": "openai", "model": "gpt-5.5"},
        },
        "fallbacks": {"local": ["local"]},
    }

    config = RuntimeConfig.model_validate(raw)

    assert config.providers["openai"].api_key_name == "OPENAI_API_KEY"
