from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_runtime_kit.config import (
    RuntimeConfig,
    is_cloud_model_tag,
    load_config,
    normalize_ollama_base_url,
)


def minimal_config() -> dict:
    return {
        "default_profile": "local",
        "allow_cloud": False,
        "providers": {
            "ollama_local": {
                "kind": "ollama",
                "api_style": "openai_compatible",
                "base_url": "http://localhost:11434",
                "timeout_seconds": 30,
                "require_loopback": True,
            }
        },
        "profiles": {
            "local": {
                "provider": "ollama_local",
                "model": "qwen3.6:35b-a3b-coding-mxfp8",
                "max_tokens": 1024,
                "max_context_tokens": 32000,
            }
        },
        "fallbacks": {"local": ["local"]},
    }


def test_load_example_config_validates() -> None:
    config = load_config(Path(__file__).parents[1] / "config_llm.yaml")

    assert config.default_profile == "ollama_default"
    assert config.default_profile not in config.profiles
    assert config.providers["ollama_local"].base_url == "http://localhost:11434"
    assert config.providers["ollama_local"].api_style == "openai_compatible"
    assert config.providers["ollama_cloud_api"].base_url == "https://ollama.com/api"
    assert config.providers["ollama_cloud_api"].api_style == "native"
    assert config.providers["openrouter"].base_url == "https://openrouter.ai/api"
    assert config.providers["openrouter"].api_style == "openai_compatible"
    assert config.output.mode == "text"
    assert config.output.repair_json is True
    assert config.output.validation_failure_is_retryable is True
    assert config.profiles["openrouter_default"].model == "openai/gpt-5.2"
    assert config.profiles["ollama_cloud_direct"].max_context_tokens == 128000
    assert config.profiles["ollama_local_default"].max_context_tokens == 128000
    assert config.fallbacks["ollama_default"] == [
        "ollama_cloud_direct",
        "ollama_local_cloud_proxy_default",
        "ollama_local_default",
        "openai_default",
        "openrouter_default",
        "anthropic_default",
    ]


def test_invalid_profile_max_context_tokens_rejected() -> None:
    raw = minimal_config()
    raw["profiles"]["local"]["max_context_tokens"] = 0

    with pytest.raises(ValidationError, match="max_context_tokens"):
        RuntimeConfig.model_validate(raw)


def test_default_profile_can_be_fallback_route_alias() -> None:
    raw = minimal_config()
    raw["default_profile"] = "route_alias"
    raw["fallbacks"] = {"route_alias": ["local"]}

    config = RuntimeConfig.model_validate(raw)

    assert config.default_profile == "route_alias"
    assert config.fallbacks["route_alias"] == ["local"]


def test_empty_fallback_route_alias_rejected() -> None:
    raw = minimal_config()
    raw["default_profile"] = "route_alias"
    raw["fallbacks"] = {"route_alias": []}

    with pytest.raises(ValueError, match="Fallback route alias has empty chain"):
        RuntimeConfig.model_validate(raw)


def test_unknown_default_profile_rejected() -> None:
    raw = minimal_config()
    raw["default_profile"] = "missing"

    with pytest.raises(ValueError, match="default_profile is unknown"):
        RuntimeConfig.model_validate(raw)


def test_normalize_ollama_base_url_rewrites_wildcard_to_loopback() -> None:
    assert normalize_ollama_base_url("http://0.0.0.0:11434") == "http://127.0.0.1:11434"


def test_cloud_tag_detection_handles_colon_and_dash_forms() -> None:
    assert is_cloud_model_tag("deepseek-v4-flash:cloud") is True
    assert is_cloud_model_tag("qwen3-coder-480b-cloud") is True
    assert is_cloud_model_tag("qwen3.6:35b-a3b-coding-mxfp8") is False


def test_unknown_provider_reference_rejected() -> None:
    raw = minimal_config()
    raw["profiles"]["local"]["provider"] = "missing"

    with pytest.raises(ValueError, match="unknown provider"):
        RuntimeConfig.model_validate(raw)


def test_fallbacks_can_target_cloud_model_tags() -> None:
    raw = minimal_config()
    raw["profiles"]["cloud"] = {
        "provider": "ollama_local",
        "model": "deepseek-v4-flash:cloud",
    }
    raw["fallbacks"] = {"local": ["local", "cloud"]}

    config = RuntimeConfig.model_validate(raw)

    assert config.fallbacks["local"] == ["local", "cloud"]


def test_unsupported_api_style_for_provider_rejected() -> None:
    raw = minimal_config()
    raw["providers"]["ollama_local"]["api_style"] = "anthropic"

    with pytest.raises(ValueError, match="does not support api_style"):
        RuntimeConfig.model_validate(raw)


def test_provider_client_key_reflects_api_style() -> None:
    raw = minimal_config()
    raw["providers"]["ollama_cloud_api"] = {
        "kind": "ollama",
        "api_style": "native",
        "base_url": "https://ollama.com/api",
        "api_key_name": "OLLAMA_API_KEY",
        "require_loopback": False,
    }
    raw["providers"]["openrouter"] = {
        "kind": "openrouter",
        "api_style": "openai_compatible",
        "base_url": "https://openrouter.ai/api",
        "api_key_name": "OPENROUTER_API_KEY",
        "require_loopback": False,
    }
    config = RuntimeConfig.model_validate(raw)

    assert config.providers["ollama_local"].client_key == "openai_compatible"
    assert config.providers["ollama_cloud_api"].client_key == "ollama_native"
    assert config.providers["openrouter"].client_key == "openai_compatible"


def test_output_config_defaults_are_text_mode() -> None:
    config = RuntimeConfig.model_validate(minimal_config())

    assert config.output.mode == "text"
    assert config.output.require_valid_json is False
    assert config.output.repair_json is True


def test_output_config_rejects_unknown_mode() -> None:
    raw = minimal_config()
    raw["output"] = {"mode": "xml"}

    with pytest.raises(ValueError, match="output.mode"):
        RuntimeConfig.model_validate(raw)


def test_extra_config_fields_rejected() -> None:
    raw = minimal_config()
    raw["unexpected"] = True

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(raw)
