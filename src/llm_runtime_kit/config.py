"""External YAML configuration for LLM Runtime Kit."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from llm_runtime_kit.credentials import CredentialConfig

LOCAL_HOSTS = {"", "0.0.0.0", "::", "[::]"}
CLOUD_TAG_MARKERS = (":cloud", "-cloud")


def normalize_ollama_base_url(value: str) -> str:
    """Normalize unsafe wildcard Ollama bind addresses to loopback clients."""

    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    host = parsed.hostname or ""
    if host in LOCAL_HOSTS:
        netloc = "127.0.0.1"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    return value.rstrip("/")


def is_cloud_model_tag(model: str) -> bool:
    """Return True if a model tag clearly routes to cloud infrastructure."""

    lowered = model.lower()
    return any(marker in lowered for marker in CLOUD_TAG_MARKERS)


def _default_api_style(kind: str) -> str:
    """Return the default API style for a supported provider kind."""

    if kind in {"ollama", "openai", "openrouter"}:
        return "openai_compatible"
    if kind == "anthropic":
        return "native"
    raise ValueError(f"Unsupported provider kind: {kind}")


def _allowed_api_styles(kind: str) -> set[str]:
    """Return supported API styles for a provider kind."""

    if kind == "ollama":
        return {"openai_compatible", "native"}
    if kind in {"openai", "openrouter"}:
        return {"openai_compatible"}
    if kind == "anthropic":
        return {"native"}
    raise ValueError(f"Unsupported provider kind: {kind}")


class LoggingConfig(BaseModel):
    """Logging behavior for the runtime."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    log_prompts: bool = False
    log_responses: bool = False
    redact_keys: list[str] = Field(default_factory=list)


class RetryConfig(BaseModel):
    """Retry behavior for idempotent LLM calls."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_seconds: float = Field(default=1.0, ge=0)
    max_delay_seconds: float = Field(default=20.0, ge=0)
    jitter_seconds: float = Field(default=0.0, ge=0)
    retryable_statuses: set[int] = Field(
        default_factory=lambda: {408, 409, 429, 500, 502, 503, 504}
    )


class OutputConfig(BaseModel):
    """Generic structured-output policy defaults."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "text"
    require_valid_json: bool = False
    repair_json: bool = True
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    validation_failure_is_retryable: bool = True
    include_validation_error_in_retry_prompt: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """Validate supported output modes."""

        if value not in {"text", "json"}:
            raise ValueError("output.mode must be text or json")
        return value


class ProviderConfig(BaseModel):
    """Provider endpoint, API style, and credential source."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    api_style: str | None = None
    base_url: str
    api_key_name: str = Field(
        default="",
        validation_alias=AliasChoices("api_key_name", "api_key_env"),
    )
    timeout_seconds: int = Field(default=120, ge=1)
    require_loopback: bool = False

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        """Validate supported provider types."""

        if value not in {"ollama", "openai", "openrouter", "anthropic"}:
            raise ValueError(f"Unsupported provider kind: {value}")
        return value

    @model_validator(mode="after")
    def validate_api_style_and_normalize_url(self) -> ProviderConfig:
        """Validate API style and normalize endpoint URLs."""

        style = self.api_style or _default_api_style(self.kind)
        allowed_styles = _allowed_api_styles(self.kind)
        if style not in allowed_styles:
            msg = f"Provider kind {self.kind} does not support api_style {style}"
            raise ValueError(msg)
        object.__setattr__(self, "api_style", style)
        if self.kind == "ollama" and self.require_loopback:
            object.__setattr__(
                self, "base_url", normalize_ollama_base_url(self.base_url)
            )
        else:
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if self.kind in {"openai", "openrouter", "anthropic"} and not self.api_key_name:
            raise ValueError(f"Provider kind {self.kind} requires api_key_name")
        return self

    @property
    def api_key_env(self) -> str:
        """Deprecated alias for api_key_name."""

        return self.api_key_name

    @property
    def client_key(self) -> str:
        """Return the router client key for this provider/API style."""

        if self.kind == "ollama" and self.api_style == "native":
            return "ollama_native"
        if self.kind == "anthropic":
            return "anthropic"
        return "openai_compatible"


class ModelProfile(BaseModel):
    """Named model profile used by application code."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    max_context_tokens: int | None = Field(default=None, ge=1)
    think: bool | None = None


class RuntimeConfig(BaseModel):
    """Validated runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    default_profile: str
    allow_cloud: bool = False
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    credentials: CredentialConfig = Field(default_factory=CredentialConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    providers: dict[str, ProviderConfig]
    profiles: dict[str, ModelProfile]
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references_and_cloud_policy(self) -> RuntimeConfig:
        """Validate profile/provider references and fallback chains."""

        if (
            self.default_profile not in self.profiles
            and self.default_profile not in self.fallbacks
        ):
            raise ValueError(f"default_profile is unknown: {self.default_profile}")
        for profile_name, profile in self.profiles.items():
            if profile.provider not in self.providers:
                msg = (
                    f"Profile {profile_name} references unknown provider "
                    f"{profile.provider}"
                )
                raise ValueError(msg)
        for source, chain in self.fallbacks.items():
            if source not in self.profiles and not chain:
                raise ValueError(f"Fallback route alias has empty chain: {source}")
            for target in chain:
                if target not in self.profiles:
                    raise ValueError(f"Fallback target profile is unknown: {target}")
        return self


def load_config(path: str | Path) -> RuntimeConfig:
    """Load and validate a runtime YAML config file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    config = RuntimeConfig.model_validate(raw)
    resolved_credentials = config.credentials.with_base_path(config_path.parent)
    return config.model_copy(update={"credentials": resolved_credentials})
