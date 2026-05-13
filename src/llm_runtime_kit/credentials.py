"""Lazy credential resolution for provider API keys."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CredentialSource = Literal["env", "dotenv", "keyring"]
DotenvLoader = Callable[[Path], Mapping[str, str | None]]
KeyringGetter = Callable[[str, str], str | None]


class CredentialConfig(BaseModel):
    """Credential resolution policy for provider API keys."""

    model_config = ConfigDict(extra="forbid")

    sources: list[CredentialSource] = Field(
        default_factory=lambda: ["env", "dotenv", "keyring"]
    )
    dotenv_path: Path | None = Path(".env")
    keyring_service: str = "llm-runtime-kit"

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[CredentialSource]) -> list[CredentialSource]:
        """Reject duplicate or empty credential source chains."""

        if not value:
            raise ValueError("credential sources must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("credential sources must not contain duplicates")
        return value

    def with_base_path(self, base_path: Path) -> CredentialConfig:
        """Return a copy with relative dotenv paths resolved from a config directory."""

        if self.dotenv_path is None or self.dotenv_path.is_absolute():
            return self
        return self.model_copy(update={"dotenv_path": base_path / self.dotenv_path})


class CredentialResolver:
    """Resolve provider credentials lazily from env, .env, then OS keyring."""

    def __init__(
        self,
        config: CredentialConfig | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        dotenv_loader: DotenvLoader | None = None,
        keyring_getter: KeyringGetter | None = None,
    ) -> None:
        self.config = config or CredentialConfig()
        self._environ = environ if environ is not None else os.environ
        self._dotenv_loader = dotenv_loader
        self._keyring_getter = keyring_getter
        self._dotenv_cache: dict[str, str | None] | None = None

    def get(self, name: str) -> str | None:
        """Resolve a credential by configured name without logging or revealing it."""

        if not name:
            return None
        for source in self.config.sources:
            if source == "env":
                value = self._from_env(name)
            elif source == "dotenv":
                value = self._from_dotenv(name)
            else:
                value = self._from_keyring(name)
            if value:
                return value
        return None

    def _from_env(self, name: str) -> str | None:
        """Resolve from process environment."""

        return self._normalize(self._environ.get(name))

    def _from_dotenv(self, name: str) -> str | None:
        """Resolve from configured .env file without mutating process environment."""

        if self.config.dotenv_path is None:
            return None
        values = self._load_dotenv()
        return self._normalize(values.get(name))

    def _from_keyring(self, name: str) -> str | None:
        """Resolve from OS keyring when the optional dependency is available."""

        getter = self._keyring_getter or self._default_keyring_getter()
        if getter is None:
            return None
        try:
            return self._normalize(getter(self.config.keyring_service, name))
        except (RuntimeError, OSError):
            return None

    def _load_dotenv(self) -> Mapping[str, str | None]:
        """Load .env values once and cache them for this resolver instance."""

        if self._dotenv_cache is not None:
            return self._dotenv_cache
        path = self.config.dotenv_path
        if path is None or not path.exists():
            self._dotenv_cache = {}
            return self._dotenv_cache
        if self._dotenv_loader is not None:
            loaded = self._dotenv_loader(path)
        else:
            loaded = self._default_dotenv_loader(path)
        self._dotenv_cache = dict(loaded)
        return self._dotenv_cache

    @staticmethod
    def _default_dotenv_loader(path: Path) -> Mapping[str, str | None]:
        """Load .env with python-dotenv if installed, otherwise use a small parser."""

        try:
            from dotenv import dotenv_values
        except ImportError:
            return _parse_simple_dotenv(path)
        return dotenv_values(path)

    @staticmethod
    def _default_keyring_getter() -> KeyringGetter | None:
        """Return keyring.get_password when the optional keyring package is usable."""

        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError:
            return None

        def get_password(service: str, name: str) -> str | None:
            try:
                return keyring.get_password(service, name)
            except KeyringError as error:
                raise RuntimeError(str(error)) from error

        return get_password

    @staticmethod
    def _normalize(value: str | None) -> str | None:
        """Normalize missing and blank credential values to None."""

        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


def _parse_simple_dotenv(path: Path) -> dict[str, str | None]:
    """Parse simple KEY=VALUE lines when python-dotenv is unavailable."""

    values: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    return values
