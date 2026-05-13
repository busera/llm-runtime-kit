"""Logging helpers for LLM runtime operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

DEFAULT_REDACT_KEYS = {"api_key", "authorization", "token", "password", "secret"}


def redact_mapping(
    data: Mapping[str, Any], redact_keys: set[str] | None = None
) -> dict[str, Any]:
    """Return a copy with sensitive-looking keys redacted."""

    keys = {key.lower() for key in (redact_keys or DEFAULT_REDACT_KEYS)}
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in keys or any(marker in key.lower() for marker in keys):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def get_logger():
    """Return the shared loguru logger."""

    return logger
