"""Bounded retry helpers for idempotent LLM calls."""

from __future__ import annotations

import random
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError

from llm_runtime_kit.config import RetryConfig

TRANSIENT_NETWORK_ERRORS = (TimeoutError, socket.timeout, URLError)


@dataclass(frozen=True)
class RetryDecision:
    """Retry classification result."""

    retryable: bool
    status_code: int | None = None
    reason: str = ""


def classify_exception(
    error: BaseException, retryable_statuses: set[int]
) -> RetryDecision:
    """Classify whether an exception is transient."""

    if isinstance(error, HTTPError):
        return RetryDecision(
            error.code in retryable_statuses, error.code, f"http_{error.code}"
        )
    if isinstance(error, TRANSIENT_NETWORK_ERRORS):
        return RetryDecision(True, None, type(error).__name__)
    return RetryDecision(False, None, type(error).__name__)


def calculate_retry_delay(
    policy: RetryConfig,
    attempt: int,
    jitter: Callable[[float, float], float] | None = None,
) -> float:
    """Return bounded exponential delay plus optional positive jitter."""

    delay = min(
        policy.max_delay_seconds,
        policy.base_delay_seconds * (2 ** (attempt - 1)),
    )
    if policy.jitter_seconds:
        jitter_value = (jitter or random.uniform)(0.0, policy.jitter_seconds)
        delay += jitter_value
    return delay


def retry_call(
    operation: Callable[[], object],
    policy: RetryConfig,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] | None = None,
) -> tuple[object, int]:
    """Run an idempotent operation with bounded exponential retry."""

    attempt = 1
    while True:
        try:
            return operation(), attempt
        except (HTTPError, URLError, TimeoutError) as error:
            decision = classify_exception(error, policy.retryable_statuses)
            if not decision.retryable or attempt >= policy.max_attempts:
                raise
            sleep(calculate_retry_delay(policy, attempt, jitter))
            attempt += 1
