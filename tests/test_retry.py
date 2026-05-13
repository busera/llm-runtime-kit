from urllib.error import HTTPError, URLError

import pytest

from llm_runtime_kit.config import RetryConfig
from llm_runtime_kit.retry import calculate_retry_delay, classify_exception, retry_call


def test_classify_exception_marks_retryable_http_status() -> None:
    error = HTTPError("https://example.test", 503, "unavailable", {}, None)

    decision = classify_exception(error, {503})

    assert decision.retryable is True
    assert decision.status_code == 503


def test_classify_exception_rejects_non_retryable_http_status() -> None:
    error = HTTPError("https://example.test", 401, "auth", {}, None)

    decision = classify_exception(error, {503})

    assert decision.retryable is False
    assert decision.status_code == 401


def test_retry_call_retries_transient_url_error_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("timed out")
        return "ok"

    result, attempts = retry_call(
        operation,
        RetryConfig(max_attempts=3, base_delay_seconds=0.5, jitter_seconds=0),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [0.5]


def test_retry_call_applies_configured_jitter_to_delay() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("timed out")
        return "ok"

    result, attempts = retry_call(
        operation,
        RetryConfig(max_attempts=3, base_delay_seconds=0.5, jitter_seconds=0.25),
        sleep=sleeps.append,
        jitter=lambda start, stop: stop,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [0.75]


def test_calculate_retry_delay_caps_base_then_adds_jitter() -> None:
    delay = calculate_retry_delay(
        RetryConfig(
            max_attempts=3,
            base_delay_seconds=10,
            max_delay_seconds=12,
            jitter_seconds=0.25,
        ),
        attempt=3,
        jitter=lambda start, stop: stop,
    )

    assert delay == 12.25


def test_retry_call_stops_after_max_attempts() -> None:
    def operation() -> str:
        raise URLError("timed out")

    with pytest.raises(URLError):
        retry_call(
            operation,
            RetryConfig(max_attempts=2, base_delay_seconds=0, jitter_seconds=0),
            sleep=lambda _delay: None,
        )
