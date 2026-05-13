"""Generic output-contract helpers for LLM responses."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputValidationResult:
    """Structured result for generic output validation."""

    success: bool
    normalized_text: str
    parsed: object | None = None
    validation_error: str | None = None


_FENCED_JSON_RE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```",
    re.DOTALL,
)
_TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")


def extract_json_candidate(text: str) -> str:
    """Extract a JSON object/array candidate from messy model output."""

    stripped = text.strip()
    fenced = _FENCED_JSON_RE.search(stripped)
    if fenced:
        return fenced.group("body").strip()

    for candidate in _json_candidates(stripped, prefer_objects=True):
        if _candidate_is_parseable(candidate):
            return candidate
    for candidate in _json_candidates(stripped, prefer_objects=True):
        return candidate
    raise ValueError("No JSON object or array found in model output")


def parse_json(text: str) -> object:
    """Parse JSON text with standard-library json.loads."""

    return json.loads(text)


def repair_json_candidate(text: str) -> str:
    """Repair minor JSON defects and return parseable JSON text.

    The built-in repair intentionally stays conservative: it handles trailing
    commas and Python-literal style single quotes. If the optional `json_repair`
    package is installed, it is tried first for broader repair coverage.
    """

    stripped = text.strip()
    try:
        parse_json(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return stripped

    repaired_by_dependency = _repair_with_optional_dependency(stripped)
    if repaired_by_dependency is not None:
        return repaired_by_dependency

    no_trailing_commas = _TRAILING_COMMA_RE.sub("", stripped)
    try:
        parsed = ast.literal_eval(no_trailing_commas)
    except (SyntaxError, ValueError):
        try:
            parsed = parse_json(no_trailing_commas)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON repair failed: {error}") from error
    return json.dumps(parsed)


def validate_output(
    text: str,
    config: Any,
    *,
    output_model: type[Any] | None = None,
    validator: Callable[[object], object | None] | None = None,
) -> OutputValidationResult:
    """Validate model output according to a generic output contract."""

    try:
        if not _requires_json_contract(config, output_model, validator):
            return OutputValidationResult(True, text, parsed=None)

        candidate = extract_json_candidate(text)
        parse_error: Exception | None = None
        try:
            parsed = parse_json(candidate)
        except json.JSONDecodeError as error:
            parse_error = error
            if not getattr(config, "repair_json", False):
                raise
            parsed = _parse_repaired(candidate, config)

        if parse_error and parsed is None:
            raise parse_error

        validated = _validate_parsed(parsed, output_model, validator)
        normalized_text = json.dumps(parsed, separators=(",", ":"))
        return OutputValidationResult(True, normalized_text, parsed=validated)
    except Exception as error:  # noqa: BLE001 - validation failures are data
        return OutputValidationResult(
            False,
            text,
            parsed=None,
            validation_error=_safe_validation_error(error),
        )


def _safe_validation_error(error: Exception) -> str:
    """Return a validation error string without raw input payload values."""

    errors_method = getattr(error, "errors", None)
    if callable(errors_method):
        try:
            errors = errors_method(include_input=False, include_context=False)
        except TypeError:
            errors = errors_method()
        safe_errors = []
        for item in errors:
            if isinstance(item, dict):
                safe_errors.append(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"input", "input_value", "ctx"}
                    }
                )
        if safe_errors:
            return f"Validation failed: {safe_errors}"
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON: {error.msg} at line {error.lineno} column {error.colno}"
    return str(error)


def _requires_json_contract(
    config: Any,
    output_model: type[Any] | None,
    validator: Callable[[object], object | None] | None,
) -> bool:
    return (
        getattr(config, "mode", "text") == "json"
        or getattr(config, "require_valid_json", False)
        or output_model is not None
        or validator is not None
    )


def _parse_repaired(candidate: str, config: Any) -> object:
    max_attempts = max(0, int(getattr(config, "max_repair_attempts", 0)))
    if max_attempts < 1:
        return parse_json(candidate)
    repaired = repair_json_candidate(candidate)
    return parse_json(repaired)


def _validate_parsed(
    parsed: object,
    output_model: type[Any] | None,
    validator: Callable[[object], object | None] | None,
) -> object:
    validated: object = parsed
    if output_model is not None:
        model_validate = getattr(output_model, "model_validate", None)
        if model_validate is None:
            raise TypeError("output_model must expose model_validate")
        validated = model_validate(parsed)
    if validator is not None:
        result = validator(validated)
        if result is not None:
            validated = result
    return validated


def _repair_with_optional_dependency(text: str) -> str | None:
    try:
        from json_repair import repair_json  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency may be absent/broken
        return None
    repaired = repair_json(text)
    if isinstance(repaired, str):
        return repaired
    return json.dumps(repaired)


def _json_candidates(text: str, *, prefer_objects: bool) -> list[str]:
    """Return balanced JSON-looking candidates from model output."""

    starts = ["{", "["] if prefer_objects else ["[", "{"]
    candidates: list[tuple[int, str]] = []
    for opening in starts:
        search_at = 0
        while True:
            start = text.find(opening, search_at)
            if start < 0:
                break
            end = _matching_json_end(text, start)
            if end is not None:
                candidates.append((start, text[start : end + 1].strip()))
                search_at = end + 1
            else:
                search_at = start + 1
    if prefer_objects:
        object_candidates = [
            candidate for _, candidate in candidates if candidate.startswith("{")
        ]
        array_candidates = [
            candidate for _, candidate in candidates if candidate.startswith("[")
        ]
        return object_candidates + array_candidates
    return [candidate for _, candidate in sorted(candidates, key=lambda item: item[0])]


def _candidate_is_parseable(candidate: str) -> bool:
    try:
        parse_json(candidate)
    except json.JSONDecodeError:
        return False
    return True


def _matching_json_end(text: str, start: int) -> int | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escape = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index
        elif char in "}]":
            return None
    return None
