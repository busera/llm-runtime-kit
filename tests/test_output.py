from pydantic import BaseModel

from llm_runtime_kit.config import OutputConfig
from llm_runtime_kit.output import (
    OutputValidationResult,
    extract_json_candidate,
    repair_json_candidate,
    validate_output,
)


class Questions(BaseModel):
    objective: str
    questions: list[str]


def test_extract_json_candidate_from_fenced_block() -> None:
    text = 'Here is the result:\n```json\n{"objective":"o","questions":["q"]}\n```'

    assert extract_json_candidate(text) == '{"objective":"o","questions":["q"]}'


def test_extract_json_candidate_from_prose() -> None:
    text = 'Use this payload: {"objective":"o","questions":["q"]} Done.'

    assert extract_json_candidate(text) == '{"objective":"o","questions":["q"]}'


def test_extract_json_candidate_prefers_parseable_object_over_prior_array() -> None:
    text = 'First [1]. Payload: {"ok": true}'

    assert extract_json_candidate(text) == '{"ok": true}'


def test_extract_json_candidate_skips_invalid_braces_for_later_json() -> None:
    text = 'I used {braces} then {"ok": true}'

    assert extract_json_candidate(text) == '{"ok": true}'


def test_repair_json_candidate_fixes_trailing_commas_and_single_quotes() -> None:
    broken = "{'objective': 'o', 'questions': ['q',],}"

    assert repair_json_candidate(broken) == '{"objective": "o", "questions": ["q"]}'


def test_validate_output_parses_and_validates_pydantic_model() -> None:
    result = validate_output(
        '```json\n{"objective":"o","questions":["q"]}\n```',
        OutputConfig(mode="json", require_valid_json=True),
        output_model=Questions,
    )

    assert isinstance(result, OutputValidationResult)
    assert result.success is True
    assert result.parsed == Questions(objective="o", questions=["q"])
    assert result.normalized_text == '{"objective":"o","questions":["q"]}'
    assert result.validation_error is None


def test_validate_output_invalid_json_returns_structured_failure() -> None:
    result = validate_output(
        "not json",
        OutputConfig(mode="json", require_valid_json=True, repair_json=False),
    )

    assert result.success is False
    assert result.parsed is None
    assert "No JSON object or array found" in str(result.validation_error)


def test_validate_output_pydantic_error_omits_raw_input_values() -> None:
    result = validate_output(
        '{"objective":123,"questions":["q"]}',
        OutputConfig(mode="json", require_valid_json=True),
        output_model=Questions,
    )

    assert result.success is False
    assert "input_value" not in str(result.validation_error)
    assert "123" not in str(result.validation_error)
    assert "objective" in str(result.validation_error)


def test_validate_output_custom_validator_can_raise_semantic_failure() -> None:
    def validator(value: object) -> object:
        if not isinstance(value, dict) or value.get("status") != "ok":
            raise ValueError("status must be ok")
        return value

    result = validate_output(
        '{"status":"bad"}',
        OutputConfig(mode="json", require_valid_json=True),
        validator=validator,
    )

    assert result.success is False
    assert "status must be ok" in str(result.validation_error)
