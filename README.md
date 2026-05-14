# LLM Runtime Kit

Reusable Python runtime for projects that call LLMs through Ollama local models, Ollama Cloud via local daemon proxy, direct Ollama Cloud API access, OpenAI-compatible APIs, OpenRouter, and Anthropic.

The point is to stop rebuilding the same provider/config/retry/fallback/logging layer in every LLM project.

## What it standardizes

- External YAML configuration via `config_llm.yaml`
- Strict Pydantic v2 validation with `extra="forbid"`
- Local-vs-cloud privacy guardrails
- Explicit provider/API-style separation (`api_style: openai_compatible` or `native`)
- Ollama local daemon, local daemon cloud proxy, and direct Ollama Cloud API routes
- OpenAI-compatible API support
- OpenRouter API support
- Anthropic Messages API support
- Retry handling for transient failures
- Provider fallback chains and route aliases
- Loguru-based logging defaults
- Generic structured-output contract layer: JSON extraction, conservative repair, validation hooks, optional Pydantic validation, and validation-aware retry/fallback behavior
- No live API calls in the test suite

## Agent guide included

This repository includes a root `SKILL.md` for AI coding agents such as Hermes, Claude Code, OpenClaw, and Codex. Treat it as the operational guide for installing, configuring, and safely using `llm-runtime-kit` in another project. It covers provider boundaries, credential handling, local-vs-cloud routing, structured-output use, testing expectations, common pitfalls, and verification steps.

## Current status

This is an early v0.1 package: a reusable starting point for shared LLM projects, not a v1.0 runtime yet. The public repository is https://github.com/busera/llm-runtime-kit. CI currently runs compile, pytest, Ruff lint, and Ruff format checks on Python 3.10, 3.11, and 3.12. The test suite uses mocked tests for config validation, clients, credential lookup, output contracts, retry behavior, and routing/fallback behavior. It does not make live API calls.

Known next implementation areas:

- Add structured runtime logging for each fallback decision without prompt/response leakage.
- Add a CLI smoke command for safe local route checks.


## Installation

From the public GitHub repository:

```bash
python -m pip install 'llm-runtime-kit @ git+https://github.com/busera/llm-runtime-kit.git'
```

From a local checkout during development:

```bash
python -m pip install -e .
```

For package development:

```bash
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[credentials]'  # Python keyring support
python -m pip install -e '.[structured]'   # optional json-repair integration
```

## Quick start

```python
from llm_runtime_kit import LLMRequest, LLMRouter, load_config

config = load_config("config_llm.yaml")
router = LLMRouter(config)

response = router.complete(
    LLMRequest(
        prompt="Summarize this note in one sentence.",
        profile="ollama_default",
    )
)

if response.success:
    print(response.text)
else:
    raise RuntimeError(response.error)
```

## Public API and main functions

The package intentionally exposes a small public API from `llm_runtime_kit`:

```python
from llm_runtime_kit import (
    CredentialConfig,
    CredentialResolver,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    OutputConfig,
    OutputValidationResult,
    RuntimeConfig,
    load_config,
)
```

### `load_config(path: str | Path) -> RuntimeConfig`

Loads an external YAML configuration file and validates it into a `RuntimeConfig` Pydantic model.

What it does:

- Expands and resolves the supplied path.
- Reads YAML with `yaml.safe_load()`.
- Requires the top-level YAML document to be a mapping.
- Applies strict Pydantic validation with unknown fields forbidden.
- Validates provider/profile/fallback references before runtime use.

Typical use:

```python
from llm_runtime_kit import load_config

config = load_config("config_llm.yaml")
```

Raises:

- `ValueError` if the YAML root is not a mapping.
- Pydantic validation errors if fields are missing, invalid, unknown, or reference unknown providers/profiles.
- File-related exceptions if the config path cannot be read.

### `RuntimeConfig`

Validated runtime configuration model. This is the in-memory representation of `config_llm.yaml`.

Important fields:

- `default_profile`: default route name. Can be either a concrete profile name or a fallback route alias.
- `allow_cloud`: execution guard. When `false`, cloud/API-key profiles are skipped.
- `logging`: `LoggingConfig` for log level and prompt/response logging policy.
- `retry`: `RetryConfig` for retry attempts, exponential backoff, jitter, and retryable status codes.
- `credentials`: `CredentialConfig` for lazy API-key lookup order and project `.env` path.
- `output`: `OutputConfig` for generic structured-output policy defaults.
- `providers`: provider endpoint definitions keyed by provider name, with explicit `kind` and `api_style`.
- `profiles`: named model presets keyed by profile name.
- `fallbacks`: ordered fallback chains. Keys can be concrete profiles or route aliases.

Validation rules:

- `default_profile` must exist either in `profiles` or `fallbacks`.
- Every profile must reference an existing provider.
- Every fallback target must reference an existing profile.
- A fallback route alias that is not itself a profile must have a non-empty chain.
- Unknown fields are rejected via `ConfigDict(extra="forbid")`.

### `CredentialConfig` and `CredentialResolver`

`CredentialConfig` controls lazy API-key lookup. The default source order is:

1. process environment variable
2. project `.env` file
3. OS keyring through the optional Python `keyring` package

`load_config("config_llm.yaml")` resolves a relative `credentials.dotenv_path` from the config file's directory. `CredentialResolver.get(name)` returns a secret string or `None`; it does not log or print secret values.

Key behavior:

- Provider configs store credential references only through `api_key_name`.
- `api_key_env` remains a deprecated backward-compatible alias and is internally treated as `api_key_name`.
- Resolution is lazy: keys are only resolved when the provider is actually called.
- Missing OpenAI/OpenRouter/Anthropic/Ollama Cloud keys do not break local Ollama config loading or local-only routing.
- OS keyring is optional and best for interactive local dev. In CI, containers, Linux servers, cron, and locked macOS sessions, prefer process env vars or `.env`.

Typical direct use:

```python
from llm_runtime_kit import CredentialConfig, CredentialResolver

resolver = CredentialResolver(CredentialConfig())
api_key = resolver.get("OPENAI_API_KEY")
```

### `LLMRequest`

Provider-agnostic request dataclass passed to the router.

Fields:

- `prompt`: user prompt text. Required.
- `system_prompt`: system instruction. Defaults to `"You are a precise, evidence-led assistant."`.
- `profile`: optional profile or route alias override. If omitted, `RuntimeConfig.default_profile` is used.
- `model`: optional one-call model override. If omitted, the selected profile model is used.
- `temperature`: optional one-call temperature override.
- `max_tokens`: optional one-call max-token override.
- `metadata`: optional string metadata for caller-side tracking. The current clients do not send this metadata to providers.
- `max_context_tokens`: optional one-call context-window budget override. The native Ollama client sends this as `options.num_ctx`; other clients retain it as caller-side policy for prompt packing/budgeting.
- `output_mode`: optional one-call output mode override: `"text"` or `"json"`.
- `require_valid_json`, `repair_json`, `max_repair_attempts`, `validation_failure_is_retryable`, `include_validation_error_in_retry_prompt`: optional per-request overrides for `OutputConfig`.
- `output_model`: optional Pydantic-style model class exposing `model_validate()`.
- `output_validator`: optional callable receiving parsed/validated output and returning a replacement object or raising an exception.

Typical use:

```python
request = LLMRequest(
    prompt="Draft three project discovery questions.",
    profile="ollama_default",
    temperature=0.1,
)
```

### `LLMResponse`

Provider-agnostic response dataclass returned by clients and the router.

Fields:

- `success`: `True` only when usable response text was returned.
- `text`: model response text.
- `provider`: provider name used for the attempt.
- `model`: model used for the attempt.
- `tokens_used`: total token usage when the provider returns it; otherwise `0`.
- `error`: error text when the attempt failed.
- `status_code`: HTTP status code for failed HTTP responses when available.
- `attempts`: number of attempts made for the returned candidate profile.
- `fallback_used`: `True` when the response came from a non-first fallback candidate.
- `parsed`: validated object/dict for structured-output calls; otherwise `None`.
- `validation_error`: content/contract validation error when structured output failed.
- `raw_text`: original model text before JSON normalization when output validation ran.

Typical handling:

```python
response = router.complete(request)
if not response.success:
    raise RuntimeError(response.error or "LLM call failed")
```

### `LLMRouter`

Routes an `LLMRequest` through configured profiles and fallback chains.

Constructor:

```python
router = LLMRouter(config)
```

Optional test seams:

```python
router = LLMRouter(
    config,
    clients={
        "openai_compatible": fake_client,
        "ollama_native": fake_client,
        "anthropic": fake_client,
    },
    sleep=lambda seconds: None,
)
```

Main method:

#### `LLMRouter.complete(request: LLMRequest) -> LLMResponse`

What it does:

1. Selects `request.profile` or `config.default_profile`.
2. Resolves that name to a fallback chain if it exists in `config.fallbacks`; otherwise treats it as a concrete profile.
3. Iterates candidates in order.
4. Skips cloud/API-key candidates when `allow_cloud` is `false`.
5. Calls the matching provider client.
6. Retries transient failures according to `RetryConfig`, including capped exponential backoff plus `jitter_seconds`.
7. If an output contract is active, extracts/parses/repairs JSON and runs the optional Pydantic model or validator.
8. If output validation fails and `validation_failure_is_retryable` is true, retries the same profile up to `retry.max_attempts`; then falls back to the next profile.
9. If all candidates fail or are skipped, returns the last failed `LLMResponse`.

Privacy behavior:

- `allow_cloud: false` blocks OpenAI/OpenRouter/Anthropic/API-key providers and Ollama model tags containing `:cloud` or `-cloud`.
- Local Ollama is treated as local only when the provider is `kind: ollama`, `require_loopback: true`, and the selected model is not a cloud tag.

### `ProviderClient` protocol

Interface implemented by provider clients.

```python
class ProviderClient(Protocol):
    def complete(
        self,
        request: LLMRequest,
        provider_name: str,
        provider: ProviderConfig,
        profile: ModelProfile,
    ) -> LLMResponse: ...
```

Use this protocol when injecting fake clients into `LLMRouter` tests or when adding another provider implementation.

### `OpenAICompatibleClient`

HTTP client for `/v1/chat/completions` APIs. It is used for:

- local Ollama daemon when configured as `api_style: openai_compatible`
- local Ollama daemon proxying cloud model tags after `ollama signin`
- OpenAI-compatible remote APIs
- OpenAI itself
- OpenRouter

Behavior:

- Sends `developer`/user messages for OpenAI and OpenRouter, and system/user messages for Ollama-compatible endpoints.
- Adds an authorization bearer header when `provider.api_key_name` is configured and the credential resolves.
- Returns a failed `LLMResponse` before the HTTP call when a non-Ollama provider config names an API key credential name that cannot be resolved.
- Sends `temperature`, `stream: false`, and an output-token limit. OpenAI and OpenRouter receive `max_completion_tokens`; Ollama-compatible endpoints receive `max_tokens`.
- Sends Ollama `think` only for `provider.kind == "ollama"` when the profile sets `think`.
- Uses `provider.timeout_seconds` on the HTTP call.
- Parses `choices[0].message.content` first, then falls back to `reasoning` or `thinking` fields for thinking-capable responses.
- Returns failed `LLMResponse` objects for HTTP, URL, timeout, missing choices, or empty response cases.

Important configuration detail:

- Set `base_url` to the root endpoint, for example `https://api.openai.com` or `https://openrouter.ai/api`, because the client appends `/v1/chat/completions`.
- OpenAI's current Chat Completions reference recommends `developer` messages for o1 and newer model families and marks `max_tokens` as deprecated in favor of `max_completion_tokens`; this client follows that convention for `kind: openai`.
- OpenRouter's current OpenAPI/reference docs support the `developer` role and both `max_tokens` and `max_completion_tokens`; this client uses `max_completion_tokens` for `kind: openrouter`.

Recommended OpenRouter configuration:

```yaml
providers:
  openrouter:
    kind: openrouter
    api_style: openai_compatible
    base_url: https://openrouter.ai/api
    api_key_name: OPENROUTER_API_KEY
    timeout_seconds: 120
    require_loopback: false

profiles:
  openrouter_default:
    provider: openrouter
    model: openai/gpt-5.2
```

OpenRouter documents `POST /api/v1/chat/completions` with bearer-token authentication and OpenAI-compatible request/response shapes. Optional `HTTP-Referer` and `X-OpenRouter-Title` attribution headers are documented by OpenRouter but are not sent by the current runtime client.

### `OllamaNativeClient`

HTTP client for Ollama native `/api/chat` APIs. It is used for direct API-key access to `https://ollama.com/api`, and can also support a local native endpoint when configured with an `/api` base URL.

Behavior:

- Sends requests to `{base_url}/chat`.
- Requires `provider.api_key_name` only when the provider config names a credential reference.
- Adds an authorization bearer header when an API key is configured and present.
- Sends system/user messages, `stream: false`, `think` when configured, and Ollama `options` with `temperature`, `num_predict`, and optional `num_ctx` from `max_context_tokens`.
- Parses `message.content` first, then falls back to `message.thinking`.
- Sums `prompt_eval_count` and `eval_count` when present.
- Returns failed `LLMResponse` objects for missing configured API key, HTTP, URL, timeout, or empty response cases.

Recommended direct Ollama Cloud configuration:

```yaml
providers:
  ollama_cloud_api:
    kind: ollama
    api_style: native
    base_url: https://ollama.com/api
    api_key_name: OLLAMA_API_KEY
    require_loopback: false
```

Do not include `/chat` in the `base_url`; the client appends `/chat`.

### `AnthropicClient`

HTTP client for Anthropic's Messages API.

Behavior:

- Sends requests to `{base_url}/v1/messages`.
- Requires an API key credential configured by `provider.api_key_name`.
- Sends `x-api-key` and `anthropic-version: 2023-06-01` headers.
- Sends `system`, one user message, `temperature`, and `max_tokens`.
- Sums `usage.input_tokens` and `usage.output_tokens` when present.
- Returns failed `LLMResponse` objects for missing API key, HTTP, URL, timeout, or empty text cases.

Recommended `base_url`:

```yaml
providers:
  anthropic:
    kind: anthropic
    api_style: native
    base_url: https://api.anthropic.com
```

Do not include `/v1` in the Anthropic `base_url`; the client appends `/v1/messages`.

### Retry helpers

#### `classify_exception(error: BaseException, retryable_statuses: set[int]) -> RetryDecision`

Classifies exceptions as retryable or non-retryable.

Retryable by default:

- `urllib.error.HTTPError` when its status code is in `retryable_statuses`
- `urllib.error.URLError`
- `TimeoutError`
- `socket.timeout`

Returns a `RetryDecision` with:

- `retryable`: whether the operation should be retried
- `status_code`: HTTP code when available
- `reason`: compact reason string, such as `http_429` or the exception class name

#### `calculate_retry_delay(policy, attempt, jitter=None) -> float`

Returns the retry sleep for one failed attempt. The base delay is `base_delay_seconds * 2 ** (attempt - 1)`, capped by `max_delay_seconds`; when `jitter_seconds` is non-zero, a random value in `[0, jitter_seconds]` is added after the cap. The optional `jitter` callable is a deterministic test seam.

#### `retry_call(operation, policy, sleep=time.sleep, jitter=None) -> tuple[object, int]`

Runs an idempotent operation with bounded exponential backoff.

Behavior:

- Calls `operation()`.
- Returns `(result, attempt_count)` on success.
- Retries retryable exceptions until `policy.max_attempts` is reached.
- Uses `base_delay_seconds * 2 ** (attempt - 1)`, capped by `max_delay_seconds`, then adds jitter.
- Adds random jitter in `[0, jitter_seconds]` when `jitter_seconds` is non-zero.
- Re-raises the original exception when it is non-retryable or attempts are exhausted.

Use this helper for side-effect-safe/idempotent calls only.

### Logging helpers

#### `redact_mapping(data, redact_keys=None) -> dict[str, Any]`

Returns a shallow copy of a mapping with sensitive-looking fields replaced by `<redacted>`.

Default redaction markers:

- `api_key`
- `authorization`
- `token`
- `password`
- `secret`

Matching is case-insensitive and checks both exact keys and keys containing a marker.

#### `get_logger()`

Returns the shared Loguru `logger` instance. The runtime uses Loguru but keeps prompt/response logging disabled by default because prompts may contain private data.

### Structured output helpers

Structured-output support is intentionally generic. The runtime kit standardizes the control loop; consuming projects own their schemas, semantic validators, and domain quality checks.

#### `OutputConfig`

Generic structured-output policy defaults from `config_llm.yaml`:

- `mode`: `text` or `json`. `text` preserves current behavior unless a request supplies an `output_model`, `output_validator`, or per-call JSON override.
- `require_valid_json`: require a JSON object or array.
- `repair_json`: try conservative JSON repair for minor syntax defects.
- `max_repair_attempts`: bounded repair attempts. The built-in repair is intentionally small; install `llm-runtime-kit[structured]` to allow optional `json-repair` integration when available.
- `validation_failure_is_retryable`: whether content/contract failures retry the same profile before fallback.
- `include_validation_error_in_retry_prompt`: whether retry prompts include a sanitized validation error as feedback. Defaults to `false` to avoid carrying prior output contents across retries/fallback providers.

#### `validate_output(text, config, output_model=None, validator=None)`

Runs the generic output contract:

1. If no contract is active, returns success with original text and no parsed object.
2. If JSON is required, extracts a fenced or embedded JSON candidate from messy output.
3. Parses JSON with `json.loads()`.
4. Optionally repairs minor JSON defects such as trailing commas and Python-literal style single quotes.
5. Optionally validates with a Pydantic-style `output_model.model_validate()`.
6. Optionally runs a caller validator. The validator should raise a clean exception for semantic/domain failure.
7. Returns `OutputValidationResult(success, normalized_text, parsed, validation_error)`.

Example with a project-owned schema:

```python
from pydantic import BaseModel

from llm_runtime_kit import LLMRequest, LLMRouter, load_config


class ProjectSummary(BaseModel):
    title: str
    bullets: list[str]


router = LLMRouter(load_config("config_llm.yaml"))
response = router.complete(
    LLMRequest(
        prompt="Return a short project summary as JSON.",
        profile="ollama_default",
        output_model=ProjectSummary,
    )
)

if not response.success:
    raise RuntimeError(response.validation_error or response.error)

validated = response.parsed
```

Failure types:

- Transport failure: timeout, HTTP 5xx/429, URL error. Handled by retry/fallback.
- Content failure: model answered, but JSON extraction/parsing/model validation failed. Handled generically by output-contract retry/fallback when enabled.
- Semantic failure: JSON shape is valid, but project/domain rules reject it. The project owns those rules through `output_validator`; the runtime only carries the failure through the same structured retry/fallback path.

### Config utility functions

#### `normalize_ollama_base_url(value: str) -> str`

Normalizes unsafe wildcard Ollama bind addresses to loopback client URLs.

Examples:

- `http://0.0.0.0:11434` -> `http://127.0.0.1:11434`
- `http://[::]:11434` -> `http://127.0.0.1:11434`
- `http://localhost:11434/` -> `http://localhost:11434`

Raises `ValueError` for malformed URLs.

#### `is_cloud_model_tag(model: str) -> bool`

Returns `True` when a model tag clearly indicates cloud execution. Current markers are:

- `:cloud`
- `-cloud`

This function supports the router's `allow_cloud` gate.

## Configuration

Configuration lives outside application code. See `config_llm.yaml` for a complete example.

A `profile` is only a named model preset: provider + model tag + generation options. The name can be whatever your project wants. A fallback key such as `ollama_default` can also act as a route alias that points to an ordered chain of profiles.

Minimal shape:

```yaml
default_profile: ollama_default
allow_cloud: false

credentials:
  sources: [env, dotenv, keyring]
  dotenv_path: .env
  keyring_service: llm-runtime-kit

logging:
  level: INFO
  log_prompts: false
  log_responses: false
  redact_keys:
    - api_key
    - authorization
    - token

retry:
  max_attempts: 3
  base_delay_seconds: 1.0
  max_delay_seconds: 20.0
  jitter_seconds: 0.25
  retryable_statuses:
    - 408
    - 409
    - 429
    - 500
    - 502
    - 503
    - 504

providers:
  ollama_local:
    kind: ollama
    api_style: openai_compatible
    base_url: http://localhost:11434
    api_key_name: ""
    timeout_seconds: 120
    require_loopback: true
  ollama_local_cloud_proxy:
    kind: ollama
    api_style: openai_compatible
    base_url: http://localhost:11434
    api_key_name: ""
    timeout_seconds: 120
    require_loopback: true
  ollama_cloud_api:
    kind: ollama
    api_style: native
    base_url: https://ollama.com/api
    api_key_name: OLLAMA_API_KEY
    timeout_seconds: 120
    require_loopback: false
  openai:
    kind: openai
    api_style: openai_compatible
    base_url: https://api.openai.com
    api_key_name: OPENAI_API_KEY
    timeout_seconds: 120
  openrouter:
    kind: openrouter
    api_style: openai_compatible
    base_url: https://openrouter.ai/api
    api_key_name: OPENROUTER_API_KEY
    timeout_seconds: 120
    require_loopback: false
  anthropic:
    kind: anthropic
    api_style: native
    base_url: https://api.anthropic.com
    api_key_name: ANTHROPIC_API_KEY
    timeout_seconds: 120

profiles:
  ollama_cloud_direct:
    provider: ollama_cloud_api
    model: gpt-oss:120b
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
    think: false
  ollama_local_cloud_proxy_default:
    provider: ollama_local_cloud_proxy
    model: gpt-oss:120b-cloud
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
    think: false
  ollama_local_default:
    provider: ollama_local
    model: qwen3.6:35b-a3b-coding-mxfp8
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
    think: false
  openai_default:
    provider: openai
    model: gpt-5.5
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
  openrouter_default:
    provider: openrouter
    model: openai/gpt-5.2
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
  anthropic_default:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 200000

fallbacks:
  ollama_default:
    - ollama_cloud_direct
    - ollama_local_cloud_proxy_default
    - ollama_local_default
    - openai_default
    - openrouter_default
    - anthropic_default
```

Fallback chains are ordered. The first successful profile wins. These chains can mix local daemon, local-daemon cloud proxy, direct Ollama Cloud API, and vendor API profiles when that is what the project wants:

```yaml
fallbacks:
  ollama_default:
    - ollama_cloud_direct
    - ollama_local_cloud_proxy_default
    - ollama_local_default
    - openai_default
    - openrouter_default
    - anthropic_default
```

Fallback trigger behavior:

- Fallback is evaluated per profile, after that profile fails or is skipped.
- For retryable failures, the current profile is retried up to `retry.max_attempts` before moving to the next fallback profile. With `max_attempts: 3`, this means initial try + 2 retries; if all 3 attempts fail, the fallback kicks in.
- For non-retryable provider/client failures, such as missing credentials or non-transient HTTP 4xx responses, the router can move to the next fallback immediately because another retry of the same profile would not help.
- Cloud-blocked profiles under `allow_cloud: false` are skipped without consuming retry attempts.
- `fallback_used` in `LLMResponse` is `True` when the successful response came from a later fallback candidate rather than the first candidate in the resolved chain.

With `allow_cloud: false`, remote/cloud entries are skipped. With `allow_cloud: true`, they are eligible.

## Deferred runtime parameter ideas

The following project-style LLM runtime parameters were considered but intentionally kept out of the v0.1 core unless a future implementation enforces them:

| Parameter | Status in `llm-runtime-kit` | Rationale |
|---|---|---|
| `json_log_max_chars` | Defer | Useful only after the package implements actual prompt/response logging sinks. |
| `streaming` / `evidence_streaming` | Defer | Streaming changes the response contract and callback surface. Keep v0.1 request/response synchronous. |
| `stall_seconds`, `stall_retries`, `stall_backoff_seconds` | Defer | Provider timeout and retry already exist. Stall detection is useful for streaming/long-running project pipelines, not the current non-streaming client core. |
| `empty_response_retries`, `empty_response_backoff_seconds` | Defer | JSON/content-contract failures are now handled by the generic output layer. Empty plain-text responses still fail and fall back through normal router behavior; a separate empty-text retry policy is only useful if a project needs that distinction. |


## Privacy and provider routing rules

Fallbacks are configurable policy. A fallback chain may point to any profile: local Ollama, Ollama Cloud, OpenAI-compatible, OpenRouter, or Anthropic. The runtime does not assume fallbacks are local.

Guardrails currently implemented:

- `allow_cloud: false` blocks execution of cloud/API-key profiles, including direct Ollama Cloud API, OpenAI, OpenRouter, Anthropic, and Ollama `:cloud`/`-cloud` tags through the local daemon.
- `allow_cloud: true` permits configured cloud/API-key profiles and cloud fallbacks.
- `api_style` selects the request/response shape explicitly: `openai_compatible` for `/v1/chat/completions`, `native` for Ollama `/api/chat` or Anthropic Messages.
- Ollama local providers can require loopback via `require_loopback: true`.
- Wildcard Ollama bind addresses such as `http://0.0.0.0:11434` normalize to `http://127.0.0.1:11434` for client calls.
- API keys are resolved lazily from credential references, not stored in config.

## Ollama API documentation cross-check

The Ollama routes in `config_llm.yaml` were checked against the current Ollama documentation.

Confirmed settings:

- Native Ollama chat uses `POST /api/chat`. The runtime sets `base_url: https://ollama.com/api` for direct cloud access and appends `/chat`, producing `https://ollama.com/api/chat`.
- Native `/api/chat` supports `messages`, `options`, `stream`, and `think`. The runtime sends `stream: false`, maps `max_tokens` to Ollama `options.num_predict`, sends `temperature` under `options`, and forwards `think` when configured.
- Native `/api/chat` responses include `message.content`, optional `message.thinking`, `prompt_eval_count`, and `eval_count`. The runtime parses content first, then thinking, and sums those token counts when present.
- Direct Ollama Cloud API access requires an API key for `https://ollama.com/api` and uses an authorization bearer header. The example config references `OLLAMA_API_KEY` but does not store the key.
- Direct Ollama Cloud API examples use base model tags such as `gpt-oss:120b`, while local-daemon cloud proxy examples use cloud tags such as `gpt-oss:120b-cloud` after `ollama signin`.
- Local Ollama requires no authentication when accessed at `http://localhost:11434`.
- Ollama's OpenAI-compatible API supports `/v1/chat/completions` on the local daemon. This runtime stores the daemon root URL, for example `http://localhost:11434`, and appends `/v1/chat/completions` internally.

Caution:

- Ollama's Cloud documentation demonstrates local-daemon cloud model calls through native `/api/chat`. The runtime also allows `ollama_local_cloud_proxy` through the OpenAI-compatible client because the local daemon exposes `/v1/chat/completions`; if a project depends on that exact cloud-proxy route in production, smoke-test it against the installed Ollama version and model tag.
- Ollama's documented API error statuses include `429`, `500`, and `502`; the default retry list also includes broader transient statuses commonly used by API gateways (`408`, `409`, `503`, `504`).

## OpenAI API documentation cross-check

OpenAI support was checked against the current OpenAI API reference.

Confirmed settings:

- OpenAI Chat Completions are exposed under the v1 REST API. The runtime stores `base_url: https://api.openai.com` and appends `/v1/chat/completions`, producing `https://api.openai.com/v1/chat/completions`.
- OpenAI authenticates with a bearer authorization header. The example config references `OPENAI_API_KEY` but does not store the key.
- The current Chat Completions reference recommends `developer` messages instead of `system` messages for o1 and newer model families. The runtime sends a `developer` instruction message for `kind: openai`.
- The current Chat Completions reference marks `max_tokens` as deprecated in favor of `max_completion_tokens`, and notes incompatibility with o-series models. The runtime sends the profile/request `max_tokens` budget as `max_completion_tokens` for `kind: openai`.
- OpenAI recommends trying the newer Responses API for new projects. This package intentionally remains on Chat Completions for v0.1 OpenAI-compatible provider reuse; adding a dedicated Responses client is a future extension, not a silent behavior change.

## OpenRouter API documentation cross-check

OpenRouter support was checked against the current OpenRouter documentation and OpenAPI specification.

Confirmed settings:

- OpenRouter uses `POST /api/v1/chat/completions` and OpenAI-compatible request/response shapes.
- The runtime stores `base_url: https://openrouter.ai/api` and appends `/v1/chat/completions`, producing `https://openrouter.ai/api/v1/chat/completions`.
- OpenRouter authenticates with an authorization bearer token. The example config references `OPENROUTER_API_KEY` but does not store the key.
- OpenRouter supports the `developer` message role, so the runtime sends `developer`/user messages for `kind: openrouter`.
- OpenRouter supports `max_completion_tokens` and still documents `max_tokens` as deprecated. The runtime sends the profile/request `max_tokens` budget as `max_completion_tokens` for `kind: openrouter`.
- OpenRouter model names use provider/model identifiers, for example `openai/gpt-5.2`.
- Optional `HTTP-Referer` and `X-OpenRouter-Title` attribution headers can be useful for OpenRouter rankings, but they are not required and are not sent by the current runtime client.

## Anthropic API documentation cross-check

Anthropic support was checked against the current Claude API overview and Messages API reference.

Confirmed settings:

- The direct Claude API root is `https://api.anthropic.com`. The runtime stores that root and appends `/v1/messages`, producing `https://api.anthropic.com/v1/messages`.
- Anthropic requires `anthropic-version`; the runtime sends `anthropic-version: 2023-06-01`.
- Anthropic supports API-key authentication through `x-api-key`. The runtime sends `x-api-key` from the configured `ANTHROPIC_API_KEY` credential reference.
- Anthropic Messages requests use a top-level `system` field plus user/assistant messages. The runtime sends the configured system prompt as top-level `system` and the prompt as one user message.
- Anthropic Messages requests use `max_tokens` for the output budget. The runtime sends the profile/request `max_tokens` value as `max_tokens` for `kind: anthropic`.
- Anthropic non-streaming responses return `content` blocks and token `usage`; the runtime concatenates text blocks and sums `input_tokens` plus `output_tokens` when present.

## API key based providers

API-key providers are configured by naming a credential reference, not by storing a secret value. Prefer `api_key_name`; the older `api_key_env` key remains a deprecated compatibility alias.

Credential resolution order is configured once:

```yaml
credentials:
  sources: [env, dotenv, keyring]
  dotenv_path: .env
  keyring_service: llm-runtime-kit
```

Resolution order:

1. process environment variable
2. project `.env`
3. OS keyring through Python `keyring` when installed and available

The repository commits `.env.example` and ignores `.env` / `.env.*`. Use `.env` for local projects, CI-like smoke runs, containers, cron, and simple onboarding when that plaintext trade-off is acceptable. For interactive local development, OS keyring/Keychain is preferred when available. For headless Linux, cron, containers, and locked macOS sessions, environment variables or `.env` are usually more reliable than keyring.

Example:

```yaml
providers:
  openai:
    kind: openai
    api_style: openai_compatible
    base_url: https://api.openai.com
    api_key_name: OPENAI_API_KEY
    timeout_seconds: 120

  openrouter:
    kind: openrouter
    api_style: openai_compatible
    base_url: https://openrouter.ai/api
    api_key_name: OPENROUTER_API_KEY
    timeout_seconds: 120
    require_loopback: false

  anthropic:
    kind: anthropic
    api_style: native
    base_url: https://api.anthropic.com
    api_key_name: ANTHROPIC_API_KEY
    timeout_seconds: 120
```

Do not commit `.env` files or real credentials. The repository `.gitignore` excludes common secret and runtime artifact files.

## Retry behavior

The runtime is designed for idempotent completion-style calls. Transient retry status codes default to:

- 408
- 409
- 429
- 500
- 502
- 503
- 504

Network timeout and URL errors are handled by the client layer and surfaced as failed responses for router/fallback handling. The lower-level `retry_call()` helper can also re-run an idempotent operation when it raises retryable HTTP/network exceptions.

## Logging

Logging uses Loguru. Prompt and response logging are disabled by default because prompts may contain private data.

Enable prompt/response logging only for controlled debugging sessions:

```yaml
logging:
  level: DEBUG
  log_prompts: true
  log_responses: true
```

## Development checks

Use the same checks before committing:

```bash
python -m compileall -q src tests
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

If you work from the local development environment that uses `uv`, the equivalent command set is:

```bash
uv run python -m compileall -q src tests
uv run python -m pytest -q
uv run python -m ruff check src tests
uv run python -m ruff format --check src tests
```

## Repository layout

```text
llm-runtime-kit/
├── .env.example
├── config_llm.yaml
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── llm_runtime_kit/
│       ├── config.py
│       ├── credentials.py
│       ├── logging.py
│       ├── output.py
│       ├── retry.py
│       ├── router.py
│       ├── types.py
│       └── clients/
│           ├── anthropic.py
│           ├── base.py
│           ├── ollama_native.py
│           └── openai_compatible.py
└── tests/
```

## References

Ollama documentation checked for the current configuration:

- Ollama Chat API, `POST /api/chat`: https://docs.ollama.com/api/chat
- Ollama Cloud, local cloud model tags and direct Cloud API examples: https://docs.ollama.com/cloud
- Ollama API authentication, local no-auth vs `OLLAMA_API_KEY` for `https://ollama.com/api`: https://docs.ollama.com/api/authentication
- Ollama OpenAI compatibility, local `/v1/chat/completions`: https://docs.ollama.com/api/openai-compatibility
- Ollama API errors and documented status codes: https://docs.ollama.com/api/errors

OpenAI documentation checked for the current configuration:

- OpenAI API overview, bearer auth, `https://api.openai.com/v1` examples, request IDs, and compatibility notes: https://developers.openai.com/api/reference/overview/
- OpenAI Chat Completions overview and note that new projects should consider Responses: https://developers.openai.com/api/reference/chat-completions/overview/
- OpenAI Create Chat Completion reference, `developer` message guidance for o1 and newer, and `max_completion_tokens` vs deprecated `max_tokens`: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/

OpenRouter documentation checked for the current configuration:

- OpenRouter quickstart and raw API endpoint: https://openrouter.ai/docs/quickstart
- OpenRouter authentication and optional attribution headers: https://openrouter.ai/docs/api/reference/authentication
- OpenRouter API overview and request schema: https://openrouter.ai/docs/api/reference/overview
- OpenRouter Chat Completion endpoint reference: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
- OpenRouter OpenAPI specification: https://openrouter.ai/openapi.yaml

Anthropic documentation checked for the current configuration:

- Claude API overview, root URL, available APIs, authentication headers, and request limits: https://docs.anthropic.com/en/api/overview
- Claude Messages API reference, `POST /v1/messages`, request/response shape, `max_tokens`, content blocks, and usage fields: https://docs.anthropic.com/en/api/messages
- Legacy Claude API overview URL also checked because search results still surface it: https://docs.anthropic.com/claude/reference/getting-started-with-the-api

## AI assistance statement

AI assistance was used to refine the scripts, tests, and documentation. The implementation decisions, privacy guardrails, and acceptance criteria remain human-directed. Generated content was reviewed and validated before commit.

## License

MIT License. See `LICENSE`.
