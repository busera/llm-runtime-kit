---
name: llm-runtime-kit
description: Use when an agent needs to add, install, configure, or call the llm-runtime-kit package for reusable multi-provider LLM access across local Ollama, Ollama Cloud, OpenAI-compatible APIs, OpenRouter, and Anthropic without putting secrets in config files.
version: 0.1.0
author: Andrew Buser
license: MIT
metadata:
  hermes:
    tags: [llm, ollama, openai, anthropic, credentials, python]
    related_skills: [code-agent, code-review]
---

# LLM Runtime Kit

## Overview

`llm-runtime-kit` is a reusable Python package for projects that need standard LLM runtime behavior instead of rebuilding provider clients in every repo.

It standardizes:

- external YAML configuration
- strict Pydantic config validation
- local-vs-cloud routing boundaries
- Ollama local daemon usage
- Ollama local-daemon cloud proxy usage
- direct Ollama Cloud API usage
- OpenAI-compatible chat completion APIs
- OpenRouter API usage
- Anthropic Messages API usage
- retry behavior for transient failures
- ordered fallback chains and route aliases
- loguru logging defaults
- generic structured-output contracts: JSON extraction, conservative repair, validation hooks, optional Pydantic validation, and validation-aware retry/fallback behavior
- test seams for mocked HTTP clients

Use this skill as the operational guide. Use `README.md` as the fuller project reference.

## When to Use

Use this skill when:

- adding LLM calls to a Python project
- replacing project-specific Ollama/OpenAI/OpenRouter/Anthropic helper code
- configuring fallback chains across local and cloud models
- enforcing a visible privacy/cost boundary between local and cloud execution
- preparing an agent such as Hermes, Claude Code, OpenClaw, Codex, or another coding agent to install and use the package
- reviewing whether a project's LLM configuration leaks API keys or silently routes private prompts to cloud models

Do not use this skill when:

- the project needs only a one-off curl command
- the project already has a mature provider abstraction with equivalent tests and guardrails
- the caller requires live API credentials during tests; tests must mock external calls

## Non-Negotiable Rules

1. Never put API keys in `config_llm.yaml`.
2. Never commit `.env`, `.env.*`, credentials, logs, or runtime state.
3. Prefer explicit provider routes over magic model-tag behavior.
4. Keep local and cloud routes visibly separate.
5. Resolve optional credentials lazily, only when the selected provider is actually called.
6. Do not let missing OpenAI/OpenRouter/Anthropic/Ollama Cloud credentials break local Ollama usage.
7. Tests must not hit live LLM APIs.
8. Do not log prompts, responses, API keys, Authorization headers, or provider payloads unless a project explicitly opts in and redaction is verified.
9. For cloud execution, require an explicit config policy such as `allow_cloud: true`.
10. If `allow_cloud: false`, cloud model overrides such as `request.model='...-cloud'` must remain blocked.
11. Keep structured-output schemas and semantic/domain validators in consuming projects; the package owns only the generic control loop.

## Install in a Project

From the consuming project, install from the approved package source, for example a local checkout during development:

```bash
python -m pip install -e /path/to/llm-runtime-kit
```

For development of the package itself:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

If packaging later moves to GitHub or a package index, replace the editable install with the pinned package source approved for that project.

## Minimal Runtime Use

```python
from llm_runtime_kit import LLMRequest, LLMRouter, load_config

config = load_config("config_llm.yaml")
router = LLMRouter(config)

response = router.complete(
    LLMRequest(
        prompt="Summarize the key operational risk in one sentence.",
        profile="ollama_default",
    )
)

if not response.success:
    raise RuntimeError(response.error or "LLM call failed")

text = response.text
```

Keep provider clients behind `LLMRouter`. Do not import client classes directly in application code unless writing tests or extending the package.

## Structured Output Use

Use the built-in output-contract layer when a project needs JSON-shaped model output without embedding project schemas in the runtime package.

```python
from pydantic import BaseModel

from llm_runtime_kit import LLMRequest, LLMRouter, load_config


class AuditQuestions(BaseModel):
    objective: str
    questions: list[str]
    risks: list[str]


router = LLMRouter(load_config("config_llm.yaml"))
response = router.complete(
    LLMRequest(
        prompt="Return audit discovery questions as JSON.",
        profile="ollama_default",
        output_model=AuditQuestions,
    )
)

if not response.success:
    raise RuntimeError(response.validation_error or response.error)

validated = response.parsed
```

Rules for agents:

- Put generic policy defaults in `config_llm.yaml` under `output`.
- Put actual schemas, required fields, semantic validators, and domain quality checks in the consuming project.
- Use `output_model` for Pydantic validation and `output_validator` for project-owned semantic validation.
- Treat validation failures as content failures, not transport failures. They can retry and then fall back when `validation_failure_is_retryable` is true.
- Keep prompt/response logging off by default; validation errors can be included in retry prompts only when enabled, and the runtime sanitizes Pydantic errors before reuse.

## Config Pattern

Use `config_llm.yaml` for non-secret runtime policy only.

Recommended shape:

```yaml
default_profile: ollama_default
allow_cloud: false

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
  retryable_statuses: [408, 409, 429, 500, 502, 503, 504]

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
    require_loopback: false

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
    require_loopback: false

profiles:
  ollama_local_default:
    provider: ollama_local
    model: qwen3.6:35b-a3b-coding-mxfp8
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
    think: false

  ollama_cloud_direct:
    provider: ollama_cloud_api
    model: gpt-oss:120b
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000
    think: false

  openrouter_default:
    provider: openrouter
    model: openai/gpt-5.2
    temperature: 0.2
    max_tokens: 4096
    max_context_tokens: 128000

fallbacks:
  ollama_default:
    - ollama_cloud_direct
    - ollama_local_default
    - openrouter_default
```

Important distinction:

- `ollama_local` is for the local daemon with local models.
- `ollama_local_cloud_proxy` is for the local daemon proxying cloud-tagged models after `ollama signin`.
- `ollama_cloud_api` is for direct API-key access to `https://ollama.com/api`.

This separation is deliberate. It makes privacy, cost, and authentication boundaries visible.

Model parameter guidance:

- `max_tokens` is the generated-output budget.
- `max_context_tokens` is the context-window budget. Native Ollama sends it as `options.num_ctx`; other clients retain it as profile/request policy for project prompt packing.
- Keep prompt and response logging disabled unless the project has explicit approval and redaction controls.

## Credential Policy

Current package behavior resolves provider credentials lazily by `api_key_name` using the configured source order: process environment, project `.env`, then optional OS keyring. The deprecated `api_key_env` config key is still accepted as an alias for backward compatibility.

Accepted current patterns:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export OLLAMA_API_KEY='...'
export OPENROUTER_API_KEY='...'
```

Recommended project-local files:

```text
.env.example   # committed, empty placeholders only
.env           # local only, never committed
```

`.env.example`:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OLLAMA_API_KEY=
OPENROUTER_API_KEY=
```

`.gitignore` must include:

```gitignore
.env
.env.*
!.env.example
```

`.env` is loaded by the package-level `CredentialResolver` when `credentials.sources` includes `dotenv`. Keep `.env` local and use mode `600` where practical.

Implemented credential resolver behavior:

- `CredentialConfig` lives in `llm_runtime_kit.credentials` and is embedded in `RuntimeConfig`.
- `CredentialResolver.get(name)` checks process environment, project `.env`, then optional Python `keyring` in configured order.
- Provider configs should use `api_key_name`; `api_key_env` remains a deprecated alias.
- Credential resolution is lazy and provider-specific.
- Do not add ad-hoc secret parsing in provider clients.

## macOS Keychain and Cross-Platform Keyring

The package reads OS keyrings through the optional Python `keyring` package when `credentials.sources` includes `keyring`. If `keyring` is not installed or the backend is unavailable, the resolver treats that source as missing and the provider fails cleanly only if no earlier source supplied the credential.

For local interactive projects, prefer Python `keyring` as the cross-platform abstraction rather than shelling out to macOS `security` directly.

Expected behavior by platform:

- macOS: `keyring` uses Keychain.
- Linux desktop: `keyring` uses Secret Service, GNOME Keyring, or KWallet when available.
- Headless Linux/server/containers: a usable keyring may not exist; prefer environment variables or injected secrets.
- CI: use the CI secret store mapped to environment variables.
- cron/headless launch contexts: do not assume Keychain or a desktop keyring is accessible.

Manual keyring setup example:

```bash
python -m keyring set llm-runtime-kit OPENAI_API_KEY
python -m keyring set llm-runtime-kit ANTHROPIC_API_KEY
python -m keyring set llm-runtime-kit OLLAMA_API_KEY
python -m keyring set llm-runtime-kit OPENROUTER_API_KEY
```

Non-revealing availability check:

```bash
python - <<'PY'
import keyring
name = "OPENAI_API_KEY"
value = keyring.get_password("llm-runtime-kit", name)
print(f"{name}: {'present' if value else 'missing'}")
PY
```

Do not run `python -m keyring get ...` in an agent transcript unless the user explicitly asks to reveal the secret; it prints the secret to stdout.

If keyring access fails in a non-interactive context, fall back to environment variables or a local `.env` file with file mode `600`.

Do not call macOS Keychain through raw subprocess calls from package code. Use the Python keyring abstraction or an injected credential resolver.

## Authentication Behavior

OpenAI-compatible providers, including OpenRouter:

- Credential name: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or a provider-specific configured name
- Header: Authorization bearer token
- Endpoint appended by client: `/v1/chat/completions`

Anthropic:

- Credential name: `ANTHROPIC_API_KEY`
- Header: `x-api-key: <key>`
- Header: `anthropic-version: 2023-06-01`
- Endpoint appended by client: `/v1/messages`

Direct Ollama Cloud API:

- Credential name: `OLLAMA_API_KEY`
- Header: Authorization bearer token
- Endpoint appended by client: `/chat` against `https://ollama.com/api`

Local Ollama daemon:

- No API key required.
- Use loopback endpoint such as `http://localhost:11434`.
- If local daemon cloud proxying is used, authenticate with the Ollama CLI outside Python, for example `ollama signin`.

## Agent Implementation Checklist

When using this package in a consuming repo:

1. Inspect the consuming repo's existing LLM code and tests.
2. Install the package or add it to the repo's dependency manifest.
3. Add or update `config_llm.yaml` with explicit provider/profile/fallback sections.
4. Add `.env.example` if API-key providers are configured, including `OPENROUTER_API_KEY` when OpenRouter is used.
5. Verify `.gitignore` excludes `.env`, `.env.*`, logs, caches, and runtime state.
6. Replace direct provider calls with `LLMRouter.complete()`.
7. Keep application code provider-agnostic; profiles should be config names, not hardcoded model strings scattered through code.
9. Add tests with mocked clients or mocked HTTP openers.
10. Test cloud-disabled behavior with `allow_cloud: false`.
11. Test missing optional credentials do not break local-only execution.
12. Run syntax, unit tests, lint, format check, and secret scan before committing.

## Testing Pattern

Use dependency injection rather than live API calls.

Router-level test shape:

```python
from llm_runtime_kit import LLMRequest, LLMResponse, LLMRouter

class FakeClient:
    def complete(self, request, provider_name, provider, profile):
        return LLMResponse(True, "ok", provider_name, profile.model)

router = LLMRouter(
    config,
    clients={
        "openai_compatible": FakeClient(),
        "ollama_native": FakeClient(),
        "anthropic": FakeClient(),
    },
    sleep=lambda seconds: None,
)

response = router.complete(LLMRequest(prompt="test", profile="ollama_default"))
assert response.success is True
```

Client-level tests should mock `urlopen` and assert:

- URL path
- HTTP method
- headers without exposing secret values
- payload shape
- timeout
- response parsing
- HTTP error handling
- timeout/URL error handling
- invalid JSON handling where implemented

## Common Pitfalls

1. Treating README as enough for agents.
   - README explains the package, but agents need operational steps, guardrails, and failure-mode checks. Keep this `SKILL.md` as the agent-facing contract.

2. Hiding cloud behavior behind a model name.
   - Use separate providers/profiles for local daemon, local cloud proxy, and direct cloud API.

3. Putting API keys in YAML.
   - YAML may be committed, copied into prompts, or logged. Store only credential names in YAML.

4. Resolving all credentials during config load.
   - This breaks local-only usage when optional cloud provider keys are missing. Resolve only at call time.

5. Assuming macOS Keychain works from every context.
   - Interactive user sessions usually work. cron, headless agents, and containers may not.

6. Using Python keyring without a fallback plan on Linux.
   - Linux keyring depends on desktop/session services. Servers and containers often need env vars or injected secret files.

7. Letting `allow_cloud: false` only check profile names.
   - The effective model must include one-call overrides from `LLMRequest.model`.

8. Broad retry behavior for non-idempotent calls.
   - Retry only idempotent completion calls. Do not retry operations with external side effects unless explicitly designed for idempotency.

9. Logging prompts or responses by default.
   - Prompts may contain private data. Keep `log_prompts: false` and `log_responses: false` unless a project has explicit approval.

## Verification Checklist

Before declaring integration complete:

- [ ] `config_llm.yaml` contains no secret values.
- [ ] `.env.example` exists if API-key providers are used.
- [ ] `.env` and `.env.*` are gitignored.
- [ ] Local-only route works without OpenAI, OpenRouter, Anthropic, or Ollama Cloud keys.
- [ ] Missing required API key returns a clean failed `LLMResponse`, not an import/config crash.
- [ ] `allow_cloud: false` blocks OpenAI, OpenRouter, Anthropic, direct Ollama Cloud API, and Ollama cloud model tags.
- [ ] Tests mock external HTTP calls.
- [ ] Retry tests cover transient HTTP statuses and timeout/network errors where the consuming project relies on retries.
- [ ] No prompts, responses, Authorization headers, or API keys appear in logs or test output.
- [ ] Syntax, tests, lint, format check, and secret scan pass.
