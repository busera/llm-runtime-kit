# PLAN — LLM Runtime Kit

CONTEXT
  Build a reusable Python package for standard LLM usage across projects so provider config, retry, logging, and fallback behavior do not need to be re-created per repo.
  Source examples reviewed from existing local implementations; public docs avoid project-specific provenance.
  Prior decisions loaded:
  - Fallback chains are fully configurable and may intentionally mix local Ollama, Ollama Cloud, OpenAI-compatible, and Anthropic profiles.
  - Cloud execution is blocked at runtime by `allow_cloud: false`, including one-call model overrides such as `LLMRequest.model`.
  - Ollama local and Ollama Cloud need explicit privacy distinction.
  - Ollama thinking-capable models can return empty content unless thinking is disabled / content fallback is handled.
  - Cloud transient retries should cover 408, 409, 429, 500, 502, 503, 504 and network timeout/URLError classes, only for idempotent calls.
  Relevant URS: FR-INF-001, FR-INF-002, FR-INF-006, FR-INF-090+, SR-INF-006/SR-INF-007.
  Context7 grounding: /websites/python_3 for urllib timeout and HTTPError/URLError handling; /pydantic/pydantic for v2 config models.

FILES
  Create: pyproject.toml — package/test/lint metadata.
  Create: config_llm.yaml — safe example external config.
  Create: src/llm_runtime_kit/config.py — Pydantic v2 validated external config.
  Create: src/llm_runtime_kit/retry.py — bounded retry policy and transient classification.
  Create: src/llm_runtime_kit/types.py — provider request/response dataclasses.
  Create: src/llm_runtime_kit/router.py — profile/fallback routing with local-only guard.
  Create: src/llm_runtime_kit/clients/*.py — Ollama/OpenAI-compatible and Anthropic HTTP clients.
  Create: tests/* — config, retry, routing, and mocked client behavior.
  No standalone recreate-from-scratch blueprint is maintained; README.md and SKILL.md are the consolidated documentation sources.

ARCHITECTURE
  - Data flow: config_llm.yaml -> RuntimeConfig -> LLMRouter -> ProviderClient -> LLMResponse.
  - Data models: Pydantic config at boundaries; frozen dataclasses for runtime request/response.
  - Test seams: HTTP transport is injectable; sleep function injectable in retry; environment lookup injectable through process env in tests.
  - Boundaries: config validation, retry policy, provider adapters, routing/fallback are separate modules.
  - Change locality: adding a provider should mean one client module + config enum/registry update, not edits in retry/config internals.

QUALITY GATES
  - No live API calls in tests.
  - No hardcoded API keys; config carries env var names only.
  - No production print(); package uses loguru.
  - All HTTP calls use timeout.
  - YAML loading uses yaml.safe_load.
  - Config parse allows configurable fallback chains; runtime `allow_cloud` blocks cloud profiles and cloud model overrides when disabled.
  - Ollama loopback normalization handles 0.0.0.0 safely.

RISKS
  - Provider schema drift -> keep client adapters thin and tested with representative payloads.
  - Cloud leakage -> explicit allow_cloud flag, separate local/cloud providers, and effective-model override checks.
  - Retry storms -> bounded attempts, retry only classified transient failures.
  - Prompt/response logging leaks -> default no prompt/response logging; redaction helpers.

EXECUTION TASK LIST
  1. Create project structure and plan — verify folder exists.
  2. Implement config/types/retry — verify py_compile and unit tests.
  3. Implement provider clients/router — verify mocked HTTP tests only.
  4. Add docs/example config — verify no secrets and local/cloud distinction is explicit.
  5. Run pytest, py_compile, ruff if available, initialize git and commit initial scaffold.
