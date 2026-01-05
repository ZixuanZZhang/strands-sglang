# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Shared HTTP Client Support**: Accept optional `client` parameter for connection pooling in high-concurrency scenarios. When running many concurrent requests (e.g., 256 samples per batch in RL training), sharing an `httpx.AsyncClient` significantly reduces connection overhead.

  ```python
  import httpx

  # Create shared client for connection pooling
  client = httpx.AsyncClient(
      base_url="http://localhost:8000",
      timeout=httpx.Timeout(600.0, connect=5.0),
  )

  # Pass to SGLangModel - reused across all requests
  model = SGLangModel(tokenizer=tokenizer, client=client)
  ```

### Changed

- **Timeout Defaults Aligned with OpenAI**: Default timeout is now 600s (10 minutes) with 5s connect timeout, matching OpenAI client behavior for long-running generations.
- **Improved Error Handling**: SGLang HTTP errors now properly raise `ContextWindowOverflowException` for context length errors and `ModelThrottledException` for rate limiting (429/503).

### Fixed

- Default `max_new_tokens` increased for thinking models that require longer outputs.
- Documentation: Added `strands-agents-tools` to pip installation path.

## [0.0.1] - 2026-01-03

### Added

- Initial release with SGLang native `/generate` API support.
- Token-In/Token-Out (TITO) tracking via `TokenManager`.
- Hermes/Qwen tool call parsing with `HermesToolCallParser`.
- `ToolIterationLimiter` hook for clean trajectory truncation.
- Integration with Strands Agents SDK.
