Write integration tests for a given module or feature.

The user provides a target as $ARGUMENTS (e.g., a module name, class, or feature). If not provided, ask.

## Conventions

Follow the existing test style in `tests/integration/`:

- **License header**: Every `.py` file must start with the Apache 2.0 license header (copy from any existing source file)
- **File naming**: `test_<feature>.py` (e.g., `test_sglang_integration.py`)
- **Location**: `tests/integration/`
- **Docstring**: Start with `"""Integration tests for <feature> with a real SGLang server."""`
- **Async tests**: Use `async def test_*` directly — `asyncio_mode = "auto"` is configured
- **Shared fixtures**: `conftest.py` provides `sglang_base_url`, `sglang_server_info`, `tokenizer`, `model`, `calculator_tool` — use these, don't redefine them
- **Assertions**: Use plain `assert`

## What to Test

Integration tests validate behavior with a **real SGLang server**. Each test should cover one observable behavior:

- Model generation produces valid output
- Token tracking captures correct token IDs and logprobs
- Tool parsing works end-to-end with real model output
- Error handling for real server responses (context length, throttling)
- Multi-turn conversation with tool use

Do NOT duplicate unit test coverage. Integration tests are expensive (real LLM calls) — keep them focused.

## Steps

1. Read the module source to understand the feature being tested
2. Read `tests/integration/conftest.py` for available fixtures
3. Read an existing integration test file (e.g., `test_sglang_integration.py`) as reference
4. Write tests that exercise the full pipeline with a real server
5. After writing, remind the user to run `/run-integration-tests` (requires a running SGLang server)
