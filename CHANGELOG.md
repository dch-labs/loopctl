# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Fixed

- Restored the no-`#[allow(clippy::*)]` lint contract. Fixed: a private `TextStreamer` type alias, lossless integer-to-float casts (centralized in an internal `numeric` module), `PartLane`/`TerminalStage` lane enums replacing bool fields, and stale-allow deletions. No public API change.

## [0.2.0] - 2026-08-02

### Added

- `TurnMode` enum: engine runs non-streaming (`create_message`) or streaming (`StreamHandler`), selectable at runtime via `set_turn_mode`. Default is feature-dependent.
- `streaming` feature: gates `StreamHandler`, per-delta callbacks, `async-stream`. With `default = []`, no streaming code is compiled.
- Sans-IO `LoopMachine`: serializable, owns all loop decisions. Exposed via `machine()` / `into_machine()` / `from_machine()`.
- Session/Run/Turn lifetime model: `SessionConfig` (session-scoped) + `RunConfig` (per-run budgets).
- Reasoning-model support: `DeltaPart::Thinking` + `on_thinking_delta`.
- Structured output: `StructuredOutput` trait, `ResponseFormat`, `request_structured::<T>()`.
- Tool constraints (`ToolConstraint`): `Strict` and `Grammar` modes.
- Tool reflection (`LlmReflector`): model classifies failed tool calls and suggests corrections.
- Parallel tool dispatch (`ParallelDispatchConfig`).
- `StreamHandler`: retry, timeout, rate-limit backoff, non-streaming fallback.
- Client-side rate limiting: `TokenBucket` / `RateLimiter`, one bucket per `base_url`.
- `ContextContributor` trait: turn-boundary message injection.
- `LoopMemory` trait wired: stores trajectories, retrieves before turns, consolidates on success.
- `DisplayHint` on `ToolOutput`: rendering hints (Text, Diff, Json, Code, Suppress, Markdown).
- Middleware: `VerifyMiddleware`, `MemoizingMiddleware`.
- Presets: `ConstrainedProfile`, `FrontierProfile`, `GoalReminder`.
- `StreamRequest`: bundles `(messages, system, tools)` for all `ApiClient` methods.
- `Role::System` variant.
- Pluggable `TokenCounter` (`HeuristicTokenCounter` default).
- `OpenAiClientBuilder::with_stream_usage(bool)`.
- `with_tcp_nodelay(bool)` on all provider builders.
- HTTP connection-pool injection: shared `reqwest::Client`, pool knobs.
- Fluent `with_*()` builders on `BareLoop` and provider builders.
- `LoopError` derives `Serialize`, `Deserialize`, `PartialEq`, `Eq`.

### Changed

- **Breaking:** `default = []` no longer pulls `async-stream`; streaming is opt-in via `streaming`. HTTP providers imply it, so `features = ["openai"]` is unchanged. Migration: add `streaming` if you used `providers` alone.
- **Breaking:** `TurnMode` no longer implements `Default`. Migration: use `turn_mode()` / `set_turn_mode()`.
- **Breaking:** `create_message` returns typed `NonStreamingResponse` instead of `serde_json::Value`.
- **Breaking:** `extract_structured` takes `&Message`; per-provider overrides removed.
- **Breaking:** `run()` is `run(&mut self, &str, &RunConfig) -> Result<Run, LoopError>`. `Loop::initialize` / `config()` removed.
- **Breaking:** `on_session_start`/`on_session_end` renamed to `on_run_start`/`on_run_end`.
- **Breaking:** `LoopConfig` removed; split into `SessionConfig` + `RunConfig`.
- **Breaking:** compaction thresholds are `u8` percentages (0–100) instead of `f64`.
- **Breaking:** builders uniformly `with_`-prefixed; `Option<T>` builders take `Option<T>`.
- **Breaking:** `StreamHandler::stream_turn` returns `impl Stream<Item = Result<HandlerEvent, StreamHandlerError>>`.
- **Breaking:** `ApiClient` methods take `&StreamRequest` instead of positional params.
- **Breaking:** `DeltaPart`, `Role`, `ToolOutput`, `ToolDispatchResult`, `ToolPostContext` are `#[non_exhaustive]`.
- **Breaking:** `Reflector::analyze` gains `tool_schema: Option<&ToolSchema>`.
- **Breaking:** `MessagePart::ToolResult` gains `name` field (serde-defaulted for old data).
- **Breaking:** `NonStreamingResponse.usage` is `Option<Usage>`.
- Cancellation no longer trips the fallback breaker (`record_turn_failure` guards `Cancelled`).
- OpenAI streaming sets `stream_options.include_usage`; all providers report usage on both paths.
- OpenAI malformed `function.arguments` returns `ApiError` instead of defaulting to `{}`.
- Gemini parses `functionCall.id` (Gemini 3) and echoes it in responses.
- SSE invalid UTF-8 surfaced as protocol error instead of `U+FFFD` replacement.
- `StreamStopReason::from_api_str` accepts `"tool_use"` (Anthropic alias).
- MSRV bumped to 1.94.

### Removed

- `record_stream_success` / `record_stream_failure` (renamed to `record_turn_*`).
- `StreamCapable` trait now requires the `streaming` feature.
- `StreamHandler::with_config(timeout, retry)` — use `with_timeout_config` / `with_retry_config`.
- `FallbackManager::record_api_failure` / `record_model_failure` — merged into `record_failure(FailureKind)`.
- `Loop::process_turn`, `BareLoop::run_turn_body` — replaced by machine-driven `run()`.
- `StreamTurnResult`, `StreamHandler::with_request_options`, `StreamHandlerError::RateLimitEscalation.prior`.
- `parking_lot` dependency.

### Fixed

- Release profile no longer uses `panic = "abort"` (it disabled `catch_unwind` tool-panic isolation).
- OpenAI streaming dropped multi-chunk tool-call argument fragments.
- `StreamAccumulator` dropped parallel tool calls with interleaved arguments.
- `BareLoop` was dead after one cancellation (`CancelSignal` not re-armed; now resets in `finalize()`).
- Tool results split across multiple user messages instead of merged per turn.
- `StreamHandler` accepted invalid timeout/retry configs; `jitter_factor` was validated but never applied.
- Zero-event streams could hang ~20 min before failing.
- `ToolHealthRegistry::is_tool_available` consumed the HalfOpen probe as a read side effect.
- Anthropic provider hardcoded text-block index to 0.
- Per-run manager reset wiped session-scoped state.

### Security

- Auto-commit hook's `git add -A` on empty file list staged the whole working tree; now refuses.
- Response body size guard (10 MB) now pre-checks `Content-Length` instead of firing after full materialization.

## [0.1.0] - 2025-07-01

Initial crates.io release.

### Added

- Trait-based LLM loop framework with pluggable clients, tools, and memory
- Core message, error, and session types
- Convergence detection using Jaccard similarity
- Loop detector with configurable thresholds and hard-stop
- Fallback manager for model failover
- LLM streaming support
- Provider clients for OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Grok, and ZAI
- Tool dispatch with panic isolation
- Tool health monitoring and tool shield middleware
- Interactive sessions with hooks
- Context compaction with truncating compactor
- Output limit middleware for token budget enforcement
- Built-in testing utilities for writing LLM loop tests
- Example CLIs: hello, REPL, echo tool, and multi-provider chat

[0.2.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.2.0
[0.1.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.1.0
