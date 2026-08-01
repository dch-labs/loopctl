# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- **Non-streaming engine turn path** (`TurnMode`): the engine can now drive each
  turn via `ApiClient::create_message` instead of streaming, selected at runtime
  with `BareLoop::set_turn_mode` / `with_turn_mode`. The default is
  `TurnMode::Streaming` when `streaming` is compiled in, `TurnMode::NonStreaming`
  otherwise. Under the non-streaming path no per-delta observer callbacks fire;
  the full assistant text still surfaces via `on_response`.
- **`streaming` feature flag**: gates `StreamHandler`, `stream::handler`,
  `StreamCapable`, `text_streamer`, and the `on_text_delta` / `on_thinking_delta`
  firing sites. With `default = []`, `async-stream` is no longer pulled and the
  engine compiles and runs without any streaming machinery.

### Changed

- **`default = []` now means non-streaming.** `async-stream` is an optional
  dependency enabled by `streaming`; it is no longer pulled into a bare
  `cargo add loopctl`. The HTTP provider features (`openai`, `anthropic`,
  `gemini`, and anything that chains from them) now imply `streaming`, so the
  common `features = ["openai"]` case preserves the previous streaming-by-default
  behavior. Migration: users who enabled only `providers` (not a named provider)
  and relied on streaming must add `streaming` explicitly.
- **MSRV corrected to 1.94** (was misdocumented as 1.85 in `AGENTS.md`; the
  `Cargo.toml` `rust-version` and `.clippy.toml` already required 1.94).
- `AGENTS.md` feature table, `providers` dependency list, core-deps list, and the
  `LoopMemory` object-safety note corrected to match the code.

### Removed

- The `record_stream_success` / `record_stream_failure` private methods are
  renamed to `record_turn_success` / `record_turn_failure` (shared by both turn
  paths). The `StreamCapable` trait and its `LoopManagers` impl now require the
  `streaming` feature.


### Added

- **Sans-IO state machine** (`engine::machine::LoopMachine`): serializable,
  owns every agent-loop decision (turn counting, max-turn enforcement,
  tool-call validity, compaction trigger, history, cancellation). Exposed via
  `BareLoop::machine()` / `into_machine()` / `from_machine()` for inspection
  and serialize-and-resume. `BareLoop::run()` now drives the machine internally.
- **Session/Run/Turn lifetime model** (`engine::core`): one `Session` spans the
  process, one `Run` per `run()` call, one `Turn` per loop iteration. Session
  derives per-session totals; `Run` carries per-run turns, tokens, and error.
  Construction splits into `SessionConfig` (session-scoped) and `RunConfig`
  (per-run budgets).
- **Reasoning-model support** (`DeltaPart::Thinking` + `on_thinking_delta`):
  reasoning tokens (Claude extended-thinking, OpenAI o-series, Gemini 2.5+) are
  routed as their own stream kind. Stream-only — not accumulated into `Message`.
- **Structured output** (`structured` module): `StructuredOutput` trait,
  `ResponseFormat`, `request_structured::<T>()`. All three providers override
  `stream_messages_with_options` / `create_message_with_options` to inject the
  schema natively.
- **Tool constraints** (`ToolConstraint` enum): `Strict` tightens tool schemas
  via the provider's native strict mode; `Grammar` compiles schemas into a
  grammar for vLLM-style samplers (`grammar` feature).
- **Tool reflection** (`LlmReflector`): asks the model to classify failed tool
  calls and suggest corrections via `request_structured`.
- **Parallel tool dispatch** (`ParallelDispatchConfig`): independent,
  concurrency-safe calls within a single turn run concurrently. Sequential by
  default.
- **Stream resilience** (`StreamHandler`): retries, timeouts, rate-limit
  backoff, and non-streaming fallback. Configurable via
  `with_timeout_config` / `with_retry_config` / `with_rate_limit_config`.
  `HandlerEvent` enum provides real-time observability during streaming.
- **Client-side rate limiting** (`TokenBucket` / `RateLimiter`): proactive
  per-provider token-bucket. One bucket per `base_url`.
- **Context contributors** (`ContextContributor` trait): turn-boundary hook for
  injecting messages before each model call.
- **Agent memory** (`LoopMemory` trait, now wired): stores tool-call
  trajectories, retrieves relevant entries before each turn, consolidates on
  successful runs. Object-safe; configure via `BareLoop::set_memory`.
- **Display hints** (`DisplayHint` on `ToolOutput`): advisory rendering hints
  (Text, Diff, Json, Code, Suppress, Markdown) for presentation layers.
- **Middleware**: `VerifyMiddleware` (post-execution verification),
  `MemoizingMiddleware` (tool-call result caching with path-aware invalidation).
- **Presets** (`ConstrainedProfile`, `FrontierProfile`, `GoalReminder`): named
  runtime profiles for small-model-tuned and frontier configurations.
- **`StreamRequest`**: bundles `(messages, system, tools)` into one parameter
  for all `ApiClient` methods.
- **`Role::System`** variant: framework-injected system context. Providers map
  to their native representation.
- **Pluggable `TokenCounter`**: `HeuristicTokenCounter` (4 chars/token) default;
  swap in a real tokenizer. Synced bidirectionally with `ContextManager`.
- **`OpenAiClientBuilder::with_stream_usage(bool)`**: controls
  `stream_options.include_usage`. `ollama()` disables it automatically.
- **`with_tcp_nodelay(bool)`** on all three provider builders + `HttpClientConfig`.
- **HTTP connection-pool injection**: shared `reqwest::Client`, pool knobs
  (`pool_max_idle_per_host`, `pool_idle_timeout`, `tcp_keepalive`).
- **Fluent `with_*()` builders** on `LoopConfig`, `BareLoop`, and all provider
  builders. `CancelSignal::reset()` for multi-run agents.
- `LoopError` now derives `Serialize`, `Deserialize`, `PartialEq`, `Eq`.

### Changed

- **Breaking (`create_message` returns typed `NonStreamingResponse`):** no longer
  returns raw `serde_json::Value`. The struct carries `message: Message`,
  `stop_reason: StreamStopReason`, `usage: Option<Usage>`. Migration: read fields
  from the struct.
- **Breaking (`extract_structured` takes `&Message`):** default implementation
  derives from the message; per-provider overrides removed. Migration: change
  parameter type or delete override.
- **Breaking (`run()` signature):** `run(&mut self, input: &str, &RunConfig)`
  returns `Result<Run, LoopError>`. `Loop::initialize` / `config()` removed.
- **Breaking (session→run lifecycle rename):** `on_session_start` / `on_session_end`
  → `on_run_start` / `on_run_end`. Fire on every `run()` call. Migration: rename
  methods and context types.
- **Breaking (`LoopConfig` removed):** replaced by `SessionConfig` + `RunConfig`.
  The `model` field lives on `ApiClient`.
- **Breaking (compaction thresholds → percentages):** `f64` fractions → `u8`
  percentages (0–100). `0.80 → 80`.
- **Breaking (builder renames):** all consuming builders uniformly `with_`-prefixed.
  `Option<T>` builders take `Option<T>` (no `_opt` suffix). Migration: add prefix,
  wrap literals in `Some(...)`.
- **Breaking (`StreamHandler::stream_turn`):** returns `impl Stream<Item =
  Result<HandlerEvent, StreamHandlerError>>` instead of a future.
- **Breaking (`StreamRequest` parameter):** all `ApiClient` streaming/creation
  methods take `&StreamRequest` instead of positional params.
- **Breaking (`DeltaPart` non-exhaustive):** add `_ =>` arm to downstream matches.
- **Breaking (`Role::System` variant):** add `System =>` arm to downstream matches.
- **Breaking (`ToolOutput` / `ToolDispatchResult` / `ToolPostContext`
  non-exhaustive):** use named constructors or `..Default::default()`.
- **Breaking (`Reflector::analyze`):** gains `tool_schema: Option<&ToolSchema>`
  parameter.
- **Breaking (`MessagePart::ToolResult` gains `name` field):**
  `tool_result()` gains a `name` argument (second param). `#[serde(default)]`
  allows deserializing older data.
- **Breaking (`NonStreamingResponse.usage` is `Option<Usage>`):** symmetric with
  streaming. `None` = provider omitted usage.
- **OpenAI streaming usage:** now sets `stream_options.include_usage` and captures
  token counts from the final chunk. All three providers report usage on both paths.
  All-zero usage collapses to `None`.
- **OpenAI malformed arguments:** non-empty `function.arguments` that fail JSON
  parse now return an `ApiError` instead of silently defaulting to `{}`.
- **Gemini tool-call ids:** `functionCall.id` is now parsed (Gemini 3) and echoed
  in `functionResponse.id`. `functionResponse` sends both `name` and `id`.
- **SSE invalid-UTF-8:** `take_line` surfaces invalid UTF-8 as a protocol error
  instead of silent `U+FFFD` replacement.
- **`StreamStopReason::from_api_str`** accepts `"tool_use"` as alias for
  `"tool_call"` (Anthropic).
- **`set_token_counter`** now propagates to the `ContextManager` if one is set,
  regardless of setter order.
- **Memory injection** uses `Role::User` (not `Role::System`) with explicit
  "reference only" delimitation.
- **Context token estimate** includes the model response message before counting.
- MSRV bumped to 1.94 (let-chain syntax). Removed `parking_lot` dependency.

### Removed

- `StreamHandler::with_config(timeout, retry)` — use `with_timeout_config` /
  `with_retry_config`.
- `FallbackManager::record_api_failure` / `record_model_failure` — merged into
  `record_failure(FailureKind)`.
- `Loop::process_turn`, `BareLoop::run_turn_body` — replaced by machine-driven
  `run()`.
- `StreamTurnResult` — engine assembles result from event stream.
- `StreamHandler::with_request_options` — use `BareLoop::set_request_options`.
- `StreamHandlerError::RateLimitEscalation.prior` field (never read).

### Fixed

- OpenAI streaming dropped multi-chunk tool-call argument fragments after the
  first; re-opened tool-call parts on every chunk carrying `function`.
- `StreamAccumulator` dropped parallel tool calls whose arguments arrived
  interleaved.
- `BareLoop` was permanently dead after a single cancellation (`CancelSignal`
  not re-armed). Now resets in `finalize()`.
- Tool results were split across multiple user messages instead of merged into
  one per turn.
- `StreamHandler` accepted invalid timeout/retry configs (validation never
  called); `jitter_factor` was validated but never applied.
- Empty-stream fast-fail: zero-event streams could hang for ~20 min before
  failing.
- `ToolHealthRegistry::is_tool_available` consumed the HalfOpen recovery probe
  as a side effect of a read.
- Anthropic provider hardcoded text-block index to 0, ignoring server index.
- Per-run manager reset wiped session-scoped state between runs.

### Security

- Auto-commit hook's `git add -A` on empty file list staged the entire working
  tree. Now refuses with an error.
- Non-streaming response body size guard (10 MB) fired after full
  materialization. Now pre-checks `Content-Length` and caps streaming reads.

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

[0.1.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.1.0
