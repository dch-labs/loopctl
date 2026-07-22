# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- `DeltaPart::Thinking { text }` variant + `on_thinking_delta` observer event
  (`ThinkingDeltaContext`): reasoning-model tokens (Claude extended-thinking,
  DeepSeek-R1, OpenAI o-series, Gemini 2.5+) are now routed as their own
  stream kind instead of being dropped or misrouted into text. Stream-only —
  reasoning is not accumulated into the `Message` and does not reach
  `ResponseContext.text`; consume it via `on_thinking_delta` or the raw
  `IndexedDelta(Thinking)` stream event. Anthropic parses `thinking_delta`
  (and emits an empty Thinking delta for `redacted_thinking`); OpenAI parses
  `reasoning_content` (aliased to `reasoning`); Gemini parses per-part
  `thought: true` flags. An empty `delta` signals redacted reasoning (render
  a placeholder). For Gemini, `GeminiClientBuilder::include_thoughts(true)`
  opts into `generationConfig.thinkingConfig.includeThoughts` on the request
  side — opt-in because the Gemini API rejects `thinkingConfig` with
  `400 INVALID_ARGUMENT` on non-reasoning models; defaults to `false`.
- `DisplayHint` advisory rendering hint on `ToolOutput` (`with_hint()` builder),
  threaded through `ToolDispatchResult` and `ToolPostContext` so presentation
  layers (TUI, headless console) can render a tool result by the tool's own
  declaration instead of inferring the strategy from the tool name. Six
  variants: `Text`, `Diff`, `Json`, `Code { language }`, `Suppress`,
  `Markdown`. Advisory only — compaction, loop-detection hashing, and loop
  semantics are unaffected (the hint terminates at the observer context and
  never enters the message model).
- `ConstrainedProfile`, `FrontierProfile`, and `GoalReminder` (`presets`
  module): a named small-model-tuned runtime profile and its frontier opt-out
  counterpart. `ConstrainedProfile::apply(&mut loop_)` wires the small-model
  middleware stack (verify with `NoopVerifier`, memoize with
  `NoopPathExtractor`, output cap) and registers a `GoalReminder` contributor;
  `loop_config()` returns a tighter `LoopConfig` (120k window, 100 turns) and
  `request_options()` returns `tool_constraint: Strict`. Compose the pieces
  individually via `pipeline_builder()` / `loop_config()` / `request_options()`.
- `NoopVerifier` (`middleware::verify`): a `Verifier` that always passes,
  co-located with the `Verifier` trait. Default verifier for
  `ConstrainedProfile`; swap in a real build/lint step when available.
- `NoopPathExtractor` (`middleware::memoize`): a `PathExtractor` that extracts
  no paths, disabling path-based cache invalidation (TTL-only caching).
  Default extractor for `ConstrainedProfile`.
- `BareLoop::set_request_options(opts)` builder: set the per-turn
  `RequestOptions` (carrying `tool_constraint`) applied to every provider call.
  Default is `RequestOptions::default()` (no constraint), reproducing prior
  behavior.
- `StreamHandler::passthrough()` constructor + `passthrough_default()` — a
  no-resilience handler (no retries, no timeouts, no fallback) used as the
  engine default when no handler is configured.
- `HandlerEvent` enum (`stream::handler`): events yielded by the new
  stream-based `StreamHandler::stream_turn`. Variants: `Stream(StreamEvent)`
  for raw provider events, `AttemptReset` on retry, `Fallback { message,
  stop_reason }` on non-streaming fallback.
- `StreamRequest` struct (`api`): bundles `(messages, system, tools)` into a
  single parameter for `ApiClient` methods. Replaces the positional
  `(Vec<Message>, Option<String>, Option<Vec<ToolSchema>>)` parameter lists on
  `stream_messages`, `create_message`, and their `_with_options` variants.
  Builders: `new`, `system`, `system_opt`, `tools`, `tools_opt`.
- `GeminiClientBuilder::include_thoughts(bool)` builder: opt into Gemini's
  `thinkingConfig.includeThoughts` for reasoning-capable models (2.5 Pro/Flash,
  Gemini 3). Defaults to `false` — the Gemini API rejects `thinkingConfig`
  with `400 INVALID_ARGUMENT` on non-reasoning models, so the caller must opt
  in once they know their model supports thinking.
- HTTP connection-pool injection and tuning on all three provider builders
  (`OpenAiClientBuilder`, `AnthropicClientBuilder`, `GeminiClientBuilder`):
  `.http_client(reqwest::Client)` injects a shared client so multiple providers
  can reuse one connection pool. `.pool_max_idle_per_host(usize)`,
  `.pool_idle_timeout(Duration)`, and `.tcp_keepalive(Duration)` expose the
  underlying `reqwest` pool knobs (default to reqwest's built-in defaults when
  unset). When an injected client is used, these knobs and `.timeout()` /
  `.connect_timeout()` are ignored — configure them on the injected client.
- `ContextContributor` trait and `ContributorContext<'a>` (`engine::contributor`
  module): a write-side hook at the turn boundary. Implementors return an
  optional `Message` that the loop appends to the conversation before the next
  model call. Register on `BareLoop` via `add_contributor`; with no contributors
  registered, the loop behaves identically to before (the turn-top consultation
  is a single cheap branch).
- `Role::System` variant on `message::Role`. Serialized as `"system"`. Used for
  framework-injected context such as a turn-boundary reminder. Providers map it
  to their native system representation: an inline `{role: "system"}` message on
  OpenAI, or the top-level system field on Anthropic and Gemini (which do not
  accept an inline system role mid-conversation).
- `ApiError::rate_limited(message, retry_after)` constructor for the structured
  rate-limit carrier variant.
- `LoopError::RateLimitEscalation { attempts, retry_after }` variant,
  recoverable, raised when the stream handler exhausts rate-limit retries on a
  model and escalates to the circuit breaker.
- `StreamHandlerError::RateLimitEscalation { attempts, retry_after }`
  variant. After `RateLimitConfig::fallback_after_retries` rate-limit retries,
  `stream_turn` yields this as a stream error instead of looping
  indefinitely or falling back to the same model's non-streaming endpoint.
- Rate-limit backoff sleeps are now clamped to the turn's `total_stream_timeout`
  so a large `Retry-After` cannot overrun the turn budget.
- The engine routes `RateLimitEscalation` to
  `FallbackManager::record_model_failure`, so a sustained rate limit on one
  model trips the circuit breaker and subsequent turns route to the fallback
  model.
- `TokenBucket` and `RateLimiter` (`stream::rate_limit`): a proactive
  client-side token-bucket rate limiter, one bucket per provider `base_url`.
  Attaches to `StreamHandler` via `with_rate_limiter`; gates each stream
  attempt before it fires, with a `max_wait` ceiling that degrades to reactive
  (proceed, risk the 429) rather than hang.
- `ApiClient::base_url()` trait method (default `""`), overridden by the
  OpenAI, Anthropic, and Gemini clients to expose their configured endpoint for
  per-provider bucket keying.
- `ParallelMode` + `ParallelDispatchConfig` in `config.rs`: opt-in parallel
  tool dispatch for independent, concurrency-safe calls within a single turn.
  `LoopConfig` is now `#[non_exhaustive]`; the new `parallel_tool_dispatch`
  field defaults to `Sequential` (v0.1.0 behaviour unchanged).
- `Tool::resource_key(&self, &Value) -> Option<String>` trait method (default
  `None`) for parallel-dispatch resource-conflict detection, plus the
  `FnTool::with_resource_key` builder.
- `MockTool::with_delay(Duration)` builder for timing-sensitive tests.
- `StructuredOutput` trait, `ResponseFormat`, `RequestOptions`, and
  `StructuredError` (`structured` module): request guaranteed-schema JSON
  responses from the model. Includes a lenient JSON extraction helper (handles
  markdown fences/prose prefixes) and a `request_structured::<T>()`
  convenience function.
- `ApiClient::stream_messages_with_options` and
  `create_message_with_options` default methods (additive — existing impls
  compile unchanged). `OpenAiClient`, `AnthropicClient`, and `GeminiClient`
  override both to inject the schema (OpenAI via native `response_format`,
  Anthropic via forced-tool tool-forcing, Gemini via `generationConfig`
  `responseMimeType` + `responseSchema`).
- `ToolOutput::structured<T>`, `structured_value()`, and
  `structured_as::<T>()` for typed tool results that round-trip through JSON.
- `ToolConstraint` enum (`structured` module, `#[non_exhaustive]`,
  default `None`) and `RequestOptions::tool_constraint` field + builder for
  constraining the model's tool-call output to the registered tool schemas.
  `ToolConstraint::Strict` makes malformed tool calls structurally impossible
  via the provider's native strict-tool mode; `ToolConstraint::Grammar`
  (requires the new `grammar` feature) compiles the schemas into a grammar
  for a grammar-aware sampler (vLLM `guided_json`).
- `ToolGrammarProvider` trait and `JsonSchemaGrammar` default impl
  (`provider::grammar` module, `grammar` feature): the extension point for
  compiling a tool registry's schemas into a sampler grammar.
- `grammar` feature flag (depends on `providers`): opt-in grammar / sampler
  support for the `Grammar` mode of `ToolConstraint`.
- `LlmReflector` (`reflection::llm` module): a `Reflector` that asks the
  model to classify failed tool calls and suggest corrections via
  `request_structured::<FailureAnalysis>`. First in-tree consumer of
  `StructuredOutput`. Opt-in via `BareLoop::set_reflector`; the default
  stays `NoopReflector`. Each analyzed failure triggers one model
  round-trip (see its rustdoc for the latency/cost note).
- `impl StructuredOutput for FailureAnalysis` (`reflection` module) with a
  hand-written JSON Schema covering the 5 fields and the nested
  `CorrectionType` snake_case enum.
- `schema_validation` feature flag (pulls `jsonschema` as an optional
  dependency): when enabled, `LlmReflector` validates the model's
  `Correction::modified_input` against the failing tool's `input_schema`
  and returns `ReflectionError::Internal` on a mismatch. When disabled,
  validation is skipped.
- `VerifyMiddleware`, `Verifier` trait, and `VerifyResult`
  (`middleware::verify` module): opt-in post-execution middleware that
  runs a caller-supplied verifier after configured write-class tools and
  appends the pass/fail + diagnostics to the `ToolOutput` so the next
  turn sees it. The verifier impl (`cargo check`, `tsc`) is
  domain-specific and supplied by the consumer; loopctl ships only the
  trait and the middleware.
- `MemoizingMiddleware` and `PathExtractor` trait
  (`middleware::memoize` module): opt-in middleware that caches
  successful tool-call results keyed on `(tool_name, hash(canonical
  input))` and returns cached output with a `[cached]` marker on hit.
  Path-aware invalidation via the caller-supplied `PathExtractor` trait;
  TTL-based expiry per `ttl_turns`. Default-off; only successful results
  are cached.

### Changed

- **Breaking:** `stream::DeltaPart` is now `#[non_exhaustive]`. Every
  exhaustive `match` on `DeltaPart` in downstream code must add a `_ =>` arm
  (or a `Thinking =>` arm for the new variant). Same-crate matches are
  unaffected. Future variant additions (e.g. `Image`, `Audio`) will arrive
  non-breaking.
- **Breaking:** `ApiClient::stream_messages`, `stream_messages_with_options`,
  `create_message`, and `create_message_with_options` now take a single
  `StreamRequest` parameter instead of positional `(messages, system, tools)`.
  Every `impl ApiClient` must update its signatures.
- **Breaking:** `StreamHandler::stream_turn` now returns
  `impl Stream<Item = Result<HandlerEvent, StreamHandlerError>>` instead of
  `Future<Result<StreamTurnResult, _>>`. Callers must drive the stream and
  accumulate events themselves. The engine's `stream_turn` does this internally
  and fires observer callbacks (`on_text_delta`, `on_thinking_delta`,
  `text_streamer`) per event — configuring a `StreamHandler` for resilience no
  longer drops real-time observability.
- **Breaking:** `StreamCapable::stream_handler` now returns `&StreamHandler`
  (not `Option<&StreamHandler>`). When no handler is configured, returns a
  shared `StreamHandler::passthrough_default()` (no-resilience default).
- **Breaking:** `StreamHandlerError::RateLimitEscalation` lost its `prior`
  field. The variant is now `{ attempts, retry_after }`.
- **Breaking:** `StreamHandler::stream_turn` now takes `options:
  RequestOptions` as an explicit parameter. `StreamHandler::with_request_options`
  is removed — use `BareLoop::set_request_options` instead.
- **Breaking:** All three provider builders (`OpenAiClientBuilder`,
  `AnthropicClientBuilder`, `GeminiClientBuilder`) now use `with_` prefix on
  consuming builder methods (e.g. `.with_api_key()`, `.with_model()`,
  `.with_timeout()`, `.with_http_client()`). The old no-prefix names
  (`.api_key()`, `.model()`, etc.) are removed.
- **Breaking:** `ToolOutput`, `ToolDispatchResult`, and `ToolPostContext` are
  now `#[non_exhaustive]`, matching `DisplayHint`. Downstream code that
  constructs these via struct literal must switch to the named constructors
  (`ToolOutput::text`/`success`/`error`/`error_text`, `ToolDispatchResult::ok`/
  `err`/`from_tool_output`/`from_result`, or the `From<ToolOutput>` impl) or
  add `..Default::default()` where a `Default` exists. Same-crate construction
  is unaffected. Future fields on these types will now arrive non-breaking.
- The engine now calls `ApiClient::stream_messages_with_options` instead of
  `stream_messages` on every turn (both the inline-streaming path in
  `engine::bare::stream` and the `StreamHandler` path), passing the loop's
  `RequestOptions`. This is additive — the default `RequestOptions::default()`
  has no `response_format` and `tool_constraint: None`, which reproduces the
  prior behavior exactly. Custom `ApiClient` impls that do not override
  `stream_messages_with_options` inherit the trait default (which delegates to
  `stream_messages` when `response_format` is `None`), so they continue to
  work; a `tool_constraint` other than `None` set via `set_request_options`
  only takes effect on clients that override `_with_options` (the built-in
  OpenAI, Anthropic, and Gemini clients do).
- **Breaking:** `message::Role` gains a `System` variant. Every exhaustive
  `match` on `Role` in downstream code must add a `System =>` arm (or a `_ =>`
  wildcard). Migration: add `Role::System => /* your mapping */` to each match,
  or switch to a wildcard arm. The three built-in providers are already
  updated; the Anthropic and Gemini serializers fold any inline `Role::System`
  messages into the top-level system request field (`system` / `systemInstruction`)
  rather than emitting them inline, because those providers accept system content
  only as a top-level field. When both a caller-supplied system prompt and inline
  `Role::System` messages are present, they are joined (caller prompt first,
  folded text appended, newline-separated). OpenAI emits `Role::System` as an
  inline `{role: "system"}` message, its native form.
- **Breaking:** `Reflector::analyze` gains a new `tool_schema:
  Option<&ToolSchema>` parameter between `tool_input` and `context`. The
  engine's call site now resolves the failing tool's schema from the
  registry (passing `None` when the tool isn't found). Every `Reflector`
  impl must add the new parameter; `NoopReflector` and the trait-doc
  example have been updated.
  Migration: add `_tool_schema: Option<&loopctl::tool::ToolSchema>` to
  your `analyze` signature. Ignore it if your reflector does not validate
  suggested corrections; otherwise use it to validate `modified_input`
  before returning the analysis.
- `OpenAiClient`, `AnthropicClient`, and `GeminiClient` now honor
  `RequestOptions::tool_constraint`. Under `Strict`, each tool's schema is
  tightened (recursive `additionalProperties: false` and full `required`);
  OpenAI additionally sets `strict: true` on each `function` entry. Under
  `Grammar`, the OpenAI client injects `guided_json` for vLLM-style
  grammar-aware samplers. Default `ToolConstraint::None` reproduces prior
  behaviour exactly; when `response_format` is also set, it wins and
  `tool_constraint` is ignored.
- Removed `parking_lot` dependency entirely. All `parking_lot::Mutex`
  usages migrated to `std::sync::Mutex` with the
  `.unwrap_or_else(std::sync::PoisonError::into_inner)` recovery pattern.
  The crate now uses a single mutex family with no external lock
  dependency.
- MSRV bumped from 1.85 to 1.94. The crate uses let-chain syntax
  (`if x && let Some(y) = ...`) which stabilized in Rust 1.88.
- Both sequential and parallel tool dispatch now check the cancel signal
  between calls. Previously, a Ctrl-C during a multi-tool batch was only
  honored at the next turn boundary; now it aborts the remaining calls in the
  batch.
- Internally-built `reqwest::Client`s now set `tcp_nodelay(true)` by default.
  SSE streaming emits many small packets; disabling Nagle's algorithm reduces
  per-delta latency. No correctness impact.

### Removed

- `StreamTurnResult` (the handler no longer accumulates; the engine assembles
  the result from the event stream).
- `StreamHandler::with_request_options` builder (options now flow via
  `stream_turn`'s parameter; configure via `BareLoop::set_request_options`).
- `StreamHandlerError::RateLimitEscalation.prior: StreamOutcome` field (never
  read by any consumer).

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
