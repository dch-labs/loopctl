# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- `engine::machine::LoopMachine` and supporting types — a sans-IO, serializable
  state machine that owns every agent-loop decision (turn counting, max-turn
  enforcement, tool-call validity, stop-reason routing, compaction trigger,
  history, cancellation). `Serialize + Deserialize`, with no `async`, no
  `tokio`, and no `ApiClient` in its surface. Includes `RunConfig`,
  `MachineStep` (`CallLLM`/`CallTools`/`Compact`/`Done`), `ModelTurn`,
  `PendingToolCall`, `MachineOutcome`, and `MachineState`. `BareLoop` now drives
  a `LoopMachine` internally (`run()` is a `match machine.next_step()` loop); the
  machine is exposed via `BareLoop::machine()` / `into_machine()` /
  `from_machine()` for inspection and serialize-and-resume.
- `LoopMachine::inject(message)` — add an arbitrary message to the machine's
  history (host steering, or `ContextContributor` goal re-injection).
- `Session`/`Run`/`Turn`/`Run` lifetime types (`engine::core`) —
  the Session ⊃ [Run ⊃ [Turn]] hierarchy: one `Session` spans the process,
  one `Run` per `run()` prompt, one `Turn` per loop iteration. `Session`
  derives per-session totals (`total_turns`/`total_duration`/
  `total_input_tokens`/`total_output_tokens`) from its run list. `Run`
  is the result of a `run()`; `Run::turn_count()`/`duration()`/`total_tokens()`.
- `SessionConfig` (`config`) — the session-scoped config slice (`session_id`,
  `system_prompt`, `context_window`) with `with_*` builders, replacing the
  session fields that lived on the old `LoopConfig`.
- `LoopError` now derives `Serialize`, `Deserialize`, `PartialEq`, and `Eq`.
- `compact::types::CompactReason` now derives `Serialize` and `Deserialize`.
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
  Builders: `new`, `with_system`, `with_tools` (both take `Option`).
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
- Fluent `with_*()` builder methods on `LoopConfig` (one per public field,
  e.g. `with_model`, `with_max_turns`, `with_session_id`,
  `with_parallel_tool_dispatch`). Each is `#[must_use]`, consuming, and
  returns `Self` for chaining. Purely additive — `Default`-based and direct
  struct-literal construction are unchanged and remain first-class.
- Consuming `with_*()` fluent builders on `BareLoop`, mirroring the existing
  `&mut self` setters so a loop can be assembled as a chain off `BareLoop::new`:
  `with_reflector`, `with_recovery_strategy`, `with_context_manager`,
  `with_stream_handler`, `with_hook_executor` (`hooks` feature),
  `with_health_registry` (`tool_health` feature), `with_pipeline` (returns
  `Result<Self, LoopError>` — chain with `?`), `with_observer`,
  `with_text_streamer`, `with_contributor`, `with_request_options`. The
  original `set_*`/`register_*` setters are unchanged and remain available.
- `CancelSignal::reset()` — re-arm a fired signal by swapping in a fresh
  underlying `CancellationToken`. Required because the token is one-shot
  by design; once fired it cannot be revived. All clones of an
  `Arc<CancelSignal>` observe the new token, so a handle returned by
  `BareLoop::cancel_signal()` keeps working across resets. `BareLoop`
  calls this in `finalize()` so each `run()` starts with a clean signal.

### Changed

- **Breaking (machine-driven engine):** `BareLoop::run()` is now a
  `match machine.next_step()` loop driving a `LoopMachine`. The machine owns the
  conversation history and every loop decision (turn count, max-turn, tool-call
  validity, compaction trigger, cancellation); the driver owns IO (LLM call,
  tool dispatch, compaction execution) and fires observers from the match-arms.
  Observer event ordering is unchanged (pinned by golden tests). The
  conversation is owned by the machine — read it via `BareLoop::conversation()`
  (delegates to the machine) or `BareLoop::machine().history()`.
- **Breaking (Session/Run lifetime model):** the agent-loop lifetime is now
  explicit. `LoopConfig` is **removed**; construction splits into
  `SessionConfig` (session-scoped: `session_id`, `system_prompt`,
  `context_window`) and `engine::RunConfig` (per-run: turn/token budgets,
  compaction policy, dispatch mode). `SessionResult` is **removed** and unified
  with the new `Run`/`Run` (`engine::core`): the per-run accumulator
  is `Run`-shaped (`turns: Vec<Turn>`, `error: Option<LoopError>`). The `model`
  field is gone from config entirely — it lives on the `ApiClient`
  (`ApiClient::model` / `set_model`). Migration: build `BareLoop::new` with a
  `SessionConfig`; read `session_id`/`system_prompt`/`context_window` from
  `SessionConfig`, run budgets from `RunConfig`, the model from the client.
- **Breaking (`Loop::run` signature + `initialize` removed):** `Loop::run` now
  takes the per-run config — `run(&mut self, user_input: &str, run_config:
  &RunConfig)` — and returns `Result<Run, LoopError>`. Session
  initialization happens once at construction (not per `run()`); each `run()`
  receives a fresh `RunConfig`. `Loop::initialize` and `Loop::config()` are
  removed. Migration: pass `&RunConfig::default()` (or a specific run config)
  as the second `run()` argument; move any `initialize` setup into
  construction.
- **Breaking (compaction thresholds → percentages):** the compaction trigger
  threshold and compaction-target fraction are now `u8` percentages (0–100;
  100 = 100%) instead of `f64` fractions. Affected APIs:
  `ContextManager::with_threshold(u8)` and `with_compact_target_pct(u8)`
  (were `f64`); `ContextManager::threshold() -> u8` and
  `compact_target_pct() -> u8` (were `f64`);
  `SessionConfig::with_compact_threshold(u8)` and the `compact_threshold` field
  (were `f64`). The default is `80` (was `0.80`); the clamp range is
  `[1, 100]` (was `[0.1, 1.0]`). Migration: multiply existing `f64`
  values by 100 and round — `0.80 → 80`, `0.50 → 50`, `0.70 → 70`.
- **Breaking (renames):** consuming builder methods that return `Self` are now
  uniformly prefixed `with_`, matching the crate-wide convention. The old
  no-prefix names are removed. Affected types and methods (old → new):
  `SessionResultBuilder` (`session_id`/`total_turns`/`input_tokens`/
  `output_tokens`/`total_duration`/`tool_calls`/`success`/`final_output`/
  `error` → `with_*`); `ModelSwitch` (`context_window`, `max_tokens` →
  `with_*`); `StreamRequest` (`system`, `system_opt`, `tools`, `tools_opt` →
  `with_*`); `RequestOptions` (`response_format`, `tool_constraint` → `with_*`);
  `UnixShieldBuilder` (`warn_threshold`, `block_threshold`, `pattern`,
  `combination_rule` → `with_*`); `AutoCommitConfigBuilder` (`enabled`,
  `message_template`, `auto_push`, `files` → `with_*`); `ToolPipelineBuilder`
  (`core` → `with_core`; the middleware accumulators `with` and `with_arc` are
  renamed to `with_middleware` and `with_middleware_arc` so they no longer read
  ambiguously next to `with_core`). Migration: add the `with_` prefix at each
  call site.
- `ToolPipelineBuilder::build` now distinguishes its two failure cases:
  `PipelineError::Empty` when neither middleware nor a core was added (the
  builder is untouched), and `PipelineError::MissingCore` when at least one
  middleware was added but no core registry was set. Previously both cases
  returned `MissingCore`, and `Empty` was unreachable. The `Empty` and
  `MissingCore` variants now carry multiline rustdoc explaining each condition.
- **Breaking (optional-field builders unified):** builder methods that set an
  `Option<T>` field now uniformly take `Option<T>` and drop the `_opt` suffix —
  the parameter type already conveys "optional," so the name should not.
  `StreamRequest` loses `with_system_opt`/`with_tools_opt`; `with_system` and
  `with_tools` now take `Option<String>`/`Option<Vec<ToolSchema>>` (pass
  `Some(…)` to set, `None` to clear). `LoopConfig::with_system_prompt` changes
  from `impl Into<String>` to `Option<String>`, so it can now clear the
  override with `None` (previously it could only set). `AttemptRecord::with_reason`
  and `Operation::with_result_hash` already followed this shape and are
  unchanged. Migration: wrap existing literal/value arguments in `Some(…)` (or
  rename `with_*_opt` → `with_*`).
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

- `Loop::process_turn` trait method and `BareLoop::run_turn_body` — the
  machine-driven `run()` replaces the old per-turn execution path. The
  `LoopMachine` is the new turn unit; drive it via `BareLoop::run()` (or
  `LoopMachine::next_step()` directly for a custom driver).
- `StreamTurnResult` (the handler no longer accumulates; the engine assembles
  the result from the event stream).
- `StreamHandler::with_request_options` builder (options now flow via
  `stream_turn`'s parameter; configure via `BareLoop::set_request_options`).
- `StreamHandlerError::RateLimitEscalation.prior: StreamOutcome` field (never
  read by any consumer).

### Fixed

- OpenAI streaming silently dropped every multi-chunk tool-call argument
  fragment after the first, truncating the tool input JSON. The
  deserialization structs declared `id` (on the tool-call delta) and
  `name` (on the function object) as required `String` fields, but the
  real OpenAI streaming protocol omits both on continuation chunks —
  those carry only `index` and an `arguments` fragment. Serde rejected
  those chunks with `missing field`, `OpenAiChunk::parse` returned
  `None`, and the stream loop silently skipped them, leaving the
  accumulated JSON incomplete. Both fields are now `Option<String>` with
  `#[serde(default)]`; the emitter latches `id` and `name` on the first
  chunk for each `index` and ignores them on continuations.
- OpenAI streaming re-opened a tool-call part on every chunk carrying a
  `function` field. Because every chunk (including argument-fragment
  continuations) carries `function`, the `StreamEmitter` emitted
  `PartStart` for each one, which wiped the downstream accumulator's
  buffered JSON. The emitter now tracks each tool call's `index` and
  emits `PartStart` exactly once per call.
- `StreamAccumulator` dropped parallel tool calls whose argument fragments
  arrived interleaved (the shape OpenAI streams for
  `parallel_tool_calls`). The accumulator tracked a single in-progress
  part, so a second `PartStart` arriving before the first `PartStop`
  overwrote the first call's buffer, and `IndexedDelta` fragments whose
  `index` did not match the single current slot were silently dropped.
  It now holds a `Vec` of open slots keyed by `index`, routes each delta
  to the matching slot, and flushes slots in `PartStart` arrival order
  (FIFO) on `PartStop`. Anthropic (strictly sequential) and Gemini
  (atomic per-chunk tool calls) are unaffected.
- `BareLoop` was permanently dead after a single cancellation. Once
  `cancel()` fired, the `CancelSignal` (a one-shot
  `CancellationToken`) stayed cancelled forever, so every subsequent
  `run()` returned `LoopError::Cancelled` immediately. `run()` now
  re-arms the signal in `finalize()` — the single chokepoint every run
  exit path passes through — so the next `run()` starts clean. A cancel
  that arrives *before* a run still cancels that run (the signal is
  cleared only after the run observes it and returns), preserving the
  pre-run-cancel contract.

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
