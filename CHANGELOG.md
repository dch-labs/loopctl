# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- `LoopMachine::compaction_noop(tokens_before, tokens_after)` — a second compaction feed method for passes that changed nothing (no compactor ran, a pre-compact hook vetoed, or the compactor returned the conversation unchanged). Unlike `compaction_result`, it leaves the committed history and the pending buffer untouched and adopts `tokens_after` as the current context size. Both compaction feeds compare the driver's measured pre-pass and post-pass token counts: when nothing was shaved off, the machine terminates the run with `ContextExceeded` instead of trusting an estimate recorded at compaction-request time (which could be stale — set before the last turn's tool results, or zero at run start). Alt drivers servicing `MachineStep::Compact` should pass both measurements from one counter.
- `McpToolProvider::with_call_timeout(Duration)` — a per-call budget for every adapted tool's `tools/call` round-trip (default 60s). A call that exceeds it resolves to a *soft* error result naming the tool and the budget, so a wedged MCP server costs one tool result instead of hanging the agent loop indefinitely. Updates already-discovered tools and applies to later refreshes. Pinned by `call_timeout_cuts_a_slow_tool_with_a_soft_error` / `default_timeout_lets_a_quick_tool_through` in `tests/mcp_tool_provider.rs`.
- MCP server adapter (`McpServerAdapter`) — serve a loopctl `ToolRegistry` over MCP (stdio), consumable by any MCP client. `McpServerAdapter::new(registry, ctx, name, version)` + `serve_stdio()` implements `ServerHandler` (`list_tools` → `all_schemas`, `call_tool` → `Tool::call`). Served calls are raced against the request's cancellation token: an already-cancelled request resolves to a cancelled result without invoking the tool, and a client cancel (`notifications/cancelled`) or disconnect drops the in-flight tool future and resolves to a cancelled tool-level result — a wedged tool no longer leaks a task per call (tools must be cancellation-safe, the same contract the engine's dispatch path imposes). `list_tools` forwards each tool's `is_read_only` as the MCP `annotations.readOnlyHint` (with `destructiveHint: false`); tools whose input schema does not compile as JSON Schema — malformed keywords, uncompilable regexes, dangling or external `$ref`s (external references are refused, never fetched) — or is not object-typed are omitted from the listing with a warning rather than advertised with a schema strict clients may reject (the `mcp` feature now pulls `jsonschema` for this check); unknown tool names return a `METHOD_NOT_FOUND` protocol error listing the registered names; empty descriptions are omitted rather than sent as `""`. Tool names are forwarded verbatim — the MCP spec recommends `^[a-zA-Z0-9_-]{1,64}$` and conforming names are the embedding application's responsibility (documented in the module docs). Transport-agnostic: `serve(impl IntoTransport)` works for future HTTP/SSE. The module promoted to `mcp/{convert,server}.rs` with symmetric inbound/outbound converters co-located. Example at `examples/mcp_server.rs` (echo + failing tool, Ctrl-C → graceful cancel; includes a piped JSON-RPC acceptance recipe).
- MCP transports (stdio + Streamable HTTP/SSE) — `McpClient::stdio(command)` spawns an MCP server as a child process over stdio; `McpClient::http_sse(endpoint)` and `McpClient::http_sse_with_client(endpoint, reqwest::Client)` connect via the Streamable HTTP transport (rmcp handles `Mcp-Session-Id`, JSON-vs-SSE response splitting, and DELETE-on-close). `McpClient::reconnect(&StreamRetryConfig)` re-establishes a dropped connection using the crate's existing backoff strategy (the one retry strategy for the crate, not a second one). New public `CommandSpec` describes what to spawn. A stdio example server ships at `examples/mcp-stdio-server.rs` for transport testing. The `mcp` feature now enables `streaming` (for `StreamRetryConfig`) and `reqwest` (for the HTTP client); `default = []` is unchanged.
- `mcp` feature + `loopctl::mcp` module — adapt any MCP server's tools as loopctl `Tool` implementations. New public types: `McpClient` (a connected client handle), `McpToolProvider` (discovers a server's tools and registers them into a `ToolRegistry`), `McpTool` (one server tool as a `Tool`), `McpError`. Real-server transports ship alongside (stdio child processes and Streamable HTTP/SSE — see the transports entry above); `McpClient::in_process` connects an in-process rmcp server for tests and bundled-server use. The optional `rmcp` dependency is pulled in only by the `mcp` feature (`default = []` is unchanged). A runnable end-to-end demo ships at `examples/mcp-adapter.rs` (`cargo run --example mcp-adapter --features mcp`).
- `LoopError::ToolRecoveryExhausted { tool, attempts }` — the driver now enforces `MAX_RECOVERY_ATTEMPTS` (5) as a hard ceiling. A recovery strategy that always returns `Retry` is stopped after 5 retries (attempt 6), returning this variant instead of looping forever. Pinned by `recovery_ceiling_stops_retry_forever_strategy`.
- `RunConfig::memory_top_k` — configurable number of memory entries retrieved and injected per turn (default 3; was a hardcoded magic number).
- `MachineStep::CallTools { turn, calls }` — the machine now emits the 0-indexed turn number on `CallTools` (matching `CallLLM`), so both handlers source the turn identically from the machine rather than one reading a field and the other querying a counter.
- `parallel_hard_error_discards_sibling_results` test — pins the documented contract that a hard error in a parallel wave aborts the batch and discards already-completed sibling results.

### Changed

- `BareLoop` constructors now seed a default `ContextManager` (a `TruncatingCompactor` with window and threshold synced from the session config) when the supplied manager bundle carries none: `new`, `new_with_managers` (only when the bundle has no context manager), and `from_machine`. Behavior change: default-configured loops with `auto_compact` on now actually compact at the threshold (observer-visible via `on_compaction`) instead of growing unbounded; a compaction pass that cannot reduce the history terminates the run with a typed `LoopError::ContextExceeded` rather than silently resetting the estimate and sending over-window conversations. `auto_compact: false` still disables threshold compaction, and hosts installing their own `ContextManager` are unaffected. Pinned by `tests/compaction_noop.rs` (`over_limit_context_is_never_sent_to_the_provider`, `default_loop_compacts_at_the_threshold`, `host_installed_context_manager_is_unaffected`).
- `ConstrainedProfile::apply` attaches a `ContextManager` synced from the loop's session config (replacing whatever the constructor seeded), so the profile's context budgeting is enforced machinery rather than a marketing claim. Pinned by `small_model_profile_compacts_at_the_threshold` in `tests/compaction_noop.rs`.
- `ContextManager::compact_with_reason` (and `compact_manual` / `ensure_context_fits`) now report a successful pass that shrinks neither the message list nor the token count as `EnsureContextResult::NoAction` instead of `Compacted`. Classification and the returned token fields use the **manager's configured counter**, not the compactor's self-report: `tokens_after`/`tokens_saved` on the `Compacted` outcome are normalized to the manager's measurements, and `compact_with_reason`'s overflow check re-counts the result with the same counter (previously it trusted the compactor-reported value). Callers matching `Compacted` to learn "compaction occurred" no longer see no-action passes; `on_compaction` observers and post-compact hooks stay silent for them (the engine already skips both on `NoAction`). Migration: code that treated any `Ok(Compacted(..))` as "the messages may have changed" should use `into_messages()` — the returned list is identical under `NoAction`.
- **Breaking:** `LoopMachine::compaction_result` now takes `(compacted, tokens_before, tokens_after)` — the compacted history plus the driver's measured full-history size ahead of the pass and the compacted size after it, replacing the estimate the machine used to record at compaction-request time. The machine's `last_compaction_tokens` field is gone (its serialized state changes accordingly for checkpoints). Migration: pass the two measurements from the same counter the driver uses for its context estimate; the no-progress guard fails the run when `tokens_after >= tokens_before`.
- Provider HTTP clients now set a **read timeout** (maximum gap between response bytes) instead of a total request timeout, and `with_timeout` configures that read timeout. A total HTTP-layer cap aborted every SSE stream longer than the configured duration (default 2 minutes) — pre-empting the `StreamHandler`'s per-event/total-stream deadlines and the engine's turn timeout, which own generation-length budgets. Long healthy streams now run as long as they keep producing bytes; a server silent for the configured gap (default 120s) is still aborted. Behavior change: generations longer than the old total cap no longer fail at the HTTP layer.

- **Breaking:** Machine turn indices are now 0-indexed (`CallLLM { turn: 0 }` for the first turn). Previously 1-indexed (`turn: 1`). `AwaitingModel { turn }` and `AwaitingTools { turn }` follow the same convention. Callers matching on these variants in tests or drivers must adjust.
- **Breaking:** `LoopError` gains `ToolRecoveryExhausted` variant. Exhaustive matches on `LoopError` must add this arm.
- **Breaking:** `RunConfig` gains `memory_top_k` field. Struct-literal construction must add it (use `..Default::default()` or the builders).
- **Breaking:** `MachineStep::CallTools` gains a `turn` field. Pattern matches must update.
- `engine/bare.rs` decomposed from 6,745 lines into a ~1,200-line facade plus focused submodules: `llm_turn` (both LLM-turn paths + shared request builder), `config` (set_*/with_* builders), `emission` (all observer/hook fan-out centralized), `model_switch`, and `tests`. Each `MachineStep` arm now maps to exactly one submodule.
- Streaming and non-streaming LLM-turn paths merged into `llm_turn.rs` with a shared `build_turn_request`, eliminating the duplicated request-construction block.
- Observer fan-out centralized in `emission.rs` — every `on_*` event family (run, turn, tool, stream, compaction, fallback) now lives in one module instead of scattered across five files.
- `MachineOutcome::to_loop_error` — canonical outcome→error translator in `core/outcome.rs`, replacing three duplicated mapping sites.
- `RecoveryDecision` enum replaces `Result<(u32, Option<Correction>), RecoveryOutcome>` — three clear variants (`Retry`, `Soft`, `Cancelled`) instead of a `Result` where `Err` meant "not an error."
- `TurnAccounting` struct bundles the turn start + token pair forwarded through `dispatch_and_record`, shrinking the signature from 5 positional params to 3.
- `dispatch_tool` and `dispatch_via_pipeline` return `ToolDispatchResult` directly (were `Result<ToolDispatchResult, LoopError>` but never returned `Err`).
- `apply_loop_detection` no longer calls `set_error_state` — the single `set_error_state` in `run()`'s error path handles all terminal-state transitions, eliminating the double-invocation.
- Tool-dispatch turn-accounting lookup keyed on `current_turn` explicitly (was `turns.last()` positional), so a future reorder produces clean `(0, 0)` rather than the wrong turn's tokens.
- `token_counter` single-source: `set_context_manager` no longer syncs its counter onto the driver field; `count_context()` prefers the manager's counter, falling back to the driver field only when no manager is configured.
- `millis_u64` unified across `emission.rs` and `compact.rs` (was duplicated with different overflow fallbacks: `u64::MAX` vs `0`).
- `current_run`/`current_run_mut` wrappers removed; callers delegate to `Session`'s existing methods.
- Loop-detection decision logic (`decide_detected_pattern`, `apply_loop_detection`) moved to `llm_turn.rs` (response-side), separating it from tool-operation detection (`pre_detection`/`post_detection`) in `dispatch.rs`.

### Fixed

- No-op compaction passes no longer report a hard-coded zero estimate or commit the in-flight run's pending messages. Previously the driver's no-manager, pre-compact-hook-veto, and `NoAction` paths returned `tokens_after = 0`, which blinded the machine's no-progress guard and reset its context estimate — the loop kept calling the provider with an over-window conversation until the turn budget was spent. Worse, feeding the uncompacted history back through `compaction_result` committed the current run's partial messages mid-run, so a later failure leaked the aborted run's prompt, tool calls, and results into committed history forever (`discard_pending` could no longer undo it). These paths now return the measured estimate (`count_context`) through an explicit no-op signal that leaves pending untouched; when compaction genuinely cannot reduce, the run terminates with a typed `LoopError::ContextExceeded`. Pinned by `failed_run_after_noop_compaction_leaves_history_clean` (engine) and `pre_compact_hook_veto_reports_measured_estimate` (hooks) — the hook-veto pass now surfaces the real size instead of zero.
- A no-change compaction pass is no longer reported as `EnsureContextResult::Compacted`: `ContextManager` used to wrap any successful compactor outcome as `Compacted`, so short-but-over-threshold conversations fired `on_compaction` observers (and post-compact hooks) on every triggering turn with zero savings and identical messages, violating both contracts. Same-list/zero-savings outcomes now map to `NoAction`. Pinned by `no_change_pass_is_not_reported_as_compacted` (compact.rs).
- The non-streaming fallback forwards the turn's `RequestOptions` and is bounded by the total-stream deadline (`StreamHandler::fallback_non_streaming`): it previously called `create_message` (dropping any configured `response_format`/`tool_constraint`) and raced only the cancel signal, so a hanging fallback request hung the turn. Pinned by `fallback_non_streaming_forwards_request_options`, `fallback_non_streaming_honors_the_total_deadline`, and `completed_fallback_response_racing_the_deadline_is_accepted` (the deadline bounds waiting, not completion — a resolved response is accepted even past expiry, while a hanging one is still cut).
- MCP `tools/call` round-trips are bounded by the new per-call timeout (see `McpToolProvider::with_call_timeout` above) — previously a wedged server hung the agent loop with cancellation as the only exit.
- Tool-result parts in a `CallTools` turn preserve **model request order** across preresolved (unknown-tool) and dispatched calls. Previously the turn's results were assembled as `[all preresolved, then all dispatched]`, which reordered the parts the model saw relative to the calls it made. Provider-safe in practice (providers match by `tool_call_id`, not position), but order-non-preserving and surprising to hosts that assume positional alignment. Pinned by `test_mixed_known_unknown_tools_preserve_request_order`.
- The `run()` `Done` arm now matches every `MachineOutcome` variant explicitly (`Completed`, `MaxTurnsExceeded`, `Cancelled`, `Failed`) instead of using a wildcard `other => ... unwrap_or(Cancelled)` fallback. `MachineOutcome` is `#[non_exhaustive]` but defined in this crate, so the compiler proves this exhaustive — a future variant forces a compile error here rather than being silently mislabelled as `Cancelled`.
- `handle_call_tools` doc corrected to state where cancellation is actually honored (the in-flight tool call is raced against the cancel signal in `execute_tool_call`'s `select!`; the sequential path checks the signal between calls), instead of claiming a `select!` that does not exist in the function itself.
- `set_token_counter` doc corrected — no longer claims a sync with `ContextManager` that the code doesn't perform. The driver field is documented as a fallback used only when no manager is configured.
- `ModelSwitch` doc corrected — removed stale "max-tokens" reference (the builder only has `context_window`).
- `MAX_RECOVERY_ATTEMPTS` doc rewritten — states the one-knob design (strategy sees the same ceiling the driver enforces) instead of implying two independent limits.
- `dispatch_tools_parallel` doc now documents hard-error semantics: a hard error from any call in a wave discards sibling results.

## [0.2.1] - 2026-08-04

### Fixed

- Detection-layer long-tail correctness (six items): `check_file_reads` now normalizes the query under the recorded op's real tool name (was: empty string) and matches by bidirectional path containment (was: exact equality), and rejects empty normalized params (an empty string is a substring of every query, which would inflate the read count); `LoopDetector::new` clamps `window_size == 0` to 1 with a warning; `find_best_match` rejects zero-score candidates even at threshold 0.0 and breaks ties lexicographically; `ToolShield::with_thresholds` swaps inverted warn/block pairs with a warning and rejects non-finite (NaN/inf) values, falling back to the band's default — a stored NaN would silently disable the band because every `score >= NaN` comparison is false. All non-breaking.
- `SessionConfig::compact_threshold` now clamps into its documented `0..=100` range on the validating construction paths (`Default`, `with_compact_threshold`, and `Deserialize`). Previously only the `with_compact_threshold` builder clamped; deserialized configs could carry an out-of-range value (e.g. `200` from disk), which the compaction subsystem would then interpret as "never compact." A single canonical clamp method plus a field-level serde deserialize helper now enforce the range silently. `Default` was already in range (80); its clamp call is defensive. Direct public struct-literal construction (`SessionConfig { compact_threshold: 200, .. }`) still bypasses normalization — the field is `pub`, and callers using that form are responsible for honoring the documented range. Non-breaking (silent clamp; no signature changes).
- `TokenBucket` refill clock no longer jumps to a future instant. The shared `elapsed_refill` helper advances `last_refill` only to the fill point (the instant capacity is reached) instead of to `at`, so a caller that passes a far-future instant — directly or via `take`/`acquire`/`available` — can no longer freeze the refill clock on the production rate-limit path. Non-breaking (callers passing correct `Instant::now()` values see no change).
- Corrected the `ParallelMode::Parallel` doc, which falsely claimed detection/observer side-effects "are not thread-safe" and fire "once on the final result only" in parallel mode. They are thread-safe (`DetectionManager` and the observer registry use `Mutex`/immutable-`Vec` interiors; `ToolHealthRegistry` uses atomics) and fire on every retry attempt in both modes, exactly as the code already does. No behavior change; the code matched the corrected doc all along.
- Fixed code-level doc contradictions: the `ApiClient` trait example showed `request: StreamRequest` (by-value) instead of `&StreamRequest` (matches the real trait), and `BareLoop::machine` was described as an "empty placeholder" rather than the real "empty machine (no history, no pending messages)".
- Reconciled the planning docs (ROADMAP, CONTEXT, ARCHITECTURE, README, DEPENDENCIES, DCH-DESIGN, the v0.2.0 release file) to the shipped 0.2.0 reality: status Planned→Shipped, `compact_threshold` u16→u8, `Loop::process_turn` soft-deprecated→removed, `LoopRuntime`/`LoopConfig`/`SessionResult`/`run_session` → their shipped replacements (`managers`/`SessionConfig`+`RunConfig`/`Run`+`Session`/`run`), MSRV 1.85→1.94, doctest count 303→286. Added a staleness banner to `LOOPCTL-DESIGN.md`.
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

[0.2.1]: https://github.com/dch-labs/loopctl/releases/tag/v0.2.1
[0.2.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.2.0
[0.1.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.1.0
