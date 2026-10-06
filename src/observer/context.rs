//! Typed context structs for [`LoopObserver`](crate::observer::LoopObserver) callbacks.
//!
//! Each struct carries the relevant fields for a specific lifecycle point.
//! Observers receive shared references (`&Context`) — the structs are
//! notification-only data carriers.

/// Context for [`LoopObserver::on_run_start`](crate::observer::LoopObserver::on_run_start).
///
/// Carries the session identifier so observers can correlate lifecycle
/// events with a specific agent run. One run is one `run()` call on the
/// loop; a session may contain many runs.
#[derive(Debug, Clone)]
pub struct RunStartContext {
    /// Unique session identifier.
    ///
    /// Correlates all lifecycle events belonging to the same agent session.
    /// Stable across every `run()` call on the same loop.
    pub session_id: uuid::Uuid,
}

/// Context for [`LoopObserver::on_run_end`](crate::observer::LoopObserver::on_run_end).
///
/// Captures the run's completion status, optional error description, total
/// turns executed, and wall-clock duration in milliseconds.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases — construct through [`new`](Self::new); struct
/// literals compile only inside the crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunEndContext {
    /// Whether the run completed successfully.
    ///
    /// `true` when the run exited normally, `false` on error or cancellation.
    pub success: bool,

    /// Error description, if the run ended due to an error.
    ///
    /// `None` when [`success`](Self::success) is `true`.
    pub error: Option<String>,

    /// Total turns completed during this run.
    ///
    /// Counts only turns that finished; an in-flight turn at the time of
    /// a fatal error is not included.
    pub total_turns: usize,

    /// Run duration in milliseconds.
    ///
    /// Measured wall-clock from [`on_run_start`](crate::observer::LoopObserver::on_run_start)
    /// to [`on_run_end`](crate::observer::LoopObserver::on_run_end).
    pub duration_ms: u64,
}

impl RunEndContext {
    /// Create a run-end context from its four facts.
    ///
    /// The construction path for code outside the crate — the type is
    /// `#[non_exhaustive]`, so struct literals compile only inside the
    /// crate. Hosts testing their observers build synthetic events
    /// through this constructor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::observer::RunEndContext;
    ///
    /// let ctx = RunEndContext::new(true, None, 3, 1_500);
    /// assert!(ctx.success);
    /// assert_eq!(ctx.total_turns, 3);
    /// ```
    #[must_use]
    pub fn new(success: bool, error: Option<String>, total_turns: usize, duration_ms: u64) -> Self {
        Self {
            success,
            error,
            total_turns,
            duration_ms,
        }
    }
}

/// Context for [`LoopObserver::on_turn_start`](crate::observer::LoopObserver::on_turn_start).
///
/// Provides the turn number and the user query that initiated it.
#[derive(Debug, Clone)]
pub struct TurnStartContext {
    /// Turn number (0-indexed).
    ///
    /// Monotonically increasing within a session; resets on session restart.
    pub turn: usize,

    /// The user query that initiated this turn.
    ///
    /// Contains the full text of the latest user message added to the
    /// conversation before the turn began.
    pub query: String,
}

/// Context for [`LoopObserver::on_turn_end`](crate::observer::LoopObserver::on_turn_end).
///
/// Reports whether the turn succeeded, any error, its duration,
/// and the token counts consumed during the turn.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code
/// reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TurnEndContext {
    /// Turn number.
    ///
    /// Matches the value passed to the corresponding
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start).
    pub turn: usize,

    /// Whether the turn completed successfully.
    ///
    /// `false` if the turn was interrupted by an error or cancellation.
    pub success: bool,

    /// Error description, if the turn failed.
    ///
    /// `None` when [`success`](Self::success) is `true`.
    pub error: Option<String>,

    /// Wall-clock duration of the turn in milliseconds.
    ///
    /// Measured from [`on_turn_start`](crate::observer::LoopObserver::on_turn_start)
    /// to [`on_turn_end`](crate::observer::LoopObserver::on_turn_end).
    pub duration_ms: u64,

    /// Input tokens consumed this turn.
    ///
    /// Sum of tokens in the prompt sent to the model.
    pub input_tokens: u64,

    /// Output tokens generated this turn.
    ///
    /// Sum of tokens across all assistant responses in the turn,
    /// including intermediate tool-call rounds.
    pub output_tokens: u64,

    /// Why the model stopped producing output for this turn.
    ///
    /// Mirrors [`Turn::stop_reason`](crate::engine::core::Turn::stop_reason)
    /// from the run record, so an observer sees the truncation-vs-finished
    /// distinction without reading the record back.
    /// [`MaxTokens`](crate::stream::StreamStopReason::MaxTokens) marks a
    /// truncated answer even though [`success`](Self::success) is `true`. A
    /// turn that failed before the model finished carries the
    /// [`EndTurn`](crate::stream::StreamStopReason::EndTurn) default — the
    /// [`error`](Self::error) field tells that story.
    pub stop_reason: crate::stream::StreamStopReason,

    /// The engine's context-size estimate at the moment the turn ended.
    ///
    /// The same figure
    /// [`LoopMachine::context_tokens`](crate::engine::core::LoopMachine::context_tokens)
    /// reports at rest: the estimated payload the provider would
    /// receive — history plus per-request overhead and known
    /// transients. Distinct from
    /// [`input_tokens`](Self::input_tokens), which is the provider's
    /// after-the-fact count of the call it actually served: this is
    /// the engine's forward-looking estimate, present even when the
    /// provider reports no usage.
    pub context_tokens: u64,

    /// The compaction denominator the run actually measured against.
    ///
    /// The resolved context window — the client's disclosed window
    /// when the provider offers one (see
    /// [`ApiClient::model_context_window`](crate::api::ApiClient::model_context_window)),
    /// otherwise the session's declared
    /// [`context_window`](crate::config::SessionConfig::context_window)
    /// — frozen for the run's whole life so the trigger and the
    /// emergency line share one number. `None` when the window policy
    /// is disabled (a resolved window of zero): there is no
    /// denominator to report. Divide
    /// [`context_tokens`](Self::context_tokens) by this field for a
    /// utilization view against the limit the engine enforced.
    pub context_window: Option<u64>,
}

/// Context for [`LoopObserver::on_stream_success`](crate::observer::LoopObserver::on_stream_success).
///
/// Provides the model name and input/output token counts for
/// a successful streaming response.
#[derive(Debug, Clone)]
pub struct StreamContext {
    /// Turn number.
    ///
    /// Identifies which turn this stream response belongs to.
    pub turn: usize,

    /// Model that was streamed.
    ///
    /// The model identifier used for this request, which may differ from
    /// the session default when model fallback occurred.
    pub model: String,

    /// Input tokens consumed.
    ///
    /// Tokens in the prompt sent to the model for this request.
    pub input_tokens: u64,

    /// Output tokens generated.
    ///
    /// Tokens in the model's streamed response.
    pub output_tokens: u64,
}

/// Context for [`LoopObserver::on_stream_failure`](crate::observer::LoopObserver::on_stream_failure).
///
/// Carries the model name and the [`LoopError`](crate::error::LoopError)
/// that caused the streaming failure.
#[derive(Debug, Clone)]
pub struct StreamFailureContext {
    /// Turn number.
    ///
    /// Identifies which turn this failure occurred in.
    pub turn: usize,

    /// Model that failed.
    ///
    /// The model identifier used for the failed request.
    pub model: String,

    /// The error that occurred.
    ///
    /// See [`LoopError`](crate::error::LoopError) for the full set of
    /// failure categories.
    pub error: crate::error::LoopError,
}

/// Context for [`LoopObserver::on_response`](crate::observer::LoopObserver::on_response).
///
/// Contains the model's text response and optional token usage
/// for the turn.
#[derive(Debug, Clone)]
pub struct ResponseContext {
    /// Turn number.
    ///
    /// Identifies which turn produced this response.
    pub turn: usize,

    /// The model's text response.
    ///
    /// Concatenated text content from the assistant message.
    /// Tool-call content is excluded; see [`ToolPostContext`]
    /// for tool result information.
    pub text: String,

    /// Token usage for this turn, if available.
    ///
    /// Populated when the API returns usage data in the
    /// streaming response. `None` when the provider does
    /// not report usage.
    pub usage: Option<crate::stream::Usage>,
}

/// Context for [`LoopObserver::on_text_delta`](crate::observer::LoopObserver::on_text_delta).
///
/// Carries one incremental text chunk from the model's streaming response. Fires
/// once per `IndexedDelta(Text)` event, *during* the stream — as opposed to
/// [`ResponseContext`], which fires once after the whole assistant text is
/// assembled.
///
/// Concatenate `delta` across all `on_text_delta` calls for a given `turn`, in
/// arrival order, to reconstruct the per-turn text. Observers that need a
/// running total maintain their own accumulator; the context carries only the
/// per-chunk slice.
///
/// # Examples
///
/// ```
/// use loopctl::observer::TextDeltaContext;
///
/// let ctx = TextDeltaContext { turn: 0, delta: "Hello".to_string() };
/// assert_eq!(ctx.turn, 0);
/// assert_eq!(ctx.delta, "Hello");
/// ```
#[derive(Debug, Clone)]
pub struct TextDeltaContext {
    /// Turn number (0-indexed).
    ///
    /// Matches the `turn` passed to the surrounding
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) /
    /// [`on_turn_end`](crate::observer::LoopObserver::on_turn_end) and to
    /// [`ResponseContext::turn`]. Lets observers correlate deltas with the
    /// turn they belong to and detect a fresh turn.
    pub turn: usize,

    /// The incremental text chunk for this delta.
    ///
    /// A small fragment of the assistant's text output. Concatenate in arrival
    /// order per turn to reconstruct the full text. No normalization, no
    /// trimming — what the provider sent, verbatim.
    pub delta: String,
}

/// Context for [`LoopObserver::on_thinking_delta`](crate::observer::LoopObserver::on_thinking_delta).
///
/// Carries one incremental reasoning ("thinking") chunk. Parallel in shape to
/// [`TextDeltaContext`]: same fields, same lifetime semantics (concatenate in
/// arrival order per turn to reconstruct the full reasoning trace). Reasoning
/// is distinct from the assistant's visible text and is never included in
/// [`ResponseContext`].
///
/// # Empty deltas
///
/// An empty `delta` arrives when the event carries something other
/// than displayable reasoning — a redacted block's opaque payload
/// (e.g. Anthropic `redacted_thinking`) or a block signature.
/// Consumers should render a placeholder, not the empty string.
///
/// # Example
///
/// ```
/// use loopctl::observer::ThinkingDeltaContext;
///
/// let ctx = ThinkingDeltaContext { turn: 0, delta: "considering options…".to_string() };
/// assert_eq!(ctx.turn, 0);
/// assert_eq!(ctx.delta, "considering options…");
/// ```
#[derive(Debug, Clone)]
pub struct ThinkingDeltaContext {
    /// Turn number (0-indexed), matching `on_turn_start` / `on_turn_end`.
    ///
    /// Same value [`TextDeltaContext::turn`] carries for the same turn, so an
    /// observer can interleave thinking and text deltas correctly.
    pub turn: usize,

    /// The incremental reasoning chunk. Concatenate in arrival order per turn.
    ///
    /// Empty when the delta carries something other than displayable
    /// reasoning — a redacted block's opaque payload or a block
    /// signature; render a placeholder, not the empty string.
    pub delta: String,
}

/// Context for [`LoopObserver::on_attempt_reset`](crate::observer::LoopObserver::on_attempt_reset).
///
/// Fired once before the first event of each retried stream attempt — never
/// before the first attempt — telling delta-buffering observers to discard
/// the failed attempt's partial text/thinking for the same turn. The cue
/// covers retries only: it never fires after a final stream failure or for
/// a non-streaming fallback, which surface through `on_stream_failure` /
/// `on_response` instead.
///
/// # Examples
///
/// ```
/// use loopctl::observer::AttemptResetContext;
///
/// let ctx = AttemptResetContext { turn: 0, attempt: 2 };
/// assert_eq!(ctx.turn, 0);
/// assert_eq!(ctx.attempt, 2);
/// ```
#[derive(Debug, Clone)]
pub struct AttemptResetContext {
    /// Turn number (0-indexed), matching `on_turn_start` / `on_turn_end`.
    ///
    /// Same value [`ThinkingDeltaContext::turn`] carries for the same
    /// turn, so an observer can key its per-turn delta buffer by it.
    pub turn: usize,

    /// The attempt the stream is entering, 1-indexed.
    ///
    /// The first attempt never fires a reset; the first retry carries
    /// `2`. A turn's resets are consecutive from the handler's retry
    /// ladder.
    pub attempt: usize,
}

/// Context for [`LoopObserver::on_gate_decision`](crate::observer::LoopObserver::on_gate_decision).
///
/// One permission-gate verdict, as the engine emitted it. Fired when
/// the deciding record becomes final — a middleware's after the
/// dispatch returns, a parked ask's denial at the park, an approved
/// ask's at the dispatch return or at the exit that stopped its call —
/// so every gated dispatch produces exactly one, while dispatches no
/// gate consulted produce none. Engine-level gates also emit records
/// with no dispatch at all: a budget refusal refuses the next turn's
/// model request before any tool is called, so its record stands
/// alone — no surrounding tool events, an empty `call_id`. The
/// `call_id` pairs a dispatch decision with the surrounding
/// [`on_tool_pre`](crate::observer::LoopObserver::on_tool_pre) /
/// [`on_tool_post`](crate::observer::LoopObserver::on_tool_post)
/// events; the decision itself carries the tool, the argument digest,
/// the verdict, and the rule provenance.
#[derive(Debug, Clone)]
pub struct GateDecisionContext {
    /// Turn number, 0-indexed.
    ///
    /// The turn whose dispatch the gate decided about, matching the
    /// value on the surrounding tool events; for engine-level
    /// decisions, the turn whose request was refused.
    pub turn: usize,

    /// The model-assigned call id the decision is about.
    ///
    /// Pairs this record with the dispatch's `tool.call` /
    /// `tool.result` ledger lines and its pre/post observer events;
    /// empty for engine-level decisions that are not about a tool
    /// call, such as a budget refusal.
    pub call_id: String,

    /// The decision itself.
    ///
    /// Verdict, argument digest, rule id and source, matched
    /// pattern, reason, and the engine-stamped timestamp.
    pub decision: crate::tool::permission::GateDecision,
}

/// Context for [`LoopObserver::on_budget_warn`](crate::observer::LoopObserver::on_budget_warn).
///
/// The budget gate's soft-line crossing report: which budget line
/// crossed, the spend at the crossing, and the hard limit the soft
/// line derives from — the numbers a "N of M" warning needs.
///
/// # Examples
///
/// ```
/// use loopctl::observer::BudgetWarnContext;
///
/// let ctx = BudgetWarnContext {
///     dimension: loopctl::budget::BudgetDimension::Tokens,
///     spent: 8_100,
///     limit: 10_000,
/// };
/// assert_eq!(ctx.spent, 8_100);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetWarnContext {
    /// The budget line whose soft threshold was crossed.
    ///
    /// Tokens, turns, or wall-clock — the same vocabulary the
    /// exhausted error and the gate records use.
    pub dimension: crate::budget::BudgetDimension,

    /// The spend at the crossing.
    ///
    /// Tokens or turns counted, or elapsed milliseconds for the
    /// wall-clock line.
    pub spent: u64,

    /// The hard limit the soft line derives from.
    ///
    /// The operator's configured ceiling, not the derived soft
    /// line, so consumers report spend against the number that was
    /// set.
    pub limit: u64,
}

/// Context for [`LoopObserver::on_tool_call_received`](crate::observer::LoopObserver::on_tool_call_received).
///
/// Fired once per tool call after the streaming response has been accumulated
/// and before dispatch of that call begins — strictly earlier than
/// [`ToolPreContext`]. Use this to surface a pending indicator the moment the
/// model decides to call a tool, before execution starts.
///
/// Unlike [`ToolPreContext`], this context also carries the call's `input`,
/// because the input is fully known at accumulation time and downstream
/// consumers may want to render it before execution. It fires exactly once per
/// call regardless of how many recovery retries the call later undergoes.
///
/// # Examples
///
/// ```
/// use loopctl::observer::ToolCallReceivedContext;
///
/// let ctx = ToolCallReceivedContext {
///     turn: 0,
///     tool: "edit".to_string(),
///     call_id: "call_1".to_string(),
///     input: serde_json::json!({"path": "/tmp/a"}),
/// };
/// assert_eq!(ctx.tool, "edit");
/// assert_eq!(ctx.call_id, "call_1");
/// ```
#[derive(Debug, Clone)]
pub struct ToolCallReceivedContext {
    /// Turn number (0-indexed).
    ///
    /// Matches the value passed to the corresponding `on_response` and
    /// `on_tool_pre` for the same assistant message.
    pub turn: usize,

    /// Tool name.
    ///
    /// Matches the name the tool was registered under. Same value as
    /// [`ToolPreContext::tool`] for the same call.
    pub tool: String,

    /// Tool call ID assigned by the API.
    ///
    /// Correlates with [`ToolPreContext::tool_call_id`] for the same call.
    pub call_id: String,

    /// JSON input the model supplied for this call.
    ///
    /// The full input object, as accumulated from the stream. Not present on
    /// `ToolPreContext` (which fires later but omits input).
    pub input: serde_json::Value,
}

/// Context for [`LoopObserver::on_tool_pre`](crate::observer::LoopObserver::on_tool_pre).
///
/// Sent before a tool is executed, providing the tool name and
/// the call ID assigned by the API.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code
/// reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolPreContext {
    /// Turn number.
    ///
    /// Identifies which turn this tool call belongs to.
    pub turn: usize,

    /// Tool name.
    ///
    /// Matches the name the tool was registered under.
    pub tool: String,

    /// Tool call ID from the API response.
    ///
    /// Unique identifier assigned by the model for this specific
    /// tool invocation, used to correlate with the tool result.
    pub tool_call_id: String,
}

/// Context for [`LoopObserver::on_tool_post`](crate::observer::LoopObserver::on_tool_post).
///
/// Sent after a tool completes, providing the tool name, a hash
/// of the output, whether an error occurred, and the execution duration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolPostContext {
    /// The model-issued call id this result answers.
    ///
    /// Mirrors [`ToolPreContext::tool_call_id`](crate::observer::ToolPreContext::tool_call_id)
    /// so observers can pair pre/post events exactly — including
    /// same-tool retries and parallel calls, where the `(turn, tool)`
    /// pair is ambiguous.
    pub tool_call_id: String,

    /// Turn number.
    ///
    /// Identifies which turn this tool result belongs to.
    pub turn: usize,

    /// Tool name.
    ///
    /// Matches the name the tool was registered under.
    pub tool: String,

    /// Deterministic hash of the tool output, if available.
    ///
    /// Used by loop detection to identify repeated tool results
    /// without exposing the full output content.
    pub result_hash: Option<u64>,

    /// Whether the tool returned an error.
    ///
    /// `true` when the tool execution resulted in an error
    /// response rather than a successful output.
    pub is_error: bool,

    /// Wall-clock execution duration.
    ///
    /// Measured from tool dispatch to completion, including any
    /// permission prompts.
    pub duration: std::time::Duration,

    /// Advisory rendering hint forwarded from the tool's
    /// [`ToolOutput`](crate::tool::ToolOutput).
    ///
    /// `None` when the call was blocked before execution (no `ToolOutput`
    /// exists), when the tool set no hint, or on error/panic paths.
    /// Presentation layers read this to pick a render strategy; loop code
    /// never reads it.
    pub display_hint: Option<crate::tool::DisplayHint>,
}

/// Context for [`LoopObserver::on_compaction`](crate::observer::LoopObserver::on_compaction).
///
/// Reports the estimated token counts before and after compaction and the
/// number of tokens saved.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code
/// reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CompactedContext {
    /// Estimated token count before compaction.
    ///
    /// The payload estimate before the compactor ran — the history plus
    /// the per-request overhead (system prompt, tool schemas) —
    /// reconstructed from `tokens_after + tokens_saved`.
    pub tokens_before: u64,

    /// Estimated token count after compaction.
    ///
    /// The payload estimate of the compacted history plus the same
    /// overhead. Transient context (contributors, retrieved memories)
    /// is not part of it — a retried turn regenerates its own, so
    /// this number does not predict the next request's size when
    /// transients ride it.
    pub tokens_after: u64,

    /// Estimated tokens saved by compaction.
    ///
    /// `tokens_before - tokens_after`, the net reduction achieved by
    /// the compactor.
    pub tokens_saved: u64,

    /// Why this pass ran.
    ///
    /// The [`CompactReason`](crate::compact::CompactReason) that
    /// triggered the pass — threshold, emergency, or manual — the same
    /// value the pre-compaction event carried, so a start/end pair can
    /// be matched without guessing.
    pub reason: crate::compact::CompactReason,

    /// How many messages this pass removed from the feed.
    ///
    /// The size of the evicted slice the engine handed to the demotion
    /// sink before adopting the compacted history — message-granular,
    /// in conversation order, empty on genuinely-unchanged passes.
    /// Pairs with the sink's own delivery record to reconcile what
    /// left the window against what memory received.
    pub evicted_messages: usize,

    /// The full pass statistics.
    ///
    /// Everything [`CompactTelemetry`](crate::compact::CompactTelemetry)
    /// computes for this pass — pre/post message-shape breakdowns,
    /// role-scoped token distribution, density, compression ratio,
    /// headroom against the manager's window, duration, and the
    /// compactor name when the host supplied one. The three flat
    /// token fields above stay payload-comparable; this field is the
    /// history-only deep view.
    pub telemetry: crate::compact::CompactTelemetry,

    /// The compactor's stage trail for this pass.
    ///
    /// A copy of [`CompactionOutcome`'s stage
    /// field](crate::compact::CompactionOutcome::stage): one line
    /// naming each internal stage the pass declined through — with its
    /// reason — and the stage that won, e.g. `QaSummarizer: error
    /// sending request; StructuredSummarizer (won)`. `None` when the
    /// configured compactor reports no internal stages, so a host
    /// distinguishes "no provenance available" from "a chain stage
    /// carried the pass" without installing a tracing subscriber.
    pub stage: Option<String>,
}

/// Context for
/// [`LoopObserver::on_compaction_failed`](crate::observer::LoopObserver::on_compaction_failed).
///
/// The pass-failure notification: fired when a compaction pass was
/// attempted and died, so a display layer can row it like a retry
/// episode even with no tracing subscriber installed. The matching
/// [`on_pre_compaction`](crate::observer::LoopObserver::on_pre_compaction)
/// already fired at pass start; the success
/// [`on_compaction`](crate::observer::LoopObserver::on_compaction)
/// never follows. `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code reads it,
/// never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CompactionFailedContext {
    /// Why the pass ran.
    ///
    /// The [`CompactReason`](crate::compact::CompactReason) that
    /// triggered the failed pass — the same value the pass-start
    /// event carried, so the pair matches without guessing.
    pub reason: crate::compact::CompactReason,

    /// The turn the pass ran in.
    ///
    /// The engine's current turn number, matching the turn carried by
    /// the surrounding turn-start/turn-end events.
    pub turn: usize,

    /// Estimated token count before the failed pass.
    ///
    /// The payload estimate the pass ran against — the history plus
    /// the per-request overhead, the same figure the pass-start event
    /// reported.
    pub tokens_before: u64,

    /// The context window the pass was compacting toward.
    ///
    /// The configured [`ContextManager`](crate::compact::ContextManager)
    /// window, so an observer can compute utilization at failure
    /// without holding the configuration itself.
    pub context_window: u64,

    /// The compactor's own error text, when the compactor failed.
    ///
    /// `Some(cause)` when a compactor ran and errored — the verbatim
    /// text the run failure's `cause` field also carries. `None` when
    /// the pass *succeeded* but its result still did not fit the
    /// window: a different failure (the compactor did its job and the
    /// conversation still overflows), reported through the same event
    /// so hosts have one place to watch.
    pub error: Option<String>,
}

/// Context for
/// [`LoopObserver::on_pre_compaction`](crate::observer::LoopObserver::on_pre_compaction).
///
/// The pass-start notification: fired when a compaction pass is about
/// to run, before the pre-compact hooks are consulted, so observers
/// can pair start and end. The outcome is not yet known — a pass that
/// the hooks veto, the run's cancellation cuts short, or the manager
/// classifies as no action still fires this event and simply never
/// fires the matching
/// [`on_compaction`](crate::observer::LoopObserver::on_compaction).
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code
/// reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreCompactionContext {
    /// Why the pass is starting.
    ///
    /// The [`CompactReason`](crate::compact::CompactReason) that
    /// triggered the pass — threshold, emergency, or manual.
    pub reason: crate::compact::CompactReason,

    /// The turn the pass runs in.
    ///
    /// The engine's current turn number, matching the turn carried by
    /// the surrounding turn-start/turn-end events.
    pub turn: usize,

    /// Estimated token count before compaction.
    ///
    /// The payload estimate the compaction decision was made against —
    /// the history plus per-request overhead, the same figure the
    /// post-event's `tokens_before` will report.
    pub tokens_before: u64,

    /// The context window the pass compacts toward.
    ///
    /// The configured [`ContextManager`](crate::compact::ContextManager)
    /// window, so an observer can compute utilization at pass start
    /// without holding the configuration itself.
    pub context_window: u64,

    /// How many messages the pass operates on.
    ///
    /// The history length at pass start — the input size, before any
    /// reduction.
    pub message_count: usize,

    /// Unique session identifier.
    ///
    /// Correlates the event with the session's other lifecycle events;
    /// stable across every `run()` call on the same loop.
    pub session_id: uuid::Uuid,
}

/// Context for [`LoopObserver::on_fallback`](crate::observer::LoopObserver::on_fallback).
///
/// Indicates which model failed (`from`) and which replacement
/// model was selected (`to`).
#[derive(Debug, Clone)]
pub struct FallbackContext {
    /// Model that failed.
    ///
    /// The model identifier that produced the error triggering fallback.
    pub from: String,

    /// Replacement model.
    ///
    /// The model identifier that will be used for subsequent requests.
    pub to: String,
}

/// Context for
/// [`LoopObserver::on_transport_fallback`](crate::observer::LoopObserver::on_transport_fallback).
///
/// Fired when a turn is served by the non-streaming transport fallback —
/// streaming exhausted its retry ceiling and the handler's last-chance
/// `create_message` produced the answer. The fallback itself is a
/// degradation report, not a failure, and the turn proceeds normally from
/// here; unrelated policies can still abort the turn afterwards, in which
/// case no flagged record exists for it (see
/// [`on_transport_fallback`](crate::observer::LoopObserver::on_transport_fallback)
/// for the tally-vs-count distinction).
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is constructed by the engine — external code
/// reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TransportFallbackContext {
    /// Turn number.
    ///
    /// Matches the value passed to the corresponding
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) —
    /// the event fires after the turn's model response is known and before
    /// its [`on_turn_end`](crate::observer::LoopObserver::on_turn_end).
    pub turn: usize,

    /// Why the fallback response's model stopped.
    ///
    /// The stop reason carried by the non-streaming response, the same
    /// value the turn's record holds in
    /// [`Turn::stop_reason`](crate::engine::core::Turn::stop_reason).
    pub stop_reason: crate::stream::StreamStopReason,
}

/// Context for [`LoopObserver::on_model_switched`](crate::observer::LoopObserver::on_model_switched).
///
/// Emitted when the model is hot-swapped via
/// [`BareLoop::switch_model`](crate::engine::BareLoop::switch_model).
/// Carries the previous and new model identifiers.
#[derive(Debug, Clone)]
pub struct ModelSwitchedContext {
    /// Model identifier before the switch.
    ///
    /// The model the session was using up to the point of the hot-swap, so
    /// observers can log or revert the transition.
    pub from: String,

    /// Model identifier after the switch.
    ///
    /// The model that will handle subsequent requests, which may differ in
    /// capability or cost from the previous one.
    pub to: String,
}

/// Context for [`LoopObserver::on_loop_detected`](crate::observer::LoopObserver::on_loop_detected).
///
/// Describes the repeating tool pattern and how many times it
/// was observed.
#[derive(Debug, Clone)]
pub struct LoopDetectedContext {
    /// Description of the repeating tool pattern.
    ///
    /// Human-readable summary of the tool operation(s) that were
    /// detected as repeating.
    pub pattern: String,

    /// Number of times the pattern was observed.
    ///
    /// Counts consecutive repetitions of the same tool operation
    /// with the same result hash.
    pub repetitions: usize,
}

/// Context for [`LoopObserver::on_convergence_detected`](crate::observer::LoopObserver::on_convergence_detected).
///
/// Carries the configured action string (e.g. `"stop"`, `"warn"`,
/// `"compact"`) determined by the detection policy.
#[derive(Debug, Clone)]
pub struct ConvergenceDetectedContext {
    /// Configured action to take (e.g. `"stop"`, `"warn"`, `"compact"`).
    ///
    /// Determined by the convergence detection policy configuration.
    /// `"stop"` halts the loop, `"warn"` logs and continues,
    /// `"compact"` triggers context compaction.
    pub action: String,
}
