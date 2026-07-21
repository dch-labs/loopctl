//! Typed context structs for [`LoopObserver`](crate::observer::LoopObserver) callbacks.
//!
//! Each struct carries the relevant fields for a specific lifecycle point.
//! Observers receive shared references (`&Context`) — the structs are
//! notification-only data carriers.

// ==================================================
// Context structs
// ==================================================

/// Context for [`LoopObserver::on_session_start`](crate::observer::LoopObserver::on_session_start).
///
/// Carries the session identifier so observers can correlate
/// lifecycle events with a specific agent run.
#[derive(Debug, Clone)]
pub struct SessionStartContext {
    /// Unique session identifier.
    ///
    /// Correlates all lifecycle events belonging to the same agent run.
    pub session_id: uuid::Uuid,
}

/// Context for [`LoopObserver::on_session_end`](crate::observer::LoopObserver::on_session_end).
///
/// Captures the session's completion status, optional error description,
/// total turns executed, and wall-clock duration in milliseconds.
#[derive(Debug, Clone)]
pub struct SessionEndContext {
    /// Whether the session completed successfully.
    ///
    /// `true` when the loop exited normally, `false` on error or cancellation.
    pub success: bool,

    /// Error description, if the session ended due to an error.
    ///
    /// `None` when [`success`](Self::success) is `true`.
    pub error: Option<String>,

    /// Total turns completed during the session.
    ///
    /// Counts only turns that finished; an in-flight turn at the time of
    /// a fatal error is not included.
    pub total_turns: usize,

    /// Total session duration in milliseconds.
    ///
    /// Measured wall-clock from [`on_session_start`](crate::observer::LoopObserver::on_session_start)
    /// to [`on_session_end`](crate::observer::LoopObserver::on_session_end).
    pub duration_ms: u64,
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
#[derive(Debug, Clone)]
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
/// # Redacted reasoning
///
/// An empty `delta` signals redacted reasoning (e.g. Anthropic
/// `redacted_thinking`) — the provider withheld the content. Consumers should
/// render a placeholder ("reasoning redacted"), not the empty string.
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
    /// Empty string for redacted/encrypted reasoning (e.g. Anthropic
    /// `redacted_thinking`) — render a placeholder, not the empty string.
    pub delta: String,
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct CompactedContext {
    /// Estimated token count before compaction.
    ///
    /// The token count of the conversation before the compactor ran,
    /// reconstructed from `tokens_after + tokens_saved`.
    pub tokens_before: u64,

    /// Estimated token count after compaction.
    ///
    /// The token count of the compacted conversation that will be used
    /// for subsequent model calls.
    pub tokens_after: u64,

    /// Estimated tokens saved by compaction.
    ///
    /// `tokens_before - tokens_after`, the net reduction achieved by
    /// the compactor.
    pub tokens_saved: u64,
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

/// Context for [`LoopObserver::on_model_switched`](crate::observer::LoopObserver::on_model_switched).
///
/// Emitted when the model is hot-swapped via
/// [`BareLoop::switch_model`](crate::engine::BareLoop::switch_model).
/// Carries the previous and new model identifiers.
#[derive(Debug, Clone)]
pub struct ModelSwitchedContext {
    /// Model identifier before the switch.
    pub from: String,
    /// Model identifier after the switch.
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
