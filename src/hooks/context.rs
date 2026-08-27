//! Hook context types — data passed to hooks at each lifecycle point.
//!
//! Each context type captures the relevant state for its trigger point.
//! Contexts are read-only snapshots; hooks cannot mutate agent state
//! directly (they return `HookAction` or [`CompactResult`] to influence
//! flow).
//!
//! # Lifecycle
//!
//! Every run opens with [`RunStartContext`], then interleaves
//! any number of tool calls (`PreToolUse` → tool → `PostToolUse`) and
//! compaction passes (`PreCompact` → compact → `PostCompact`), and
//! finally closes with [`RunEndContext`]. Every callback receives a
//! context whose fields mirror the moment it fires: pre-contexts carry
//! the *about-to-happen* inputs, post-contexts carry the
//! *what-happened* results, and the run contexts bracket the whole
//! run.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Context provided to `on_pre_tool_use` hooks.
///
/// Carries everything a hook needs to decide whether to allow, block, or
/// ask about a tool invocation: the tool name, its JSON input, and the
/// surrounding session/turn coordinates. The fields are the same values
/// that will reach the tool, so a pre-hook acts as a gatekeeper with
/// full visibility into the upcoming call.
#[derive(Debug, Clone)]
pub struct PreToolUseContext {
    /// Name of the tool about to be invoked.
    ///
    /// Matches the identifier the tool was registered under, so hooks can
    /// dispatch on it (e.g. treat `"Edit"` differently from `"Read"`).
    pub tool_name: String,

    /// JSON input that will be passed to the tool.
    ///
    /// The full arguments object the model generated, exactly as it will be
    /// delivered to the tool's execution. Hooks may inspect it to validate,
    /// redact, or block based on field values.
    pub input: Value,

    /// Session ID of the running agent.
    ///
    /// Correlates this tool call with the surrounding agent session, letting
    /// hooks share per-session state across lifecycle callbacks.
    pub session_id: uuid::Uuid,

    /// Current turn number within the session (0-indexed).
    ///
    /// How many turns have elapsed since the session started, so hooks can
    /// reason about timing (e.g. only enforce a policy after turn *N*).
    pub turn_number: usize,
}

/// Context provided to `on_post_tool_use` hooks.
///
/// Captures the outcome of a tool invocation: the input that was sent,
/// the output (or error) it produced, how long it took, and the
/// surrounding session/turn coordinates. Post-hooks are
/// notification-only and cannot alter the result — use them for
/// logging, metrics, and side-effects like
/// [`AutoCommitHook`](crate::hooks::builtin::AutoCommitHook)'s file
/// tracking.
#[derive(Debug, Clone)]
pub struct PostToolUseContext {
    /// Name of the tool that was invoked.
    ///
    /// The same identifier the tool was registered under, mirrored from the
    /// matching [`PreToolUseContext::tool_name`].
    pub tool_name: String,

    /// JSON input that was passed to the tool.
    ///
    /// The arguments object the tool actually received, useful for logging
    /// or correlating an error with its triggering input.
    pub input: Value,

    /// Output produced by the tool (stringified).
    ///
    /// The tool's return value rendered as a string, whether the call
    /// succeeded or failed. For errors this typically holds the error message.
    pub output: String,

    /// Whether the tool reported an error.
    ///
    /// `true` when the tool returned an error result rather than a normal
    /// output, so hooks can branch on failure without parsing `output`.
    pub is_error: bool,

    /// Wall-clock execution time in milliseconds.
    ///
    /// Measured from tool dispatch to completion, for latency tracking and
    /// slow-tool detection.
    pub duration_ms: u64,

    /// Session ID of the running agent.
    ///
    /// Correlates this result with the surrounding agent session, mirroring
    /// the value carried by [`PreToolUseContext::session_id`].
    pub session_id: uuid::Uuid,

    /// Turn number within the session.
    ///
    /// The 0-indexed turn in which this tool call executed, mirroring the
    /// value from the corresponding [`PreToolUseContext::turn_number`].
    pub turn_number: usize,
}

/// Why compaction was triggered.
///
/// Distinguishes between automatic threshold-based compaction and
/// manual compaction requested by the agent or user. Hooks consult this
/// to apply different policies per source — for example, allowing a
/// manual compaction to proceed while aborting an automatic one during a
/// sensitive operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactTrigger {
    /// Automatic compaction triggered by a token threshold.
    ///
    /// Fired when token usage crosses
    /// [`SessionConfig::compact_threshold`](crate::config::SessionConfig::compact_threshold)
    /// or the 95% emergency line. The compactor decided to run on its
    /// own; no caller asked for it.
    Auto,

    /// Manual compaction requested by the agent or user.
    ///
    /// Fired by an explicit compaction request — for example the agent
    /// invoking a "compact now" tool, or a host application reacting to
    /// a user gesture. The caller asked for it, independent of token
    /// usage.
    Manual,
}

impl From<crate::compact::types::CompactReason> for CompactTrigger {
    /// Map the compaction pipeline's `CompactReason` into the hook-level
    /// [`CompactTrigger`].
    ///
    /// [`ThresholdExceeded`](crate::compact::types::CompactReason::ThresholdExceeded)
    /// and [`Emergency`](crate::compact::types::CompactReason::Emergency)
    /// are both automatic triggers (they collapse to
    /// [`Auto`](CompactTrigger::Auto)), while
    /// [`Manual`](crate::compact::types::CompactReason::Manual) maps to
    /// [`Manual`](CompactTrigger::Manual).
    fn from(reason: crate::compact::types::CompactReason) -> Self {
        match reason {
            crate::compact::types::CompactReason::ThresholdExceeded
            | crate::compact::types::CompactReason::Emergency => CompactTrigger::Auto,
            crate::compact::types::CompactReason::Manual => CompactTrigger::Manual,
        }
    }
}

/// Context provided to `on_pre_compact` hooks.
///
/// Carries the state at the moment compaction was about to run — why it
/// was triggered, how big the conversation is, and how close token usage
/// is to the limit. Hooks use this to decide whether to abort compaction
/// or to inject extra instructions/context the summarizer should
/// preserve. The returned [`CompactResult`] accumulates across hooks.
#[derive(Debug, Clone)]
pub struct PreCompactContext {
    /// Why compaction was triggered.
    ///
    /// [`CompactTrigger::Auto`] when a token threshold was exceeded,
    /// [`CompactTrigger::Manual`] when the agent or user requested it. Hooks
    /// can use this to apply different policies per trigger source.
    pub trigger: CompactTrigger,

    /// Custom instructions from prior hooks (accumulated).
    ///
    /// Instructions that earlier `on_pre_compact` hooks have already appended,
    /// so a later hook can build on them rather than overwrite. `None` when no
    /// hook has contributed custom instructions yet.
    pub custom_instructions: Option<String>,

    /// Number of messages in the conversation before compaction.
    ///
    /// The size of the message history at the moment compaction was
    /// triggered, useful for deciding how aggressively to summarize.
    pub message_count: usize,

    /// Estimated token count before compaction.
    ///
    /// The payload estimate right before the compactor runs — the history
    /// plus the per-request overhead (system prompt, tool schemas) —
    /// against which `context_window` is compared to trigger compaction.
    pub tokens_before: u64,

    /// Model context window size in tokens.
    ///
    /// The configured window for the active model, providing the denominator
    /// against which `tokens_before` is compared.
    pub context_window: u64,

    /// Session ID of the running agent.
    ///
    /// Correlates this compaction event with the surrounding agent session.
    pub session_id: uuid::Uuid,
}

/// Context provided to `on_post_compact` hooks.
///
/// Reports how compaction turned out — how many messages it removed,
/// how many tokens it saved, how long it took — plus the trigger that
/// started it. Notification-only; cannot alter the compaction result.
/// Use it for metrics, budget tracking, and post-compaction logging.
#[derive(Debug, Clone)]
pub struct PostCompactContext {
    /// Why compaction was triggered.
    ///
    /// The same trigger value carried by the matching
    /// [`PreCompactContext::trigger`], so post-hooks can correlate.
    pub trigger: CompactTrigger,

    /// Number of messages removed by compaction.
    ///
    /// How many messages were dropped or summarized away, indicating how
    /// aggressively the compactor pruned the history.
    pub messages_compacted: usize,

    /// Tokens saved by compaction.
    ///
    /// The net token reduction achieved, computed as `tokens_before -
    /// tokens_after`, surfaced for metrics and budget tracking.
    pub tokens_saved: u64,

    /// Estimated tokens after compaction.
    ///
    /// The payload estimate once the compactor has finished — the
    /// compacted history plus the per-request overhead. Deferred
    /// transient context (contributors, retrieved memories) is
    /// excluded: a retried turn regenerates its own, so this number
    /// does not predict that request's size.
    pub tokens_after: u64,

    /// Compaction duration in milliseconds.
    ///
    /// Wall-clock time spent running the compactor, useful for spotting
    /// expensive summarization strategies.
    pub duration_ms: u64,

    /// Session ID of the running agent.
    ///
    /// Correlates this compaction result with the surrounding agent session.
    pub session_id: uuid::Uuid,
}

/// Result of a pre-compact hook check.
///
/// Unlike tool hooks (Allow/Block/Ask), compact hooks have richer
/// semantics: multiple hooks may want to inject context or instructions,
/// and any one of them may abort. The executor accumulates results
/// across hooks, with [`abort`](Self::abort) taking priority over every
/// other field. Construct with [`allow`](Self::allow) for the common
/// "proceed" case, or use the builder methods
/// ([`with_context`](Self::with_context),
/// [`with_instructions`](Self::with_instructions)) to shape the
/// summary.
#[derive(Debug, Clone, Default)]
pub struct CompactResult {
    /// Whether to abort compaction.
    ///
    /// When `true` the executor skips compaction entirely for this trigger;
    /// abort takes priority over every other field. Set via
    /// [`CompactResult::abort`].
    pub abort: bool,

    /// Reason for abort (shown to the user/agent).
    ///
    /// Human-readable explanation surfaced when [`abort`](Self::abort) is
    /// `true`, so callers can report why compaction was suppressed. `None`
    /// unless compaction was aborted.
    pub abort_reason: Option<String>,

    /// Override custom instructions for the summarizer.
    ///
    /// When `Some`, replaces the summarizer's default instructions entirely.
    /// When multiple hooks return instructions, the last-registered hook's
    /// value is the one delivered (each hook sees earlier contributions via
    /// [`PreCompactContext::custom_instructions`]);
    /// [`additional_context`](Self::additional_context) is the channel that
    /// accumulates across hooks. Use sparingly; prefer appending to it when
    /// you only want to add guidance.
    pub new_instructions: Option<String>,

    /// Additional context strings to include in the summary.
    ///
    /// Extra facts the hook wants preserved during summarization, accumulated
    /// across hooks. The executor feeds these to the summarizer alongside the
    /// conversation history.
    pub additional_context: Vec<String>,
}

impl CompactResult {
    /// Allow compaction to proceed with no modifications.
    ///
    /// Returns the default result — no abort, no new instructions, no
    /// extra context. This is the value a hook returns when it has no
    /// opinion about the compaction.
    #[must_use]
    pub fn allow() -> Self {
        Self::default()
    }

    /// Abort compaction and surface `reason` to the user/agent.
    ///
    /// Sets [`abort`](Self::abort) to `true` and records `reason` in
    /// [`abort_reason`](Self::abort_reason). The executor treats this as
    /// a veto: compaction is skipped for this trigger, regardless of
    /// what other hooks returned.
    pub fn abort(reason: impl Into<String>) -> Self {
        Self {
            abort: true,
            abort_reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// Append an additional context string for the summarizer to preserve.
    ///
    /// Builder-style helper that pushes `ctx` onto
    /// [`additional_context`](Self::additional_context). Multiple calls
    /// accumulate; the summarizer receives every entry alongside the
    /// conversation history. Prefer this over
    /// [`with_instructions`](Self::with_instructions) when you only want
    /// to add a fact, not rewrite the whole prompt.
    #[must_use]
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.additional_context.push(ctx.into());
        self
    }

    /// Replace the summarizer's default instructions entirely.
    ///
    /// Builder-style helper that sets
    /// [`new_instructions`](Self::new_instructions). Use sparingly — it
    /// overrides the summarizer's prompt rather than augmenting it.
    /// Prefer [`with_context`](Self::with_context) for additive guidance.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.new_instructions = Some(instructions.into());
        self
    }
}

/// Why the run ended.
///
/// Carried by [`RunEndContext::reason`] so end-hooks can branch on
/// the terminal condition. The variants cover the four ways a run
/// can stop — normal completion, cancellation, an unrecoverable error,
/// hitting a turn cap — plus context overflow when compaction could not
/// recover. Use this for outcome-specific logging and cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunEndReason {
    /// The agent finished normally.
    ///
    /// The agent declared its task complete (for example the model
    /// emitted a terminal stop reason) and the loop wound down cleanly.
    /// This is the "happy path" outcome.
    Complete,

    /// The user interrupted the run.
    ///
    /// A cancel signal — Ctrl+C in an interactive host, or a programmatic
    /// cancellation — stopped the loop. The run may have partial
    /// results available in its state at the time of interruption.
    Cancelled,

    /// The agent hit an unrecoverable error.
    ///
    /// A [`LoopError`](crate::error::LoopError) (other than
    /// cancellation) terminated the loop — for example an API failure
    /// after retries were exhausted, or a misconfiguration. The carried
    /// error is available in the agent state.
    Error,

    /// The configured turn cap was reached.
    ///
    /// The agent ran for [`RunConfig::max_turns`](crate::engine::RunConfig::max_turns)
    /// turns without converging. Not necessarily a failure — long tasks
    /// can legitimately hit the cap — but worth surfacing so the caller
    /// can decide whether to raise the limit or accept partial results.
    MaxTurns,

    /// Token usage exceeded the context window and compaction failed.
    ///
    /// The conversation grew past [`SessionConfig::context_window`](crate::config::SessionConfig::context_window),
    /// compaction was attempted but could not bring it back under the
    /// limit, and the loop terminated. This usually means the task
    /// genuinely needs a larger model context than configured.
    ContextOverflow,
}

/// Context provided to `on_run_start` hooks.
///
/// Fires at the start of each `run()` call, before any turns run. Carries
/// the session identifier, the model that will handle the first
/// request, and the working directory the agent treats as its root.
/// Notification-only — use it to initialize per-run state (open
/// resources, set up tracking, log the start).
#[derive(Debug, Clone)]
pub struct RunStartContext {
    /// Session ID.
    ///
    /// Unique identifier for the agent session that owns this run, letting
    /// hooks initialize any per-session state they track.
    pub session_id: uuid::Uuid,

    /// Model identifier.
    ///
    /// The model the session will use for its first request (e.g. a provider
    /// name like `"claude-3-opus"`). May change later if fallback or an
    /// explicit model switch occurs.
    pub model: String,

    /// Working directory the agent was started in.
    ///
    /// The filesystem path the agent treats as its root for relative tool
    /// calls, useful for hooks that log or restrict operations by location.
    pub working_directory: String,
}

/// Context provided to `on_run_end` hooks.
///
/// Fires at the end of each `run()` call, after the loop has terminated.
/// Carries the session identifier, the terminal
/// [`RunEndReason`], and aggregate counters (turns, tokens,
/// duration) for bookkeeping and cost accounting. Notification-only —
/// use it to flush resources, finalize tracking, and emit a summary log
/// line.
#[derive(Debug, Clone)]
pub struct RunEndContext {
    /// Session ID.
    ///
    /// Unique identifier for the session that owns this run, matching the
    /// value carried by [`RunStartContext::session_id`].
    pub session_id: uuid::Uuid,

    /// Why the run ended.
    ///
    /// The terminal condition — normal completion, cancellation, error, turn
    /// cap, or context overflow — so hooks can branch on the outcome.
    pub reason: RunEndReason,

    /// Total turns executed.
    ///
    /// How many turns ran to completion during this run, for bookkeeping
    /// and budget reporting.
    pub total_turns: usize,

    /// Total tokens consumed (input + output across all turns).
    ///
    /// Aggregate token usage over the whole run, combining input and
    /// output tokens from every turn for cost accounting.
    pub total_tokens: u64,

    /// Wall-clock run duration in seconds.
    ///
    /// Elapsed time from run start to run end, rounded to whole
    /// seconds, for latency and uptime reporting.
    pub duration_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_trigger_maps_every_compaction_reason() {
        use crate::compact::types::CompactReason;
        assert_eq!(
            CompactTrigger::from(CompactReason::ThresholdExceeded),
            CompactTrigger::Auto
        );
        assert_eq!(
            CompactTrigger::from(CompactReason::Emergency),
            CompactTrigger::Auto
        );
        assert_eq!(
            CompactTrigger::from(CompactReason::Manual),
            CompactTrigger::Manual
        );
    }

    #[test]
    fn compact_result_allow_is_default() {
        let result = CompactResult::allow();
        assert!(!result.abort);
        assert!(result.abort_reason.is_none());
        assert!(result.new_instructions.is_none());
        assert!(result.additional_context.is_empty());
    }

    #[test]
    fn compact_result_abort_sets_reason() {
        let result = CompactResult::abort("too risky");
        assert!(result.abort);
        assert_eq!(result.abort_reason.as_deref(), Some("too risky"));
    }

    #[test]
    fn compact_result_builder_pattern() {
        let result = CompactResult::allow()
            .with_context("keep file X")
            .with_context("remember Y")
            .with_instructions("focus on Z");
        assert!(!result.abort);
        assert_eq!(result.additional_context.len(), 2);
        assert_eq!(result.new_instructions.as_deref(), Some("focus on Z"));
    }

    #[test]
    fn compact_trigger_serialization() {
        for trigger in [CompactTrigger::Auto, CompactTrigger::Manual] {
            let json = serde_json::to_string(&trigger).expect("serialize");
            let back: CompactTrigger = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, trigger);
        }
    }

    #[test]
    fn run_end_reason_serialization() {
        let reasons = [
            RunEndReason::Complete,
            RunEndReason::Cancelled,
            RunEndReason::Error,
            RunEndReason::MaxTurns,
            RunEndReason::ContextOverflow,
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).expect("serialize");
            let back: RunEndReason = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, reason);
        }
    }

    #[test]
    fn compact_result_abort_with_no_additional_fields() {
        let result = CompactResult::abort("reason");
        assert!(result.abort);
        assert_eq!(result.abort_reason.as_deref(), Some("reason"));
        assert!(
            result.new_instructions.is_none(),
            "should have no new_instructions"
        );
        assert!(
            result.additional_context.is_empty(),
            "should have empty additional_context"
        );
    }
}
