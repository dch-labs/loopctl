//! Supporting types for context compaction.
//!
//! Data types used across the compaction pipeline:
//!
//! - [`CompactReason`] — why compaction was triggered.
//! - [`CompactionContext`] — input metadata passed to compactors.
//! - [`CompactionOutcome`] — result of a single compaction pass.
//! - [`CompactTelemetry`] — telemetry data for compaction operations.
//! - [`PreCompactStats`] / [`PostCompactStats`] — stats before/after compaction.
//! - [`ContextOverflow`] — error when the conversation cannot fit.
//! - [`EnsureContextResult`] — result of [`ContextManager::ensure_context_fits`](super::ContextManager::ensure_context_fits).

use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::fmt;

// ===================================================
// CompactReason
// ===================================================

/// Why compaction was triggered.
///
/// Different triggers may warrant different compaction strategies.
/// For example, an [`Emergency`](CompactReason::Emergency) compaction
/// should be more aggressive than a routine threshold check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactReason {
    /// Token usage exceeded the configured threshold percentage.
    ///
    /// This is the routine, expected trigger: estimated context size crossed the
    /// [`ContextManager`](super::ContextManager)'s threshold (80% by default),
    /// so a compaction pass runs proactively before the next turn to keep the
    /// context comfortably below the window.
    ThresholdExceeded,

    /// Token usage is dangerously close to the context window limit.
    ///
    /// This is the fallback safety trigger, firing when usage reaches the
    /// emergency zone (95% of the window) regardless of the configured
    /// threshold. An emergency compaction should compact more aggressively
    /// than a routine threshold pass because the conversation is on the verge
    /// of overflowing the model's window.
    Emergency,

    /// Compaction was explicitly requested (e.g. by the agent or a tool).
    ///
    /// Compaction was forced by an explicit caller rather than by a size-based
    /// trigger — for example a host application compacting on demand, or a tool
    /// that wants to free context before producing a large result.
    Manual,
}

impl fmt::Display for CompactReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThresholdExceeded => write!(f, "threshold exceeded"),
            Self::Emergency => write!(f, "emergency"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

// ===================================================
// CompactionContext
// ===================================================

/// Metadata passed to [`ContextCompactor::compact`](super::ContextCompactor::compact)
/// describing the compaction trigger and current state.
///
/// Compactors can use this information to decide how aggressively to
/// compact — e.g. an emergency compaction may use more aggressive
/// summarization than a routine threshold check.
#[derive(Debug, Clone)]
pub struct CompactionContext {
    /// Estimated token count before compaction.
    ///
    /// The compactor's input size, computed with the crate's standard
    /// 4-chars-per-token heuristic. Compaction aims to bring the post-compaction
    /// size below this so the conversation fits with headroom for the next turn.
    pub tokens_before: u64,

    /// Why compaction was triggered.
    ///
    /// Compactors may use the trigger to pick a strategy — an
    /// [`Emergency`](CompactReason::Emergency) trigger warrants more aggressive
    /// summarization than a routine [`ThresholdExceeded`](CompactReason::ThresholdExceeded).
    pub reason: CompactReason,

    /// The model's context window size.
    ///
    /// The hard upper bound on tokens the model accepts in one request. This is
    /// the denominator every compaction threshold and target is expressed
    /// against, so the compactor can decide how much to keep.
    pub context_window: u64,

    /// The current turn number in the session.
    ///
    /// Zero-indexed within the run. Useful for compaction strategies that weight
    /// recent turns more heavily, or for correlating a compaction pass back to
    /// the turn that triggered it in logs.
    pub turn: usize,
}

// ===================================================
// CompactionOutcome
// ===================================================

/// Result of a single compaction pass.
///
/// Returned by [`ContextCompactor::compact`](super::ContextCompactor::compact),
/// this struct contains the compacted message list along with telemetry data
/// about what happened.
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    /// The compacted message list.
    ///
    /// The messages that remain after compaction — typically a summary or
    /// truncation of the original conversation. The caller feeds this list back
    /// into the loop as the new history. May equal the input when compaction
    /// decided no change was needed (see [`CompactionOutcome::no_change`]).
    pub messages: Vec<Message>,

    /// Estimated token count after compaction.
    ///
    /// The post-compaction size of [`messages`](Self::messages), estimated with
    /// the same heuristic used for the pre-compaction count, so before/after
    /// values are directly comparable.
    pub tokens_after: u64,

    /// Estimated tokens saved by compaction.
    ///
    /// The difference between the pre-compaction token count and
    /// [`tokens_after`](Self::tokens_after). Zero when compaction made no change
    /// or when it enlarged the conversation (e.g. injecting a summary that
    /// outweighs the messages it replaced).
    pub tokens_saved: u64,

    /// Whether compaction succeeded.
    ///
    /// `true` when the compactor produced a usable message list, even if that
    /// list is unchanged. `false` only when the compactor itself failed — in
    /// that case [`error`](Self::error) describes the failure and
    /// [`messages`](Self::messages) typically holds the original input.
    pub success: bool,

    /// Error message if compaction failed.
    ///
    /// `Some(description)` when [`success`](Self::success) is `false`, carrying
    /// the compactor's human-readable failure reason. `None` on success. Typed
    /// as a [`String`] because it surfaces to observers/logs, not to programmatic
    /// control flow (the loop treats any failed compaction uniformly).
    pub error: Option<String>,
}

impl CompactionOutcome {
    /// Create an outcome representing no change (compaction was not needed).
    ///
    /// Use this when the compactor decides the messages don't need
    /// compaction — e.g. when the message count is below the minimum.
    #[must_use]
    pub fn no_change(messages: Vec<Message>) -> Self {
        let tokens = Self::estimate_tokens(&messages);
        Self {
            messages,
            tokens_after: tokens,
            tokens_saved: 0,
            success: true,
            error: None,
        }
    }

    /// Create an outcome representing successful compaction.
    ///
    /// Computes [`tokens_saved`](Self::tokens_saved) automatically from the
    /// difference between `tokens_before` and `tokens_after`.
    #[must_use]
    pub fn compacted(messages: Vec<Message>, tokens_before: u64, tokens_after: u64) -> Self {
        Self {
            tokens_saved: tokens_before.saturating_sub(tokens_after),
            messages,
            tokens_after,
            success: true,
            error: None,
        }
    }

    /// Estimate the token count for a slice of messages.
    ///
    /// Uses the standard 4-chars-per-token heuristic, the same
    /// heuristic used by [`ContextManager::estimate_tokens`](super::ContextManager::estimate_tokens).
    #[must_use]
    pub fn estimate_tokens(messages: &[Message]) -> u64 {
        super::ContextManager::estimate_tokens(messages)
    }
}

// ===================================================
// CompactTelemetry
// ===================================================

/// Telemetry data for a single compaction operation.
///
/// Produced by [`ContextManager::ensure_context_fits`](super::ContextManager::ensure_context_fits)
/// when compaction occurs. Observers receive this via
/// [`on_compaction`](crate::observer::LoopObserver::on_compaction).
#[derive(Debug, Clone)]
pub struct CompactTelemetry {
    /// Why compaction was triggered.
    ///
    /// The [`CompactReason`] that caused this pass, useful for distinguishing
    /// routine threshold compactions from emergency or manual ones when reading
    /// the telemetry.
    pub trigger: CompactReason,

    /// Conversation stats before compaction.
    ///
    /// A snapshot of the conversation as it was when compaction began — message
    /// counts broken down by role and token estimate. See [`PreCompactStats`].
    pub pre_compact: PreCompactStats,

    /// Conversation stats after compaction.
    ///
    /// A snapshot of the conversation after compaction completed, plus the
    /// savings achieved. Compare against [`pre_compact`](Self::pre_compact) to
    /// measure the effect of the pass. See [`PostCompactStats`].
    pub post_compact: PostCompactStats,

    /// Wall-clock duration of the compaction.
    ///
    /// Time spent inside the compactor's `compact` call for this pass, measured
    /// from just before the call to just after. Excludes the token-estimate
    /// bookkeeping done before and after the call itself.
    pub duration: std::time::Duration,
}

/// Conversation statistics captured before compaction.
///
/// A breakdown of the conversation's shape at the moment compaction begins:
/// how many messages there are, their estimated token cost, and how they split
/// across user/assistant/tool roles. Captured by
/// [`ContextManager::build_telemetry`](super::ContextManager::build_telemetry)
/// and bundled into [`CompactTelemetry::pre_compact`].
#[derive(Debug, Clone)]
pub struct PreCompactStats {
    /// Total number of messages in the conversation.
    ///
    /// Every message in the history about to be compacted, regardless of role
    /// or content. This is the input size the compactor operates on.
    pub total_messages: usize,

    /// Estimated token count.
    ///
    /// The pre-compaction token estimate of the whole conversation, using the
    /// standard 4-chars-per-token heuristic. This is the number compared against
    /// the threshold to decide whether compaction was needed.
    pub estimated_tokens: u64,

    /// Number of user-role messages.
    ///
    /// Messages whose role is [`User`](crate::message::Role::User). Includes
    /// both genuine user turns and tool-result messages, which are conventionally
    /// sent with the user role.
    pub user_messages: usize,

    /// Number of assistant-role messages.
    ///
    /// Messages whose role is [`Assistant`](crate::message::Role::Assistant) —
    /// the model's own responses, including any that carried tool-call requests.
    pub assistant_messages: usize,

    /// Number of messages containing tool calls or results.
    ///
    /// Messages with at least one tool-call or tool-result part, regardless of
    /// role. These are often worth preserving across compaction because they
    /// carry the intermediate state of the tool loop.
    pub tool_messages: usize,
}

/// Conversation statistics captured after compaction.
///
/// A summary of the conversation's shape after compaction completes, together
/// with how much the pass reclaimed. Captured by
/// [`ContextManager::build_telemetry`](super::ContextManager::build_telemetry)
/// and bundled into [`CompactTelemetry::post_compact`].
#[derive(Debug, Clone)]
pub struct PostCompactStats {
    /// Total number of messages after compaction.
    ///
    /// The size of the compacted message list. Smaller than the pre-compaction
    /// [`total_messages`](PreCompactStats::total_messages) when compaction
    /// removed or summarized messages; equal when it made no change.
    pub total_messages: usize,
    /// Estimated token count after compaction.
    ///
    /// The post-compaction token estimate, comparable to
    /// [`estimated_tokens`](PreCompactStats::estimated_tokens) from the
    /// pre-compaction snapshot. The difference is [`tokens_saved`](Self::tokens_saved).
    pub estimated_tokens: u64,
    /// Tokens removed by compaction.
    ///
    /// How many tokens the pass reclaimed: the pre-compaction estimate minus
    /// the post-compaction estimate. Saturates at zero, so it never goes
    /// negative even if a summary injection made the conversation larger.
    pub tokens_saved: u64,
    /// Percentage of tokens saved (0–100).
    ///
    /// [`tokens_saved`](Self::tokens_saved) as a share of the pre-compaction
    /// estimate, expressed as a whole-number percentage. Clamped to `0..=100`;
    /// `0` when nothing was saved or when the pre-compaction estimate was zero.
    pub percent_saved: u8,
}

// ===================================================
// ContextOverflow error
// ===================================================

/// Error returned when the conversation cannot fit within the context
/// window, even after compaction.
///
/// Terminal condition — the conversation is too large and the
/// compactor was unable to reduce it sufficiently.
#[derive(Debug, Clone)]
pub struct ContextOverflow {
    /// Estimated token count of the conversation.
    ///
    /// How many tokens the conversation occupies when it overflows — the same
    /// heuristic estimate used everywhere else in the subsystem. Compare
    /// against [`context_window`](Self::context_window) (or use
    /// [`overflow`](Self::overflow)) to see by how much it exceeded the limit.
    pub tokens_used: u64,

    /// The model's context window size.
    ///
    /// The hard token limit the conversation failed to fit under, even after a
    /// compaction pass. The denominator [`utilization`](Self::utilization) is
    /// measured against.
    pub context_window: u64,

    /// How many messages were in the conversation.
    ///
    /// The message count at the point of overflow, useful for diagnosing
    /// whether the overflow came from many small messages or a few large ones.
    pub message_count: usize,

    /// The reason compaction was attempted.
    ///
    /// The [`CompactReason`] that triggered the (failed) compaction attempt.
    /// An [`Emergency`](CompactReason::Emergency) trigger here means even an
    /// aggressive compaction could not bring the conversation back under the
    /// window.
    pub trigger: CompactReason,

    /// Error from the compactor, if compaction was attempted.
    ///
    /// `Some(description)` when a compactor ran but returned an error that
    /// prevented recovery; `None` when the conversation was simply too large
    /// to reduce (compaction succeeded but the result still overflowed).
    pub compactor_error: Option<String>,
}

impl ContextOverflow {
    /// How many tokens the conversation exceeds the window by.
    #[must_use]
    pub fn overflow(&self) -> u64 {
        self.tokens_used.saturating_sub(self.context_window)
    }

    /// The fraction of the context window used (0.0–1.0+).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.context_window == 0 {
            return f64::INFINITY;
        }
        f64::from(u32::try_from(self.tokens_used).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(self.context_window).unwrap_or(u32::MAX))
    }
}

impl fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "context overflow: {} tokens used of {} window ({} messages, {} overflow)",
            self.tokens_used,
            self.context_window,
            self.message_count,
            self.overflow()
        )
    }
}

impl std::error::Error for ContextOverflow {}

// ===================================================
// EnsureContextResult
// ===================================================

/// Result of [`ContextManager::ensure_context_fits`](super::ContextManager::ensure_context_fits).
///
/// Tells the caller whether compaction occurred and provides the
/// (possibly compacted) message list.
#[derive(Debug, Clone)]
pub enum EnsureContextResult {
    /// Compaction occurred and produced a shorter message list.
    ///
    /// The wrapped [`CompactionOutcome`] carries the compacted messages, the
    /// token savings, and whether the pass succeeded. Feed
    /// [`outcome.messages`](CompactionOutcome::messages) back into the loop as
    /// the new history.
    Compacted(CompactionOutcome),

    /// No compaction was needed; messages returned as-is.
    ///
    /// The conversation fit comfortably within the threshold, so no compaction
    /// pass ran. The wrapped message list is the original input, unchanged; use
    /// it directly as the next-turn history.
    NoAction(Vec<Message>),
}

impl EnsureContextResult {
    /// Extract the message list from this result, regardless of variant.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        match self {
            Self::Compacted(outcome) => outcome.messages,
            Self::NoAction(messages) => messages,
        }
    }
}
