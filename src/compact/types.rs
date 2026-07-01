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
use std::fmt;

// ===================================================
// CompactReason
// ===================================================

/// Why compaction was triggered.
///
/// Different triggers may warrant different compaction strategies.
/// For example, an [`Emergency`](CompactReason::Emergency) compaction
/// should be more aggressive than a routine threshold check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    /// Token usage exceeded the configured threshold percentage.
    ThresholdExceeded,
    /// Token usage is dangerously close to the context window limit.
    Emergency,
    /// Compaction was explicitly requested (e.g. by the agent or a tool).
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
    pub tokens_before: u64,
    /// Why compaction was triggered.
    pub reason: CompactReason,
    /// The model's context window size.
    pub context_window: u64,
    /// The current turn number in the session.
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
    pub messages: Vec<Message>,
    /// Estimated token count after compaction.
    pub tokens_after: u64,
    /// Estimated tokens saved by compaction.
    pub tokens_saved: u64,
    /// Whether compaction succeeded.
    pub success: bool,
    /// Error message if compaction failed.
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
    pub trigger: CompactReason,
    /// Conversation stats before compaction.
    pub pre_compact: PreCompactStats,
    /// Conversation stats after compaction.
    pub post_compact: PostCompactStats,
    /// Wall-clock duration of the compaction.
    pub duration: std::time::Duration,
}

/// Conversation statistics captured before compaction.
#[derive(Debug, Clone)]
pub struct PreCompactStats {
    /// Total number of messages in the conversation.
    pub total_messages: usize,
    /// Estimated token count.
    pub estimated_tokens: u64,
    /// Number of user-role messages.
    pub user_messages: usize,
    /// Number of assistant-role messages.
    pub assistant_messages: usize,
    /// Number of messages containing tool calls or results.
    pub tool_messages: usize,
}

/// Conversation statistics captured after compaction.
#[derive(Debug, Clone)]
pub struct PostCompactStats {
    /// Total number of messages after compaction.
    pub total_messages: usize,
    /// Estimated token count after compaction.
    pub estimated_tokens: u64,
    /// Tokens removed by compaction.
    pub tokens_saved: u64,
    /// Percentage of tokens saved (0–100).
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
    pub tokens_used: u64,
    /// The model's context window size.
    pub context_window: u64,
    /// How many messages were in the conversation.
    pub message_count: usize,
    /// The reason compaction was attempted.
    pub trigger: CompactReason,
    /// Error from the compactor, if compaction was attempted.
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
    Compacted(CompactionOutcome),
    /// No compaction was needed; messages returned as-is.
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
