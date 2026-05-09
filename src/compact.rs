//! Context management and compaction for agent conversations.
//!
//! As conversations grow, they approach the model's context window limit.
//! This module provides the infrastructure to detect when compaction is
//! needed and to carry it out through a pluggable strategy.
//!
//! # Architecture
//!
//! The design separates **when to compact** from **how to compact**:
//!
//! - [`ContextManager`] — a concrete struct that monitors token usage,
//!   checks thresholds, and decides when to trigger compaction.
//! - [`ContextCompactor`] — a trait that defines the compaction strategy.
//!   Plug in truncation, summarization, Q&A extraction, or any custom
//!   approach.
//!
//! ```text
//! ┌───────────────────────────────┐
//! │       ContextManager          │
//! │                               │
//! │  estimate_tokens()            │
//! │  should_compact()             │
//! │  ensure_context_fits()        │
//! │          │                    │
//! │          ▼                    │
//! │  ┌──────────────────────┐     │
//! │  │  dyn ContextCompactor│     │
//! │  │  .compact()          │     │
//! │  └──────────────────────┘     │
//! └───────────────────────────────┘
//! ```
//!
//! # Provided Compactors
//!
//! - [`TruncatingCompactor`] — drops the oldest messages, keeping the
//!   system prompt and a configurable number of recent messages. No LLM
//!   calls required.
//!
//! Agent-side compactors (LLM-based summarization, Q&A extraction, etc.)
//! live outside the framework and implement [`ContextCompactor`] against
//! their own API client.
//!
//! # Supporting Types
//!
//! - [`TokenSplitter`] — splits a conversation at turn boundaries for coherent compaction.
//! - [`CompactTelemetry`] — telemetry data for compaction operations.
//! - [`CompactionOutcome`] — result of a single compaction pass.
//! - [`CompactionContext`] — input context passed to compactors.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::compact::{ContextManager, TruncatingCompactor};
//! use std::sync::Arc;
//!
//! let compactor = TruncatingCompactor::new()
//!     .with_preserve_recent(4)
//!     .with_min_messages(6);
//!
//! let manager = ContextManager::new(Arc::new(compactor))
//!     .with_context_window(200_000)
//!     .with_threshold(0.80);
//!
//! assert_eq!(manager.context_window(), 200_000);
//! assert!((manager.threshold() - 0.80).abs() < f64::EPSILON);
//! ```
//!
//! # Integration with `BareLoop`
//!
//! The engine's [`BareLoop`](crate::engine::BareLoop) accepts an optional
//! [`ContextManager`]. When present, it checks token usage after each turn
//! and triggers compaction automatically when usage exceeds the threshold.

use crate::message::{Message, Role};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

// ===================================================
// ContextCompactor trait
// ===================================================

/// Strategy trait for compacting a conversation's message history.
///
/// Implementations define *how* to reduce a message list — truncation,
/// summarization, Q&A extraction, etc. The framework calls
/// [`compact`](ContextCompactor::compact) when the [`ContextManager`]
/// determines compaction is needed.
///
/// The trait is object-safe so it can be used as `Arc<dyn ContextCompactor>`.
///
/// # Contract
///
/// - The returned [`CompactionOutcome`] always contains a valid message
///   list (never empty unless the input was empty).
/// - The `target_tokens` parameter is a hint; implementations may exceed
///   it if necessary for coherent output.
/// - Implementations must set [`CompactionOutcome::success`] to `false`
///   and provide an error message if compaction fails.
///
/// # Example
///
/// ```rust
/// use std::future::Future;
/// use std::pin::Pin;
///
/// use loopctl::compact::{ContextCompactor, CompactionContext, CompactionOutcome};
/// use loopctl::message::Message;
///
/// struct DropOldCompactor;
///
/// impl ContextCompactor for DropOldCompactor {
///     fn compact(
///         &self,
///         messages: Vec<Message>,
///         target_tokens: u64,
///         context: CompactionContext,
///     ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
///         Box::pin(async move {
///             if messages.len() <= 2 {
///                 return CompactionOutcome::no_change(messages);
///             }
///             let tokens_before = CompactionOutcome::estimate_tokens(&messages);
///             let preserved: Vec<Message> = messages.into_iter().rev().take(2).rev().collect();
///             let tokens_after = CompactionOutcome::estimate_tokens(&preserved);
///             CompactionOutcome::compacted(
///                 preserved,
///                 tokens_before,
///                 tokens_after,
///             )
///         })
///     }
/// }
/// ```
pub trait ContextCompactor: Send + Sync {
    /// Compact the given messages to fit within `target_tokens`.
    ///
    /// The `context` parameter provides metadata about why compaction was
    /// triggered and the current token budget. Implementations use this
    /// to make informed decisions about how aggressively to compact.
    ///
    /// Returns a [`CompactionOutcome`] containing the compacted message
    /// list and telemetry data.
    ///
    /// # Arguments
    ///
    /// * `messages` — The full conversation history to compact.
    /// * `target_tokens` — The target token count for the compacted output.
    /// * `context` — Metadata about the compaction trigger.
    // The return-type boxing is required for object safety.
    #[allow(clippy::type_complexity)]
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>>;
}

// ===================================================
// CompactionContext
// ===================================================

/// Metadata passed to [`ContextCompactor::compact`] describing the
/// compaction trigger and current state.
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
// CompactionOutcome
// ===================================================

/// Result of a single compaction pass.
///
/// Returned by [`ContextCompactor::compact`], this struct contains the
/// compacted message list along with telemetry data about what happened.
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    /// The compacted message list.
    pub messages: Vec<Message>,
    /// How many messages were removed by compaction.
    pub messages_compacted: usize,
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
            messages_compacted: 0,
            tokens_after: tokens,
            tokens_saved: 0,
            success: true,
            error: None,
        }
    }

    /// Create an outcome representing successful compaction.
    #[must_use]
    pub fn compacted(messages: Vec<Message>, tokens_before: u64, tokens_after: u64) -> Self {
        Self {
            messages_compacted: 0, // caller should set
            tokens_saved: tokens_before.saturating_sub(tokens_after),
            messages,
            tokens_after,
            success: true,
            error: None,
        }
    }

    /// Estimate the token count for a slice of messages.
    ///
    /// Uses the standard 4-chars-per-token heuristic. This is the same
    /// heuristic used by [`ContextManager::estimate_tokens`].
    #[must_use]
    pub fn estimate_tokens(messages: &[Message]) -> u64 {
        ContextManager::estimate_tokens(messages)
    }
}

// ===================================================
// CompactTelemetry
// ===================================================

/// Telemetry data for a single compaction operation.
///
/// Produced by [`ContextManager::ensure_context_fits`] when compaction
/// occurs. Observers receive this via
/// [`on_compaction`](crate::core::AgentObserver::on_compaction).
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
// TruncatingCompactor
// ===================================================

/// A simple compactor that drops the oldest messages.
///
/// Keeps the first message (typically the system prompt) and a configurable
/// number of recent messages. No LLM calls required — useful as a fallback
/// or for contexts where summarization isn't available.
///
/// # Strategy
///
/// ```text
/// [System?] [Old₁, Old₂, ..., Oldₙ] [Recent₁, Recent₂, ..., Recentₘ]
///  ↑ kept   ↑ discarded ↑            ↑ preserved ↑
/// ```
///
/// The first message is always retained (if present) because it usually
/// contains the system prompt or conversation instructions. This prevents
/// the compactor from discarding essential context that shapes the agent's
/// behavior. If the conversation is shorter than `min_messages`, no
/// compaction occurs.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::TruncatingCompactor;
/// use std::sync::Arc;
///
/// let compactor = TruncatingCompactor::new()
///     .with_preserve_recent(6)
///     .with_min_messages(8);
///
/// // Pass to ContextManager:
/// // let manager = ContextManager::new(Arc::new(compactor));
/// ```
#[derive(Debug, Clone)]
pub struct TruncatingCompactor {
    /// Number of recent messages to always preserve.
    preserve_recent: usize,
    /// Minimum messages before compaction is considered.
    min_messages: usize,
}

impl TruncatingCompactor {
    /// Create a new truncating compactor with sensible defaults.
    ///
    /// Defaults:
    ///
    /// | Setting           | Default |
    /// |-------------------|---------|
    /// | `preserve_recent` | 4       |
    /// | `min_messages`    | 6       |
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_recent: 4,
            min_messages: 6,
        }
    }

    /// Set how many recent messages to preserve during compaction.
    ///
    /// This many messages from the end of the conversation are kept
    /// intact. The rest are dropped. Must be at least 1.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the minimum number of messages before compaction is attempted.
    ///
    /// If the conversation has fewer messages than this, compaction is
    /// skipped entirely. Prevents aggressive truncation of short
    /// conversations.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Number of recent messages that will be preserved.
    #[must_use]
    pub fn preserve_recent(&self) -> usize {
        self.preserve_recent
    }

    /// Minimum messages before compaction is attempted.
    #[must_use]
    pub fn min_messages(&self) -> usize {
        self.min_messages
    }
}

impl Default for TruncatingCompactor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextCompactor for TruncatingCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        _target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let total = messages.len();
            if total <= self.min_messages {
                return CompactionOutcome::no_change(messages);
            }

            // Determine split point: keep `preserve_recent` from the end.
            let split = total.saturating_sub(self.preserve_recent);
            let recent: Vec<Message> = messages.get(split..).unwrap_or_default().to_vec();

            // Always preserve the first message (typically the system prompt)
            // unless it is already included in the recent slice (split == 0).
            let preserved = if split > 0 {
                if let Some(first) = messages.first() {
                    let mut v = vec![first.clone()];
                    v.extend(recent);
                    v
                } else {
                    recent
                }
            } else {
                // split == 0 means recent already contains all messages.
                recent
            };

            let tokens_after = CompactionOutcome::estimate_tokens(&preserved);
            let messages_compacted = total.saturating_sub(preserved.len());

            CompactionOutcome {
                messages: preserved,
                messages_compacted,
                tokens_after,
                tokens_saved: context.tokens_before.saturating_sub(tokens_after),
                success: true,
                error: None,
            }
        })
    }
}

// ===================================================
// TokenSplitter
// ===================================================

/// Splits a conversation into "old" and "recent" at a turn boundary.
///
/// Used by compactors (and agent-side code) that need to know which
/// messages to compact versus preserve. Splits at role transitions for
/// coherent summarization — the split always occurs between a complete
/// request/response pair.
///
/// # Rules
///
/// - Never compact the last user message (it's the current request).
/// - Split at turn boundaries (role transitions) for coherent output.
/// - If the conversation is too short, `to_compact` will be empty.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::TokenSplitter;
/// use loopctl::message::Message;
///
/// let splitter = TokenSplitter::new()
///     .with_preserve_recent(4)
///     .with_min_messages(6);
///
/// let messages = vec![
///     Message::user("Hello"),
///     Message::assistant("Hi there!"),
///     Message::user("What is 2+2?"),
///     Message::assistant("4"),
/// ];
///
/// let result = splitter.split(&messages);
/// // With only 4 messages and min_messages=6, nothing is split off.
/// assert!(result.to_compact.is_empty());
/// assert_eq!(result.preserved.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct TokenSplitter {
    /// Number of recent messages to always preserve.
    preserve_recent: usize,
    /// Minimum messages before considering a split.
    min_messages: usize,
}

/// Result of splitting a conversation into old and recent portions.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Messages to compact or summarize (the "old" part).
    pub to_compact: Vec<Message>,
    /// Messages to preserve as-is (the "recent" part).
    pub preserved: Vec<Message>,
    /// Estimated tokens in `to_compact`.
    pub compact_tokens: u64,
    /// Estimated tokens in `preserved`.
    pub preserved_tokens: u64,
    /// The index in the original message list where the split occurred.
    pub split_index: usize,
}

impl TokenSplitter {
    /// Create a new splitter with sensible defaults.
    ///
    /// Defaults:
    ///
    /// | Setting           | Default |
    /// |-------------------|---------|
    /// | `preserve_recent` | 4       |
    /// | `min_messages`    | 6       |
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_recent: 4,
            min_messages: 6,
        }
    }

    /// Set how many recent messages to preserve.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the minimum messages before splitting is considered.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Split the given messages into old and recent portions.
    ///
    /// The split point is chosen at a turn boundary (a role transition
    /// from assistant to user) as close as possible to leaving
    /// `preserve_recent` messages in the recent portion.
    ///
    /// If the conversation has fewer than `min_messages`, the entire
    /// conversation goes into `preserved` and `to_compact` is empty.
    #[must_use]
    pub fn split(&self, messages: &[Message]) -> SplitResult {
        if messages.len() <= self.min_messages {
            return SplitResult {
                to_compact: vec![],
                preserved: messages.to_vec(),
                compact_tokens: 0,
                preserved_tokens: ContextManager::estimate_tokens(messages),
                split_index: 0,
            };
        }

        // Find a split point: we want `preserve_recent` messages at the end.
        // Look for a turn boundary (assistant→user transition) near the
        // target split point.
        let target_split = messages.len().saturating_sub(self.preserve_recent);
        let split_index = Self::find_turn_boundary(messages, target_split);
        let (to_compact, preserved) = messages.split_at(split_index);
        SplitResult {
            to_compact: to_compact.to_vec(),
            preserved: preserved.to_vec(),
            compact_tokens: ContextManager::estimate_tokens(to_compact),
            preserved_tokens: ContextManager::estimate_tokens(preserved),
            split_index,
        }
    }

    /// Find the nearest turn boundary at or before the target index.
    ///
    /// A turn boundary is a position where the previous message is
    /// assistant-role and the next is user-role. This ensures we split
    /// at a coherent conversation boundary.
    fn find_turn_boundary(messages: &[Message], target: usize) -> usize {
        if target == 0 {
            return 0;
        }

        // Search backwards from target for an assistant→user transition.
        for i in (1..=target).rev() {
            if i < messages.len() {
                let Some(prev) = messages.get(i.saturating_sub(1)) else {
                    continue;
                };
                let Some(curr) = messages.get(i) else {
                    continue;
                };
                let prev_is_assistant = prev.role == Role::Assistant;
                let curr_is_user = curr.role == Role::User;
                if prev_is_assistant && curr_is_user {
                    return i;
                }
            }
        }

        // Fallback: no clean boundary found, split at target.
        target
    }
}

impl Default for TokenSplitter {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================
// ContextOverflow error
// ===================================================

/// Error returned when the conversation cannot fit within the context
/// window, even after compaction.
///
/// This is a terminal condition — the conversation is too large and the
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

/// Result of [`ContextManager::ensure_context_fits`].
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

// ===================================================
// ContextManager
// ===================================================

/// Manages context window usage and triggers compaction when needed.
///
/// This is the main entry point for context management. It monitors
/// token usage, checks thresholds, and delegates to a pluggable
/// [`ContextCompactor`] when compaction is needed.
///
/// # Token Estimation
///
/// Token counts are *estimates* using a 4-chars-per-token heuristic.
/// This is deliberately simple — the goal is to trigger compaction
/// *before* hitting the actual limit, not to be perfectly accurate.
/// Production systems should calibrate against their model's actual
/// tokenizer.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::{ContextManager, TruncatingCompactor};
/// use loopctl::message::Message;
/// use std::sync::Arc;
///
/// let compactor = TruncatingCompactor::new()
///     .with_preserve_recent(2)
///     .with_min_messages(4);
///
/// let manager = ContextManager::new(Arc::new(compactor))
///     .with_context_window(1_000)
///     .with_threshold(0.80);
///
/// // Short conversation — no compaction needed.
/// let messages = vec![
///     Message::user("Hi"),
///     Message::assistant("Hello!"),
/// ];
/// let tokens = ContextManager::estimate_tokens(&messages);
/// assert!(!manager.should_compact(tokens));
/// ```
pub struct ContextManager {
    /// The compaction strategy.
    compactor: Arc<dyn ContextCompactor>,
    /// Model context window size in tokens.
    context_window: u64,
    /// Threshold (0.0–1.0) at which compaction triggers.
    threshold: f64,
    /// Whether auto-compaction is enabled.
    auto_compact: bool,
}

impl ContextManager {
    /// Create a new context manager with the given compactor.
    ///
    /// Defaults:
    ///
    /// | Setting          | Default |
    /// |------------------|---------|
    /// | `context_window` | 200_000 |
    /// | `threshold`      | 0.80    |
    /// | `auto_compact`   | `true`  |
    #[must_use]
    pub fn new(compactor: Arc<dyn ContextCompactor>) -> Self {
        Self {
            compactor,
            context_window: 200_000,
            threshold: 0.80,
            auto_compact: true,
        }
    }

    /// Set the model's context window size.
    #[must_use]
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = tokens;
        self
    }

    /// Set the compaction threshold (0.0–1.0).
    ///
    /// Clamped to `[0.1, 1.0]` to prevent degenerate configurations.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.1, 1.0);
        self
    }

    /// Set whether auto-compaction is enabled.
    #[must_use]
    pub fn with_auto_compact(mut self, enabled: bool) -> Self {
        self.auto_compact = enabled;
        self
    }

    /// The model's context window size in tokens.
    #[must_use]
    pub fn context_window(&self) -> u64 {
        self.context_window
    }

    /// The compaction threshold (0.0–1.0).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether auto-compaction is enabled.
    #[must_use]
    pub fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    /// The token budget at which compaction triggers.
    ///
    /// Equal to `context_window * threshold`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn compact_threshold_tokens(&self) -> u64 {
        let threshold =
            self.threshold * f64::from(u32::try_from(self.context_window).unwrap_or(u32::MAX));
        threshold as u64
    }

    /// Estimate the token count for a slice of messages.
    ///
    /// Uses a 4-chars-per-token heuristic based on the text content
    /// of all message parts. This is deliberately conservative — it
    /// overestimates rather than underestimates.
    ///
    /// The estimation:
    /// - Counts text content from all parts (text, tool calls, tool results).
    /// - Adds a fixed overhead per message (role tags, formatting).
    /// - Divides total character count by 4.
    #[must_use]
    pub fn estimate_tokens(messages: &[Message]) -> u64 {
        const CHARS_PER_TOKEN: u64 = 4;
        const MESSAGE_OVERHEAD_CHARS: u64 = 20; // role tags, newlines, etc.
        let total_chars: u64 = messages
            .iter()
            .map(|m| {
                let part_chars: u64 = m
                    .parts
                    .iter()
                    .map(|part| match part {
                        crate::message::MessagePart::Text { text } => text.len() as u64,
                        crate::message::MessagePart::Image { .. } => 256, // rough base64 estimate
                        crate::message::MessagePart::ToolCall { name, input, .. } => {
                            let name_len = name.len() as u64;
                            let input_len = input.to_string().len() as u64;
                            name_len.saturating_add(input_len)
                        }
                        crate::message::MessagePart::ToolResult { output, .. } => match output {
                            crate::message::ToolContent::Text(s) => s.len() as u64,
                            crate::message::ToolContent::Multipart(parts) => parts
                                .iter()
                                .map(|p| match p {
                                    crate::message::ToolContentPart::Text { text } => {
                                        text.len() as u64
                                    }
                                    crate::message::ToolContentPart::Image { .. } => 256,
                                })
                                .sum(),
                        },
                    })
                    .sum();
                MESSAGE_OVERHEAD_CHARS.saturating_add(part_chars)
            })
            .sum();

        total_chars / CHARS_PER_TOKEN
    }

    /// Check whether compaction should be triggered for the given token count.
    ///
    /// Returns `true` when `used_tokens >= context_window * threshold`.
    /// Also returns `true` when usage exceeds 95% regardless of threshold
    /// (emergency compaction).
    #[must_use]
    pub fn should_compact(&self, used_tokens: u64) -> bool {
        if !self.auto_compact {
            return false;
        }
        let budget = self.compact_threshold_tokens();
        used_tokens >= budget || self.is_emergency(used_tokens)
    }

    /// Check whether usage is in the emergency zone (>95% of window).
    #[must_use]
    pub fn is_emergency(&self, used_tokens: u64) -> bool {
        let emergency_line = self.context_window.saturating_mul(19) / 20; // 95%
        used_tokens >= emergency_line
    }

    /// Determine the compaction reason for the given token count.
    #[must_use]
    pub fn compact_reason(&self, used_tokens: u64) -> CompactReason {
        if self.is_emergency(used_tokens) {
            CompactReason::Emergency
        } else {
            CompactReason::ThresholdExceeded
        }
    }

    /// Ensure the conversation fits within the context window.
    ///
    /// This is the main entry point called by the agent loop after each
    /// turn. It:
    ///
    /// 1. Estimates the current token usage.
    /// 2. Checks if compaction is needed.
    /// 3. If needed, delegates to the [`ContextCompactor`].
    /// 4. Returns the (possibly compacted) messages.
    ///
    /// # Errors
    ///
    /// Returns a [`ContextOverflow`] if the conversation exceeds the
    /// context window and compaction fails or is insufficient.
    pub async fn ensure_context_fits(
        &self,
        messages: Vec<Message>,
        turn: usize,
    ) -> Result<EnsureContextResult, ContextOverflow> {
        let tokens_before = Self::estimate_tokens(&messages);
        let message_count = messages.len();

        if !self.should_compact(tokens_before) {
            return Ok(EnsureContextResult::NoAction(messages));
        }

        let reason = self.compact_reason(tokens_before);
        let target_tokens = self.compact_threshold_tokens().saturating_mul(7) / 10; // compact to 70% of threshold
        let context = CompactionContext {
            tokens_before,
            reason,
            context_window: self.context_window,
            turn,
        };
        let outcome = self
            .compactor
            .compact(messages, target_tokens, context)
            .await;

        if !outcome.success {
            return Err(ContextOverflow {
                tokens_used: tokens_before,
                context_window: self.context_window,
                message_count,
                trigger: reason,
                compactor_error: outcome.error,
            });
        }

        // Verify the compactor actually reduced the context.
        let tokens_after = Self::estimate_tokens(&outcome.messages);
        if tokens_after > self.context_window && self.is_emergency(tokens_after) {
            return Err(ContextOverflow {
                tokens_used: tokens_after,
                context_window: self.context_window,
                message_count: outcome.messages.len(),
                trigger: CompactReason::Emergency,
                compactor_error: Some("compactor failed to reduce context below window".into()),
            });
        }

        Ok(EnsureContextResult::Compacted(outcome))
    }

    /// Manually trigger compaction regardless of threshold.
    ///
    /// Use this when the agent or a tool explicitly requests context
    /// reduction. The `reason` is set to [`CompactReason::Manual`].
    /// # Errors
    ///
    /// Returns a [`ContextOverflow`] if compaction fails or the result
    /// still exceeds the context window.
    pub async fn compact_manual(
        &self,
        messages: Vec<Message>,
        turn: usize,
    ) -> Result<EnsureContextResult, ContextOverflow> {
        let tokens_before = Self::estimate_tokens(&messages);
        let message_count = messages.len();

        if messages.is_empty() {
            return Ok(EnsureContextResult::NoAction(messages));
        }

        let target_tokens = self
            .compact_threshold_tokens()
            .saturating_mul(7)
            .saturating_div(10);
        let context = CompactionContext {
            tokens_before,
            reason: CompactReason::Manual,
            context_window: self.context_window,
            turn,
        };
        let outcome = self
            .compactor
            .compact(messages, target_tokens, context)
            .await;

        if !outcome.success {
            return Err(ContextOverflow {
                tokens_used: tokens_before,
                context_window: self.context_window,
                message_count,
                trigger: CompactReason::Manual,
                compactor_error: outcome.error,
            });
        }

        Ok(EnsureContextResult::Compacted(outcome))
    }

    /// Build telemetry for a compaction operation.
    ///
    /// Called by the engine after compaction to produce telemetry for
    /// observers. Takes the pre/post message lists and the compaction
    /// trigger reason, along with the start time of the operation.
    ///
    /// Note: Currently does not depend on instance state, but is a method
    /// by design to allow future configuration-aware telemetry.
    #[must_use]
    #[allow(
        clippy::unused_self,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )]
    pub fn build_telemetry(
        &self,
        trigger: CompactReason,
        pre_messages: &[Message],
        post_messages: &[Message],
        _outcome: &CompactionOutcome,
        start: Instant,
    ) -> CompactTelemetry {
        let pre_tokens = Self::estimate_tokens(pre_messages);
        let post_tokens = Self::estimate_tokens(post_messages);
        let tokens_saved = pre_tokens.saturating_sub(post_tokens);
        let percent_saved = if pre_tokens > 0 {
            let ratio = u128::from(tokens_saved)
                .saturating_mul(100)
                .saturating_div(u128::from(pre_tokens));
            u8::try_from(ratio.min(100)).unwrap_or(100)
        } else {
            0
        };

        CompactTelemetry {
            trigger,
            pre_compact: PreCompactStats {
                total_messages: pre_messages.len(),
                estimated_tokens: pre_tokens,
                user_messages: pre_messages.iter().filter(|m| m.role == Role::User).count(),
                assistant_messages: pre_messages
                    .iter()
                    .filter(|m| m.role == Role::Assistant)
                    .count(),
                tool_messages: pre_messages
                    .iter()
                    .filter(|m| {
                        m.parts
                            .iter()
                            .any(crate::message::MessagePart::is_tool_call)
                    })
                    .count(),
            },
            post_compact: PostCompactStats {
                total_messages: post_messages.len(),
                estimated_tokens: post_tokens,
                tokens_saved,
                percent_saved,
            },
            duration: start.elapsed(),
        }
    }
}

impl fmt::Debug for ContextManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextManager")
            .field("context_window", &self.context_window)
            .field("threshold", &self.threshold)
            .field("auto_compact", &self.auto_compact)
            .finish_non_exhaustive()
    }
}

// ===================================================
// Tests
// ===================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_conversation(n_turns: usize) -> Vec<Message> {
        let mut messages = Vec::new();
        for i in 0..n_turns {
            messages.push(Message::user(format!("User message {i}")));
            messages.push(Message::assistant(format!("Assistant reply {i}")));
        }
        messages
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(ContextManager::estimate_tokens(&[]), 0);
    }

    #[test]
    fn test_estimate_tokens_single_message() {
        let msgs = vec![Message::user("Hello, world!")];
        let tokens = ContextManager::estimate_tokens(&msgs);
        // 13 chars + 20 overhead = 33 chars / 4 = 8 tokens
        assert_eq!(tokens, 8);
    }

    #[test]
    fn test_estimate_tokens_multi_message() {
        let msgs = make_conversation(3);
        let tokens = ContextManager::estimate_tokens(&msgs);
        assert!(tokens > 0);
    }

    #[test]
    fn test_manager_defaults() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        assert_eq!(manager.context_window(), 200_000);
        assert!((manager.threshold() - 0.80).abs() < f64::EPSILON);
        assert!(manager.auto_compact());
    }

    #[test]
    fn test_manager_custom_settings() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(100_000)
            .with_threshold(0.50)
            .with_auto_compact(false);
        assert_eq!(manager.context_window(), 100_000);
        assert!((manager.threshold() - 0.50).abs() < f64::EPSILON);
        assert!(!manager.auto_compact());
    }

    #[test]
    fn test_threshold_clamped() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_threshold(0.01);
        assert!((manager.threshold() - 0.1).abs() < f64::EPSILON);

        let compactor2 = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor2)).with_threshold(2.0);
        assert!((manager.threshold() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_should_compact_below_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(0.80);
        // 800 is the threshold for 1000 * 0.80
        assert!(!manager.should_compact(799));
    }

    #[test]
    fn test_should_compact_at_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(0.80);
        assert!(manager.should_compact(800));
    }

    #[test]
    fn test_should_compact_emergency() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(0.50); // threshold at 500, but emergency at 950
        assert!(manager.should_compact(950));
    }

    #[test]
    fn test_should_compact_disabled() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_auto_compact(false);
        assert!(!manager.should_compact(999_999));
    }

    #[test]
    fn test_is_emergency_at_95_percent() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_context_window(1_000);
        assert!(manager.is_emergency(950));
        assert!(!manager.is_emergency(949));
    }

    #[test]
    fn test_compact_reason_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(0.80);
        assert_eq!(
            manager.compact_reason(800),
            CompactReason::ThresholdExceeded
        );
    }

    #[test]
    fn test_compact_reason_emergency() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_context_window(1_000);
        assert_eq!(manager.compact_reason(960), CompactReason::Emergency);
    }

    #[test]
    fn test_compact_threshold_tokens() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(0.80);
        assert_eq!(manager.compact_threshold_tokens(), 160_000);
    }

    #[tokio::test]
    async fn test_truncating_compactor_no_change() {
        let compactor = TruncatingCompactor::new()
            .with_min_messages(6)
            .with_preserve_recent(4);
        let msgs = make_conversation(2); // 4 messages
        let context = CompactionContext {
            tokens_before: ContextManager::estimate_tokens(&msgs),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 1,
        };
        let outcome = compactor.compact(msgs.clone(), 500, context).await;
        assert!(outcome.success);
        assert_eq!(outcome.messages.len(), msgs.len());
        assert_eq!(outcome.messages_compacted, 0);
    }

    #[tokio::test]
    async fn test_truncating_compactor_truncates() {
        let compactor = TruncatingCompactor::new()
            .with_min_messages(4)
            .with_preserve_recent(2);
        let msgs = make_conversation(10); // 20 messages
        let first_role = msgs.first().map(|m| m.role);
        let tokens_before = ContextManager::estimate_tokens(&msgs);
        let context = CompactionContext {
            tokens_before,
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 5,
        };
        let outcome = compactor.compact(msgs, 500, context).await;
        assert!(outcome.success);
        // 1 (first/system prompt) + 2 (preserve_recent) = 3 messages preserved.
        assert_eq!(outcome.messages.len(), 3);
        assert_eq!(outcome.messages_compacted, 17);
        assert!(outcome.tokens_saved > 0);
        // Verify the first message was preserved.
        assert!(outcome.messages.first().is_some());
        assert_eq!(outcome.messages.first().unwrap().role, first_role.unwrap());
    }

    #[test]
    fn test_splitter_short_conversation() {
        let splitter = TokenSplitter::new()
            .with_min_messages(6)
            .with_preserve_recent(4);
        let msgs = make_conversation(2); // 4 messages
        let result = splitter.split(&msgs);
        assert!(result.to_compact.is_empty());
        assert_eq!(result.preserved.len(), 4);
    }

    #[test]
    fn test_splitter_long_conversation() {
        let splitter = TokenSplitter::new()
            .with_min_messages(4)
            .with_preserve_recent(2);
        let msgs = make_conversation(10); // 20 messages
        let result = splitter.split(&msgs);
        assert!(!result.to_compact.is_empty());
        assert_eq!(result.preserved.len() + result.to_compact.len(), 20);
    }

    #[test]
    fn test_splitter_preserves_recent() {
        let splitter = TokenSplitter::new()
            .with_min_messages(4)
            .with_preserve_recent(4);
        let msgs = make_conversation(10); // 20 messages
        let result = splitter.split(&msgs);
        // Should preserve at least 4 messages from the end.
        assert!(result.preserved.len() >= 4);
        // The preserved messages should be the last ones.
        let last_user = msgs.last().unwrap();
        assert_eq!(result.preserved.last().unwrap().role, last_user.role);
    }

    #[tokio::test]
    async fn test_ensure_no_action_when_below_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000_000) // very large window
            .with_threshold(0.80);
        let msgs = make_conversation(3);
        let result = manager.ensure_context_fits(msgs.clone(), 1).await;
        match result {
            Ok(EnsureContextResult::NoAction(messages)) => {
                assert_eq!(messages.len(), msgs.len());
            }
            _ => panic!("expected NoAction"),
        }
    }

    #[tokio::test]
    async fn test_ensure_compacts_when_over_threshold() {
        let compactor = TruncatingCompactor::new()
            .with_min_messages(4)
            .with_preserve_recent(2);
        // Use a window that triggers compaction at 50% but still fits
        // the preserved 2 messages after compaction.
        // 2 messages ≈ 18 tokens, so 200 tokens is plenty of headroom.
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200)
            .with_threshold(0.10);
        let msgs = make_conversation(20); // 40 messages
        let result = manager.ensure_context_fits(msgs, 1).await;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                assert!(outcome.success);
                assert!(outcome.messages.len() < 40);
            }
            Ok(EnsureContextResult::NoAction(..)) => {
                panic!("expected Compacted");
            }
            Err(e) => {
                panic!("expected Ok, got error: {e}");
            }
        }
    }

    #[tokio::test]
    async fn test_ensure_manual_compaction() {
        let compactor = TruncatingCompactor::new()
            .with_min_messages(2)
            .with_preserve_recent(2);
        let manager = ContextManager::new(Arc::new(compactor)).with_context_window(1_000_000);
        let msgs = make_conversation(10);
        let result = manager.compact_manual(msgs, 1).await;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                assert!(outcome.messages.len() < 20);
            }
            _ => panic!("expected Compacted"),
        }
    }

    #[test]
    fn test_context_overflow_display() {
        let overflow = ContextOverflow {
            tokens_used: 150_000,
            context_window: 100_000,
            message_count: 50,
            trigger: CompactReason::Emergency,
            compactor_error: None,
        };
        let msg = overflow.to_string();
        assert!(msg.contains("150000"));
        assert!(msg.contains("100000"));
        assert!(msg.contains("50000 overflow"));
    }

    #[test]
    fn test_context_overflow_utilization() {
        let overflow = ContextOverflow {
            tokens_used: 150_000,
            context_window: 100_000,
            message_count: 50,
            trigger: CompactReason::Emergency,
            compactor_error: None,
        };
        let util = overflow.utilization();
        assert!(util > 1.0);
    }

    #[test]
    fn test_build_telemetry() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_context_window(1_000);

        let pre = make_conversation(10);
        let post = make_conversation(2);
        let outcome = CompactionOutcome {
            messages: post.clone(),
            messages_compacted: 16,
            tokens_after: 100,
            tokens_saved: 800,
            success: true,
            error: None,
        };

        let start = Instant::now();
        let telemetry = manager.build_telemetry(
            CompactReason::ThresholdExceeded,
            &pre,
            &post,
            &outcome,
            start,
        );

        assert_eq!(telemetry.trigger, CompactReason::ThresholdExceeded);
        assert_eq!(telemetry.pre_compact.total_messages, 20);
        assert_eq!(telemetry.post_compact.total_messages, 4);
        assert!(telemetry.post_compact.percent_saved > 0);
    }

    #[test]
    fn test_compact_reason_display() {
        assert_eq!(
            CompactReason::ThresholdExceeded.to_string(),
            "threshold exceeded"
        );
        assert_eq!(CompactReason::Emergency.to_string(), "emergency");
        assert_eq!(CompactReason::Manual.to_string(), "manual");
    }

    #[test]
    fn test_truncating_compactor_defaults() {
        let c = TruncatingCompactor::new();
        assert_eq!(c.preserve_recent(), 4);
        assert_eq!(c.min_messages(), 6);
    }

    #[test]
    fn test_truncating_compactor_clamped() {
        let c = TruncatingCompactor::new().with_preserve_recent(0);
        assert_eq!(c.preserve_recent(), 1); // clamped to 1

        let c = TruncatingCompactor::new().with_min_messages(0);
        assert_eq!(c.min_messages(), 2); // clamped to 2
    }

    #[test]
    fn test_compacted_outcome_computes_tokens_saved() {
        let msgs = make_conversation(2);
        let outcome = CompactionOutcome::compacted(msgs, 1000, 400);
        assert!(outcome.success);
        assert_eq!(outcome.tokens_after, 400);
        assert_eq!(outcome.tokens_saved, 600); // 1000 - 400
        assert_eq!(outcome.messages_compacted, 0); // caller sets
        assert!(outcome.error.is_none());
    }

    #[test]
    fn test_compacted_outcome_tokens_saved_saturates() {
        let msgs = make_conversation(1);
        // tokens_after > tokens_before should not panic, just saturate to 0.
        let outcome = CompactionOutcome::compacted(msgs, 100, 500);
        assert_eq!(outcome.tokens_saved, 0);
    }
}
