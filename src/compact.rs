//! Context management and compaction for agent conversations.
//!
//! As conversations grow, they approach the model's context window limit.
//! Infrastructure to detect when compaction is needed and to carry it out
//! through a pluggable strategy.
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
//! ┌────────────────────────────┐
//! │       ContextManager       │
//! │                            │
//! │     estimate_tokens()      │
//! │     should_compact()       │
//! │     ensure_context_fits()  │
//! │             │              │
//! │             ▼              │
//! │  ┌──────────────────────┐  │
//! │  │  dyn ContextCompactor│  │
//! │  │  .compact()          │  │
//! │  └──────────────────────┘  │
//! └────────────────────────────┘
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

use crate::message::{Message, MessagePart, Role};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

pub mod truncating;
pub mod types;

pub use truncating::{SplitResult, TokenSplitter, TruncatingCompactor};
pub use types::{
    CompactReason, CompactTelemetry, CompactionContext, CompactionOutcome, ContextOverflow,
    EnsureContextResult, PostCompactStats, PreCompactStats,
};

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
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>>;
}

// ===================================================
// CompactBase
// ===================================================

/// Determines the base used to calculate the compaction target.
///
/// When compaction triggers, the manager asks the compactor to reduce
/// the conversation to some target token count. This enum controls
/// *what* that target is a percentage *of*:
///
/// - [`Threshold`](CompactBase::Threshold): the target is a percentage
///   of the trigger threshold (`context_window × threshold`).
/// - [`Context`](CompactBase::Context): the target is a percentage
///   of the full context window.
///
/// The percentage itself is configured via
/// [`with_compact_target_pct`](ContextManager::with_compact_target_pct).
///
/// # Example
///
/// ```rust
/// use loopctl::compact::{CompactBase, ContextManager, TruncatingCompactor};
/// use std::sync::Arc;
///
/// let compactor = TruncatingCompactor::new();
/// let manager = ContextManager::new(Arc::new(compactor))
///     .with_context_window(200_000)
///     .with_threshold(0.80)                // triggers at 160k tokens
///     .with_compact_target(CompactBase::Context)   // target = % of 200k
///     .with_compact_target_pct(0.50);      // compact to 50% of 200k = 100k
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompactBase {
    /// Target is a percentage of the full context window.
    ///
    /// `target = context_window × compact_target_pct`
    ///
    /// Use this when you want compaction to aim for a fixed fraction
    /// of the model's total capacity regardless of the trigger threshold.
    Context,

    /// Target is a percentage of the trigger threshold.
    ///
    /// `target = compact_threshold_tokens × compact_target_pct`
    ///
    /// This is the default. With the default `threshold = 0.80` and
    /// `compact_target_pct = 0.70`, compaction targets 56% of the
    /// context window (0.80 × 0.70 = 0.56).
    #[default]
    Threshold,
}

// ===================================================
// ContextManager
// ===================================================

/// Manages context window usage and triggers compaction when needed.
///
/// Main entry point for context management. It monitors
/// token usage, checks thresholds, and delegates to a pluggable
/// [`ContextCompactor`] when compaction is needed.
///
/// # Token Estimation
///
/// Token counts are *estimates* using a 4-chars-per-token heuristic.
/// Deliberately simple — the goal is to trigger compaction
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
    /// The base used to compute the compaction target.
    compact_base: CompactBase,
    /// The fraction (0.0–1.0) of the target base to compact down to.
    compact_target: f64,
}

impl ContextManager {
    /// Create a new context manager with the given compactor.
    ///
    /// Defaults:
    ///
    /// | Setting              | Default                      |
    /// |----------------------|------------------------------|
    /// | `context_window`     | 200_000                      |
    /// | `threshold`          | 0.80                         |
    /// | `auto_compact`       | `true`                       |
    /// | `compact_target`     | [`CompactBase::Threshold`] |
    /// | `compact_target_pct` | 0.70                         |
    #[must_use]
    pub fn new(compactor: Arc<dyn ContextCompactor>) -> Self {
        Self {
            compactor,
            context_window: 200_000,
            threshold: 0.80,
            auto_compact: true,
            compact_base: CompactBase::Threshold,
            compact_target: 0.70,
        }
    }

    /// Set the model's context window size.
    ///
    /// Determines the upper bound on estimated tokens the manager
    /// will allow before triggering compaction.
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
    ///
    /// When disabled, [`should_compact`](Self::should_compact) always
    /// returns `false` and [`ensure_context_fits`](Self::ensure_context_fits)
    /// will never trigger compaction.
    #[must_use]
    pub fn with_auto_compact(mut self, enabled: bool) -> Self {
        self.auto_compact = enabled;
        self
    }

    /// Set the base used to compute the compaction target.
    ///
    /// See [`CompactBase`] for details. Defaults to
    /// [`CompactBase::Threshold`].
    #[must_use]
    pub fn with_compact_target(mut self, target: CompactBase) -> Self {
        self.compact_base = target;
        self
    }

    /// Set the fraction (0.0–1.0) of the target base to compact down to.
    ///
    /// Clamped to `[0.1, 1.0]` to prevent degenerate configurations.
    /// Defaults to `0.70` (70%).
    #[must_use]
    pub fn with_compact_target_pct(mut self, pct: f64) -> Self {
        self.compact_target = pct.clamp(0.1, 1.0);
        self
    }

    /// The model's context window size in tokens.
    ///
    /// This is the upper limit the manager uses to decide when compaction
    /// is necessary. See [`with_context_window`](Self::with_context_window).
    #[must_use]
    pub fn context_window(&self) -> u64 {
        self.context_window
    }

    /// The compaction threshold (0.0–1.0).
    ///
    /// Compaction triggers when estimated tokens reach
    /// `context_window * threshold`. See [`with_threshold`](Self::with_threshold).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether auto-compaction is enabled.
    ///
    /// When `false`, the manager never triggers compaction automatically.
    /// See [`with_auto_compact`](Self::with_auto_compact).
    #[must_use]
    pub fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    /// The base used to compute the compaction target.
    ///
    /// See [`CompactBase`] and [`with_compact_target`](Self::with_compact_target).
    #[must_use]
    pub fn compact_target(&self) -> CompactBase {
        self.compact_base
    }

    /// The fraction (0.0–1.0) of the target base to compact down to.
    ///
    /// See [`with_compact_target_pct`](Self::with_compact_target_pct).
    #[must_use]
    pub fn compact_target_pct(&self) -> f64 {
        self.compact_target
    }

    /// The token budget at which compaction triggers.
    ///
    /// Equal to `context_window * threshold`. The result is always
    /// non-negative (percentage × positive count), so the f64→u64
    /// cast is safe in practice.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn compact_threshold_tokens(&self) -> u64 {
        (self.threshold * self.context_window as f64) as u64
    }

    /// The token count to compact down to.
    ///
    /// Computed from [`compact_target`](Self::compact_target) and
    /// [`compact_target_pct`](Self::compact_target_pct):
    ///
    /// - [`CompactBase::Threshold`]: `compact_threshold_tokens × pct`
    /// - [`CompactBase::Context`]: `context_window × pct`
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn compact_target_tokens(&self) -> u64 {
        let base: u64 = match self.compact_base {
            CompactBase::Threshold => self.compact_threshold_tokens(),
            CompactBase::Context => self.context_window,
        };
        (self.compact_target * base as f64) as u64
    }

    /// Estimate the token count for a slice of messages.
    ///
    /// Uses a 4-chars-per-token heuristic based on the text content
    /// of all message parts. Conservative — overestimates rather than underestimates.
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
                    .map(|p| match p {
                        MessagePart::Text { text } => text.chars().count() as u64,
                        MessagePart::Image { .. } => 256, // rough base64 estimate
                        MessagePart::ToolCall { name, input, .. } => {
                            let name_len = name.len() as u64;
                            let input_len = input.to_string().len() as u64;
                            name_len.saturating_add(input_len)
                        }
                        MessagePart::ToolResult { output, .. } => match output {
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
    ///
    /// Emergency compaction is more aggressive because the context is
    /// dangerously close to overflowing the model's window.
    #[must_use]
    pub fn is_emergency(&self, used_tokens: u64) -> bool {
        let emergency_line = self.context_window.saturating_mul(19) / 20; // 95%
        used_tokens >= emergency_line
    }

    /// Determine the compaction reason for the given token count.
    ///
    /// Returns [`CompactReason::Emergency`] when usage exceeds 95% of
    /// the window, or [`CompactReason::ThresholdExceeded`] otherwise.
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
    /// Main entry point called by the agent loop after each
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

        if !self.should_compact(tokens_before) {
            return Ok(EnsureContextResult::NoAction(messages));
        }

        let message_count = messages.len();
        let reason = self.compact_reason(tokens_before);
        let target_tokens = self.compact_target_tokens();
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

        let tokens_after = Self::estimate_tokens(&outcome.messages);
        if tokens_after > self.context_window {
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

        if messages.is_empty() {
            return Ok(EnsureContextResult::NoAction(messages));
        }

        let message_count = messages.len();
        let target_tokens = self.compact_target_tokens();
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

        if outcome.tokens_after > self.context_window {
            return Err(ContextOverflow {
                tokens_used: outcome.tokens_after,
                context_window: self.context_window,
                message_count,
                trigger: CompactReason::Manual,
                compactor_error: None,
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
                        m.parts.iter().any(|p| {
                            crate::message::MessagePart::is_tool_call(p)
                                || crate::message::MessagePart::is_tool_result(p)
                        })
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
            .field("threshold", &self.threshold())
            .field("auto_compact", &self.auto_compact)
            .field("compact_target", &self.compact_base)
            .field("compact_target_pct", &self.compact_target)
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

    #[test]
    fn test_compact_target_tokens_default() {
        // Default: CompactBase::Threshold with pct 0.70
        // threshold = 200_000 * 0.80 = 160_000
        // target   = 160_000 * 0.70 = 112_000
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(0.80);
        assert_eq!(manager.compact_target_tokens(), 112_000);
    }

    #[test]
    fn test_compact_target_tokens_context_base() {
        // CompactBase::Context with pct 0.50
        // target = 200_000 * 0.50 = 100_000
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(0.80)
            .with_compact_target(CompactBase::Context)
            .with_compact_target_pct(0.50);
        assert_eq!(manager.compact_target_tokens(), 100_000);
    }

    #[test]
    fn test_compact_target_tokens_threshold_base() {
        // CompactBase::Threshold with pct 0.50
        // threshold = 200_000 * 0.80 = 160_000
        // target   = 160_000 * 0.50 = 80_000
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(0.80)
            .with_compact_target(CompactBase::Threshold)
            .with_compact_target_pct(0.50);
        assert_eq!(manager.compact_target_tokens(), 80_000);
    }

    #[test]
    fn test_compact_target_pct_clamped() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_compact_target_pct(0.01);
        assert!((manager.compact_target_pct() - 0.1).abs() < f64::EPSILON);

        let compactor2 = TruncatingCompactor::new();
        let manager2 = ContextManager::new(Arc::new(compactor2)).with_compact_target_pct(2.0);
        assert!((manager2.compact_target_pct() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compact_target_default_is_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        assert_eq!(manager.compact_target(), CompactBase::Threshold);
        assert!((manager.compact_target_pct() - 0.70).abs() < f64::EPSILON);
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
        assert!(outcome.tokens_saved > 0);
        // Verify the first message was preserved.
        assert!(!outcome.messages.is_empty());
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
