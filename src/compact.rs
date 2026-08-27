//! Context management and compaction for agent conversations.
//!
//! As conversations grow, they approach the model's context window limit.
//! Infrastructure to detect when compaction is needed and to carry it out
//! through a pluggable strategy.
//!
//! # Architecture
//!
//! The design separates when to compact from how to compact:
//!
//! 1. [`ContextManager`] monitors token usage, checks thresholds, and
//!    decides when to trigger compaction.
//! 2. [`ContextCompactor`] is the trait that defines the compaction
//!    strategy. Plug in truncation, summarization, or any custom approach.
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
//!     .with_threshold(80);
//!
//! assert_eq!(manager.context_window(), 200_000);
//! assert_eq!(manager.threshold(), 80);
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

/// Strategy for estimating the token cost of a message slice.
///
/// The engine uses a token counter in two places: the driver estimates the
/// context size after each model response (to decide whether to trigger
/// compaction), and the compactor estimates the size of the compacted
/// history (to report `tokens_after` and verify progress). Both must use the
/// same counter so the before/after comparison is consistent — mixing a
/// provider's billed token count with a heuristic estimate causes
/// compaction flapping.
///
/// The default implementation, [`HeuristicTokenCounter`], uses a
/// characters-per-token ratio and is intentionally conservative (it
/// overestimates rather than underestimates). For production accuracy,
/// implement this trait on top of a real tokenizer (e.g. `tiktoken` for
/// OpenAI) and attach it via
/// [`BareLoop::set_token_counter`](crate::engine::BareLoop::set_token_counter)
/// and
/// [`ContextManager::with_token_counter`](ContextManager::with_token_counter).
///
/// # Example
///
/// ```rust
/// use loopctl::compact::{TokenCounter, HeuristicTokenCounter};
/// use loopctl::message::Message;
///
/// let counter = HeuristicTokenCounter;
/// let tokens = counter.count(&[Message::user("hello world")]);
/// assert!(tokens > 0);
/// ```
pub trait TokenCounter: Send + Sync {
    /// Estimate the token count for a slice of messages.
    ///
    /// The count should include all message content the counter considers
    /// part of the context — text, tool calls, tool results — but the exact
    /// set depends on the implementation. The only requirement is
    /// consistency: the same counter must be used on both the trigger and
    /// the post-compaction path.
    fn count(&self, messages: &[Message]) -> u64;
}

/// A zero-dependency token estimator using a characters-per-token ratio.
///
/// Counts the character length of all message parts (text, tool calls, tool
/// results), adds a fixed per-message overhead for role tags and formatting,
/// and divides by 4 (`CHARS_PER_TOKEN`). Conservative — overestimates rather
/// than underestimates, so compaction triggers slightly early rather than
/// late. Accuracy is roughly ±30% on real content; for production use,
/// swap in a real tokenizer via the [`TokenCounter`] trait.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::{HeuristicTokenCounter, TokenCounter};
/// use loopctl::message::Message;
///
/// let counter = HeuristicTokenCounter;
/// let tokens = counter.count(&[
///     Message::user("hello"),
///     Message::assistant("hi there"),
/// ]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, messages: &[Message]) -> u64 {
        const CHARS_PER_TOKEN: u64 = 4;
        const MESSAGE_OVERHEAD_CHARS: u64 = 20;
        let total_chars: u64 = messages
            .iter()
            .map(|m| {
                let part_chars: u64 = m
                    .parts
                    .iter()
                    .map(|p| match p {
                        MessagePart::Text { text } => text.chars().count() as u64,
                        MessagePart::Image { .. } => 256,
                        MessagePart::ToolCall { name, input, .. } => (name.chars().count() as u64)
                            .saturating_add(input.to_string().chars().count() as u64),
                        MessagePart::ToolResult { output, .. } => match output {
                            crate::message::ToolContent::Text(s) => s.chars().count() as u64,
                            crate::message::ToolContent::Multipart(parts) => parts
                                .iter()
                                .map(|p| match p {
                                    crate::message::ToolContentPart::Text { text } => {
                                        text.chars().count() as u64
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
}

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
    /// * `target_tokens` — The target token count for the compacted
    ///   output, already reduced by any budget reserved for content
    ///   riding the request alongside the history (see
    ///   [`ContextManager::compact_with_reason`](ContextManager::compact_with_reason)),
    ///   so fitting the target leaves room for it.
    /// * `context` — Metadata about the compaction trigger.
    // The return-type boxing is required for object safety.
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>>;
}

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
///     .with_threshold(80)               // triggers at 160k tokens
///     .with_compact_target(CompactBase::Context)   // target = % of 200k
///     .with_compact_target_pct(50);     // compact to 50% of 200k = 100k
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompactBase {
    /// Target is a percentage of the full context window.
    ///
    /// `target = context_window × compact_target_pct / 100`
    ///
    /// Use this when you want compaction to aim for a fixed fraction
    /// of the model's total capacity regardless of the trigger threshold.
    Context,

    /// Target is a percentage of the trigger threshold.
    ///
    /// `target = compact_threshold_tokens × compact_target_pct / 100`
    ///
    /// This is the default. With the default `threshold = 80` (80%) and
    /// `compact_target_pct = 70` (70%), compaction targets 56% of the
    /// context window (`0.8 × 0.7 = 0.56`).
    #[default]
    Threshold,
}

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
///     .with_threshold(80);
///
/// // Short conversation — no compaction needed.
/// let messages = vec![
///     Message::user("Hi"),
///     Message::assistant("Hello!"),
/// ];
/// let tokens = manager.estimate_tokens(&messages);
/// assert!(!manager.should_compact(tokens));
/// ```
#[derive(Clone)]
pub struct ContextManager {
    /// The compaction strategy.
    ///
    /// The [`ContextCompactor`](super::compact::ContextCompactor) implementation
    /// that performs the actual summarization or truncation when compaction runs.
    /// The manager decides *when* and *how much* to compact; this field decides
    /// *how* the messages are rewritten.
    compactor: Arc<dyn ContextCompactor>,

    /// Model context window size in tokens.
    ///
    /// The hard upper bound on tokens the model accepts in one request. Every
    /// threshold and target is expressed as a fraction of this value, so it is
    /// the denominator for [`compact_threshold_tokens`](Self::compact_threshold_tokens)
    /// and the emergency-zone and compaction-target calculations.
    context_window: u64,

    /// Threshold as a percentage (0–100) at which compaction triggers.
    ///
    /// When estimated token usage reaches `threshold * context_window / 100`,
    /// [`should_compact`](Self::should_compact) returns `true` and a compaction
    /// pass runs before the next turn. Set via [`with_threshold`](Self::with_threshold),
    /// which clamps to `[1, 100]`; defaults to `80`.
    threshold: u8,

    /// Whether auto-compaction is enabled.
    ///
    /// When `false`, [`should_compact`](Self::should_compact) always returns
    /// `false` and [`ensure_context_fits`](Self::ensure_context_fits) never
    /// triggers compaction — the host must manage context size manually (useful
    /// in tests or fixed-length sessions). Toggle via
    /// [`with_auto_compact`](Self::with_auto_compact).
    auto_compact: bool,

    /// The base used to compute the compaction target.
    ///
    /// Whether [`compact_target_tokens`](Self::compact_target_tokens) measures
    /// the post-compaction size against the trigger threshold
    /// ([`CompactBase::Threshold`], the default) or the full context window
    /// ([`CompactBase::Context`]). Set via
    /// [`with_compact_target`](Self::with_compact_target).
    compact_base: CompactBase,

    /// Fraction of the target base to compact down to, as a percentage (0–100).
    ///
    /// The post-compaction size aimed for: `compact_target * base / 100`,
    /// where `base` is determined by [`compact_base`](Self::compact_base).
    /// Defaults to `70` (70%); set and clamped to `[1, 100]` via
    /// [`with_compact_target_pct`](Self::with_compact_target_pct).
    compact_target: u8,

    /// Token counter used for all size estimates.
    ///
    /// Both the compaction trigger (checked by the driver before each model
    /// call) and the post-compaction verification use this counter, ensuring
    /// the before/after comparison is consistent. Defaults to
    /// [`HeuristicTokenCounter`]; swap in a real tokenizer via
    /// [`with_token_counter`](Self::with_token_counter).
    token_counter: Arc<dyn TokenCounter>,
}

impl ContextManager {
    /// Create a new context manager with the given compactor.
    ///
    /// Defaults:
    ///
    /// | Setting              | Default                      |
    /// |----------------------|------------------------------|
    /// | `context_window`     | 200_000                      |
    /// | `threshold`          | 80 (80%)                     |
    /// | `auto_compact`       | `true`                       |
    /// | `compact_target`     | [`CompactBase::Threshold`]   |
    /// | `compact_target_pct` | 70 (70%)                     |
    /// | `token_counter`      | [`HeuristicTokenCounter`]    |
    #[must_use]
    pub fn new(compactor: Arc<dyn ContextCompactor>) -> Self {
        Self {
            compactor,
            context_window: 200_000,
            threshold: 80,
            auto_compact: true,
            compact_base: CompactBase::Threshold,
            compact_target: 70,
            token_counter: Arc::new(HeuristicTokenCounter),
        }
    }

    /// Set the token counter used for size estimates (builder-style).
    ///
    /// Both the compaction trigger and the post-compaction verification will
    /// use this counter. Use this to plug in a real tokenizer (e.g.
    /// `tiktoken` for OpenAI) for more accurate estimates than the default
    /// [`HeuristicTokenCounter`].
    #[must_use]
    pub fn with_token_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = counter;
        self
    }

    /// Borrow the token counter.
    ///
    /// Exposed so the driver can use the same counter for its pre-model
    /// estimate, keeping the trigger and post-compaction paths consistent.
    #[must_use]
    pub fn token_counter(&self) -> &Arc<dyn TokenCounter> {
        &self.token_counter
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

    /// Set the compaction threshold as a percentage (0–100).
    ///
    /// Clamped to `[1, 100]` to prevent degenerate configurations.
    #[must_use]
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold.clamp(1, 100);
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

    /// Set the fraction of the target base to compact down to, as a percentage
    /// (`0–100`; `100` = 100%).
    ///
    /// Clamped to `[1, 100]` to prevent degenerate configurations.
    /// Defaults to `70`.
    #[must_use]
    pub fn with_compact_target_pct(mut self, pct: u8) -> Self {
        self.compact_target = pct.clamp(1, 100);
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

    /// The compaction threshold as a percentage (0–100).
    ///
    /// Compaction triggers when estimated tokens reach
    /// `context_window * threshold / 10_000`. See [`with_threshold`](Self::with_threshold).
    #[must_use]
    pub fn threshold(&self) -> u8 {
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

    /// The fraction of the target base to compact down to, as a percentage
    /// (`0–100`; `100` = 100%).
    ///
    /// See [`with_compact_target_pct`](Self::with_compact_target_pct).
    #[must_use]
    pub fn compact_target_pct(&self) -> u8 {
        self.compact_target
    }

    /// The token budget at which compaction triggers.
    ///
    /// Equal to `context_window * threshold`.
    #[must_use]
    pub fn compact_threshold_tokens(&self) -> u64 {
        self.context_window
            .saturating_mul(u64::from(self.threshold))
            / 100
    }

    /// The token count to compact down to.
    ///
    /// Computed from [`compact_target`](Self::compact_target) and
    /// [`compact_target_pct`](Self::compact_target_pct):
    ///
    /// - [`CompactBase::Threshold`]: `compact_threshold_tokens × pct / 100`
    /// - [`CompactBase::Context`]: `context_window × pct / 100`
    #[must_use]
    pub fn compact_target_tokens(&self) -> u64 {
        let base: u64 = match self.compact_base {
            CompactBase::Threshold => self.compact_threshold_tokens(),
            CompactBase::Context => self.context_window,
        };
        base.saturating_mul(u64::from(self.compact_target)) / 100
    }

    /// Estimate the token count for a slice of messages using the
    /// configured [`TokenCounter`].
    ///
    /// Delegates to [`token_counter`](Self::token_counter), so both the
    /// compaction trigger (driver-side) and the post-compaction check
    /// (compactor-side) use the same estimation strategy.
    #[must_use]
    pub fn estimate_tokens(&self, messages: &[Message]) -> u64 {
        self.token_counter.count(messages)
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
    /// Self-driven entry point for callers that manage the conversation
    /// themselves: it decides whether compaction is needed and runs it in
    /// one call. It:
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
        let tokens_before = self.estimate_tokens(&messages);

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
            counter: Arc::clone(&self.token_counter),
            instructions: None,
            additional_context: Vec::new(),
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

        let tokens_after = self.estimate_tokens(&outcome.messages);
        if tokens_after > self.context_window {
            return Err(ContextOverflow {
                tokens_used: tokens_after,
                context_window: self.context_window,
                message_count: outcome.messages.len(),
                trigger: CompactReason::Emergency,
                compactor_error: Some("compactor failed to reduce context below window".into()),
            });
        }

        if outcome.messages.len() == message_count && tokens_after >= tokens_before {
            return Ok(EnsureContextResult::NoAction(outcome.messages));
        }
        let mut normalized = outcome;
        normalized.tokens_after = tokens_after;
        normalized.tokens_saved = tokens_before.saturating_sub(tokens_after);
        Ok(EnsureContextResult::Compacted(normalized))
    }

    /// Manually trigger compaction regardless of threshold.
    ///
    /// Use this when the agent or a tool explicitly requests context
    /// reduction. The `reason` is set to [`CompactReason::Manual`]. A
    /// successful pass is classified with this manager's configured
    /// [`TokenCounter`], not the compactor's self-report: when the message
    /// count is unchanged and the measured token count did not shrink, the
    /// pass is reported as [`EnsureContextResult::NoAction`] — compaction
    /// did not occur, so compaction observers and hooks stay silent. On the
    /// [`EnsureContextResult::Compacted`] path the outcome's token fields
    /// are normalized to the same counter, so telemetry reflects the
    /// manager's measurements.
    /// # Errors
    ///
    /// Returns a [`ContextOverflow`] if compaction fails or the result
    /// still exceeds the context window.
    pub async fn compact_manual(
        &self,
        messages: Vec<Message>,
        turn: usize,
    ) -> Result<EnsureContextResult, ContextOverflow> {
        self.compact_with_reason(messages, turn, CompactReason::Manual, None, Vec::new(), 0)
            .await
    }

    /// Trigger compaction with a specific reason, bypassing the threshold check.
    ///
    /// Use this when the decision to compact was already made (e.g. by the
    /// driving state machine) and the `reason` should reach the compactor's
    /// [`CompactionContext`] so it can vary strategy (e.g. more aggressive
    /// summarization for [`CompactReason::Emergency`]). A successful pass
    /// is classified with this manager's configured [`TokenCounter`], not
    /// the compactor's self-report: when the message count is unchanged and
    /// the measured token count did not shrink, the pass is reported as
    /// [`EnsureContextResult::NoAction`] — compaction did not occur, so
    /// compaction observers and hooks stay silent. On the
    /// [`EnsureContextResult::Compacted`] path the outcome's token fields
    /// are normalized to the same counter, so telemetry reflects the
    /// manager's measurements.
    ///
    /// `reserved_tokens` is budget the compacted history must leave room
    /// for — per-request overhead plus, when compaction serves a deferred
    /// turn, that turn's transient messages: both the compaction target
    /// and the fit check subtract it, so the post-compaction history plus
    /// the reserve fits the window. Pass `0` when nothing rides the
    /// request beyond the history.
    ///
    /// # Errors
    ///
    /// Returns a [`ContextOverflow`] if compaction fails or the result
    /// still exceeds the context window.
    pub async fn compact_with_reason(
        &self,
        messages: Vec<Message>,
        turn: usize,
        reason: CompactReason,
        instructions: Option<String>,
        additional_context: Vec<String>,
        reserved_tokens: u64,
    ) -> Result<EnsureContextResult, ContextOverflow> {
        let tokens_before = self.estimate_tokens(&messages);

        if messages.is_empty() {
            return Ok(EnsureContextResult::NoAction(messages));
        }

        let message_count = messages.len();
        let target_tokens = self.compact_target_tokens().saturating_sub(reserved_tokens);
        let context = CompactionContext {
            tokens_before,
            reason,
            context_window: self.context_window,
            turn,
            counter: Arc::clone(&self.token_counter),
            instructions,
            additional_context,
        };
        let outcome = self
            .compactor
            .compact(messages, target_tokens, context)
            .await;

        if !outcome.success {
            return Err(ContextOverflow {
                tokens_used: tokens_before.saturating_add(reserved_tokens),
                context_window: self.context_window,
                message_count,
                trigger: reason,
                compactor_error: outcome.error,
            });
        }

        let tokens_after = self.estimate_tokens(&outcome.messages);
        if tokens_after > self.context_window.saturating_sub(reserved_tokens) {
            return Err(ContextOverflow {
                tokens_used: tokens_after.saturating_add(reserved_tokens),
                context_window: self.context_window,
                message_count,
                trigger: reason,
                compactor_error: None,
            });
        }

        if outcome.messages.len() == message_count && tokens_after >= tokens_before {
            return Ok(EnsureContextResult::NoAction(outcome.messages));
        }
        let mut normalized = outcome;
        normalized.tokens_after = tokens_after;
        normalized.tokens_saved = tokens_before.saturating_sub(tokens_after);
        Ok(EnsureContextResult::Compacted(normalized))
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
    pub fn build_telemetry(
        trigger: CompactReason,
        pre_messages: &[Message],
        post_messages: &[Message],
        start: Instant,
    ) -> CompactTelemetry {
        let pre_tokens = CompactionOutcome::estimate_tokens(pre_messages);
        let post_tokens = CompactionOutcome::estimate_tokens(post_messages);
        let tokens_saved = pre_tokens.saturating_sub(post_tokens);
        let percent_saved: u8 = tokens_saved
            .checked_mul(100)
            .and_then(|v| v.checked_div(pre_tokens))
            .map_or(0, |ratio| ratio.min(100) as u8);

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
            .field("compact_target_pct", &self.compact_target_pct())
            .finish_non_exhaustive()
    }
}

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
        let manager = ContextManager::new(Arc::new(TruncatingCompactor::new()));
        assert_eq!(manager.estimate_tokens(&[]), 0);
    }

    #[test]
    fn test_estimate_tokens_single_message() {
        let manager = ContextManager::new(Arc::new(TruncatingCompactor::new()));
        let msgs = vec![Message::user("Hello, world!")];
        let tokens = manager.estimate_tokens(&msgs);
        // 13 chars + 20 overhead = 33 chars / 4 = 8 tokens (integer division)
        assert_eq!(tokens, 8);
    }

    #[test]
    fn test_estimate_tokens_multi_message() {
        let manager = ContextManager::new(Arc::new(TruncatingCompactor::new()));
        let msgs = make_conversation(3);
        let tokens = manager.estimate_tokens(&msgs);
        assert!(tokens > 0);
    }

    #[test]
    fn test_manager_defaults() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        assert_eq!(manager.context_window(), 200_000);
        assert_eq!(manager.threshold(), 80);
        assert!(manager.auto_compact());
    }

    #[test]
    fn test_manager_custom_settings() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(100_000)
            .with_threshold(50)
            .with_auto_compact(false);
        assert_eq!(manager.context_window(), 100_000);
        assert_eq!(manager.threshold(), 50);
        assert!(!manager.auto_compact());
    }

    #[test]
    fn test_threshold_clamped() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_threshold(1);
        assert_eq!(manager.threshold(), 1);

        let compactor2 = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor2)).with_threshold(200);
        assert_eq!(manager.threshold(), 100);
    }

    #[test]
    fn test_should_compact_below_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(80);
        // 800 is the threshold for 1000 * 8000 / 10_000
        assert!(!manager.should_compact(799));
    }

    #[test]
    fn test_should_compact_at_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(80);
        assert!(manager.should_compact(800));
    }

    #[test]
    fn test_should_compact_emergency() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(1_000)
            .with_threshold(50); // threshold at 500, but emergency at 950
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
            .with_threshold(80);
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
            .with_threshold(80);
        assert_eq!(manager.compact_threshold_tokens(), 160_000);
    }

    #[test]
    fn test_compact_target_tokens_default() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(80);
        assert_eq!(manager.compact_target_tokens(), 112_000);
    }

    #[test]
    fn test_compact_target_tokens_context_base() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(80)
            .with_compact_target(CompactBase::Context)
            .with_compact_target_pct(50);
        assert_eq!(manager.compact_target_tokens(), 100_000);
    }

    #[test]
    fn test_compact_target_tokens_threshold_base() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200_000)
            .with_threshold(80)
            .with_compact_target(CompactBase::Threshold)
            .with_compact_target_pct(50);
        assert_eq!(manager.compact_target_tokens(), 80_000);
    }

    #[test]
    fn test_compact_target_pct_clamped() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor)).with_compact_target_pct(1);
        assert_eq!(manager.compact_target_pct(), 1);

        let compactor2 = TruncatingCompactor::new();
        let manager2 = ContextManager::new(Arc::new(compactor2)).with_compact_target_pct(200);
        assert_eq!(manager2.compact_target_pct(), 100);
    }

    #[test]
    fn test_compact_target_default_is_threshold() {
        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        assert_eq!(manager.compact_target(), CompactBase::Threshold);
        assert_eq!(manager.compact_target_pct(), 70);
    }

    #[tokio::test]
    async fn test_truncating_compactor_no_change() {
        let compactor = TruncatingCompactor::new()
            .with_min_messages(6)
            .with_preserve_recent(4);
        let msgs = make_conversation(2); // 4 messages
        let context = CompactionContext {
            tokens_before: CompactionOutcome::estimate_tokens(&msgs),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 1,
            counter: Arc::new(HeuristicTokenCounter),

            instructions: None,
            additional_context: Vec::new(),
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
        let tokens_before = CompactionOutcome::estimate_tokens(&msgs);
        let context = CompactionContext {
            tokens_before,
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 5,
            counter: Arc::new(HeuristicTokenCounter),

            instructions: None,
            additional_context: Vec::new(),
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
            .with_threshold(80);
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
        // Use a window that triggers compaction at 10% but still fits
        // the preserved 2 messages after compaction.
        // 2 messages ≈ 18 tokens, so 200 tokens is plenty of headroom.
        let manager = ContextManager::new(Arc::new(compactor))
            .with_context_window(200)
            .with_threshold(10);
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
        let _manager = ContextManager::new(Arc::new(compactor)).with_context_window(1_000);

        let pre = make_conversation(10);
        let post = make_conversation(2);
        let _outcome = CompactionOutcome {
            messages: post.clone(),
            tokens_after: 100,
            tokens_saved: 800,
            success: true,
            error: None,
        };

        let start = Instant::now();
        let telemetry =
            ContextManager::build_telemetry(CompactReason::ThresholdExceeded, &pre, &post, start);

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

    #[tokio::test]
    async fn no_change_pass_is_not_reported_as_compacted() {
        use crate::compact::truncating::TruncatingCompactor;
        use crate::message::Message;

        let manager = ContextManager::new(std::sync::Arc::new(
            TruncatingCompactor::new().with_min_messages(10),
        ));
        let messages: Vec<Message> = (0..3).map(|i| Message::user(format!("msg {i}"))).collect();
        let result = manager
            .compact_manual(messages.clone(), 1)
            .await
            .expect("manual compaction must not error");
        if let EnsureContextResult::Compacted(outcome) = result {
            assert!(
                outcome.messages.len() < messages.len() || outcome.tokens_saved > 0,
                "doc: Compacted means compaction occurred and produced a shorter message list — got identical list with zero savings"
            );
        }
    }
}
