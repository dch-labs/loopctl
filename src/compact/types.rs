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

use crate::compact::TokenCounter;
use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Why compaction was triggered.
///
/// Different triggers may warrant different compaction strategies.
/// For example, an [`Emergency`](CompactReason::Emergency) compaction
/// should be more aggressive than a routine threshold check.
/// `#[non_exhaustive]` so new reasons can arrive in minor
/// releases — matches need a `_` wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
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

/// Metadata passed to [`ContextCompactor::compact`](super::ContextCompactor::compact)
/// describing the compaction trigger and current state.
///
/// Compactors can use this information to decide how aggressively to
/// compact — e.g. an emergency compaction may use more aggressive
/// summarization than a routine threshold check.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases — construct through [`new`](Self::new); struct
/// literals compile only inside the crate.
#[derive(Clone)]
#[non_exhaustive]
pub struct CompactionContext {
    /// Estimated token count before compaction.
    ///
    /// The compactor's input size, computed by the configured
    /// [`TokenCounter`]. Compaction aims to bring the post-compaction size
    /// below this so the conversation fits with headroom for the next turn.
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

    /// The token counter for estimating message sizes.
    ///
    /// The same counter the driver uses for its compaction trigger — so the
    /// compactor can self-report `tokens_after` consistently. Compact
    /// implementations should use `context.counter.count(&messages)` instead
    /// of the static [`CompactionOutcome::estimate_tokens`] to match the
    /// driver's configured counter.
    pub counter: Arc<dyn TokenCounter>,

    /// Merged pre-compact hook instructions, when a hook supplied any.
    ///
    /// Populated by the driver from the pre-compact hooks' merged
    /// [`CompactResult`](crate::hooks::context::CompactResult): the
    /// last hook's `new_instructions` when hooks ran and supplied one.
    /// `None` when no hooks are configured or none supplied
    /// instructions — compactors that don't care simply ignore it.
    pub instructions: Option<String>,

    /// Merged pre-compact hook context fragments, when hooks supplied
    /// any.
    ///
    /// Every hook's `additional_context` entry, in hook order.
    /// Compact-but-informative fragments an LLM summarizer should weave
    /// into its prompt; empty when no hooks are configured or none
    /// supplied any.
    pub additional_context: Vec<String>,
}

impl CompactionContext {
    /// Create a compaction context with no hook contributions.
    ///
    /// The construction path for code outside the crate — the type is
    /// `#[non_exhaustive]`, so struct literals compile only inside the
    /// crate. The driver populates the hook fields (`instructions`,
    /// `additional_context`) when hooks supply them; this constructor
    /// covers the hook-less shape, which is what a compactor's unit
    /// fixtures need.
    #[must_use]
    pub fn new(
        tokens_before: u64,
        reason: CompactReason,
        context_window: u64,
        turn: usize,
        counter: Arc<dyn TokenCounter>,
    ) -> Self {
        Self {
            tokens_before,
            reason,
            context_window,
            turn,
            counter,
            instructions: None,
            additional_context: Vec::new(),
        }
    }
}

/// Result of a single compaction pass.
///
/// Returned by [`ContextCompactor::compact`](super::ContextCompactor::compact),
/// this struct contains the compacted message list along with telemetry data
/// about what happened.
/// `#[non_exhaustive]` so fields can be added in minor
/// releases — compactors construct it through
/// [`compacted`](Self::compacted), [`no_change`](Self::no_change),
/// or [`failed`](Self::failed).
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// Everything this pass removed from the history.
    ///
    /// Every input message with no surviving representative in
    /// [`messages`](Self::messages), in conversation order — message-granular,
    /// so a kept message with parts stripped during reconstruction is not
    /// listed, while a message the pass dropped wholesale is. Empty on
    /// genuinely-unchanged passes and on failures: nothing left the feed, so
    /// nothing was evicted. The engine hands this slice to the configured
    /// [`DemotionSink`](crate::compact::demote::DemotionSink) before adopting
    /// the compacted history; hosts running compaction directly through
    /// [`compact_manual`](super::ContextManager::compact_manual) receive it
    /// here and own its demotion themselves.
    ///
    /// The field is the compactor's claim, not a verified measurement: the
    /// manager classifies the pass by message count and measured tokens alone
    /// and never diffs the field against the output. A compactor that lists a
    /// message still present in [`messages`](Self::messages) double-presents
    /// it — delivered to the sink once and retained in the history — so the
    /// no-surviving-representative rule above is an obligation on compactors,
    /// not something the machinery polices.
    pub evicted: Vec<Message>,
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
            evicted: Vec::new(),
        }
    }

    /// Create an outcome representing successful compaction.
    ///
    /// Computes [`tokens_saved`](Self::tokens_saved) automatically from the
    /// difference between `tokens_before` and `tokens_after`. The outcome
    /// starts with no [`evicted`](Self::evicted) content; a compactor that
    /// removed messages attaches them with
    /// [`with_evicted`](Self::with_evicted).
    #[must_use]
    pub fn compacted(messages: Vec<Message>, tokens_before: u64, tokens_after: u64) -> Self {
        Self {
            tokens_saved: tokens_before.saturating_sub(tokens_after),
            messages,
            tokens_after,
            success: true,
            error: None,
            evicted: Vec::new(),
        }
    }

    /// Create an outcome representing a failed compaction pass.
    ///
    /// For compactors that could not produce a usable result — the
    /// carried error explains why, and the message list passes through
    /// unchanged so the caller can compare real measurements. Token
    /// savings are zero: nothing was committed, so nothing was saved.
    /// Nothing was evicted either — the feed keeps every message.
    #[must_use]
    pub fn failed(messages: Vec<Message>, tokens_after: u64, error: impl Into<String>) -> Self {
        Self {
            tokens_saved: 0,
            messages,
            tokens_after,
            success: false,
            error: Some(error.into()),
            evicted: Vec::new(),
        }
    }

    /// Attach the messages this pass removed, consuming `self`.
    ///
    /// The producer-side path for [`evicted`](Self::evicted): compactors
    /// chain it over [`compacted`](Self::compacted) with every input
    /// message that has no surviving representative in the output, in
    /// conversation order. Leave it unset when the pass removed nothing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::compact::CompactionOutcome;
    /// use loopctl::message::Message;
    ///
    /// let kept = vec![Message::user("recent")];
    /// let dropped = vec![Message::assistant("old")];
    /// let outcome = CompactionOutcome::compacted(kept, 100, 20)
    ///     .with_evicted(dropped);
    /// assert_eq!(outcome.evicted.len(), 1);
    /// ```
    #[must_use]
    pub fn with_evicted(mut self, evicted: Vec<Message>) -> Self {
        self.evicted = evicted;
        self
    }

    /// Estimate the token count for a slice of messages.
    ///
    /// Convenience static method for compactor implementations that need to
    /// self-report token counts. Uses the default
    /// [`HeuristicTokenCounter`](super::HeuristicTokenCounter). The
    /// [`ContextManager`](super::ContextManager) re-counts the result with
    /// its own configured counter after compaction, so the self-reported
    /// value is a hint — only the manager's count is authoritative for the
    /// before/after comparison.
    #[must_use]
    pub fn estimate_tokens(messages: &[Message]) -> u64 {
        use super::TokenCounter;
        super::HeuristicTokenCounter.count(messages)
    }
}

/// Telemetry data for a single compaction operation.
///
/// Produced by [`ContextManager::build_telemetry`](super::ContextManager::build_telemetry),
/// which the engine calls on every compacting pass (and hosts may call
/// directly when they drive compaction themselves). Observers receive
/// this via
/// [`on_compaction`](crate::observer::LoopObserver::on_compaction).
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is produced by the compaction machinery —
/// external code reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// Wall-clock duration of the compaction pass.
    ///
    /// The span the engine measures for the whole pass, opened just
    /// before the history snapshot is taken and closed after the
    /// demotion handoff completes: the snapshot clone, the token
    /// estimates around the compactor call (the manager re-counts the
    /// history before and the result after), the compactor's `compact`
    /// call itself, and the demotion sink's delivery. Not the compactor
    /// call in isolation — a slow sink or a large history is part of
    /// what this number reports, matching the duration the post-compact
    /// hook receives from the same start.
    pub duration: std::time::Duration,

    /// Compression ratio achieved by the pass.
    ///
    /// The pre-compaction token estimate divided by the post-compaction
    /// estimate: `4.0` means the pass shrank the conversation to a quarter.
    /// Values below `1.0` are honest — a compactor whose summary outweighed
    /// what it removed grew the conversation, and the ratio says so.
    /// `f64::INFINITY` when the result is empty (`post == 0`), so dashboards
    /// can rank compactors without dividing in the consumer.
    pub compression_ratio: f64,

    /// Fraction of the context window now free, post-compaction.
    ///
    /// `1.0 − post_tokens / context_window`, clamped to `0.0..=1.0` — the
    /// headroom signal: roughly how much conversation can accrue before the
    /// next pass fires. Computed against the manager's own window, so two
    /// managers with different windows report different headroom for the
    /// same compacted output. A window explicitly set to `0` (accepted by
    /// [`with_context_window`](super::ContextManager::with_context_window))
    /// reports `0.0` — a sentinel for "no window to measure against", not
    /// a genuinely full window.
    pub headroom_pct: f64,

    /// Which compactor produced this outcome, if identifiable.
    ///
    /// The host's name for the configured compactor (for example
    /// `"QaSummarizer"` or `"FallbackCompactor(QaSummarizer)"`), driving
    /// per-strategy dashboards. `None` when the caller does not supply a
    /// name — the engine itself always passes `None`, because the concrete
    /// type behind its `Arc<dyn ContextCompactor>` is unrecoverable without
    /// `Any` bounds the trait does not carry.
    pub compactor_name: Option<String>,
}

/// Conversation statistics captured before compaction.
///
/// A breakdown of the conversation's shape at the moment compaction begins:
/// how many messages there are, their estimated token cost, and how they split
/// across user/assistant/tool roles. Captured by
/// [`ContextManager::build_telemetry`](super::ContextManager::build_telemetry)
/// and bundled into [`CompactTelemetry::pre_compact`].
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is produced by the compaction machinery —
/// external code reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreCompactStats {
    /// Total number of messages in the conversation.
    ///
    /// Every message in the history about to be compacted, regardless of role
    /// or content. This is the input size the compactor operates on.
    pub total_messages: usize,

    /// Estimated token count.
    ///
    /// The pre-compaction token estimate of the whole conversation, using the
    /// standard 4-chars-per-token heuristic — history only, without
    /// per-request overhead or transients.
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

    /// Estimated tokens in user-role messages.
    ///
    /// The same 4-chars-per-token estimate as
    /// [`estimated_tokens`](Self::estimated_tokens), scoped to the slice of
    /// messages whose role is [`User`](crate::message::Role::User) — including
    /// tool-result messages, which conventionally ride the user role.
    /// `user_tokens + assistant_tokens` equals `estimated_tokens` on any
    /// conversation built from the two roles.
    pub user_tokens: u64,

    /// Estimated tokens in assistant-role messages.
    ///
    /// `estimated_tokens − user_tokens` — the partition's other half, so
    /// the two role figures sum to the whole-slice estimate exactly. The
    /// subtraction carries the whole-slice flooring remainder into this
    /// figure (up to one token on two-role lists) and, on lists with
    /// more than the two roles, their tokens as well.
    pub assistant_tokens: u64,

    /// Average tokens per message.
    ///
    /// `estimated_tokens / total_messages` — how dense the conversation is.
    /// High density with high message count is the summarize-first signal;
    /// low density means many small turns a truncator can drop wholesale.
    /// `0.0` for an empty conversation.
    pub density: f64,
}

/// Conversation statistics captured after compaction.
///
/// A summary of the conversation's shape after compaction completes, together
/// with how much the pass reclaimed. Captured by
/// [`ContextManager::build_telemetry`](super::ContextManager::build_telemetry)
/// and bundled into [`CompactTelemetry::post_compact`].
/// `#[non_exhaustive]` so fields can be added in minor
/// releases; it is produced by the compaction machinery —
/// external code reads it, never builds it.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// Number of user-role messages after compaction.
    ///
    /// The role breakdown of the compacted list, mirroring
    /// [`PreCompactStats::user_messages`](super::PreCompactStats::user_messages)
    /// so a before/after diff shows which roles the compactor favored.
    pub user_messages: usize,

    /// Number of assistant-role messages after compaction.
    ///
    /// The assistant share of the compacted list — the role a summarizer
    /// typically collapses hardest, since consecutive assistant turns merge
    /// into one summary message.
    pub assistant_messages: usize,

    /// Number of tool-bearing messages after compaction.
    ///
    /// Messages with at least one tool-call or tool-result part in the
    /// compacted list. Dropping these orphans the loop state, so a healthy
    /// compactor preserves them — this count is the quick check that it did.
    pub tool_messages: usize,

    /// Estimated tokens in user-role messages after compaction.
    ///
    /// The role-scoped estimate over the compacted list, mirroring
    /// [`PreCompactStats::user_tokens`](super::PreCompactStats::user_tokens).
    pub user_tokens: u64,

    /// The assistant half of the compacted estimate, by partition.
    ///
    /// `estimated_tokens − user_tokens` over the compacted list — the
    /// same partition semantics as
    /// [`PreCompactStats::assistant_tokens`](super::PreCompactStats::assistant_tokens):
    /// it carries the flooring remainder and any third-role tokens, so
    /// the two role figures sum to the whole exactly.
    pub assistant_tokens: u64,

    /// Average tokens per message after compaction.
    ///
    /// The compacted list's
    /// [`estimated_tokens`](Self::estimated_tokens) divided by its
    /// [`total_messages`](Self::total_messages) — compare against the
    /// pre-compaction density to see whether the pass thinned small turns
    /// or condensed large ones. `0.0` for an empty result.
    pub density: f64,
}

/// Error returned when compaction cannot bring the request payload
/// within the context window.
///
/// Terminal condition — either the payload is too large for the
/// compactor to reduce sufficiently, or the compactor itself failed
/// (see [`compactor_error`](Self::compactor_error)).
#[derive(Debug, Clone)]
pub struct ContextOverflow {
    /// Estimated token count of the conversation.
    ///
    /// How many tokens the conversation occupies when it overflows — the same
    /// heuristic estimate used everywhere else in the subsystem. When a
    /// reserve rode the request (see
    /// [`compact_with_reason`](super::ContextManager::compact_with_reason)),
    /// the estimate includes it, so the comparison below explains the
    /// failure. Compare
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
    /// How many tokens the conversation exceeds the context window by.
    ///
    /// Returns zero when the conversation fits within the window. Uses
    /// saturating subtraction so an underflow never panics.
    #[must_use]
    pub fn overflow(&self) -> u64 {
        self.tokens_used.saturating_sub(self.context_window)
    }

    /// The fraction of the context window currently consumed.
    ///
    /// Returns a value between `0.0` and `1.0` when the conversation fits,
    /// or above `1.0` when it overflows. Returns infinity when the context
    /// window is zero (division by zero).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.context_window == 0 {
            return f64::INFINITY;
        }
        crate::numeric::unit_ratio(self.tokens_used, self.context_window)
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

/// Result of [`ContextManager::ensure_context_fits`](super::ContextManager::ensure_context_fits).
///
/// Tells the caller whether compaction occurred and provides the
/// (possibly compacted) message list.
/// `#[non_exhaustive]` so new variants can arrive in minor
/// releases — matches need a `_` wildcard arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    ///
    /// Returns the compacted messages from [`Compacted`](Self::Compacted) or
    /// the unchanged messages from [`NoAction`](Self::NoAction). Use this when
    /// you only care about the resulting history and not whether compaction
    /// actually occurred.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        match self {
            Self::Compacted(outcome) => outcome.messages,
            Self::NoAction(messages) => messages,
        }
    }
}
