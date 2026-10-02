//! LLM-driven three-step compaction: summarize, question gaps, answer.
//!
//! [`QaSummarizer`] implements [`ContextCompactor`] against a
//! caller-supplied [`ApiClient`](crate::api::ApiClient), replacing
//! information-destructive
//! truncation with a compaction pass that preserves the facts a future
//! turn will need. Each pass makes up to three non-streaming LLM calls:
//!
//! 1. **Summarize** the about-to-be-dropped portion into a dense running
//!    summary.
//! 2. **Question gaps** — ask the model what it would need to know from
//!    the dropped context that the summary does not carry. Surfacing the
//!    *questions* instead of guessing means nothing critical is silently
//!    lost.
//! 3. **Answer** — pull the answers to those questions out of the full
//!    pre-compaction context before it is discarded, and fold them into
//!    the summary.
//!
//! The final summary becomes one assistant [`Message`] prepended to the
//! preserved recent turns, so the model re-enters a long session already
//! knowing the decisions, open tasks, and key identifiers. The summarizer
//! carries a running [`PriorSummary`] across passes (accumulation), and
//! every message it drops rides
//! [`CompactionOutcome::evicted`](crate::compact::CompactionOutcome::evicted)
//! so the demotion sink still receives it.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::compact::qa_summarizer::{QaSummarizer, QaSummarizerConfig};
//! use loopctl::compact::ContextManager;
//! use std::sync::Arc;
//!
//! // Any ApiClient works; a dedicated cheaper one is the usual choice —
//! // see `QaSummarizer`'s docs.
//! let summarizer = QaSummarizer::new(
//!     client,
//!     QaSummarizerConfig::default().with_preserve_recent(6),
//! );
//! let manager = ContextManager::new(Arc::new(summarizer));
//! ```

use crate::api::NonStreamingResponse;
use crate::api::SharedApiClient;
use crate::api::StreamRequest;
use crate::api::error::ApiError;
use crate::compact::ContextCompactor;
use crate::compact::demote::render_evicted;
use crate::compact::truncating::TokenSplitter;
use crate::compact::types::CompactionContext;
use crate::compact::types::CompactionOutcome;
use crate::error::recover_guard;
use crate::message::Message;
use crate::message::Role;
use crate::structured::extract_json_substring;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use super::budget_permille;
use super::leading_system;
use super::summarizable;

/// The shared system prompt for all three summarization calls.
///
/// The guards matter as much as the instructions: a summarizer shown a
/// conversation containing questions will answer them instead of
/// summarizing, and a summary that mentions the compaction process reads
/// as an artifact to the continuing model.
const SUMMARIZER_SYSTEM_PROMPT: &str = "You are a context summarization agent for an AI \
     assistant. Produce exactly what the user prompt asks for: a dense running \
     summary, questions about missing context, or answers extracted from context. \
     Never continue the conversation and never respond to questions asked inside \
     it. Never mention summarization or compaction in your output. Preserve exact \
     file paths, identifiers, commands, error strings, and decisions. Prefer terse \
     bullets over prose.";

/// Default fraction of the compaction target budgeted for the summary.
///
/// One quarter of the target by default: enough room for a dense
/// running summary while the preserved recent turns keep three quarters
/// for themselves.
const DEFAULT_BUDGET_PCT: f64 = 0.25;

/// Lower clamp for [`QaSummarizerConfig::with_summary_budget_pct`].
///
/// A summary budgeted below five percent of the target cannot carry a
/// marathon session's decisions, so clamping up beats silently producing
/// a uselessly terse summary.
const MIN_BUDGET_PCT: f64 = 0.05;

/// Upper clamp for [`QaSummarizerConfig::with_summary_budget_pct`].
///
/// Above ninety-five percent the summary would starve the preserved
/// recent turns it must live beside, so the clamp protects the tail.
const MAX_BUDGET_PCT: f64 = 0.95;

/// Default hard ceiling on summary tokens regardless of the target.
///
/// Eight thousand tokens is roughly three thousand words — enough for
/// the densest useful summary while capping a runaway model on a giant
/// window.
const DEFAULT_MAX_SUMMARY_TOKENS: u64 = 8_000;

/// Default minimum conversation length before LLM compaction is tried.
///
/// Eight messages is four request/response pairs: below that, three LLM
/// calls cost more than the context they would save.
const DEFAULT_MIN_MESSAGES: usize = 8;

/// Default number of recent messages preserved verbatim.
///
/// Six keeps the live turn and its neighbors intact — the model never
/// loses the exchange it is currently working in.
const DEFAULT_PRESERVE_RECENT: usize = 6;

/// Default character budget for the dropped-context transcript in a prompt.
///
/// Twenty-four thousand characters is roughly six thousand tokens of
/// prompt: a full marathon drop rendered without overwhelming a small
/// local model's input window.
const DEFAULT_TRANSCRIPT_MAX_CHARS: usize = 24_000;

/// Character budget for the preserved-tail excerpt in the question prompt.
///
/// The questioner only needs the shape of what follows the summary, not
/// its full text, so the excerpt stays tighter than the main transcript.
const RECENT_EXCERPT_MAX_CHARS: usize = 4_000;

/// How many gap questions step 3 will answer, at most.
///
/// Bounds the answer prompt on a model that emits a long question list;
/// the first questions are the ones the questioner judged most important.
const MAX_QUESTIONS: usize = 8;

/// Characters per token, matching the crate's heuristic counters.
///
/// The same long-standing cross-tokenizer average the heuristic token
/// counter divides by, kept in step so prompt-size estimates and budget
/// figures speak one unit.
const CHARS_PER_TOKEN: u64 = 4;

/// One round of an existing running summary, if compaction has happened before.
///
/// `QaSummarizer` carries the previous pass's summary forward so a session
/// that compacts many times accumulates one growing summary rather than
/// restarting from scratch each pass. `None` until the first successful
/// compaction; read it with
/// [`prior_summary`](QaSummarizer::prior_summary).
#[derive(Debug, Clone)]
pub struct PriorSummary {
    /// The text of the summary produced by the previous compaction pass.
    ///
    /// Exactly the text of the summary message that pass prepended to the
    /// history, which is how the next pass recognizes and excludes that
    /// message from its transcript.
    pub text: String,

    /// Which compaction pass produced it, 1-indexed.
    ///
    /// Increments once per successful pass; purely informational — the
    /// summary message itself carries no pass header, so the continuing
    /// model never reads compaction metadata.
    pub pass: u32,
}

/// The summary half-way through a compaction pass.
///
/// Produced by the summarize step and refined by the answer step; the
/// distinct type lets tests inspect each stage and lets the answer step
/// reference the text it is enriching.
#[derive(Debug, Clone)]
pub struct CompactionSummary {
    /// The summary text as produced so far.
    ///
    /// Step 1's output verbatim, or that output plus the folded answers
    /// once step 3 ran.
    pub text: String,

    /// Estimated tokens in [`text`](Self::text).
    ///
    /// The crate's heuristic estimate — the same figure every compactor
    /// self-reports with — used for the fold cap and telemetry.
    pub estimated_tokens: u64,
}

/// Configuration for [`QaSummarizer`].
///
/// Built from [`Default`](Self::default) and adjusted with the `with_*`
/// builders; every setter clamps or rejects degenerate values the way the
/// rest of the compaction module does, so a stored config is always
/// usable.
///
/// # Example
///
/// ```
/// use loopctl::compact::qa_summarizer::QaSummarizerConfig;
///
/// let config = QaSummarizerConfig::default()
///     .with_summary_budget_pct(0.3)
///     .with_max_summary_tokens(4_000)
///     .with_preserve_recent(8);
/// assert_eq!(config.summary_budget_pct(), 0.3);
/// assert_eq!(config.max_summary_tokens(), 4_000);
/// assert_eq!(config.preserve_recent(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct QaSummarizerConfig {
    /// Fraction of the compaction target budgeted for the summary.
    ///
    /// The summary's share of what the pass must fit into; the remainder
    /// stays available to the preserved recent turns that ride beside it.
    budget_pct: f64,

    /// The budget fraction as an exact per-mille integer.
    ///
    /// Derived from `budget_pct`'s decimal rendering through the module's
    /// shared rational parser, so the budget computation never runs
    /// through a float cast.
    budget_permille: u64,

    /// Hard ceiling on summary tokens regardless of the target.
    ///
    /// Bounds a model that ignores the budget instruction on a giant
    /// window; the budget formula takes the smaller of this ceiling and
    /// the fraction's share.
    max_summary_tokens: u64,

    /// Minimum conversation length before LLM compaction is tried.
    ///
    /// Passes below this length return `no_change` without spending any
    /// calls — the guard that keeps short chats free.
    min_messages: usize,

    /// Number of recent messages preserved verbatim.
    ///
    /// Handed to the internal [`TokenSplitter`], which draws a pair-safe
    /// boundary the way [`TruncatingCompactor`](crate::compact::TruncatingCompactor)
    /// does; the two boundaries are deliberately not identical — see
    /// [`with_preserve_recent`](Self::with_preserve_recent).
    preserve_recent: usize,

    /// Whether the step-1 prompt receives and merges the prior summary.
    ///
    /// Accumulation is what makes a session that compacts many times
    /// coherent; fresh mode trades that thread for a cheaper prompt.
    accumulate: bool,

    /// Character budget for the dropped-context transcript in a prompt.
    ///
    /// Bounds the summarizer's *input* independently of the output
    /// budget — the small-model knob the transcript renderer honors.
    transcript_max_chars: usize,
}

impl QaSummarizerConfig {
    /// The validated configuration with every field at its default.
    ///
    /// The single construction site for defaults; `Default` delegates
    /// here and every builder starts from a `fresh` value.
    fn fresh() -> Self {
        Self {
            budget_pct: DEFAULT_BUDGET_PCT,
            budget_permille: budget_permille(DEFAULT_BUDGET_PCT),
            max_summary_tokens: DEFAULT_MAX_SUMMARY_TOKENS,
            min_messages: DEFAULT_MIN_MESSAGES,
            preserve_recent: DEFAULT_PRESERVE_RECENT,
            accumulate: true,
            transcript_max_chars: DEFAULT_TRANSCRIPT_MAX_CHARS,
        }
    }

    /// Clamp a budget fraction and derive its exact per-mille form.
    ///
    /// A non-finite or non-positive fraction is rejected with a warning
    /// and the default takes its place — a stored NaN would poison the
    /// budget — then the value is clamped into `[0.05, 0.95]` so a
    /// summary can neither vanish nor starve the preserved tail.
    fn validated_pct(pct: f64) -> (f64, u64) {
        let usable = pct.is_finite() && pct > 0.0;
        if !usable {
            tracing::warn!(
                target: "loopctl::compact",
                pct,
                "summary budget fraction not finite or not positive; using the 0.25 default"
            );
        }
        let clamped = if usable {
            pct.clamp(MIN_BUDGET_PCT, MAX_BUDGET_PCT)
        } else {
            DEFAULT_BUDGET_PCT
        };
        (clamped, budget_permille(clamped))
    }

    /// Set the fraction of the compaction target budgeted for the summary.
    ///
    /// Clamped to `[0.05, 0.95]`; NaN and infinities are rejected with a
    /// warning and fall back to the default `0.25`. The remaining share
    /// of the target stays available to the preserved recent turns.
    #[must_use]
    pub fn with_summary_budget_pct(mut self, pct: f64) -> Self {
        (self.budget_pct, self.budget_permille) = Self::validated_pct(pct);
        self
    }

    /// Set the hard ceiling on summary tokens.
    ///
    /// Stops a pathological huge summary on a giant context window;
    /// values below `1` clamp to `1`.
    #[must_use]
    pub fn with_max_summary_tokens(mut self, tokens: u64) -> Self {
        self.max_summary_tokens = tokens.max(1);
        self
    }

    /// Set the minimum conversation length before LLM compaction is tried.
    ///
    /// Below this length `compact` returns a no-change outcome without
    /// spending any LLM calls; values below `2` clamp to `2`, matching
    /// [`TokenSplitter::with_min_messages`].
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Set the number of recent messages preserved verbatim.
    ///
    /// The value reaches the internal [`TokenSplitter`], which draws the
    /// drop/preserve boundary with the same occurrence-aware tool-pair
    /// safety [`TruncatingCompactor`](crate::compact::TruncatingCompactor)
    /// applies to its own split. The two boundaries are not identical:
    /// the splitter additionally snaps to an assistant-to-user turn
    /// transition the truncator never consults, and where the truncator
    /// keeps any first message verbatim this compactor pulls back only a
    /// leading system-role message — a non-system first message joins
    /// the summarized slice. Values below `1` clamp to `1`.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set whether passes accumulate on the prior summary.
    ///
    /// `true` (the default) feeds the previous pass's summary to step 1
    /// with merge instructions; `false` produces a fresh summary each
    /// pass — a cheaper prompt that loses the running thread.
    #[must_use]
    pub fn with_accumulate(mut self, accumulate: bool) -> Self {
        self.accumulate = accumulate;
        self
    }

    /// Set the character budget for the dropped-context transcript.
    ///
    /// The transcript is the rendered form of the messages about to be
    /// summarized; the budget bounds the summarizer's *input* prompt
    /// independently of the output budget, which matters on small local
    /// models. Values below `1` clamp to `1`.
    #[must_use]
    pub fn with_transcript_max_chars(mut self, max_chars: usize) -> Self {
        self.transcript_max_chars = max_chars.max(1);
        self
    }

    /// The fraction of the compaction target budgeted for the summary.
    ///
    /// The stored value after clamping — `0.25` unless overridden.
    #[must_use]
    pub fn summary_budget_pct(&self) -> f64 {
        self.budget_pct
    }

    /// The hard ceiling on summary tokens.
    ///
    /// The stored value after the `max(1)` clamp — a zero ceiling would
    /// make every pass fail its own budget.
    #[must_use]
    pub fn max_summary_tokens(&self) -> u64 {
        self.max_summary_tokens
    }

    /// The minimum conversation length before LLM compaction is tried.
    ///
    /// The stored value after the `max(2)` clamp, matching the
    /// splitter's own floor.
    #[must_use]
    pub fn min_messages(&self) -> usize {
        self.min_messages
    }

    /// The number of recent messages preserved verbatim.
    ///
    /// The stored value after the `max(1)` clamp; zero would preserve
    /// nothing and summarize the live turn out from under the model.
    #[must_use]
    pub fn preserve_recent(&self) -> usize {
        self.preserve_recent
    }

    /// Whether passes accumulate on the prior summary.
    ///
    /// Read this before deciding whether a stored prior reaches the next
    /// prompt — the flag gates both the merge block and the transcript
    /// hiding.
    #[must_use]
    pub fn accumulate(&self) -> bool {
        self.accumulate
    }

    /// The character budget for the dropped-context transcript.
    ///
    /// The stored value after the `max(1)` clamp, applied per prompt the
    /// transcript is built for.
    #[must_use]
    pub fn transcript_max_chars(&self) -> usize {
        self.transcript_max_chars
    }

    /// The summary token budget for a compaction pass at `target_tokens`.
    ///
    /// Reserve-then-cap: the smaller of the hard ceiling and the
    /// fraction's exact share of the target, so the summary can neither
    /// starve the preserved recent turns nor balloon on a giant window.
    /// The ceiling is enforced, not merely instructed: a step-1 response
    /// whose heuristic token estimate exceeds this figure fails the pass
    /// (strictly — a summary estimating exactly at the budget passes),
    /// with the folded answer section bounded separately by the same
    /// budget in characters.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::compact::qa_summarizer::QaSummarizerConfig;
    ///
    /// let config = QaSummarizerConfig::default();
    /// assert_eq!(config.summary_budget(40_000), 8_000);
    /// assert_eq!(config.summary_budget(20_000), 5_000);
    /// ```
    #[must_use]
    pub fn summary_budget(&self, target_tokens: u64) -> u64 {
        let share = target_tokens.saturating_mul(self.budget_permille) / 1_000;
        share.min(self.max_summary_tokens)
    }
}

impl Default for QaSummarizerConfig {
    fn default() -> Self {
        Self::fresh()
    }
}

/// LLM-driven three-step compactor.
///
/// Construct with any [`ApiClient`](crate::api::ApiClient) and a [`QaSummarizerConfig`], then
/// install on a [`ContextManager`](crate::compact::ContextManager) as
/// `Arc<dyn ContextCompactor>`. The client may be the same one the loop
/// uses for the main conversation or a dedicated cheaper one —
/// summarization is lower-stakes than generation, and a second client
/// pointed at a smaller model is the recommended production shape; the
/// constructor accepts either because the calls ride plain
/// [`create_message`](crate::api::ApiClient::create_message) with no request
/// options.
///
/// The summarizer's own LLM calls are *outside* the loop's request path:
/// their tokens are never counted against the run's usage, and the pass
/// reports its spend through the `loopctl.compaction.summarizer.tokens`
/// metric events so the exclusion is auditable. Each pass costs up to
/// three calls — compaction is infrequent, which is what makes that
/// acceptable.
///
/// State: the running [`PriorSummary`] lives behind a plain mutex, is
/// read once at pass entry, and is written once at pass exit — no lock is
/// held across an LLM call, and a pass cancelled mid-flight (the trait's
/// drop-safe contract) leaves the prior at its last committed value.
/// Call [`reset`](Self::reset) at session start when reusing one
/// summarizer across sessions.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::compact::qa_summarizer::{QaSummarizer, QaSummarizerConfig};
/// use loopctl::compact::ContextManager;
/// use std::sync::Arc;
///
/// let summarizer = Arc::new(QaSummarizer::new(
///     client,
///     QaSummarizerConfig::default(),
/// ));
/// let manager = ContextManager::new(summarizer);
/// ```
pub struct QaSummarizer {
    /// The client the three summarization calls ride.
    ///
    /// Shared, not owned: the same `Arc` the loop holds, or a dedicated
    /// client pointed at a cheaper model — the calls are plain
    /// `create_message` either way.
    client: SharedApiClient,

    /// The clamped, always-usable configuration.
    ///
    /// Every setter normalizes on write, so a stored config can never
    /// hold a degenerate value the budget math would trip over.
    config: QaSummarizerConfig,

    /// The running summary across passes within a session.
    ///
    /// Written only at the end of a successful pass; recovered on poison
    /// because the record is a plain value nothing can desynchronize.
    prior: Mutex<Option<PriorSummary>>,
}

impl QaSummarizer {
    /// Create a QA summarizer bound to the given API client.
    ///
    /// The client is kept as-is for the instance's whole life; hosts
    /// wanting per-session state should build one summarizer per session
    /// or call [`reset`](Self::reset) at the boundary.
    #[must_use]
    pub fn new(client: SharedApiClient, config: QaSummarizerConfig) -> Self {
        Self {
            client,
            config,
            prior: Mutex::new(None),
        }
    }

    /// The current accumulated prior summary, if a pass has completed.
    ///
    /// `None` until the first successful `compact`; hosts that persist
    /// session state can serialize this value to checkpoint where a
    /// session's accumulation stands. The record reflects the last pass
    /// this compactor completed, not the last history the manager
    /// adopted: a pass the manager later classifies as no-action (same
    /// message count, no measured shrink) still commits its summary
    /// here, because a compactor cannot see the manager's
    /// classification — so on such degenerate configurations the next
    /// pass's `<prior-summary>` may describe a message that is still
    /// live in the conversation.
    #[must_use]
    pub fn prior_summary(&self) -> Option<PriorSummary> {
        recover_guard(self.prior.lock()).clone()
    }

    /// Clear the accumulated prior summary.
    ///
    /// Call at session start when one summarizer instance serves more
    /// than one session; the next pass then produces a first-pass summary
    /// with no merge block. The cleared value is whatever the last
    /// completed pass committed — including one whose output the manager
    /// classified as no-action, per [`prior_summary`](Self::prior_summary).
    pub fn reset(&self) {
        *recover_guard(self.prior.lock()) = None;
    }

    /// The splitter whose boundary this compactor delegates to.
    ///
    /// Rebuilt from the config each pass — a plain value, so the
    /// construction cost is nothing next to the LLM calls that follow.
    fn splitter(&self) -> TokenSplitter {
        TokenSplitter::new()
            .with_preserve_recent(self.config.preserve_recent)
            .with_min_messages(self.config.min_messages)
    }

    /// The dropped-context transcript for a summarizer prompt.
    ///
    /// Renders through the module's shared renderer (role lines, tool
    /// call/result one-liners, per-part truncation) at the configured
    /// character budget. In accumulate mode the previous pass's summary
    /// message — recognizable by exact text equality, since this
    /// compactor authored it — is filtered out: its content already
    /// reaches the prompt through the `<prior-summary>` block, and
    /// presenting it twice invites the model to re-summarize the summary.
    fn transcript(&self, dropped: &[Message], prior: Option<&PriorSummary>) -> String {
        let prior_text = prior
            .filter(|_| self.config.accumulate)
            .map_or(String::new(), |prior| prior.text.clone());
        let hiding_prior = !prior_text.is_empty();
        let rendered_source: Vec<Message> = if hiding_prior {
            dropped
                .iter()
                .filter(|msg| !(msg.role == Role::Assistant && msg.text_content() == prior_text))
                .cloned()
                .collect()
        } else {
            dropped.to_vec()
        };
        render_evicted(&rendered_source, self.config.transcript_max_chars)
    }

    /// Run the three steps over one split.
    ///
    /// The orchestrator of a pass: summarize, question, and — only when
    /// the question step found gaps — answer and fold. Returns the final
    /// summary, how many LLM calls ran, and how many gap questions the
    /// pass answered; the step name in the error position names the arm
    /// that failed.
    ///
    /// # Errors
    ///
    /// Propagates the failing step's name and [`ApiError`] — the first
    /// LLM call that errored ends the pass.
    async fn run_steps(
        &self,
        dropped: &[Message],
        preserved: &[Message],
        budget: u64,
        prior: Option<&PriorSummary>,
        context: &CompactionContext,
        spend: &mut TokenSpend,
    ) -> Result<(CompactionSummary, u8, usize), (&'static str, ApiError)> {
        let mergeable_prior = prior.filter(|_| self.config.accumulate);
        let summary = self
            .summarize(dropped, mergeable_prior, budget, context, spend)
            .await
            .map_err(|error| ("summarize", error))?;
        let questions = self
            .question_gaps(&summary, preserved, spend)
            .await
            .map_err(|error| ("question", error))?;
        if questions.is_empty() {
            return Ok((summary, 2, 0));
        }
        let enriched = self
            .answer_and_merge(summary, &questions, dropped, mergeable_prior, budget, spend)
            .await
            .map_err(|error| ("answer", error))?;
        Ok((enriched, 3, questions.len()))
    }

    /// Step 1: summarize the dropped messages into a running summary.
    ///
    /// In accumulate mode the prompt carries the prior summary with merge
    /// instructions — carry forward facts the new excerpt does not
    /// mention, the excerpt wins on conflict, anything not carried
    /// forward is lost. Hook contributions from the
    /// [`CompactionContext`] ride along as host instructions and context
    /// fragments.
    ///
    /// A response whose text is empty or whitespace-only fails the step:
    /// committing it would assemble an empty summary message, hand the
    /// dropped slice to the demotion sink with nothing replacing it in
    /// the feed, and store an empty prior that silently disables the
    /// next pass's prior-summary filtering — the information loss this
    /// compactor exists to prevent. A response whose heuristic token
    /// estimate exceeds the pass budget fails the step for the mirror
    /// reason: the budget is a ceiling, not a suggestion, and shipping an
    /// oversized summary would silently break the reserve-then-cap
    /// guarantee that the preserved tail keeps its share of the window.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] from the underlying
    /// `create_message` call, and fails with a typed error when the
    /// response carries no usable summary text or a summary whose
    /// estimate exceeds the pass budget.
    async fn summarize(
        &self,
        dropped: &[Message],
        prior: Option<&PriorSummary>,
        budget: u64,
        context: &CompactionContext,
        spend: &mut TokenSpend,
    ) -> Result<CompactionSummary, ApiError> {
        let transcript = self.transcript(dropped, prior);
        let prompt = Self::summarize_prompt(&transcript, prior, budget, context);
        let response = self.call_with_spend(prompt, spend).await?;
        let text = response.message.text_content();
        if text.trim().is_empty() {
            return Err(ApiError::api("the model returned an empty summary"));
        }
        let summary = Self::summary_of(text);
        if summary.estimated_tokens > budget {
            return Err(ApiError::api(format!(
                "the model returned a summary over the {budget}-token budget"
            )));
        }
        Ok(summary)
    }

    /// Step 2: ask what the summary is missing.
    ///
    /// The questioner sees the summary and a bounded excerpt of the
    /// preserved tail — the context the continuing model actually works
    /// from — and lists what it would need from the dropped portion,
    /// which it cannot see. Returns at most [`MAX_QUESTIONS`] questions;
    /// an empty list is the model judging the summary sufficient, and a
    /// parse failure degrades to that same empty list with a warning
    /// rather than failing the pass.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] from the underlying
    /// `create_message` call.
    async fn question_gaps(
        &self,
        summary: &CompactionSummary,
        preserved: &[Message],
        spend: &mut TokenSpend,
    ) -> Result<Vec<String>, ApiError> {
        let recent = render_evicted(preserved, RECENT_EXCERPT_MAX_CHARS);
        let prompt = Self::question_prompt(&summary.text, &recent);
        let response = self.call_with_spend(prompt, spend).await?;
        let raw = response.message.text_content();
        let questions = Self::parse_questions(&raw);
        let answered_sufficient = raw.trim().is_empty() || raw.trim() == "[]";
        if questions.is_empty() && !answered_sufficient {
            tracing::warn!(
                target: "loopctl::compact",
                "question step produced no parseable questions; treating the summary as sufficient"
            );
        }
        Ok(questions)
    }

    /// Step 3: answer the gap questions from the dropped context.
    ///
    /// One call for every question, numbered; answers the dropped context
    /// does not contain come back `UNKNOWN` and are dropped, so no
    /// hallucinated fact ever enters the summary. The transcript is the
    /// same accumulate-filtered render step 1 saw, so the prior summary
    /// never re-enters the answer prompt as answerable conversation. The
    /// folded section is capped to the pass budget, shedding the latest
    /// answers first.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] from the underlying
    /// `create_message` call.
    async fn answer_and_merge(
        &self,
        summary: CompactionSummary,
        questions: &[String],
        dropped: &[Message],
        prior: Option<&PriorSummary>,
        budget: u64,
        spend: &mut TokenSpend,
    ) -> Result<CompactionSummary, ApiError> {
        let transcript = self.transcript(dropped, prior);
        let prompt = Self::answer_prompt(questions, &transcript);
        let response = self.call_with_spend(prompt, spend).await?;
        let text = fold_answers(
            &summary.text,
            questions,
            &response.message.text_content(),
            budget,
        );
        Ok(Self::summary_of(text))
    }

    /// Wrap summary text with its heuristic token estimate.
    ///
    /// The estimate is the crate's standard heuristic, the same figure
    /// every compactor self-reports with.
    fn summary_of(text: String) -> CompactionSummary {
        let estimated_tokens = estimate_text_tokens(&text);
        CompactionSummary {
            text,
            estimated_tokens,
        }
    }

    /// Run one non-streaming call with the shared system prompt.
    ///
    /// The single call shape every step uses: one user message carrying
    /// the step prompt, no tools, no request options — a summarizer must
    /// work against every [`ApiClient`](crate::api::ApiClient), including ones that reject
    /// options they cannot forward. The call's tokens fold into the
    /// pass's spend: the provider's reported usage when it supplies one,
    /// the heuristic estimate otherwise.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] when the call fails.
    async fn call_with_spend(
        &self,
        user_prompt: String,
        spend: &mut TokenSpend,
    ) -> Result<NonStreamingResponse, ApiError> {
        let estimated_input = char_tokens(&user_prompt);
        let request = StreamRequest::new(vec![Message::user(user_prompt)])
            .with_system(Some(SUMMARIZER_SYSTEM_PROMPT.to_string()));
        let response = self.client.create_message(&request).await?;
        let output_text = response.message.text_content();
        match response.usage.as_ref() {
            Some(usage) => spend.record(
                u64::from(usage.input_tokens),
                u64::from(usage.output_tokens),
                true,
            ),
            None => spend.record(estimated_input, char_tokens(&output_text), false),
        }
        Ok(response)
    }

    /// The step-1 user prompt.
    ///
    /// Optionally opens with the prior-summary merge block, always
    /// carries the tagged transcript and token budget, and closes with
    /// any hook instructions and context fragments.
    fn summarize_prompt(
        transcript: &str,
        prior: Option<&PriorSummary>,
        budget: u64,
        context: &CompactionContext,
    ) -> String {
        use std::fmt::Write as _;
        let mut prompt = String::new();
        if let Some(prior) = prior {
            prompt.push_str(
                "The summary of everything before this excerpt is in <prior-summary>. Merge it \
                 with the excerpt into one new summary: carry forward objectives, decisions, and \
                 constraints the excerpt does not mention; where they conflict, the excerpt wins; \
                 drop only what is finished and no longer needed. The prior summary is discarded \
                 after this — anything you do not carry into your output is lost.\n\n\
                 <prior-summary>\n",
            );
            prompt.push_str(&prior.text);
            prompt.push_str("\n</prior-summary>\n\n");
        }
        let _ignored = write!(
            prompt,
            "Summarize the conversation in <conversation> so another agent can continue the \
             work without re-reading it. Capture the objective, decisions made and why, important \
             constraints, active and blocked work, and the key files, paths, and identifiers \
             involved. Keep it under roughly {budget} tokens. Emit only the summary.\n\n\
             <conversation>\n{transcript}\n</conversation>"
        );
        if let Some(instructions) = context.instructions.as_deref() {
            let _ignored = write!(
                prompt,
                "\n\nAdditional instructions from the host:\n{instructions}"
            );
        }
        for fragment in &context.additional_context {
            let _ignored = write!(prompt, "\n\nAdditional context to weave in:\n{fragment}");
        }
        prompt
    }

    /// The step-2 user prompt.
    ///
    /// Shows the questioner the summary and the preserved tail but never
    /// the dropped portion — it must ask for what it cannot see.
    fn question_prompt(summary: &str, recent: &str) -> String {
        format!(
            "An agent continues the work with only <summary> (of earlier context) and <recent> \
             (the messages that follow it). List what facts from the earlier context are missing \
             from <summary> that the agent would need to continue effectively. Emit a JSON array \
             of short question strings — [] if the summary is sufficient. No other text.\n\n\
             <summary>\n{summary}\n</summary>\n\n<recent>\n{recent}\n</recent>"
        )
    }

    /// The step-3 user prompt.
    ///
    /// Numbers the questions for the model to echo back, and pairs them
    /// with the full dropped transcript the answers must come from.
    fn answer_prompt(questions: &[String], transcript: &str) -> String {
        let numbered = questions
            .iter()
            .enumerate()
            .map(|(index, question)| format!("{}. {question}", index.saturating_add(1)))
            .collect::<Vec<String>>()
            .join("\n");
        format!(
            "Answer each numbered question using only the conversation in <conversation>. Emit \
             one line per question as `N. <answer>`; if the conversation does not contain the \
             answer, emit `N. UNKNOWN`. No other text.\n\n<questions>\n{numbered}\n</questions>\
             \n\n<conversation>\n{transcript}\n</conversation>"
        )
    }

    /// Parse the question step's answer into at most [`MAX_QUESTIONS`]
    /// questions.
    ///
    /// Prefers the JSON array (found leniently — fences and surrounding
    /// prose tolerated by the shared scanner); falls back to numbered or
    /// bulleted lines; any other shape yields no questions, which the
    /// caller treats as "the summary was sufficient".
    fn parse_questions(text: &str) -> Vec<String> {
        if let Some(value) = extract_json_substring(text.trim())
            && let serde_json::Value::Array(items) = &value
        {
            return items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .map(|question| question.trim().to_string())
                .filter(|question| !question.is_empty())
                .take(MAX_QUESTIONS)
                .collect();
        }
        list_lines(text)
    }

    /// The assembled output list for a successful pass.
    ///
    /// A leading system-role history message survives at the head, the
    /// summary follows as one assistant message, and the preserved tail
    /// rides verbatim.
    fn assemble(summary: &str, preserved: &[Message], messages: &[Message]) -> Vec<Message> {
        let mut out = Vec::with_capacity(preserved.len().saturating_add(2));
        if leading_system(messages)
            && let Some(first) = messages.first()
        {
            out.push(first.clone());
        }
        out.push(Message::assistant(summary.to_string()));
        out.extend_from_slice(preserved);
        out
    }

    /// The failure outcome: the original messages intact, the step named.
    ///
    /// Pure construction — the caller owns the decision to stop, and the
    /// messages return untouched so a fallback chain can try the next
    /// tier.
    fn fail(
        original: Vec<Message>,
        context: &CompactionContext,
        step: &str,
        error: &ApiError,
    ) -> CompactionOutcome {
        tracing::warn!(
            target: "loopctl::compact",
            step,
            error = %error,
            "qa summarizer pass failed; returning the original messages"
        );
        CompactionOutcome::failed(
            original,
            context.tokens_before,
            format!("qa summarizer {step} step failed: {error}"),
        )
    }

    /// Commit the pass's summary as the prior for the next pass.
    ///
    /// The one write to the running state, after the outcome is fully
    /// assembled; a pass that never reaches here leaves the prior at its
    /// last committed value.
    fn commit_prior(&self, text: String, previous_pass: Option<u32>) {
        let pass = previous_pass.map_or(1, |pass| pass.saturating_add(1));
        *recover_guard(self.prior.lock()) = Some(PriorSummary { text, pass });
    }

    /// Emit the pass's telemetry.
    ///
    /// One summary line for the pass shape and one metric event per token
    /// direction. Pass duration is deliberately absent: the engine
    /// already reports the whole pass through
    /// [`CompactTelemetry::duration`](crate::compact::CompactTelemetry::duration)
    /// measured from the clock seam, and a second wall-clock read inside
    /// the compactor would bypass that seam.
    fn emit_telemetry(steps: u8, questions: usize, budget: u64, spend: &TokenSpend) {
        tracing::debug!(
            target: "loopctl::compact",
            steps,
            questions,
            budget,
            "qa summarizer pass complete"
        );
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.compaction.summarizer.tokens",
            direction = "in",
            tokens = spend.input_tokens,
            estimated = spend.estimated,
            "summarizer input tokens for one compaction pass"
        );
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.compaction.summarizer.tokens",
            direction = "out",
            tokens = spend.output_tokens,
            estimated = spend.estimated,
            "summarizer output tokens for one compaction pass"
        );
    }
}

impl ContextCompactor for QaSummarizer {
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let original = messages.clone();
            if messages.len() <= self.config.min_messages {
                return CompactionOutcome::no_change(messages);
            }
            let split = self.splitter().split(&messages);
            let dropped = summarizable(&split.to_compact);
            if dropped.is_empty() {
                return CompactionOutcome::no_change(messages);
            }

            let budget = self.config.summary_budget(target_tokens);
            let prior = self.prior_summary();
            let mut spend = TokenSpend::default();
            let (enriched, steps, questions) = match self
                .run_steps(
                    dropped,
                    &split.preserved,
                    budget,
                    prior.as_ref(),
                    &context,
                    &mut spend,
                )
                .await
            {
                Ok(passed) => passed,
                Err((step, error)) => return Self::fail(original, &context, step, &error),
            };

            let out = Self::assemble(&enriched.text, &split.preserved, &messages);
            let tokens_after = context.counter.count(&out);
            self.commit_prior(enriched.text, prior.as_ref().map(|prior| prior.pass));
            Self::emit_telemetry(steps, questions, budget, &spend);
            CompactionOutcome::compacted(out, context.tokens_before, tokens_after)
                .with_evicted(dropped.to_vec())
        })
    }
}

/// The summarizer's own token spend for one pass.
///
/// Input and output totals across the pass's calls. Providers that report
/// [`Usage`](crate::stream::Usage) are taken at their word; the heuristic
/// fallback marks the figures estimated so a dashboard never mistakes a
/// characters-per-token guess for a billed count.
#[derive(Debug, Default)]
struct TokenSpend {
    /// Input tokens across the pass's calls.
    ///
    /// Reported usage when the provider supplies it, the characters-
    /// per-token estimate otherwise.
    input_tokens: u64,

    /// Output tokens across the pass's calls.
    ///
    /// The same reported-else-estimated discipline as the input total,
    /// summed across every call the pass made.
    output_tokens: u64,

    /// Whether any figure is a heuristic estimate rather than a report.
    ///
    /// Sticky once set: a mixed total should never read as fully billed
    /// on a dashboard.
    estimated: bool,
}

impl TokenSpend {
    /// Fold one call's figures into the running totals.
    ///
    /// The estimated flag is sticky once any call lacked usage, because
    /// the totals are then a mix no consumer should read as fully billed.
    fn record(&mut self, input_tokens: u64, output_tokens: u64, reported: bool) {
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        self.estimated = self.estimated || !reported;
    }
}

/// The heuristic token estimate for one summary-sized text.
///
/// Wraps the text in a single assistant message and runs the standard
/// estimate over it, so the figure lands in the same unit as every other
/// compactor self-report.
fn estimate_text_tokens(text: &str) -> u64 {
    CompactionOutcome::estimate_tokens(std::slice::from_ref(&Message::assistant(text.to_string())))
}

/// The heuristic token estimate for one prompt-sized text.
///
/// A plain characters-per-token division — prompt texts have no message
/// framing to count, so the shared counter would add overhead that is
/// not there.
fn char_tokens(text: &str) -> u64 {
    text.chars().count() as u64 / CHARS_PER_TOKEN
}

/// Parse numbered or bulleted list lines into question strings.
///
/// The fallback parser for models that answer the question step with a
/// plain list: a line beginning `- `, `* `, or `N.`/`N)` yields its
/// remainder, trimmed; everything else is skipped. Capped at
/// [`MAX_QUESTIONS`] like the JSON path.
fn list_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(parse_list_line)
        .take(MAX_QUESTIONS)
        .collect()
}

/// Parse one list line into its question text, if it is one.
///
/// Recognizes the two bullet spellings and delegates numbered lines to
/// the marker stripper; anything else is not a question.
fn parse_list_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let question = if let Some(rest) = trimmed.strip_prefix("- ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("* ") {
        rest
    } else {
        numbered_text(trimmed)?
    };
    let question = question.trim();
    (!question.is_empty()).then(|| question.to_string())
}

/// Strip a `N.` or `N)` prefix, requiring digits before the marker.
///
/// Requiring at least one digit keeps prose like `3.14 radians` from
/// parsing as question three.
fn numbered_text(line: &str) -> Option<&str> {
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let after_digits = line.get(digits..)?;
    after_digits
        .strip_prefix(". ")
        .or_else(|| after_digits.strip_prefix(") "))
}

/// Fold the answer step's output into the summary.
///
/// Numbered answers pair with questions by their number; `UNKNOWN`
/// answers are dropped — an unanswerable question stays unrecorded rather
/// than hallucinated — and each folded answer renders as one bullet. The
/// section is capped to the pass's character budget by shedding answers
/// from the end, so the summary itself always survives whole.
fn fold_answers(summary: &str, questions: &[String], answer_text: &str, budget: u64) -> String {
    let answers = numbered_answers(answer_text);
    let cap_chars = usize::try_from(budget.saturating_mul(CHARS_PER_TOKEN)).unwrap_or(usize::MAX);
    let mut section = String::from("\n\n## Facts recovered from the dropped context");
    let mut folded = 0usize;
    for (index, question) in questions.iter().enumerate() {
        let Some(answer) = answers.get(&index.saturating_add(1)) else {
            continue;
        };
        let bullet = format!("\n- {question}: {answer}");
        if summary
            .chars()
            .count()
            .saturating_add(section.chars().count())
            .saturating_add(bullet.chars().count())
            > cap_chars
        {
            break;
        }
        section.push_str(&bullet);
        folded = folded.saturating_add(1);
    }
    if folded == 0 {
        return summary.to_string();
    }
    format!("{summary}{section}")
}

/// Parse `N. answer` lines into a number-to-answer map.
///
/// `UNKNOWN` answers (any case) map to nothing — the caller's loop skips
/// the missing numbers, which is how they are dropped.
fn numbered_answers(text: &str) -> std::collections::HashMap<usize, String> {
    let mut answers = std::collections::HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 {
            continue;
        }
        let Some(number) = trimmed
            .get(..digits)
            .and_then(|digits_text| digits_text.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(rest) = trimmed.get(digits..) else {
            continue;
        };
        let Some(answer) = rest.strip_prefix(". ") else {
            continue;
        };
        if answer.trim().eq_ignore_ascii_case("unknown") {
            continue;
        }
        answers.insert(number, answer.trim().to_string());
    }
    answers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ApiClient;
    use crate::api::error::ApiError;
    use crate::compact::TokenCounter;
    use crate::compact::types::CompactReason;
    use crate::compact::{ContextManager, EnsureContextResult, HeuristicTokenCounter};
    use crate::stream::{StreamEvent, StreamStopReason};
    use futures::Stream;
    use std::collections::VecDeque;
    use std::sync::Arc;

    fn ok(text: &str) -> Result<NonStreamingResponse, ApiError> {
        Ok(NonStreamingResponse {
            message: Message::assistant(text),
            stop_reason: StreamStopReason::EndTurn,
            usage: None,
        })
    }

    /// An `ApiClient` double that records every call's prompt and serves
    /// the scripted responses in order.
    struct RecordingClient {
        prompts: Mutex<Vec<String>>,
        script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
    }

    impl RecordingClient {
        fn new(script: Vec<Result<NonStreamingResponse, ApiError>>) -> Arc<Self> {
            Arc::new(Self {
                prompts: Mutex::new(Vec::new()),
                script: Mutex::new(script.into()),
            })
        }

        fn prompts(&self) -> Vec<String> {
            recover_guard(self.prompts.lock()).clone()
        }
    }

    impl ApiClient for RecordingClient {
        fn model(&self) -> String {
            "qa-test".to_string()
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            let prompt = request
                .messages
                .first()
                .map_or(String::new(), Message::text_content);
            recover_guard(self.prompts.lock()).push(prompt);
            let next = recover_guard(self.script.lock())
                .pop_front()
                .expect("every scripted call gets a response");
            Box::pin(std::future::ready(next))
        }
    }

    fn scripted(
        script: Vec<Result<NonStreamingResponse, ApiError>>,
    ) -> (Arc<RecordingClient>, QaSummarizer) {
        let client = RecordingClient::new(script);
        let summarizer = QaSummarizer::new(
            Arc::clone(&client) as SharedApiClient,
            QaSummarizerConfig::default(),
        );
        (client, summarizer)
    }

    /// Six user/assistant pairs with unique, probe-able text per message.
    fn conversation() -> Vec<Message> {
        let mut messages = Vec::new();
        for turn in 0..6 {
            messages.push(Message::user(format!(
                "user turn {turn} asks about topic-{turn}"
            )));
            messages.push(Message::assistant(format!(
                "assistant turn {turn} decided fact-{turn}"
            )));
        }
        messages
    }

    fn context_for(messages: &[Message]) -> CompactionContext {
        CompactionContext {
            tokens_before: CompactionOutcome::estimate_tokens(messages),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000_000,
            turn: 3,
            counter: Arc::new(HeuristicTokenCounter),
            instructions: None,
            additional_context: Vec::new(),
        }
    }

    #[tokio::test]
    async fn a_short_conversation_returns_no_change_with_zero_calls() {
        let (client, summarizer) = scripted(vec![]);
        let short: Vec<Message> = conversation().into_iter().take(8).collect();
        let outcome = summarizer
            .compact(short.clone(), 40_000, context_for(&short))
            .await;
        assert!(outcome.success, "a short conversation is a no-op success");
        assert_eq!(outcome.messages.len(), 8, "every message returns unchanged");
        assert_eq!(outcome.tokens_saved, 0, "nothing was saved");
        assert!(
            client.prompts().is_empty(),
            "no LLM call is spent below min_messages"
        );
    }

    #[tokio::test]
    async fn the_split_delegates_to_the_token_splitter() {
        let messages = conversation();
        let expected = TokenSplitter::new()
            .with_preserve_recent(6)
            .with_min_messages(8)
            .split(&messages);
        let (_client, summarizer) = scripted(vec![ok("S"), ok("[]")]);
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let preserved = &outcome.messages[1..];
        assert_eq!(
            preserved.len(),
            expected.preserved.len(),
            "the preserved tail is the splitter's, not a reinvented boundary"
        );
        for (got, want) in preserved.iter().zip(&expected.preserved) {
            assert_eq!(got.text_content(), want.text_content());
        }
    }

    #[test]
    fn the_budget_takes_the_smaller_of_the_cap_and_the_share() {
        let config = QaSummarizerConfig::default();
        assert_eq!(config.summary_budget(40_000), 8_000, "the cap wins at 40k");
        assert_eq!(
            config.summary_budget(20_000),
            5_000,
            "the share wins at 20k"
        );
        let capped = QaSummarizerConfig::default().with_max_summary_tokens(2_000);
        assert_eq!(
            capped.summary_budget(40_000),
            2_000,
            "a lowered cap wins at 40k"
        );
        let low = QaSummarizerConfig::default().with_summary_budget_pct(0.01);
        assert!(
            (low.summary_budget_pct() - 0.05).abs() < f64::EPSILON,
            "0.01 clamps up to 0.05"
        );
        assert_eq!(low.summary_budget(1_000), 50, "the clamped share applies");
        let high = QaSummarizerConfig::default().with_summary_budget_pct(2.0);
        assert!(
            (high.summary_budget_pct() - 0.95).abs() < f64::EPSILON,
            "2.0 clamps down to 0.95"
        );
        let rejected = QaSummarizerConfig::default()
            .with_summary_budget_pct(f64::NAN)
            .with_max_summary_tokens(100_000);
        assert!(
            (rejected.summary_budget_pct() - 0.25).abs() < f64::EPSILON,
            "NaN falls back to the default fraction"
        );
        assert_eq!(rejected.summary_budget(40_000), 10_000);
    }

    #[tokio::test]
    async fn the_three_steps_fire_in_order_and_feed_each_other() {
        let (client, summarizer) = scripted(vec![
            ok("SUMMARY-A"),
            ok("[\"Which config path was chosen?\"]"),
            ok("1. /etc/app.conf"),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 3, "a full pass is three LLM calls");
        assert!(
            prompts[0].contains("<conversation>")
                && prompts[0].contains("user turn 0 asks about topic-0"),
            "step 1 summarizes the dropped transcript"
        );
        assert!(
            prompts[1].contains("<summary>")
                && prompts[1].contains("SUMMARY-A")
                && prompts[1].contains("<recent>"),
            "step 2 sees the step-1 summary and the preserved tail"
        );
        assert!(
            prompts[2].contains("<questions>")
                && prompts[2].contains("Which config path was chosen?")
                && prompts[2].contains("<conversation>"),
            "step 3 answers the step-2 questions from the transcript"
        );
        let summary_text = outcome.messages.first().map(Message::text_content).unwrap();
        assert!(
            summary_text.contains("SUMMARY-A") && summary_text.contains("/etc/app.conf"),
            "the folded answer rides the summary message: {summary_text}"
        );
    }

    #[tokio::test]
    async fn no_questions_skips_the_answer_step() {
        let (client, summarizer) = scripted(vec![ok("SUMMARY-B"), ok("[]")]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        assert_eq!(
            client.prompts().len(),
            2,
            "a sufficient summary costs two calls, not three"
        );
        assert_eq!(
            outcome.messages.first().map(Message::text_content),
            Some("SUMMARY-B".to_string()),
            "the summary message is the step-1 output verbatim"
        );
    }

    #[tokio::test]
    async fn the_summary_message_is_assistant_role_and_precedes_the_preserved_tail() {
        let (_client, summarizer) = scripted(vec![ok("SUMMARY-C"), ok("[]")]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        let first = outcome.messages.first().expect("the output is non-empty");
        assert_eq!(
            first.role,
            Role::Assistant,
            "the summary is an assistant message"
        );
        assert_eq!(first.text_content(), "SUMMARY-C");
        assert_eq!(
            outcome.messages.len(),
            7,
            "one summary message plus six preserved"
        );
    }

    #[tokio::test]
    async fn a_leading_system_message_survives_at_the_head() {
        let (_client, summarizer) = scripted(vec![ok("SUMMARY-S"), ok("[]")]);
        let mut messages = conversation();
        messages.insert(
            0,
            Message::new(
                Role::System,
                vec![crate::message::MessagePart::text("standing instructions")],
            ),
        );
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        let first = outcome.messages.first().expect("the output is non-empty");
        assert_eq!(first.role, Role::System, "the system message survives");
        assert_eq!(first.text_content(), "standing instructions");
        assert_eq!(
            outcome.messages.get(1).map(|m| m.role),
            Some(Role::Assistant),
            "the summary follows the system message"
        );
        assert_eq!(
            outcome.evicted.len(),
            6,
            "the pulled system message is the one message not evicted"
        );
    }

    #[tokio::test]
    async fn a_leading_system_message_never_rides_the_summarizer_transcript() {
        let (client, summarizer) = scripted(vec![
            ok("SUMMARY-S"),
            ok("[\"what fact was fixed?\"]"),
            ok("1. the fixed fact"),
        ]);
        let mut messages = conversation();
        messages.insert(
            0,
            Message::new(
                Role::System,
                vec![crate::message::MessagePart::text("standing instructions")],
            ),
        );
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 3, "the scripted pass runs all three steps");
        assert!(
            prompts[0].contains("<conversation>")
                && prompts[0].contains("user turn 0 asks about topic-0"),
            "the step-1 transcript still renders the dropped conversation: {}",
            prompts[0]
        );
        assert!(
            !prompts[0].contains("standing instructions"),
            "the system message survives at the output head, so it must not ride the \
             step-1 transcript and get restated by the summary: {}",
            prompts[0]
        );
        assert!(
            prompts[2].contains("<conversation>") && !prompts[2].contains("standing instructions"),
            "the step-3 transcript draws from the same system-free slice: {}",
            prompts[2]
        );
        let head = outcome.messages.first().expect("the output is non-empty");
        assert_eq!(
            (head.role, head.text_content()),
            (Role::System, "standing instructions".to_string()),
            "the survival contract is unchanged — the message still heads the output"
        );
        assert_eq!(
            outcome.messages.get(1).map(|m| m.role),
            Some(Role::Assistant),
            "the summary follows the system message"
        );
        assert!(
            outcome
                .evicted
                .iter()
                .all(|msg| msg.text_content() != "standing instructions"),
            "the pulled-back system message is never demoted"
        );
    }

    #[test]
    fn the_summarizable_slice_excludes_only_a_leading_system_message() {
        let system =
            |text: &str| Message::new(Role::System, vec![crate::message::MessagePart::text(text)]);
        let slice = vec![
            system("standing instructions"),
            Message::user("question"),
            Message::assistant("answer"),
        ];
        let texts: Vec<String> = summarizable(&slice)
            .iter()
            .map(Message::text_content)
            .collect();
        assert_eq!(
            texts,
            vec!["question".to_string(), "answer".to_string()],
            "a leading system message is the one slice member never summarized"
        );
        let user_led = vec![Message::user("question"), Message::assistant("answer")];
        assert_eq!(
            summarizable(&user_led).len(),
            2,
            "a slice with no leading system message passes through whole"
        );
        let only_system = system("the only compactable message");
        assert!(
            summarizable(std::slice::from_ref(&only_system)).is_empty(),
            "a slice holding only the system message summarizes nothing — the \
             caller's no-change case"
        );
    }

    #[tokio::test]
    async fn a_second_pass_accumulates_on_the_first_summary() {
        let (client, summarizer) =
            scripted(vec![ok("SUMMARY-1"), ok("[]"), ok("SUMMARY-2"), ok("[]")]);
        let first = conversation();
        let outcome_one = summarizer
            .compact(first.clone(), 40_000, context_for(&first))
            .await;
        assert!(outcome_one.success);
        let prior = summarizer
            .prior_summary()
            .expect("pass one commits a prior");
        assert_eq!(prior.pass, 1);
        assert_eq!(prior.text, "SUMMARY-1");

        let mut second = outcome_one.messages.clone();
        second.push(Message::user("user turn 6 asks about topic-6"));
        second.push(Message::assistant("assistant turn 6 decided fact-6"));
        second.push(Message::user("user turn 7 asks about topic-7"));
        second.push(Message::assistant("assistant turn 7 decided fact-7"));
        let outcome_two = summarizer
            .compact(second, 40_000, context_for(&first))
            .await;
        assert!(outcome_two.success);
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 4, "two passes, two calls each");
        assert!(
            prompts[2].contains("<prior-summary>") && prompts[2].contains("SUMMARY-1"),
            "the second pass's step-1 prompt merges the first summary"
        );
        let prior = summarizer
            .prior_summary()
            .expect("pass two commits a prior");
        assert_eq!(prior.pass, 2, "the pass counter increments");
        assert_eq!(prior.text, "SUMMARY-2");
    }

    #[tokio::test]
    async fn accumulate_off_produces_a_fresh_prompt_each_pass() {
        let client =
            RecordingClient::new(vec![ok("SUMMARY-1"), ok("[]"), ok("SUMMARY-2"), ok("[]")]);
        let summarizer = QaSummarizer::new(
            Arc::clone(&client) as SharedApiClient,
            QaSummarizerConfig::default().with_accumulate(false),
        );
        let first = conversation();
        let outcome_one = summarizer
            .compact(first.clone(), 40_000, context_for(&first))
            .await;
        assert!(outcome_one.success);
        let mut second = outcome_one.messages.clone();
        second.push(Message::user("user turn 6 asks about topic-6"));
        second.push(Message::assistant("assistant turn 6 decided fact-6"));
        second.push(Message::user("user turn 7 asks about topic-7"));
        second.push(Message::assistant("assistant turn 7 decided fact-7"));
        let outcome_two = summarizer
            .compact(second, 40_000, context_for(&first))
            .await;
        assert!(outcome_two.success);
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 4);
        assert!(
            !prompts[2].contains("<prior-summary>"),
            "fresh mode never merges the prior summary"
        );
    }

    #[tokio::test]
    async fn an_api_error_fails_the_outcome_and_keeps_the_original_messages() {
        let (_client, summarizer) = scripted(vec![Err(ApiError::config("provider exploded"))]);
        let messages = conversation();
        let context = context_for(&messages);
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context.clone())
            .await;
        assert!(!outcome.success, "a step failure fails the pass");
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains("summarize") && error.contains("provider exploded"),
            "the error names the step and the cause: {error}"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(
            outcome.messages.first().map(Message::text_content),
            messages.first().map(Message::text_content),
        );
        assert!(
            summarizer.prior_summary().is_none(),
            "a failed pass commits no prior"
        );
        assert_eq!(outcome.evicted.len(), 0, "nothing left the feed");
    }

    #[tokio::test]
    async fn an_empty_summary_fails_the_pass_and_keeps_the_original_messages() {
        let (client, summarizer) = scripted(vec![ok(""), ok("[]")]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "a response with no usable text must fail the pass, not commit an empty summary"
        );
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains("summarize") && error.contains("empty summary"),
            "the error names the step and the cause: {error}"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(
            outcome.evicted.len(),
            0,
            "nothing is demoted when no summary replaces it"
        );
        assert!(
            summarizer.prior_summary().is_none(),
            "an empty summary is never committed as the prior"
        );
        assert_eq!(client.prompts().len(), 1, "the pass stops at step 1");

        let (_client, whitespace) = scripted(vec![ok("   \n\t"), ok("[]")]);
        let messages = conversation();
        let outcome = whitespace
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "a whitespace-only summary is the same empty class: {outcome:?}"
        );
        assert!(
            outcome
                .error
                .is_some_and(|error| error.contains("empty summary")),
            "the whitespace direction fails for the same named cause"
        );
    }

    #[tokio::test]
    async fn an_over_budget_summary_fails_the_pass() {
        let scripted = "a summary text of a known length";
        let as_message = Message::assistant(scripted.to_string());
        let at_budget = CompactionOutcome::estimate_tokens(std::slice::from_ref(&as_message));
        let one_under = at_budget.saturating_sub(1);

        let client = RecordingClient::new(vec![ok(scripted), ok("[]")]);
        let summarizer = QaSummarizer::new(
            Arc::clone(&client) as SharedApiClient,
            QaSummarizerConfig::default().with_max_summary_tokens(one_under),
        );
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "a summary over the hard ceiling must fail the pass, not ship through it"
        );
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains(&format!("{one_under}-token budget")),
            "the error names the budget figure it was measured against: {error}"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(
            outcome.evicted.len(),
            0,
            "nothing is demoted on a failed pass"
        );
        assert!(
            summarizer.prior_summary().is_none(),
            "an over-budget summary is never committed as the prior"
        );
        assert_eq!(client.prompts().len(), 1, "the pass stops at step 1");

        let boundary_client = RecordingClient::new(vec![ok(scripted), ok("[]")]);
        let boundary = QaSummarizer::new(
            Arc::clone(&boundary_client) as SharedApiClient,
            QaSummarizerConfig::default().with_max_summary_tokens(at_budget),
        );
        let messages = conversation();
        let outcome = boundary
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            outcome.success,
            "a summary estimating exactly at the budget passes — the ceiling rejects strictly"
        );
        assert_eq!(
            outcome.messages.first().map(Message::text_content),
            Some(scripted.to_string()),
            "the at-budget summary rides the output verbatim"
        );
    }

    #[test]
    fn question_parsing_tolerates_fences_lists_and_garbage() {
        assert_eq!(
            QaSummarizer::parse_questions("```json\n[\"first?\", \"second?\"]\n```"),
            vec!["first?".to_string(), "second?".to_string()],
            "a fenced JSON array parses"
        );
        assert_eq!(
            QaSummarizer::parse_questions("1. first?\n2) second?"),
            vec!["first?".to_string(), "second?".to_string()],
            "numbered list lines parse in both marker spellings"
        );
        assert_eq!(
            QaSummarizer::parse_questions("- first?\n* second?"),
            vec!["first?".to_string(), "second?".to_string()],
            "bulleted list lines parse in both marker spellings"
        );
        assert!(
            QaSummarizer::parse_questions("[]").is_empty(),
            "the explicit sufficient answer parses to no questions"
        );
        assert!(
            QaSummarizer::parse_questions("The summary looked complete to me.").is_empty(),
            "unparseable prose parses to no questions"
        );
        let many: Vec<String> = (0..12).map(|index| format!("q{index}?")).collect();
        let payload = serde_json::to_string(&many).unwrap();
        assert_eq!(
            QaSummarizer::parse_questions(&payload).len(),
            8,
            "the question list caps at eight"
        );
    }

    #[tokio::test]
    async fn unknown_answers_are_dropped_and_known_answers_fold_in() {
        let (_client, summarizer) = scripted(vec![
            ok("BASE-SUMMARY"),
            ok("[\"known question?\", \"unknown question?\"]"),
            ok("1. a real fact\n2. UNKNOWN"),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let summary_text = outcome.messages.first().map(Message::text_content).unwrap();
        assert!(
            summary_text.contains("## Facts recovered from the dropped context"),
            "the fold adds its section header: {summary_text}"
        );
        assert!(
            summary_text.contains("known question?: a real fact"),
            "answerable questions fold with their answers"
        );
        assert!(
            !summary_text.contains("unknown question?"),
            "an UNKNOWN answer drops its question entirely"
        );
    }

    #[tokio::test]
    async fn reset_clears_the_prior_summary() {
        let (client, summarizer) = scripted(vec![
            ok("SUMMARY-1"),
            ok("[]"),
            ok("SUMMARY-FRESH"),
            ok("[]"),
        ]);
        let first = conversation();
        let outcome_one = summarizer
            .compact(first.clone(), 40_000, context_for(&first))
            .await;
        assert!(outcome_one.success);
        summarizer.reset();
        assert!(
            summarizer.prior_summary().is_none(),
            "reset clears the accumulated prior"
        );
        let mut second = outcome_one.messages.clone();
        second.push(Message::user("user turn 6 asks about topic-6"));
        second.push(Message::assistant("assistant turn 6 decided fact-6"));
        second.push(Message::user("user turn 7 asks about topic-7"));
        second.push(Message::assistant("assistant turn 7 decided fact-7"));
        let outcome_two = summarizer
            .compact(second, 40_000, context_for(&first))
            .await;
        assert!(outcome_two.success);
        let prompts = client.prompts();
        assert!(
            !prompts[2].contains("<prior-summary>"),
            "the pass after reset produces a first-pass prompt"
        );
    }

    /// A counter answering a constant per message — a figure the
    /// heuristic estimate cannot produce on these fixtures, so the pin
    /// distinguishes the context counter from the static estimate.
    struct ConstantPerMessageCounter(u64);

    impl TokenCounter for ConstantPerMessageCounter {
        fn count(&self, messages: &[Message]) -> u64 {
            self.0.saturating_mul(messages.len() as u64)
        }
    }

    #[tokio::test]
    async fn the_outcome_counts_tokens_with_the_context_counter() {
        let counter = Arc::new(ConstantPerMessageCounter(7));
        let (_client, summarizer) = scripted(vec![ok("S"), ok("[]")]);
        let messages = conversation();
        let mut context = context_for(&messages);
        context.tokens_before = counter.count(&messages);
        context.counter = Arc::clone(&counter) as Arc<dyn TokenCounter>;
        let tokens_before = context.tokens_before;
        let outcome = summarizer.compact(messages, 40_000, context).await;
        assert_eq!(
            outcome.messages.len(),
            7,
            "one summary message plus the six preserved messages"
        );
        assert_eq!(
            outcome.tokens_after,
            7 * 7,
            "tokens_after is the context counter's figure — 7 per message — not the static heuristic estimate"
        );
        assert_eq!(
            outcome.tokens_saved,
            tokens_before - outcome.tokens_after,
            "tokens_saved is the before/after difference"
        );
    }

    #[tokio::test]
    async fn evicted_carries_the_dropped_slice_for_the_sink() {
        let (_client, summarizer) = scripted(vec![ok("S"), ok("[]")]);
        let messages = conversation();
        let expected_split = TokenSplitter::new()
            .with_preserve_recent(6)
            .with_min_messages(8)
            .split(&messages);
        let outcome = summarizer
            .compact(
                messages,
                40_000,
                context_for(&expected_split_to_compact_placeholder()),
            )
            .await;
        assert_eq!(
            outcome.evicted.len(),
            expected_split.to_compact.len(),
            "every dropped message rides evicted in conversation order"
        );
        for (evicted, dropped) in outcome.evicted.iter().zip(&expected_split.to_compact) {
            assert_eq!(evicted.text_content(), dropped.text_content());
        }
        let kept_texts: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        assert!(
            outcome
                .evicted
                .iter()
                .all(|msg| !kept_texts.contains(&msg.text_content())),
            "nothing listed as evicted survives in the output"
        );
    }

    fn expected_split_to_compact_placeholder() -> Vec<Message> {
        conversation()
    }

    #[tokio::test]
    async fn the_prior_summary_message_is_not_resummarized_as_conversation() {
        let (client, summarizer) = scripted(vec![
            ok("UNIQUE-SUMMARY-TEXT"),
            ok("[]"),
            ok("SUMMARY-2"),
            ok("[]"),
        ]);
        let first = conversation();
        let outcome_one = summarizer
            .compact(first.clone(), 40_000, context_for(&first))
            .await;
        assert!(outcome_one.success);
        let mut second = outcome_one.messages.clone();
        second.push(Message::user("user turn 6 asks about topic-6"));
        second.push(Message::assistant("assistant turn 6 decided fact-6"));
        second.push(Message::user("user turn 7 asks about topic-7"));
        second.push(Message::assistant("assistant turn 7 decided fact-7"));
        let outcome_two = summarizer
            .compact(second, 40_000, context_for(&first))
            .await;
        assert!(outcome_two.success);
        let second_prompt = &client.prompts()[2];
        assert_eq!(
            second_prompt.matches("UNIQUE-SUMMARY-TEXT").count(),
            1,
            "the prior summary appears exactly once — in <prior-summary>, not in <conversation>"
        );
    }

    #[tokio::test]
    async fn hook_instructions_and_additional_context_reach_the_prompt() {
        let (client, summarizer) = scripted(vec![ok("S"), ok("[]")]);
        let messages = conversation();
        let mut context = context_for(&messages);
        context.instructions = Some("Focus on API decisions".to_string());
        context.additional_context = vec![
            "The service is deploy-vacation".to_string(),
            "The branch is feat/qa-summarizer".to_string(),
        ];
        let outcome = summarizer.compact(messages, 40_000, context).await;
        assert!(outcome.success);
        let prompt = &client.prompts()[0];
        assert!(
            prompt.contains("Additional instructions from the host:\nFocus on API decisions"),
            "hook instructions ride the step-1 prompt"
        );
        assert!(
            prompt.contains("Additional context to weave in:\nThe service is deploy-vacation")
                && prompt
                    .contains("Additional context to weave in:\nThe branch is feat/qa-summarizer"),
            "every hook context fragment rides the step-1 prompt"
        );
    }

    #[tokio::test]
    async fn the_summarizer_compacts_through_the_manager() {
        let (_client, summarizer) = scripted(vec![ok("MANAGED-SUMMARY"), ok("[]")]);
        let manager = ContextManager::new(Arc::new(summarizer)).with_context_window(1_000_000);
        let messages = conversation();
        let result = manager.compact_manual(messages, 1).await;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                assert!(outcome.success);
                assert_eq!(
                    outcome.messages.first().map(Message::text_content),
                    Some("MANAGED-SUMMARY".to_string()),
                    "the summary message leads the manager's compacted history"
                );
            }
            _ => panic!("a scripted pass through the manager classifies Compacted"),
        }
    }

    #[test]
    fn a_computed_fraction_budgets_at_its_full_per_mille() {
        assert_eq!(
            budget_permille(0.1 + 0.2),
            300,
            "a computed fraction budgets at its true per-mille, not the saturated 184"
        );
        assert_eq!(
            budget_permille(0.4 * 0.75),
            300,
            "the second computed-fraction shape reaches its true share too"
        );
        assert_eq!(
            budget_permille(0.3),
            300,
            "short decimal literals are unchanged"
        );
        assert_eq!(budget_permille(0.25), 250);
        let config = QaSummarizerConfig::default().with_summary_budget_pct(0.1 + 0.2);
        assert_eq!(
            config.summary_budget(20_000),
            6_000,
            "a 30% share of a 20k target budgets 6 000 tokens, not the saturated 3 680"
        );
    }

    #[tokio::test]
    async fn step_three_never_resummarizes_the_prior_summary() {
        let (client, summarizer) = scripted(vec![
            ok("PRIOR-PASS-ONE"),
            ok("[]"),
            ok("SUMMARY-2"),
            ok("[\"what fact was fixed?\"]"),
            ok("1. the fixed fact"),
        ]);
        let first = conversation();
        let outcome_one = summarizer
            .compact(first.clone(), 40_000, context_for(&first))
            .await;
        assert!(outcome_one.success);
        let mut second = outcome_one.messages.clone();
        second.push(Message::user("user turn 6 asks about topic-6"));
        second.push(Message::assistant("assistant turn 6 decided fact-6"));
        second.push(Message::user("user turn 7 asks about topic-7"));
        second.push(Message::assistant("assistant turn 7 decided fact-7"));
        let outcome_two = summarizer
            .compact(second, 40_000, context_for(&first))
            .await;
        assert!(outcome_two.success);
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 5, "pass two runs all three steps");
        assert_eq!(
            prompts[2].matches("PRIOR-PASS-ONE").count(),
            1,
            "pass two's step-1 prompt carries the prior exactly once, in <prior-summary>"
        );
        assert!(
            prompts[4].contains("<conversation>") && prompts[4].contains("what fact was fixed?"),
            "the answer prompt pairs the questions with a transcript: {}",
            prompts[4]
        );
        assert!(
            !prompts[4].contains("PRIOR-PASS-ONE"),
            "step 3 renders the same accumulate-filtered transcript as step 1 — the prior \
             summary never re-enters the answer prompt as answerable conversation: {}",
            prompts[4]
        );
    }

    #[tokio::test]
    async fn a_transcript_budget_bounds_the_step_one_prompt() {
        let client = RecordingClient::new(vec![ok("S"), ok("[]")]);
        let summarizer = QaSummarizer::new(
            Arc::clone(&client) as SharedApiClient,
            QaSummarizerConfig::default().with_transcript_max_chars(60),
        );
        let messages = conversation();
        let context = context_for(&messages);
        let outcome = summarizer.compact(messages, 40_000, context).await;
        assert!(outcome.success);
        let prompt = &client.prompts()[0];
        assert!(
            prompt.contains("user turn 0 asks about topic-0"),
            "the head of the dropped slice still renders: {prompt}"
        );
        assert!(
            prompt.contains("[evicted "),
            "a saturated render closes with the truncation marker: {prompt}"
        );
        assert!(
            !prompt.contains("topic-3"),
            "messages past the transcript budget never reach the prompt: {prompt}"
        );
    }

    #[test]
    fn the_fold_cap_sheds_the_latest_answers_first() {
        let questions: Vec<String> = ["first question?", "second question?", "third question?"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let answers = "1. first answer\n2. second answer\n3. third answer";
        let two_fit = fold_answers("BASE", &questions, answers, 30);
        assert!(
            two_fit.contains("first question?: first answer")
                && two_fit.contains("second question?: second answer"),
            "a budget admitting two bullets folds exactly those two: {two_fit}"
        );
        assert!(
            !two_fit.contains("third question?"),
            "the latest answers are the ones shed: {two_fit}"
        );
        assert!(
            two_fit.starts_with("BASE"),
            "the summary itself always survives whole: {two_fit}"
        );
        let none_fit = fold_answers("BASE", &questions, answers, 10);
        assert_eq!(
            none_fit, "BASE",
            "when even the first bullet cannot fit, the summary returns unchanged with no section"
        );
    }
}
