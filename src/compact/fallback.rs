//! Compactor chain: run inner compactors in sequence until one
//! reduces the conversation **and lands within the compaction
//! target**.
//!
//! [`FallbackCompactor`] holds an ordered chain of [`ContextCompactor`]s and
//! runs them in order, returning the first outcome that succeeds,
//! reduces, and measures at-or-under the passed `target_tokens` with
//! the caller's own counter — a success that changed nothing does not
//! win (a later stage, the terminal truncator can always drop, still
//! might), and neither does a landing above the target (the target is
//! enforced, not advised; a later stage might land within it). The
//! default chain is quality-descending — [`QaSummarizer`]
//! → [`StructuredSummarizer`]
//! → [`TerminalCapture`] around a [`TruncatingCompactor`] — so
//! compaction degrades from best-effort summarization down to
//! deterministic truncation and, because the terminal stage cannot
//! fail, the chain as a whole never fails to compact: an
//! out-of-tokens condition stops surfacing as a failed pass with the
//! original messages returned intact.
//!
//! Each failed or declined stage is logged at `warn` and recorded in a
//! [`ChainReport`](FallbackCompactor::last_report), so the chain stays
//! debuggable while it survives; the winning stage's outcome rides out
//! with its [`evicted`](CompactionOutcome::evicted) handoff intact and
//! the one-line stage trail stamped on
//! [`stage`](CompactionOutcome::stage). A failed stage's returned
//! messages are discarded rather than handed
//! onward — every stage runs against the original input, so a stage that
//! fails with a partial or emptied list cannot corrupt the history its
//! successors compact. When every stage declines without carrying the
//! pass — errored, unchanged, or landed over the target — the last
//! non-failing outcome rides out as a success, so the caller's own
//! no-action classification (not a manufactured failure) decides what
//! the pass meant; the all-stages-failed outcome is reserved for chains
//! whose every stage errored.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::compact::FallbackCompactor;
//! use loopctl::compact::ContextManager;
//! use std::sync::Arc;
//!
//! let fallback = FallbackCompactor::default_chain(client);
//! let manager = ContextManager::new(Arc::new(fallback));
//! ```

use crate::api::SharedApiClient;
use crate::compact::ContextCompactor;
use crate::compact::qa_summarizer::QaSummarizer;
use crate::compact::qa_summarizer::QaSummarizerConfig;
use crate::compact::structured_summarizer::StructuredSummarizer;
use crate::compact::structured_summarizer::StructuredSummaryConfig;
use crate::compact::truncating::TruncatingCompactor;
use crate::compact::types::CompactionContext;
use crate::compact::types::CompactionOutcome;
use crate::error::recover_guard;
use crate::message::Message;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

/// The default floor [`TerminalCapture`] enforces on the terminal stage.
///
/// Four matches `TruncatingCompactor`'s own `preserve_recent` default, so the
/// default chain enforces what the truncator already promises — the floor is
/// insurance against a misconfigured custom chain, not a behavior change.
const DEFAULT_TERMINAL_MIN_RECENT: usize = 4;

/// The error text the all-stages-failed branch carries.
///
/// Reachable only in a chain with no non-failing terminal stage; the wording
/// names every stage, not just the last, because the honest statement is that
/// the whole chain declined.
const ALL_STAGES_FAILED: &str = "all compactor stages failed";

/// The decline reason recorded for a stage that succeeded without
/// reducing anything.
///
/// A success that changed nothing cannot carry the pass while a later
/// stage might still reduce — the decline is a chain verdict, not the
/// compactor's own error, so it gets its own label rather than riding
/// the failure text verbatim.
const NO_REDUCTION: &str = "stage returned success without reducing";

/// The decline reason recorded for a stage whose landing measured
/// above the chain's compaction target.
///
/// The target is enforced, not advised: a stage that reduced the
/// conversation but landed above `target_tokens` cannot carry the pass
/// while a later stage might land within it — the landing check runs
/// with the same counter the reduction check uses.
const OVER_TARGET: &str = "stage landed above the compaction target";

/// The trail closing for a run no stage landed within the target.
///
/// The mirror of `no stage reduced`: every stage that ran was declined
/// for its landing (or errored), and the last non-failing outcome rode
/// out for the caller's own classification.
const NO_FIT_CLOSING: &str = "no stage landed within the compaction target";

/// One stage in a fallback chain.
///
/// Carries the compactor and the name logs, reports, and telemetry address it
/// by. Build chains through [`FallbackCompactor::builder`] or
/// [`FallbackCompactor::default_chain`] and mutate the public fields only
/// in-crate — the type is `#[non_exhaustive]`, so struct literals compile
/// only inside the crate.
#[derive(Clone)]
#[non_exhaustive]
pub struct ChainStage {
    /// The name logs, reports, and telemetry address this stage by.
    ///
    /// The default chain uses the compactors' own type names; a custom stage
    /// carries whatever name its builder gave it.
    pub name: &'static str,

    /// The compactor this stage runs.
    ///
    /// Held as `Arc<dyn ContextCompactor>` so stages are object-safe and one
    /// compactor can serve several chains.
    pub compactor: Arc<dyn ContextCompactor>,
}

/// How one stage of a chain run fared.
///
/// One entry per stage that ran, in execution order; the entry after the
/// first success never exists, because the chain returns at the winner.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StageOutcome {
    /// The stage's name, as configured on the [`ChainStage`].
    ///
    /// The report's vocabulary — identical to the winner's stage name for
    /// the entry that carried the pass.
    pub name: String,

    /// Whether this stage carried the pass.
    ///
    /// `true` only on the stage whose reducing outcome the chain
    /// returned. A stage that succeeded without reducing anything is
    /// recorded `false` — its outcome did not carry the pass — with
    /// the decline reason in [`error`](Self::error).
    pub success: bool,

    /// The stage's own error string when it failed, or its decline
    /// reason when it succeeded without reducing.
    ///
    /// `None` on the winner and on stages that never ran (there are
    /// none — entries exist only for stages that ran).
    pub error: Option<String>,

    /// The token figure the stage's outcome self-reported.
    ///
    /// For failed stages this is the failing compactor's own figure; the
    /// chain records it verbatim rather than re-deriving one.
    pub tokens_after: u64,

    /// Wall-clock time spent inside the stage.
    ///
    /// Monotonic (`Instant`) diagnostics on the report, never a metric
    /// emission and never an input to a decision — the engine's clock-seamed
    /// telemetry owns duration reporting for the pass as a whole.
    pub duration: Duration,
}

/// Diagnostic record of one chain run.
///
/// Stored after each `compact` and readable through
/// [`FallbackCompactor::last_report`]; the one-line digest of this
/// report also rides every outcome out on
/// [`stage`](CompactionOutcome::stage) — each declined stage with its
/// reason, the winner marked — so a host reads provenance off the
/// outcome and reserves this full record for per-stage detail the
/// trail compresses away (durations, per-stage token counts).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ChainReport {
    /// The stages that ran, in execution order.
    ///
    /// The last entry is the stage that determined the result.
    pub stages: Vec<StageOutcome>,

    /// The index in `stages` of the stage whose output was used.
    ///
    /// `None` when no stage reduced the conversation — every stage
    /// errored, or every stage declined without reducing. Impossible
    /// with a [`TerminalCapture`]-terminated chain over a splittable
    /// history, but typed for honesty.
    pub winning_stage: Option<usize>,
}

/// Configuration for [`FallbackCompactor`].
///
/// Built from [`Default`](Self::default) and adjusted with the `with_*`
/// builders, so a stored config is always usable.
///
/// # Example
///
/// ```
/// use loopctl::compact::fallback::FallbackConfig;
///
/// let config = FallbackConfig::default().with_terminal_min_recent(8);
/// assert_eq!(config.terminal_min_recent(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// The floor the terminal stage's `preserve_recent` is raised to.
    ///
    /// Insurance against a misconfigured custom chain dropping the active
    /// turn at the last-resort truncate; clamped to at least one.
    terminal_min_recent: usize,
}

impl FallbackConfig {
    /// The validated configuration with every field at its default.
    ///
    /// The single construction site for defaults; [`Default`] delegates here.
    fn fresh() -> Self {
        Self {
            terminal_min_recent: DEFAULT_TERMINAL_MIN_RECENT,
        }
    }

    /// Set the floor the terminal stage's `preserve_recent` is raised to.
    ///
    /// Guarantees the active turn and its neighbors survive every stage,
    /// including the last-resort truncate; values below `1` clamp to `1`.
    #[must_use]
    pub fn with_terminal_min_recent(mut self, floor: usize) -> Self {
        self.terminal_min_recent = floor.max(1);
        self
    }

    /// The floor the terminal stage's `preserve_recent` is raised to.
    ///
    /// The stored value after the `max(1)` clamp.
    #[must_use]
    pub fn terminal_min_recent(&self) -> usize {
        self.terminal_min_recent
    }
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self::fresh()
    }
}

/// Guarantees the last messages survive the terminal compaction stage.
///
/// Wraps a [`TruncatingCompactor`] and enforces a structural floor: the inner
/// truncator is reconstructed with
/// `preserve_recent = max(configured, floor)`, so the guarantee lives in the
/// compactor that actually runs — no per-call rewriting, no post-hoc
/// stitching that could violate the truncator's tool-pair boundary. The inner
/// truncator's `min_messages` carries over unchanged. A chain whose terminal
/// is an LLM compactor needs no capture: both LLM compactors preserve their
/// recent tail verbatim through their own `TokenSplitter`.
#[derive(Clone)]
pub struct TerminalCapture {
    /// The truncator, configured with the enforced floor.
    ///
    /// Reconstructed from the caller's truncator at wrap time, so the
    /// guarantee lives in the compactor that actually runs.
    inner: Arc<TruncatingCompactor>,

    /// The floor the truncator's `preserve_recent` was raised to.
    ///
    /// The configured floor, even when the wrapped truncator arrived with a
    /// higher `preserve_recent` of its own.
    floor: usize,
}

impl TerminalCapture {
    /// Wrap a truncator, enforcing `floor` preserved recent messages.
    ///
    /// The wrapped truncator keeps its `min_messages` and takes
    /// `preserve_recent = max(its own, floor)`; values below `1` clamp
    /// through the truncator's own builder. The truncator is read, not
    /// consumed — the caller keeps its copy untouched.
    #[must_use]
    pub fn new(truncator: &TruncatingCompactor, floor: usize) -> Self {
        let enforced = truncator.preserve_recent().max(floor);
        let rebuilt = TruncatingCompactor::new()
            .with_preserve_recent(enforced)
            .with_min_messages(truncator.min_messages());
        Self {
            inner: Arc::new(rebuilt),
            floor,
        }
    }

    /// The floor this capture enforces.
    ///
    /// The configured floor, not the possibly-higher `preserve_recent` the
    /// inner truncator arrived with.
    #[must_use]
    pub fn floor(&self) -> usize {
        self.floor
    }
}

impl ContextCompactor for TerminalCapture {
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        self.inner.compact(messages, target_tokens, context)
    }
}

/// Compactor chain: try each stage in order, return the first success.
///
/// Construct with [`default_chain`](Self::default_chain) for the
/// quality-descending LLM → structured → truncate chain, or
/// [`builder`](Self::builder) for a custom one. A chain whose last stage
/// cannot fail (any [`TerminalCapture`]) never fails itself, so
/// [`ContextManager::ensure_context_fits`](crate::compact::ContextManager::ensure_context_fits)
/// stops returning [`ContextOverflow`](crate::compact::ContextOverflow) for
/// an out-of-tokens condition.
pub struct FallbackCompactor {
    /// The stages, in execution order.
    ///
    /// The first success wins; the chain never revisits an earlier stage.
    stages: Vec<ChainStage>,

    /// The last completed run's report.
    ///
    /// Written once at each `compact` exit and never held across an await;
    /// reflects the last completed run, concurrent runs each overwriting in
    /// completion order.
    last_report: Mutex<Option<ChainReport>>,
}

impl FallbackCompactor {
    /// Build the default chain for the given API client.
    ///
    /// `QaSummarizer` → `StructuredSummarizer` →
    /// `TerminalCapture(TruncatingCompactor)`, both LLM stages on the
    /// caller's client with their default configs — pass a dedicated cheaper
    /// client if the loop's own model is the expensive one. The terminal
    /// enforces the configured floor (4 by default).
    #[must_use]
    pub fn default_chain(client: SharedApiClient) -> Self {
        Self::builder()
            .stage(
                "QaSummarizer",
                Arc::new(QaSummarizer::new(
                    Arc::clone(&client),
                    QaSummarizerConfig::default(),
                )),
            )
            .stage(
                "StructuredSummarizer",
                Arc::new(StructuredSummarizer::new(
                    client,
                    StructuredSummaryConfig::default(),
                )),
            )
            .terminal(&TruncatingCompactor::new())
            .build()
    }

    /// Start a custom chain builder.
    ///
    /// Append stages with [`ChainBuilder::stage`], finish with
    /// [`ChainBuilder::terminal`] so the chain cannot fail, then `build`.
    /// Configuration — the terminal floor — is set on the builder through
    /// [`ChainBuilder::config`] before `terminal()`, because the floor is
    /// captured inside the [`TerminalCapture`] when the terminal stage is
    /// appended.
    #[must_use]
    pub fn builder() -> ChainBuilder {
        ChainBuilder {
            stages: Vec::new(),
            config: FallbackConfig::default(),
        }
    }

    /// The last completed run's report, if any.
    ///
    /// `None` until the first `compact`; reflects the last completed run,
    /// not the last adopted history.
    pub fn last_report(&self) -> Option<ChainReport> {
        recover_guard(self.last_report.lock()).clone()
    }

    /// The number of stages in the chain.
    ///
    /// The default chain answers three; an empty chain answers zero and
    /// fails honestly at compact time.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the chain holds no stages.
    ///
    /// A misconfiguration: `compact` on an empty chain returns the
    /// all-stages-failed outcome rather than a silent no-op.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// The names of the stages, in execution order.
    ///
    /// The report's vocabulary without running a pass; the default chain
    /// answers with its three compactors' type names.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.stages.iter().map(|stage| stage.name).collect()
    }

    /// Record the report of a finished run.
    ///
    /// The one write site per `compact` exit; the lock is taken and released
    /// here only, never across an await.
    fn store_report(&self, report: ChainReport) {
        *recover_guard(self.last_report.lock()) = Some(report);
    }
}

/// Builder for a custom [`FallbackCompactor`] chain.
///
/// Append any stages with [`stage`](Self::stage), then finish with
/// [`terminal`](Self::terminal) — the never-fail shape — before `build`.
pub struct ChainBuilder {
    /// The stages appended so far, in execution order.
    ///
    /// `build` freezes this list into the chain verbatim.
    stages: Vec<ChainStage>,

    /// The configuration the built chain carries.
    ///
    /// The terminal floor applies at every later `terminal` call.
    config: FallbackConfig,
}

impl ChainBuilder {
    /// Append a stage.
    ///
    /// The name is the report/log/telemetry label; it should identify the
    /// compactor for a human reading `last_report`.
    #[must_use]
    pub fn stage(mut self, name: &'static str, compactor: Arc<dyn ContextCompactor>) -> Self {
        self.stages.push(ChainStage { name, compactor });
        self
    }

    /// Append a `TerminalCapture(TruncatingCompactor)` as the final stage.
    ///
    /// The capture enforces the configured `terminal_min_recent` floor, so
    /// the chain cannot fail. The recommended last call before `build`.
    #[must_use]
    pub fn terminal(mut self, truncator: &TruncatingCompactor) -> Self {
        let floor = self.config.terminal_min_recent();
        self.stages.push(ChainStage {
            name: "TruncatingCompactor",
            compactor: Arc::new(TerminalCapture::new(truncator, floor)),
        });
        self
    }

    /// Set the configuration.
    ///
    /// The `terminal_min_recent` floor applies to every later
    /// [`terminal`](Self::terminal) call.
    #[must_use]
    pub fn config(mut self, config: FallbackConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the chain.
    ///
    /// An empty chain builds (and fails honestly at compact time). A chain
    /// whose last stage is not the terminal truncate draws a best-effort
    /// heuristic warning — a `dyn` compactor cannot be inspected, so the
    /// check reads the stage name only.
    #[must_use]
    pub fn build(self) -> FallbackCompactor {
        if let Some(last) = self.stages.last()
            && !looks_terminal(last.name)
        {
            tracing::warn!(
                target: "loopctl::compact",
                stage = last.name,
                "fallback chain ends in a non-terminal stage; it can fail to compact"
            );
        }
        FallbackCompactor {
            stages: self.stages,
            last_report: Mutex::new(None),
        }
    }
}

/// Whether a stage name heuristically names a non-failing terminal.
///
/// The honest bound on `dyn` inspection: the default terminal's exact name,
/// or a name that says truncate/capture. A custom never-fail terminal under
/// another name draws the warning; it is advisory only.
fn looks_terminal(name: &str) -> bool {
    name == "TruncatingCompactor"
        || name.to_lowercase().contains("truncat")
        || name.to_lowercase().contains("capture")
}

impl ContextCompactor for FallbackCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let mut report = ChainReport::default();
            let original = messages;
            let mut ride_out: Option<CompactionOutcome> = None;
            for (index, stage) in self.stages.iter().enumerate() {
                let started = Instant::now();
                let tokens_before = context.counter.count(&original);
                let mut stage_context = context.clone();
                stage_context.tokens_before = tokens_before;
                let outcome = stage
                    .compactor
                    .compact(original.clone(), target_tokens, stage_context)
                    .await;
                let duration = started.elapsed();
                if outcome.success {
                    let measured_after = context.counter.count(&outcome.messages);
                    let reduced = measured_after < tokens_before;
                    if !reduced {
                        tracing::warn!(
                            target: "loopctl::compact",
                            stage = stage.name,
                            "fallback chain stage returned success without reducing; trying the next stage"
                        );
                        report.stages.push(StageOutcome {
                            name: stage.name.to_string(),
                            success: false,
                            error: Some(NO_REDUCTION.to_string()),
                            tokens_after: outcome.tokens_after,
                            duration,
                        });
                        ride_out = Some(outcome);
                        continue;
                    }
                    let fits = measured_after <= target_tokens;
                    if !fits {
                        tracing::warn!(
                            target: "loopctl::compact",
                            stage = stage.name,
                            measured_after,
                            target_tokens,
                            "fallback chain stage landed above the compaction target; trying the next stage"
                        );
                        report.stages.push(StageOutcome {
                            name: stage.name.to_string(),
                            success: false,
                            error: Some(OVER_TARGET.to_string()),
                            tokens_after: outcome.tokens_after,
                            duration,
                        });
                        ride_out = Some(outcome);
                        continue;
                    }
                    for failed in &report.stages {
                        emit_degradation(&failed.name, stage.name);
                    }
                    report.stages.push(StageOutcome {
                        name: stage.name.to_string(),
                        success: true,
                        error: None,
                        tokens_after: outcome.tokens_after,
                        duration,
                    });
                    report.winning_stage = Some(index);
                    let trail = stage_trail(&report);
                    self.store_report(report);
                    tracing::info!(
                        target: "loopctl::compact",
                        stage = stage.name,
                        tokens_after = outcome.tokens_after,
                        "fallback chain succeeded at a stage"
                    );
                    emit_pass(stage.name);
                    return stamped(outcome, &trail);
                }
                tracing::warn!(
                    target: "loopctl::compact",
                    stage = stage.name,
                    error = outcome.error.as_deref().unwrap_or("unknown"),
                    "fallback chain stage failed; trying the next stage"
                );
                report.stages.push(StageOutcome {
                    name: stage.name.to_string(),
                    success: false,
                    error: outcome.error.clone(),
                    tokens_after: outcome.tokens_after,
                    duration,
                });
            }
            let tokens_after = context.counter.count(&original);
            let trail = stage_trail(&report);
            self.store_report(report);
            if let Some(ride_out) = ride_out {
                return stamped(ride_out, &trail);
            }
            stamped(
                CompactionOutcome::failed(original, tokens_after, ALL_STAGES_FAILED),
                &trail,
            )
        })
    }
}

/// Stamp the trail on an outcome unless there is none.
///
/// An empty trail — a chain with no stages — leaves
/// [`stage`](crate::compact::CompactionOutcome::stage) `None`, the
/// documented "no provenance available" value, instead of an empty
/// string a host would have to special-case.
fn stamped(outcome: CompactionOutcome, trail: &str) -> CompactionOutcome {
    if trail.is_empty() {
        outcome
    } else {
        outcome.with_stage(trail)
    }
}

/// Render the one-line stage trail for a chain outcome.
///
/// Each stage the pass declined through appears as `name: reason` — its
/// recorded error or decline text — and the winning stage as
/// `name (won)`; when no stage won, every stage lists its reason and
/// the trail closes with the wall the chain hit: `no stage landed
/// within the compaction target` when any decline was an over-target
/// landing (the target bound the run, even if some stages also failed
/// to reduce), otherwise `no stage reduced`. The trail is the
/// digest a host reads off the outcome;
/// [`ChainReport`](FallbackCompactor::last_report) stays the full
/// record.
fn stage_trail(report: &ChainReport) -> String {
    use std::fmt::Write as _;
    let mut trail = String::new();
    for (index, stage) in report.stages.iter().enumerate() {
        if index > 0 {
            trail.push_str("; ");
        }
        if report.winning_stage == Some(index) {
            let _ignored = write!(trail, "{} (won)", stage.name);
            continue;
        }
        let _ignored = write!(
            trail,
            "{}: {}",
            stage.name,
            stage.error.as_deref().unwrap_or("unknown")
        );
    }
    if report.winning_stage.is_none() && !report.stages.is_empty() {
        let any_over_target = report
            .stages
            .iter()
            .any(|stage| stage.error.as_deref() == Some(OVER_TARGET));
        if any_over_target {
            trail.push_str("; ");
            trail.push_str(NO_FIT_CLOSING);
        } else {
            trail.push_str("; no stage reduced");
        }
    }
    trail
}

/// Emit the pass counter event at the stage that served it.
///
/// One event per succeeded run, labeled with the winning stage's name — the
/// honest label for custom chains; the default chain's names are its three
/// tiers. Failed runs emit nothing here (their visibility rides the stage
/// warnings and the report).
fn emit_pass(stage: &str) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.compaction.passes",
        stage,
        "one compaction pass served by one fallback stage"
    );
}

/// Emit one degradation event per failed stage the winner declined through.
///
/// `from` names the stage that failed, `to` the stage that carried the pass —
/// a climbing `to=TruncatingCompactor` rate is the degradation alarm.
fn emit_degradation(from: &str, to: &str) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.compaction.degradations",
        from,
        to,
        "one fallback stage declined and a later stage carried the pass"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ApiClient;
    use crate::api::NonStreamingResponse;
    use crate::api::StreamRequest;
    use crate::api::error::ApiError;
    use crate::compact::HeuristicTokenCounter;
    use crate::compact::TokenCounter;
    use crate::compact::types::CompactReason;
    use crate::compact::{ContextManager, EnsureContextResult};
    use crate::stream::StreamEvent;
    use futures::Stream;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    /// What one scripted stage returns for its call.
    ///
    /// One entry per call the double serves; the script runs out only if a
    /// chain calls a stage more times than the test scripted.
    enum Scripted {
        /// Fail the pass with this error text.
        ///
        /// The error surfaces verbatim in the chain's report; the returned
        /// messages are the stage's input verbatim, the honest shape.
        Fail(&'static str),

        /// Fail the pass and return an emptied message list.
        ///
        /// The corrupted-history shape a misbehaving stage can produce —
        /// a failed outcome only "typically" carries the original input,
        /// so the chain's defenses are pinned against this arm.
        FailEmpty(&'static str),

        /// Succeed by compacting down to the final message.
        ///
        /// The shrink is drastic on purpose — the winner's output must be
        /// unmistakably this stage's.
        ShrinkToLast,

        /// Succeed without changing anything.
        ///
        /// The no-change shape the chain must not treat as a win: an
        /// outcome that reduced nothing cannot carry the pass while a
        /// later stage might still reduce.
        Unchanged,

        /// Succeed by compacting down to one verbose message.
        ///
        /// The length-only reduction shape: fewer messages, more
        /// tokens — a summary wordier than the conversation it
        /// replaced, the "reduction" a token-budget consumer cannot
        /// use.
        GrowToVerbose,

        /// Succeed by compacting down to one message of the given
        /// token size.
        ///
        /// The landing-control shape: a genuine reduction whose size
        /// the script pins exactly, so a test can place a stage's
        /// landing above or below the chain's target at will.
        ShrinkToTokens(u64),
    }

    /// What one scripted stage observed per call.
    ///
    /// The wiring pins assert against this record rather than the chain's
    /// own bookkeeping.
    #[derive(Debug, Clone)]
    struct Seen {
        target_tokens: u64,
        tokens_before: u64,
        reason: CompactReason,
        context_window: u64,
        turn: usize,
        messages: Vec<String>,
    }

    /// A `ContextCompactor` double serving scripted results in order and
    /// recording everything each call observed.
    struct ScriptedCompactor {
        script: StdMutex<VecDeque<Scripted>>,
        calls: StdMutex<Vec<Seen>>,
    }

    impl ScriptedCompactor {
        /// A double whose calls consume the script front-first.
        ///
        /// A call past the script's end panics — a test bug, not a fixture.
        fn new(script: Vec<Scripted>) -> Arc<Self> {
            Arc::new(Self {
                script: StdMutex::new(script.into()),
                calls: StdMutex::new(Vec::new()),
            })
        }

        /// How many calls this stage served.
        ///
        /// The short-circuit pin reads zero on a stage that never ran.
        fn call_count(&self) -> usize {
            self.calls.lock().expect("calls lock").len()
        }

        /// What each call observed, in order.
        ///
        /// Cloned out from under the lock; tests read after the await.
        fn seen(&self) -> Vec<Seen> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    impl ContextCompactor for ScriptedCompactor {
        fn compact(
            &self,
            messages: Vec<Message>,
            target_tokens: u64,
            context: CompactionContext,
        ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
            let next = self
                .script
                .lock()
                .expect("script lock")
                .pop_front()
                .expect("every scripted call gets a result");
            self.calls.lock().expect("calls lock").push(Seen {
                target_tokens,
                tokens_before: context.tokens_before,
                reason: context.reason,
                context_window: context.context_window,
                turn: context.turn,
                messages: messages.iter().map(Message::text_content).collect(),
            });
            Box::pin(std::future::ready(match next {
                Scripted::Fail(error) => {
                    CompactionOutcome::failed(messages, context.tokens_before, error)
                }
                Scripted::FailEmpty(error) => {
                    CompactionOutcome::failed(Vec::new(), context.tokens_before, error)
                }
                Scripted::ShrinkToLast => {
                    let last = messages.last().cloned().unwrap_or_else(|| {
                        Message::user("the scripted shrink always has a last message")
                    });
                    CompactionOutcome::compacted(vec![last], context.tokens_before, 1)
                }
                Scripted::Unchanged => CompactionOutcome::no_change(messages),
                Scripted::GrowToVerbose => CompactionOutcome::compacted(
                    vec![Message::assistant("v".repeat(5_000))],
                    context.tokens_before,
                    19,
                ),
                Scripted::ShrinkToTokens(tokens) => CompactionOutcome::compacted(
                    vec![Message::assistant("s".repeat(
                        usize::try_from(tokens.saturating_mul(4)).unwrap_or(usize::MAX),
                    ))],
                    context.tokens_before,
                    tokens,
                ),
            }))
        }
    }

    /// An `ApiClient` double whose every call fails.
    ///
    /// Drives both LLM stages of a default chain to decline without any
    /// network — the shape every truncate-carries-the-pass pin needs.
    struct FailingClient;

    impl ApiClient for FailingClient {
        fn model(&self) -> String {
            "fallback-test".to_string()
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            Box::pin(std::future::ready(Err(ApiError::api(
                "the scripted provider is unreachable",
            ))))
        }
    }

    /// Ten user/assistant pairs with unique, probe-able text per message.
    ///
    /// Each message names its turn twice, so prompt-presence and
    /// intact-handoff assertions probe exact substrings without colliding.
    fn conversation() -> Vec<Message> {
        let mut messages = Vec::new();
        for turn in 0..10 {
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
            pinned: Vec::new(),
        }
    }

    #[test]
    fn the_default_chain_holds_three_stages_in_quality_order() {
        let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
        assert_eq!(chain.len(), 3, "the default chain is three stages");
        assert_eq!(
            chain.stage_names(),
            vec![
                "QaSummarizer",
                "StructuredSummarizer",
                "TruncatingCompactor"
            ],
            "the stages are quality-descending"
        );
        assert!(!chain.is_empty());
    }

    #[tokio::test]
    async fn a_first_stage_success_short_circuits_the_chain() {
        let first = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .terminal(&TruncatingCompactor::new())
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert!(outcome.success);
        assert_eq!(
            outcome.messages.len(),
            1,
            "the returned outcome is the first stage's"
        );
        assert_eq!(
            second.call_count(),
            0,
            "a first-stage success never runs the later stages"
        );
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(
            report.winning_stage,
            Some(0),
            "the report names the first stage as the winner"
        );
        assert_eq!(report.stages.len(), 1, "only the winner ran");
    }

    #[tokio::test]
    async fn a_failed_first_stage_falls_through_to_the_second() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("stage one refused")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .terminal(&TruncatingCompactor::new())
            .build();
        let outcome = chain
            .compact(conversation(), 40_000, context_for(&conversation()))
            .await;
        assert!(outcome.success, "the second stage carries the pass");
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(report.winning_stage, Some(1));
        assert_eq!(
            report.stages[0].error.as_deref(),
            Some("stage one refused"),
            "the failed stage's error is recorded: {:?}",
            report.stages[0]
        );
        assert!(!report.stages[0].success);
        assert!(report.stages[1].success);
    }

    #[tokio::test]
    async fn two_llm_failures_fall_through_to_the_terminal_truncate() {
        let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
        let messages = conversation();
        let outcome = chain
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success, "the terminal truncate carries the pass");
        assert!(
            outcome.messages.len() < messages.len(),
            "the terminal actually compacted: {} -> {}",
            messages.len(),
            outcome.messages.len()
        );
        assert_eq!(
            outcome.messages.last().map(Message::text_content),
            messages.last().map(Message::text_content),
            "the newest message survives the truncate"
        );
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(report.winning_stage, Some(2));
        assert_eq!(report.stages.len(), 3, "all three stages ran");
        assert!(
            report.stages.iter().take(2).all(|stage| !stage.success),
            "both LLM stages failed first"
        );
        assert!(!outcome.evicted.is_empty(), "the truncate evicted content");
    }

    #[tokio::test]
    async fn a_terminal_terminated_chain_never_fails_to_compact() {
        for shape in [1usize, 6, 20] {
            let failing = ScriptedCompactor::new(vec![Scripted::Fail("no"), Scripted::Fail("no")]);
            let chain = FallbackCompactor::builder()
                .stage("failing", Arc::clone(&failing) as Arc<dyn ContextCompactor>)
                .terminal(&TruncatingCompactor::new())
                .build();
            let messages: Vec<Message> = conversation().into_iter().take(shape * 2).collect();
            let outcome = chain
                .compact(messages, 40_000, context_for(&conversation()))
                .await;
            assert!(
                outcome.success,
                "the headline invariant: a terminal-terminated chain succeeds ({shape} pairs)"
            );
        }
    }

    #[tokio::test]
    async fn an_all_stages_failed_chain_fails_honestly() {
        let only = ScriptedCompactor::new(vec![Scripted::Fail("the lone stage failed")]);
        let chain = FallbackCompactor::builder()
            .stage("only", Arc::clone(&only) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "without a terminal the chain fails honestly, never silently no-ops"
        );
        assert_eq!(
            outcome.error.as_deref(),
            Some("all compactor stages failed"),
            "the failure names the whole chain: {:?}",
            outcome.error
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert!(outcome.evicted.is_empty(), "nothing left the feed");
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(report.winning_stage, None, "no stage won");
    }

    #[tokio::test]
    async fn terminal_capture_raises_a_low_preserve_recent_to_the_floor() {
        let capture = TerminalCapture::new(&TruncatingCompactor::new().with_preserve_recent(1), 4);
        assert_eq!(capture.floor(), 4, "the floor reports the configured value");
        let messages = conversation();
        let outcome = capture
            .compact(messages.clone(), 65, context_for(&messages))
            .await;
        assert!(outcome.success);
        let kept: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        let mut expected = vec![messages[0].text_content()];
        expected.extend(
            messages
                .iter()
                .rev()
                .take(4)
                .map(Message::text_content)
                .collect::<Vec<String>>()
                .into_iter()
                .rev(),
        );
        assert_eq!(
            kept, expected,
            "the capture preserves the leading message plus the last four verbatim"
        );
        let low = TruncatingCompactor::new()
            .with_preserve_recent(1)
            .with_min_messages(8);
        let raised = TerminalCapture::new(&low, 4);
        let outcome = raised
            .compact(conversation(), 65, context_for(&conversation()))
            .await;
        assert_eq!(
            outcome.messages.len(),
            5,
            "the raised truncator keeps the leading message plus four — unwrapped it would \
             have kept two"
        );
    }

    #[tokio::test]
    async fn the_default_chain_terminal_preserves_the_configured_floor() {
        let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
        let messages = conversation();
        let outcome = chain
            .compact(messages.clone(), 65, context_for(&messages))
            .await;
        assert!(outcome.success);
        assert_eq!(
            outcome.messages.len(),
            5,
            "the default chain's terminal keeps the leading message plus the default floor"
        );
        let floored = FallbackCompactor::builder()
            .config(FallbackConfig::default().with_terminal_min_recent(6))
            .stage(
                "StructuredSummarizer",
                Arc::new(StructuredSummarizer::new(
                    Arc::new(FailingClient),
                    StructuredSummaryConfig::default(),
                )) as Arc<dyn ContextCompactor>,
            )
            .terminal(&TruncatingCompactor::new())
            .build();
        let messages = conversation();
        let outcome = floored
            .compact(messages.clone(), 90, context_for(&messages))
            .await;
        assert!(outcome.success);
        assert_eq!(
            outcome.messages.len(),
            7,
            "a configured floor of six survives the failing LLM stage (plus the leading \
             message)"
        );
    }

    #[tokio::test]
    async fn each_stage_receives_the_running_tokens_before_and_a_stable_context() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("stage one refused")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let expected = CompactionOutcome::estimate_tokens(&messages);
        chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        let seen = second.seen();
        assert_eq!(
            seen[0].tokens_before, expected,
            "a failed stage changes nothing, so stage two sees the original size"
        );
        assert_eq!(
            first.seen()[0].tokens_before,
            expected,
            "stage one saw the original size too"
        );
        for stage in [first.seen(), second.seen()] {
            assert_eq!(stage[0].reason, CompactReason::ThresholdExceeded);
            assert_eq!(stage[0].context_window, 1_000_000);
            assert_eq!(stage[0].turn, 3);
        }
    }

    #[tokio::test]
    async fn every_stage_receives_the_same_target_tokens() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("stage one refused")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        chain
            .compact(conversation(), 12_345, context_for(&conversation()))
            .await;
        for stage in [first.seen(), second.seen()] {
            for seen in stage {
                assert_eq!(
                    seen.target_tokens, 12_345,
                    "every stage aims at the chain's one target"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_failed_stage_hands_the_original_messages_to_the_next() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("stage one refused")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let expected: Vec<String> = messages.iter().map(Message::text_content).collect();
        chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert_eq!(
            second.seen()[0].messages,
            expected,
            "no partial or half-compacted state leaks between stages"
        );
    }

    #[tokio::test]
    async fn an_unchanged_success_does_not_win_the_chain() {
        let first = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert_eq!(
            second.call_count(),
            1,
            "a stage that returned success without reducing anything cannot \
             win the chain — the pass must fall through to the next stage"
        );
        assert_eq!(
            outcome.messages.len(),
            1,
            "the reducing stage's output is the pass's output"
        );
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.winning_stage,
            Some(1),
            "the winner is the stage that actually reduced"
        );
        assert!(
            !report.stages[0].success,
            "the unchanged stage is recorded as not carrying the pass"
        );
    }

    #[tokio::test]
    async fn chain_outcome_carries_the_winning_stage_trail() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("transport died")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let third = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .stage("third", Arc::clone(&third) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        let trail = outcome
            .stage
            .as_deref()
            .expect("a chain outcome names the stages that decided it");
        assert!(
            trail.starts_with("first: transport died; "),
            "the trail names every declined stage with its reason: {trail}"
        );
        assert!(
            trail.ends_with("second (won)"),
            "the trail names the winning stage as the winner: {trail}"
        );
        assert!(
            third.call_count() == 0,
            "a stage after the winner never runs"
        );
    }

    #[tokio::test]
    async fn a_no_reduction_ride_out_names_every_stage_and_states_none_reduced() {
        let first = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let second = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        let trail = outcome
            .stage
            .as_deref()
            .expect("a ride-out outcome still names the stages that decided it");
        assert!(
            trail.contains("first: stage returned success without reducing")
                && trail.contains("second: stage returned success without reducing"),
            "the trail names every stage with its decline reason: {trail}"
        );
        assert!(
            trail.ends_with("no stage reduced"),
            "a pass no stage carried states that outright: {trail}"
        );
    }

    #[tokio::test]
    async fn a_length_reducing_token_growing_stage_does_not_win_the_chain() {
        let first = ScriptedCompactor::new(vec![Scripted::GrowToVerbose]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert_eq!(
            second.call_count(),
            1,
            "a stage that answered with fewer messages but more tokens has \
             not reduced the conversation — the chain's progress test is the \
             token drop the machine's own no-progress guard measures, so the \
             pass must fall through to the truncating stage"
        );
        assert_eq!(
            outcome.messages.len(),
            1,
            "the reducing stage's output is the pass's output"
        );
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.winning_stage,
            Some(1),
            "the winner is the stage that reduced the token count"
        );
        assert!(
            !report.stages[0].success,
            "the token-growing stage is recorded as declined"
        );
    }

    #[tokio::test]
    async fn a_chain_whose_stages_never_reduce_returns_the_last_unchanged_outcome() {
        let first = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let second = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let expected: Vec<String> = messages.iter().map(Message::text_content).collect();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert_eq!(
            second.call_count(),
            1,
            "every stage runs when none of them reduces"
        );
        assert!(
            outcome.success,
            "no stage errored, so the chain does not manufacture a failure"
        );
        let rendered: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        assert_eq!(
            rendered, expected,
            "the conversation rides out unchanged for the caller's own \
             no-action classification"
        );
        assert!(
            outcome.error.is_none(),
            "the all-stages-failed text is reserved for stages that errored"
        );
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.winning_stage, None,
            "no stage reduced, so no stage won"
        );
    }

    #[tokio::test]
    async fn a_failed_stage_s_emptied_output_never_reaches_the_next_stage() {
        let first =
            ScriptedCompactor::new(vec![Scripted::FailEmpty("stage one mangled its return")]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let expected: Vec<String> = messages.iter().map(Message::text_content).collect();
        chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert_eq!(
            second.seen()[0].messages,
            expected,
            "every stage sees the original input — a failed stage's returned list is \
             discarded, never handed onward"
        );
    }

    #[tokio::test]
    async fn the_all_failed_outcome_carries_the_original_not_a_stage_s_emptied_list() {
        let first =
            ScriptedCompactor::new(vec![Scripted::FailEmpty("stage one mangled its return")]);
        let second =
            ScriptedCompactor::new(vec![Scripted::FailEmpty("stage two mangled its return")]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let expected: Vec<String> = messages.iter().map(Message::text_content).collect();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert!(
            !outcome.success,
            "both stages failed; the chain fails honestly"
        );
        let returned: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        assert_eq!(
            returned, expected,
            "the chain's failure returns the original input, never a failed stage's emptied list"
        );
        assert!(
            outcome.evicted.is_empty(),
            "a mangled stage's drops are not evictions — nothing left the chain's feed"
        );
    }

    #[tokio::test]
    async fn last_report_records_every_stage_that_ran() {
        let first = ScriptedCompactor::new(vec![Scripted::Fail("stage one refused")]);
        let second = ScriptedCompactor::new(vec![Scripted::Fail("stage two refused")]);
        let third = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .stage("third", Arc::clone(&third) as Arc<dyn ContextCompactor>)
            .build();
        chain
            .compact(conversation(), 40_000, context_for(&conversation()))
            .await;
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(report.stages.len(), 3, "three stages ran, three recorded");
        assert_eq!(report.winning_stage, Some(2));
        assert_eq!(
            report
                .stages
                .iter()
                .map(|stage| stage.name.as_str())
                .collect::<Vec<&str>>(),
            vec!["first", "second", "third"],
            "the report is in execution order"
        );
    }

    #[tokio::test]
    async fn an_empty_chain_builds_and_fails_honestly() {
        let chain = FallbackCompactor::builder().build();
        assert!(chain.is_empty(), "an empty chain builds");
        assert_eq!(chain.len(), 0);
        let messages = conversation();
        let outcome = chain
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(!outcome.success);
        assert_eq!(
            outcome.error.as_deref(),
            Some("all compactor stages failed"),
            "the empty chain fails with the chain-level error"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the messages pass through untouched"
        );
        assert!(
            outcome.stage.is_none(),
            "a chain with no stages reports no provenance, not an empty trail"
        );
        let terminal_chain = FallbackCompactor::builder()
            .terminal(&TruncatingCompactor::new())
            .build();
        assert_eq!(
            terminal_chain.stage_names(),
            vec!["TruncatingCompactor"],
            "terminal() appends the capture stage"
        );
    }

    #[tokio::test]
    async fn a_custom_chain_stops_at_its_first_success() {
        let custom = ScriptedCompactor::new(vec![Scripted::ShrinkToLast]);
        let chain = FallbackCompactor::builder()
            .stage("custom", Arc::clone(&custom) as Arc<dyn ContextCompactor>)
            .terminal(&TruncatingCompactor::new())
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 40_000, context_for(&conversation()))
            .await;
        assert!(outcome.success);
        assert_eq!(
            outcome.messages.len(),
            1,
            "the custom stage's outcome is the chain's outcome"
        );
        let report = chain.last_report().expect("the run stored a report");
        assert_eq!(report.stages.len(), 1, "the terminal never ran");
    }

    #[tokio::test]
    async fn the_chain_compacts_through_the_manager_without_overflow() {
        let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
        let manager = ContextManager::new(Arc::new(chain)).with_context_window(280);
        let messages = conversation();
        let result = manager.ensure_context_fits(messages, 3).await;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                assert!(
                    outcome.messages.len() < 20,
                    "the chain compacted through the manager"
                );
            }
            other => {
                panic!("an over-window conversation compacts instead of overflowing: {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn an_over_target_landing_declines_to_the_next_stage() {
        let first = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(150)]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(40)]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let messages = conversation();
        let outcome = chain
            .compact(messages, 120, context_for(&conversation()))
            .await;
        assert_eq!(
            second.call_count(),
            1,
            "a stage that reduced but landed above the target cannot carry the \
             pass — the chain must fall through to the next stage"
        );
        assert_eq!(
            outcome.messages.len(),
            1,
            "the fitting stage's output is the pass's output"
        );
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.winning_stage,
            Some(1),
            "the winner is the stage that landed within the target"
        );
        assert_eq!(
            report.stages[0].error.as_deref(),
            Some(OVER_TARGET),
            "the decline names the landing, not a failure: {:?}",
            report.stages[0]
        );
    }

    #[tokio::test]
    async fn a_chain_whose_stages_never_fit_rides_out_the_last_outcome() {
        let first = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(150)]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(180)]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        let outcome = chain
            .compact(conversation(), 120, context_for(&conversation()))
            .await;
        assert!(
            outcome.success,
            "no stage errored, so the chain does not manufacture a failure"
        );
        assert_eq!(
            outcome.messages.len(),
            1,
            "the last non-failing outcome rides out for the caller's own \
             classification"
        );
        let trail = outcome
            .stage
            .as_deref()
            .expect("a ride-out outcome still names the stages that decided it");
        assert!(
            trail.contains("first: stage landed above the compaction target")
                && trail.contains("second: stage landed above the compaction target"),
            "the trail names every stage with its decline reason: {trail}"
        );
        assert!(
            trail.ends_with("no stage landed within the compaction target"),
            "a pass no stage fit states that outright: {trail}"
        );
    }

    #[tokio::test]
    async fn a_winning_stage_never_lands_above_the_target() {
        let mut winners = 0_usize;
        for size in [40_u64, 90, 150, 400] {
            let first = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(size)]);
            let second = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(30)]);
            let chain = FallbackCompactor::builder()
                .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
                .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
                .build();
            let outcome = chain
                .compact(conversation(), 120, context_for(&conversation()))
                .await;
            let report = chain.last_report().expect("a report is stored per run");
            if let Some(winner) = report.winning_stage {
                winners = winners.saturating_add(1);
                let measured = HeuristicTokenCounter.count(&outcome.messages);
                assert!(
                    measured <= 120,
                    "a winning stage's landing measures within the target \
                     (size {size}, winner {winner}, measured {measured})"
                );
            }
        }
        assert!(
            winners >= 2,
            "the grid produced winners to check ({winners}) — a chain that stopped \
             selecting any would make the landing assert vacuous"
        );
    }

    #[tokio::test]
    async fn a_stage_that_neither_reduced_nor_fit_records_no_reduction() {
        let first = ScriptedCompactor::new(vec![Scripted::Unchanged]);
        let second = ScriptedCompactor::new(vec![Scripted::ShrinkToTokens(30)]);
        let chain = FallbackCompactor::builder()
            .stage("first", Arc::clone(&first) as Arc<dyn ContextCompactor>)
            .stage("second", Arc::clone(&second) as Arc<dyn ContextCompactor>)
            .build();
        chain
            .compact(conversation(), 120, context_for(&conversation()))
            .await;
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.stages[0].error.as_deref(),
            Some(NO_REDUCTION),
            "the missing reduction is the more fundamental finding — an \
             unchanged stage records it even when its landing is also over \
             the target"
        );
    }
}
