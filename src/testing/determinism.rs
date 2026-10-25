//! Determinism test fixtures — a stepping clock and a capturing observer.
//!
//! The engine's reproducibility seams ([`Clock`], [`IdGen`], and the retry
//! rng — see [`crate::determinism`]) make its reported artifacts functions
//! of the installed implementations. These fixtures are how a test spends
//! that property: [`StepClock`] turns every duration the engine measures
//! into an exact multiple of a chosen step, and [`CapturingObserver`]
//! records every lifecycle callback into a comparable log, so two engines
//! built from the same seeds can be asserted equal event for event.
//!
//! Install them the same way a host installs any pinned implementation:
//! `LoopManagers::new().with_clock(Arc::new(StepClock::new(...)))` and
//! `loop.register_observer(Arc::new(CapturingObserver::new()))`.
//!
//! [`Clock`]: crate::determinism::Clock
//! [`IdGen`]: crate::determinism::IdGen

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::determinism::Clock;
use crate::observer::LoopObserver;
use crate::observer::{
    AttemptResetContext, BudgetWarnContext, CompactedContext, CompactionFailedContext,
    ConvergenceDetectedContext, FallbackContext, GateDecisionContext, LoopDetectedContext,
    ModelSwitchedContext, PreCompactionContext, ResponseContext, RunEndContext, RunStartContext,
    StreamContext, StreamFailureContext, TextDeltaContext, ThinkingDeltaContext,
    ToolCallReceivedContext, ToolPostContext, ToolPreContext, ToolSurfaceContext,
    TransportFallbackContext, TurnEndContext, TurnStartContext,
};

/// A [`Clock`] whose monotonic half advances a fixed step per read.
///
/// Deterministic timing without sleeps: every `monotonic()` call returns
/// the base instant advanced by one step per prior read, so any duration
/// the engine measures through the seam is an exact multiple of the step —
/// the property duration pins assert on. The wall half is frozen at the
/// Unix epoch; the engine's one wall read stamps permission-gate records,
/// so every gate decision a run emits under this clock carries a zero
/// timestamp. Install through
/// [`LoopManagers::with_clock`](crate::managers::LoopManagers::with_clock).
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use loopctl::determinism::Clock;
/// use loopctl::testing::StepClock;
///
/// let clock = StepClock::new(Duration::from_millis(5));
/// let first = clock.monotonic();
/// let second = clock.monotonic();
/// assert_eq!(second - first, Duration::from_millis(5));
/// assert_eq!(clock.elapsed_since(second), Duration::from_millis(5));
/// ```
#[derive(Debug)]
pub struct StepClock {
    /// The zeroth reading, captured once at construction.
    ///
    /// Every monotonic read returns this instant advanced by `step` times
    /// the number of prior reads, so the sequence is reproducible from a
    /// single captured base — no re-reading the real clock mid-test.
    base: Instant,

    /// The fixed advance per monotonic read.
    ///
    /// The quantum of every duration the engine can measure through the
    /// seam: each elapsed span is an exact multiple of this step, which
    /// the duration pins assert on.
    step: Duration,

    /// Reads served so far.
    ///
    /// Counts monotonic calls to derive the next reading's multiplier;
    /// atomic because the engine may consult the clock from tool futures.
    calls: AtomicUsize,
}

impl StepClock {
    /// Freeze a clock advancing `step` per read from now.
    ///
    /// The base is captured from the real clock once, here — after
    /// construction the clock is fully deterministic and never touches
    /// the system again.
    #[must_use]
    pub fn new(step: Duration) -> Self {
        Self {
            base: Instant::now(),
            step,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Clock for StepClock {
    fn now(&self) -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH
    }

    fn monotonic(&self) -> Instant {
        let reads = u32::try_from(self.calls.fetch_add(1, Ordering::SeqCst)).unwrap_or(0);
        let offset = self.step.checked_mul(reads).unwrap_or(Duration::MAX);
        self.base.checked_add(offset).unwrap_or(self.base)
    }
}

/// One observer callback's engine-sourced fields, projected for comparison.
///
/// The comparable form [`CapturingObserver`] records: one variant per
/// [`LoopObserver`] callback, each carrying that callback's
/// engine-sourced fields — ids, durations, token counts, stop reasons,
/// model names, error text. Two logs captured from two engines built from
/// the same seeds can be compared with `==` event for event, which is what
/// a same-seed equality pin (or a replay differencing suite) asserts.
///
/// Three context fields have no equality and are deliberately not carried:
/// a turn-end's history snapshot and a compaction's deep telemetry are
/// mirrors of engine state rather than seam artifacts (their scalar
/// summaries are carried instead — the compaction's pass statistics
/// reduce to `telemetry_duration`, the duration the clock seam produced),
/// and a stream failure's typed error is carried as its rendered text.
/// The enum is `#[non_exhaustive]`: a callback added to the trait widens
/// it in a minor release.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapturedEvent {
    /// A run began, carrying the session identifier.
    ///
    /// The first event of every `run()` call; a same-seed pin reads it to
    /// prove both engines minted the same first id from the seam.
    RunStart {
        /// The session's identifier, minted by the id seam.
        ///
        /// Stable across every `run()` call on the same loop, so two
        /// run-start events from one engine carry the same value — the
        /// field that ties a whole captured stream to one session.
        session_id: Uuid,
    },

    /// A run ended, carrying its outcome, duration, and usage totals.
    ///
    /// The last event of a `run()` call; its duration and totals are the
    /// run-level artifacts a pinned clock and a scripted backend leave
    /// fully determined.
    RunEnd {
        /// Whether the run exited normally.
        ///
        /// `false` on error or cancellation; pairs with `error` to tell
        /// the two apart from a clean exit.
        success: bool,

        /// The run's failure text, when it failed.
        ///
        /// `None` on success; a cancellation-shaped failure carries its
        /// own diagnostic here, so the rendered text stays comparable.
        error: Option<String>,

        /// How many turns the run completed.
        ///
        /// Counts only turns that finished; an in-flight turn at a fatal
        /// error is not included, so the figure matches the run record.
        total_turns: usize,

        /// The run's duration in whole milliseconds.
        ///
        /// Measured through the clock seam from run start to run end, so
        /// a pinned clock makes it exactly zero or an exact step multiple.
        duration_ms: u64,

        /// The source of the run's final context-size figure.
        ///
        /// `Some` when the engine had a figure in force at run end —
        /// anchored on a usage report or flying on the heuristic — and
        /// `None` only on host-built synthetic events.
        figure_source: Option<crate::observer::FigureSource>,

        /// How many completed turns old the figure's anchor was.
        ///
        /// `Some(age)` only when `figure_source` is anchored; carries the
        /// staleness the engine reported for its last real number.
        figure_anchor_age_turns: Option<usize>,

        /// The user input of a run that died on the context window.
        ///
        /// `Some(input)` only on the overflow family of endings, so a
        /// differencing host can offer resubmission; `None` otherwise.
        unfinished_input: Option<String>,

        /// The run's usage-fed estimate-calibration ratio, in thousandths.
        ///
        /// `None` when no usable usage report ever arrived — no ground
        /// truth, not a ratio of one — so absence stays comparable.
        calibration_ratio_bp: Option<u64>,

        /// Total input tokens the run's turns consumed.
        ///
        /// Saturating sum of the per-turn figures the provider reported;
        /// zero on a run whose backend never reported usage.
        input_tokens: u64,

        /// Total output tokens the run's turns produced.
        ///
        /// Saturating sum of the per-turn figures, matching
        /// `input_tokens` in provenance; the reasoning share is a subset
        /// of it, never additional.
        output_tokens: u64,

        /// Total input tokens served from the provider's cache.
        ///
        /// Saturating sum of the per-turn cached shares; zero means no
        /// turn reported a cached share — unreported, not zero-known.
        cached_input_tokens: u64,

        /// Total tokens written to the provider's cache.
        ///
        /// Saturating sum of the per-turn cache-write spends, with the
        /// same zero-means-unreported convention as the cached share.
        cache_write_tokens: u64,

        /// Total reasoning tokens inside the run's output.
        ///
        /// A subset of `output_tokens`; zero means no turn reported a
        /// reasoning share.
        reasoning_tokens: u64,
    },

    /// A turn began, carrying its index and driving query.
    ///
    /// One per model call the run makes; the query is the text the engine
    /// used for memory retrieval and observer tagging on this turn.
    TurnStart {
        /// The turn's 0-indexed number.
        ///
        /// Monotonically increasing within a run; matches the value the
        /// turn's other events carry, so a diff can pair them.
        turn: usize,

        /// The user query that initiated the turn.
        ///
        /// The full text of the latest user message before the turn
        /// began — script-determined, so two same-seed engines must
        /// agree on it.
        query: String,
    },

    /// A turn ended, carrying its outcome, duration, and token figures.
    ///
    /// Fired once per turn from both ending phases — after the model
    /// response and after tool dispatch — so a tool turn and a text turn
    /// each produce exactly one.
    TurnEnd {
        /// The turn's 0-indexed number.
        ///
        /// Matches the value the turn's start event carried; the pair is
        /// what a differencing consumer keys its rows by.
        turn: usize,

        /// Whether the turn completed successfully.
        ///
        /// `false` on a turn-level failure; the `error` field tells the
        /// story when it is.
        success: bool,

        /// The turn's failure text, when it failed.
        ///
        /// `None` on success; rendered text, so the value stays
        /// comparable across engines.
        error: Option<String>,

        /// The turn's duration in whole milliseconds.
        ///
        /// Measured through the clock seam from the turn's start read to
        /// its end read, so a pinned clock makes it exactly zero or an
        /// exact step multiple.
        duration_ms: u64,

        /// Input tokens the turn consumed.
        ///
        /// The provider-reported prompt figure for the turn's request;
        /// zero when the backend reported no usage.
        input_tokens: u64,

        /// Output tokens the turn produced.
        ///
        /// The provider-reported completion figure, including any
        /// intermediate tool-call rounds the turn made.
        output_tokens: u64,

        /// Why the model stopped producing output for this turn.
        ///
        /// `MaxTokens` marks a truncated answer even on a successful
        /// turn; a turn that failed before the model finished carries
        /// the `EndTurn` default.
        stop_reason: crate::stream::StreamStopReason,

        /// The engine's context-size estimate as the turn ended.
        ///
        /// The same forward-looking figure the machine reports at rest,
        /// present even when the provider reports no usage — the number
        /// that drove the run's window decisions.
        context_tokens: u64,

        /// The compaction denominator the run measured against.
        ///
        /// The resolved context window, or `None` when the window policy
        /// is disabled and there is no denominator to report.
        context_window: Option<u64>,

        /// The turn's cached input share.
        ///
        /// Verbatim from the provider's usage report; zero means
        /// unreported, per the crate-wide convention.
        cached_input_tokens: u64,

        /// The turn's cache-write spend.
        ///
        /// Verbatim from the usage report, with the same
        /// zero-means-unreported convention as the cached share.
        cache_write_tokens: u64,

        /// The reasoning share inside the turn's output.
        ///
        /// A subset of `output_tokens`; zero means the provider reported
        /// no reasoning share for the turn.
        reasoning_tokens: u64,
    },

    /// A streaming call succeeded, carrying its model and token figures.
    ///
    /// One per completed provider request; the model name is the field a
    /// fallback pin reads to see which model actually served the turn.
    StreamSuccess {
        /// The turn the streamed response belongs to.
        ///
        /// Matches the value the surrounding turn-start/end events
        /// carry, so per-request rows pair with per-turn rows.
        turn: usize,

        /// The model that served the request.
        ///
        /// May differ from the session default when model fallback
        /// occurred — the field a fallback pin reads.
        model: String,

        /// Input tokens consumed by the request.
        ///
        /// The cache-aware prompt total, including any share served from
        /// the provider's prompt cache.
        input_tokens: u64,

        /// Output tokens generated by the response.
        ///
        /// The completion figure, including any reasoning share reported
        /// inside it.
        output_tokens: u64,

        /// The request's cached input share.
        ///
        /// Zero means the provider reported no cached share —
        /// unreported, not zero-known.
        cached_input_tokens: u64,

        /// The request's cache-write spend.
        ///
        /// Zero means unreported, with the same convention as the cached
        /// share.
        cache_write_tokens: u64,

        /// The reasoning share inside the output.
        ///
        /// A subset of `output_tokens`; zero means the provider reported
        /// no reasoning share for this request.
        reasoning_tokens: u64,
    },

    /// A streaming call failed, carrying its model and rendered error.
    ///
    /// Fired on transport-level failures before any retry or fallback
    /// decision is applied; the engine may still recover the turn
    /// afterwards.
    StreamFailure {
        /// The turn the failed request belonged to.
        ///
        /// Matches the value the turn's other events carry, so failure
        /// rows pair with the turn they interrupted.
        turn: usize,

        /// The model the failed request targeted.
        ///
        /// The model identifier the request carried, which a fallback
        /// flow then changes for the retry.
        model: String,

        /// The failure's rendered text.
        ///
        /// The typed error's `Display` form; the type itself has no
        /// equality, so the log carries the rendered text a consumer
        /// would have shown.
        error: String,
    },

    /// A model response completed, carrying its text and usage.
    ///
    /// The assembled assistant text of the turn, delivered once the
    /// stream ended — the committed view, as opposed to the per-chunk
    /// delta events.
    Response {
        /// The turn the response belongs to.
        ///
        /// Matches the value the turn's start/end events carry.
        turn: usize,

        /// The model's text response.
        ///
        /// Concatenated text content; tool-call content is excluded and
        /// arrives through the tool events instead.
        text: String,

        /// The turn's token usage, when the provider reported one.
        ///
        /// `None` when the backend reports no usage — absence carried
        /// as absence, so backends stay distinguishable.
        usage: Option<crate::stream::Usage>,
    },

    /// An incremental text chunk arrived.
    ///
    /// One per streamed text delta, in arrival order; concatenating the
    /// chunks of a turn reconstructs its live text.
    TextDelta {
        /// The turn the chunk belongs to.
        ///
        /// Same value the surrounding response and turn-end events
        /// carry, so buffered chunks key correctly.
        turn: usize,

        /// The incremental text chunk.
        ///
        /// Concatenate across calls, in arrival order, to reconstruct
        /// the per-turn text.
        delta: String,
    },

    /// An incremental reasoning chunk arrived.
    ///
    /// The thinking-stream counterpart of the text delta; reasoning is
    /// distinct from visible assistant text and never concatenates with
    /// it.
    ThinkingDelta {
        /// The turn the chunk belongs to.
        ///
        /// Same value the text deltas of the same stream carry, so an
        /// observer can interleave the two lanes correctly.
        turn: usize,

        /// The incremental reasoning chunk.
        ///
        /// Empty when the delta carried something other than displayable
        /// reasoning — a redacted block's opaque payload or a signature.
        delta: String,
    },

    /// A retried stream attempt discarded the failed attempt's events.
    ///
    /// Fired exactly once before the first event of each retried attempt
    /// — the cue a delta-buffering consumer resets on.
    AttemptReset {
        /// The turn the retry belongs to.
        ///
        /// Matches the value the discarded attempt's delta events
        /// carried.
        turn: usize,

        /// The attempt the stream is entering, 1-indexed.
        ///
        /// The first attempt never fires a reset; the first retry
        /// carries `2`, so the value counts the ladder's rungs.
        attempt: usize,
    },

    /// A permission gate decided about a tool call.
    ///
    /// One per gated dispatch, when the deciding record becomes final;
    /// dispatches no gate consulted produce no event.
    GateDecision {
        /// The turn whose dispatch the gate decided about.
        ///
        /// Matches the value the surrounding tool pre/post events carry,
        /// so the decision rows pair with the dispatch.
        turn: usize,

        /// The model-assigned call id the decision is about.
        ///
        /// Pairs the decision with the same call's tool-pre and
        /// tool-post events, where the turn and tool pair alone would be
        /// ambiguous.
        call_id: String,

        /// The deciding record: verdict, argument digest, provenance.
        ///
        /// Carried whole — the record is itself comparable, and its
        /// fields are what an audit trail rows on.
        decision: crate::tool::permission::GateDecision,
    },

    /// A budget line's soft threshold was crossed.
    ///
    /// Fired once per dimension per run, at the first pre-request check
    /// past the derived soft line — a warning, not the hard refusal.
    BudgetWarn {
        /// The budget line whose soft threshold was crossed.
        ///
        /// The one vocabulary for "which budget" shared with the typed
        /// exhaustion error and the gate-decision reasons.
        dimension: crate::budget::BudgetDimension,

        /// The spend at the crossing.
        ///
        /// The figure the check observed, in the dimension's own unit —
        /// tokens, turns, or wall-clock as the line defines it.
        spent: u64,

        /// The hard limit the soft line derives from.
        ///
        /// The operator's configured ceiling, not the derived soft line,
        /// so a reader can compute the crossing point itself.
        limit: u64,
    },

    /// A tool call was accumulated, before dispatch.
    ///
    /// Fired once per call, after the stream assembles it and before any
    /// dispatch begins — strictly earlier than the tool-pre event.
    ToolCallReceived {
        /// The turn the call belongs to.
        ///
        /// Matches the value the later tool-pre/post events carry for
        /// the same call.
        turn: usize,

        /// The tool's registered name.
        ///
        /// The name the model asked for, as the registry will resolve
        /// it; same value tool-pre repeats.
        tool: String,

        /// The model-issued call id.
        ///
        /// Unique per invocation; the field that pairs a call with its
        /// result when the same tool is called twice in one turn.
        call_id: String,

        /// The JSON input the model supplied for the call.
        ///
        /// The full input object as accumulated from the stream — the
        /// one event that carries it.
        input: serde_json::Value,
    },

    /// A tool dispatch is about to run.
    ///
    /// The last event before the tool executes; it does not repeat on
    /// recovery retries of the same call.
    ToolPre {
        /// The turn the dispatch belongs to.
        ///
        /// Matches the value the call's received and post events carry.
        turn: usize,

        /// The tool's registered name.
        ///
        /// Same value the received event carried; the dispatch resolves
        /// it against the registry here.
        tool: String,

        /// The model-issued call id the dispatch answers.
        ///
        /// Pairs with the tool-post event's id — including on same-tool
        /// retries and parallel calls, where names alone are ambiguous.
        tool_call_id: String,
    },

    /// A tool dispatch finished, carrying its outcome and duration.
    ///
    /// One per completed dispatch, including error results; the duration
    /// is the seam artifact a pinned clock controls on every dispatch
    /// path.
    ToolPost {
        /// The model-issued call id this result answers.
        ///
        /// Mirrors the tool-pre id, so pre/post pairs match exactly even
        /// across same-tool repeats.
        tool_call_id: String,

        /// The turn the dispatch belongs to.
        ///
        /// Matches the value the pairing tool-pre event carried.
        turn: usize,

        /// The tool's registered name.
        ///
        /// Same value the pairing events carried; carried again so a
        /// post row stands alone.
        tool: String,

        /// Deterministic hash of the tool's output, when available.
        ///
        /// The loop detector's identity for the result — comparing it
        /// across engines proves identical outputs without carrying the
        /// output itself.
        result_hash: Option<u64>,

        /// Whether the tool returned an error.
        ///
        /// `true` for an error response rather than a successful output;
        /// a soft error still completes the dispatch.
        is_error: bool,

        /// The dispatch's execution duration.
        ///
        /// Measured through the clock seam on every dispatch path,
        /// including the middleware pipeline, so a pinned clock makes it
        /// exactly zero or an exact step multiple.
        duration: Duration,

        /// The result's advisory rendering hint.
        ///
        /// `None` when the call was blocked before execution or the
        /// tool set no hint; presentation-only, never read by loop code.
        display_hint: Option<crate::tool::DisplayHint>,
    },

    /// A compaction pass is about to run.
    ///
    /// The pass-start notification; the matching success event fires only
    /// when the pass actually compacted, so a vetoed or no-action pass
    /// shows this event alone.
    PreCompaction {
        /// Why the pass is starting.
        ///
        /// Threshold, emergency, or manual — the same value the pass's
        /// end event carries, so a start/end pair matches without
        /// guessing.
        reason: crate::compact::CompactReason,

        /// The turn the pass runs in.
        ///
        /// The engine's current turn number, matching the surrounding
        /// turn events.
        turn: usize,

        /// The payload estimate the pass runs against.
        ///
        /// History plus per-request overhead — the same figure the
        /// post-event's `tokens_before` reports.
        tokens_before: u64,

        /// The context window the pass compacts toward.
        ///
        /// The configured manager window, so an observer can compute
        /// utilization at pass start without holding the configuration.
        context_window: u64,

        /// The history length at pass start.
        ///
        /// The input size before any reduction; the evicted count the
        /// post-event reports is a fraction of it.
        message_count: usize,

        /// The session's identifier, minted by the id seam.
        ///
        /// Same value every run-start event carried — the field that
        /// ties a compaction row to its session.
        session_id: Uuid,
    },

    /// A compaction pass completed, carrying its token delta and duration.
    ///
    /// Fired only when compaction actually occurred — not on no-action
    /// passes — with the before/after estimates and the pass's
    /// seam-sourced duration.
    Compaction {
        /// The payload estimate before the compactor ran.
        ///
        /// Reconstructed from after-plus-saved; the same figure the
        /// pass-start event reported.
        tokens_before: u64,

        /// The payload estimate after the compactor ran.
        ///
        /// The compacted history plus the same overhead; excludes
        /// per-request transients, which the next turn regenerates.
        tokens_after: u64,

        /// The net reduction the pass achieved.
        ///
        /// `tokens_before - tokens_after`; the machine's no-progress
        /// guard reads the same relationship.
        tokens_saved: u64,

        /// Why the pass ran.
        ///
        /// The same value the pass-start event carried, so the pair
        /// matches.
        reason: crate::compact::CompactReason,

        /// How many messages the pass removed from the feed.
        ///
        /// The size of the evicted slice handed to the demotion sink —
        /// message-granular, in conversation order.
        evicted_messages: usize,

        /// The pass's duration, measured through the clock seam.
        ///
        /// The one figure of the compactor's telemetry the log carries:
        /// the deep statistics have no equality, and the duration is the
        /// seam artifact a pinned clock controls.
        telemetry_duration: Duration,

        /// The compactor's stage trail, when it reported one.
        ///
        /// One line per internal stage the pass declined through plus
        /// the stage that won; `None` when the compactor reports no
        /// stages.
        stage: Option<String>,
    },

    /// A compaction pass was attempted and failed.
    ///
    /// The pass-death or deferral notification; the success event never
    /// follows this one, and the run has already failed by the time the
    /// death variants arrive.
    CompactionFailed {
        /// Why the failed pass ran.
        ///
        /// The same value the pass-start event carried, so the pair
        /// matches without guessing.
        reason: crate::compact::CompactReason,

        /// The turn the failed pass ran in.
        ///
        /// The engine's current turn number, matching the surrounding
        /// turn events.
        turn: usize,

        /// The payload estimate the pass ran against.
        ///
        /// The same figure the pass-start event reported as
        /// `tokens_before`.
        tokens_before: u64,

        /// The window the pass was compacting toward.
        ///
        /// The configured manager window, so an observer can compute
        /// utilization at the failure.
        context_window: u64,

        /// The compactor's error text, the doesn't-fit marker, or the
        /// deferral notice.
        ///
        /// `None` only for the succeeded-but-did-not-fit death; the
        /// non-fatal deferral carries its notice here, so a reader tells
        /// the three modes apart.
        error: Option<String>,
    },

    /// A model fallback occurred.
    ///
    /// Reports the policy-driven switch to a fallback model; the model
    /// the next request carries is the `to` value.
    Fallback {
        /// The model that failed.
        ///
        /// The identifier whose error tripped the breaker — the model
        /// the failed request targeted.
        from: String,

        /// The replacement model for subsequent requests.
        ///
        /// The override the next request carries; a same-seed pin sees
        /// identical values on both engines.
        to: String,
    },

    /// A turn was served by the non-streaming transport fallback.
    ///
    /// The degradation report for a turn the handler's last-chance
    /// `create_message` rescued; the turn proceeds normally afterwards.
    TransportFallback {
        /// The turn the fallback served.
        ///
        /// Matches the value the turn's other events carry; the event
        /// fires before that turn's turn-end.
        turn: usize,

        /// Why the fallback response's model stopped.
        ///
        /// The stop reason of the non-streaming response — the same
        /// value the turn's record holds.
        stop_reason: crate::stream::StreamStopReason,
    },

    /// The active model was hot-swapped at runtime.
    ///
    /// Fired by an explicit switch after the client accepted the new
    /// model; distinct from the policy-driven fallback event.
    ModelSwitched {
        /// The model before the switch.
        ///
        /// The identifier the session used up to the hot-swap point.
        from: String,

        /// The model after the switch.
        ///
        /// The identifier that will handle subsequent requests.
        to: String,
    },

    /// The registry's tool surface changed between turns.
    ///
    /// Fired at the turn seam when the engine observes the registry's
    /// tool set moved since its previous observation.
    ToolSurfaceChanged {
        /// The registry's mutation counter at the observation.
        ///
        /// Monotonic across the registry's lifetime; two reports compare
        /// by it even when turns interleave.
        version: u64,

        /// Names that entered the advertised surface.
        ///
        /// Sorted set difference; empty when the change was removals or
        /// overwrites only.
        added: Vec<String>,

        /// Names that left the advertised surface.
        ///
        /// Sorted; empty when the change was additions or overwrites
        /// only.
        removed: Vec<String>,

        /// Names whose advertised schema changed in place.
        ///
        /// Sorted; an overwrite under the same name enters here — the
        /// name never left the surface, only its schema moved.
        changed: Vec<String>,
    },

    /// A repeating tool pattern was detected.
    ///
    /// Fired when the same tool operation produced the same result
    /// repeatedly, past the configured threshold.
    LoopDetected {
        /// The repeating pattern's description.
        ///
        /// Human-readable summary of the tool operations detected as
        /// repeating — rendered text, so it stays comparable.
        pattern: String,

        /// How many times the pattern was observed.
        ///
        /// Counts consecutive repetitions of the same operation with
        /// the same result hash.
        repetitions: usize,
    },

    /// Response convergence was detected.
    ///
    /// Fired when consecutive assistant responses became semantically
    /// similar per the convergence policy's configuration.
    ConvergenceDetected {
        /// The configured action to take.
        ///
        /// The policy's verdict — `"stop"`, `"warn"`, or `"compact"` —
        /// as determined by the detection configuration.
        action: String,
    },
}

/// An observer that records every callback into a comparable log.
///
/// Implements [`LoopObserver`] by projecting each callback's engine-sourced
/// fields into a [`CapturedEvent`] and appending it to a shared buffer, so
/// the whole stream of a run — or of two runs from two same-seed engines —
/// can be read back with [`snapshot`](Self::snapshot) and compared with
/// `==`. Register one per engine the same way any observer is registered;
/// the buffer grows unboundedly, which is the point: a capture must not
/// drop the event that would have exposed a divergence.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use loopctl::observer::{LoopObserver, RunStartContext};
/// use loopctl::testing::CapturingObserver;
///
/// let observer = Arc::new(CapturingObserver::new());
/// let session_id = uuid::Uuid::new_v4();
/// observer.on_run_start(&RunStartContext { session_id });
///
/// let snapshot = observer.snapshot();
/// assert_eq!(snapshot.len(), 1);
/// assert!(matches!(
///     snapshot.first(),
///     Some(loopctl::testing::CapturedEvent::RunStart { .. })
/// ));
/// ```
#[derive(Debug, Default)]
pub struct CapturingObserver {
    /// The captured events, in arrival order.
    ///
    /// Shared behind a mutex because the engine may fire callbacks from
    /// tool futures; the lock recovers from poison, since a panicked
    /// sibling must not lose the events already recorded.
    events: Mutex<Vec<CapturedEvent>>,
}

impl CapturingObserver {
    /// Create an observer with an empty log.
    ///
    /// The single construction path; register the returned observer (an
    /// `Arc` of it) with each engine whose stream is being captured.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read back the captured events, in arrival order.
    ///
    /// Clones the buffer, so a test can assert on the snapshot while the
    /// engine keeps appending; compare two snapshots with `==` to assert
    /// two engines produced identical streams.
    pub fn snapshot(&self) -> Vec<CapturedEvent> {
        crate::error::recover_guard(self.events.lock()).clone()
    }

    /// Append one captured event to the log.
    ///
    /// The single write path every callback funnels through; keeping it
    /// in one place is what makes the observer's own behavior trivially
    /// uniform across callbacks.
    fn record(&self, event: CapturedEvent) {
        crate::error::recover_guard(self.events.lock()).push(event);
    }
}

impl LoopObserver for CapturingObserver {
    fn name(&self) -> &'static str {
        "capturing-observer"
    }

    fn on_run_start(&self, ctx: &RunStartContext) {
        self.record(CapturedEvent::RunStart {
            session_id: ctx.session_id,
        });
    }

    fn on_run_end(&self, ctx: &RunEndContext) {
        self.record(CapturedEvent::RunEnd {
            success: ctx.success,
            error: ctx.error.clone(),
            total_turns: ctx.total_turns,
            duration_ms: ctx.duration_ms,
            figure_source: ctx.context_figure.as_ref().map(|figure| figure.source),
            figure_anchor_age_turns: ctx
                .context_figure
                .as_ref()
                .and_then(|figure| figure.anchor_age_turns),
            unfinished_input: ctx.unfinished_input.clone(),
            calibration_ratio_bp: ctx.calibration_ratio_bp,
            input_tokens: ctx.input_tokens,
            output_tokens: ctx.output_tokens,
            cached_input_tokens: ctx.cached_input_tokens,
            cache_write_tokens: ctx.cache_write_tokens,
            reasoning_tokens: ctx.reasoning_tokens,
        });
    }

    fn on_turn_start(&self, ctx: &TurnStartContext) {
        self.record(CapturedEvent::TurnStart {
            turn: ctx.turn,
            query: ctx.query.clone(),
        });
    }

    fn on_turn_end(&self, ctx: &TurnEndContext) {
        self.record(CapturedEvent::TurnEnd {
            turn: ctx.turn,
            success: ctx.success,
            error: ctx.error.clone(),
            duration_ms: ctx.duration_ms,
            input_tokens: ctx.input_tokens,
            output_tokens: ctx.output_tokens,
            stop_reason: ctx.stop_reason,
            context_tokens: ctx.context_tokens,
            context_window: ctx.context_window,
            cached_input_tokens: ctx.cached_input_tokens,
            cache_write_tokens: ctx.cache_write_tokens,
            reasoning_tokens: ctx.reasoning_tokens,
        });
    }

    fn on_stream_success(&self, ctx: &StreamContext) {
        self.record(CapturedEvent::StreamSuccess {
            turn: ctx.turn,
            model: ctx.model.clone(),
            input_tokens: ctx.input_tokens,
            output_tokens: ctx.output_tokens,
            cached_input_tokens: ctx.cached_input_tokens,
            cache_write_tokens: ctx.cache_write_tokens,
            reasoning_tokens: ctx.reasoning_tokens,
        });
    }

    fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        self.record(CapturedEvent::StreamFailure {
            turn: ctx.turn,
            model: ctx.model.clone(),
            error: ctx.error.to_string(),
        });
    }

    fn on_response(&self, ctx: &ResponseContext) {
        self.record(CapturedEvent::Response {
            turn: ctx.turn,
            text: ctx.text.clone(),
            usage: ctx.usage,
        });
    }

    fn on_text_delta(&self, ctx: &TextDeltaContext) {
        self.record(CapturedEvent::TextDelta {
            turn: ctx.turn,
            delta: ctx.delta.clone(),
        });
    }

    fn on_thinking_delta(&self, ctx: &ThinkingDeltaContext) {
        self.record(CapturedEvent::ThinkingDelta {
            turn: ctx.turn,
            delta: ctx.delta.clone(),
        });
    }

    fn on_attempt_reset(&self, ctx: &AttemptResetContext) {
        self.record(CapturedEvent::AttemptReset {
            turn: ctx.turn,
            attempt: ctx.attempt,
        });
    }

    fn on_gate_decision(&self, ctx: &GateDecisionContext) {
        self.record(CapturedEvent::GateDecision {
            turn: ctx.turn,
            call_id: ctx.call_id.clone(),
            decision: ctx.decision.clone(),
        });
    }

    fn on_budget_warn(&self, ctx: &BudgetWarnContext) {
        self.record(CapturedEvent::BudgetWarn {
            dimension: ctx.dimension,
            spent: ctx.spent,
            limit: ctx.limit,
        });
    }

    fn on_tool_call_received(&self, ctx: &ToolCallReceivedContext) {
        self.record(CapturedEvent::ToolCallReceived {
            turn: ctx.turn,
            tool: ctx.tool.clone(),
            call_id: ctx.call_id.clone(),
            input: ctx.input.clone(),
        });
    }

    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        self.record(CapturedEvent::ToolPre {
            turn: ctx.turn,
            tool: ctx.tool.clone(),
            tool_call_id: ctx.tool_call_id.clone(),
        });
    }

    fn on_tool_post(&self, ctx: &ToolPostContext) {
        self.record(CapturedEvent::ToolPost {
            tool_call_id: ctx.tool_call_id.clone(),
            turn: ctx.turn,
            tool: ctx.tool.clone(),
            result_hash: ctx.result_hash,
            is_error: ctx.is_error,
            duration: ctx.duration,
            display_hint: ctx.display_hint.clone(),
        });
    }

    fn on_pre_compaction(&self, ctx: &PreCompactionContext) {
        self.record(CapturedEvent::PreCompaction {
            reason: ctx.reason,
            turn: ctx.turn,
            tokens_before: ctx.tokens_before,
            context_window: ctx.context_window,
            message_count: ctx.message_count,
            session_id: ctx.session_id,
        });
    }

    fn on_compaction(&self, ctx: &CompactedContext) {
        self.record(CapturedEvent::Compaction {
            tokens_before: ctx.tokens_before,
            tokens_after: ctx.tokens_after,
            tokens_saved: ctx.tokens_saved,
            reason: ctx.reason,
            evicted_messages: ctx.evicted_messages,
            telemetry_duration: ctx.telemetry.duration,
            stage: ctx.stage.clone(),
        });
    }

    fn on_compaction_failed(&self, ctx: &CompactionFailedContext) {
        self.record(CapturedEvent::CompactionFailed {
            reason: ctx.reason,
            turn: ctx.turn,
            tokens_before: ctx.tokens_before,
            context_window: ctx.context_window,
            error: ctx.error.clone(),
        });
    }

    fn on_fallback(&self, ctx: &FallbackContext) {
        self.record(CapturedEvent::Fallback {
            from: ctx.from.clone(),
            to: ctx.to.clone(),
        });
    }

    fn on_transport_fallback(&self, ctx: &TransportFallbackContext) {
        self.record(CapturedEvent::TransportFallback {
            turn: ctx.turn,
            stop_reason: ctx.stop_reason,
        });
    }

    fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        self.record(CapturedEvent::ModelSwitched {
            from: ctx.from.clone(),
            to: ctx.to.clone(),
        });
    }

    fn on_tool_surface_changed(&self, ctx: &ToolSurfaceContext) {
        self.record(CapturedEvent::ToolSurfaceChanged {
            version: ctx.version,
            added: ctx.added.clone(),
            removed: ctx.removed.clone(),
            changed: ctx.changed.clone(),
        });
    }

    fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        self.record(CapturedEvent::LoopDetected {
            pattern: ctx.pattern.clone(),
            repetitions: ctx.repetitions,
        });
    }

    fn on_convergence_detected(&self, ctx: &ConvergenceDetectedContext) {
        self.record(CapturedEvent::ConvergenceDetected {
            action: ctx.action.clone(),
        });
    }
}
