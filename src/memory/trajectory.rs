//! Agent-run trajectories captured as serializable records.
//!
//! [`TrajectoryObserver`] is a [`LoopObserver`]
//! that listens to a run's lifecycle events and assembles them into one
//! [`TrajectoryRecord`] per run — every turn's query and response, every tool
//! call paired by `tool_call_id`, durations, and token sums. Records are kept
//! in memory ([`records`](TrajectoryObserver::records)) and, optionally,
//! appended to a JSON Lines file. The record is the substrate for
//! downstream experience extraction and is suitable for attachment to
//! bug reports.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::memory::trajectory::TrajectoryObserver;
//! use loopctl::observer::{LoopObserver, RunEndContext, RunStartContext};
//!
//! let observer = TrajectoryObserver::in_memory();
//!
//! // In a host application the observer is attached to the loop instead:
//! // BareLoop::new(client, registry, config).with_observer(Arc::new(observer))
//! observer.on_run_start(&RunStartContext {
//!     session_id: uuid::Uuid::new_v4(),
//! });
//! observer.on_run_end(&RunEndContext {
//!     success: true,
//!     error: None,
//!     total_turns: 0,
//!     duration_ms: 0,
//! });
//!
//! let records = observer.records();
//! assert_eq!(records.len(), 1);
//! assert_eq!(
//!     records[0].outcome,
//!     loopctl::memory::TrajectoryOutcome::Success
//! );
//! ```
//!
//! Capture is opt-in per host: `in_memory()` keeps finished records in
//! process for [`records`](TrajectoryObserver::records) to hand over;
//! [`writing_to`](TrajectoryObserver::writing_to) additionally appends
//! each record to a JSON Lines ledger on disk.
//!
//! # When to Use
//!
//! Use the observer when a host needs the whole run as data — experience
//! extraction, debugging artifacts, telemetry — but does not want capture
//! to influence the run. Hosts that need to *control* what happens use
//! hooks, which are a different contract.
//!
//! # Capture contract
//!
//! The observer is a pure listener: it never vetoes or fails a run.
//! Events that arrive without their expected predecessor are still
//! captured: a tool result whose dispatch was never observed (the observer
//! attached mid-run) is recorded, and a tool call still in flight when the
//! run ends is closed as `ok: false` with its duration measured to the
//! run end.
//!
//! # JSONL interchange schema
//!
//! Each finished record is one line: `serde_json::to_string` of a
//! [`TrajectoryRecord`], followed by `\n`, appended to
//! `<sink dir>/trajectory.jsonl`. The schema is an interchange contract:
//!
//! ```json
//! {"session_id":"…","run_id":"…","outcome":"partial","started_at":"2026-08-31T12:00:00Z",
//!  "duration_ms":4200,"total_turns":2,
//!  "token_summary":{"input_tokens":120,"output_tokens":340,
//!                   "cached_input_tokens":null,"cache_write_tokens":null,
//!                   "reasoning_tokens":null},
//!  "turns":[{"turn":0,"query":"fix the bug","response_text":"…",
//!            "tool_calls":[{"tool_call_id":"call_a","tool":"bash","ok":true,
//!                           "duration_ms":12}],
//!            "duration_ms":2100,"input_tokens":60,"output_tokens":170}]}
//! ```
//!
//! Rules external consumers may rely on: fields serialize in declaration
//! order; `outcome` is one of `success`, `failure`, `partial`
//! (`snake_case`);
//! unknown fields are tolerated when deserializing (forward compatibility);
//! `token_summary`'s detail fields are `null` when the provider did not
//! report them — `null` means *unreported*, never zero.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::recover_guard;
use crate::observer::{
    LoopObserver, ResponseContext, RunEndContext, RunStartContext, ToolPostContext, ToolPreContext,
    TurnEndContext, TurnStartContext,
};

/// Default bound on the response text captured per turn, in characters.
const DEFAULT_CAPTURE_LIMIT: usize = 2_000;

/// File name of the JSON Lines ledger inside the sink directory.
const LEDGER_FILE: &str = "trajectory.jsonl";

/// One run's full trajectory, captured by [`TrajectoryObserver`].
///
/// A session with N runs produces N records, correlated by `session_id` and
/// uniquely identified by `run_id`. The record is plain data — serializable,
/// cloneable, and independent of engine types — so it can be written to
/// disk, attached to a bug report, or consumed by downstream analysis
/// without reference to the loop that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    /// The session the run belonged to.
    ///
    /// Correlates all of a session's runs; stable across every `run()` call
    /// on the same loop.
    pub session_id: String,

    /// Unique identifier of this run.
    ///
    /// Minted by the observer when the run started; two records never share
    /// a `run_id`.
    pub run_id: String,

    /// How the run ended, classified from the run outcome and its tool work.
    ///
    /// The classification is deterministic: `Success` when the run ended
    /// cleanly, `Failure` when it ended in failure with no successful tool
    /// work, and `Partial` when failure followed real progress.
    pub outcome: TrajectoryOutcome,

    /// When the run started, RFC 3339 in UTC.
    ///
    /// Rendered from the system clock at capture time, whole-second
    /// resolution, `Z`-suffixed.
    pub started_at: String,

    /// Wall-clock duration of the whole run, in milliseconds.
    ///
    /// Carried from the run's own end report, not re-measured by the
    /// observer.
    pub duration_ms: u64,

    /// Number of turns the run executed.
    ///
    /// The engine's turn count for the run; the captured `turns` list may
    /// be shorter when the observer attached mid-run.
    pub total_turns: usize,

    /// Token totals over the run's turns, plus the provider's usage detail
    /// when it reported any.
    ///
    /// See [`TokenSummary`] for the unreported-versus-zero distinction on
    /// the detail fields.
    pub token_summary: TokenSummary,

    /// The run's turns, in execution order.
    ///
    /// Each entry is self-contained: the query, the truncated response,
    /// and the turn's tool calls.
    pub turns: Vec<TrajectoryTurn>,
}

/// How a run ended, in three grades.
///
/// Finer-grained than the run's `success: bool`: a run that performed
/// successful work and afterwards exceeded a terminal limit — for example,
/// completing its objective and then exhausting the turn budget during
/// cleanup — is [`Partial`](TrajectoryOutcome::Partial). The distinction
/// is material for downstream extraction and is not visible in a
/// boolean outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryOutcome {
    /// Completed cleanly.
    ///
    /// The run reported success, regardless of any failed tool calls
    /// along the way — the model recovered from them.
    Success,

    /// Terminal failure with no successful tool work.
    ///
    /// The run's own outcome was failure and nothing in the trajectory
    /// contradicts it: every observed tool call failed or was abandoned
    /// in flight.
    Failure,

    /// Real progress, imperfect ending: the run reported failure while at
    /// least one of its tool calls succeeded.
    Partial,
}

/// Token totals over a run's turns, plus usage detail when reported.
///
/// The sums aggregate every captured turn. The detail fields forward the
/// provider's per-run usage breakdown when present; `None` means the
/// provider did not report the figure. An unreported value is never
/// represented as a measured zero.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenSummary {
    /// Total input tokens across the run's turns.
    ///
    /// Saturating sum of the per-turn figures; a run with no captured
    /// turns reports zero.
    pub input_tokens: u64,

    /// Total output tokens across the run's turns.
    ///
    /// Saturating sum of the per-turn figures; a run with no captured
    /// turns reports zero.
    pub output_tokens: u64,

    /// Input tokens served from the provider's cache, if reported.
    ///
    /// `None` until the provider's usage detail reaches the observer —
    /// unreported is never collapsed to a measured zero.
    pub cached_input_tokens: Option<u64>,

    /// Tokens written to the provider's cache, if reported.
    ///
    /// Same `None`-means-unreported contract as
    /// [`cached_input_tokens`](Self::cached_input_tokens).
    pub cache_write_tokens: Option<u64>,

    /// Reasoning tokens the provider charged separately, if reported.
    ///
    /// Same `None`-means-unreported contract as
    /// [`cached_input_tokens`](Self::cached_input_tokens).
    pub reasoning_tokens: Option<u64>,
}

/// One turn of a captured run.
///
/// `query` carries the user-visible input of the turn. On continuation
/// turns after tool dispatch it carries the tool-result text, so each
/// turn is self-contained within the record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryTurn {
    /// Turn number, 0-indexed within the run.
    ///
    /// Matches the engine's turn indexing; a turn opened mid-run (its
    /// start event never observed) still carries the correct index.
    pub turn: usize,

    /// The query that initiated the turn.
    ///
    /// On continuation turns after tool dispatch this is the tool-result
    /// text, so each turn is self-contained within the record.
    pub query: String,

    /// The model's text response, truncated to the observer's capture limit.
    ///
    /// Truncation happens on character boundaries when the response is
    /// captured; the default limit is 2,000 characters.
    pub response_text: String,

    /// The turn's tool calls, in completion order.
    ///
    /// Pairing is by `tool_call_id`, so parallel and interleaved calls
    /// land on their own entries. Calls closed at run end (never
    /// completed) are appended afterwards, in unspecified order.
    pub tool_calls: Vec<TrajectoryToolCall>,

    /// Wall-clock duration of the turn, in milliseconds.
    ///
    /// Carried from the turn's end event, whenever it arrives: a late end
    /// patches the already-closed turn's figures; a turn that never
    /// received one (mid-run attach, run ending first) reports zero.
    pub duration_ms: u64,

    /// Input tokens the turn consumed.
    ///
    /// From the turn's end event; zero when the turn closed without one.
    pub input_tokens: u64,

    /// Output tokens the turn produced.
    ///
    /// From the turn's end event; zero when the turn closed without one.
    pub output_tokens: u64,
}

/// One tool call within a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryToolCall {
    /// The model-issued call id — the pairing key between dispatch and
    /// result, stable under parallel and same-tool-retry traffic.
    pub tool_call_id: String,

    /// The name the tool is registered under.
    ///
    /// Copied from the dispatch (or the result, when the dispatch was
    /// never observed), so a lazily opened slot still names its tool.
    pub tool: String,

    /// Whether the call succeeded.
    ///
    /// `false` also indicates a call that was still in flight when the
    /// run ended; such calls are recorded as unsuccessful rather than
    /// inferred to have succeeded.
    pub ok: bool,

    /// Wall-clock execution duration, in milliseconds.
    ///
    /// For a call closed unfinished at run end, the duration is counted
    /// from dispatch to run end.
    pub duration_ms: u64,
}

/// The run currently being assembled.
#[derive(Debug)]
struct RecordBuilder {
    /// Session correlation id, captured at run start.
    ///
    /// Rendered from the run-start event's session UUID.
    session_id: String,

    /// Freshly minted unique id for this run.
    ///
    /// Generated in [`RecordBuilder::new`]; never reused across runs.
    run_id: String,

    /// RFC 3339 render of the run's start instant.
    ///
    /// Frozen once, at construction, so the record's timestamp is the
    /// start regardless of when the run ends.
    started_at: String,

    /// Response-text cap, carried from the observer's configuration.
    ///
    /// Applied when a response is captured, so the builder never holds
    /// more than the configured limit.
    capture_limit: usize,

    /// Turns that received their end event, in execution order.
    ///
    /// Immutable once closed; converted from the in-flight builder at
    /// close time.
    turns: Vec<TrajectoryTurn>,

    /// The turn in flight, if any event of a turn has been seen.
    ///
    /// `None` between turns; swapped out on every turn end.
    current: Option<TurnBuilder>,

    /// Dispatches seen but not yet answered, keyed by `tool_call_id`.
    ///
    /// Drained at run end — anything still here is abandoned in flight.
    open_calls: HashMap<String, OpenCall>,
}

impl RecordBuilder {
    /// A builder for a run's record, opened at the run's start.
    fn new(session_id: String, capture_limit: usize) -> Self {
        Self {
            session_id,
            run_id: Uuid::new_v4().to_string(),
            started_at: rfc3339_now(),
            capture_limit,
            turns: Vec::new(),
            current: None,
            open_calls: HashMap::new(),
        }
    }

    /// The turn a within-turn event belongs to, opened on demand.
    ///
    /// Reuses the in-flight turn while its index matches; a mismatch (or
    /// a missing start event, as when the observer attached mid-run)
    /// closes the open turn and opens a turn whose query is empty.
    fn turn_mut(&mut self, turn: usize) -> &mut TurnBuilder {
        let stale = self.current.as_ref().is_some_and(|open| open.turn != turn);
        if stale {
            self.close_current();
        }
        self.current
            .get_or_insert_with(|| TurnBuilder::new(turn, String::new()))
    }

    /// Move the in-flight turn, if any, into the finished list.
    ///
    /// The builder is converted into its serializable shape here — the one
    /// place a turn becomes immutable.
    fn close_current(&mut self) {
        if let Some(turn) = self.current.take() {
            let response_text = truncate_chars(&turn.response_text, self.capture_limit);
            self.turns.push(TrajectoryTurn {
                turn: turn.turn,
                query: turn.query,
                response_text,
                tool_calls: turn.tool_calls,
                duration_ms: turn.duration_ms,
                input_tokens: turn.input_tokens,
                output_tokens: turn.output_tokens,
            });
        }
    }

    /// Close a still-open tool call as unfinished, timed to now.
    fn abandon_call(&mut self, tool_call_id: String, open: OpenCall) {
        let duration_ms = millis(open.started.elapsed());
        let turn = self.turn_mut(open.turn);
        turn.tool_calls.push(TrajectoryToolCall {
            tool_call_id,
            tool: open.tool,
            ok: false,
            duration_ms,
        });
    }
}

/// A turn being assembled.
#[derive(Debug)]
struct TurnBuilder {
    /// Turn number.
    ///
    /// Matches the engine's indexing for the turn whose events are being
    /// collected.
    turn: usize,

    /// Query that initiated the turn.
    ///
    /// Empty when the turn was opened without a start event (mid-run
    /// attach).
    query: String,

    /// Model text response, already capture-limited.
    ///
    /// Truncated when captured, so the builder never holds more than the
    /// configured limit regardless of response size.
    response_text: String,

    /// Tool calls completed so far, in completion order.
    ///
    /// A call whose result arrived without a dispatch (mid-run attach)
    /// is appended here like any other.
    tool_calls: Vec<TrajectoryToolCall>,

    /// Turn duration, filled at turn end.
    ///
    /// Zero stays zero when no end event arrives before the turn closes.
    duration_ms: u64,

    /// Turn input tokens, filled at turn end.
    ///
    /// Zero until a matching end event supplies the figure.
    input_tokens: u64,

    /// Turn output tokens, filled at turn end.
    ///
    /// Zero until a matching end event supplies the figure.
    output_tokens: u64,
}

impl TurnBuilder {
    /// A fresh turn builder with nothing captured yet.
    fn new(turn: usize, query: String) -> Self {
        Self {
            turn,
            query,
            response_text: String::new(),
            tool_calls: Vec::new(),
            duration_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
        }
    }
}

/// A tool dispatch awaiting its result.
#[derive(Debug)]
struct OpenCall {
    /// Tool name from the dispatch.
    ///
    /// Kept so an abandoned call can still be rendered with the tool it
    /// targeted, even though its result never arrived.
    tool: String,

    /// Turn the dispatch belonged to.
    ///
    /// Carried from the dispatch event so a call abandoned at run end is
    /// attributed to the correct turn even when no turn was open.
    turn: usize,

    /// Monotonic dispatch instant.
    ///
    /// The duration source for a call abandoned at run end — the only
    /// measurement available when no result event ever arrives.
    started: Instant,
}

/// Captures runs into [`TrajectoryRecord`]s and optionally writes them out.
///
/// Pure listener: nothing here blocks, vetoes, or fails the run — a sink
/// write error is a `warn!` and the record's file output is dropped (the
/// alternative, failing a successful run because its log could not flush,
/// inverts the dependency). The in-memory record survives either way and is
/// reachable via [`records`](Self::records). Captured records contain
/// prompt, response, and tool text; the records and any ledger directory
/// are to be treated as sensitive data. The sink append at run end is
/// the one synchronous operation on the run path; it is accepted in
/// exchange for crash-safety and assumes local-disk latency.
///
/// # Thread Safety
///
/// [`TrajectoryObserver`] is `Send + Sync`. Interior mutability is handled
/// via internal mutexes (recovered, never propagated), so every callback
/// takes `&self` and the observer can be shared as
/// `Arc<TrajectoryObserver>` or registered on several loops. Locks are
/// held only for in-memory state updates; the sink write at run end
/// happens outside the locks. Sharing one
/// observer between *concurrently running* loops interleaves their
/// events; the clobbering run start warns and discards the orphaned
/// record.
///
/// # Example
///
/// ```
/// use loopctl::memory::trajectory::TrajectoryObserver;
///
/// let observer = TrajectoryObserver::in_memory().with_capture_limit(4_000);
/// // Attach with `BareLoop::with_observer(Arc::new(observer))` and run;
/// // finished records are then available via `records()`.
/// # assert!(observer.records().is_empty());
/// ```
pub struct TrajectoryObserver {
    /// The run being assembled, if one is in flight.
    ///
    /// Category 1 mutex: guarded by `recover_guard`, since a panicked
    /// holder's half-built record is discardable — the observer keeps
    /// listening either way.
    inner: Mutex<Option<RecordBuilder>>,

    /// Records of runs that ended, oldest first.
    ///
    /// Unbounded unless [`with_memory_retention`](TrajectoryObserver::with_memory_retention)
    /// set a cap; past the cap, the oldest records are dropped.
    finished: Mutex<VecDeque<TrajectoryRecord>>,

    /// Directory the JSONL ledger is appended to, when disk capture is on.
    sink: Option<PathBuf>,

    /// Maximum characters of response text kept per turn.
    capture_limit: usize,

    /// How many finished records memory retains; `None` retains all.
    retention: Option<usize>,
}

impl std::fmt::Debug for TrajectoryObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let finished = recover_guard(self.finished.lock()).len();
        let in_flight = recover_guard(self.inner.lock()).is_some();
        f.debug_struct("TrajectoryObserver")
            .field("finished_records", &finished)
            .field("run_in_flight", &in_flight)
            .field("sink", &self.sink)
            .field("capture_limit", &self.capture_limit)
            .field("retention", &self.retention)
            .finish_non_exhaustive()
    }
}

impl Default for TrajectoryObserver {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl TrajectoryObserver {
    /// Capture in memory only.
    ///
    /// Finished records accumulate and are handed to the host via
    /// [`records`](Self::records) — the form tests and in-process consumers
    /// use. No disk writes happen.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            inner: Mutex::new(None),
            finished: Mutex::new(VecDeque::new()),
            sink: None,
            capture_limit: DEFAULT_CAPTURE_LIMIT,
            retention: None,
        }
    }

    /// Capture and append each finished record as one JSONL line.
    ///
    /// The ledger lives at `<dir>/trajectory.jsonl`; the directory is
    /// created best-effort on first write. Every record is written with an
    /// open-write-close pass — crash-safe by construction, nothing buffered
    /// to lose — as a single append, so each record lands as one whole
    /// line. A write failure emits one `warn!` per affected record and is
    /// otherwise silent; the in-memory record survives.
    ///
    /// Two trade-offs apply. The append is performed synchronously during
    /// run completion to preserve crash-safety; where storage latency is
    /// significant, point the sink at fast local storage or use
    /// [`in_memory`](Self::in_memory). The ledger grows without bound;
    /// rotation and pruning are the host's responsibility. Captured
    /// content is plaintext prompt, response, and tool text; the ledger
    /// and its directory are to be treated as sensitive.
    ///
    /// One ledger per process: this crate does not serialize appends
    /// across processes, so point two concurrently running hosts at the
    /// same `dir` only if interleaving their records in one file is
    /// acceptable.
    #[must_use]
    pub fn writing_to(dir: impl Into<PathBuf>) -> Self {
        Self {
            sink: Some(dir.into()),
            ..Self::in_memory()
        }
    }

    /// Set the maximum characters of response text kept per turn.
    ///
    /// Defaults to 2,000 characters: sized for extraction while bounding
    /// the ledger growth of long, many-turn sessions.
    #[must_use]
    pub fn with_capture_limit(mut self, chars: usize) -> Self {
        self.capture_limit = chars;
        self
    }

    /// The finished records, oldest first.
    ///
    /// Only runs whose end was observed appear here; a run in flight is not
    /// included. The retained set is cloned on every call, so the cost is
    /// proportional to the retained records' size.
    #[must_use]
    pub fn records(&self) -> Vec<TrajectoryRecord> {
        recover_guard(self.finished.lock())
            .iter()
            .cloned()
            .collect()
    }

    /// Set how many finished records are retained in memory.
    ///
    /// Retention is first-in, first-out: once the cap is reached, each
    /// new record drops the oldest. `None` (the default) retains every
    /// record, which is appropriate for short-lived processes and tests;
    /// a long-lived host should set a cap sized to its consumers. Disk
    /// capture is unaffected: the ledger retains every record it
    /// successfully wrote.
    #[must_use]
    pub fn with_memory_retention(mut self, limit: Option<usize>) -> Self {
        self.retention = limit;
        self
    }
}

impl LoopObserver for TrajectoryObserver {
    fn name(&self) -> &'static str {
        "trajectory"
    }

    fn on_run_start(&self, ctx: &RunStartContext) {
        let mut guard = recover_guard(self.inner.lock());
        if guard
            .replace(RecordBuilder::new(
                ctx.session_id.to_string(),
                self.capture_limit,
            ))
            .is_some()
        {
            tracing::warn!(
                target: "loopctl::trajectory",
                records_lost = 1,
                "a run started before the previous run ended; its unfinished \
                 record is discarded — attach one observer per concurrently \
                 running loop"
            );
        }
    }

    fn on_turn_start(&self, ctx: &TurnStartContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(builder) = guard.as_mut() else {
            return;
        };
        builder.close_current();
        builder.current = Some(TurnBuilder::new(ctx.turn, ctx.query.clone()));
    }

    fn on_response(&self, ctx: &ResponseContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(builder) = guard.as_mut() else {
            return;
        };
        let text = truncate_chars(&ctx.text, self.capture_limit);
        builder.turn_mut(ctx.turn).response_text = text;
    }

    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(builder) = guard.as_mut() else {
            return;
        };
        builder.open_calls.insert(
            ctx.tool_call_id.clone(),
            OpenCall {
                tool: ctx.tool.clone(),
                turn: ctx.turn,
                started: Instant::now(),
            },
        );
    }

    fn on_tool_post(&self, ctx: &ToolPostContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(builder) = guard.as_mut() else {
            return;
        };
        let _ = builder.open_calls.remove(&ctx.tool_call_id);
        let turn = builder.turn_mut(ctx.turn);
        turn.tool_calls.push(TrajectoryToolCall {
            tool_call_id: ctx.tool_call_id.clone(),
            tool: ctx.tool.clone(),
            ok: !ctx.is_error,
            duration_ms: millis(ctx.duration),
        });
    }

    fn on_turn_end(&self, ctx: &TurnEndContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(builder) = guard.as_mut() else {
            return;
        };
        if let Some(turn) = builder
            .current
            .as_mut()
            .filter(|open| open.turn == ctx.turn)
        {
            turn.duration_ms = ctx.duration_ms;
            turn.input_tokens = ctx.input_tokens;
            turn.output_tokens = ctx.output_tokens;
            builder.close_current();
        } else if let Some(done) = builder
            .turns
            .last_mut()
            .filter(|done| done.turn == ctx.turn)
        {
            done.duration_ms = ctx.duration_ms;
            done.input_tokens = ctx.input_tokens;
            done.output_tokens = ctx.output_tokens;
        } else {
            builder.turn_mut(ctx.turn);
            builder.close_current();
        }
    }

    fn on_run_end(&self, ctx: &RunEndContext) {
        let mut guard = recover_guard(self.inner.lock());
        let Some(mut builder) = guard.take() else {
            return;
        };
        drop(guard);

        let unanswered: Vec<(String, OpenCall)> = builder.open_calls.drain().collect();
        for (tool_call_id, open) in unanswered {
            builder.abandon_call(tool_call_id, open);
        }
        builder.close_current();

        let any_call_ok = builder
            .turns
            .iter()
            .flat_map(|turn| turn.tool_calls.iter())
            .any(|call| call.ok);
        let outcome = if ctx.success {
            TrajectoryOutcome::Success
        } else if any_call_ok {
            TrajectoryOutcome::Partial
        } else {
            TrajectoryOutcome::Failure
        };

        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;
        for turn in &builder.turns {
            input_tokens = input_tokens.saturating_add(turn.input_tokens);
            output_tokens = output_tokens.saturating_add(turn.output_tokens);
        }

        let record = TrajectoryRecord {
            session_id: builder.session_id,
            run_id: builder.run_id,
            outcome,
            started_at: builder.started_at,
            duration_ms: ctx.duration_ms,
            total_turns: ctx.total_turns,
            token_summary: TokenSummary {
                input_tokens,
                output_tokens,
                cached_input_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
            turns: builder.turns,
        };
        emit_run_signals(&record);
        self.append_to_ledger(&record);
        let mut finished = recover_guard(self.finished.lock());
        finished.push_back(record);
        if let Some(cap) = self.retention {
            while finished.len() > cap {
                finished.pop_front();
            }
        }
    }
}

impl TrajectoryObserver {
    /// Append one finished record to the JSONL ledger, when a sink is set.
    ///
    /// Best-effort by contract: any failure along the way emits exactly one
    /// warning for the record and drops the file output. The in-memory copy
    /// is unaffected. A failed write is repaired by truncating back to the
    /// pre-write length, so a partial line never corrupts the ledger's
    /// one-record-per-line contract for the records around it.
    fn append_to_ledger(&self, record: &TrajectoryRecord) {
        let Some(dir) = &self.sink else {
            return;
        };
        let line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!(
                    target: "loopctl::trajectory",
                    error = %error,
                    records_lost = 1,
                    "trajectory record could not be serialized; file output dropped"
                );
                return;
            }
        };
        if std::fs::create_dir_all(dir).is_err() {
            tracing::warn!(
                target: "loopctl::trajectory",
                records_lost = 1,
                dir = %dir.display(),
                "trajectory ledger directory could not be created; file output dropped"
            );
            return;
        }
        let path = dir.join(LEDGER_FILE);
        let bytes = line.len().saturating_add(1) as u64;
        let write = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| {
                use std::io::Write as _;
                let start = file.metadata().map_or(0, |m| m.len());
                let mut out = line;
                out.push('\n');
                match file.write_all(out.as_bytes()) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        if file.set_len(start).is_err() {
                            tracing::debug!(
                                target: "loopctl::trajectory",
                                "ledger truncation after a failed write did not succeed"
                            );
                        }
                        Err(error)
                    }
                }
            });
        match write {
            Ok(()) => {
                tracing::debug!(
                    target: "loopctl::metrics",
                    metric = "loopctl.trajectory.jsonl_bytes",
                    value = bytes,
                    "trajectory ledger flushed"
                );
            }
            Err(error) => {
                tracing::warn!(
                    target: "loopctl::trajectory",
                    error = %error,
                    records_lost = 1,
                    path = %path.display(),
                    "trajectory ledger write failed; file output dropped"
                );
            }
        }
    }
}

/// Emit the run-end telemetry signals: the summary span and the outcome
/// counter.
fn emit_run_signals(record: &TrajectoryRecord) {
    let outcome = outcome_label(&record.outcome);
    let tool_calls: usize = record.turns.iter().map(|turn| turn.tool_calls.len()).sum();
    let span = tracing::debug_span!(
        target: "loopctl::trajectory",
        "trajectory.run",
        outcome = %outcome,
        turns = record.total_turns,
        duration_ms = record.duration_ms,
        tool_calls = tool_calls,
        input_tokens = record.token_summary.input_tokens,
        output_tokens = record.token_summary.output_tokens,
    );
    {
        let _entered = span.enter();
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.trajectory.records",
            outcome = %outcome,
            "trajectory record finished"
        );
    }
}

/// The stable JSONL/telemetry label for an outcome.
fn outcome_label(outcome: &TrajectoryOutcome) -> &'static str {
    match outcome {
        TrajectoryOutcome::Success => "success",
        TrajectoryOutcome::Failure => "failure",
        TrajectoryOutcome::Partial => "partial",
    }
}

/// Truncate to at most `limit` characters, on char boundaries.
fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}

/// Milliseconds of a duration, saturating at `u64` instead of overflowing.
fn millis(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// The current instant as an RFC 3339 UTC timestamp.
///
/// Falls back to the raw Unix-seconds rendering in the event of an
/// arithmetic overflow, which realistic clock values cannot produce; the
/// fallback exists so that capture never panics.
fn rfc3339_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339(now.as_secs()).unwrap_or_else(|| format!("<unix:{}>", now.as_secs()))
}

/// Render Unix seconds as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Exact-integer civil-date conversion, valid across the whole `u64` second
/// range short of intermediate overflow; `None` only when that arithmetic
/// overflows.
fn format_rfc3339(unix_secs: u64) -> Option<String> {
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(unix_secs.div_euclid(86_400))?;
    let hour = secs_of_day.div_euclid(3_600);
    let minute = secs_of_day.div_euclid(60).rem_euclid(60);
    let second = secs_of_day.rem_euclid(60);
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

/// Convert days since the Unix epoch to a proleptic-Gregorian date.
///
/// Howard Hinnant's `civil_from_days` algorithm, restructured onto
/// checked/saturating and division methods so no arithmetic can overflow or
/// panic; `None` only when an intermediate overflows.
fn civil_from_days(days_since_epoch: u64) -> Option<(u64, u64, u64)> {
    let z = days_since_epoch.checked_add(719_468)?;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = doe
        .saturating_sub(doe.div_euclid(1_460))
        .saturating_add(doe.div_euclid(36_524))
        .saturating_sub(doe.div_euclid(146_096))
        .div_euclid(365);
    let doy = doe.saturating_sub(
        yoe.saturating_mul(365)
            .saturating_add(yoe.div_euclid(4))
            .saturating_sub(yoe.div_euclid(100)),
    );
    let mp = doy.saturating_mul(5).checked_add(2)?.div_euclid(153);
    let day = doy
        .saturating_sub(mp.saturating_mul(153).checked_add(2)?.div_euclid(5))
        .checked_add(1)?;
    let month = if mp < 10 {
        mp.checked_add(3)?
    } else {
        mp.checked_sub(9)?
    };
    let year = yoe.checked_add(era.saturating_mul(400))?;
    let year = if month <= 2 {
        year.checked_add(1)?
    } else {
        year
    };
    Some((year, month, day))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as TestMutex;

    /// A minimal subscriber that captures event messages, so the
    /// bounded-warning contract is pinned rather than trusted.
    struct WarnCapture {
        /// The `message` field of every emitted event, in order.
        messages: TestMutex<Vec<String>>,
    }

    impl WarnCapture {
        fn new() -> Self {
            Self {
                messages: TestMutex::new(Vec::new()),
            }
        }

        fn messages_containing(&self, needle: &str) -> usize {
            self.messages
                .lock()
                .unwrap()
                .iter()
                .filter(|m| m.contains(needle))
                .count()
        }
    }

    impl tracing::Subscriber for WarnCapture {
        fn enabled(&self, _meta: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::Id {
            tracing::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            struct MessageVisitor {
                message: String,
            }
            impl tracing::field::Visit for MessageVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.message = format!("{value:?}");
                    }
                }
            }
            let mut visitor = MessageVisitor {
                message: String::new(),
            };
            event.record(&mut visitor);
            self.messages.lock().unwrap().push(visitor.message);
        }

        fn enter(&self, _id: &tracing::Id) {}

        fn exit(&self, _id: &tracing::Id) {}
    }

    #[test]
    fn format_rfc3339_renders_known_timestamps() {
        assert_eq!(
            format_rfc3339(0).as_deref(),
            Some("1970-01-01T00:00:00Z"),
            "the Unix epoch renders exactly"
        );
        assert_eq!(
            format_rfc3339(86_400).as_deref(),
            Some("1970-01-02T00:00:00Z")
        );
        assert_eq!(
            format_rfc3339(951_868_800).as_deref(),
            Some("2000-03-01T00:00:00Z"),
            "the day after a leap-year February lands on March 1"
        );
        assert_eq!(
            format_rfc3339(1_709_164_800).as_deref(),
            Some("2024-02-29T00:00:00Z"),
            "leap days render correctly"
        );
        assert_eq!(
            format_rfc3339(1_700_000_000).as_deref(),
            Some("2023-11-14T22:13:20Z"),
            "an arbitrary modern timestamp renders with time of day"
        );
        assert_eq!(
            format_rfc3339(4_102_444_799).as_deref(),
            Some("2099-12-31T23:59:59Z"),
            "the last second of a century's first decade renders"
        );
    }

    #[test]
    fn truncate_chars_respects_the_limit_and_boundaries() {
        assert_eq!(truncate_chars("abcdef", 3), "abc");
        assert_eq!(truncate_chars("ab", 3), "ab", "shorter text passes through");
        assert_eq!(
            truncate_chars("héllo", 2),
            "hé",
            "multi-byte chars stay whole"
        );
    }

    #[test]
    fn outcome_labels_are_the_stable_jsonl_names() {
        assert_eq!(outcome_label(&TrajectoryOutcome::Success), "success");
        assert_eq!(outcome_label(&TrajectoryOutcome::Failure), "failure");
        assert_eq!(outcome_label(&TrajectoryOutcome::Partial), "partial");
    }

    /// Feed pre/post events for two calls in one wave, deliberately
    /// interleaved so a positional implementation pairs them wrong.
    fn feed_interleaved_calls(observer: &TrajectoryObserver) {
        let session = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        observer.on_run_start(&session);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "parallel work".to_string(),
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "alpha".to_string(),
            tool_call_id: "call_a".to_string(),
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "beta".to_string(),
            tool_call_id: "call_b".to_string(),
        });
        observer.on_tool_post(&ToolPostContext {
            tool_call_id: "call_b".to_string(),
            turn: 0,
            tool: "beta".to_string(),
            result_hash: None,
            is_error: false,
            duration: std::time::Duration::from_millis(2),
            display_hint: None,
        });
        observer.on_tool_post(&ToolPostContext {
            tool_call_id: "call_a".to_string(),
            turn: 0,
            tool: "alpha".to_string(),
            result_hash: None,
            is_error: true,
            duration: std::time::Duration::from_millis(7),
            display_hint: None,
        });
        observer.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 20,
            input_tokens: 10,
            output_tokens: 5,
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 1,
            duration_ms: 25,
        });
    }

    #[test]
    fn parallel_calls_pair_by_id_not_position() {
        let observer = TrajectoryObserver::in_memory();
        feed_interleaved_calls(&observer);

        let record = &observer.records()[0];
        let calls = &record.turns[0].tool_calls;
        assert_eq!(calls.len(), 2, "both calls are captured");
        assert_eq!(
            calls[0].tool_call_id, "call_b",
            "completion order is driven by pairing, not dispatch position"
        );
        assert_eq!(calls[0].tool, "beta", "each id pairs with its own tool");
        assert!(calls[0].ok);
        assert_eq!(calls[1].tool_call_id, "call_a");
        assert_eq!(calls[1].tool, "alpha");
        assert!(
            !calls[1].ok,
            "the error result lands on its own call, not its neighbor"
        );
        assert_eq!(calls[1].duration_ms, 7);
    }

    #[test]
    fn cancelled_in_flight_call_renders_unfinished() {
        let observer = TrajectoryObserver::in_memory();
        let session = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        observer.on_run_start(&session);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "cancelled work".to_string(),
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "slow".to_string(),
            tool_call_id: "call_x".to_string(),
        });
        observer.on_run_end(&RunEndContext {
            success: false,
            error: Some("cancelled".to_string()),
            total_turns: 1,
            duration_ms: 50,
        });

        let record = &observer.records()[0];
        let calls = &record.turns[0].tool_calls;
        assert_eq!(calls.len(), 1, "the abandoned call is not lost");
        assert!(
            !calls[0].ok,
            "a call still in flight at run end renders as unfinished"
        );
        assert_eq!(
            record.outcome,
            TrajectoryOutcome::Failure,
            "no successful tool work means Failure, even with a dispatch seen"
        );
    }

    #[test]
    fn mid_run_attach_opens_slots_lazily() {
        let observer = TrajectoryObserver::in_memory();
        let session = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        // The run started before the observer attached: only the result is
        // ever observed.
        observer.on_run_start(&session);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "attached mid-run".to_string(),
        });
        observer.on_tool_post(&ToolPostContext {
            tool_call_id: "call_orphan".to_string(),
            turn: 0,
            tool: "echo".to_string(),
            result_hash: None,
            is_error: false,
            duration: std::time::Duration::from_millis(4),
            display_hint: None,
        });
        observer.on_run_end(&RunEndContext {
            success: false,
            error: None,
            total_turns: 1,
            duration_ms: 9,
        });

        let record = &observer.records()[0];
        let calls = &record.turns[0].tool_calls;
        assert_eq!(
            calls.len(),
            1,
            "an orphan post creates its own slot without a panic or a loss"
        );
        assert_eq!(calls[0].tool_call_id, "call_orphan");
        assert!(calls[0].ok);
        assert_eq!(
            record.outcome,
            TrajectoryOutcome::Partial,
            "the orphan's success counts toward the Partial classification"
        );
    }

    #[test]
    fn a_run_started_mid_run_replaces_the_orphaned_record() {
        let observer = TrajectoryObserver::in_memory();
        let first = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        observer.on_run_start(&first);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "never finished".to_string(),
        });

        // A second run starting on the same observer (two loops sharing it)
        // must not resurrect the orphaned run's half-built record.
        let second = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        observer.on_run_start(&second);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "the live run".to_string(),
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 1,
            duration_ms: 5,
        });

        let records = observer.records();
        assert_eq!(
            records.len(),
            1,
            "the orphaned run's record is discarded, loudly, at run start"
        );
        assert_eq!(records[0].turns[0].query, "the live run");
    }

    #[test]
    fn an_empty_run_captures_an_empty_record() {
        let observer = TrajectoryObserver::in_memory();
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 0,
            duration_ms: 0,
        });

        let record = &observer.records()[0];
        assert!(
            record.turns.is_empty(),
            "a run with no turns records zero turns, never a placeholder"
        );
        assert_eq!(record.outcome, TrajectoryOutcome::Success);
        assert_eq!(record.token_summary.input_tokens, 0);
    }

    #[test]
    fn capture_limit_bounds_the_captured_response() {
        let observer = TrajectoryObserver::in_memory().with_capture_limit(3);
        let session = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };
        observer.on_run_start(&session);
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "q".to_string(),
        });
        observer.on_response(&ResponseContext {
            turn: 0,
            text: "abcde".to_string(),
            usage: None,
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 1,
            duration_ms: 1,
        });

        let record = &observer.records()[0];
        assert_eq!(
            record.turns[0].response_text, "abc",
            "response text is truncated to the configured limit"
        );

        let empty = TrajectoryObserver::in_memory().with_capture_limit(0);
        empty.on_run_start(&session);
        empty.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "q".to_string(),
        });
        empty.on_response(&ResponseContext {
            turn: 0,
            text: "abcde".to_string(),
            usage: None,
        });
        empty.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 1,
            duration_ms: 1,
        });
        assert_eq!(
            empty.records()[0].turns[0].response_text,
            "",
            "a zero limit captures no response text"
        );
    }

    #[test]
    fn events_outside_a_run_are_ignored() {
        let observer = TrajectoryObserver::in_memory();

        // Every callback, with no run in flight — note `on_run_start` is
        // deliberately never called: the observer absorbs each
        // one and stays healthy for the run that follows.
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "orphan".to_string(),
        });
        observer.on_response(&ResponseContext {
            turn: 0,
            text: "orphan".to_string(),
            usage: None,
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "t".to_string(),
            tool_call_id: "call_o".to_string(),
        });
        observer.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 1,
            input_tokens: 1,
            output_tokens: 1,
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 1,
            duration_ms: 1,
        });
        assert!(
            observer.records().is_empty(),
            "events with no run in flight record nothing"
        );

        // The observer still works for a run that starts afterwards.
        feed_interleaved_calls(&observer);
        assert_eq!(observer.records().len(), 1);
        assert_eq!(observer.records()[0].turns[0].tool_calls.len(), 2);
    }

    #[test]
    fn a_clobbered_run_warns_exactly_once_per_clobber() {
        let capture = std::sync::Arc::new(WarnCapture::new());
        let observer = TrajectoryObserver::in_memory();
        let session = RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        };

        let sink = std::sync::Arc::clone(&capture);
        tracing::subscriber::with_default(sink, || {
            observer.on_run_start(&session);
            observer.on_run_start(&session);
        });
        assert_eq!(
            capture.messages_containing("record is discarded"),
            1,
            "one clobbering run start emits exactly one bounded warning"
        );

        let sink = std::sync::Arc::clone(&capture);
        tracing::subscriber::with_default(sink, || {
            observer.on_run_end(&RunEndContext {
                success: true,
                error: None,
                total_turns: 1,
                duration_ms: 1,
            });
        });
        assert_eq!(
            capture.messages_containing("record is discarded"),
            1,
            "a clean run end emits no loss warning"
        );
    }

    #[test]
    fn observer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrajectoryObserver>();
    }
    #[test]
    fn debug_rendering_does_not_expose_captured_text() {
        let observer = TrajectoryObserver::in_memory();
        feed_interleaved_calls(&observer);
        let rendered = format!("{observer:?}");
        assert!(
            !rendered.contains("parallel work"),
            "Debug output must not include captured query text: {rendered}"
        );
        assert!(
            rendered.contains("finished_records"),
            "Debug output summarizes observer state: {rendered}"
        );
    }
    #[test]
    fn abandoned_call_is_attributed_to_its_dispatch_turn() {
        let observer = TrajectoryObserver::in_memory();
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        // Mid-run attach: a dispatch on turn 5 with no turn ever opened.
        observer.on_tool_pre(&ToolPreContext {
            turn: 5,
            tool: "slow".to_string(),
            tool_call_id: "call_late".to_string(),
        });
        observer.on_run_end(&RunEndContext {
            success: false,
            error: None,
            total_turns: 6,
            duration_ms: 40,
        });

        let record = &observer.records()[0];
        assert_eq!(
            record.turns.len(),
            1,
            "the abandoned call opens exactly one turn slot"
        );
        assert_eq!(
            record.turns[0].turn, 5,
            "an abandoned call is attributed to the turn it was dispatched on"
        );
        assert_eq!(record.turns[0].tool_calls[0].tool_call_id, "call_late");
    }
    #[test]
    fn a_late_turn_end_patches_its_own_turn() {
        let observer = TrajectoryObserver::in_memory();
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_turn_start(&TurnStartContext {
            turn: 3,
            query: "first".to_string(),
        });
        // A later turn's event closes turn 3 before its end event arrives.
        observer.on_response(&ResponseContext {
            turn: 4,
            text: "next turn".to_string(),
            usage: None,
        });
        observer.on_turn_end(&TurnEndContext {
            turn: 3,
            success: true,
            error: None,
            duration_ms: 42,
            input_tokens: 7,
            output_tokens: 9,
        });
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 5,
            duration_ms: 60,
        });

        let record = &observer.records()[0];
        let indices: Vec<usize> = record.turns.iter().map(|turn| turn.turn).collect();
        assert_eq!(
            indices.len(),
            indices
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            "no two captured turns share an index"
        );
        let patched = record
            .turns
            .iter()
            .find(|turn| turn.turn == 3)
            .expect("turn 3 is captured");
        assert_eq!(
            patched.duration_ms, 42,
            "the late end event patches its turn"
        );
        assert_eq!(patched.input_tokens, 7);
        assert_eq!(patched.output_tokens, 9);
    }
}
