//! Per-event ledger records — the event stream beside the per-run ledger.
//!
//! [`TrajectoryObserver`](super::TrajectoryObserver) writes one
//! [`TrajectoryRecord`](super::TrajectoryRecord) per run;
//! [`EventLedgerObserver`] writes one [`TrajectoryEvent`] per mapped
//! lifecycle callback, as a sibling `events.jsonl` in the same sink
//! directory. The two streams never mix lines — the per-run interchange
//! contract is untouched, and each file parses as one schema. Register
//! both observers on the same loop; the per-run record still comes only
//! from [`TrajectoryObserver`](super::TrajectoryObserver), so the two
//! compose without double-writing.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::memory::trajectory::EventLedgerObserver;
//! use loopctl::observer::{LoopObserver, RunEndContext, RunStartContext};
//!
//! let dir = std::env::temp_dir().join(format!("events-doctest-{}", uuid::Uuid::new_v4()));
//! let observer = EventLedgerObserver::writing_to(dir.clone());
//! observer.on_run_start(&RunStartContext {
//!     session_id: uuid::Uuid::new_v4(),
//! });
//! observer.on_run_end(&RunEndContext::new(true, None, 0, 1));
//! observer.flush();
//!
//! let ledger = std::fs::read_to_string(dir.join("events.jsonl"))
//!     .expect("the events ledger exists after a flushed run");
//! assert_eq!(ledger.lines().count(), 2, "one line per mapped callback");
//! ```
//!
//! # Kinds
//!
//! Each line is `serde_json::to_string` of a [`TrajectoryEvent`],
//! fields in declaration order. `turn` is the context's turn number
//! for turn-scoped kinds and `null` for run-scoped ones.
//!
//! | kind | `turn` | `data` |
//! |------|--------|--------|
//! | `run.started` | `null` | `{}` |
//! | `run.ended` | `null` | `{success, error, total_turns, duration_ms}` |
//! | `turn.started` | number | `{query}` — truncated to the observer's capture limit (default 2,000 characters) |
//! | `turn.ended` | number | `{success, error, duration_ms, input_tokens, output_tokens, stop_reason}` |
//! | `compaction` | `null` | `{tokens_before, tokens_after, tokens_saved, evicted_messages, reason}` |
//! | `fallback` | `null` | `{from, to}` |
//! | `model.switched` | `null` | `{from, to}` |
//! | `detection` | `null` | `{detector, pattern, repetitions}` or `{detector, action}` |
//! | `tool.call` | number | `{tool_call_id, tool}` |
//! | `tool.result` | number | `{tool_call_id, tool, result_hash, is_error, duration_ms}` |
//!
//! # Numbering and correlation
//!
//! `run_id` is per-sink and monotonic from `1`; `seq` resets at each
//! `run.started`, which carries `seq = 1` — the event hub's numbering
//! discipline. The per-run ledger's `run_id` is its own minted uuid, so
//! cross-ledger correlation rides `session_id`, which both streams
//! carry. One observer serves at most one active run: a run start while
//! a run is still open warns once, steps the run id, and restarts the
//! sequence — attach one observer per concurrently running loop. A
//! run's end seals its stream: stray callbacks after it write nothing
//! until the next run start.
//!
//! # Excluded and reserved kinds
//!
//! v1 maps the ten kinds above. Text and thinking deltas, `response`,
//! stream success/failure, `tool_call_received`, `pre_compaction`, and
//! `transport.fallback` are excluded — delta volume, the
//! reserved-for-`usage` per-turn accounting, pre-dispatch accumulation
//! that `tool.call` already covers, and the streaming-to-non-streaming
//! transport retry, which is mechanics rather than a run semantic (the
//! mapped `fallback` kind is the model fallback). `gate.decision` and
//! `usage` are reserved labels; this version does not emit them.
//!
//! # Reader tolerance
//!
//! The ledger is appended one complete line per record under a
//! truncate-back repair, so a torn line can only be the file's last. A
//! consumer treats an unparseable trailing line as end-of-record and
//! keeps everything before it; no loader ships with this module. An
//! unknown `kind` from a newer release deserializes to
//! [`Unknown`](TrajectoryEventKind::Unknown) rather than failing the
//! line, so an older reader keeps the envelope and `data`.
//!
//! # Trust scope
//!
//! Every event line carries its callback's content — user text (a
//! post-tool turn's `query` includes the prior tool result's text),
//! model output, tool inputs and tool-result metadata — so an events
//! ledger belongs to the same trust scope as the run it records. The
//! file is created owner-only (`0600`) on unix, matching the hardened
//! per-run ledger — defense in depth, not a trust boundary; the
//! permissions of an existing file remain the host's.

use std::path::PathBuf;
use std::sync::Mutex;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::error::recover_guard;
use crate::observer::{
    CompactedContext, ConvergenceDetectedContext, FallbackContext, LoopDetectedContext,
    LoopObserver, ModelSwitchedContext, RunEndContext, RunStartContext, ToolPostContext,
    ToolPreContext, TurnEndContext, TurnStartContext,
};

use super::sink::{DEFAULT_QUEUE_CAPACITY, LedgerWriter};
use super::{DEFAULT_CAPTURE_LIMIT, millis, rfc3339_now, truncate_chars};

/// File name of the per-event ledger inside the sink directory.
///
/// Sits beside `trajectory.jsonl`; hosts locate a session's event
/// stream by joining the sink directory with this name.
const EVENTS_FILE: &str = "events.jsonl";

/// Which lifecycle moment one event line records.
///
/// One dotted JSON label per mapped callback; the reserved labels
/// `gate.decision` and `usage` are not emitted in this version.
/// The enum is `#[non_exhaustive]`: a later kind widens it in a minor
/// release, so out-of-crate consumers match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TrajectoryEventKind {
    /// A run began.
    ///
    /// The first event of each run; its `seq` is always `1`, because
    /// the observer resets its counter here.
    #[serde(rename = "run.started")]
    RunStarted,

    /// A run ended.
    ///
    /// The last event of a run, whatever its completion status.
    #[serde(rename = "run.ended")]
    RunEnded,

    /// A turn began.
    ///
    /// One per model call the run makes.
    #[serde(rename = "turn.started")]
    TurnStarted,

    /// A turn ended.
    ///
    /// Carries the turn's accounting in `data`.
    #[serde(rename = "turn.ended")]
    TurnEnded,

    /// A compaction pass completed.
    ///
    /// Fired after the pass, with its token facts in `data`.
    #[serde(rename = "compaction")]
    Compaction,

    /// The run fell back to another model.
    ///
    /// Names both models in `data`.
    #[serde(rename = "fallback")]
    Fallback,

    /// The active model was switched.
    ///
    /// Names the models involved in `data`.
    #[serde(rename = "model.switched")]
    ModelSwitched,

    /// Loop or convergence detection fired.
    ///
    /// The `detector` field of `data` says which.
    #[serde(rename = "detection")]
    Detection,

    /// A tool dispatch is about to run.
    ///
    /// The last event before the tool executes, carrying the dispatch
    /// identity.
    #[serde(rename = "tool.call")]
    ToolCall,

    /// A tool dispatch finished, as metadata.
    ///
    /// Carries the result hash, error flag, and duration — never the
    /// output text.
    #[serde(rename = "tool.result")]
    ToolResult,

    /// A kind this version does not know.
    ///
    /// Deserialization-only: a ledger written by a newer release may
    /// carry kinds this version has never heard of, and the fallback
    /// keeps the line parseable — envelope and `data` survive —
    /// instead of rejecting mid-file. The observer never emits it;
    /// serializing a hand-constructed `Unknown` is not part of the
    /// interchange.
    #[serde(other)]
    Unknown,
}

impl TrajectoryEventKind {
    /// The dotted JSONL label for this kind.
    ///
    /// Mirrors the serde renames on the variants; the serde labels are
    /// the contract, this accessor the ergonomic spelling, and the
    /// ordering pin keeps the two from drifting apart.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::RunStarted => "run.started",
            Self::RunEnded => "run.ended",
            Self::TurnStarted => "turn.started",
            Self::TurnEnded => "turn.ended",
            Self::Compaction => "compaction",
            Self::Fallback => "fallback",
            Self::ModelSwitched => "model.switched",
            Self::Detection => "detection",
            Self::ToolCall => "tool.call",
            Self::ToolResult => "tool.result",
            Self::Unknown => "unknown",
        }
    }
}

/// One event ledger line: a lifecycle moment, stamped and enveloped.
///
/// The payload [`EventLedgerObserver`] appends to `events.jsonl`. The
/// struct is `#[non_exhaustive]` so later widening stays additive;
/// fields serialize in declaration order, matching the module's
/// interchange table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrajectoryEvent {
    /// The event's per-run sequence number.
    ///
    /// `1` for the run's `run.started` and strictly increasing within
    /// the run; each run restarts at `1`.
    pub seq: u64,

    /// When the event was recorded, RFC 3339 in UTC.
    ///
    /// Rendered from the system clock at mapping time, whole-second
    /// resolution, `Z`-suffixed.
    pub ts: String,

    /// The run the event belongs to, per-sink and monotonic from `1`.
    ///
    /// Steps at each `run.started` and never resets within an
    /// observer's lifetime. The per-run ledger's `run_id` is a
    /// different, uuid-shaped identifier — correlate the two streams
    /// by `session_id`.
    pub run_id: u64,

    /// The run's session id, remembered from its start.
    ///
    /// The bridge between the per-run and per-event ledgers: both
    /// streams carry the same value for the same run.
    pub session_id: String,

    /// The turn the event belongs to, when the kind is turn-scoped.
    ///
    /// `null` for run-scoped kinds (`run.*`, `compaction`, `fallback`,
    /// `model.switched`, `detection`).
    pub turn: Option<usize>,

    /// Which lifecycle moment this line records.
    ///
    /// One of the ten v1 kinds; the module's interchange table holds
    /// the `data` shape each one carries.
    pub kind: TrajectoryEventKind,

    /// The kind-specific payload.
    ///
    /// One object shape per kind, documented in the module's
    /// interchange table; unknown fields are tolerated when
    /// deserializing.
    pub data: Value,
}

/// The stream's position, mutated under the observer's lock.
///
/// Carries everything one line's envelope needs that the callback
/// itself cannot supply: the remembered session, the current run's id
/// and sequence, and whether that run is still open.
#[derive(Default)]
struct ObserverState {
    /// The run's session id, remembered from its start.
    ///
    /// Set at each run start; consulted for every line's envelope. A
    /// callback arriving while no run is open is ignored, mirroring
    /// the per-run observer's discipline before the first start and
    /// after every end.
    session_id: Option<String>,

    /// The current run's id, per-sink and monotonic.
    ///
    /// Steps at each run start; the first run is `1`.
    run_id: u64,

    /// The current run's sequence counter.
    ///
    /// Reset to zero at each run start, so `run.started` carries
    /// `seq = 1`.
    seq: u64,

    /// Whether a run is open — started without its `run.ended` yet.
    ///
    /// The gate every mapped callback passes: a run start while this
    /// is `true` breaks the one-active-run contract (it warns once,
    /// steps the run id, and restarts the sequence), and once a run
    /// has ended this flag seals the stream until the next start.
    run_open: bool,
}

/// The per-event ledger observer — one JSONL line per mapped callback.
///
/// Register it beside [`TrajectoryObserver`](super::TrajectoryObserver)
/// on the same loop: the per-run record still comes only from that
/// observer, this one owns the sibling `events.jsonl` stream, and the
/// two never write each other's lines. Delivery is best-effort through
/// the same bounded queue as the per-run ledger — a slow or failing
/// sink drops the oldest queued line with a warning and never blocks or
/// fails a run.
pub struct EventLedgerObserver {
    /// The events ledger's background writer.
    ///
    /// All filesystem work happens on the writer's worker thread; the
    /// last dropped reference drains and joins it.
    sink: LedgerWriter,

    /// The stream's position.
    ///
    /// Category 1 mutex: guarded by `recover_guard`, since a panicked
    /// holder's counters are discardable — the observer keeps
    /// listening either way.
    state: Mutex<ObserverState>,

    /// Maximum characters of query text kept per `turn.started` line.
    ///
    /// Applied when the line is stamped, so no event ever holds more
    /// than the limit regardless of query size. Defaults to the
    /// trajectory capture limit; hosts wanting parity with a tuned
    /// per-run observer set it through
    /// [`with_capture_limit`](Self::with_capture_limit).
    capture_limit: usize,
}

impl std::fmt::Debug for EventLedgerObserver {
    /// Renders the observer's observable state, not its events.
    ///
    /// Prints the sink directory and the current position — enough to
    /// identify an observer in a log line without dumping captured
    /// content.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = recover_guard(self.state.lock());
        f.debug_struct("EventLedgerObserver")
            .field("sink", &self.sink.dir().display().to_string())
            .field("run_id", &state.run_id)
            .field("seq", &state.seq)
            .field("run_open", &state.run_open)
            .field("capture_limit", &self.capture_limit)
            .finish_non_exhaustive()
    }
}

impl EventLedgerObserver {
    /// Append each mapped callback as one event line under `dir`.
    ///
    /// The ledger lives at `<dir>/events.jsonl`; the directory is
    /// created best-effort on first write. Events are handed to a
    /// bounded background writer — see the type docs for the
    /// drop-oldest contract.
    #[must_use]
    pub fn writing_to(dir: impl Into<PathBuf>) -> Self {
        Self {
            sink: LedgerWriter::with_file(dir.into(), EVENTS_FILE, DEFAULT_QUEUE_CAPACITY, true),
            state: Mutex::new(ObserverState::default()),
            capture_limit: DEFAULT_CAPTURE_LIMIT,
        }
    }

    /// Set the maximum characters of query text kept per
    /// `turn.started` line.
    ///
    /// Defaults to the trajectory capture limit (2,000 characters):
    /// the same default the per-run observer applies, so an unconfigured
    /// pair of observers truncates identically. Hosts that tune one
    /// ledger's bound should tune the other's to match.
    #[must_use]
    pub fn with_capture_limit(mut self, chars: usize) -> Self {
        self.capture_limit = chars;
        self
    }

    /// Block until every event accepted before this call has been written.
    ///
    /// Intended for tests and orderly shutdown: returns once the
    /// writer's queue is empty and no batch is mid-write.
    pub fn flush(&self) {
        self.sink.flush();
    }

    /// Stamp one event line and hand it to the writer.
    ///
    /// No-ops outside an open run — before the first run start and
    /// after a run's end: the stream records runs, and a callback
    /// outside one is not the stream's to keep. Serialization of the
    /// envelope is total in practice; the warn-and-drop arm mirrors
    /// the per-run ledger's best-effort contract anyway.
    fn append(
        &self,
        state: &mut ObserverState,
        kind: TrajectoryEventKind,
        turn: Option<usize>,
        data: Value,
    ) {
        if !state.run_open {
            return;
        }
        let Some(session_id) = state.session_id.clone() else {
            return;
        };
        state.seq = state.seq.saturating_add(1);
        let event = TrajectoryEvent {
            seq: state.seq,
            ts: rfc3339_now(),
            run_id: state.run_id,
            session_id,
            turn,
            kind,
            data,
        };
        match serde_json::to_string(&event) {
            Ok(line) => self.sink.enqueue(line),
            Err(error) => tracing::warn!(
                target: "loopctl::trajectory",
                error = %error,
                records_lost = 1,
                "event record could not be serialized; file output dropped"
            ),
        }
    }
}

impl LoopObserver for EventLedgerObserver {
    /// Returns `"event_ledger"`, the label observers report to hosts.
    ///
    /// Constant per implementation, so hosts can key observer routing
    /// on it.
    fn name(&self) -> &'static str {
        "event_ledger"
    }

    /// Opens a run: steps the id, resets the sequence, records the
    /// session.
    ///
    /// The first line of every run. A start while a run is still open
    /// breaks the one-active-run contract — it warns once, and the
    /// already-written lines stay while the id steps and the sequence
    /// restarts, the same visible restamp the event hub carries.
    fn on_run_start(&self, ctx: &RunStartContext) {
        let mut state = recover_guard(self.state.lock());
        if state.run_open {
            tracing::warn!(
                target: "loopctl::trajectory",
                "a run started before the previous run ended; the event ledger steps its run id \
                 and restarts its sequence — attach one observer per concurrently running loop"
            );
        }
        state.run_id = state.run_id.saturating_add(1);
        state.seq = 0;
        state.run_open = true;
        state.session_id = Some(ctx.session_id.to_string());
        self.append(
            &mut state,
            TrajectoryEventKind::RunStarted,
            None,
            run_started_data(),
        );
    }

    /// Closes the run with its completion facts.
    ///
    /// The run's last line; nothing further lands until the next run
    /// start opens a new id and sequence — stray callbacks after the
    /// end are dropped, exactly as the per-run observer drops them.
    fn on_run_end(&self, ctx: &RunEndContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::RunEnded,
            None,
            run_ended_data(ctx),
        );
        state.run_open = false;
    }

    /// Records the turn's query, truncated to this observer's capture
    /// limit.
    ///
    /// One verbose turn cannot dominate the event stream; the default
    /// matches the per-run observer's, and
    /// [`with_capture_limit`](Self::with_capture_limit) tunes it.
    fn on_turn_start(&self, ctx: &TurnStartContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::TurnStarted,
            Some(ctx.turn),
            turn_started_data(ctx, self.capture_limit),
        );
    }

    /// Records the turn's accounting.
    ///
    /// Stop reason, token totals, and duration land in `data`.
    fn on_turn_end(&self, ctx: &TurnEndContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::TurnEnded,
            Some(ctx.turn),
            turn_ended_data(ctx),
        );
    }

    /// Records a tool dispatch's identity, before it runs.
    ///
    /// Pairs with the dispatch's `tool.result` by `tool_call_id`.
    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::ToolCall,
            Some(ctx.turn),
            tool_call_data(ctx),
        );
    }

    /// Records a tool dispatch's outcome, as metadata.
    ///
    /// The result hash, error flag, and duration — never the tool's
    /// output text; that rides the following turn's `query` when the
    /// engine feeds the result back.
    fn on_tool_post(&self, ctx: &ToolPostContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::ToolResult,
            Some(ctx.turn),
            tool_result_data(ctx),
        );
    }

    /// Records a completed compaction pass's token facts.
    ///
    /// Run-scoped: the line's `turn` is null.
    fn on_compaction(&self, ctx: &CompactedContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::Compaction,
            None,
            compaction_data(ctx),
        );
    }

    /// Records a model fallback by name.
    ///
    /// Run-scoped: the line's `turn` is null.
    fn on_fallback(&self, ctx: &FallbackContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::Fallback,
            None,
            switch_data(&ctx.from, &ctx.to),
        );
    }

    /// Records a model switch by name.
    ///
    /// Run-scoped: the line's `turn` is null.
    fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::ModelSwitched,
            None,
            switch_data(&ctx.from, &ctx.to),
        );
    }

    /// Records loop detection, named by its detector.
    ///
    /// The `data.detector` field distinguishes it from convergence
    /// detection under the one shared kind.
    fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::Detection,
            None,
            loop_detected_data(ctx),
        );
    }

    /// Records convergence detection, named by its detector.
    ///
    /// The `data.detector` field distinguishes it from loop detection
    /// under the one shared kind.
    fn on_convergence_detected(&self, ctx: &ConvergenceDetectedContext) {
        let mut state = recover_guard(self.state.lock());
        self.append(
            &mut state,
            TrajectoryEventKind::Detection,
            None,
            convergence_detected_data(ctx),
        );
    }
}

/// The `run.started` payload: the run's opening moment carries no
/// extra facts — the envelope already holds the identifiers.
///
/// The empty object keeps `data` a uniform, parseable object for
/// every kind.
fn run_started_data() -> Value {
    serde_json::json!({})
}

/// The `run.ended` payload from the run-end context.
///
/// The context's four facts, verbatim.
fn run_ended_data(ctx: &RunEndContext) -> Value {
    serde_json::json!({
        "success": ctx.success,
        "error": ctx.error,
        "total_turns": ctx.total_turns,
        "duration_ms": ctx.duration_ms,
    })
}

/// The `turn.started` payload: the query, bounded by the observer's
/// capture limit.
///
/// One field, through the shared truncation helper.
fn turn_started_data(ctx: &TurnStartContext, limit: usize) -> Value {
    serde_json::json!({
        "query": truncate_chars(&ctx.query, limit),
    })
}

/// The `turn.ended` payload from the turn-end context.
///
/// The stop reason serializes through its serde labels, matching the
/// run record's vocabulary.
fn turn_ended_data(ctx: &TurnEndContext) -> Value {
    serde_json::json!({
        "success": ctx.success,
        "error": ctx.error,
        "duration_ms": ctx.duration_ms,
        "input_tokens": ctx.input_tokens,
        "output_tokens": ctx.output_tokens,
        "stop_reason": ctx.stop_reason,
    })
}

/// The `compaction` payload: the pass's payload-comparable token facts.
///
/// The telemetry deep view is the per-run ledger's; the event line
/// keeps the flat counts.
fn compaction_data(ctx: &CompactedContext) -> Value {
    serde_json::json!({
        "tokens_before": ctx.tokens_before,
        "tokens_after": ctx.tokens_after,
        "tokens_saved": ctx.tokens_saved,
        "evicted_messages": ctx.evicted_messages,
        "reason": ctx.reason,
    })
}

/// The `fallback`/`model.switched` payload: the names involved.
///
/// Shared by the two name-pairing kinds.
fn switch_data(from: &str, to: &str) -> Value {
    serde_json::json!({
        "from": from,
        "to": to,
    })
}

/// The loop-detection payload, named by its detector.
///
/// The `detector` string is the discriminator inside the shared
/// `detection` kind.
fn loop_detected_data(ctx: &LoopDetectedContext) -> Value {
    serde_json::json!({
        "detector": "loop",
        "pattern": ctx.pattern,
        "repetitions": ctx.repetitions,
    })
}

/// The convergence-detection payload, named by its detector.
///
/// The `detector` string is the discriminator inside the shared
/// `detection` kind.
fn convergence_detected_data(ctx: &ConvergenceDetectedContext) -> Value {
    serde_json::json!({
        "detector": "convergence",
        "action": ctx.action,
    })
}

/// The `tool.call` payload: the dispatch identity.
///
/// `ToolPreContext` carries no input — the model's input object is the
/// `tool_call_received` callback's, which v1 does not map.
fn tool_call_data(ctx: &ToolPreContext) -> Value {
    serde_json::json!({
        "tool_call_id": ctx.tool_call_id,
        "tool": ctx.tool,
    })
}

/// The `tool.result` payload: the dispatch outcome, as metadata.
///
/// Duration converts through the shared saturating-millis helper; no
/// output text, ever.
fn tool_result_data(ctx: &ToolPostContext) -> Value {
    serde_json::json!({
        "tool_call_id": ctx.tool_call_id,
        "tool": ctx.tool,
        "result_hash": ctx.result_hash,
        "is_error": ctx.is_error,
        "duration_ms": millis(ctx.duration),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::trajectory::TrajectoryObserver;

    /// A unique temporary ledger directory for one test.
    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("events-ledger-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir created");
        dir
    }

    /// The parsed event lines of one observer's ledger.
    fn read_events(observer: &EventLedgerObserver) -> Vec<TrajectoryEvent> {
        observer.flush();
        let content = std::fs::read_to_string(observer.sink.dir().join(EVENTS_FILE))
            .expect("the events ledger exists after a flushed run");
        content
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("every ledger line parses as an event"))
            .collect()
    }

    /// A telemetry value for a constructed compaction context.
    fn compaction_telemetry() -> crate::compact::CompactTelemetry {
        let manager = crate::compact::ContextManager::new(std::sync::Arc::new(
            crate::compact::TruncatingCompactor::new(),
        ))
        .with_context_window(200_000);
        let pre = vec![crate::message::Message::user("a long conversation")];
        let post = vec![crate::message::Message::user("summary")];
        manager.build_telemetry(
            crate::compact::CompactReason::ThresholdExceeded,
            &pre,
            &post,
            Some("TruncatingCompactor"),
            std::time::Instant::now(),
        )
    }

    /// The exact `data` object each driven callback must produce.
    ///
    /// One entry per expected line, in call order — the ordering pin's
    /// interchange-table half; the serde labels `"ThresholdExceeded"`
    /// and `"EndTurn"` are the enums' default variant names.
    fn expected_event_datas() -> Vec<Value> {
        vec![
            serde_json::json!({}),
            serde_json::json!({"query": "fix the bug"}),
            serde_json::json!({"tool_call_id": "call_a", "tool": "echo"}),
            serde_json::json!({
                "tool_call_id": "call_a",
                "tool": "echo",
                "result_hash": 7,
                "is_error": false,
                "duration_ms": 3,
            }),
            serde_json::json!({
                "tokens_before": 100,
                "tokens_after": 40,
                "tokens_saved": 60,
                "evicted_messages": 1,
                "reason": "ThresholdExceeded",
            }),
            serde_json::json!({"from": "primary", "to": "backup"}),
            serde_json::json!({"from": "primary", "to": "backup"}),
            serde_json::json!({
                "detector": "loop",
                "pattern": "echo three times",
                "repetitions": 3,
            }),
            serde_json::json!({"detector": "convergence", "action": "stop"}),
            serde_json::json!({
                "success": true,
                "error": null,
                "duration_ms": 5,
                "input_tokens": 10,
                "output_tokens": 5,
                "stop_reason": "EndTurn",
            }),
            serde_json::json!({
                "success": true,
                "error": null,
                "total_turns": 1,
                "duration_ms": 5,
            }),
            serde_json::json!({}),
            serde_json::json!({
                "success": false,
                "error": null,
                "total_turns": 0,
                "duration_ms": 1,
            }),
        ]
    }

    #[test]
    fn event_records_land_in_ledger_in_order() {
        let observer = EventLedgerObserver::writing_to(temp_dir("ordered"));
        let session_id = uuid::Uuid::new_v4();
        let telemetry = compaction_telemetry();

        observer.on_run_start(&RunStartContext { session_id });
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "fix the bug".to_string(),
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "echo".to_string(),
            tool_call_id: "call_a".to_string(),
        });
        observer.on_tool_post(&ToolPostContext {
            tool_call_id: "call_a".to_string(),
            turn: 0,
            tool: "echo".to_string(),
            result_hash: Some(7),
            is_error: false,
            duration: std::time::Duration::from_millis(3),
            display_hint: None,
        });
        observer.on_compaction(&CompactedContext {
            tokens_before: 100,
            tokens_after: 40,
            tokens_saved: 60,
            reason: crate::compact::CompactReason::ThresholdExceeded,
            evicted_messages: 1,
            telemetry,
        });
        observer.on_fallback(&FallbackContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        observer.on_model_switched(&ModelSwitchedContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        observer.on_loop_detected(&LoopDetectedContext {
            pattern: "echo three times".to_string(),
            repetitions: 3,
        });
        observer.on_convergence_detected(&ConvergenceDetectedContext {
            action: "stop".to_string(),
        });
        observer.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 5,
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: crate::stream::StreamStopReason::EndTurn,
            context_tokens: 10,
            context_window: None,
        });
        observer.on_run_end(&RunEndContext::new(true, None, 1, 5));
        observer.on_run_start(&RunStartContext { session_id });
        observer.on_run_end(&RunEndContext::new(false, None, 0, 1));

        let events = read_events(&observer);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind.label()).collect();
        assert_eq!(
            kinds,
            vec![
                "run.started",
                "turn.started",
                "tool.call",
                "tool.result",
                "compaction",
                "fallback",
                "model.switched",
                "detection",
                "detection",
                "turn.ended",
                "run.ended",
                "run.started",
                "run.ended",
            ],
            "the ledger must carry exactly the mapped callbacks, in call order"
        );
        let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
        assert_eq!(
            seqs,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2],
            "each run's events must number from 1 in call order"
        );
        let run_ids: Vec<u64> = events.iter().map(|e| e.run_id).collect();
        assert_eq!(
            run_ids,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            "the second run must carry the next per-sink run id"
        );
        assert!(
            events
                .iter()
                .all(|e| e.session_id == session_id.to_string()),
            "every line must carry the run's session id — the cross-ledger bridge"
        );
        assert!(
            events.iter().all(|e| e.turn.is_none() || e.turn == Some(0)),
            "turn-scoped lines carry the context's turn and run-scoped lines carry null"
        );
        assert_eq!(
            events[1].data,
            serde_json::json!({"query": "fix the bug"}),
            "turn.started carries the query"
        );
        assert_eq!(
            events[2].data,
            serde_json::json!({"tool_call_id": "call_a", "tool": "echo"}),
            "tool.call carries the dispatch identity"
        );
        let datas: Vec<Value> = events.iter().map(|e| e.data.clone()).collect();
        assert_eq!(
            datas,
            expected_event_datas(),
            "every kind's data object must match the module's interchange table exactly — a \
             field rename or drop in any payload helper must fail here, not in a consumer"
        );
    }

    #[test]
    fn per_run_record_stays_backward_compatible() {
        let legacy = concat!(
            r#"{"session_id":"s-1","run_id":"r-1","outcome":"partial","#,
            r#""started_at":"2026-08-31T12:00:00Z","duration_ms":4200,"total_turns":1,"#,
            r#""token_summary":{"input_tokens":60,"output_tokens":170,"#,
            r#""cached_input_tokens":null,"cache_write_tokens":null,"reasoning_tokens":null},"#,
            r#""turns":[{"turn":0,"query":"fix the bug","response_text":"done","#,
            r#""tool_calls":[{"tool_call_id":"call_a","tool":"bash","ok":true,"duration_ms":12}],"#,
            r#""duration_ms":2100,"input_tokens":60,"output_tokens":170}]}"#
        );
        let parsed: crate::memory::trajectory::TrajectoryRecord =
            serde_json::from_str(legacy).expect("a pre-events ledger line still parses");
        assert_eq!(
            parsed.outcome,
            crate::memory::trajectory::TrajectoryOutcome::Partial,
            "the legacy outcome label must survive the round trip"
        );

        let dir = temp_dir("backward-compatible");
        let run_observer = TrajectoryObserver::writing_to(dir.clone());
        run_observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        run_observer.on_run_end(&RunEndContext::new(true, None, 0, 4));
        run_observer.flush();
        let records = run_observer.records();
        let expected = format!(
            "{}\n",
            serde_json::to_string(&records[0]).expect("the record serializes")
        );
        let written =
            std::fs::read_to_string(dir.join("trajectory.jsonl")).expect("the ledger exists");
        assert_eq!(
            written, expected,
            "the per-run ledger's bytes must be exactly one serialized record"
        );
        assert!(
            !dir.join(EVENTS_FILE).exists(),
            "the per-run observer must not write the event stream's file"
        );
    }

    #[test]
    fn events_outside_a_run_are_ignored() {
        let observer = EventLedgerObserver::writing_to(temp_dir("outside-run"));

        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "no run yet".to_string(),
        });
        observer.on_fallback(&FallbackContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        observer.on_run_end(&RunEndContext::new(true, None, 0, 0));
        observer.flush();
        assert!(
            !observer.sink.dir().join(EVENTS_FILE).exists(),
            "callbacks before any run start must write nothing"
        );

        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_run_end(&RunEndContext::new(true, None, 0, 0));
        let events = read_events(&observer);
        assert_eq!(
            events.len(),
            2,
            "the same observer must record normally once a run starts — the earlier absence was \
             the ignore path, not a broken sink"
        );
    }

    #[test]
    fn callbacks_after_a_run_end_are_ignored() {
        let observer = EventLedgerObserver::writing_to(temp_dir("after-run-end"));

        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_run_end(&RunEndContext::new(true, None, 0, 2));
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "stray callback".to_string(),
        });
        observer.on_run_end(&RunEndContext::new(false, None, 0, 0));

        let events = read_events(&observer);
        assert_eq!(
            events.len(),
            2,
            "a stray callback after the run's end must write nothing — the closed run's stream \
             is sealed, matching the documented contract"
        );

        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_run_end(&RunEndContext::new(true, None, 0, 1));
        let reopened = read_events(&observer);
        assert_eq!(
            reopened.len(),
            4,
            "the next run must record normally — the earlier absence was the sealed-run gate, \
             not a broken sink"
        );
    }

    #[test]
    fn a_configured_capture_limit_bounds_the_events_ledger_query() {
        let observer =
            EventLedgerObserver::writing_to(temp_dir("capture-limit")).with_capture_limit(10);
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "a query far past ten characters".to_string(),
        });

        let events = read_events(&observer);
        assert_eq!(events.len(), 2, "one run start and one turn start");
        assert_eq!(
            events[1].data,
            serde_json::json!({"query": "a query fa"}),
            "the query must truncate at the observer's configured capture limit — ten \
             characters here"
        );
    }

    #[test]
    fn an_unknown_future_kind_still_parses_with_its_envelope_intact() {
        let line = concat!(
            r#"{"seq":3,"ts":"2026-09-28T00:00:00Z","run_id":1,"session_id":"s-1","#,
            r#""turn":null,"kind":"gate.decision","data":{"rule":"budget"}}"#
        );
        let event: TrajectoryEvent = serde_json::from_str(line).expect(
            "an unknown kind from a newer release must not reject the whole line — the \
             envelope and data survive for an older reader",
        );
        assert_eq!(
            event.kind,
            TrajectoryEventKind::Unknown,
            "the unrecognized kind label must land in the fallback variant"
        );
        assert_eq!(event.seq, 3, "the envelope's other fields must survive");
        assert_eq!(
            event.data,
            serde_json::json!({"rule": "budget"}),
            "the kind's payload must survive for a reader that does know the kind"
        );
    }

    #[test]
    fn the_observer_never_emits_the_unknown_kind() {
        let observer = EventLedgerObserver::writing_to(temp_dir("never-unknown"));
        let telemetry = compaction_telemetry();

        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "fix the bug".to_string(),
        });
        observer.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "echo".to_string(),
            tool_call_id: "call_a".to_string(),
        });
        observer.on_tool_post(&ToolPostContext {
            tool_call_id: "call_a".to_string(),
            turn: 0,
            tool: "echo".to_string(),
            result_hash: None,
            is_error: false,
            duration: std::time::Duration::from_millis(1),
            display_hint: None,
        });
        observer.on_compaction(&CompactedContext {
            tokens_before: 10,
            tokens_after: 5,
            tokens_saved: 5,
            reason: crate::compact::CompactReason::ThresholdExceeded,
            evicted_messages: 0,
            telemetry,
        });
        observer.on_fallback(&FallbackContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        observer.on_model_switched(&ModelSwitchedContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        observer.on_loop_detected(&LoopDetectedContext {
            pattern: "echo".to_string(),
            repetitions: 2,
        });
        observer.on_convergence_detected(&ConvergenceDetectedContext {
            action: "stop".to_string(),
        });
        observer.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 1,
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: crate::stream::StreamStopReason::EndTurn,
            context_tokens: 10,
            context_window: None,
        });
        observer.on_run_end(&RunEndContext::new(true, None, 1, 1));

        let events = read_events(&observer);
        assert_eq!(events.len(), 11, "every mapped kind must have fired");
        assert!(
            events
                .iter()
                .all(|e| e.kind != TrajectoryEventKind::Unknown),
            "the fallback variant is deserialization-only — no callback may produce it"
        );
    }

    #[test]
    fn a_run_start_before_the_previous_run_ends_warns_and_restamps() {
        struct WarnCapture {
            messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }
        impl WarnCapture {
            fn messages_containing(
                messages: &std::sync::Mutex<Vec<String>>,
                needle: &str,
            ) -> usize {
                messages
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

        let observer = EventLedgerObserver::writing_to(temp_dir("restamp"));
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture = WarnCapture {
            messages: std::sync::Arc::clone(&messages),
        };
        tracing::subscriber::with_default(capture, || {
            observer.on_run_start(&RunStartContext {
                session_id: uuid::Uuid::new_v4(),
            });
            observer.on_turn_start(&TurnStartContext {
                turn: 0,
                query: "first run".to_string(),
            });
            observer.on_run_start(&RunStartContext {
                session_id: uuid::Uuid::new_v4(),
            });
            observer.on_turn_end(&TurnEndContext {
                turn: 0,
                success: true,
                error: None,
                duration_ms: 5,
                input_tokens: 1,
                output_tokens: 1,
                stop_reason: crate::stream::StreamStopReason::EndTurn,
                context_tokens: 10,
                context_window: None,
            });
            observer.on_run_end(&RunEndContext::new(true, None, 1, 5));
        });
        assert_eq!(
            WarnCapture::messages_containing(&messages, "before the previous run ended"),
            1,
            "exactly one warn must name the broken one-active-run contract"
        );

        let events = read_events(&observer);
        let stamped: Vec<(u64, u64, &str)> = events
            .iter()
            .map(|e| (e.run_id, e.seq, e.kind.label()))
            .collect();
        assert_eq!(
            stamped,
            vec![
                (1, 1, "run.started"),
                (1, 2, "turn.started"),
                (2, 1, "run.started"),
                (2, 2, "turn.ended"),
                (2, 3, "run.ended"),
            ],
            "the mid-flight restart must step the run id and restart the sequence, keeping every \
             already-written line"
        );
    }

    #[test]
    fn separate_ledger_files_do_not_share_lines() {
        let dir = temp_dir("disjoint");
        let run_observer = TrajectoryObserver::writing_to(dir.clone());
        let event_observer = EventLedgerObserver::writing_to(dir.clone());

        let session_id = uuid::Uuid::new_v4();
        run_observer.on_run_start(&RunStartContext { session_id });
        run_observer.on_run_end(&RunEndContext::new(true, None, 0, 3));
        event_observer.on_run_start(&RunStartContext { session_id });
        event_observer.on_run_end(&RunEndContext::new(true, None, 0, 3));
        run_observer.flush();
        event_observer.flush();

        let run_ledger = std::fs::read_to_string(dir.join("trajectory.jsonl"))
            .expect("the per-run ledger exists");
        let event_ledger =
            std::fs::read_to_string(dir.join(EVENTS_FILE)).expect("the events ledger exists");
        let run_lines: Vec<&str> = run_ledger.lines().filter(|line| !line.is_empty()).collect();
        let event_lines: Vec<&str> = event_ledger
            .lines()
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(run_lines.len(), 1, "one run, one per-run record");
        assert_eq!(event_lines.len(), 2, "one run, two mapped events");
        assert!(
            serde_json::from_str::<crate::memory::trajectory::TrajectoryRecord>(event_lines[0])
                .is_err(),
            "an event line must not parse as a per-run record"
        );
        assert!(
            serde_json::from_str::<TrajectoryEvent>(run_lines[0]).is_err(),
            "a per-run record must not parse as an event line"
        );
    }
}
