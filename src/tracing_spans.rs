//! Engine span hierarchy: `agent.run` / `agent.turn` / `agent.tool`.
//!
//! The three headline nesting points of a [`BareLoop`](crate::engine::BareLoop)
//! run emit DEBUG spans on target `loopctl::engine`, so one run renders as a
//! readable trace tree in any `tracing-subscriber`-backed collector — and,
//! through an `OTLP` exporter, in Jaeger/Tempo/Grafana. Wire-facing
//! fields carry their **final** `OTel GenAI` names from emission
//! (`gen_ai.request.model`, `gen_ai.usage.*`, `error.type`, …): an exporter
//! maps span fields one-to-one and renames nothing.
//!
//! The tree, per run:
//!
//!```text
//! agent.run            gen_ai.operation.name=invoke_agent  gen_ai.conversation.id=…  gen_ai.request.model=…
//! ├─ agent.turn        turn=0  query="…"  → records outcome, duration_ms, usage on close
//! ├─ agent.tool        gen_ai.tool.name=…  gen_ai.tool.call.id=…  → records outcome, duration_ms, error.type
//! └─ …
//!```
//!
//! Tool spans describe one dispatch each; when the engine runs the tool
//! phase from its state-machine driver (a turn's tool calls dispatch after
//! the model call returned), a tool span parents to the enclosing run scope
//! rather than the finished turn — structure the handlers dictate, not a
//! lost parent. The pre-existing subsystem spans ([`structured`](crate::structured)'s
//! `structured.extract`, [`trajectory`](crate::memory::trajectory)'s
//! `trajectory.run`, `memory.extract`) are untouched children wherever they
//! nest.
//!
//! # Spans are entered only around `await` via
//! [`Instrument`](tracing::Instrument)
//!
//! Like [`structured.extract`](crate::structured), every span here attaches
//! to its region's future through [`tracing::Instrument`] — never an
//! `EnterGuard` held across an `await`, which
//! a work-stealing runtime could migrate between threads. Recording happens
//! on the span handle while the handle is alive, so a collector sees the
//! recorded `outcome` before the span closes.
//!
//! # Level and target guide (the `RUST_LOG` contract)
//!
//! | Target | Emissions | Levels |
//! |---|---|---|
//! | `loopctl::engine` | the span tree + signal events | spans/events DEBUG; lifecycle progression INFO; problems WARN (bounded per contract — once per condition) |
//! | `loopctl::metrics` | metric events | DEBUG only; never spans. The [`MetricsLayer`](crate::metrics::MetricsLayer) host filter needs `loopctl::metrics=debug` |
//! | `loopctl::compact`, `::structured`, `::trajectory`, `::memory`, `::pricing`, `::provider::forensics` | existing events plus `structured.extract` / `trajectory.run` spans | DEBUG; forensics WARN bounded |
//! | spans in general | all loopctl spans | DEBUG — always on; an `INFO` filter keeps an events-only log |
//!
//! Presets: `loopctl=warn` production · `loopctl=info` lifecycle events ·
//! `loopctl=debug` full trace trees, signal events, plus metric events.
//!
//! # Signal events (the engine's transparency layer)
//!
//! Beyond the span tree, the engine narrates its decisions as named
//! DEBUG events on target `loopctl::engine` — one per decision point,
//! each with a fixed field set. Under an `OTLP` exporter they attach as
//! events on the span in whose scope they fire (a compaction under its
//! run, a steering drain under the turn it steers), so one trace reads
//! as the whole story. The vocabulary is closed — every name below is
//! pinned: the names ride [`SIGNAL_EVENT_NAMES`], which the vocabulary
//! pin iterates, and a new signal is added to this table first:
//!
//! | Event | Fires at | Fields |
//! |---|---|---|
//! | `loopctl.compaction` | a compaction pass settles | `reason` (`prune`/`threshold`/`emergency`/`manual`), `tokens_before`, `tokens_after`, `outcome` (`applied`/`noop`/`deferred`/`failed`) |
//! | `loopctl.steering_drain` | accepted guidance joins the next turn | `count` |
//! | `loopctl.model_switch` | the active model changes | `from`, `to` |
//! | `loopctl.transport_fallback` | a failed stream is recovered on the non-streaming transport | `turn` |
//! | `loopctl.salvage` | a salvage retry is armed toward shrinking | `kind` (`overflow`/`length_cut`), `turn`, `target_tokens` |
//! | `loopctl.cancelled` | the run ends by cancellation | `turn` |
//! | `loopctl.hook_followup` | a hook extends a clean run by one turn | `origin` (hook-declared id, never prompt content) |
//!
//! None of them carry message content — names, counts, token figures,
//! and model ids only — so turning this layer on is not turning content
//! capture on. The `query` field below remains the only content-bearing
//! field in the crate, at the same DEBUG opt-in. Figures are honest
//! counts on one scale — the engine's provider-token figure, never the
//! compaction manager's raw counts: a pass where no compaction ran
//! reports the conversation unchanged (`after == before`), a failed
//! pass that measured reports its figures (the conversation's measured
//! size, and — when the pass produced a result the window refused —
//! that result's own size), zeros only when nothing was ever measured,
//! and an outcome is settled by the machine's own decision — `applied`
//! only when it adopts the compacted history, `noop` only when it
//! accepts the unchanged feed; a result that shrank nothing is refused
//! by the no-progress guard and narrates as `failed`. A salvage reports
//! the target the retry aims at — the compaction may still decline,
//! which the run's settlement narrates.
//!
//! The `query` field rides the turn span, which is DEBUG — a host filtering
//! at DEBUG opts into user-content capture, mirroring `OTel`'s content-capture
//! convention. Provider body capture remains [`forensics`](crate::provider)'s
//! explicit, separately-targeted feature.

use tracing::Level;

use crate::observer::{ToolPreContext, TurnStartContext};

/// Span field names — the emitter↔exporter contract.
///
/// An `OTLP` exporter maps span fields one-to-one onto `OTLP` attributes
/// and renames nothing, so these constants are the durable wire names:
/// engine instrumentation writes them, exporters read them, and a test
/// pins every spelling. Field values are strings unless noted;
/// `Empty`-initialized fields record only when the outcome is known.
pub mod fields {
    /// `OTel GenAI` operation name.
    ///
    /// Values: `"invoke_agent"` on `agent.run`, `"execute_tool"` on
    /// `agent.tool`. The discriminator `OTel GenAI` consumers switch on to
    /// decide which attribute set a span claims to carry.
    pub const OPERATION: &str = "gen_ai.operation.name";

    /// Run/session id as the `OTel GenAI` conversation id.
    ///
    /// The [`RunStartContext`](crate::observer::RunStartContext) uuid,
    /// rendered with `display`, shared by every span of one run so a
    /// collector can group runs across processes.
    pub const CONVERSATION_ID: &str = "gen_ai.conversation.id";

    /// Model requested for this operation, on both spans that carry it.
    ///
    /// On the run span this is the run's starting model — the client's
    /// configured name at `run()` entry; routing fallbacks are their
    /// own events, so the run-level field stays the *requested* name a
    /// host configures. On the turn span it is the model serving that
    /// turn (see [`record_turn_model`](super::record_turn_model)),
    /// which a fallback chain or a mid-run switch can change after the
    /// run opens.
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";

    /// Turn index within the run (loopctl-neutral extra).
    ///
    /// Zero-based, matching [`TurnStartContext`](crate::observer::TurnStartContext::turn);
    /// `OTel GenAI` has no equivalent attribute, so the neutral name keeps
    /// it out of the standard namespace.
    pub const TURN: &str = "turn";

    /// Turn input text — DEBUG-only, host-filterable.
    ///
    /// Rides the DEBUG turn span, recorded when the turn is assembled
    /// — steering injections included, the same string the turn-start
    /// observers receive — so a host filtering at DEBUG opts into
    /// user-content capture; at INFO and above the span (and the text)
    /// never materializes.
    pub const QUERY: &str = "query";

    /// Tool name as `OTel GenAI` spells it for `execute_tool` spans.
    ///
    /// The registry name the model called, pre-correction: what the
    /// model asked for, which is what a trace reader correlates with
    /// the assistant message.
    pub const TOOL_NAME: &str = "gen_ai.tool.name";

    /// Provider-assigned tool-call id.
    ///
    /// The same string the metric emission carries, letting a consumer
    /// join a trace span to its counter sample and to the conversation
    /// history entry.
    pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";

    /// `OTel GenAI` input-token usage, recorded at the turn's end.
    ///
    /// The provider-reported figure from [`TurnEndContext`](crate::observer::TurnEndContext::input_tokens),
    /// not the engine's local estimate — usage fields mean billing
    /// numbers here.
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

    /// `OTel GenAI` output-token usage, recorded at the turn's end.
    ///
    /// Provider-reported, paired with the input figure; recorded on
    /// every settled turn, zero where the provider reported no usage
    /// (a turn that failed before any report never reaches the
    /// record).
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

    /// `OTel GenAI` cached-read usage (the provider usage-detail fields' own names).
    ///
    /// Cache-read input tokens; recorded on turns whose provider
    /// reported cache detail, absent otherwise — absence distinguishes
    /// "provider said nothing" from "nothing was cached".
    pub const USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

    /// `OTel GenAI` cache-write usage, the cache-creation counterpart.
    ///
    /// Tokens written to the provider cache this turn; recorded under
    /// the same presence rule as [`USAGE_CACHE_READ_INPUT_TOKENS`].
    pub const USAGE_CACHE_WRITE_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";

    /// `OTel GenAI` reasoning-token count for the turn's response.
    ///
    /// The reasoning-token figure under its `GenAI` spelling; reasoning
    /// bills as output, and this separates the thinking part from the
    /// visible one.
    pub const RESPONSE_REASONING_TOKENS: &str = "gen_ai.response.reasoning_tokens";

    /// `OTel` error classification, recorded when the operation failed.
    ///
    /// A bounded class name from [`error_type`](super::error_type) (never the message
    /// text), matching `OTel`'s `error.type` semantics: a cardinality-
    /// limited reason code, safe as a metric label downstream.
    pub const ERROR_TYPE: &str = "error.type";

    /// Loopctl-neutral rollup key: `"ok"` or `"error"`.
    ///
    /// Recorded on every span that reaches an outcome, so dashboards
    /// group by result without parsing the `OTel` fields; a span without
    /// the field was interrupted before any outcome was knowable.
    pub const OUTCOME: &str = "outcome";

    /// Wall-clock milliseconds, loopctl-neutral extra.
    ///
    /// Recorded post-settle beside [`OUTCOME`]; span duration measured
    /// by the collector remains the primary timing — this is the
    /// engine's own view, matching the metric events' unit.
    pub const DURATION_MS: &str = "duration_ms";
}

/// Span name for the whole run: one per [`run`](crate::engine::core::Loop::run)
/// call that reaches the engine.
pub const AGENT_RUN_SPAN: &str = "agent.run";

/// Span name for one turn: the model call and its immediate aftermath.
///
/// One per turn of a run, from the engine's model-call step until its
/// results are committed; the turn index and input text ride the span
/// itself.
pub const TURN_SPAN: &str = "agent.turn";

/// Span name for one settled tool dispatch.
///
/// One per dispatch attempt reaching a settled result — soft errors
/// included — mirroring the `loopctl.tools.calls` counter event so the
/// two signals describe the same fact.
pub const TOOL_SPAN: &str = "agent.tool";

/// The target every engine span carries.
///
/// One target for the hierarchy so `RUST_LOG` can raise or drop trace
/// spans without touching the metric events (`loopctl::metrics`) or
/// any subsystem's own emissions.
pub const TARGET: &str = "loopctl::engine";

/// Every signal-event name the engine emits, in dictionary order.
///
/// The single list the vocabulary pin iterates: the test fires each
/// helper and asserts its event carries one of these names, and
/// rejects an emission the list does not declare. The helpers spell
/// the names in their own emissions, so a helper that drifts from
/// this list fails the vocabulary pin.
pub const SIGNAL_EVENT_NAMES: [&str; 7] = [
    "loopctl.cancelled",
    "loopctl.compaction",
    "loopctl.hook_followup",
    "loopctl.model_switch",
    "loopctl.salvage",
    "loopctl.steering_drain",
    "loopctl.transport_fallback",
];

/// The settled outcome of one compaction pass, as the
/// `loopctl.compaction` event narrates it.
///
/// A closed vocabulary: the engine constructs these at its call
/// sites, so a misspelled outcome cannot reach the wire the way a
/// free-form string literal could. The spellings are the wire
/// contract an `OTLP` consumer groups on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactionPassOutcome {
    /// Compacted history was adopted; the conversation shrank.
    ///
    /// The pass rewrote the history and the machine took the result;
    /// `tokens_after` is the adopted conversation's measured size.
    /// Emitted only after the machine accepted the result — a
    /// shrank-nothing compacted history the no-progress guard refuses
    /// narrates as [`Failed`](Self::Failed).
    Applied,

    /// The pass changed nothing and the conversation stands.
    ///
    /// No compactor ran, a hook vetoed the pass, or the manager
    /// reported no action; both figures are measurements of the same
    /// unchanged conversation. Emitted only when the machine accepted
    /// the unchanged feed — a no-progress stall the guard refuses
    /// narrates as [`Failed`](Self::Failed).
    Noop,

    /// A threshold pass could not shrink below the line; the turn
    /// serves uncompacted.
    ///
    /// The compactor errored while the payload still fit the window,
    /// or the pass could not shrink below the threshold; the next turn
    /// retries the same compactor.
    Deferred,

    /// The pass errored into the run.
    ///
    /// Either the compactor failed, its result still did not fit, or
    /// the machine's no-progress guard refused a pass that shrank
    /// nothing — compacted or unchanged alike; the run ends. A pass
    /// that measured reports its figures — the conversation's measured
    /// size, and the rejected result's own when one was produced.
    Failed,
}

impl CompactionPassOutcome {
    /// The wire spelling the `loopctl.compaction` event carries.
    ///
    /// One word per outcome, stable across releases — a consumer's
    /// dashboard groups on it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Noop => "noop",
            Self::Deferred => "deferred",
            Self::Failed => "failed",
        }
    }
}

/// Which rejection a salvage retry answers, as the `loopctl.salvage`
/// event narrates it.
///
/// A closed vocabulary mirroring the engine's two salvage arms; the
/// engine constructs these at its call sites, so a misspelled kind
/// cannot reach the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SalvageKind {
    /// The provider reported the window: an oversized-prompt rejection.
    ///
    /// The provider's own error named the limit; the retry compacts
    /// toward a target strictly under the refused payload.
    Overflow,

    /// The response ended at the output limit with no visible text.
    ///
    /// The wire cannot report this case; the engine reads the shape
    /// and remedies it as an over-full context.
    LengthCut,
}

impl SalvageKind {
    /// The wire spelling the `loopctl.salvage` event carries.
    ///
    /// One word per arm, stable across releases — a consumer's
    /// dashboard groups on it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Overflow => "overflow",
            Self::LengthCut => "length_cut",
        }
    }
}

/// Creates the `agent.run` span for one engine run.
///
/// Built at the driver before its future starts polling, so the run's
/// first emission is already a child. Callers attach it with
/// [`Instrument`](tracing::Instrument) around the run future and record
/// [`fields::OUTCOME`] / [`fields::DURATION_MS`] / [`fields::ERROR_TYPE`]
/// on the settled tail; the conversation id is the session uuid every
/// [`RunStartContext`](crate::observer::RunStartContext) carries.
///
/// Field spellings here mirror the [`fields`] constants literally —
/// `tracing`'s macro syntax takes names, not paths — and the span
/// test suite pins every emission against the constants, so the two
/// can never drift silently.
///
/// Everything the engine records after construction is declared
/// [`Empty`](tracing::field::Empty) at construction: `Span::record`
/// resolves names against the constructed field set, so a field
/// that was never declared is silently dropped rather than added.
#[must_use = "the Span must be entered or recorded to take effect"]
pub fn agent_run_span(session_id: uuid::Uuid, model: &str) -> tracing::Span {
    tracing::span!(
        target: TARGET,
        Level::DEBUG,
        AGENT_RUN_SPAN,
        "gen_ai.operation.name" = "invoke_agent",
        "gen_ai.conversation.id" = %session_id,
        "gen_ai.request.model" = model,
        "outcome" = tracing::field::Empty,
        "duration_ms" = tracing::field::Empty,
        "error.type" = tracing::field::Empty,
    )
}

/// Creates the `agent.turn` span for one model-call turn.
///
/// Carries the turn index and — because the span is DEBUG, an explicit
/// host opt-in for content capture — the input text, which opens unset
/// and arrives through [`record_turn_query`] once the turn is assembled.
/// It opens unset rather than empty for the same reason every other
/// late-filled field here does: a recorded `tracing` field is *added* to
/// the `OTel` attribute list, not swapped in, so an opening value ships
/// alongside the recorded one and the backend receives the key twice.
/// The engine records [`fields::OUTCOME`], duration, the usage fields,
/// and — once the request is assembled — the serving model (a fallback
/// chain can change it mid-run, so the run span's model is only the
/// starting one) at the turn's boundaries before the handle closes.
#[must_use = "the Span must be entered or recorded to take effect"]
pub fn turn_span(ctx: &TurnStartContext) -> tracing::Span {
    tracing::span!(
        target: TARGET,
        Level::DEBUG,
        TURN_SPAN,
        "turn" = ctx.turn,
        "query" = tracing::field::Empty,
        "gen_ai.request.model" = tracing::field::Empty,
        "outcome" = tracing::field::Empty,
        "duration_ms" = tracing::field::Empty,
        "gen_ai.usage.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.output_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_read.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_creation.input_tokens" = tracing::field::Empty,
        "gen_ai.response.reasoning_tokens" = tracing::field::Empty,
    )
}

/// Creates the `agent.tool` span for one dispatch attempt.
///
/// Carries the `GenAI` `execute_tool` shape — name and provider-assigned
/// call id — with [`fields::OUTCOME`], [`fields::DURATION_MS`], and
/// [`fields::ERROR_TYPE`] declared empty for the post-settle record.
#[must_use = "the Span must be entered or recorded to take effect"]
pub fn tool_span(ctx: &ToolPreContext) -> tracing::Span {
    tracing::span!(
        target: TARGET,
        Level::DEBUG,
        TOOL_SPAN,
        "gen_ai.operation.name" = "execute_tool",
        "gen_ai.tool.name" = ctx.tool.as_str(),
        "gen_ai.tool.call.id" = ctx.tool_call_id.as_str(),
        "outcome" = tracing::field::Empty,
        "duration_ms" = tracing::field::Empty,
        "error.type" = tracing::field::Empty,
    )
}

/// Records the turn's assembled input onto the current turn span.
///
/// The span opens with `query` unset because the real text is only
/// known once the turn is assembled — steering injections included —
/// and the engine computes that string exactly once, for the turn-start
/// observers; this records it onto the span from the same value, so the
/// two consumers can never disagree about what the turn's input was.
/// Unset is the opening shape, not an empty string: a `tracing` record
/// adds an attribute to an `OTel` span instead of replacing one, so a
/// span that opened with `""` would export both the empty opener and the
/// recorded text under the same key, and a backend that keeps the first
/// occurrence it sees would show an empty input for every turn. A turn
/// that ends before this point exports no `query` at all, which says
/// *never assembled* rather than *assembled as nothing*.
/// The same name check as [`record_turn_end`] keeps stray callers from
/// writing turn text onto someone else's span, and on a host that
/// filters the span away the record is a no-op.
pub fn record_turn_query(query: &str) {
    let span = tracing::Span::current();
    if span.metadata().is_none_or(|meta| meta.name() != TURN_SPAN) {
        return;
    }
    span.record(fields::QUERY, query);
}

/// Record the model serving this turn onto its span.
///
/// The run span already names the run's starting model; a fallback
/// chain or a mid-run [`switch_model`](crate::engine::BareLoop::switch_model)
/// can change it, and the turn span is where a per-request reader
/// looks. Called from the turn's request-options assembly — inside
/// the turn span's scope — so [`Span::current`](tracing::Span::current)
/// is that span; the name-and-target guard makes a stray call a
/// no-op rather than a field on the wrong span — a host's own
/// `agent.turn`-named span on another target is never written.
pub fn record_turn_model(model: &str) {
    let span = tracing::Span::current();
    if span
        .metadata()
        .is_none_or(|meta| meta.name() != TURN_SPAN || meta.target() != TARGET)
    {
        return;
    }
    span.record(fields::REQUEST_MODEL, model);
}

/// Narrate one compaction pass settling.
///
/// Fired once per pass with the before/after context figures and the
/// outcome the machine adopted — [`Applied`](CompactionPassOutcome::Applied)
/// when compacted history was taken (the machine's adoption, not the
/// compactor's success, decides this), [`Noop`](CompactionPassOutcome::Noop)
/// when the pass kept the conversation as it stood,
/// [`Deferred`](CompactionPassOutcome::Deferred) when a threshold pass
/// could not shrink below the line and the turn is served uncompacted,
/// [`Failed`](CompactionPassOutcome::Failed) when the pass errored into
/// the run. Token figures are context counts on the engine's
/// provider-token scale, not usage and not the compaction manager's
/// raw counts: a pass where no compaction ran reports the conversation
/// unchanged (`after == before`), a failed pass that measured reports
/// its figures — the conversation's measured size and, when the pass
/// produced a result the window refused, that result's own — and zeros
/// only when nothing was ever measured.
pub fn event_compaction(
    reason: crate::compact::types::CompactReason,
    tokens_before: u64,
    tokens_after: u64,
    outcome: CompactionPassOutcome,
) {
    let reason = match reason {
        crate::compact::types::CompactReason::Prune => "prune",
        crate::compact::types::CompactReason::ThresholdExceeded => "threshold",
        crate::compact::types::CompactReason::Emergency => "emergency",
        crate::compact::types::CompactReason::Manual => "manual",
    };
    let outcome = outcome.as_str();
    tracing::debug!(
        target: TARGET,
        reason,
        tokens_before,
        tokens_after,
        outcome,
        "loopctl.compaction"
    );
}

/// Narrate accepted guidance joining the next turn.
///
/// Fires at the turn boundary where the steering slot drains: every
/// message accepted since the last boundary enters the pending set
/// ahead of this turn's assembly. The count is messages, not sources —
/// three pushes from one host are three.
pub fn event_steering_drain(count: usize) {
    tracing::debug!(target: TARGET, count, "loopctl.steering_drain");
}

/// Narrate the active model changing.
///
/// Fires after the client accepted the switch — `from` is the model the
/// run was serving, `to` the one it serves now. Model ids only; request
/// content never rides this event.
pub fn event_model_switch(from: &str, to: &str) {
    tracing::debug!(target: TARGET, from, to, "loopctl.model_switch");
}

/// Narrate a failed stream recovered on the non-streaming transport.
///
/// Fires when the turn is served by the `create_message` recovery after
/// streaming exhausted its retry ceiling — the same condition the
/// run's `transport_fallbacks` counter records, at the turn and span
/// a reader is already looking at.
pub fn event_transport_fallback(turn: usize) {
    tracing::debug!(target: TARGET, turn, "loopctl.transport_fallback");
}

/// Narrate an over-window turn's salvage retry being armed.
///
/// Fires when the engine decides to retry a turn the provider refused
/// (or ended at the output cap) by compacting toward `target_tokens`
/// and reissuing — [`Overflow`](SalvageKind::Overflow) for the
/// provider-reported window and [`LengthCut`](SalvageKind::LengthCut)
/// for the silent output-cap shape. Armed, not completed: the
/// compaction may still decline or fail, which the run's own
/// settlement narrates; this event is the decision to retry.
pub fn event_salvage(kind: SalvageKind, turn: usize, target_tokens: u64) {
    let kind = kind.as_str();
    tracing::debug!(
        target: TARGET,
        kind,
        turn,
        target_tokens,
        "loopctl.salvage"
    );
}

/// Narrate the run ending by cancellation.
///
/// Fired once from the run's finalize funnel — whichever arm the cancel
/// won (before the first turn, mid-stream, mid-tool), `turn` is the
/// run's settled turn count at that moment, so a cancel before any turn
/// is `0`. The run span's `outcome` settlement remains the authoritative
/// end state; this event is the timing of the interrupt itself.
pub fn event_cancelled(turn: usize) {
    tracing::debug!(target: TARGET, turn, "loopctl.cancelled");
}

/// Narrate a hook extending a clean run by one turn.
///
/// Fires when a follow-up is accepted — `origin` is the hook-declared
/// id (the `"hook:<name>"` convention), never the injected prompt,
/// which is user content and stays out of telemetry entirely.
pub fn event_hook_followup(origin: &str) {
    tracing::debug!(target: TARGET, origin, "loopctl.hook_followup");
}

/// The settled figures one turn-span settlement records.
///
/// Built wherever a turn ends — the engine's notification funnel — and
/// handed to [`record_turn_end`] (or [`record_turn_end_on`] for a span
/// the driver holds open past the model phase), so the settlement
/// sites and the span writers agree on one shape instead of seven
/// positional figures.
#[derive(Debug, Clone, Copy)]
pub struct TurnSettlement {
    /// Whether the turn reached its intended completion.
    ///
    /// `false` for a hard failure — cancellation, dispatch error, a
    /// recovery-exhausted tool; soft tool errors the model will see
    /// stay `true`.
    pub success: bool,

    /// Wall-clock duration of the settled attempt, in milliseconds.
    ///
    /// Measured by the engine from the attempt's start — the model call
    /// plus the tool phase for a tool-bearing turn — matching the
    /// `duration_ms` metric events' unit.
    pub duration_ms: u64,

    /// Provider-reported input tokens for the attempt.
    ///
    /// The billing figure from the turn-end notification, never the
    /// engine's local estimate; recorded verbatim onto the span.
    pub input_tokens: u64,

    /// Provider-reported output tokens for the attempt.
    ///
    /// The generated-token billing figure, paired with
    /// [`input_tokens`](Self::input_tokens) on every settled turn.
    pub output_tokens: u64,

    /// Cache-read detail, zero where unreported.
    ///
    /// Presence-not-zero: a zero means the provider said nothing about
    /// cache reads, so the span omits the field entirely.
    pub cached_input_tokens: u64,

    /// Cache-write detail, zero where unreported.
    ///
    /// Same presence rule as
    /// [`cached_input_tokens`](Self::cached_input_tokens) — the
    /// cache-creation share, recorded only where reported.
    pub cache_write_tokens: u64,

    /// Reasoning detail inside output, zero where unreported.
    ///
    /// The thinking subset of the output count; recorded only where the
    /// provider reported it.
    pub reasoning_tokens: u64,
}

/// Records a settled turn's outcome and usage on the current span.
///
/// The turn-end boundary knows the figures; the span lives wherever the
/// turn was instrumented. A name check keeps this honest: only an
/// `agent.turn` span is ever written, so the same call from a scope that
/// runs without a turn span (a deferred tool-phase event, an
/// uninstrumented caller) records nothing instead of writing turn facts
/// onto someone else's span. Token figures ride their `OTel GenAI` names
/// directly — the exporter renames nothing.
///
/// The cache and reasoning detail fields follow presence-not-zero:
/// recorded only when the provider reported them (the engine reads
/// zero exactly where no report arrived, so a zero here would say
/// "nothing cached" when the truth is "nothing said"). The headline
/// input and output figures record on every settled turn.
pub fn record_turn_end(settlement: TurnSettlement) {
    record_turn_end_on(&tracing::Span::current(), settlement);
}

/// Records a settled turn's outcome and usage on an explicit span.
///
/// The span-holding twin of [`record_turn_end`]: the driver keeps a
/// tool-bearing turn's span open past the model phase and settles it
/// from the turn-end notification after the tool phase ran, so the
/// record names the span it belongs to instead of relying on whichever
/// span is current where the notification fires. The same name check
/// applies — a foreign span records nothing.
pub fn record_turn_end_on(span: &tracing::Span, settlement: TurnSettlement) {
    let TurnSettlement {
        success,
        duration_ms,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        reasoning_tokens,
    } = settlement;
    if span.metadata().is_none_or(|meta| meta.name() != TURN_SPAN) {
        return;
    }
    span.record(fields::OUTCOME, if success { "ok" } else { "error" });
    span.record(fields::DURATION_MS, duration_ms);
    span.record(fields::USAGE_INPUT_TOKENS, input_tokens);
    span.record(fields::USAGE_OUTPUT_TOKENS, output_tokens);
    if cached_input_tokens != 0 {
        span.record(fields::USAGE_CACHE_READ_INPUT_TOKENS, cached_input_tokens);
    }
    if cache_write_tokens != 0 {
        span.record(fields::USAGE_CACHE_WRITE_INPUT_TOKENS, cache_write_tokens);
    }
    if reasoning_tokens != 0 {
        span.record(fields::RESPONSE_REASONING_TOKENS, reasoning_tokens);
    }
}

/// Records a settled run's outcome on the current `agent.run` span.
///
/// The run-span twin of [`record_turn_end`]: every `run()` exit path —
/// clean completion, error, cancellation, a failed manager reset before
/// the first turn — settles the span with its outcome, the
/// engine-measured duration, and the bounded `error.type` class, so a
/// collector never sees a failed run close as an outcome-less span. The
/// name check keeps stray callers from writing run facts onto someone
/// else's span.
pub fn record_run_end(success: bool, duration_ms: u64, error: Option<&crate::error::LoopError>) {
    let span = tracing::Span::current();
    if span
        .metadata()
        .is_none_or(|meta| meta.name() != AGENT_RUN_SPAN)
    {
        return;
    }
    span.record(fields::OUTCOME, if success { "ok" } else { "error" });
    span.record(fields::DURATION_MS, duration_ms);
    if let Some(err) = error {
        span.record(fields::ERROR_TYPE, error_type(err));
    }
}

/// Classifies a [`LoopError`](crate::error::LoopError) as an `OTel`
/// `error.type` name.
///
/// A bounded set of snake-case class names — never the error message,
/// which carries unbounded user content. Variants the classifier does
/// not name collapse to `"error"`: today that is `FallbackExhausted`,
/// `CompactionFailed`, `CompactionStalled`, `PhaseFailed`, `Reflection`,
/// `UserInputRequired`, `ToolLimitReached`, `ToolRecoveryExhausted`,
/// `Internal`, and `RateLimitEscalation`, plus any variant added later —
/// adding one never changes this function's compilation, and the
/// boundedness `OTel` requires is a property of the code, not of
/// today's enum.
#[must_use]
pub fn error_type(err: &crate::error::LoopError) -> &'static str {
    use crate::error::LoopError as E;
    match err {
        E::Cancelled => "cancelled",
        E::Api(_) => "api",
        E::ToolNotFound { .. } => "tool_not_found",
        E::ToolExecution { .. } => "tool_execution",
        E::InvalidInput(_) => "invalid_input",
        E::MaxTurnsExceeded { .. } => "max_turns_exceeded",
        E::ContextExceeded { .. } => "context_exceeded",
        E::BudgetExhausted { .. } => "budget_exhausted",
        E::LoopDetected { .. } => "loop_detected",
        E::LockPoisoned { .. } => "lock_poisoned",
        E::StreamError(_) => "stream_error",
        E::Config(_) => "config",
        E::Memory(_) => "memory",
        _ => "error",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc)]

    use super::error_type;
    use crate::error::LoopError;

    /// Every named class plus the wildcard: `error.type` is a wire
    /// vocabulary — a consumer's dashboard groups by it — so an arm
    /// drifting here drifts silently on every exporter. Constructed
    /// from real variants, never strings, so a payload change fails
    /// the compile rather than the assertion.
    #[test]
    fn error_type_names_every_bounded_class() {
        let cases: Vec<(LoopError, &str)> = vec![
            (LoopError::Cancelled, "cancelled"),
            (LoopError::Api("boom".to_string()), "api"),
            (
                LoopError::ToolNotFound {
                    tool: "nope".to_string(),
                    available: "echo".to_string(),
                },
                "tool_not_found",
            ),
            (
                LoopError::ToolExecution {
                    tool: "echo".to_string(),
                    message: "kaput".to_string(),
                },
                "tool_execution",
            ),
            (LoopError::InvalidInput("bad".to_string()), "invalid_input"),
            (LoopError::MaxTurnsExceeded { max: 3 }, "max_turns_exceeded"),
            (
                LoopError::ContextExceeded { used: 10, limit: 8 },
                "context_exceeded",
            ),
            (
                LoopError::BudgetExhausted {
                    dimension: crate::budget::BudgetDimension::Tokens,
                    spent: 10,
                    limit: 8,
                },
                "budget_exhausted",
            ),
            (
                LoopError::LoopDetected {
                    message: "spinning".to_string(),
                },
                "loop_detected",
            ),
            (
                LoopError::LockPoisoned {
                    what: "store".to_string(),
                },
                "lock_poisoned",
            ),
            (LoopError::StreamError("reset".to_string()), "stream_error"),
            (LoopError::Config("nope".to_string()), "config"),
            (LoopError::Memory("gone".to_string()), "memory"),
        ];
        for (err, want) in cases {
            assert_eq!(error_type(&err), want, "for {err:?}");
        }
    }

    /// Variants the classifier does not name collapse to `"error"` —
    /// boundedness is a property of the match, not of today's enum, and
    /// adding a variant must never widen the exported vocabulary. Every
    /// current unnamed variant is constructed here so a new arm cannot
    /// land silently either: naming one in the classifier without
    /// moving it out of this list fails the test.
    #[test]
    fn unnamed_variants_collapse_to_the_bounded_fallback() {
        for err in [
            LoopError::FallbackExhausted,
            LoopError::CompactionFailed {
                used: 900,
                limit: 800,
                cause: "summarizer refused".to_string(),
            },
            LoopError::CompactionStalled { used: 900 },
            LoopError::PhaseFailed {
                phase: "reflection".to_string(),
                message: "phase blew up".to_string(),
            },
            LoopError::Reflection("unhandled".to_string()),
            LoopError::UserInputRequired {
                message: "converged".to_string(),
            },
            LoopError::ToolLimitReached {
                message: "session limit".to_string(),
            },
            LoopError::ToolRecoveryExhausted {
                tool: "echo".to_string(),
                attempts: 3,
            },
            LoopError::Internal("unspecified".to_string()),
            LoopError::RateLimitEscalation {
                attempts: 4,
                retry_after: None,
            },
        ] {
            assert_eq!(error_type(&err), "error", "for {err:?}");
        }
    }
}
