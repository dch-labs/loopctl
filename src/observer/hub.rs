//! The event hub — an observer-to-broadcast adapter.
//!
//! [`LoopObserver`] callbacks are synchronous and notify-only: the
//! engine calls them directly and never waits on a consumer. The hub
//! [`EventHub`] is the opt-in bridge for consumers that live outside
//! that call chain — a second task, a UI, a daemon — implementing the
//! observer trait by forwarding every callback as an owned
//! [`ObservedEvent`] on a `tokio` broadcast channel. A slow consumer is
//! lagged and dropped by the channel, never allowed to backpressure the
//! engine.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use tokio::sync::broadcast;

use super::LoopObserver;
use super::context::{
    AttemptResetContext, BudgetWarnContext, CompactedContext, CompactionFailedContext,
    ConvergenceDetectedContext, FallbackContext, GateDecisionContext, LoopDetectedContext,
    ModelSwitchedContext, PreCompactionContext, ResponseContext, RunEndContext, RunStartContext,
    StreamContext, StreamFailureContext, TextDeltaContext, ThinkingDeltaContext,
    ToolCallReceivedContext, ToolPostContext, ToolPreContext, TransportFallbackContext,
    TurnEndContext, TurnStartContext,
};

/// One observed lifecycle moment of a run, as forwarded by [`EventHub`].
///
/// One variant per [`LoopObserver`] callback, named as the callback
/// without its `on_` prefix and carrying an owned clone of that
/// callback's context. The enum is `#[non_exhaustive]`: a future
/// callback added to the trait widens this enum in a minor release, so
/// out-of-crate consumers match with a wildcard arm instead of
/// breaking.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LoopEvent {
    /// A run began.
    ///
    /// The first event of every `run()` call; its sequence number is
    /// always `1`, because the hub resets its counter here, and it is
    /// the first event stamped with the run's id.
    RunStart(RunStartContext),

    /// A run ended.
    ///
    /// The last event of a `run()` call, carrying the completion
    /// status, error text, turn count, and duration.
    RunEnd(RunEndContext),

    /// A turn began.
    ///
    /// One per model call the run makes.
    TurnStart(TurnStartContext),

    /// A turn ended.
    ///
    /// Carries the turn's stop reason and per-turn totals.
    TurnEnd(TurnEndContext),

    /// A streaming call succeeded.
    ///
    /// Fired when a provider stream completes without transport error.
    StreamSuccess(StreamContext),

    /// A streaming call failed.
    ///
    /// Fired on transport-level stream failures before any retry or
    /// fallback decision is applied.
    StreamFailure(StreamFailureContext),

    /// A model response completed.
    ///
    /// Carries the assembled response text and token usage.
    Response(ResponseContext),

    /// An incremental text chunk arrived.
    ///
    /// One per streamed text delta; ordering within a turn follows the
    /// stream.
    TextDelta(TextDeltaContext),

    /// An incremental reasoning chunk arrived.
    ///
    /// One per streamed thinking delta, ahead of the text deltas of the
    /// same stream.
    ThinkingDelta(ThinkingDeltaContext),

    /// A retried stream attempt discarded the failed attempt's events.
    ///
    /// One per retry, before the retried attempt's first delta — the cue
    /// to drop buffered text/thinking deltas of the same turn.
    AttemptReset(AttemptResetContext),

    /// A permission gate decided about a tool call.
    ///
    /// One per gated dispatch, when the deciding record becomes final:
    /// the verdict, argument digest, and rule provenance as one record.
    GateDecision(GateDecisionContext),

    /// A budget line's soft threshold was crossed.
    ///
    /// One per dimension per run, at the first pre-request check past
    /// the derived soft line: the dimension, the spend, and the hard
    /// limit it derives from.
    BudgetWarn(BudgetWarnContext),

    /// A tool call was accumulated, before dispatch.
    ///
    /// Fired when the model's tool call is parsed, before the tool
    /// runs.
    ToolCallReceived(ToolCallReceivedContext),

    /// A tool dispatch is about to run.
    ///
    /// The last event before the tool executes.
    ToolPre(ToolPreContext),

    /// A tool dispatch finished.
    ///
    /// Carries the tool's output or error and its duration.
    ToolPost(ToolPostContext),

    /// A compaction pass is about to run.
    ///
    /// Fired before the compactor is invoked, carrying the pre-pass
    /// telemetry.
    PreCompaction(PreCompactionContext),

    /// A compaction pass completed.
    ///
    /// Carries what was compacted and the token delta.
    Compaction(CompactedContext),

    /// A compaction pass was attempted and failed.
    ///
    /// Fired from the engine's compaction error arm — the compactor
    /// errored (`error` set), or its successful result still did not
    /// fit the window (`error` absent). The success
    /// [`Compaction`](LoopEvent::Compaction) event never follows.
    CompactionFailed(CompactionFailedContext),

    /// A model fallback occurred.
    ///
    /// Fired when the loop switches to a fallback model by policy.
    Fallback(FallbackContext),

    /// A transport fallback occurred.
    ///
    /// Fired when a streaming transport degrades to the non-streaming
    /// one.
    TransportFallback(TransportFallbackContext),

    /// The active model changed.
    ///
    /// Carries the from and to model names.
    ModelSwitched(ModelSwitchedContext),

    /// The loop detector fired.
    ///
    /// A repeated-tool-call loop was detected.
    LoopDetected(LoopDetectedContext),

    /// The convergence detector fired.
    ///
    /// Consecutive similar responses were detected.
    ConvergenceDetected(ConvergenceDetectedContext),
}

/// A [`LoopEvent`] stamped with the run it belongs to and that run's
/// sequence number.
///
/// The payload [`EventHub`] broadcasts. The run id distinguishes the
/// runs flowing through one hub — it moves up by one at each
/// [`RunStart`](LoopEvent::RunStart) — and the sequence number starts
/// at `1` for each run's first event and increases by one per event,
/// so a consumer can order what it received within a run and detect
/// gaps the channel's lag dropping introduced. The struct is
/// `#[non_exhaustive]` so later widening stays additive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ObservedEvent {
    /// The id of the run this event belongs to.
    ///
    /// Assigned by the hub: `1` for the first run observed through it,
    /// one higher for each later run. Under the hub's one-active-run
    /// contract every event of a run carries the same id; a second
    /// [`RunStart`](LoopEvent::RunStart) arriving before the previous
    /// run's [`RunEnd`](LoopEvent::RunEnd) is the visible signature of
    /// that contract being broken.
    pub run_id: u64,

    /// The event's per-run sequence number.
    ///
    /// `1` for the run's first event; strictly increasing within a
    /// run. Each run restarts at `1` — order runs by
    /// [`run_id`](Self::run_id) and events within a run by `seq`.
    pub seq: u64,

    /// The observed lifecycle moment.
    ///
    /// Which callback fired, with its full context.
    pub event: LoopEvent,
}

/// A built-in [`LoopObserver`] that forwards every callback onto a
/// broadcast channel.
///
/// The adapter for consumers that cannot sit inside the engine's
/// synchronous notify chain: register one hub — `Arc`-wrapped, through
/// the engine's observer registration path — and hand each consumer a
/// receiver from [`subscribe`](Self::subscribe). The engine is never
/// blocked: sends are best-effort, and a consumer whose receive ring
/// fills is lagged by the channel, which reports the dropped count as
/// [`broadcast::error::RecvError::Lagged`] on that consumer's next
/// receive. That error is the documented lag report — the hub itself
/// cannot see receiver state and deliberately logs nothing.
///
/// One hub instance per engine, and at most one active run per hub.
/// Do not share one hub across concurrently running engines: the
/// sequence counter is shared and resets at each
/// [`on_run_start`](LoopObserver::on_run_start), and a callback
/// carries no run-scoped signal the hub could attribute it by, so
/// interleaved events would stamp overlapping, non-monotonic
/// sequences. Use a separate hub for each concurrently running
/// engine; a shared hub's misuse is at least visible — the run id
/// restamps mid-flight with no intervening
/// [`RunEnd`](LoopEvent::RunEnd). The hub holds no per-session state,
/// so the observer trait's `reset` stays the default no-op. A hub
/// also hands every subscriber the full content of every callback —
/// user text, model output, tool inputs, and tool results (a result's
/// text rides the following turn's [`TurnStart`](LoopEvent::TurnStart)
/// query; its [`ToolPost`](LoopEvent::ToolPost) event carries hash
/// metadata only) — so keep it within one trust scope.
///
/// ```
/// use loopctl::observer::EventHub;
///
/// let hub = EventHub::new(64);
/// let receiver = hub.subscribe();
/// drop(receiver);
/// ```
#[derive(Debug)]
pub struct EventHub {
    /// The broadcast sender every callback forwards through.
    ///
    /// Cloned into each receiver's subscription; holding it alive for
    /// the hub's lifetime is what makes `subscribe` callable at any
    /// time.
    sender: broadcast::Sender<ObservedEvent>,

    /// The per-run sequence counter.
    ///
    /// Reset to zero in `on_run_start` and incremented once per
    /// forwarded event, so each run's events number from `1`.
    seq: AtomicU64,

    /// The run id stamped onto forwarded events.
    ///
    /// Incremented at each `on_run_start`, so the first run is `1`; the
    /// value `0` only reaches an event when a callback fires before
    /// any run start — direct trait misuse, since the engine always
    /// opens a run before firing callbacks.
    current_run_id: AtomicU64,
}

impl EventHub {
    /// Create a hub whose receive rings hold `capacity` events each.
    ///
    /// `capacity` bounds how far a consumer may fall behind before the
    /// channel starts dropping the oldest events on its behalf —
    /// smaller rings drop sooner and report larger lag counts. Choose
    /// it above the busiest turn's event count to keep an
    /// occasionally-slow consumer gapless. A capacity of `0` is floored
    /// to `1` rather than refused: construction is infallible, and a
    /// one-slot ring still delivers a live drain — only an unobserved
    /// consumer lags.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender,
            seq: AtomicU64::new(0),
            current_run_id: AtomicU64::new(0),
        }
    }

    /// Subscribe a new consumer to the hub's events.
    ///
    /// Each call returns an independent receiver with its own ring and
    /// its own lag bookkeeping; consumers never share drops. A receiver
    /// only ever sees events sent after its creation.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<ObservedEvent> {
        self.sender.subscribe()
    }

    /// Stamp the current run id and the next per-run sequence number
    /// onto `event` and forward it.
    ///
    /// Best-effort by design: the send is synchronous and its failure
    /// is discarded, because both failure modes are benign — no
    /// receiver is subscribed, or every receiver's ring is full and the
    /// channel has already dropped the oldest events for those
    /// receivers. Either way the engine must never wait on a consumer.
    fn publish(&self, event: LoopEvent) {
        let run_id = self.current_run_id.load(Ordering::Relaxed);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        drop(self.sender.send(ObservedEvent { run_id, seq, event }));
    }
}

impl LoopObserver for EventHub {
    /// Returns the hub's diagnostic name, `"event_hub"`.
    ///
    /// Identifies the adapter in engine diagnostics when the hub is the
    /// registered observer.
    fn name(&self) -> &'static str {
        "event_hub"
    }

    /// Forwards [`on_run_start`](LoopObserver::on_run_start) as a
    /// [`RunStart`](LoopEvent::RunStart) event.
    ///
    /// Also opens the run: the hub's run id moves to the next value
    /// and the per-run sequence counter resets, so this event carries
    /// `seq = 1`.
    fn on_run_start(&self, ctx: &RunStartContext) {
        self.current_run_id.fetch_add(1, Ordering::Relaxed);
        self.seq.store(0, Ordering::Relaxed);
        self.publish(LoopEvent::RunStart(ctx.clone()));
    }

    /// Forwards [`on_run_end`](LoopObserver::on_run_end) as a
    /// [`RunEnd`](LoopEvent::RunEnd) event.
    ///
    /// The last event of a run, whatever its completion status.
    fn on_run_end(&self, ctx: &RunEndContext) {
        self.publish(LoopEvent::RunEnd(ctx.clone()));
    }

    /// Forwards [`on_turn_start`](LoopObserver::on_turn_start) as a
    /// [`TurnStart`](LoopEvent::TurnStart) event.
    ///
    /// One per model call the run makes; on a turn that follows tool
    /// dispatch, the context's `query` carries the prior tool result's
    /// text — the channel by which tool output reaches subscribers.
    fn on_turn_start(&self, ctx: &TurnStartContext) {
        self.publish(LoopEvent::TurnStart(ctx.clone()));
    }

    /// Forwards [`on_turn_end`](LoopObserver::on_turn_end) as a
    /// [`TurnEnd`](LoopEvent::TurnEnd) event.
    ///
    /// Carries the turn's stop reason and per-turn totals.
    fn on_turn_end(&self, ctx: &TurnEndContext) {
        self.publish(LoopEvent::TurnEnd(ctx.clone()));
    }

    /// Forwards [`on_stream_success`](LoopObserver::on_stream_success)
    /// as a [`StreamSuccess`](LoopEvent::StreamSuccess) event.
    ///
    /// Fired when a provider stream completes without transport error.
    fn on_stream_success(&self, ctx: &StreamContext) {
        self.publish(LoopEvent::StreamSuccess(ctx.clone()));
    }

    /// Forwards [`on_stream_failure`](LoopObserver::on_stream_failure)
    /// as a [`StreamFailure`](LoopEvent::StreamFailure) event.
    ///
    /// Fired on transport-level stream failures, before any retry or
    /// fallback decision is applied.
    fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        self.publish(LoopEvent::StreamFailure(ctx.clone()));
    }

    /// Forwards [`on_response`](LoopObserver::on_response) as a
    /// [`Response`](LoopEvent::Response) event.
    ///
    /// Carries the assembled response text and token usage.
    fn on_response(&self, ctx: &ResponseContext) {
        self.publish(LoopEvent::Response(ctx.clone()));
    }

    /// Forwards [`on_text_delta`](LoopObserver::on_text_delta) as a
    /// [`TextDelta`](LoopEvent::TextDelta) event.
    ///
    /// One per streamed text delta; ordering within a turn follows the
    /// stream.
    fn on_text_delta(&self, ctx: &TextDeltaContext) {
        self.publish(LoopEvent::TextDelta(ctx.clone()));
    }

    /// Forwards [`on_thinking_delta`](LoopObserver::on_thinking_delta)
    /// as a [`ThinkingDelta`](LoopEvent::ThinkingDelta) event.
    ///
    /// One per streamed thinking delta, ahead of the text deltas of
    /// the same stream.
    fn on_thinking_delta(&self, ctx: &ThinkingDeltaContext) {
        self.publish(LoopEvent::ThinkingDelta(ctx.clone()));
    }

    /// Forwards [`on_attempt_reset`](LoopObserver::on_attempt_reset)
    /// as an [`AttemptReset`](LoopEvent::AttemptReset) event.
    ///
    /// One per retry, before the retried attempt's first delta.
    fn on_attempt_reset(&self, ctx: &AttemptResetContext) {
        self.publish(LoopEvent::AttemptReset(ctx.clone()));
    }

    /// Forwards
    /// [`on_gate_decision`](LoopObserver::on_gate_decision) as a
    /// [`GateDecision`](LoopEvent::GateDecision) event.
    ///
    /// One per gated dispatch; the record arrives by value.
    fn on_gate_decision(&self, ctx: &GateDecisionContext) {
        self.publish(LoopEvent::GateDecision(ctx.clone()));
    }

    /// Forwards
    /// [`on_budget_warn`](LoopObserver::on_budget_warn) as a
    /// [`BudgetWarn`](LoopEvent::BudgetWarn) event.
    ///
    /// One per dimension per run; the context is small and plain,
    /// so the forward clones only the event's own copy.
    fn on_budget_warn(&self, ctx: &BudgetWarnContext) {
        self.publish(LoopEvent::BudgetWarn(ctx.clone()));
    }

    /// Forwards
    /// [`on_tool_call_received`](LoopObserver::on_tool_call_received)
    /// as a [`ToolCallReceived`](LoopEvent::ToolCallReceived) event.
    ///
    /// Fired when the model's tool call is parsed, before the tool
    /// runs.
    fn on_tool_call_received(&self, ctx: &ToolCallReceivedContext) {
        self.publish(LoopEvent::ToolCallReceived(ctx.clone()));
    }

    /// Forwards [`on_tool_pre`](LoopObserver::on_tool_pre) as a
    /// [`ToolPre`](LoopEvent::ToolPre) event.
    ///
    /// The last event before the tool executes.
    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        self.publish(LoopEvent::ToolPre(ctx.clone()));
    }

    /// Forwards [`on_tool_post`](LoopObserver::on_tool_post) as a
    /// [`ToolPost`](LoopEvent::ToolPost) event.
    ///
    /// Carries the dispatch outcome — result hash, error flag,
    /// duration — once the tool call has finished. The tool's output
    /// text is never carried here; it rides the following turn's
    /// [`TurnStart`](LoopEvent::TurnStart) query instead.
    fn on_tool_post(&self, ctx: &ToolPostContext) {
        self.publish(LoopEvent::ToolPost(ctx.clone()));
    }

    /// Forwards [`on_pre_compaction`](LoopObserver::on_pre_compaction)
    /// as a [`PreCompaction`](LoopEvent::PreCompaction) event.
    ///
    /// Fired before a compaction pass runs, with the token counts that
    /// triggered it.
    fn on_pre_compaction(&self, ctx: &PreCompactionContext) {
        self.publish(LoopEvent::PreCompaction(ctx.clone()));
    }

    /// Forwards [`on_compaction`](LoopObserver::on_compaction) as a
    /// [`Compaction`](LoopEvent::Compaction) event.
    ///
    /// Fired after the pass, with the before and after token counts
    /// and what the compactor saved.
    fn on_compaction(&self, ctx: &CompactedContext) {
        self.publish(LoopEvent::Compaction(ctx.clone()));
    }

    /// Forwards
    /// [`on_compaction_failed`](LoopObserver::on_compaction_failed) as
    /// a [`CompactionFailed`](LoopEvent::CompactionFailed) event.
    ///
    /// Fired when a pass was attempted and died — the compactor's own
    /// error text rides `error` when there was one; `None` means the
    /// pass succeeded but its result still did not fit the window.
    fn on_compaction_failed(&self, ctx: &CompactionFailedContext) {
        self.publish(LoopEvent::CompactionFailed(ctx.clone()));
    }

    /// Forwards [`on_fallback`](LoopObserver::on_fallback) as a
    /// [`Fallback`](LoopEvent::Fallback) event.
    ///
    /// Names the model the run fell back from and to.
    fn on_fallback(&self, ctx: &FallbackContext) {
        self.publish(LoopEvent::Fallback(ctx.clone()));
    }

    /// Forwards
    /// [`on_transport_fallback`](LoopObserver::on_transport_fallback)
    /// as a [`TransportFallback`](LoopEvent::TransportFallback) event.
    ///
    /// Fired when a failed streaming turn is retried over the
    /// non-streaming transport.
    fn on_transport_fallback(&self, ctx: &TransportFallbackContext) {
        self.publish(LoopEvent::TransportFallback(ctx.clone()));
    }

    /// Forwards [`on_model_switched`](LoopObserver::on_model_switched)
    /// as a [`ModelSwitched`](LoopEvent::ModelSwitched) event.
    ///
    /// Names the model the run switched from and to.
    fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        self.publish(LoopEvent::ModelSwitched(ctx.clone()));
    }

    /// Forwards [`on_loop_detected`](LoopObserver::on_loop_detected) as
    /// a [`LoopDetected`](LoopEvent::LoopDetected) event.
    ///
    /// Carries the detected pattern and its repetition count.
    fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        self.publish(LoopEvent::LoopDetected(ctx.clone()));
    }

    /// Forwards
    /// [`on_convergence_detected`](LoopObserver::on_convergence_detected)
    /// as a [`ConvergenceDetected`](LoopEvent::ConvergenceDetected)
    /// event.
    ///
    /// Carries the action the detection triggered.
    fn on_convergence_detected(&self, ctx: &ConvergenceDetectedContext) {
        self.publish(LoopEvent::ConvergenceDetected(ctx.clone()));
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]
mod tests {
    use super::*;

    fn kind_of(event: &LoopEvent) -> &'static str {
        match event {
            LoopEvent::RunStart(_) => "run_start",
            LoopEvent::RunEnd(_) => "run_end",
            LoopEvent::TurnStart(_) => "turn_start",
            LoopEvent::TurnEnd(_) => "turn_end",
            LoopEvent::StreamSuccess(_) => "stream_success",
            LoopEvent::StreamFailure(_) => "stream_failure",
            LoopEvent::Response(_) => "response",
            LoopEvent::TextDelta(_) => "text_delta",
            LoopEvent::ThinkingDelta(_) => "thinking_delta",
            LoopEvent::AttemptReset(_) => "attempt_reset",
            LoopEvent::GateDecision(_) => "gate_decision",
            LoopEvent::BudgetWarn(_) => "budget_warn",
            LoopEvent::ToolCallReceived(_) => "tool_call_received",
            LoopEvent::ToolPre(_) => "tool_pre",
            LoopEvent::ToolPost(_) => "tool_post",
            LoopEvent::PreCompaction(_) => "pre_compaction",
            LoopEvent::Compaction(_) => "compaction",
            LoopEvent::CompactionFailed(_) => "compaction_failed",
            LoopEvent::Fallback(_) => "fallback",
            LoopEvent::TransportFallback(_) => "transport_fallback",
            LoopEvent::ModelSwitched(_) => "model_switched",
            LoopEvent::LoopDetected(_) => "loop_detected",
            LoopEvent::ConvergenceDetected(_) => "convergence_detected",
        }
    }

    /// The compaction telemetry the mapping pin's events carry.
    ///
    /// A minimal real pass over a two-message conversation, built the
    /// way the engine builds it, so the pin's `pre_compaction` and
    /// `compaction` events carry genuine telemetry shapes.
    fn mapping_pin_telemetry() -> crate::compact::CompactTelemetry {
        let manager = crate::compact::ContextManager::new(std::sync::Arc::new(
            crate::compact::TruncatingCompactor::new(),
        ))
        .with_context_window(200_000);
        let pre = vec![crate::message::Message::user(
            "a long conversation now summarized",
        )];
        let post = vec![crate::message::Message::user("summary")];
        manager.build_telemetry(
            crate::compact::CompactReason::ThresholdExceeded,
            &pre,
            &post,
            Some("TruncatingCompactor"),
            std::time::Instant::now(),
        )
    }

    /// Publish one gate-decision event through the hub.
    ///
    /// The mapping pin's gate entry: a denied deploy under a named
    /// middleware rule, the shape a gated dispatch produces.
    fn publish_gate_decision(hub: &EventHub) {
        hub.on_gate_decision(&GateDecisionContext {
            turn: 0,
            call_id: "call_gate".to_string(),
            decision: crate::tool::permission::GateDecision::new(
                "deploy",
                crate::tool::permission::GateVerdict::Deny,
                "deny-write-etc",
                crate::tool::permission::GateRuleSource::Middleware,
            ),
        });
    }

    #[test]
    fn every_callback_maps_to_its_event_kind_in_call_order() {
        let hub = EventHub::new(64);
        let mut receiver = hub.subscribe();
        let session_id = uuid::Uuid::new_v4();
        let telemetry = mapping_pin_telemetry();

        hub.on_run_start(&RunStartContext { session_id });
        hub.on_turn_start(&TurnStartContext {
            turn: 0,
            query: "fix the bug".to_string(),
        });
        hub.on_stream_success(&StreamContext {
            turn: 0,
            model: "test-model".to_string(),
            input_tokens: 10,
            output_tokens: 5,
        });
        hub.on_stream_failure(&StreamFailureContext {
            turn: 0,
            model: "test-model".to_string(),
            error: crate::error::LoopError::Cancelled,
        });
        hub.on_thinking_delta(&ThinkingDeltaContext {
            turn: 0,
            delta: "considering".to_string(),
        });
        hub.on_attempt_reset(&AttemptResetContext {
            turn: 0,
            attempt: 2,
        });
        hub.on_text_delta(&TextDeltaContext {
            turn: 0,
            delta: "working".to_string(),
        });
        hub.on_response(&ResponseContext {
            turn: 0,
            text: "working".to_string(),
            usage: None,
        });
        hub.on_tool_call_received(&ToolCallReceivedContext {
            turn: 0,
            tool: "echo".to_string(),
            call_id: "call_1".to_string(),
            input: serde_json::json!({}),
        });
        hub.on_tool_pre(&ToolPreContext {
            turn: 0,
            tool: "echo".to_string(),
            tool_call_id: "call_1".to_string(),
        });
        publish_gate_decision(&hub);
        hub.on_budget_warn(&BudgetWarnContext {
            dimension: crate::budget::BudgetDimension::Tokens,
            spent: 375,
            limit: 400,
        });
        hub.on_tool_post(&ToolPostContext {
            tool_call_id: "call_1".to_string(),
            turn: 0,
            tool: "echo".to_string(),
            result_hash: Some(7),
            is_error: false,
            duration: std::time::Duration::from_millis(1),
            display_hint: None,
        });

        hub.on_pre_compaction(&PreCompactionContext {
            reason: crate::compact::CompactReason::ThresholdExceeded,
            turn: 0,
            tokens_before: 100,
            context_window: 200_000,
            message_count: 2,
            session_id,
        });
        hub.on_compaction(&CompactedContext {
            tokens_before: 100,
            tokens_after: 40,
            tokens_saved: 60,
            reason: crate::compact::CompactReason::ThresholdExceeded,
            evicted_messages: 1,
            telemetry,
        });
        hub.on_compaction_failed(&CompactionFailedContext {
            reason: crate::compact::CompactReason::Emergency,
            turn: 0,
            tokens_before: 100,
            context_window: 200_000,
            error: Some("summarizer unavailable".to_string()),
        });
        hub.on_fallback(&FallbackContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        hub.on_transport_fallback(&TransportFallbackContext {
            turn: 0,
            stop_reason: crate::stream::StreamStopReason::EndTurn,
        });
        hub.on_model_switched(&ModelSwitchedContext {
            from: "primary".to_string(),
            to: "backup".to_string(),
        });
        hub.on_loop_detected(&LoopDetectedContext {
            pattern: "echo three times".to_string(),
            repetitions: 3,
        });
        hub.on_convergence_detected(&ConvergenceDetectedContext {
            action: "stop".to_string(),
        });
        hub.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 5,
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: crate::stream::StreamStopReason::EndTurn,
            context_tokens: 12,
            context_window: None,
        });
        hub.on_run_end(&RunEndContext::new(true, None, 1, 5));

        let expected = [
            "run_start",
            "turn_start",
            "stream_success",
            "stream_failure",
            "thinking_delta",
            "attempt_reset",
            "text_delta",
            "response",
            "tool_call_received",
            "tool_pre",
            "gate_decision",
            "budget_warn",
            "tool_post",
            "pre_compaction",
            "compaction",
            "compaction_failed",
            "fallback",
            "transport_fallback",
            "model_switched",
            "loop_detected",
            "convergence_detected",
            "turn_end",
            "run_end",
        ];
        drain_and_assert_kinds(&mut receiver, &expected, 23);
    }

    /// Drain a hub receiver and pin both the event-kind order and the
    /// sequence numbering of a full `count`-event run.
    fn drain_and_assert_kinds(
        receiver: &mut tokio::sync::broadcast::Receiver<ObservedEvent>,
        expected: &[&str],
        count: u64,
    ) {
        let mut actual = Vec::new();
        let mut seqs = Vec::new();
        while let Ok(observed) = receiver.try_recv() {
            actual.push(kind_of(&observed.event));
            seqs.push(observed.seq);
        }
        assert_eq!(
            actual, expected,
            "every observer callback must forward as its own event kind, in call order"
        );
        let expected_seqs: Vec<u64> = (1..=count).collect();
        assert_eq!(
            seqs, expected_seqs,
            "the run's events must number 1 through {count} in call order"
        );
    }

    #[test]
    fn the_hub_forwards_the_turn_end_context_size() {
        let hub = EventHub::new(16);
        let mut receiver = hub.subscribe();
        hub.on_turn_end(&TurnEndContext {
            turn: 0,
            success: true,
            error: None,
            duration_ms: 5,
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: crate::stream::StreamStopReason::EndTurn,
            context_tokens: 777,
            context_window: None,
        });
        let observed = receiver
            .try_recv()
            .expect("the turn-end event is receivable");
        match observed.event {
            LoopEvent::TurnEnd(ctx) => assert_eq!(
                ctx.context_tokens, 777,
                "the hub re-emits the turn-end context by value — the context-size field arrives with it"
            ),
            other => panic!("expected a turn-end event, got {}", kind_of(&other)),
        }
    }

    #[test]
    fn a_zero_capacity_hub_is_floored_to_one() {
        let hub = EventHub::new(0);
        let mut receiver = hub.subscribe();
        let session_id = uuid::Uuid::new_v4();

        hub.on_run_start(&RunStartContext { session_id });
        let first = receiver.try_recv().expect("the first event is receivable");
        assert_eq!(first.seq, 1, "a floored ring must hold the first event");

        hub.on_run_end(&RunEndContext::new(true, None, 0, 0));
        let second = receiver.try_recv().expect("the second event is receivable");
        assert_eq!(
            second.seq, 2,
            "a drained floored ring must hold the next event"
        );
    }

    #[test]
    fn each_run_on_a_shared_hub_carries_its_own_run_id_and_sequence() {
        let hub = EventHub::new(64);
        let mut receiver = hub.subscribe();
        let session_id = uuid::Uuid::new_v4();

        hub.on_run_start(&RunStartContext { session_id });
        hub.on_run_end(&RunEndContext::new(true, None, 0, 0));
        hub.on_run_start(&RunStartContext { session_id });
        hub.on_run_end(&RunEndContext::new(false, None, 1, 5));

        let mut observed = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            observed.push((event.run_id, event.seq));
        }
        assert_eq!(
            observed,
            vec![(1, 1), (1, 2), (2, 1), (2, 2)],
            "each run's events must carry that run's id and number its sequence from 1"
        );
    }

    #[test]
    fn a_mid_flight_run_start_restamps_subsequent_events_with_the_newer_run_id() {
        let hub = EventHub::new(64);
        let mut receiver = hub.subscribe();
        let session_id = uuid::Uuid::new_v4();

        hub.on_run_start(&RunStartContext { session_id });
        hub.on_run_start(&RunStartContext { session_id });
        hub.on_run_end(&RunEndContext::new(true, None, 0, 0));

        let mut observed = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            observed.push((event.run_id, event.seq, kind_of(&event.event)));
        }
        assert_eq!(
            observed,
            vec![(1, 1, "run_start"), (2, 1, "run_start"), (2, 2, "run_end"),],
            "a run starting before the previous one ended must restamp subsequent events with \
             the newer run id — the visible signature of the unsupported concurrent share"
        );
    }
}
