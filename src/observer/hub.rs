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
    CompactedContext, ConvergenceDetectedContext, FallbackContext, LoopDetectedContext,
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
    /// always `1`, because the hub resets its counter here.
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

/// A [`LoopEvent`] stamped with its per-run sequence number.
///
/// The payload [`EventHub`] broadcasts. The sequence number starts at
/// `1` for each run's [`RunStart`](LoopEvent::RunStart) event and
/// increases by one per event, so a consumer can order what it received
/// and detect gaps the channel's lag dropping introduced. The struct is
/// `#[non_exhaustive]` so later widening stays additive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ObservedEvent {
    /// The event's per-run sequence number.
    ///
    /// `1` for the run's first event; strictly increasing within a run.
    /// Two runs observed through one hub each restart at `1` —
    /// cross-run ordering is the consumer's concern, not the hub's.
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
/// One hub instance per engine; the hub holds no per-session state, so
/// the observer trait's `reset` stays the default no-op.
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

    /// Stamp the next sequence number onto `event` and forward it.
    ///
    /// Best-effort by design: the send is synchronous and its failure
    /// is discarded, because both failure modes are benign — no
    /// receiver is subscribed, or every receiver's ring is full and the
    /// channel has already dropped the oldest events for those
    /// receivers. Either way the engine must never wait on a consumer.
    fn publish(&self, event: LoopEvent) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        drop(self.sender.send(ObservedEvent { seq, event }));
    }
}

impl LoopObserver for EventHub {
    fn name(&self) -> &'static str {
        "event_hub"
    }

    fn on_run_start(&self, ctx: &RunStartContext) {
        self.seq.store(0, Ordering::Relaxed);
        self.publish(LoopEvent::RunStart(ctx.clone()));
    }

    fn on_run_end(&self, ctx: &RunEndContext) {
        self.publish(LoopEvent::RunEnd(ctx.clone()));
    }

    fn on_turn_start(&self, ctx: &TurnStartContext) {
        self.publish(LoopEvent::TurnStart(ctx.clone()));
    }

    fn on_turn_end(&self, ctx: &TurnEndContext) {
        self.publish(LoopEvent::TurnEnd(ctx.clone()));
    }

    fn on_stream_success(&self, ctx: &StreamContext) {
        self.publish(LoopEvent::StreamSuccess(ctx.clone()));
    }

    fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        self.publish(LoopEvent::StreamFailure(ctx.clone()));
    }

    fn on_response(&self, ctx: &ResponseContext) {
        self.publish(LoopEvent::Response(ctx.clone()));
    }

    fn on_text_delta(&self, ctx: &TextDeltaContext) {
        self.publish(LoopEvent::TextDelta(ctx.clone()));
    }

    fn on_thinking_delta(&self, ctx: &ThinkingDeltaContext) {
        self.publish(LoopEvent::ThinkingDelta(ctx.clone()));
    }

    fn on_tool_call_received(&self, ctx: &ToolCallReceivedContext) {
        self.publish(LoopEvent::ToolCallReceived(ctx.clone()));
    }

    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        self.publish(LoopEvent::ToolPre(ctx.clone()));
    }

    fn on_tool_post(&self, ctx: &ToolPostContext) {
        self.publish(LoopEvent::ToolPost(ctx.clone()));
    }

    fn on_pre_compaction(&self, ctx: &PreCompactionContext) {
        self.publish(LoopEvent::PreCompaction(ctx.clone()));
    }

    fn on_compaction(&self, ctx: &CompactedContext) {
        self.publish(LoopEvent::Compaction(ctx.clone()));
    }

    fn on_fallback(&self, ctx: &FallbackContext) {
        self.publish(LoopEvent::Fallback(ctx.clone()));
    }

    fn on_transport_fallback(&self, ctx: &TransportFallbackContext) {
        self.publish(LoopEvent::TransportFallback(ctx.clone()));
    }

    fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        self.publish(LoopEvent::ModelSwitched(ctx.clone()));
    }

    fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        self.publish(LoopEvent::LoopDetected(ctx.clone()));
    }

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
            LoopEvent::ToolCallReceived(_) => "tool_call_received",
            LoopEvent::ToolPre(_) => "tool_pre",
            LoopEvent::ToolPost(_) => "tool_post",
            LoopEvent::PreCompaction(_) => "pre_compaction",
            LoopEvent::Compaction(_) => "compaction",
            LoopEvent::Fallback(_) => "fallback",
            LoopEvent::TransportFallback(_) => "transport_fallback",
            LoopEvent::ModelSwitched(_) => "model_switched",
            LoopEvent::LoopDetected(_) => "loop_detected",
            LoopEvent::ConvergenceDetected(_) => "convergence_detected",
        }
    }

    #[test]
    fn every_callback_maps_to_its_event_kind_in_call_order() {
        let hub = EventHub::new(64);
        let mut receiver = hub.subscribe();
        let session_id = uuid::Uuid::new_v4();
        let manager = crate::compact::ContextManager::new(std::sync::Arc::new(
            crate::compact::TruncatingCompactor::new(),
        ))
        .with_context_window(200_000);
        let pre = vec![crate::message::Message::user(
            "a long conversation now summarized",
        )];
        let post = vec![crate::message::Message::user("summary")];
        let telemetry = manager.build_telemetry(
            crate::compact::CompactReason::ThresholdExceeded,
            &pre,
            &post,
            Some("TruncatingCompactor"),
            std::time::Instant::now(),
        );

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
        });
        hub.on_run_end(&RunEndContext::new(true, None, 1, 5));

        let expected = [
            "run_start",
            "turn_start",
            "stream_success",
            "stream_failure",
            "thinking_delta",
            "text_delta",
            "response",
            "tool_call_received",
            "tool_pre",
            "tool_post",
            "pre_compaction",
            "compaction",
            "fallback",
            "transport_fallback",
            "model_switched",
            "loop_detected",
            "convergence_detected",
            "turn_end",
            "run_end",
        ];
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
        let expected_seqs: Vec<u64> = (1..=19).collect();
        assert_eq!(
            seqs, expected_seqs,
            "the run's events must number 1 through 19 in call order"
        );
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
}
