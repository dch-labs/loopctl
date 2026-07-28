//! Typed lifecycle observer for agent loops.
//!
//! Defines [`LoopObserver`] — a notification-only trait with one method per
//! lifecycle point — and [`ObserverHost`], which holds registered observers
//! and dispatches notifications to each in order.
//!
//! # Context Structs
//!
//! Each callback receives a typed context struct with relevant fields:
//!
//! - [`RunStartContext`] / [`RunEndContext`] — run boundaries (one per `run()` call)
//! - [`TurnStartContext`] / [`TurnEndContext`] — turn boundaries
//! - [`StreamContext`] / [`StreamFailureContext`] — stream success/failure
//! - [`ResponseContext`] — model response text and usage
//! - [`TextDeltaContext`] — incremental text chunk while streaming
//! - [`ThinkingDeltaContext`] — incremental reasoning chunk while streaming
//! - [`ToolCallReceivedContext`] — tool call accumulated, before dispatch
//! - [`ToolPreContext`] / [`ToolPostContext`] — tool dispatch lifecycle
//! - [`CompactedContext`] — context window compaction
//! - [`FallbackContext`] — model fallback event
//! - [`LoopDetectedContext`] — loop detection event
//! - [`ConvergenceDetectedContext`] — convergence detection event
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::observer::{LoopObserver, RunStartContext};
//!
//! struct MetricsObserver;
//!
//! impl LoopObserver for MetricsObserver {
//!     fn name(&self) -> &str { "metrics" }
//!
//!     fn on_run_start(&self, ctx: &RunStartContext) {
//!         println!("session {} started", ctx.session_id);
//!     }
//! }
//! ```

use std::sync::Arc;

pub mod context;

pub use context::{
    CompactedContext, ConvergenceDetectedContext, FallbackContext, LoopDetectedContext,
    ModelSwitchedContext, ResponseContext, RunEndContext, RunStartContext, StreamContext,
    StreamFailureContext, TextDeltaContext, ThinkingDeltaContext, ToolCallReceivedContext,
    ToolPostContext, ToolPreContext, TurnEndContext, TurnStartContext,
};

/// A notification observer that receives typed callbacks at agent loop lifecycle points.
///
/// Observers are registered via [`LoopManagers`](crate::managers::LoopManagers) and called at each
/// lifecycle point in registration order. All methods are **notification-only** — they
/// return `()`. Use the hook system (requires `hooks` feature) if you need to control
/// flow (block/allow actions).
///
/// All methods have default no-op implementations. Override only the callbacks you need.
pub trait LoopObserver: Send + Sync {
    /// Human-readable name for diagnostics and logging.
    ///
    /// Returned in error messages and telemetry to identify which observer
    /// produced a side-effect.
    fn name(&self) -> &str;

    /// Called when a run begins.
    ///
    /// Fired at the start of each `run()` call, before the first turn of
    /// that run. A session may contain many runs.
    fn on_run_start(&self, _ctx: &RunStartContext) {}

    /// Called when a run ends.
    ///
    /// Fired after the run completes — whether normally, by error, or by
    /// cancellation. Check [`RunEndContext::success`] to distinguish normal
    /// exit from failure.
    fn on_run_end(&self, _ctx: &RunEndContext) {}

    /// Called at the start of processing a turn.
    ///
    /// Fired before the model is called for this turn.
    fn on_turn_start(&self, _ctx: &TurnStartContext) {}

    /// Called when a turn completes.
    ///
    /// Fired after tool dispatch and any compaction has finished.
    /// Check [`TurnEndContext::success`] to detect turn-level failures.
    fn on_turn_end(&self, _ctx: &TurnEndContext) {}

    /// Called after the model streams a response successfully.
    ///
    /// Provides token counts for the completed streaming request.
    /// Not fired when the stream fails — see [`on_stream_failure`](Self::on_stream_failure).
    fn on_stream_success(&self, _ctx: &StreamContext) {}

    /// Called when the API stream fails.
    ///
    /// Fired on network errors, API errors, or stream interruptions.
    /// The loop may retry or fall back to another model after this notification.
    fn on_stream_failure(&self, _ctx: &StreamFailureContext) {}

    /// Called after extracting the model's text response.
    ///
    /// Contains the concatenated assistant text and optional token usage.
    /// Tool-call content is excluded; use [`on_tool_post`](Self::on_tool_post)
    /// for tool results.
    fn on_response(&self, _ctx: &ResponseContext) {}

    /// Called for each text delta while the model streams a response.
    ///
    /// Fires once per `IndexedDelta(Text)` stream event, *during* streaming —
    /// before [`on_response`](Self::on_response), which delivers the assembled
    /// text once the stream ends. Concatenate
    /// [`TextDeltaContext::delta`](TextDeltaContext::delta) across calls, in
    /// arrival order, to reconstruct the per-turn text.
    ///
    /// This is the per-token counterpart to the raw
    /// [`text_streamer`](crate::engine::BareLoop::set_text_streamer) callback,
    /// delivered through the observer system so multiple observers each receive
    /// every chunk. The streamer remains available for simple single-consumer
    /// use.
    ///
    /// Default no-op — override only if you need live per-chunk text. An
    /// override should do minimal work (append to a buffer, notify a waker) and
    /// must not parse, render, or perform I/O synchronously, since it runs on
    /// the stream-ingestion task.
    ///
    /// # Retry caveat (handler path)
    ///
    /// When a [`StreamHandler`](crate::stream::handler::StreamHandler) is
    /// configured, this callback fires for every event of every attempt —
    /// including partial events from a failed attempt that got cut off
    /// mid-stream. An observer that concatenates `delta` across calls will,
    /// under the handler path only, see duplicated or truncated-then-restarted
    /// fragments after a retry. Consumers that need only committed output must
    /// buffer until [`on_response`](Self::on_response) (or
    /// [`on_turn_end`](Self::on_turn_end)).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::{Arc, Mutex};
    /// use loopctl::observer::{LoopObserver, TextDeltaContext};
    ///
    /// // A minimal observer that records every streamed chunk.
    /// struct Recorder { chunks: Arc<Mutex<Vec<String>>> }
    /// impl LoopObserver for Recorder {
    ///     fn name(&self) -> &str { "recorder" }
    ///     fn on_text_delta(&self, ctx: &TextDeltaContext) {
    ///         if let Ok(mut buf) = self.chunks.lock() {
    ///             buf.push(ctx.delta.clone());
    ///         }
    ///     }
    /// }
    /// ```
    fn on_text_delta(&self, _ctx: &TextDeltaContext) {}

    /// Called for each incremental reasoning ("thinking") chunk.
    ///
    /// Fired per `DeltaPart::Thinking` during streaming, symmetric to
    /// [`on_text_delta`](Self::on_text_delta). Reasoning is distinct from
    /// visible assistant text; do not concatenate it with `on_text_delta` /
    /// [`on_response`](Self::on_response) output. Redacted reasoning arrives
    /// as an empty `delta` (render a placeholder, not the empty string).
    ///
    /// Inherits the same retry caveat as [`on_text_delta`](Self::on_text_delta):
    /// under a configured [`StreamHandler`](crate::stream::handler::StreamHandler),
    /// partial events from a failed attempt fire here too. Buffer until
    /// [`on_turn_end`](Self::on_turn_end) if you need only committed reasoning.
    fn on_thinking_delta(&self, _ctx: &ThinkingDeltaContext) {}

    /// Called when the engine has accumulated a tool call and is about to dispatch it.
    ///
    /// Fires once per call, after the streaming response is accumulated and
    /// before dispatch begins — strictly earlier than
    /// [`on_tool_pre`](Self::on_tool_pre). Unlike `on_tool_pre`, the context
    /// carries the call `input`, and the event does **not** repeat on recovery
    /// retries (it is per-call, not per-attempt).
    ///
    /// Notification-only — cannot block or modify the tool call. Use the hook
    /// system (requires `hooks` feature) for flow control.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    /// use loopctl::observer::{LoopObserver, ToolCallReceivedContext};
    ///
    /// struct PendingTracker { pending: Arc<AtomicUsize> }
    /// impl LoopObserver for PendingTracker {
    ///     fn name(&self) -> &str { "pending-tracker" }
    ///     fn on_tool_call_received(&self, _ctx: &ToolCallReceivedContext) {
    ///         self.pending.fetch_add(1, Ordering::SeqCst);
    ///     }
    /// }
    /// ```
    fn on_tool_call_received(&self, _ctx: &ToolCallReceivedContext) {}

    /// Called before a tool is dispatched.
    ///
    /// Notification-only — cannot block or modify the tool call.
    /// Use the hook system (requires `hooks` feature) for flow control.
    fn on_tool_pre(&self, _ctx: &ToolPreContext) {}

    /// Called after a tool completes execution.
    ///
    /// Reports whether the tool errored and includes a hash of the result
    /// for loop-detection correlation.
    fn on_tool_post(&self, _ctx: &ToolPostContext) {}

    /// Called after conversation compaction.
    ///
    /// Reports token counts before and after compaction. Only fired when
    /// compaction actually occurred — not on no-action passes.
    fn on_compaction(&self, _ctx: &CompactedContext) {}

    /// Called when a model fallback is triggered.
    ///
    /// Indicates that the primary model failed and a fallback model
    /// was selected for subsequent requests.
    fn on_fallback(&self, _ctx: &FallbackContext) {}

    /// Called when the model is hot-swapped at runtime.
    ///
    /// Fired by [`BareLoop::switch_model`](crate::engine::BareLoop::switch_model)
    /// after the client has accepted the new model.
    fn on_model_switched(&self, _ctx: &ModelSwitchedContext) {}

    /// Called when a loop is detected in tool operations.
    ///
    /// Fired when the same tool operation produces the same result
    /// repeatedly, exceeding the configured threshold.
    fn on_loop_detected(&self, _ctx: &LoopDetectedContext) {}

    /// Called when response convergence is detected.
    ///
    /// Fired when consecutive model responses become sufficiently similar
    /// as determined by the convergence detection policy.
    fn on_convergence_detected(&self, _ctx: &ConvergenceDetectedContext) {}

    /// Reset observer state for a new session.
    ///
    /// Called once per session (on the first `run()`), before
    /// [`on_run_start`](Self::on_run_start), to allow observers to clear
    /// per-session accumulators.
    fn reset(&self) {}
}

/// Holds registered observers and dispatches notifications to each.
///
/// Observers run in registration order. All observers are always notified —
/// there is no short-circuiting (use the hook system, requires `hooks` feature, for flow control).
///
/// An empty host (no observers registered) is effectively zero-cost:
/// each notification call iterates an empty `Vec`.
#[derive(Default)]
pub struct ObserverHost {
    observers: Vec<Arc<dyn LoopObserver>>,
}

impl ObserverHost {
    /// Create an empty observer host.
    ///
    /// Equivalent to [`ObserverHost::default`] but more explicit.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an observer.
    ///
    /// Observers are called in registration order at each lifecycle point.
    /// Registering the same observer twice will result in duplicate notifications.
    pub fn register(&mut self, observer: Arc<dyn LoopObserver>) {
        self.observers.push(observer);
    }

    /// Dispatch a closure to each observer, isolating panics.
    ///
    /// If an observer panics, it is logged and the remaining observers
    /// still receive the event. This matches the panic-isolation applied
    /// to tool dispatch.
    fn dispatch<F>(&self, f: F)
    where
        F: Fn(&dyn LoopObserver),
    {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        for obs in &self.observers {
            let obs: &dyn LoopObserver = obs.as_ref();
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| f(obs))) {
                let msg = payload
                    .downcast_ref::<&'static str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                tracing::error!(
                    observer = obs.name(),
                    panic_message = %msg,
                    "observer panicked; continuing with remaining observers"
                );
            }
        }
    }

    /// Number of observers currently registered.
    ///
    /// Cheap (`O(1)`) read of the observer vector length. Returns `0`
    /// for a freshly constructed host, in which case every dispatch
    /// method is effectively a no-op (it iterates an empty `Vec`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.observers.len()
    }

    /// Whether no observers are registered.
    ///
    /// `true` when [`len`](Self::len) is `0`. When `true`, every
    /// dispatch call short-circuits through an empty loop, so an
    /// observerless host adds negligible overhead to the agent loop.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }

    /// Reset all observers for a new session.
    ///
    /// Calls [`LoopObserver::reset`] on every registered observer,
    /// allowing them to clear per-session accumulators.
    pub fn reset_all(&self) {
        self.dispatch(|obs| obs.reset());
    }

    /// Dispatch [`LoopObserver::on_run_start`] to all observers.
    ///
    /// Fired at the start of each `run()` call, before the first turn
    /// begins. Iterates registered observers in registration order; use it
    /// to initialize per-run observer state.
    pub fn on_run_start(&self, ctx: &RunStartContext) {
        self.dispatch(|obs| obs.on_run_start(ctx));
    }

    /// Dispatch [`LoopObserver::on_run_end`] to all observers.
    ///
    /// Fired at the end of each `run()` call, after the loop has
    /// terminated. Iterates registered observers in registration order;
    /// check [`RunEndContext::success`] in each observer to distinguish a
    /// normal exit from a failure.
    pub fn on_run_end(&self, ctx: &RunEndContext) {
        self.dispatch(|obs| obs.on_run_end(ctx));
    }

    /// Dispatch [`LoopObserver::on_turn_start`] to all observers.
    ///
    /// Fired before the model is called for the turn. Iterates
    /// registered observers in registration order.
    pub fn on_turn_start(&self, ctx: &TurnStartContext) {
        self.dispatch(|obs| obs.on_turn_start(ctx));
    }

    /// Dispatch [`LoopObserver::on_turn_end`] to all observers.
    ///
    /// Fired after the turn's model call and any tool dispatch have
    /// completed. Iterates registered observers in registration order;
    /// check [`TurnEndContext::success`] to detect turn-level failures.
    pub fn on_turn_end(&self, ctx: &TurnEndContext) {
        self.dispatch(|obs| obs.on_turn_end(ctx));
    }

    /// Dispatch [`LoopObserver::on_stream_success`] to all observers.
    ///
    /// Fired when a streaming model call completes successfully.
    /// Iterates registered observers in registration order. Not fired
    /// when the stream fails — see
    /// [`on_stream_failure`](Self::on_stream_failure).
    pub fn on_stream_success(&self, ctx: &StreamContext) {
        self.dispatch(|obs| obs.on_stream_success(ctx));
    }

    /// Dispatch [`LoopObserver::on_stream_failure`] to all observers.
    ///
    /// Fired when a streaming model call fails (timeout, transport
    /// error, rate limit). Iterates registered observers in registration
    /// order; the loop may retry or fall back to another model after
    /// this notification.
    pub fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        self.dispatch(|obs| obs.on_stream_failure(ctx));
    }

    /// Dispatch [`LoopObserver::on_response`] to all observers.
    ///
    /// Fired once the full model response is assembled — the committed
    /// assistant text plus any tool calls, before the framework resolves
    /// tool results. Iterates registered observers in registration order.
    pub fn on_response(&self, ctx: &ResponseContext) {
        self.dispatch(|obs| obs.on_response(ctx));
    }

    /// Dispatch [`LoopObserver::on_text_delta`] to all observers.
    ///
    /// Iterates registered observers in registration order. Called once per
    /// streamed text delta, so each observer's `on_text_delta` must be cheap.
    pub fn on_text_delta(&self, ctx: &TextDeltaContext) {
        self.dispatch(|obs| obs.on_text_delta(ctx));
    }

    /// Dispatch [`LoopObserver::on_thinking_delta`] to all observers.
    ///
    /// Iterates registered observers in registration order. Called once per
    /// streamed reasoning delta, so each observer's `on_thinking_delta` must
    /// be cheap.
    pub fn on_thinking_delta(&self, ctx: &ThinkingDeltaContext) {
        self.dispatch(|obs| obs.on_thinking_delta(ctx));
    }

    /// Dispatch [`LoopObserver::on_tool_pre`] to all observers.
    ///
    /// Fired before a tool is dispatched, carrying the call's name and
    /// input. Iterates registered observers in registration order.
    /// Notification-only — use the hook system (requires the `hooks`
    /// feature) for flow control.
    pub fn on_tool_pre(&self, ctx: &ToolPreContext) {
        self.dispatch(|obs| obs.on_tool_pre(ctx));
    }

    /// Dispatch [`LoopObserver::on_tool_call_received`] to all observers.
    ///
    /// Fired when the framework receives a tool call from the model
    /// response, before it is dispatched. Iterates registered observers
    /// in registration order; useful for pending-call tracking.
    pub fn on_tool_call_received(&self, ctx: &ToolCallReceivedContext) {
        self.dispatch(|obs| obs.on_tool_call_received(ctx));
    }

    /// Dispatch [`LoopObserver::on_tool_post`] to all observers.
    ///
    /// Fired after a tool completes, carrying its output and timing.
    /// Iterates registered observers in registration order; useful for
    /// loop-detection correlation.
    pub fn on_tool_post(&self, ctx: &ToolPostContext) {
        self.dispatch(|obs| obs.on_tool_post(ctx));
    }

    /// Dispatch [`LoopObserver::on_compaction`] to all observers.
    ///
    /// Fired after the context manager reduces the history, carrying the
    /// before/after token counts. Iterates registered observers in
    /// registration order; fired only when compaction actually occurred,
    /// not on no-action passes.
    pub fn on_compaction(&self, ctx: &CompactedContext) {
        self.dispatch(|obs| obs.on_compaction(ctx));
    }

    /// Dispatch [`LoopObserver::on_fallback`] to all observers.
    ///
    /// Fired when the fallback manager activates a fallback model,
    /// carrying the reason and the selected model. Iterates registered
    /// observers in registration order; the fallback model will be used
    /// for subsequent requests.
    pub fn on_fallback(&self, ctx: &FallbackContext) {
        self.dispatch(|obs| obs.on_fallback(ctx));
    }

    /// Dispatch [`LoopObserver::on_model_switched`] to all observers.
    ///
    /// Fired after an explicit model switch is accepted by the client,
    /// carrying the old and new model names. Iterates registered
    /// observers in registration order.
    pub fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        self.dispatch(|obs| obs.on_model_switched(ctx));
    }

    /// Dispatch [`LoopObserver::on_loop_detected`] to all observers.
    ///
    /// Fired when the loop detector observes the same operation
    /// repeatedly, exceeding the configured threshold. Iterates
    /// registered observers in registration order.
    pub fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        self.dispatch(|obs| obs.on_loop_detected(ctx));
    }

    /// Dispatch [`LoopObserver::on_convergence_detected`] to all
    /// observers.
    ///
    /// Fired when the convergence detector observes semantically
    /// similar assistant responses, as determined by the convergence
    /// detection policy. Iterates registered observers in registration
    /// order.
    pub fn on_convergence_detected(&self, ctx: &ConvergenceDetectedContext) {
        self.dispatch(|obs| obs.on_convergence_detected(ctx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingObserver {
        name: &'static str,
        stream_success: AtomicUsize,
        resets: AtomicUsize,
    }

    impl CountingObserver {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                stream_success: AtomicUsize::new(0),
                resets: AtomicUsize::new(0),
            }
        }
    }

    impl LoopObserver for CountingObserver {
        fn name(&self) -> &str {
            self.name
        }

        fn on_stream_success(&self, _ctx: &StreamContext) {
            self.stream_success.fetch_add(1, Ordering::SeqCst);
        }

        fn reset(&self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn host_dispatches_to_single_observer() {
        let obs = Arc::new(CountingObserver::new("test"));
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs) as Arc<dyn LoopObserver>);
        host.on_stream_success(&StreamContext {
            turn: 0,
            model: "m".into(),
            input_tokens: 0,
            output_tokens: 0,
        });
        assert_eq!(obs.stream_success.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_dispatches_to_multiple_observers() {
        let obs1 = Arc::new(CountingObserver::new("a"));
        let obs2 = Arc::new(CountingObserver::new("b"));
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs1) as Arc<dyn LoopObserver>);
        host.register(Arc::clone(&obs2) as Arc<dyn LoopObserver>);
        host.on_stream_success(&StreamContext {
            turn: 0,
            model: "m".into(),
            input_tokens: 0,
            output_tokens: 0,
        });
        assert_eq!(obs1.stream_success.load(Ordering::SeqCst), 1);
        assert_eq!(obs2.stream_success.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_len_and_is_empty() {
        let mut host = ObserverHost::new();
        assert!(host.is_empty());
        assert_eq!(host.len(), 0);
        host.register(Arc::new(CountingObserver::new("x")) as Arc<dyn LoopObserver>);
        assert!(!host.is_empty());
        assert_eq!(host.len(), 1);
    }

    #[test]
    fn host_reset_all() {
        let obs = Arc::new(CountingObserver::new("p"));
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs) as Arc<dyn LoopObserver>);
        host.on_stream_success(&StreamContext {
            turn: 0,
            model: "m".into(),
            input_tokens: 0,
            output_tokens: 0,
        });
        assert_eq!(obs.stream_success.load(Ordering::SeqCst), 1);
        host.reset_all();
        assert_eq!(obs.resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_dispatches_model_switched() {
        struct SwitchRecorder {
            events: std::sync::Mutex<Vec<(String, String)>>,
        }
        impl LoopObserver for SwitchRecorder {
            fn name(&self) -> &'static str {
                "switch-recorder"
            }
            fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
                crate::error::recover_guard(self.events.lock())
                    .push((ctx.from.clone(), ctx.to.clone()));
            }
        }

        let obs = Arc::new(SwitchRecorder {
            events: std::sync::Mutex::new(Vec::new()),
        });
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs) as Arc<dyn LoopObserver>);

        host.on_model_switched(&ModelSwitchedContext {
            from: "a".into(),
            to: "b".into(),
        });
        host.on_model_switched(&ModelSwitchedContext {
            from: "b".into(),
            to: "c".into(),
        });

        let events = crate::error::recover_guard(obs.events.lock());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], ("a".into(), "b".into()));
        assert_eq!(events[1], ("b".into(), "c".into()));
    }

    #[test]
    fn model_switched_default_is_noop() {
        // The default impl of on_model_switched should be a no-op
        // (no panic, no crash).
        struct NoopObserver;
        impl LoopObserver for NoopObserver {
            fn name(&self) -> &'static str {
                "noop"
            }
        }

        let obs = NoopObserver;
        obs.on_model_switched(&ModelSwitchedContext {
            from: "x".into(),
            to: "y".into(),
        });
    }

    #[test]
    fn on_text_delta_default_is_noop() {
        struct NoopObserver;
        impl LoopObserver for NoopObserver {
            fn name(&self) -> &'static str {
                "noop"
            }
        }

        let obs = NoopObserver;
        let mut host = ObserverHost::new();
        host.register(Arc::new(obs) as Arc<dyn LoopObserver>);
        host.on_text_delta(&TextDeltaContext {
            turn: 0,
            delta: "x".into(),
        });
    }

    #[test]
    fn host_dispatches_on_text_delta_to_all_observers() {
        struct DeltaRecorder {
            deltas: std::sync::Mutex<Vec<String>>,
        }
        impl LoopObserver for DeltaRecorder {
            fn name(&self) -> &'static str {
                "delta-recorder"
            }
            fn on_text_delta(&self, ctx: &TextDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
            }
        }

        let obs1 = Arc::new(DeltaRecorder {
            deltas: std::sync::Mutex::new(Vec::new()),
        });
        let obs2 = Arc::new(DeltaRecorder {
            deltas: std::sync::Mutex::new(Vec::new()),
        });
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs1) as Arc<dyn LoopObserver>);
        host.register(Arc::clone(&obs2) as Arc<dyn LoopObserver>);

        host.on_text_delta(&TextDeltaContext {
            turn: 0,
            delta: "x".into(),
        });

        assert_eq!(
            crate::error::recover_guard(obs1.deltas.lock()).clone(),
            vec!["x".to_string()]
        );
        assert_eq!(
            crate::error::recover_guard(obs2.deltas.lock()).clone(),
            vec!["x".to_string()]
        );
    }

    #[test]
    fn host_dispatches_on_text_delta_with_no_observers() {
        let host = ObserverHost::new();
        host.on_text_delta(&TextDeltaContext {
            turn: 0,
            delta: "x".into(),
        });
    }

    #[test]
    fn on_tool_call_received_default_is_noop() {
        struct NoopObserver;
        impl LoopObserver for NoopObserver {
            fn name(&self) -> &'static str {
                "noop"
            }
        }

        let obs = NoopObserver;
        let mut host = ObserverHost::new();
        host.register(Arc::new(obs) as Arc<dyn LoopObserver>);
        host.on_tool_call_received(&ToolCallReceivedContext {
            turn: 0,
            tool: "echo".into(),
            call_id: "c1".into(),
            input: serde_json::Value::Null,
        });
    }

    #[test]
    fn host_dispatches_on_tool_call_received_to_all_observers() {
        struct ReceivedRecorder {
            calls: std::sync::Mutex<Vec<String>>,
        }
        impl LoopObserver for ReceivedRecorder {
            fn name(&self) -> &'static str {
                "received-recorder"
            }
            fn on_tool_call_received(&self, ctx: &ToolCallReceivedContext) {
                crate::error::recover_guard(self.calls.lock()).push(ctx.tool.clone());
            }
        }

        let obs1 = Arc::new(ReceivedRecorder {
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let obs2 = Arc::new(ReceivedRecorder {
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs1) as Arc<dyn LoopObserver>);
        host.register(Arc::clone(&obs2) as Arc<dyn LoopObserver>);

        host.on_tool_call_received(&ToolCallReceivedContext {
            turn: 0,
            tool: "edit".into(),
            call_id: "c1".into(),
            input: serde_json::Value::Null,
        });

        assert_eq!(
            crate::error::recover_guard(obs1.calls.lock()).clone(),
            vec!["edit".to_string()]
        );
        assert_eq!(
            crate::error::recover_guard(obs2.calls.lock()).clone(),
            vec!["edit".to_string()]
        );
    }

    #[test]
    fn host_dispatches_on_tool_call_received_with_no_observers() {
        let host = ObserverHost::new();
        host.on_tool_call_received(&ToolCallReceivedContext {
            turn: 0,
            tool: "echo".into(),
            call_id: "c1".into(),
            input: serde_json::Value::Null,
        });
    }

    #[test]
    fn host_dispatches_on_thinking_delta_to_all_observers() {
        struct ThinkingRecorder {
            deltas: std::sync::Mutex<Vec<String>>,
        }
        impl LoopObserver for ThinkingRecorder {
            fn name(&self) -> &'static str {
                "thinking-recorder"
            }
            fn on_thinking_delta(&self, ctx: &ThinkingDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
            }
        }

        let obs1 = Arc::new(ThinkingRecorder {
            deltas: std::sync::Mutex::new(Vec::new()),
        });
        let obs2 = Arc::new(ThinkingRecorder {
            deltas: std::sync::Mutex::new(Vec::new()),
        });
        let mut host = ObserverHost::new();
        host.register(Arc::clone(&obs1) as Arc<dyn LoopObserver>);
        host.register(Arc::clone(&obs2) as Arc<dyn LoopObserver>);

        host.on_thinking_delta(&ThinkingDeltaContext {
            turn: 2,
            delta: "reasoning".into(),
        });

        assert_eq!(
            crate::error::recover_guard(obs1.deltas.lock()).clone(),
            vec!["reasoning".to_string()]
        );
        assert_eq!(
            crate::error::recover_guard(obs2.deltas.lock()).clone(),
            vec!["reasoning".to_string()]
        );
    }

    #[test]
    fn on_thinking_delta_default_is_noop() {
        struct NoopObserver;
        impl LoopObserver for NoopObserver {
            fn name(&self) -> &'static str {
                "noop"
            }
        }
        let obs = NoopObserver;
        // A bare observer that doesn't override on_thinking_delta must compile
        // and not panic when the method is called.
        obs.on_thinking_delta(&ThinkingDeltaContext {
            turn: 0,
            delta: String::new(),
        });
    }
}
