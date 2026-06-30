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
//! - [`SessionStartContext`] / [`SessionEndContext`] — session boundaries
//! - [`TurnStartContext`] / [`TurnEndContext`] — turn boundaries
//! - [`StreamContext`] / [`StreamFailureContext`] — stream success/failure
//! - [`ResponseContext`] — model response text and usage
//! - [`ToolPreContext`] / [`ToolPostContext`] — tool dispatch lifecycle
//! - [`CompactedContext`] — context window compaction
//! - [`FallbackContext`] — model fallback event
//! - [`LoopDetectedContext`] — loop detection event
//! - [`ConvergenceDetectedContext`] — convergence detection event
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::observer::{LoopObserver, SessionStartContext};
//!
//! struct MetricsObserver;
//!
//! impl LoopObserver for MetricsObserver {
//!     fn name(&self) -> &str { "metrics" }
//!
//!     fn on_session_start(&self, ctx: &SessionStartContext) {
//!         println!("session {} started", ctx.session_id);
//!     }
//! }
//! ```

use std::sync::Arc;

pub mod context;

pub use context::{
    CompactedContext, ConvergenceDetectedContext, FallbackContext, LoopDetectedContext,
    ModelSwitchedContext, ResponseContext, SessionEndContext, SessionStartContext, StreamContext,
    StreamFailureContext, ToolPostContext, ToolPreContext, TurnEndContext, TurnStartContext,
};
// ==================================================
// LoopObserver Trait
// ==================================================

/// A notification observer that receives typed callbacks at agent loop lifecycle points.
///
/// Observers are registered via [`LoopRuntime`](crate::runtime::LoopRuntime) and called at each
/// lifecycle point in registration order. All methods are **notification-only** — they
/// return `()`. Use the [hook system](crate::hooks) if you need to control
/// flow (block/allow actions).
///
/// All methods have default no-op implementations. Override only the callbacks you need.
pub trait LoopObserver: Send + Sync {
    /// Human-readable name for diagnostics and logging.
    ///
    /// Returned in error messages and telemetry to identify which observer
    /// produced a side-effect.
    fn name(&self) -> &str;

    /// Called when an agent session begins.
    ///
    /// Fired once per session, before the first turn starts.
    fn on_session_start(&self, _ctx: &SessionStartContext) {}

    /// Called when an agent session ends.
    ///
    /// Fired after the last turn completes or when a fatal error stops the loop.
    /// Check [`SessionEndContext::success`] to distinguish normal exit from failure.
    fn on_session_end(&self, _ctx: &SessionEndContext) {}

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

    /// Called before a tool is dispatched.
    ///
    /// Notification-only — cannot block or modify the tool call.
    /// Use the [hook system](crate::hooks) for flow control.
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
    /// Called before [`on_session_start`](Self::on_session_start) to allow
    /// observers to clear per-session accumulators.
    fn reset(&self) {}
}

// ==================================================
// ObserverHost
// ==================================================

/// Holds registered observers and dispatches notifications to each.
///
/// Observers run in registration order. All observers are always notified —
/// there is no short-circuiting (use the [hook system](crate::hooks) for flow control).
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

    /// Number of registered observers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.observers.len()
    }

    /// Whether no observers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }

    /// Reset all observers for a new session.
    ///
    /// Calls [`LoopObserver::reset`] on every registered observer,
    /// allowing them to clear per-session accumulators.
    pub fn reset_all(&self) {
        for obs in &self.observers {
            obs.reset();
        }
    }

    /// Dispatch [`LoopObserver::on_session_start`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_session_start(&self, ctx: &SessionStartContext) {
        for obs in &self.observers {
            obs.on_session_start(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_session_end`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_session_end(&self, ctx: &SessionEndContext) {
        for obs in &self.observers {
            obs.on_session_end(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_turn_start`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_turn_start(&self, ctx: &TurnStartContext) {
        for obs in &self.observers {
            obs.on_turn_start(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_turn_end`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_turn_end(&self, ctx: &TurnEndContext) {
        for obs in &self.observers {
            obs.on_turn_end(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_stream_success`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_stream_success(&self, ctx: &StreamContext) {
        for obs in &self.observers {
            obs.on_stream_success(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_stream_failure`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        for obs in &self.observers {
            obs.on_stream_failure(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_response`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_response(&self, ctx: &ResponseContext) {
        for obs in &self.observers {
            obs.on_response(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_tool_pre`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_tool_pre(&self, ctx: &ToolPreContext) {
        for obs in &self.observers {
            obs.on_tool_pre(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_tool_post`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_tool_post(&self, ctx: &ToolPostContext) {
        for obs in &self.observers {
            obs.on_tool_post(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_compaction`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_compaction(&self, ctx: &CompactedContext) {
        for obs in &self.observers {
            obs.on_compaction(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_fallback`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_fallback(&self, ctx: &FallbackContext) {
        for obs in &self.observers {
            obs.on_fallback(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_model_switched`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
        for obs in &self.observers {
            obs.on_model_switched(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_loop_detected`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        for obs in &self.observers {
            obs.on_loop_detected(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_convergence_detected`] to all observers.
    ///
    /// Iterates registered observers in registration order.
    pub fn on_convergence_detected(&self, ctx: &ConvergenceDetectedContext) {
        for obs in &self.observers {
            obs.on_convergence_detected(ctx);
        }
    }
}

// ==================================================
// Tests
// ==================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A test observer that counts notification invocations.
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
            events: parking_lot::Mutex<Vec<(String, String)>>,
        }
        impl LoopObserver for SwitchRecorder {
            fn name(&self) -> &'static str {
                "switch-recorder"
            }
            fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
                self.events.lock().push((ctx.from.clone(), ctx.to.clone()));
            }
        }

        let obs = Arc::new(SwitchRecorder {
            events: parking_lot::Mutex::new(Vec::new()),
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

        let events = obs.events.lock();
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
}
