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
//! - [`CompactionContext`] — context window compaction
//! - [`FallbackContext`] — model fallback event
//! - [`LoopDetectedContext`] — loop detection event
//! - [`ConvergenceDetectedContext`] — convergence detection event
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::core::observer::{LoopObserver, SessionStartContext};
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

// ==================================================
// Context structs
// ==================================================

/// Context for [`LoopObserver::on_session_start`].
#[derive(Debug, Clone)]
pub struct SessionStartContext {
    /// Unique session identifier.
    pub session_id: uuid::Uuid,
}

/// Context for [`LoopObserver::on_session_end`].
#[derive(Debug, Clone)]
pub struct SessionEndContext {
    /// Whether the session completed successfully.
    pub success: bool,
    /// Error description, if the session ended due to an error.
    pub error: Option<String>,
    /// Total turns completed during the session.
    pub total_turns: usize,
    /// Total session duration in milliseconds.
    pub duration_ms: u64,
}

/// Context for [`LoopObserver::on_turn_start`].
#[derive(Debug, Clone)]
pub struct TurnStartContext {
    /// Turn number (0-indexed).
    pub turn: usize,
    /// The user query that initiated this turn.
    pub query: String,
}

/// Context for [`LoopObserver::on_turn_end`].
#[derive(Debug, Clone)]
pub struct TurnEndContext {
    /// Turn number.
    pub turn: usize,
    /// Whether the turn completed successfully.
    pub success: bool,
    /// Error description, if the turn failed.
    pub error: Option<String>,
    /// Wall-clock duration of the turn in milliseconds.
    pub duration_ms: u64,
    /// Input tokens consumed this turn.
    pub input_tokens: u64,
    /// Output tokens generated this turn.
    pub output_tokens: u64,
}

/// Context for [`LoopObserver::on_stream_success`].
#[derive(Debug, Clone)]
pub struct StreamContext {
    /// Turn number.
    pub turn: usize,
    /// Model that was streamed.
    pub model: String,
    /// Input tokens consumed.
    pub input_tokens: u64,
    /// Output tokens generated.
    pub output_tokens: u64,
}

/// Context for [`LoopObserver::on_stream_failure`].
#[derive(Debug, Clone)]
pub struct StreamFailureContext {
    /// Turn number.
    pub turn: usize,
    /// Model that failed.
    pub model: String,
    /// The error that occurred.
    pub error: crate::core::AgentError,
}

/// Context for [`LoopObserver::on_response`].
#[derive(Debug, Clone)]
pub struct ResponseContext {
    /// Turn number.
    pub turn: usize,
    /// The model's text response.
    pub text: String,
    /// Token usage for this turn, if available.
    pub usage: Option<crate::stream::Usage>,
}

/// Context for [`LoopObserver::on_tool_pre`].
#[derive(Debug, Clone)]
pub struct ToolPreContext {
    /// Turn number.
    pub turn: usize,
    /// Tool name.
    pub tool: String,
    /// Tool call ID from the API response.
    pub tool_call_id: String,
}

/// Context for [`LoopObserver::on_tool_post`].
#[derive(Debug, Clone)]
pub struct ToolPostContext {
    /// Turn number.
    pub turn: usize,
    /// Tool name.
    pub tool: String,
    /// Deterministic hash of the tool output, if available.
    pub result_hash: Option<u64>,
    /// Whether the tool returned an error.
    pub is_error: bool,
    /// Wall-clock execution duration.
    pub duration: std::time::Duration,
}

/// Context for [`LoopObserver::on_compaction`].
#[derive(Debug, Clone)]
pub struct CompactionContext {
    /// Message count before compaction.
    pub messages_before: usize,
    /// Message count after compaction.
    pub messages_after: usize,
    /// Estimated tokens saved by compaction.
    pub tokens_saved: u64,
}

/// Context for [`LoopObserver::on_fallback`].
#[derive(Debug, Clone)]
pub struct FallbackContext {
    /// Model that failed.
    pub from: String,
    /// Replacement model.
    pub to: String,
}

/// Context for [`LoopObserver::on_loop_detected`].
#[derive(Debug, Clone)]
pub struct LoopDetectedContext {
    /// Description of the repeating tool pattern.
    pub pattern: String,
    /// Number of times the pattern was observed.
    pub repetitions: usize,
}

/// Context for [`LoopObserver::on_convergence_detected`].
#[derive(Debug, Clone)]
pub struct ConvergenceDetectedContext {
    /// Configured action to take (e.g. `"stop"`, `"warn"`, `"compact"`).
    pub action: String,
}

// ==================================================
// LoopObserver Trait
// ==================================================

/// A notification observer that receives typed callbacks at agent loop lifecycle points.
///
/// Observers are registered via `BareLoop` or `ManagerBundle` and called at each
/// lifecycle point in registration order. All methods are **notification-only** — they
/// return `()`. Use the [hook system](crate::hooks) if you need to control
/// flow (block/allow actions).
///
/// All methods have default no-op implementations. Override only the callbacks you need.
pub trait LoopObserver: Send + Sync {
    /// Human-readable name for diagnostics and logging.
    fn name(&self) -> &str;

    /// Called when an agent session begins.
    fn on_session_start(&self, _ctx: &SessionStartContext) {}

    /// Called when an agent session ends.
    fn on_session_end(&self, _ctx: &SessionEndContext) {}

    /// Called at the start of processing a turn.
    fn on_turn_start(&self, _ctx: &TurnStartContext) {}

    /// Called when a turn completes.
    fn on_turn_end(&self, _ctx: &TurnEndContext) {}

    /// Called after the model streams a response successfully.
    fn on_stream_success(&self, _ctx: &StreamContext) {}

    /// Called when the API stream fails.
    fn on_stream_failure(&self, _ctx: &StreamFailureContext) {}

    /// Called after extracting the model's text response.
    fn on_response(&self, _ctx: &ResponseContext) {}

    /// Called before a tool is dispatched (notification-only).
    fn on_tool_pre(&self, _ctx: &ToolPreContext) {}

    /// Called after a tool completes execution.
    fn on_tool_post(&self, _ctx: &ToolPostContext) {}

    /// Called after conversation compaction.
    fn on_compaction(&self, _ctx: &CompactionContext) {}

    /// Called when a model fallback is triggered.
    fn on_fallback(&self, _ctx: &FallbackContext) {}

    /// Called when a loop is detected in tool operations.
    fn on_loop_detected(&self, _ctx: &LoopDetectedContext) {}

    /// Called when response convergence is detected.
    fn on_convergence_detected(&self, _ctx: &ConvergenceDetectedContext) {}

    /// Reset observer state for a new session.
    fn reset(&self) {}
}

// ==================================================
// ObserverHost
// ==================================================

/// Holds registered observers and dispatches notifications to each.
///
/// Observers run in registration order. All observers are always notified —
/// there is no short-circuiting (that's the [hook system](crate::hooks)'s job).
///
/// An empty host (no observers registered) is effectively zero-cost:
/// each notification call iterates an empty `Vec`.
#[derive(Default)]
pub struct ObserverHost {
    observers: Vec<Arc<dyn LoopObserver>>,
}

impl ObserverHost {
    /// Create an empty observer host.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an observer. Called in registration order at each notification point.
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
    pub fn reset_all(&self) {
        for obs in &self.observers {
            obs.reset();
        }
    }

    /// Dispatch [`LoopObserver::on_session_start`] to all observers.
    pub fn on_session_start(&self, ctx: &SessionStartContext) {
        for obs in &self.observers {
            obs.on_session_start(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_session_end`] to all observers.
    pub fn on_session_end(&self, ctx: &SessionEndContext) {
        for obs in &self.observers {
            obs.on_session_end(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_turn_start`] to all observers.
    pub fn on_turn_start(&self, ctx: &TurnStartContext) {
        for obs in &self.observers {
            obs.on_turn_start(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_turn_end`] to all observers.
    pub fn on_turn_end(&self, ctx: &TurnEndContext) {
        for obs in &self.observers {
            obs.on_turn_end(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_stream_success`] to all observers.
    pub fn on_stream_success(&self, ctx: &StreamContext) {
        for obs in &self.observers {
            obs.on_stream_success(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_stream_failure`] to all observers.
    pub fn on_stream_failure(&self, ctx: &StreamFailureContext) {
        for obs in &self.observers {
            obs.on_stream_failure(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_response`] to all observers.
    pub fn on_response(&self, ctx: &ResponseContext) {
        for obs in &self.observers {
            obs.on_response(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_tool_pre`] to all observers.
    pub fn on_tool_pre(&self, ctx: &ToolPreContext) {
        for obs in &self.observers {
            obs.on_tool_pre(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_tool_post`] to all observers.
    pub fn on_tool_post(&self, ctx: &ToolPostContext) {
        for obs in &self.observers {
            obs.on_tool_post(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_compaction`] to all observers.
    pub fn on_compaction(&self, ctx: &CompactionContext) {
        for obs in &self.observers {
            obs.on_compaction(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_fallback`] to all observers.
    pub fn on_fallback(&self, ctx: &FallbackContext) {
        for obs in &self.observers {
            obs.on_fallback(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_loop_detected`] to all observers.
    pub fn on_loop_detected(&self, ctx: &LoopDetectedContext) {
        for obs in &self.observers {
            obs.on_loop_detected(ctx);
        }
    }

    /// Dispatch [`LoopObserver::on_convergence_detected`] to all observers.
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
}
