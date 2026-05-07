//! Reference observer implementations — ready-to-use [`AgentObserver`] backends.
//!
//! This module provides concrete implementations of the [`AgentObserver`] trait
//! so consumers can plug in observability without implementing the trait from
//! scratch. Each implementation covers a different use case, from silent
//! no-ops to full `tracing`-based logging to composite fan-out.
//!
//! # Provided Implementations
//!
//! - **[`NoOpObserver`]** — A zero-cost no-op that silently ignores all
//!   lifecycle events. Useful as a safe default and in tests.
//! - **[`LoggingObserver`]** — Emits structured log lines for every event
//!   via the `tracing` crate. Session and compaction events are logged at
//!   `info` level; tool events at `debug`; errors and fallbacks at `warn`.
//! - **[`MultiObserver`]** — A composite observer that fans out every event
//!   to an ordered list of inner observers. Follows the composite pattern.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::builtin::observer::{LoggingObserver, MultiObserver, NoOpObserver};
//! use loopctl::core::AgentObserver;
//!
//! // Use a single logging observer:
//! let observer = LoggingObserver;
//!
//! // Or compose multiple observers:
//! let multi = MultiObserver::new()
//!     .with(LoggingObserver)
//!     .with(NoOpObserver);
//!
//! multi.on_session_start(uuid::Uuid::new_v4());
//! ```

use crate::core::AgentObserver;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// A no-op observer that silently ignores all events.
///
/// Every method body is empty — no allocation, no I/O, no side effects.
/// Use this as a safe default when no observability is needed, as a
/// placeholder in tests, or as a baseline when benchmarking the framework
/// overhead *without* observer noise.
///
/// # Example
///
/// ```
/// use loopctl::builtin::observer::NoOpObserver;
/// use loopctl::core::AgentObserver;
///
/// let observer = NoOpObserver;
/// observer.on_session_start(uuid::Uuid::new_v4()); // does nothing
/// observer.on_tool_call("Bash", r#"{"command": "ls"}"#); // does nothing
/// ```
pub struct NoOpObserver;

impl AgentObserver for NoOpObserver {}

/// An observer that logs all lifecycle events via `tracing`.
///
/// Emits a structured log line for each [`AgentObserver`] callback. The
/// log levels are chosen so that normal operation is visible at `info`
/// level while detailed tool execution is available at `debug` level:
///
/// | Event                     | Level            |
/// |---------------------------|------------------|
/// | session start / end       | `info`           |
/// | context compaction        | `info`           |
/// | turn start / end          | `info` / `debug` |
/// | tool call / complete      | `debug`          |
/// | context warning           | `warn`           |
/// | fallback triggered        | `warn`           |
/// | errors / failure reasons  | `warn`           |
///
/// Query strings in [`on_turn_start`](AgentObserver::on_turn_start) are
/// previewed to a maximum of 80 characters to avoid flooding logs with
/// long prompts.
///
/// # Structured Fields
///
/// Every log line attaches machine-readable fields (`session_id`, `tool`,
/// `duration`, `used_tokens`, etc.) so that log aggregation backends
/// can filter and chart without parsing free-form text.
///
/// # Thread Safety
///
/// [`LoggingObserver`] is a zero-sized unit struct — it carries no state
/// and can be freely shared across threads or cloned at zero cost.
///
/// # Example
///
/// ```
/// use loopctl::builtin::observer::LoggingObserver;
/// use loopctl::core::AgentObserver;
///
/// let observer = LoggingObserver;
/// observer.on_session_start(uuid::Uuid::new_v4());
/// ```
pub struct LoggingObserver;

impl AgentObserver for LoggingObserver {
    /// Log session start at `info` level.
    ///
    /// Emits the `session_id` as a structured field for correlation in
    /// log aggregation systems.
    fn on_session_start(&self, session_id: uuid::Uuid) {
        tracing::info!(%session_id, "Session started");
    }

    /// Log session end at `info` or `warn` level depending on outcome.
    ///
    /// Successful sessions are logged at `info`; failures and sessions
    /// with notes are logged at `warn` with the error message attached.
    fn on_session_end(&self, success: bool, error: Option<&str>) {
        match (success, error) {
            (true, None) => tracing::info!("Session ended successfully"),
            (true, Some(e)) => tracing::info!(error = e, "Session ended with note"),
            (false, Some(e)) => tracing::warn!(error = e, "Session failed"),
            (false, None) => tracing::warn!("Session ended unsuccessfully"),
        }
    }

    /// Log turn start at `info` level with a truncated query preview.
    ///
    /// The query is previewed to at most 80 characters to keep log lines
    /// readable when dealing with long prompts.
    fn on_turn_start(&self, query: &str) {
        let preview = if query.len() > 80 {
            let end = query
                .char_indices()
                .take_while(|(i, _)| *i < 80)
                .last()
                .map_or(0, |(i, c)| i.saturating_add(c.len_utf8()));
            &query[..end]
        } else {
            query
        };
        tracing::info!(query = preview, "Turn started");
    }

    /// Log turn completion at `debug` (success) or `warn` (failure) level.
    ///
    /// Successful turns emit a short `debug` line; failed turns include the
    /// error reason at `warn` level for visibility in production alerts.
    fn on_turn_end(&self, success: bool, error_reason: Option<&str>) {
        if success {
            tracing::debug!("Turn completed");
        } else {
            tracing::warn!(reason = error_reason.unwrap_or("unknown"), "Turn failed");
        }
    }

    /// Log a tool invocation at `debug` level.
    ///
    /// Emits the tool name as a structured field. The input payload is
    /// accepted but not logged to avoid leaking sensitive data (e.g. file
    /// contents, API keys).
    fn on_tool_call(&self, tool: &str, _input: &str) {
        tracing::debug!(tool, "Tool called");
    }

    /// Log tool completion at `debug` (success) or `warn` (failure) level.
    ///
    /// Includes the tool name, wall-clock `duration`, and — on failure —
    /// the error message. The `duration` field is useful for identifying
    /// slow tools in production.
    fn on_tool_complete(
        &self,
        tool: &str,
        _input: &str,
        _output: &str,
        duration: Duration,
        success: bool,
        error: Option<&str>,
    ) {
        if success {
            tracing::debug!(tool, ?duration, "Tool completed");
        } else {
            tracing::warn!(
                tool,
                ?duration,
                error = error.unwrap_or("unknown"),
                "Tool failed"
            );
        }
    }

    /// Log a context-window warning at `warn` level.
    ///
    /// Emits the `used_tokens` and `remaining_tokens` counts so operators
    /// can correlate context pressure with agent behaviour.
    fn on_context_warning(&self, used_tokens: u64, remaining_tokens: u64) {
        tracing::warn!(used_tokens, remaining_tokens, "Context window running low");
    }

    /// Log a context compaction event at `info` level.
    ///
    /// Emits the message counts before and after so operators can verify
    /// that compaction is reducing context size as expected.
    fn on_compaction(&self, messages_before: usize, messages_after: usize) {
        tracing::info!(messages_before, messages_after, "Context compacted");
    }

    /// Log a model fallback event at `warn` level.
    ///
    /// Emits both the `from` and `to` model identifiers so operators can
    /// detect chronic primary-model failures.
    fn on_fallback(&self, from: &str, to: &str) {
        tracing::warn!(from, to, "Fallback triggered");
    }
}

/// A composite observer that fans out every event to multiple inner observers.
///
/// Holds an ordered list of [`AgentObserver`] trait objects behind `Arc`
/// and forwards each callback to every inner observer in sequence. This
/// implements the classic **Composite** pattern, allowing you to combine
/// logging, metrics, and custom observers without writing a new struct.
///
/// Observers are called in insertion order. If any individual observer
/// panics, the remaining observers in the list are still called — this
/// is achieved internally via [`std::panic::catch_unwind`] (or the
/// caller's panic handler). This ensures one misbehaving observer does
/// not prevent others from receiving events.
///
/// # Architecture
///
/// ```text
///                MultiObserver
///              ┌──────────────┐
/// on_xxx() ──► │ observers[]  │──► obs[0].on_xxx()
///              │              │──► obs[1].on_xxx()
///              │              │──► obs[2].on_xxx()
///              └──────────────┘
/// ```
///
/// # Thread Safety
///
/// Each inner observer is stored as [`Arc`]`<dyn `[`AgentObserver`]`>`, so the
/// same observer can be shared across multiple [`MultiObserver`] instances
/// or even other parts of the system. The fan-out loop borrows `&self`,
/// meaning all callbacks must be `&self`-safe (no `&mut self`).
///
/// # Example
///
/// ```
/// use loopctl::builtin::observer::{LoggingObserver, MultiObserver, NoOpObserver};
/// use loopctl::core::AgentObserver;
/// use std::sync::Arc;
///
/// // Build with owned observers:
/// let multi = MultiObserver::new()
///     .with(LoggingObserver)
///     .with(NoOpObserver);
/// assert_eq!(multi.len(), 2);
///
/// // Build with pre-Arc'd observers:
/// let shared = Arc::new(LoggingObserver);
/// let multi = MultiObserver::new()
///     .with_arc(shared.clone())
///     .with_arc(shared);  // same observer twice
/// assert_eq!(multi.len(), 2);
/// ```
pub struct MultiObserver {
    /// The ordered list of inner observers to fan out to.
    ///
    /// Each observer is stored as `Arc<dyn AgentObserver>` so it can be
    /// shared across threads if needed. Observers are called in insertion
    /// order — first added is first notified.
    ///
    /// Starts empty when created via [`new`](MultiObserver::new) or
    /// [`default`](MultiObserver::default). Observers are added via
    /// [`with`](MultiObserver::with) or [`with_arc`](MultiObserver::with_arc).
    observers: Vec<Arc<dyn AgentObserver>>,
}

// ===================================================
// Construction
// ===================================================

impl MultiObserver {
    /// Create an empty multi-observer with no inner observers.
    ///
    /// All callbacks will be no-ops until observers are added via
    /// [`with`](MultiObserver::with) or [`with_arc`](MultiObserver::with_arc).
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::MultiObserver;
    ///
    /// let multi = MultiObserver::new();
    /// assert!(multi.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            observers: Vec::new(),
        }
    }

    /// Add an owned observer to the fan-out list.
    ///
    /// The observer is boxed as `Arc<dyn AgentObserver>` and appended to
    /// the end of the list. Returns `self` for chaining.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::{LoggingObserver, MultiObserver, NoOpObserver};
    ///
    /// let multi = MultiObserver::new()
    ///     .with(LoggingObserver)
    ///     .with(NoOpObserver);
    /// assert_eq!(multi.len(), 2);
    /// ```
    #[must_use]
    pub fn with<O: AgentObserver + 'static>(mut self, observer: O) -> Self {
        self.observers.push(Arc::new(observer));
        self
    }

    /// Add an observer that is already behind an `Arc`.
    ///
    /// Useful when multiple [`MultiObserver`] instances need to share the
    /// same inner observer, or when the observer is constructed externally
    /// and already wrapped in an `Arc`. Accepts both `Arc<ConcreteType>`
    /// and `Arc<dyn AgentObserver>`.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::{LoggingObserver, MultiObserver};
    /// use std::sync::Arc;
    ///
    /// // Arc a concrete type:
    /// let shared = Arc::new(LoggingObserver);
    /// let multi = MultiObserver::new()
    ///     .with_arc(shared.clone())
    ///     .with_arc(shared);
    /// assert_eq!(multi.len(), 2);
    /// ```
    #[must_use]
    pub fn with_arc<O: AgentObserver + 'static>(mut self, observer: Arc<O>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Number of observers in the fan-out list.
    ///
    /// Returns the count of inner observers that will receive events.
    /// Mirrors the `Vec::len` of the internal observers list.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::{LoggingObserver, MultiObserver};
    ///
    /// let multi = MultiObserver::new().with(LoggingObserver);
    /// assert_eq!(multi.len(), 1);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.observers.len()
    }

    /// Whether there are no observers in the fan-out list.
    ///
    /// Returns `true` when [`len`](MultiObserver::len) is zero. When empty,
    /// all callbacks are effectively no-ops (the internal loop body never
    /// executes).
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::MultiObserver;
    ///
    /// let multi = MultiObserver::new();
    /// assert!(multi.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }
}

impl Default for MultiObserver {
    /// Returns an empty multi-observer, equivalent to [`new`](MultiObserver::new).
    ///
    /// The default instance contains zero inner observers, so all callbacks
    /// are effectively no-ops until observers are added.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::builtin::observer::MultiObserver;
    ///
    /// let multi = MultiObserver::default();
    /// assert!(multi.is_empty());
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================
// AgentObserver implementation
// ===================================================

impl AgentObserver for MultiObserver {
    fn on_session_start(&self, session_id: uuid::Uuid) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_session_start(session_id);
            })) {
                warn!("observer panicked in on_session_start: {payload:?}");
            }
        }
    }

    fn on_session_end(&self, success: bool, error: Option<&str>) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_session_end(success, error);
            })) {
                warn!("observer panicked in on_session_end: {payload:?}");
            }
        }
    }

    fn on_turn_start(&self, query: &str) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_turn_start(query);
            })) {
                warn!("observer panicked in on_turn_start: {payload:?}");
            }
        }
    }

    fn on_turn_end(&self, success: bool, error_reason: Option<&str>) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_turn_end(success, error_reason);
            })) {
                warn!("observer panicked in on_turn_end: {payload:?}");
            }
        }
    }

    fn on_tool_call(&self, tool: &str, input: &str) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_tool_call(tool, input);
            })) {
                warn!("observer panicked in on_tool_call: {payload:?}");
            }
        }
    }

    fn on_tool_complete(
        &self,
        tool: &str,
        input: &str,
        output: &str,
        duration: Duration,
        success: bool,
        error: Option<&str>,
    ) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_tool_complete(tool, input, output, duration, success, error);
            })) {
                warn!("observer panicked in on_tool_complete: {payload:?}");
            }
        }
    }

    fn on_context_warning(&self, used_tokens: u64, remaining_tokens: u64) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_context_warning(used_tokens, remaining_tokens);
            })) {
                warn!("observer panicked in on_context_warning: {payload:?}");
            }
        }
    }

    fn on_compaction(&self, messages_before: usize, messages_after: usize) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_compaction(messages_before, messages_after);
            })) {
                warn!("observer panicked in on_compaction: {payload:?}");
            }
        }
    }

    fn on_fallback(&self, from: &str, to: &str) {
        for obs in &self.observers {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                obs.on_fallback(from, to);
            })) {
                warn!("observer panicked in on_fallback: {payload:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_observer_does_not_panic() {
        let observer = NoOpObserver;
        observer.on_session_start(uuid::Uuid::new_v4());
        observer.on_session_end(true, None);
        observer.on_turn_start("test query");
        observer.on_turn_end(true, None);
        observer.on_tool_call("Read", r#"{"file_path": "test.rs"}"#);
        observer.on_tool_complete("Read", "input", "output", Duration::ZERO, true, None);
        observer.on_context_warning(1000, 500);
        observer.on_compaction(10, 5);
        observer.on_fallback("model-a", "model-b");
    }

    #[test]
    fn logging_observer_does_not_panic() {
        let observer = LoggingObserver;
        observer.on_session_start(uuid::Uuid::new_v4());
        observer.on_session_end(true, None);
        observer.on_session_end(false, Some("test error"));
        observer.on_turn_start("test query");
        observer.on_turn_end(true, None);
        observer.on_turn_end(false, Some("some reason"));
        observer.on_tool_call("Bash", r#"{"command": "ls"}"#);
        observer.on_tool_complete(
            "Bash",
            "input",
            "output",
            Duration::from_millis(100),
            true,
            None,
        );
        observer.on_tool_complete(
            "Bash",
            "input",
            "",
            Duration::from_millis(50),
            false,
            Some("command failed"),
        );
        observer.on_context_warning(5000, 2000);
        observer.on_compaction(20, 10);
        observer.on_fallback("primary", "fallback");
    }

    #[test]
    fn multi_observer_fans_out() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));

        struct CountingObserver {
            count: Arc<AtomicUsize>,
        }

        impl AgentObserver for CountingObserver {
            fn on_session_start(&self, _session_id: uuid::Uuid) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
        }

        let multi = MultiObserver::new()
            .with(CountingObserver {
                count: call_count.clone(),
            })
            .with(CountingObserver {
                count: call_count.clone(),
            })
            .with(CountingObserver {
                count: call_count.clone(),
            });

        multi.on_session_start(uuid::Uuid::new_v4());
        assert_eq!(call_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn multi_observer_default_is_empty() {
        let multi = MultiObserver::default();
        assert!(multi.is_empty());
        assert_eq!(multi.len(), 0);
    }

    #[test]
    fn multi_observer_with_arc() {
        let multi = MultiObserver::new().with_arc(Arc::new(NoOpObserver));
        assert_eq!(multi.len(), 1);
    }
}
