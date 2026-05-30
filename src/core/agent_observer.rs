//! Agent observer trait — lifecycle event hooks for monitoring agents.
//!
//! Observers receive callbacks at key lifecycle points.
//! This enables logging, metrics, trajectory recording,
//! and other cross-cutting concerns without modifying agent logic.
//!
//! This is the **canonical** observer trait, shared between the framework
//! and production agent crates. All methods have default no-op implementations,
//! so consumers only override what they need.
//!
//! # Provided Implementations
//!
//! - **`NoOpObserver`** — Default no-op (does nothing).
//! - **`LoggingObserver`** — Logs all events via `tracing`.
//! - **`MultiObserver`** — Fans out to multiple observers.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::core::agent_observer::AgentObserver;
//! use std::time::Duration;
//!
//! struct MetricsObserver;
//!
//! impl AgentObserver for MetricsObserver {
//!     fn on_turn_start(&self, _query: &str) {}
//!     fn on_tool_complete(&self, tool: &str, _input: &str, _output: &str,
//!                         duration: Duration, success: bool, _error: Option<&str>) {}
//!     fn on_session_end(&self, success: bool, _error: Option<&str>) {}
//! }
//! ```

use std::time::Duration;

/// Observer trait for agent lifecycle events.
///
/// All methods have default no-op implementations, so consumers
/// only override what they need. Implementors must be [`Send + Sync`]
/// because observers are shared across async tasks.
///
/// # Lifecycle
///
/// ```text
/// on_session_start(session_id)
///   → on_turn_start(query)                [once per turn]
///     → on_tool_call(tool, input)          [before each tool]
///     → on_tool_complete(tool, ...)        [after each tool]
///   → on_turn_end(success, error)          [once per turn]
///   → ...
/// on_session_end(success, error)
/// ```
///
/// # Implementing
///
/// Override only the methods you care about. Every method has a no-op
/// default, so a minimal observer can be just an empty `impl`:
///
/// ```
/// use loopctl::core::agent_observer::AgentObserver;
///
/// struct MyObserver;
/// impl AgentObserver for MyObserver {} // all methods are no-ops
/// ```
///
/// # Example
///
/// ```
/// use loopctl::core::agent_observer::AgentObserver;
/// use std::time::Duration;
///
/// struct LoggingObserver;
///
/// impl AgentObserver for LoggingObserver {
///     fn on_turn_start(&self, _query: &str) {}
///
///     fn on_tool_complete(&self, tool: &str, _input: &str, _output: &str,
///                         duration: Duration, _success: bool, _error: Option<&str>) {}
/// }
/// ```
pub trait AgentObserver: Send + Sync {
    /// Called when an agent session begins.
    ///
    /// Fired once at the start of `AgentCore::initialize`. The
    /// `session_id` matches the one passed to the agent configuration
    /// and can be used to correlate all subsequent events for this session.
    fn on_session_start(&self, _session_id: uuid::Uuid) {}

    /// Called when an agent session ends.
    ///
    /// Fired once after `AgentCore::finalize`. `success` indicates
    /// whether the session completed normally; `error` contains a
    /// description when the session ended due to a failure.
    fn on_session_end(&self, _success: bool, _error: Option<&str>) {}

    /// Called at the start of processing a turn.
    ///
    /// Fired before `AgentCore::process_turn` is invoked. The `query`
    /// is the raw user input for this turn.
    fn on_turn_start(&self, _query: &str) {}

    /// Called when a turn completes.
    ///
    /// Fired after `AgentCore::process_turn` returns. `success`
    /// indicates whether the turn completed without error; when
    /// `false`, `error_reason` describes what went wrong.
    fn on_turn_end(&self, _success: bool, _error_reason: Option<&str>) {}

    /// Called before a tool is executed.
    ///
    /// Fired just before the framework dispatches a tool call. `tool`
    /// is the tool name and `input` is the raw input string. Use this
    /// to log tool usage or record the start time for custom timing.
    fn on_tool_call(&self, _tool: &str, _input: &str) {}

    /// Called after a tool completes execution.
    ///
    /// Fired after the tool returns. The parameters provide the full
    /// execution context:
    ///
    /// - `tool` — Tool name that was invoked.
    /// - `input` — Raw input string passed to the tool.
    /// - `output` — Result string returned by the tool.
    /// - `duration` — Wall-clock execution time.
    /// - `success` — Whether the tool reported success.
    /// - `error` — Error message if the tool failed, `None` otherwise.
    fn on_tool_complete(
        &self,
        _tool: &str,
        _input: &str,
        _output: &str,
        _duration: Duration,
        _success: bool,
        _error: Option<&str>,
    ) {
    }

    /// Called when the context window is running low.
    ///
    /// Fired when the number of remaining tokens drops below a
    /// configurable threshold defined in the agent configuration.
    /// Use this to trigger
    /// pre-emptive compaction or warn downstream consumers.
    fn on_context_warning(&self, _used_tokens: u64, _remaining_tokens: u64) {}

    /// Called when a context compaction occurs.
    ///
    /// Fired after the framework compresses the conversation history.
    /// `messages_before` and `messages_after` indicate how aggressive
    /// the compaction was — useful for monitoring whether compaction
    /// is losing too much context.
    fn on_compaction(&self, _messages_before: usize, _messages_after: usize) {}

    /// Called when a fallback is triggered (e.g. switching models).
    ///
    /// Fired by the fallback manager when the primary model fails
    /// and the framework switches to a backup.
    /// `from` is the model that failed; `to` is the replacement.
    fn on_fallback(&self, _from: &str, _to: &str) {}

    /// Called when a loop is detected in tool operations.
    ///
    /// Fired by the detection manager when the same tool call pattern
    /// has been repeated beyond the configured loop threshold.
    /// `tool` is the repeating tool name; `repetitions` is the count.
    fn on_loop_detected(&self, _tool: &str, _repetitions: usize) {}

    /// Called when convergence is detected in agent responses.
    ///
    /// Fired by the detection manager when recent agent responses
    /// have become semantically similar beyond the configured threshold.
    /// `action` describes the configured response (e.g. `"stop"`, `"warn"`).
    fn on_convergence_detected(&self, _action: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn default_impl_noop_does_not_panic() {
        struct Nop;
        impl AgentObserver for Nop {}

        let nop = Nop;
        nop.on_session_start(uuid::Uuid::nil());
        nop.on_session_end(true, None);
        nop.on_turn_start("hello");
        nop.on_turn_end(true, None);
        nop.on_tool_call("read_file", "/tmp/x");
        nop.on_tool_complete(
            "read_file",
            "in",
            "out",
            Duration::from_millis(5),
            true,
            None,
        );
        nop.on_context_warning(100_000, 28_000);
        nop.on_compaction(50, 20);
        nop.on_fallback("llm-1", "llm-2");
    }

    #[test]
    fn empty_impl_is_send_sync() {
        struct EmptyObs;
        impl AgentObserver for EmptyObs {}

        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmptyObs>();
    }
}
