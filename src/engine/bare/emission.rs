//! Observer notifications and event-sink emissions.
//!
//! Split from [`BareLoop`] for clarity — these methods are thin wrappers
//! that fan out to every registered [`AgentObserver`] and/or the
//! [`EventSink`]. Keeping them in a dedicated file makes the emission
//! surface area easy to audit and prevents the main loop file from
//! ballooning with boilerplate.
//!
//! Two categories:
//!
//! - **`notify_*`** — iterate `observers` and optionally run hooks.
//! - **`emit_*`** — send a single [`ObserveEvent`] to the [`EventSink`].

use super::{ApiClient, BareLoop, Duration, EndReason, ObserveEvent, SessionEndInfo};
#[cfg(feature = "hooks")]
use crate::hooks::context::{SessionEndContext, SessionEndReason, SessionStartContext};

// ==================================================
// Observer notifications
// ==================================================

impl<C: ApiClient> BareLoop<C> {
    /// Notify all observers that the session has started.
    ///
    /// Called once at the beginning of [`run()`](BareLoop::run),
    /// before the first turn. Iterates over every registered
    /// [`AgentObserver`] and calls
    /// [`on_session_start()`](AgentObserver::on_session_start) with the
    /// session ID from [`AgentConfig`].
    pub(super) fn notify_session_start(&self) {
        for obs in &self.observers {
            obs.on_session_start(self.config.session_id);
        }
        self.event_sink.on_event(&ObserveEvent::SessionStart {
            session_id: self.config.session_id,
        });

        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let ctx = SessionStartContext {
                session_id: self.config.session_id,
                model: self.config.model.clone(),
                working_directory: std::env::current_dir()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            };
            executor.notify_session_start(&ctx);
        }
    }

    /// Notify all observers that the session has ended.
    ///
    /// Called once when [`run()`](BareLoop::run) returns — whether
    /// successfully, due to an error, or because of cancellation.
    pub(super) fn notify_session_end(&self, info: &SessionEndInfo) {
        let reason_str = match &info.reason {
            EndReason::Complete => None,
            EndReason::Cancelled => Some("cancelled"),
            EndReason::MaxTurns => Some("max turns exceeded"),
            EndReason::Error => Some("session ended with error"),
        };
        for obs in &self.observers {
            obs.on_session_end(info.success, reason_str);
        }

        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let reason = match &info.reason {
                EndReason::Complete => SessionEndReason::Complete,
                EndReason::Cancelled => SessionEndReason::Cancelled,
                EndReason::Error => SessionEndReason::Error,
                EndReason::MaxTurns => SessionEndReason::MaxTurns,
            };
            let ctx = SessionEndContext {
                session_id: self.config.session_id,
                reason,
                total_turns: info.total_turns,
                total_tokens: info.total_tokens,
                duration_secs: info.duration_secs,
            };
            executor.notify_session_end(&ctx);
        }
    }

    /// Notify all observers that a turn has started.
    ///
    /// Called at the top of every iteration of the main loop, before
    /// the API streaming request. The `query` parameter is the original
    /// user input (the same for every turn within a single
    /// [`run()`](BareLoop::run) call).
    pub(super) fn notify_turn_start(&self, query: &str) {
        for obs in &self.observers {
            obs.on_turn_start(query);
        }
    }

    /// Notify all observers that a turn has ended.
    ///
    /// Called after each turn completes — whether it produced a tool
    /// call, ended with text, or encountered an error. The `success`
    /// flag is `true` for normal turns and `false` when the API stream
    /// returned an error.
    pub(super) fn notify_turn_end(&self, success: bool, error: Option<&str>) {
        for obs in &self.observers {
            obs.on_turn_end(success, error);
        }
    }

    /// Notify all observers that a tool is about to be invoked.
    ///
    /// Called just before the tool's [`call()`](crate::tool::Tool::call)
    /// method is invoked. The `tool` parameter is the tool name and
    /// `input` is the JSON input serialized to a string.
    pub(super) fn notify_tool_call(&self, tool: &str, input: &str) {
        for obs in &self.observers {
            obs.on_tool_call(tool, input);
        }
    }

    /// Notify all observers that a tool invocation has completed.
    ///
    /// Called after the tool's [`call()`](crate::tool::Tool::call)
    /// method returns — whether successfully or with an error.
    /// Includes the tool's output, execution `duration`, a `success`
    /// flag, and an optional `error` message.
    ///
    /// Parameter count is dictated by the [`AgentObserver::on_tool_complete`]
    /// trait method.
    pub(super) fn notify_tool_complete(
        &self,
        tool: &str,
        input: &str,
        output: &str,
        duration: Duration,
        success: bool,
        error: Option<&str>,
    ) {
        for obs in &self.observers {
            obs.on_tool_complete(tool, input, output, duration, success, error);
        }
    }

    // ==================================================
    // EventSink emissions
    // ==================================================

    /// Convert a [`Duration`] to milliseconds as `u64`.
    ///
    /// Clamps at `u64::MAX` if the duration exceeds ~584 million years,
    /// which is safe for any practical agent session.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// Emit a [`TurnStart`](ObserveEvent::TurnStart) event.
    pub(super) fn emit_turn_start(&self, turn: usize, query: &str) {
        self.event_sink.on_event(&ObserveEvent::TurnStart {
            turn,
            query: query.to_string(),
        });
    }

    /// Emit a [`TurnComplete`](ObserveEvent::TurnComplete) event.
    pub(super) fn emit_turn_complete(
        &self,
        turn: usize,
        duration: Duration,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        self.event_sink.on_event(&ObserveEvent::TurnComplete {
            turn,
            duration_ms: Self::millis_u64(duration),
            input_tokens,
            output_tokens,
        });
    }

    /// Emit a [`TurnFailed`](ObserveEvent::TurnFailed) event.
    pub(super) fn emit_turn_failed(&self, turn: usize, duration: Duration, error: &str) {
        self.event_sink.on_event(&ObserveEvent::TurnFailed {
            turn,
            duration_ms: Self::millis_u64(duration),
            error: error.to_string(),
        });
    }

    /// Emit a [`ToolStart`](ObserveEvent::ToolStart) event.
    pub(super) fn emit_tool_start(&self, name: &str, input: &str) {
        self.event_sink.on_event(&ObserveEvent::ToolStart {
            name: name.to_string(),
            input: input.to_string(),
        });
    }

    /// Emit a [`ToolComplete`](ObserveEvent::ToolComplete) event.
    pub(super) fn emit_tool_complete(
        &self,
        name: &str,
        output: &str,
        is_error: bool,
        duration: Duration,
    ) {
        self.event_sink.on_event(&ObserveEvent::ToolComplete {
            name: name.to_string(),
            output: output.to_string(),
            is_error,
            duration_ms: Self::millis_u64(duration),
        });
    }

    /// Emit a [`SessionStop`](ObserveEvent::SessionStop) event.
    pub(super) fn emit_session_stop(
        &self,
        total_turns: usize,
        duration: Duration,
        success: bool,
        reason: &str,
    ) {
        self.event_sink.on_event(&ObserveEvent::SessionStop {
            session_id: self.config.session_id,
            success,
            reason: reason.to_string(),
            total_turns,
            duration_ms: Self::millis_u64(duration),
        });
    }
}
