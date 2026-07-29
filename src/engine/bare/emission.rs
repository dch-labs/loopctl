//! Run lifecycle notifications — start and end events.
//!
//! Fires observer callbacks and hook notifications when a run begins and
//! ends. Other observer events (`on_turn_start`, `on_response`, etc.) are fired
//! directly at their call sites in the driver loop.

use super::{ApiClient, BareLoop, Duration, LoopError, Run};
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    RunEndContext as HookRunEndContext, RunEndReason, RunStartContext as HookRunStartContext,
};
use crate::observer::{RunEndContext, RunStartContext};

impl<C: ApiClient> BareLoop<C> {
    /// Notify all observers and hooks that a run has started.
    ///
    /// Fires [`on_run_start`](crate::observer::LoopObserver::on_run_start)
    /// on every registered observer with the session id, then fires the
    /// `on_run_start` hook (when the `hooks` feature is enabled and a
    /// hook executor is configured). Called once at the beginning of a
    /// run, before the first turn.
    pub(super) fn notify_run_start(&self) {
        self.managers.observers().on_run_start(&RunStartContext {
            session_id: self.session.id,
        });
        self.notify_run_start_hook();
    }

    /// Notify hooks and observers that a run has ended.
    ///
    /// Fires `on_run_end` hooks first (when the `hooks` feature is
    /// enabled), then fires
    /// [`on_run_end`](crate::observer::LoopObserver::on_run_end) on
    /// every registered observer. The [`Run`] supplies per-run totals;
    /// `error` carries the terminal [`LoopError`] when the run failed,
    /// or `None` on success; `duration` is the wall-clock run length.
    pub(super) fn notify_run_end(
        &self,
        result: &Run,
        duration: Duration,
        error: Option<&LoopError>,
    ) {
        self.notify_run_end_hook(result, error, duration);
        self.managers.observers().on_run_end(&RunEndContext {
            success: error.is_none(),
            error: error.map_or_else(|| None, |e| Some(e.to_string())),
            total_turns: result.turn_count(),
            duration_ms: Self::millis_u64(duration),
        });
    }

    /// Derive the structured [`RunEndReason`] from the terminal error.
    ///
    /// Maps the authoritative terminal [`LoopError`] carried out of
    /// [`run`](crate::engine::core::Loop::run) — never the turn count,
    /// since a run that legitimately completes on exactly the
    /// `max_turns`-th turn reaches `error = None` and must read as
    /// [`Complete`](RunEndReason::Complete), not `MaxTurns`. Cancellation
    /// (signalled or carried by [`LoopError::Cancelled`]) takes
    /// precedence; then [`LoopError::ContextExceeded`] maps to
    /// [`ContextOverflow`](RunEndReason::ContextOverflow),
    /// [`LoopError::MaxTurnsExceeded`] to
    /// [`MaxTurns`](RunEndReason::MaxTurns), any other error to
    /// [`Error`](RunEndReason::Error), and `None` to
    /// [`Complete`](RunEndReason::Complete).
    #[cfg(feature = "hooks")]
    fn run_end_reason(&self, error: Option<&LoopError>) -> RunEndReason {
        if self.is_cancelled() {
            return RunEndReason::Cancelled;
        }
        match error {
            Some(LoopError::ContextExceeded { .. }) => RunEndReason::ContextOverflow,
            Some(LoopError::MaxTurnsExceeded { .. }) => RunEndReason::MaxTurns,
            Some(LoopError::Cancelled) => RunEndReason::Cancelled,
            Some(_) => RunEndReason::Error,
            None => RunEndReason::Complete,
        }
    }

    /// Fire the `on_run_start` hook when a hook executor is
    /// configured.
    ///
    /// Builds a [`HookRunStartContext`] from the session id, the
    /// client's current model, and the process working directory, then
    /// dispatches it to every registered run-start hook. No-op when
    /// no hook executor is set.
    #[cfg(feature = "hooks")]
    fn notify_run_start_hook(&self) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let ctx = HookRunStartContext {
            session_id: self.session.id,
            model: self.client.model(),
            working_directory: std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        executor.notify_run_start(&ctx);
    }

    /// No-op run-start hook for builds without the `hooks` feature.
    ///
    /// Does nothing — there are no hooks to notify. Kept so
    /// [`notify_run_start`](Self::notify_run_start) compiles
    /// identically with and without the feature.
    #[cfg(not(feature = "hooks"))]
    fn notify_run_start_hook(&self) {}

    /// Fire the `on_run_end` hook when a hook executor is
    /// configured.
    ///
    /// Derives the [`RunEndReason`] via
    /// [`run_end_reason`](Self::run_end_reason), then builds a
    /// [`HookRunEndContext`] from the session id, reason, turn
    /// count, total tokens, and run duration, and dispatches it to
    /// every registered run-end hook. No-op when no hook executor
    /// is set.
    #[cfg(feature = "hooks")]
    fn notify_run_end_hook(&self, result: &Run, error: Option<&LoopError>, duration: Duration) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let reason = self.run_end_reason(error);
        let ctx = HookRunEndContext {
            session_id: self.session.id,
            reason,
            total_turns: result.turn_count(),
            total_tokens: result.total_tokens(),
            duration_secs: duration.as_secs(),
        };
        executor.notify_run_end(&ctx);
    }

    /// No-op run-end hook for builds without the `hooks` feature.
    ///
    /// Does nothing — there are no hooks to notify. Kept so
    /// [`notify_run_end`](Self::notify_run_end) compiles
    /// identically with and without the feature.
    #[cfg(not(feature = "hooks"))]
    fn notify_run_end_hook(&self, _result: &Run, _error: Option<&LoopError>, _duration: Duration) {}

    /// Convert a [`Duration`] to milliseconds as a `u64`.
    ///
    /// Saturates at `u64::MAX` if the duration exceeds the `u64` range
    /// (only possible with a platform-specific `u128` millisecond count
    /// far beyond any realistic run length). Used to populate the
    /// observer/hook end-context `duration_ms` fields from a
    /// [`Duration`].
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }
}
