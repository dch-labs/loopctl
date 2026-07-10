//! Session lifecycle notifications — start and end events.
//!
//! Fires observer callbacks and hook notifications when a session begins and
//! ends. Other observer events (`on_turn_start`, `on_response`, etc.) are fired
//! directly at their call sites in `process_turn`.

use super::{ApiClient, BareLoop, Duration, SessionResult};
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    SessionEndContext as HookSessionEndContext, SessionEndReason,
    SessionStartContext as HookSessionStartContext,
};
use crate::observer::{SessionEndContext, SessionStartContext};

// ==================================================
// Session lifecycle notifications
// ==================================================

impl<C: ApiClient> BareLoop<C> {
    /// Notify all observers and hooks that the session has started.
    pub(super) fn notify_session_start(&self) {
        self.managers
            .observers()
            .on_session_start(&SessionStartContext {
                session_id: self.config.session_id,
            });
        self.notify_session_start_hook();
    }

    /// Notify all observers and hooks that the session has ended.
    pub(super) fn notify_session_end(&self, result: &SessionResult, duration: Duration) {
        self.managers
            .observers()
            .on_session_end(&SessionEndContext {
                success: result.success,
                error: result.error.clone(),
                total_turns: result.total_turns,
                duration_ms: Self::millis_u64(duration),
            });
        self.notify_session_end_hook(result, duration);
    }

    /// Derive the structured [`SessionEndReason`] from the loop's
    /// terminal state.
    ///
    /// Unlike a simple `success` boolean, this distinguishes
    /// cancellation, max-turns exhaustion, and context overflow.
    #[cfg(feature = "hooks")]
    fn session_end_reason(&self, success: bool) -> SessionEndReason {
        if self.is_cancelled() {
            SessionEndReason::Cancelled
        } else if !success {
            if self
                .budget
                .error
                .as_ref()
                .is_some_and(|e| e.contains("context") || e.contains("overflow"))
            {
                SessionEndReason::ContextOverflow
            } else {
                SessionEndReason::Error
            }
        } else if self.budget.total_turns >= self.config.max_turns {
            SessionEndReason::MaxTurns
        } else {
            SessionEndReason::Complete
        }
    }

    #[cfg(feature = "hooks")]
    fn notify_session_start_hook(&self) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let ctx = HookSessionStartContext {
            session_id: self.config.session_id,
            model: self.config.model.clone(),
            working_directory: std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        executor.notify_session_start(&ctx);
    }

    #[cfg(not(feature = "hooks"))]
    fn notify_session_start_hook(&self) {}

    #[cfg(feature = "hooks")]
    fn notify_session_end_hook(&self, result: &SessionResult, duration: Duration) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let reason = self.session_end_reason(result.success);
        let ctx = HookSessionEndContext {
            session_id: result.session_id,
            reason,
            total_turns: result.total_turns,
            total_tokens: result.input_tokens.saturating_add(result.output_tokens),
            duration_secs: duration.as_secs(),
        };
        executor.notify_session_end(&ctx);
    }

    #[cfg(not(feature = "hooks"))]
    fn notify_session_end_hook(&self, _result: &SessionResult, _duration: Duration) {}

    /// Convert a [`Duration`] to milliseconds as `u64`.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }
}
