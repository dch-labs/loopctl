//! Session lifecycle notifications.
//!
//! Split from [`BareLoop`] for clarity — these methods dispatch to
//! the [`ObserverHost`](crate::observer::ObserverHost) and the hook executor.
//!
//! Only session start/end live here because they do *two* things:
//! observer notification + hook dispatch. All other observer notifications
//! are called directly at their call sites via
//! `self.managers.observers().on_*()`.

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

        #[cfg(feature = "hooks")]
        if let Some(executor) = self.managers.hook_executor() {
            let ctx = HookSessionStartContext {
                session_id: self.config.session_id,
                model: self.config.model.clone(),
                working_directory: std::env::current_dir()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            };
            executor.notify_session_start(&ctx);
        }
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

        #[cfg(feature = "hooks")]
        if let Some(executor) = self.managers.hook_executor() {
            let reason = if result.success {
                SessionEndReason::Complete
            } else {
                SessionEndReason::Error
            };
            let ctx = HookSessionEndContext {
                session_id: result.session_id,
                reason,
                total_turns: result.total_turns,
                total_tokens: result.input_tokens.saturating_add(result.output_tokens),
                duration_secs: duration.as_secs(),
            };
            executor.notify_session_end(&ctx);
        }
    }

    /// Convert a [`Duration`] to milliseconds as `u64`.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }
}
