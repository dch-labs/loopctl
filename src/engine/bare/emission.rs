//! Session lifecycle notifications.
//!
//! Split from [`BareLoop`] for clarity — these methods dispatch to
//! the [`ObserverHost`](crate::observer::ObserverHost) and the hook executor.
//!
//! Only session start/end live here because they do *two* things:
//! observer notification + hook dispatch. All other observer notifications
//! are called directly at their call sites via
//! `self.managers.observers().on_*()`.

use super::{ApiClient, BareLoop, Duration, EndReason, SessionEndInfo};
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
        if let Some(ref executor) = self.hook_executor {
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
    pub(super) fn notify_session_end(&self, info: &SessionEndInfo) {
        let reason_str = match &info.reason {
            EndReason::Complete => None,
            EndReason::Cancelled => Some("cancelled"),
            EndReason::MaxTurns => Some("max turns exceeded"),
            EndReason::Error => Some("session ended with error"),
        };
        self.managers
            .observers()
            .on_session_end(&SessionEndContext {
                success: info.success,
                error: reason_str.map(std::string::ToString::to_string),
                total_turns: info.total_turns,
                duration_ms: u64::try_from(
                    std::time::Duration::from_secs(info.duration_secs).as_millis(),
                )
                .unwrap_or(u64::MAX),
            });

        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let reason = match &info.reason {
                EndReason::Complete => SessionEndReason::Complete,
                EndReason::Cancelled => SessionEndReason::Cancelled,
                EndReason::Error => SessionEndReason::Error,
                EndReason::MaxTurns => SessionEndReason::MaxTurns,
            };
            let ctx = HookSessionEndContext {
                session_id: self.config.session_id,
                reason,
                total_turns: info.total_turns,
                total_tokens: info.total_tokens,
                duration_secs: info.duration_secs,
            };
            executor.notify_session_end(&ctx);
        }
    }

    /// Convert a [`Duration`] to milliseconds as `u64`.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }
}
