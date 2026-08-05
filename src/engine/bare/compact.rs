//! Context compaction for agent conversations.
//!
//! The driving state machine decides *when* to compact by emitting
//! [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact); the
//! [`ContextManager`](crate::compact::ContextManager) consulted here decides
//! *how*. [`run_compaction`](BareLoop::run_compaction) runs the configured
//! compactor over the machine-owned history, fires the compaction observer and
//! hook events, and returns the compacted messages plus a post-compaction token
//! estimate for the driver to feed back into the machine.

use super::{ApiClient, BareLoop, LoopError};
#[cfg(feature = "hooks")]
use super::{CompactTrigger, Instant, PostCompactContext, PreCompactContext};
use crate::compact::EnsureContextResult;

use crate::capabilities::Compactable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
use crate::message::Message;
use crate::observer::CompactedContext;

impl<C: ApiClient> BareLoop<C> {
    /// Compact the conversation context owned by the driving machine.
    ///
    /// Run when the machine requests it via
    /// [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact).
    /// Reads the history from [`LoopMachine::history`](crate::engine::core::LoopMachine::history),
    /// asks the configured [`ContextManager`](crate::compact::ContextManager)
    /// to reduce it, fires
    /// [`on_compaction`](crate::observer::LoopObserver::on_compaction) and the
    /// post-compact hook when compaction occurred, and returns the compacted
    /// messages alongside the post-compaction token estimate. The caller feeds
    /// both back via
    /// [`LoopMachine::compaction_result`](crate::engine::core::LoopMachine::compaction_result).
    ///
    /// When no `ContextManager` is set the history is returned unchanged with a
    /// zero token estimate, so the machine can resume without compaction. When
    /// a pre-compact hook aborts compaction the history is likewise returned
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::ContextExceeded`] when compaction was required but
    /// the compactor could not reduce the history enough to fit the context
    /// window.
    pub(super) async fn run_compaction(
        &mut self,
        turn: usize,
        reason: crate::compact::types::CompactReason,
    ) -> Result<(Vec<Message>, u64), LoopError> {
        let history = self.machine.full_history();
        let Some(ctx_manager) = self.managers.context_manager() else {
            return Ok((history, 0));
        };

        #[cfg(feature = "hooks")]
        if self.pre_compact_hook_aborts(&history) {
            return Ok((history, 0));
        }

        #[cfg(feature = "hooks")]
        let messages_before = history.len();
        #[cfg(feature = "hooks")]
        let compact_start = Instant::now();
        let result = ctx_manager.compact_with_reason(history, turn, reason).await;

        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                let tokens_after = outcome.tokens_after;
                let tokens_saved = outcome.tokens_saved;
                #[cfg(feature = "hooks")]
                let messages_after = outcome.messages.len();
                let tokens_before = tokens_after.saturating_add(tokens_saved);
                self.managers.observers().on_compaction(&CompactedContext {
                    tokens_before,
                    tokens_after,
                    tokens_saved,
                });
                #[cfg(feature = "hooks")]
                self.notify_post_compact_hook(
                    messages_before,
                    messages_after,
                    tokens_after,
                    tokens_saved,
                    compact_start.elapsed(),
                );
                Ok((outcome.messages, tokens_after))
            }
            Ok(EnsureContextResult::NoAction(messages)) => Ok((messages, 0)),
            Err(overflow) => Err(LoopError::ContextExceeded {
                used: overflow.tokens_used,
                limit: overflow.context_window,
            }),
        }
    }

    /// Run the pre-compact hook, returning `true` if any hook aborts.
    ///
    /// Builds a [`PreCompactContext`] from the current history and
    /// session config, then asks the
    /// [`HookExecutor`](crate::hooks::HookExecutor) to run every
    /// registered `on_pre_compact` hook. If any hook returns
    /// [`abort: true`](crate::hooks::context::CompactResult::abort),
    /// this returns `true` and [`run_compaction`](Self::run_compaction)
    /// skips compaction entirely for this trigger. Returns `false`
    /// when there is no hook executor or no hook vetoes the compaction.
    #[cfg(feature = "hooks")]
    fn pre_compact_hook_aborts(&self, history: &[Message]) -> bool {
        let Some(executor) = self.managers.hook_executor() else {
            return false;
        };
        let tokens_before = crate::compact::CompactionOutcome::estimate_tokens(history);
        let ctx = PreCompactContext {
            trigger: CompactTrigger::Auto,
            custom_instructions: None,
            message_count: history.len(),
            tokens_before,
            context_window: self.session.config.context_window,
            session_id: self.session.id,
        };
        executor.check_pre_compact(&ctx).abort
    }

    /// Notify the post-compact hook that compaction completed.
    ///
    /// Builds a [`PostCompactContext`] from the before/after message
    /// counts, the post-compaction token estimate, the tokens saved,
    /// and the wall-clock compaction duration, then fires every
    /// registered `on_post_compact` hook via the
    /// [`HookExecutor`](crate::hooks::HookExecutor). No-op when there
    /// is no hook executor. Called only on the
    /// [`EnsureContextResult::Compacted`] path — the `NoAction` path
    /// skips it.
    #[cfg(feature = "hooks")]
    fn notify_post_compact_hook(
        &self,
        messages_before: usize,
        messages_after: usize,
        tokens_after: u64,
        tokens_saved: u64,
        duration: std::time::Duration,
    ) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let ctx = PostCompactContext {
            trigger: CompactTrigger::Auto,
            messages_compacted: messages_before.saturating_sub(messages_after),
            tokens_saved,
            tokens_after,
            duration_ms: Self::millis_u64(duration),
            session_id: self.session.id,
        };
        executor.notify_post_compact(&ctx);
    }
}
