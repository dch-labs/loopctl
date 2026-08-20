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

/// The result of servicing a [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact),
/// before it is fed back into the machine.
///
/// Distinguishes a pass that rewrote the history from one that left it
/// unchanged, because the machine treats the two differently: a rewritten
/// history replaces the committed one and clears the pending buffer
/// ([`LoopMachine::compaction_result`](crate::engine::core::LoopMachine::compaction_result)),
/// while an unchanged history must leave both alone
/// ([`LoopMachine::compaction_noop`](crate::engine::core::LoopMachine::compaction_noop))
/// — committing the in-flight run's partial messages mid-run would make a
/// later failure un-discardable.
pub(super) enum CompactStepOutcome {
    /// Compaction rewrote the history.
    ///
    /// Carries the compacted message list and its post-compaction token
    /// estimate, ready to feed into
    /// [`LoopMachine::compaction_result`](crate::engine::core::LoopMachine::compaction_result).
    Compacted(Vec<Message>, u64),

    /// Compaction changed nothing.
    ///
    /// No compactor ran, a pre-compact hook vetoed the pass, or the manager
    /// reported no action. Carries the driver's measured estimate of the
    /// unchanged conversation, ready to feed into
    /// [`LoopMachine::compaction_noop`](crate::engine::core::LoopMachine::compaction_noop).
    Unchanged(u64),
}

impl<C: ApiClient> BareLoop<C> {
    /// Compact the conversation context owned by the driving machine.
    ///
    /// Run when the machine requests it via
    /// [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact).
    /// Reads the history from [`LoopMachine::history`](crate::engine::core::LoopMachine::history),
    /// asks the configured [`ContextManager`](crate::compact::ContextManager)
    /// to reduce it, fires
    /// [`on_compaction`](crate::observer::LoopObserver::on_compaction) and the
    /// post-compact hook when compaction occurred, and returns a
    /// [`CompactStepOutcome`] for the driver to feed back into the machine.
    ///
    /// When no `ContextManager` is set, a pre-compact hook aborts the pass,
    /// or the manager reports [`EnsureContextResult::NoAction`], the
    /// conversation is returned unchanged with a *measured* estimate of its
    /// size — never a hard-coded zero — so the machine's no-progress guard
    /// sees the true size and terminates with a typed error when compaction
    /// cannot reduce instead of silently looping.
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
    ) -> Result<CompactStepOutcome, LoopError> {
        let history = self.machine.full_history();
        let Some(ctx_manager) = self.managers.context_manager() else {
            let tokens_after = self.count_context(&history);
            return Ok(CompactStepOutcome::Unchanged(tokens_after));
        };

        #[cfg(feature = "hooks")]
        if self.pre_compact_hook_aborts(&history) {
            let tokens_after = self.count_context(&history);
            return Ok(CompactStepOutcome::Unchanged(tokens_after));
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
                Ok(CompactStepOutcome::Compacted(
                    outcome.messages,
                    tokens_after,
                ))
            }
            Ok(EnsureContextResult::NoAction(messages)) => {
                let tokens_after = self.count_context(&messages);
                Ok(CompactStepOutcome::Unchanged(tokens_after))
            }
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
