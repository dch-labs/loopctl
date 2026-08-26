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
/// Carries the driver's token measurements of the conversation, both taken
/// with the same counter: ahead of the pass (`tokens_before`) and after it
/// (`tokens_after`). `compacted` decides which machine feed applies: a
/// rewritten history replaces the committed one and clears the pending
/// buffer
/// ([`LoopMachine::compaction_result`](crate::engine::core::LoopMachine::compaction_result)),
/// while an unchanged history must leave both alone
/// ([`LoopMachine::compaction_noop`](crate::engine::core::LoopMachine::compaction_noop))
/// — committing the in-flight run's partial messages mid-run would make a
/// later failure un-discardable.
pub(super) struct CompactStepOutcome {
    /// Measured token size of the full history ahead of the compaction pass.
    ///
    /// Taken by the driver before the compactor runs, so the machine's
    /// no-progress guard compares two real measurements of the same
    /// conversation rather than estimates recorded at different times.
    pub(super) tokens_before: u64,

    /// Measured token size of the conversation after the pass.
    ///
    /// The size of the compacted list when `compacted` is `Some`, or of the
    /// unchanged history otherwise. Equal to `tokens_before` whenever the
    /// pass changed nothing.
    pub(super) tokens_after: u64,

    /// The compacted message list, when the pass rewrote the history.
    ///
    /// `None` when compaction changed nothing — no compactor ran, a
    /// pre-compact hook vetoed the pass, or the manager reported no action.
    pub(super) compacted: Option<Vec<Message>>,
}

impl<C: ApiClient> BareLoop<C> {
    /// Compact the conversation context owned by the driving machine.
    ///
    /// Run when the machine requests it via
    /// [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact).
    /// Reads the history from [`LoopMachine::history`](crate::engine::core::LoopMachine::history),
    /// measures its token size, asks the configured
    /// [`ContextManager`](crate::compact::ContextManager) to reduce it, fires
    /// [`on_compaction`](crate::observer::LoopObserver::on_compaction) and the
    /// post-compact hook when compaction occurred, and returns a
    /// [`CompactStepOutcome`] — the measured before/after pair plus the
    /// compacted list when one was produced — for the driver to feed back
    /// into the machine.
    ///
    /// When no `ContextManager` is set, a pre-compact hook aborts the pass,
    /// or the manager reports [`EnsureContextResult::NoAction`], the
    /// conversation is returned unchanged with measured before/after sizes —
    /// never a hard-coded zero — so the machine's no-progress guard compares
    /// real measurements and terminates with a typed error when compaction
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
        let tokens_before = self.count_context(&history);
        let Some(ctx_manager) = self.managers.context_manager() else {
            return Ok(CompactStepOutcome {
                tokens_before,
                tokens_after: tokens_before,
                compacted: None,
            });
        };

        #[cfg(feature = "hooks")]
        let hook = self.pre_compact_hook(&history, reason);
        #[cfg(feature = "hooks")]
        if hook.abort {
            return Ok(CompactStepOutcome {
                tokens_before,
                tokens_after: tokens_before,
                compacted: None,
            });
        }
        #[cfg(not(feature = "hooks"))]
        let (instructions, additional_context) = (None, Vec::new());

        #[cfg(feature = "hooks")]
        let messages_before = history.len();
        #[cfg(feature = "hooks")]
        let compact_start = Instant::now();
        #[cfg(feature = "hooks")]
        let (instructions, additional_context) = (hook.new_instructions, hook.additional_context);
        let result = ctx_manager
            .compact_with_reason(history, turn, reason, instructions, additional_context)
            .await;

        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                let tokens_after = outcome.tokens_after;
                let tokens_saved = tokens_before.saturating_sub(tokens_after);
                #[cfg(feature = "hooks")]
                let messages_after = outcome.messages.len();
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
                Ok(CompactStepOutcome {
                    tokens_before,
                    tokens_after,
                    compacted: Some(outcome.messages),
                })
            }
            Ok(EnsureContextResult::NoAction(messages)) => {
                let tokens_after = self.count_context(&messages);
                Ok(CompactStepOutcome {
                    tokens_before,
                    tokens_after,
                    compacted: None,
                })
            }
            Err(overflow) => Err(LoopError::ContextExceeded {
                used: overflow.tokens_used,
                limit: overflow.context_window,
            }),
        }
    }

    /// Run the pre-compact hooks, returning their merged result.
    ///
    /// Builds a [`PreCompactContext`] from the current history and
    /// session config — the trigger derived from the compaction
    /// [`reason`](crate::compact::types::CompactReason), so `Manual`
    /// reaches hooks as [`Manual`](CompactTrigger::Manual) — then asks
    /// the [`HookExecutor`](crate::hooks::HookExecutor) to run every
    /// registered `on_pre_compact` hook. An `abort: true` result makes
    /// [`run_compaction`](Self::run_compaction) skip compaction
    /// entirely for this trigger; the merged `new_instructions` and
    /// `additional_context` are threaded into the compactor's
    /// [`CompactionContext`](crate::compact::types::CompactionContext).
    /// Returns an empty allow-result when there is no hook executor.
    #[cfg(feature = "hooks")]
    fn pre_compact_hook(
        &self,
        history: &[Message],
        reason: crate::compact::types::CompactReason,
    ) -> crate::hooks::context::CompactResult {
        let Some(executor) = self.managers.hook_executor() else {
            return crate::hooks::context::CompactResult::allow();
        };
        let tokens_before = crate::compact::CompactionOutcome::estimate_tokens(history);
        let ctx = PreCompactContext {
            trigger: CompactTrigger::from(reason),
            custom_instructions: None,
            message_count: history.len(),
            tokens_before,
            context_window: self.session.config.context_window,
            session_id: self.session.id,
        };
        executor.check_pre_compact(&ctx)
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
