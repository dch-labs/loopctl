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
use super::{CompactTrigger, PostCompactContext, PreCompactContext};
use crate::compact::EnsureContextResult;

use crate::capabilities::Compactable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
use crate::message::Message;
use crate::observer::{CompactedContext, CompactionFailedContext, PreCompactionContext};

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

/// A failed compaction pass: the propagating error plus the driver's
/// measured figures for the signal event.
///
/// `tokens_before` is the same driver-measured, provider-token-scale
/// figure the pass's observer context carried, so the event and the
/// observer can never disagree. `tokens_after` is the measured size
/// the failure narrates — the conversation unchanged when the
/// compactor produced nothing, or the rejected result's own measured
/// size when the pass succeeded but did not fit.
pub(super) struct CompactStepFailure {
    /// The error to propagate, exactly as the pass produced it.
    ///
    /// Its own `used`/`limit` pair stays manager-scale, like with
    /// like — the deferral guard compares them raw, and the machine
    /// feed routes the figure through the driver's conversion
    /// separately.
    pub(super) error: LoopError,

    /// Driver-measured conversation size ahead of the pass.
    ///
    /// Taken with the same counter and the same provider-token
    /// conversion as every successful pass's `tokens_before`, so all
    /// outcomes of the event family report one scale.
    pub(super) tokens_before: u64,

    /// The measured size the failure narrates for the pass.
    ///
    /// Equal to `tokens_before` when the compactor errored (nothing
    /// was produced); the `context_figure`-scaled size of the rejected
    /// result when the pass succeeded but did not fit.
    pub(super) tokens_after: u64,
}

impl<C: ApiClient> BareLoop<C> {
    /// Record what one compaction pass bought, in tokens and in time.
    ///
    /// `loopctl.compaction.tokens_saved` is the figure the pass reported on
    /// the engine's own scale — the provider-token figure the machine
    /// compares, not the compactor's raw count — so a pass that saved
    /// nothing reads as zero rather than being skipped.
    /// `loopctl.compaction.summarize.duration_ms` measures the pass from the
    /// driver's side, which includes the summarizer call and everything the
    /// pass owns; a pass that never reached this site (no action, a vetoed
    /// hook, an errored compactor) records neither, because a failed pass has
    /// no saving to report and its duration is a different question —
    /// `loopctl.compaction.no_action` and the signal event carry those
    /// readings.
    fn record_pass_savings(&self, tokens_saved: u64, compact_start: std::time::Instant) {
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.compaction.tokens_saved",
            value = tokens_saved,
            "one compaction pass reported its saving"
        );
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.compaction.summarize.duration_ms",
            value = self.managers.clock().elapsed_since(compact_start).as_millis(),
            "one compaction pass settled"
        );
    }

    /// Tell the observers a compaction pass is starting, before anything
    /// is reduced.
    ///
    /// Fires [`on_pre_compaction`](crate::observer::LoopObserver::on_pre_compaction)
    /// with the pass's opening figures — the trigger, the turn, the measured
    /// context against its window, and the message count the pass starts
    /// from — on the session's own id. A named step rather than an inline
    /// block so [`run_compaction`](Self::run_compaction) reads as the
    /// sequence of decisions it makes: notify, reduce, record.
    fn note_pre_compaction(
        &self,
        reason: crate::compact::types::CompactReason,
        turn: usize,
        tokens_before: u64,
        message_count: usize,
    ) {
        self.managers
            .observers()
            .on_pre_compaction(&PreCompactionContext {
                reason,
                turn,
                tokens_before,
                context_window: self.effective_context_window(),
                message_count,
                session_id: self.session.id,
            });
    }

    /// Compact the conversation context owned by the driving machine.
    ///
    /// Run when the machine requests it via
    /// [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact).
    /// Reads the history from [`LoopMachine::history`](crate::engine::core::LoopMachine::history),
    /// measures its token size, asks the configured
    /// [`ContextManager`](crate::compact::ContextManager) to reduce it, fires
    /// [`on_pre_compaction`](crate::observer::LoopObserver::on_pre_compaction)
    /// before the pass starts (after the manager-exists check, before the
    /// pre-compact hook veto) and
    /// [`on_compaction`](crate::observer::LoopObserver::on_compaction) with
    /// the full pass telemetry when compaction occurred, and the
    /// post-compact hook; returns a
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
    /// A failed pass returns [`CompactStepFailure`]: the propagating
    /// error beside the driver-measured, provider-token-scale figure
    /// pair the signal event emits — a compactor error leaves the
    /// conversation unchanged (`after == before`), while a pass whose
    /// result still does not fit reports that result's own measured
    /// size as `after`.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::CompactionFailed`] when the configured
    /// compactor ran and errored — the real cause rides the variant —
    /// and [`LoopError::ContextExceeded`] when a pass succeeded but
    /// its result still does not fit the context window.
    pub(super) async fn run_compaction(
        &mut self,
        turn: usize,
        reason: crate::compact::types::CompactReason,
    ) -> Result<CompactStepOutcome, CompactStepFailure> {
        let history = self.machine.full_history();
        let tokens_before = self.context_figure(
            self.count_context(&history)
                .saturating_add(self.overhead_tokens()),
        );
        let Some(ctx_manager) = self.managers.context_manager() else {
            return Ok(CompactStepOutcome {
                tokens_before,
                tokens_after: tokens_before,
                compacted: None,
            });
        };

        let message_count = history.len();
        self.note_pre_compaction(reason, turn, tokens_before, message_count);

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
        let compact_start = self.managers.clock().monotonic();
        let pre_snapshot = history.clone();
        #[cfg(feature = "hooks")]
        let (instructions, additional_context) = (hook.new_instructions, hook.additional_context);
        let reserved = self
            .overhead_tokens()
            .saturating_add(std::mem::take(&mut self.deferred_transient_tokens));
        let salvage_target = std::mem::take(&mut self.salvage_target);
        let sizing = crate::compact::CompactSizing {
            reserved_tokens: reserved,
            target_override: salvage_target,
        };
        let result = ctx_manager
            .compact_with_reason(
                history,
                turn,
                reason,
                instructions,
                additional_context,
                sizing,
            )
            .await;

        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                self.demote_evicted(&outcome.evicted, reason, turn).await;
                let tokens_after = self
                    .context_figure(outcome.tokens_after.saturating_add(self.overhead_tokens()));
                let tokens_saved = tokens_before.saturating_sub(tokens_after);
                self.record_pass_savings(tokens_saved, compact_start);
                #[cfg(feature = "hooks")]
                let messages_after = outcome.messages.len();
                let telemetry = ctx_manager.build_telemetry_with_duration(
                    reason,
                    &pre_snapshot,
                    &outcome.messages,
                    None,
                    self.managers.clock().elapsed_since(compact_start),
                );
                self.managers.observers().on_compaction(&CompactedContext {
                    tokens_before,
                    tokens_after,
                    tokens_saved,
                    reason,
                    evicted_messages: outcome.evicted.len(),
                    telemetry,
                    stage: outcome.stage.clone(),
                });
                #[cfg(feature = "hooks")]
                self.notify_post_compact_hook(
                    messages_before,
                    messages_after,
                    tokens_after,
                    tokens_saved,
                    self.managers.clock().elapsed_since(compact_start),
                );
                Ok(CompactStepOutcome {
                    tokens_before,
                    tokens_after,
                    compacted: Some(outcome.messages),
                })
            }
            Ok(EnsureContextResult::NoAction(messages)) => {
                tracing::debug!(
                    target: "loopctl::metrics",
                    metric = "loopctl.compaction.no_action",
                    "a compaction pass found nothing to give back"
                );
                let tokens_after = self.context_figure(
                    self.count_context(&messages)
                        .saturating_add(self.overhead_tokens()),
                );
                Ok(CompactStepOutcome {
                    tokens_before,
                    tokens_after,
                    compacted: None,
                })
            }
            Err(overflow) => {
                if let Some(error) = &overflow.compactor_error {
                    tracing::warn!(
                        error = %error,
                        "compaction failed; reporting the context overflow it could not fix"
                    );
                }
                self.managers
                    .observers()
                    .on_compaction_failed(&CompactionFailedContext {
                        reason,
                        turn,
                        tokens_before,
                        context_window: self.effective_context_window(),
                        error: overflow.compactor_error.clone(),
                    });
                match overflow.compactor_error {
                    Some(cause) => Err(CompactStepFailure {
                        error: LoopError::CompactionFailed {
                            used: overflow.tokens_used,
                            limit: overflow.context_window,
                            cause,
                        },
                        tokens_before,
                        tokens_after: tokens_before,
                    }),
                    None => Err(CompactStepFailure {
                        error: LoopError::ContextExceeded {
                            used: overflow.tokens_used,
                            limit: overflow.context_window,
                        },
                        tokens_before,
                        tokens_after: self.context_figure(overflow.tokens_used),
                    }),
                }
            }
        }
    }

    /// Hand a pass's evicted messages to the demotion sink.
    ///
    /// A successful compaction may evict turns; the sink is where they
    /// become retrievable memories. An empty eviction list is a no-op,
    /// and a sink that rejects the batch is warned about, never failed
    /// — compaction's own success must not hinge on the memory layer.
    pub(super) async fn demote_evicted(
        &self,
        evicted: &[Message],
        reason: crate::compact::types::CompactReason,
        turn: usize,
    ) {
        if evicted.is_empty() {
            return;
        }
        let meta = crate::compact::demote::DemotionContext {
            reason,
            turn,
            session_id: self.session.id,
        };
        if let Err(e) = self.managers.demotion_sink().demote(evicted, meta).await {
            tracing::warn!(
                error = %e,
                count = evicted.len(),
                "demotion sink rejected evicted turns"
            );
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
        let tokens_before = self.context_figure(
            self.count_context(history)
                .saturating_add(self.overhead_tokens()),
        );
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
