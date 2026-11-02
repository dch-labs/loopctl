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
use crate::compact::{CompactionOutcome, EnsureContextResult};

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

    /// How many messages the pass evicted from the feed.
    ///
    /// Carried out rather than counted inside the pass, because a pass that
    /// produced a history is not yet a conversation that lost messages: the
    /// driver books `loopctl.compaction.evicted.messages` only once the
    /// machine adopts the compacted history. A pass dropped on the way there
    /// — cancelled while a demotion sink was awaited, or refused by the
    /// machine — leaves the original feed in place and must book nothing.
    pub(super) evicted_messages: usize,
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

/// The verdict of one compaction pass, before it is shaped for a
/// consumer.
///
/// [`BareLoop::run_compaction`](BareLoop::run_compaction) reshapes it
/// into the [`CompactStepOutcome`] the driver feeds back into the
/// machine when the machine requested the pass;
/// [`BareLoop::compact_manual`](BareLoop::compact_manual) installs the
/// compacted feed into a resting machine and hands the verdict's own
/// record to the host.
pub(super) enum CompactPass {
    /// The pass produced a rewritten history.
    ///
    /// The full [`CompactionOutcome`](crate::compact::CompactionOutcome)
    /// rides along — the compacted list, the figures, the eviction
    /// list, and the stage account — so both consumers act on one pass
    /// record: the step flow takes the list and the eviction count,
    /// the manual seam hands the whole record to the host after
    /// installing the list.
    Compacted {
        /// The manager's pass record, whole.
        ///
        /// The compacted message list, the before/after figures the
        /// manager normalized to its own counter, the eviction slice
        /// already handed to the demotion sink, and the stage account —
        /// the same record the automatic path reports to its observers.
        /// The step flow reaches only for the list and the eviction
        /// count; the manual seam hands the whole record to the host
        /// verbatim after installing the list, so what the host reads
        /// is what the pass did.
        outcome: CompactionOutcome,

        /// Driver-measured conversation size ahead of the pass.
        ///
        /// Taken before the manager was consulted, with the loop's own
        /// counter and calibration — the provider-token scale every
        /// `tokens_*` figure in this family shares, so the pair compares
        /// like with like and against the machine's trigger.
        tokens_before: u64,

        /// Driver-scale measurement of the compacted feed.
        ///
        /// The post-pass conversation size on that same scale, request
        /// overhead included — the figure a manual install sets the
        /// machine's context count to, and the figure the `on_compaction`
        /// observer event and the signal event carry beside
        /// `tokens_before`.
        tokens_after: u64,
    },

    /// The pass ran and declined: no manager installed, a pre-compact
    /// hook vetoed the pass, or the compaction found nothing to give
    /// back.
    ///
    /// The conversation is unchanged; both figures are real
    /// measurements of it (never a hard-coded zero) so the machine's
    /// no-progress guard keeps comparing evidence.
    Declined {
        /// Driver-measured conversation size ahead of the pass.
        ///
        /// The same figure every outcome of the family opens with —
        /// the counter's and calibration's reading of the conversation
        /// before anything ran — so a decline still reports evidence on
        /// the shared scale rather than silence.
        tokens_before: u64,

        /// Driver-scale measurement of the unchanged conversation.
        ///
        /// Re-measured after the decline when the manager hands back a
        /// list it declined to compact (overhead included, as a
        /// successful pass's figure is) — a conversation that grew
        /// while the pass was in flight reads honestly, which is what
        /// keeps the machine's no-progress guard comparing two real
        /// measurements. The declines that never reach a list — no
        /// manager installed, a pre-compact hook veto — have nothing
        /// new to measure and carry `tokens_before` unchanged.
        tokens_after: u64,
    },

    /// The pass rewrote the feed, but not smaller.
    ///
    /// The manager classified the pass `Compacted` — its message
    /// count changed — while the driver's own measurement says the
    /// rewritten feed is at or above the conversation's pre-pass
    /// size, or the pass is a manual one whose rewrite carries more
    /// messages than the committed history it rewrites — the
    /// install splits at `history_len`, so the excess would stage
    /// as pending input, and a feed the seam cannot install is as
    /// empty-handed as one that grew. That is the machine's own
    /// no-progress rule, applied here inside
    /// the pass and ahead of every side effect: nothing was demoted,
    /// nothing recorded, no observer spoke and no hook ran, so the
    /// conversation, the sink, and the ledger stand exactly as the
    /// pass found them. Both consumers end the way the machine's
    /// guard ends a run — with
    /// [`CompactionStalled`](crate::error::LoopError::CompactionStalled)
    /// — because a compactor that grew the feed has nothing to offer:
    /// the automatic path fails its run, the manual seam refuses to
    /// install, and the history stays intact either way.
    NoProgress {
        /// Driver-measured conversation size ahead of the pass.
        ///
        /// The pre-pass figure the rewritten feed failed to go under
        /// — the shared-scale reading every outcome of this family
        /// opens with, so the pair still compares after the refusal
        /// and the signal event still narrates both sides of it.
        tokens_before: u64,

        /// Driver-scale measurement of the rewritten feed.
        ///
        /// At or above `tokens_before` — the measurement that
        /// triggered the refusal and the `used` payload the resulting
        /// `CompactionStalled` names; on the manual seam's
        /// message-count refusal, the measured size of the outgrown
        /// feed. The refused feed itself is
        /// dropped here, never installed, never demoted.
        tokens_after: u64,
    },
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
    /// block so [`compaction_pass`](Self::compaction_pass) reads as the
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
    /// post-compact hook; returns a [`CompactPass`] — the measured
    /// before/after pair plus the full [`CompactionOutcome`] when one
    /// was produced — which [`run_compaction`](Self::run_compaction)
    /// reshapes for the driver and which
    /// [`compact_manual`](Self::compact_manual) installs directly.
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
    pub(super) async fn compaction_pass(
        &mut self,
        turn: usize,
        reason: crate::compact::types::CompactReason,
    ) -> Result<CompactPass, CompactStepFailure> {
        let history = self.machine.full_history();
        let tokens_before = self.context_figure(
            self.count_context(&history)
                .saturating_add(self.overhead_tokens()),
        );
        let Some(ctx_manager) = self.managers.context_manager() else {
            return Ok(CompactPass::Declined {
                tokens_before,
                tokens_after: tokens_before,
            });
        };

        let message_count = history.len();
        self.note_pre_compaction(reason, turn, tokens_before, message_count);

        #[cfg(feature = "hooks")]
        let hook = self.pre_compact_hook(&history, reason);
        #[cfg(feature = "hooks")]
        if hook.abort {
            return Ok(CompactPass::Declined {
                tokens_before,
                tokens_after: tokens_before,
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
                let tokens_after = self
                    .context_figure(outcome.tokens_after.saturating_add(self.overhead_tokens()));
                let manual_outgrew_history = reason == crate::compact::types::CompactReason::Manual
                    && outcome.messages.len() > self.machine.history().len();
                if tokens_after >= tokens_before || manual_outgrew_history {
                    return Ok(CompactPass::NoProgress {
                        tokens_before,
                        tokens_after,
                    });
                }
                self.demote_evicted(&outcome.evicted, reason, turn).await;
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
                    CompactTrigger::from(reason),
                    messages_before,
                    messages_after,
                    tokens_after,
                    tokens_saved,
                    self.managers.clock().elapsed_since(compact_start),
                );
                Ok(CompactPass::Compacted {
                    outcome,
                    tokens_before,
                    tokens_after,
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
                Ok(CompactPass::Declined {
                    tokens_before,
                    tokens_after,
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

    /// The step-shaped face of [`compaction_pass`](Self::compaction_pass):
    /// runs the same routine and reshapes its verdict into the
    /// [`CompactStepOutcome`] the driver feeds back into the machine
    /// when the machine requested the pass via
    /// [`MachineStep::Compact`](crate::engine::core::MachineStep::Compact).
    ///
    /// # Errors
    ///
    /// Propagates the pass's failures unchanged, and turns the pass's
    /// `NoProgress` verdict into the machine's own
    /// [`CompactionStalled`](LoopError::CompactionStalled) — the
    /// automatic path ends its run on the stall exactly as the
    /// machine's no-progress guard would; see
    /// [`compaction_pass`](Self::compaction_pass).
    pub(super) async fn run_compaction(
        &mut self,
        turn: usize,
        reason: crate::compact::types::CompactReason,
    ) -> Result<CompactStepOutcome, CompactStepFailure> {
        match self.compaction_pass(turn, reason).await? {
            CompactPass::Compacted {
                outcome,
                tokens_before,
                tokens_after,
            } => Ok(CompactStepOutcome {
                evicted_messages: outcome.evicted.len(),
                tokens_before,
                tokens_after,
                compacted: Some(outcome.messages),
            }),
            CompactPass::Declined {
                tokens_before,
                tokens_after,
            } => Ok(CompactStepOutcome {
                tokens_before,
                tokens_after,
                compacted: None,
                evicted_messages: 0,
            }),
            CompactPass::NoProgress {
                tokens_before,
                tokens_after,
            } => Err(CompactStepFailure {
                error: LoopError::CompactionStalled { used: tokens_after },
                tokens_before,
                tokens_after,
            }),
        }
    }

    /// Run a manual compaction pass over this loop's history.
    ///
    /// The host seam for a `/compact`-style command and for headless
    /// scheduled passes: the loop drives its own installed
    /// [`ContextManager`](crate::compact::ContextManager) through the
    /// engine's own compaction ceremony — the opening observer event,
    /// the pre-compact hook veto (hooks see the pass under the manual
    /// trigger), demotion of everything the pass evicted, telemetry,
    /// the `on_compaction` observer event, the post-compact hook, and
    /// the `loopctl.compaction` signal event with reason `manual` —
    /// and then installs what the pass won: the machine takes the
    /// compacted history through
    /// [`install_manual_compaction`](crate::engine::core::LoopMachine::install_manual_compaction)
    /// — the guarded install, which folds runs of adjacent assistant
    /// messages the way the in-run adoption does so a summary landing
    /// before an assistant preserved tail never reaches history or
    /// the wire — the token recount lands with it, and the usage
    /// anchor and any prepared turn are invalidated exactly as a rewind
    /// invalidates them. The host reads the figures *before* the next
    /// run: the reported count and the next run's trigger both see the
    /// compacted size.
    ///
    /// The pass works only while the loop is not running: the machine
    /// is guarded like [`rewind_history_to`](Self::rewind_history_to),
    /// and a turn in flight refuses with an error naming the state and
    /// the cancel-then-resolve path — text written to be displayed to
    /// the user who asked. A machine at rest between runs (a
    /// `Terminal` one awaiting the next
    /// [`accept_input`](crate::engine::core::LoopMachine::accept_input))
    /// is explicitly admitted: this is where a long-lived host's
    /// `/compact` finds it.
    ///
    /// [`EnsureContextResult::NoAction`] is the honest decline: the
    /// conversation sat below the threshold, a pre-compact hook vetoed
    /// the pass, or no manager was installed — history, figures, and
    /// anchor are left untouched, and the signal event narrates a noop.
    ///
    /// # Errors
    ///
    /// [`LoopError::InvalidInput`] when a turn is in flight or input
    /// is staged unstarted;
    /// [`LoopError::CompactionFailed`] when the compactor ran and
    /// errored (its own text rides the variant),
    /// [`LoopError::ContextExceeded`] when the pass succeeded but its
    /// result still does not fit the window, and
    /// [`LoopError::CompactionStalled`] when the pass rewrote the feed
    /// but did not shrink it, or rewrote it into more messages than
    /// the committed history holds — the install commits `history_len`
    /// messages and stages the rest as pending input, so the excess
    /// would run uninvited — the shared no-progress rule refuses
    /// both before anything is demoted, recorded, or narrated, and
    /// the conversation is left exactly as it was. The failure
    /// mappings are the ones the automatic pass propagates to
    /// its run.
    pub async fn compact_manual(&mut self) -> Result<EnsureContextResult, LoopError> {
        self.machine.guard_manual_compaction()?;
        let turn = self.machine.turns_taken();
        let history_len = self.machine.history().len();
        match self
            .compaction_pass(turn, crate::compact::types::CompactReason::Manual)
            .await
        {
            Ok(CompactPass::Compacted {
                outcome,
                tokens_before,
                tokens_after,
            }) => {
                let evicted = outcome.evicted.len();
                self.machine.install_manual_compaction(
                    outcome.messages.clone(),
                    history_len,
                    tokens_after,
                )?;
                self.anchor = None;
                self.prepared_turn = None;
                crate::tracing_spans::event_compaction(
                    crate::compact::types::CompactReason::Manual,
                    tokens_before,
                    tokens_after,
                    crate::tracing_spans::CompactionPassOutcome::Applied,
                );
                if evicted > 0 {
                    tracing::debug!(
                        target: "loopctl::metrics",
                        metric = "loopctl.compaction.evicted.messages",
                        messages = evicted,
                        "messages left the feed as a manual compaction was installed"
                    );
                }
                Ok(EnsureContextResult::Compacted(outcome))
            }
            Ok(CompactPass::Declined {
                tokens_before,
                tokens_after,
            }) => {
                crate::tracing_spans::event_compaction(
                    crate::compact::types::CompactReason::Manual,
                    tokens_before,
                    tokens_after,
                    crate::tracing_spans::CompactionPassOutcome::Noop,
                );
                Ok(EnsureContextResult::NoAction(self.machine.full_history()))
            }
            Ok(CompactPass::NoProgress {
                tokens_before,
                tokens_after,
            }) => {
                crate::tracing_spans::event_compaction(
                    crate::compact::types::CompactReason::Manual,
                    tokens_before,
                    tokens_after,
                    crate::tracing_spans::CompactionPassOutcome::Failed,
                );
                Err(LoopError::CompactionStalled { used: tokens_after })
            }
            Err(failure) => {
                crate::tracing_spans::event_compaction(
                    crate::compact::types::CompactReason::Manual,
                    failure.tokens_before,
                    failure.tokens_after,
                    crate::tracing_spans::CompactionPassOutcome::Failed,
                );
                Err(failure.error)
            }
        }
    }

    /// Hand a pass's evicted messages to the demotion sink.
    ///
    /// A successful compaction may evict turns; the sink is where they
    /// become retrievable memories. An empty eviction list is a no-op,
    /// and a sink that rejects the batch is warned about, never failed
    /// — compaction's own success must not hinge on the memory layer.
    ///
    /// This seam books nothing on the metric target. The eviction count rides
    /// out with [`CompactStepOutcome::evicted_messages`](super::compact::CompactStepOutcome)
    /// and is booked by the driver once the machine has adopted the compacted
    /// history — until then the conversation still holds those messages, and a
    /// pass cancelled here (a storing sink that never returns is enough) drops
    /// the whole attempt with the history intact. Whether the sink stored,
    /// refused, or is the default that keeps nothing is its own story, told by
    /// `loopctl.demotion.delivered` and `loopctl.demotion.errors`.
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
    /// [`compaction_pass`](Self::compaction_pass) skip compaction
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
    /// skips it. The `trigger` names the pass for the hook: the
    /// machine's own passes report
    /// [`Auto`](CompactTrigger::Auto); a host-requested
    /// [`compact_manual`](Self::compact_manual) reports
    /// [`Manual`](CompactTrigger::Manual).
    #[cfg(feature = "hooks")]
    fn notify_post_compact_hook(
        &self,
        trigger: CompactTrigger,
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
            trigger,
            messages_compacted: messages_before.saturating_sub(messages_after),
            tokens_saved,
            tokens_after,
            duration_ms: Self::millis_u64(duration),
            session_id: self.session.id,
        };
        executor.notify_post_compact(&ctx);
    }
}
