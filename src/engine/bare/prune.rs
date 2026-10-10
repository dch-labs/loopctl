//! The standalone tool-output reclamation pass — no model call, no
//! state transition, wired to the demotion seam.
//!
//! One entry point, [`maybe_prune`](BareLoop::maybe_prune), called by
//! the pre-request estimate site before the compaction deferral
//! re-check: when the configured
//! [`PruneConfig`](crate::compact::PruneConfig) arms and the context
//! figure has crossed the prune line, completed tool results beyond
//! the protected tail are cleared from the feed — their content
//! replaced by [`CLEARED_MARKER`], their originals handed to the
//! [`DemotionSink`](crate::compact::DemotionSink) with their identity
//! intact (call id, tool name, error flag, retention class), in a
//! deterministic oldest-first order — so a session whose growth is
//! mostly finished tool results reclaims its bytes in milliseconds
//! instead of paying a summarizer pass.

use super::BareLoop;
use crate::api::ApiClient;
use crate::compact::PruneConfig;
use crate::engine::bare::Message;
use crate::error::LoopError;
use crate::message::MessagePart;
use crate::tool::Retention;

/// The model-facing text that replaces a pruned tool result's content.
///
/// Short and neutral by design: it states what happened without
/// narrating mechanics, the same shape family as the truncating
/// terminal's `[evicted N more messages]` marker.
pub(super) const CLEARED_MARKER: &str = "[cleared: prior tool output]";

impl<C: ApiClient> BareLoop<C> {
    /// Configure the standalone tool-output reclamation pass.
    ///
    /// The pass is off until this is called. The prune line must sit
    /// strictly below the session's compaction threshold — pruning is
    /// the cheaper reclaim and gets its chance first; a configuration
    /// that would not is rejected rather than clamped. The
    /// minimum-free gate must stay at or below 100 percent — a gate
    /// the whole window could never satisfy would disable the pass
    /// silently.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidInput`] when
    /// [`prune_line_pct`](PruneConfig::prune_line_pct) is not strictly
    /// below the session's
    /// [`compact_threshold`](crate::config::SessionConfig::compact_threshold),
    /// or when [`min_free_pct`](PruneConfig::min_free_pct) exceeds
    /// `100`.
    pub fn set_prune(&mut self, config: PruneConfig) -> Result<(), LoopError> {
        if u32::from(config.prune_line_pct) >= u32::from(self.session.config.compact_threshold) {
            return Err(LoopError::InvalidInput(format!(
                "the prune line ({}% of the window) must sit strictly below the compaction \
                 threshold ({}%) so pruning runs before the summarizer",
                config.prune_line_pct, self.session.config.compact_threshold
            )));
        }
        if u32::from(config.min_free_pct) > 100 {
            return Err(LoopError::InvalidInput(format!(
                "the minimum-free gate ({}% of the window) cannot exceed 100% — a gate at \
                 least the whole window is never satisfiable and would disable pruning \
                 silently",
                config.min_free_pct
            )));
        }
        self.prune = Some(config);
        Ok(())
    }

    /// Run the reclamation pass if it is armed and worthwhile.
    ///
    /// Returns whether the feed changed. No-op — without touching the
    /// conversation — when no [`PruneConfig`] is set, the window policy
    /// is disabled, the figure has not crossed the prune line, or the
    /// projection would reclaim less than the minimum-free gate. The
    /// gate is converted from its provider-token fraction into the
    /// counter's raw units — the scale the projection counts in — so a
    /// settled calibration ratio cannot distort the gate's verdict.
    /// A pruned feed clears the usage anchor (the wire changed
    /// wholesale) and hands the cleared originals to the demotion
    /// sink, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] when the cancel signal fires
    /// while the demotion handoff is in flight — raced the same biased
    /// way the compaction arm races its sink, and cut off before the
    /// rewritten feed is adopted, so a cancelled pass leaves the
    /// conversation untouched. The history rewrite itself is
    /// infallible and a demotion sink that rejects its batch is warned
    /// about, never failed on.
    pub(super) async fn maybe_prune(&mut self) -> Result<bool, LoopError> {
        let Some(config) = self.prune else {
            return Ok(false);
        };
        let window = self.effective_context_window();
        if window == 0 {
            return Ok(false);
        }
        let line = window.saturating_mul(u64::from(config.prune_line_pct)) / 100;
        if self.machine.context_tokens() < line {
            return Ok(false);
        }
        let minimum_free = self
            .calibration
            .raw_window(window.saturating_mul(u64::from(config.min_free_pct)) / 100);
        let messages = self.machine.full_history();
        let history_len = self.machine.history().len();
        let Some((pruned, cleared, reclaimed)) = plan_prune(self, &messages, config, minimum_free)
        else {
            return Ok(false);
        };
        let cancelled = std::sync::Arc::clone(&self.cancelled);
        let turn = self.machine.turns_taken();
        tokio::select! {
            biased;
            () = cancelled.notified() => return Err(LoopError::Cancelled),
            () = self.demote_evicted(
                &cleared,
                crate::compact::CompactReason::Prune,
                turn,
            ) => {}
        }
        self.machine.adopt_pruned(pruned, history_len);
        self.anchor = None;
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.prune.cleared",
            parts = cleared.len(),
            tokens = reclaimed,
            "tool-output reclamation pass cleared prior results"
        );
        Ok(true)
    }
}

/// One clearable tool result beyond the protected tail.
///
/// The rank orders the pools: untagged results clear first,
/// [`Requery`](Retention::Requery) only once the untagged pool cannot
/// satisfy the minimum-free gate, [`Durable`](Retention::Durable)
/// never becomes a candidate.
struct PruneCandidate {
    /// The class pool the candidate belongs to.
    ///
    /// `0` for untagged results — the pool that clears first — and
    /// `1` for [`Requery`](Retention::Requery), admitted only once
    /// the untagged pool cannot satisfy the minimum-free gate.
    rank: usize,

    /// The candidate's message index in the snapshot.
    ///
    /// Locates the message whose part the pass rewrites on admission;
    /// the sort uses it to admit oldest-first within a pool.
    position: usize,

    /// The candidate's part index within its message.
    ///
    /// A message may carry several tool results; the index names the
    /// one part this candidate clears.
    part_index: usize,

    /// The projected reclaim: the original content's estimate minus
    /// the marker's.
    reclaim: u64,
}

/// Plan the pass over one conversation snapshot.
///
/// Walks from the end charging the protected tail budget — content
/// inside it is never cleared — and collects clearable tool results
/// beyond it as [`PruneCandidate`]s, admitted oldest-first within
/// each pool (the untagged pool in full, then requery results until
/// the projection clears `minimum_free`). Returns the rewritten
/// conversation, the cleared originals (for the demotion sink, in
/// admission order — oldest-first within each pool, the pools in
/// class order), and the reclaimed estimate; `None` when the gate
/// says the pass is not worth running. Each cleared original reaches
/// the sink with its identity intact — call id, tool name, error
/// flag, retention class — the same verbatim handoff a compaction
/// pass gives its evicted messages.
fn plan_prune<C: ApiClient>(
    loop_: &BareLoop<C>,
    messages: &[Message],
    config: PruneConfig,
    minimum_free: u64,
) -> Option<(Vec<Message>, Vec<Message>, u64)> {
    let mut budget = config.protected_tail_tokens;
    let mut candidates: Vec<PruneCandidate> = Vec::new();
    for (position, message) in messages.iter().enumerate().rev() {
        let cost = loop_.count_context(std::slice::from_ref(message));
        if budget > 0 {
            budget = budget.saturating_sub(cost);
            continue;
        }
        for (part_index, part) in message.parts.iter().enumerate() {
            let MessagePart::ToolResult {
                output, retention, ..
            } = part
            else {
                continue;
            };
            let rank = match retention {
                Some(Retention::Requery) => 1,
                None => 0,
                Some(Retention::Durable) => continue,
            };
            let reclaim = tool_result_cost(loop_, output).saturating_sub(marker_cost(loop_));
            if reclaim > 0 {
                candidates.push(PruneCandidate {
                    rank,
                    position,
                    part_index,
                    reclaim,
                });
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then(left.position.cmp(&right.position))
            .then(left.part_index.cmp(&right.part_index))
    });

    let mut admitted: Vec<&PruneCandidate> = Vec::new();
    let mut projected = 0_u64;
    for candidate in &candidates {
        if candidate.rank == 1 && projected >= minimum_free {
            break;
        }
        admitted.push(candidate);
        projected = projected.saturating_add(candidate.reclaim);
    }
    if admitted.is_empty() || projected < minimum_free {
        return None;
    }

    let mut pruned = messages.to_vec();
    let mut cleared: Vec<Message> = Vec::new();
    for candidate in admitted {
        let Some(message) = pruned.get_mut(candidate.position) else {
            continue;
        };
        let Some(MessagePart::ToolResult {
            output,
            call_id,
            name,
            is_error,
            retention,
            ..
        }) = message.parts.get(candidate.part_index)
        else {
            continue;
        };
        cleared.push(Message::new(
            message.role,
            vec![MessagePart::ToolResult {
                call_id: call_id.clone(),
                name: name.clone(),
                output: output.clone(),
                is_error: *is_error,
                retention: *retention,
            }],
        ));
        if let Some(MessagePart::ToolResult {
            output, is_error, ..
        }) = message.parts.get_mut(candidate.part_index)
        {
            *output = crate::message::ToolContent::Text(CLEARED_MARKER.to_string());
            *is_error = Some(false);
        }
    }
    let reclaimed = projected;
    Some((pruned, cleared, reclaimed))
}

/// The counter's estimate of one tool-result payload.
///
/// Wrapped in a synthetic user message so the counter's per-message
/// envelope applies — the same shaping the projection and the actual
/// feed count share, keeping the gate's arithmetic consistent with
/// what a pass really reclaims.
fn tool_result_cost<C: ApiClient>(
    loop_: &BareLoop<C>,
    output: &crate::message::ToolContent,
) -> u64 {
    loop_.count_context(&[Message::new(
        crate::message::Role::User,
        vec![MessagePart::ToolResult {
            call_id: String::new(),
            name: String::new(),
            output: output.clone(),
            is_error: Some(false),
            retention: None,
        }],
    )])
}

/// The counter's estimate of the marker that replaces cleared content.
///
/// The projection's subtraction term: a candidate's reclaim is its
/// original estimate minus this, so a result already smaller than the
/// marker contributes nothing and never becomes a candidate.
fn marker_cost<C: ApiClient>(loop_: &BareLoop<C>) -> u64 {
    tool_result_cost(
        loop_,
        &crate::message::ToolContent::Text(CLEARED_MARKER.to_string()),
    )
}
