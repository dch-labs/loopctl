//! Truncating compactor and token splitter.
//!
//! Contents:
//!
//! - [`TruncatingCompactor`] — a simple compactor that drops the oldest messages.
//! - [`TokenSplitter`] — splits a conversation into "old" and "recent" at a turn boundary.
//! - [`SplitResult`] — result of splitting a conversation.

use crate::compact::ContextCompactor;
use crate::compact::types::{CompactionContext, CompactionOutcome};
use crate::message::{Message, MessagePart, Role};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;

/// The pairing state of one tool part.
///
/// Built by [`ToolPairing::scan`] as it walks the message list: every
/// call and result part ends up either paired with the occurrence it
/// belongs to or marked unpaired.
#[derive(Debug, Clone, Copy)]
enum PartMate {
    /// The part is paired with a counterpart in another message.
    ///
    /// Calls carry their result's location and results their call's —
    /// which side is which follows from message order, since a call
    /// never appears after its own result in a well-formed history.
    Paired {
        /// Message index of the counterpart part.
        ///
        /// Pairing decisions work at message granularity (split
        /// points, pulls, live-message sets), so the part index within
        /// that message is not recorded.
        message: usize,
    },
    /// The part is a call or result whose counterpart never appears.
    ///
    /// A call that never received a result, or a result with no
    /// preceding call. Split decisions leave such parts untouched:
    /// repairing a conversation already broken at the input is not the
    /// compactor's job.
    Unpaired,
}

/// Occurrence-aware pairing between tool calls and tool results.
///
/// Pairing is positional, not by id alone: a result part pairs with the
/// most recent *preceding* call part carrying the same id that no
/// earlier result has already claimed, so a call id reused across
/// separate conversation turns yields two distinct pairs instead of one
/// conflated one. A call that never receives a result, or a result with
/// no preceding call, is [`Unpaired`](PartMate::Unpaired) — split
/// decisions leave such parts alone rather than inventing a pair for
/// them.
struct ToolPairing {
    /// The pairing state of every part, mirroring the message list's
    /// shape.
    ///
    /// Indexed as `mates[message][part]`: `Some(Paired)` or
    /// `Some(Unpaired)` for call and result parts, `None` for every
    /// other part kind.
    mates: Vec<Vec<Option<PartMate>>>,
}

impl ToolPairing {
    /// Pair every call and result occurrence in a message list.
    ///
    /// One forward scan with a per-id stack of unconsumed calls: each
    /// result claims the most recent unconsumed preceding call with the
    /// same id (last-in-first-out), which keeps reused ids in separate
    /// turns as separate pairs.
    fn scan(messages: &[Message]) -> Self {
        let mut mates: Vec<Vec<Option<PartMate>>> = messages
            .iter()
            .map(|msg| msg.parts.iter().map(|_| None).collect())
            .collect();
        let mut pending: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for (i, msg) in messages.iter().enumerate() {
            for (p, part) in msg.parts.iter().enumerate() {
                match part {
                    MessagePart::ToolCall { id, .. } => {
                        pending.entry(id.clone()).or_default().push((i, p));
                    }
                    MessagePart::ToolResult { call_id, .. } => {
                        let claimed = pending.get_mut(call_id).and_then(Vec::pop);
                        let state = match claimed {
                            Some((cm, cp)) => {
                                if let Some(slot) =
                                    mates.get_mut(cm).and_then(|row| row.get_mut(cp))
                                {
                                    *slot = Some(PartMate::Paired { message: i });
                                }
                                PartMate::Paired { message: cm }
                            }
                            None => PartMate::Unpaired,
                        };
                        if let Some(slot) = mates.get_mut(i).and_then(|row| row.get_mut(p)) {
                            *slot = Some(state);
                        }
                    }
                    _ => {}
                }
            }
        }
        for (i, p) in pending.into_values().flatten() {
            if let Some(slot) = mates.get_mut(i).and_then(|row| row.get_mut(p)) {
                *slot = Some(PartMate::Unpaired);
            }
        }
        Self { mates }
    }

    /// Move a split back to the earliest call a straddling result
    /// would orphan, or return it unchanged.
    ///
    /// A straddling result is a paired result part at or after `split`
    /// whose paired call sits before it (calls precede their results),
    /// so a call id reused later in the kept range does not count as a
    /// match for the earlier occurrence.
    fn adjusted_split(&self, split: usize) -> usize {
        let mut new_split = split;
        for row in self.mates.iter().skip(split) {
            for mate in row.iter().flatten() {
                if let PartMate::Paired { message: m, .. } = mate
                    && *m < split
                    && *m < new_split
                {
                    new_split = *m;
                }
            }
        }
        new_split
    }

    /// Whether splitting at `index` keeps every paired result with its
    /// call.
    ///
    /// `false` when any paired result at or after `index` has its call
    /// before `index` — judged per occurrence, so a reused id's later
    /// pair does not vouch for the earlier one.
    fn boundary_pair_safe(&self, index: usize) -> bool {
        !self.mates.iter().enumerate().skip(index).any(|(i, row)| {
            row.iter().flatten().any(|mate| {
                matches!(
                    mate,
                    PartMate::Paired { message: m, .. } if *m < index && *m < i
                )
            })
        })
    }

    /// Result-message indices in `1..split` paired with calls carried
    /// by the first message.
    ///
    /// The exact occurrences mated to message 0's call parts — a later
    /// pair reusing the same id does not satisfy the first message's
    /// call. Sorted ascending; one index per message even when several
    /// of message 0's calls resolved in it.
    fn first_message_dropped_result_indices(&self, split: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = Vec::new();
        if let Some(first_row) = self.mates.first() {
            for mate in first_row.iter().flatten() {
                if let PartMate::Paired { message: m, .. } = mate
                    && *m > 0
                    && *m < split
                    && !indices.contains(m)
                {
                    indices.push(*m);
                }
            }
        }
        indices.sort_unstable();
        indices
    }

    /// Whether the part at `origin`'s given position keeps its pair in
    /// the assembled output.
    ///
    /// Non-tool parts always survive; a tool part survives exactly when
    /// its mate's message is among the live originals. Mates inside
    /// the pulled region survive mutually, so a single filtering pass
    /// is consistent.
    fn part_survives(&self, origin: usize, part: usize, live: &HashSet<usize>) -> bool {
        match self
            .mates
            .get(origin)
            .and_then(|row| row.get(part))
            .and_then(|state| state.as_ref())
        {
            None => true,
            Some(PartMate::Unpaired) => false,
            Some(PartMate::Paired { message: m, .. }) => live.contains(m),
        }
    }
}

/// A simple compactor that drops the oldest messages.
///
/// Keeps the first message (typically the system prompt) and a configurable
/// number of recent messages. No LLM calls required — useful as a fallback
/// or for contexts where summarization isn't available.
///
/// # Strategy
///
/// ```text
/// [System?] [Old₁, Old₂, ..., Oldₙ] [Recent₁, Recent₂, ..., Recentₘ]
///  ↑ kept   ↑ discarded ↑            ↑ preserved ↑
/// ```
///
/// The first message is always retained (if present) because it usually
/// contains the system prompt or conversation instructions. This prevents
/// the compactor from discarding essential context that shapes the agent's
/// behavior. Tool-call/result pairs are never split: the split point
/// moves to keep a pair together, and a result that would be dropped
/// behind a call carried by the preserved first message is pulled back
/// alongside it. Pairs are matched per occurrence — a call id reused
/// in a later turn is a different pair, never a substitute for an
/// earlier one. If the conversation is shorter than `min_messages`, no
/// compaction occurs.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::TruncatingCompactor;
/// use std::sync::Arc;
///
/// let compactor = TruncatingCompactor::new()
///     .with_preserve_recent(6)
///     .with_min_messages(8);
///
/// // Pass to ContextManager:
/// // let manager = ContextManager::new(Arc::new(compactor));
/// ```
#[derive(Debug, Clone)]
pub struct TruncatingCompactor {
    /// Number of recent messages to always preserve during compaction.
    ///
    /// This many messages from the end of the conversation are kept intact;
    /// everything before them is dropped. Defaults to 4.
    preserve_recent: usize,

    /// Minimum number of messages before compaction is attempted.
    ///
    /// If the conversation has fewer messages than this, compaction is
    /// skipped entirely. Prevents aggressive truncation of short
    /// conversations. Defaults to 6.
    min_messages: usize,
}

impl TruncatingCompactor {
    /// Create a new truncating compactor with sensible defaults.
    ///
    /// Preserves the 4 most recent messages and requires at least 6 messages
    /// before compaction is attempted.
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_recent: 4,
            min_messages: 6,
        }
    }

    /// Set how many recent messages to preserve during compaction.
    ///
    /// This many messages from the end of the conversation are kept
    /// intact. The rest are dropped. Must be at least 1.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the minimum number of messages before compaction is attempted.
    ///
    /// If the conversation has fewer messages than this, compaction is
    /// skipped entirely. Prevents aggressive truncation of short
    /// conversations.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Number of recent messages that will be preserved during compaction.
    ///
    /// This many messages from the end of the conversation are always kept.
    #[must_use]
    pub fn preserve_recent(&self) -> usize {
        self.preserve_recent
    }

    /// Minimum number of messages required before compaction is attempted.
    ///
    /// Conversations shorter than this are left untouched.
    #[must_use]
    pub fn min_messages(&self) -> usize {
        self.min_messages
    }
}

impl Default for TruncatingCompactor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextCompactor for TruncatingCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        _target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let total = messages.len();
            if total <= self.min_messages {
                return CompactionOutcome::no_change(messages);
            }

            // Determine split point: keep `preserve_recent` from the end.
            let initial_split = total.saturating_sub(self.preserve_recent);

            // Adjust split to avoid orphaning tool-call/result pairs.
            // If the "recent" portion contains a ToolResult whose matching
            // ToolCall would be dropped, move the split back to include the
            // message containing that ToolCall. Each backward move admits
            // previously dropped messages that can themselves carry results
            // whose calls stay dropped, so re-adjust until the split stops
            // moving: the adjustment is non-increasing and bottoms out at
            // 0, so the loop always terminates.
            let mut split = initial_split;
            loop {
                let adjusted = Self::adjust_for_tool_pairs(&messages, split);
                if adjusted == split {
                    break;
                }
                split = adjusted;
            }

            // A split of 0 means the adjustment pulled the whole
            // conversation into the recent slice — nothing is dropped, so
            // report no change instead of a compaction that reduced
            // nothing.
            if split == 0 {
                return CompactionOutcome::no_change(messages);
            }

            let recent: Vec<Message> = messages.get(split..).unwrap_or_default().to_vec();

            // Always preserve the first message (typically the system
            // prompt); split > 0 guarantees it is not already part of the
            // recent slice.
            let mut preserved: Vec<Message> = Vec::with_capacity(recent.len().saturating_add(1));
            if let Some(first) = messages.first() {
                preserved.push(first.clone());
            }
            preserved.extend(recent);
            let preserved = Self::reattach_dropped_results(&messages, split, preserved);

            let tokens_after = context.counter.count(&preserved);
            CompactionOutcome {
                messages: preserved,
                tokens_after,
                tokens_saved: context.tokens_before.saturating_sub(tokens_after),
                success: true,
                error: None,
            }
        })
    }
}

impl TruncatingCompactor {
    /// Adjust the split index to avoid orphaning tool-call/result pairs.
    ///
    /// If the "recent" portion (from `split` onward) contains any
    /// [`MessagePart::ToolResult`] whose paired
    /// [`MessagePart::ToolCall`] would be in the dropped portion
    /// (before `split`), the split is moved backward to include the
    /// message containing that call. Pairing is per occurrence (see
    /// [`ToolPairing`]): a call id reused in a later turn is a
    /// different pair and never substitutes for the stranded one.
    fn adjust_for_tool_pairs(messages: &[Message], split: usize) -> usize {
        if split == 0 {
            return 0;
        }
        ToolPairing::scan(messages).adjusted_split(split)
    }

    /// Pull dropped [`MessagePart::ToolResult`]s back into the kept slice
    /// when the preserved first message carries their calls.
    ///
    /// The first message is kept unconditionally, but its `ToolCall`s can
    /// have results that land in the dropped range (before `split`) —
    /// [`adjust_for_tool_pairs`](Self::adjust_for_tool_pairs) repairs only
    /// the mirror direction (a result in the recent slice whose call
    /// would be dropped). Each dropped message carrying a still-missing
    /// result for a first-message call is inserted into the kept slice
    /// right after the first message, keeping the pair adjacent.
    /// Results already present anywhere in the kept slice satisfy their
    /// calls and are not pulled twice.
    ///
    /// A pulled message can carry parts beyond the results it was pulled
    /// for — e.g. a result for a call that stays dropped. Such parts are
    /// removed from the pulled copy rather than stranded without their
    /// pair-mate: both directions of the pairing contract hold in the
    /// output (no call without its result that the input did not already
    /// carry, and no result without its call). The first message and the
    /// recent slice are emitted exactly as received. Pairing is per
    /// occurrence (see [`ToolPairing`]): a result message is pulled only
    /// when it is the exact occurrence mated to a first-message call —
    /// a later pair reusing the same id neither satisfies that call nor
    /// gets pulled in its place.
    fn reattach_dropped_results(
        messages: &[Message],
        split: usize,
        kept: Vec<Message>,
    ) -> Vec<Message> {
        let pairing = ToolPairing::scan(messages);
        let pull_indices = pairing.first_message_dropped_result_indices(split);
        if pull_indices.is_empty() {
            return kept;
        }

        let pulled: Vec<Message> = pull_indices
            .iter()
            .filter_map(|&i| messages.get(i).cloned())
            .collect();
        let mut kept_iter = kept.into_iter();
        let Some(first) = kept_iter.next() else {
            return pulled;
        };
        let pulled_len = pulled.len();
        let mut out =
            Vec::with_capacity(pulled_len.saturating_add(kept_iter.len()).saturating_add(1));
        let mut origins: Vec<usize> = Vec::with_capacity(out.capacity());
        out.push(first);
        origins.push(0);
        out.extend(pulled);
        origins.extend(pull_indices.iter().copied());
        origins.extend(split..messages.len());
        out.extend(kept_iter);
        Self::drop_unmatched_pair_parts(&mut out, &origins, &pairing, pulled_len);
        out
    }

    /// Remove pair-mate-less tool parts from the pulled region of a
    /// reassembled list.
    ///
    /// A part is removed exactly when its paired counterpart's message
    /// is not among the assembled output (`origins` maps each output
    /// slot to its original message index) or when it has no
    /// counterpart at all — removing such a part can never orphan a
    /// kept one, and mates inside the pulled region survive mutually,
    /// so one filtering pass is consistent. Only the pulled messages
    /// (output indices `1..=pulled_len`) are filtered; the first
    /// message and the recent slice pass through as-is. Every pulled
    /// message keeps at least the result it was pulled for (its call
    /// rides in the first message), so filtering cannot empty a
    /// message.
    fn drop_unmatched_pair_parts(
        out: &mut [Message],
        origins: &[usize],
        pairing: &ToolPairing,
        pulled_len: usize,
    ) {
        if pulled_len == 0 {
            return;
        }
        let live: HashSet<usize> = origins.iter().copied().collect();
        let pulled_end = pulled_len.saturating_add(1);
        for (slot, msg) in out.iter_mut().enumerate().take(pulled_end).skip(1) {
            let Some(&origin) = origins.get(slot) else {
                continue;
            };
            let keeps: Vec<bool> = (0..msg.parts.len())
                .map(|p| pairing.part_survives(origin, p, &live))
                .collect();
            msg.parts = msg
                .parts
                .iter()
                .zip(keeps)
                .filter(|(_, keep)| *keep)
                .map(|(part, _)| part.clone())
                .collect();
        }
    }
}

/// Splits a conversation into "old" and "recent" at a turn boundary.
///
/// Used by compactors (and agent-side code) that need to know which
/// messages to compact versus preserve. Splits at role transitions for
/// coherent summarization — the split always occurs between a complete
/// request/response pair.
///
/// # Rules
///
/// - Never compact the last user message (it's the current request).
/// - Split at turn boundaries (role transitions) for coherent output.
/// - If the conversation is too short, `to_compact` will be empty.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::TokenSplitter;
/// use loopctl::message::Message;
///
/// let splitter = TokenSplitter::new()
///     .with_preserve_recent(4)
///     .with_min_messages(6);
///
/// let messages = vec![
///     Message::user("Hello"),
///     Message::assistant("Hi there!"),
///     Message::user("What is 2+2?"),
///     Message::assistant("4"),
/// ];
///
/// let result = splitter.split(&messages);
/// // With only 4 messages and min_messages=6, nothing is split off.
/// assert!(result.to_compact.is_empty());
/// assert_eq!(result.preserved.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct TokenSplitter {
    /// Number of recent messages to always preserve during a split.
    ///
    /// This many messages from the end of the conversation are kept in the
    /// preserved portion. Defaults to 4.
    preserve_recent: usize,

    /// Minimum number of messages before a split is attempted.
    ///
    /// Conversations shorter than this go entirely into the preserved
    /// portion. Defaults to 6.
    min_messages: usize,
}

/// Result of splitting a conversation into old and recent portions.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Messages to compact or summarize (the old portion).
    ///
    /// These are the messages before the split point. They are candidates
    /// for summarization, truncation, or removal by the compactor.
    pub to_compact: Vec<Message>,

    /// Messages to preserve as-is (the recent portion).
    ///
    /// These are the messages after the split point. They are kept
    /// untouched in the conversation history.
    pub preserved: Vec<Message>,

    /// Estimated token count of the old portion ([`to_compact`](Self::to_compact)).
    ///
    /// Approximate token usage of the messages eligible for summarization or
    /// removal, computed from the pre-split history so callers can budget how
    /// much context compaction should reclaim.
    pub compact_tokens: u64,

    /// Estimated token count of the recent portion ([`preserved`](Self::preserved)).
    ///
    /// Approximate token usage of the messages kept intact after the split,
    /// giving callers the residual context footprint that will carry into the
    /// next model call.
    pub preserved_tokens: u64,

    /// The index in the original message list where the split occurred.
    ///
    /// Zero when no split was needed (the entire conversation was
    /// preserved), or when no split is possible without separating a
    /// tool call from its result.
    pub split_index: usize,
}

impl TokenSplitter {
    /// Create a new splitter with sensible defaults.
    ///
    /// Preserves the 4 most recent messages and requires at least 6 messages
    /// before a split is attempted.
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_recent: 4,
            min_messages: 6,
        }
    }

    /// Set how many recent messages to preserve during a split.
    ///
    /// This many messages from the end of the conversation are kept in the
    /// preserved portion. The rest are candidates for compaction.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the minimum number of messages before a split is attempted.
    ///
    /// Conversations shorter than this are returned entirely as preserved.
    /// Prevents splitting very short conversations into meaningless fragments.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Split the given messages into old and recent portions.
    ///
    /// The split point is chosen at a turn boundary (a role transition
    /// from assistant to a user message carrying no tool results) as
    /// close as possible to leaving `preserve_recent` messages in the
    /// recent portion. Splitting never separates a tool call from its
    /// result: when no pair-safe boundary exists at or before the
    /// target, the entire conversation is preserved and `to_compact`
    /// is empty.
    ///
    /// If the conversation has fewer than `min_messages`, the entire
    /// conversation goes into `preserved` and `to_compact` is empty.
    #[must_use]
    pub fn split(&self, messages: &[Message]) -> SplitResult {
        if messages.len() <= self.min_messages {
            return SplitResult {
                to_compact: vec![],
                preserved: messages.to_vec(),
                compact_tokens: 0,
                preserved_tokens: CompactionOutcome::estimate_tokens(messages),
                split_index: 0,
            };
        }

        // Find a split point: we want `preserve_recent` messages at the end.
        // Look for a turn boundary (assistant→user transition) near the
        // target split point.
        let target_split = messages.len().saturating_sub(self.preserve_recent);
        let split_index = Self::find_turn_boundary(messages, target_split);
        let (to_compact, preserved) = messages.split_at(split_index);
        SplitResult {
            to_compact: to_compact.to_vec(),
            preserved: preserved.to_vec(),
            compact_tokens: CompactionOutcome::estimate_tokens(to_compact),
            preserved_tokens: CompactionOutcome::estimate_tokens(preserved),
            split_index,
        }
    }

    /// Find the nearest pair-safe turn boundary at or before the target
    /// index.
    ///
    /// A turn boundary is a position where the previous message is
    /// assistant-role and the next is user-role, and no tool result in
    /// the kept portion references a call that would be split off (see
    /// [`split_is_pair_safe`](Self::split_is_pair_safe)) — a user
    /// message delivering tool results continues the same turn, so
    /// splitting there would separate a call from its result. This
    /// ensures we split at a coherent conversation boundary. When no
    /// such boundary exists at or before the target, `0` is returned:
    /// nothing can be split off without breaking a call/result pair,
    /// so the whole conversation stays preserved.
    fn find_turn_boundary(messages: &[Message], target: usize) -> usize {
        if target == 0 {
            return 0;
        }

        for i in (1..=target).rev() {
            if i >= messages.len() {
                continue;
            }
            let Some(prev) = messages.get(i.saturating_sub(1)) else {
                continue;
            };
            let Some(curr) = messages.get(i) else {
                continue;
            };
            if prev.role == Role::Assistant
                && curr.role == Role::User
                && Self::split_is_pair_safe(messages, i)
            {
                return i;
            }
        }

        0
    }

    /// Whether splitting at `index` keeps every tool call with its
    /// result.
    ///
    /// `false` when any paired result in the kept portion (from
    /// `index` on) has its call in the dropped portion (before
    /// `index`) — judged per occurrence regardless of how far apart
    /// the two messages are, so histories with interleaved or
    /// consecutive user messages are covered, and a call id reused by
    /// a later complete pair does not vouch for an earlier occurrence
    /// being split off.
    fn split_is_pair_safe(messages: &[Message], index: usize) -> bool {
        ToolPairing::scan(messages).boundary_pair_safe(index)
    }
}

impl Default for TokenSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::ContextCompactor;
    use crate::compact::types::{CompactReason, CompactionContext};
    use crate::message::{Message, MessagePart, Role, ToolContent};
    use serde_json::json;

    fn tool_text(s: &str) -> ToolContent {
        ToolContent::from_string(s)
    }

    fn make_context(msgs: &[Message]) -> CompactionContext {
        CompactionContext {
            tokens_before: CompactionOutcome::estimate_tokens(msgs),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 5,
            counter: std::sync::Arc::new(crate::compact::HeuristicTokenCounter),
        }
    }

    fn convo_with_straddling_tool_pair() -> Vec<Message> {
        vec![
            Message::user("msg0"),
            Message::assistant("reply0"),
            Message::user("msg1"),
            Message::assistant("reply1"),
            Message::user("msg2"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "call_a",
                    "search",
                    json!({"q": "rust"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "call_a",
                    "search",
                    tool_text("result data"),
                    false,
                )],
            ),
            Message::assistant("final reply"),
        ]
    }

    fn has_tool_call(msgs: &[Message], id: &str) -> bool {
        msgs.iter()
            .flat_map(|m| m.parts.iter())
            .any(|p| matches!(p, MessagePart::ToolCall { id: tool_id, .. } if tool_id == id))
    }

    fn has_tool_result(msgs: &[Message], call_id: &str) -> bool {
        msgs.iter()
            .flat_map(|m| m.parts.iter())
            .any(|p| matches!(p, MessagePart::ToolResult { call_id: cid, .. } if cid == call_id))
    }

    #[tokio::test]
    async fn compact_preserves_tool_call_when_result_is_in_recent() {
        let messages = convo_with_straddling_tool_pair();
        let compactor = TruncatingCompactor::new()
            .with_preserve_recent(2)
            .with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 500, context).await;

        // The tool-call ("call_a") and tool-result ("call_a") must both
        // be in the compacted output — neither should be orphaned.
        assert!(
            has_tool_call(&outcome.messages, "call_a"),
            "tool-call 'call_a' must be preserved"
        );
        assert!(
            has_tool_result(&outcome.messages, "call_a"),
            "tool-result for 'call_a' must be preserved"
        );
    }

    #[tokio::test]
    async fn compact_does_not_orphan_when_pairs_are_together_in_recent() {
        // When both call and result are already in the recent portion,
        // no adjustment is needed — the split should stay at the naive point.
        let messages = vec![
            Message::user("msg0"),
            Message::assistant("reply0"),
            Message::user("msg1"),
            Message::assistant("reply1"),
            Message::user("msg2"),
            Message::assistant("reply2"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("call_b", "calc", json!({}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "call_b",
                    "calc",
                    tool_text("42"),
                    false,
                )],
            ),
        ];

        let compactor = TruncatingCompactor::new()
            .with_preserve_recent(2)
            .with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 500, context).await;

        assert!(
            has_tool_call(&outcome.messages, "call_b"),
            "tool-call 'call_b' must be preserved"
        );
        assert!(
            has_tool_result(&outcome.messages, "call_b"),
            "tool-result for 'call_b' must be preserved"
        );
    }

    #[tokio::test]
    async fn compact_drops_both_call_and_result_when_in_old_portion() {
        // When both call and result are entirely in the old (dropped)
        // portion, the split should NOT be adjusted — both are dropped
        // together, which is correct.
        let messages = vec![
            Message::user("msg0"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("call_c", "tool", json!({}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "call_c",
                    "tool",
                    tool_text("done"),
                    false,
                )],
            ),
            Message::assistant("reply1"),
            Message::user("msg2"),
            Message::assistant("reply2"),
            Message::user("msg3"),
            Message::assistant("reply3"),
        ];

        let compactor = TruncatingCompactor::new()
            .with_preserve_recent(4)
            .with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 500, context).await;

        // Neither call_c nor its result should appear — both dropped.
        assert!(
            !has_tool_call(&outcome.messages, "call_c"),
            "tool-call 'call_c' should be dropped"
        );
        assert!(
            !has_tool_result(&outcome.messages, "call_c"),
            "tool-result for 'call_c' should be dropped"
        );
    }

    #[tokio::test]
    async fn compact_rechecks_newly_admitted_messages_for_orphaned_results() {
        // Interleaved ordering: pulling the split back for result "z"
        // admits a message carrying result "w" whose call stays dropped.
        // The adjustment must be re-applied until the split is stable, or
        // an unmatched result reaches the compacted output.
        let messages = vec![
            Message::user("q0"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("w", "Read", json!({"path": "w.rs"}))],
            ),
            Message::user("intermediate"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("z", "Read", json!({"path": "z.rs"}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "w",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "z",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::assistant("done"),
        ];
        let compactor = TruncatingCompactor::new()
            .with_preserve_recent(2)
            .with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 500, context).await;

        let call_ids: Vec<&str> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let result_ids: Vec<&str> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        let orphaned_results: Vec<&str> = result_ids
            .iter()
            .filter(|id| !call_ids.contains(id))
            .copied()
            .collect();
        assert!(
            orphaned_results.is_empty(),
            "newly admitted messages must be rechecked for orphaned results: {orphaned_results:?}"
        );
        assert!(
            has_tool_call(&outcome.messages, "w") && has_tool_result(&outcome.messages, "w"),
            "the second adjustment must pull in call 'w' alongside its admitted result"
        );
    }

    #[test]
    fn adjust_for_tool_pairs_returns_zero_when_split_is_zero() {
        let messages = convo_with_straddling_tool_pair();
        assert_eq!(TruncatingCompactor::adjust_for_tool_pairs(&messages, 0), 0);
    }

    #[test]
    fn adjust_for_tool_pairs_no_orphans_returns_original_split() {
        // No tool results in the recent portion → no adjustment.
        let messages = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
            Message::assistant("d"),
            Message::user("e"),
            Message::assistant("f"),
        ];
        assert_eq!(TruncatingCompactor::adjust_for_tool_pairs(&messages, 4), 4);
    }

    #[test]
    fn adjust_for_tool_pairs_moves_split_back_for_orphaned_result() {
        let messages = convo_with_straddling_tool_pair();
        // Naive split at index 6 would keep result (idx 6) but drop call (idx 5).
        // Should adjust back to 5.
        assert_eq!(TruncatingCompactor::adjust_for_tool_pairs(&messages, 6), 5);
    }

    #[tokio::test]
    async fn compact_short_conversation_returns_unchanged() {
        // Below min_messages, the conversation should pass through unchanged.
        let messages = vec![
            Message::user("hello"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("call_d", "tool", json!({}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "call_d",
                    "tool",
                    tool_text("ok"),
                    false,
                )],
            ),
        ];
        let compactor = TruncatingCompactor::new()
            .with_preserve_recent(2)
            .with_min_messages(6);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages.clone(), 500, context).await;
        assert_eq!(outcome.messages.len(), messages.len());
    }

    #[tokio::test]
    async fn preserved_first_message_does_not_orphan_its_tool_call() {
        let messages = vec![
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c1",
                    "Read",
                    json!({"path": "a.rs"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "c1",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::user("q2"),
            Message::assistant("a2"),
            Message::user("q3"),
            Message::assistant("a3"),
            Message::user("q4"),
            Message::assistant("a4"),
        ];
        let compactor = TruncatingCompactor::new().with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 1, context).await;
        assert!(outcome.success);
        let has_call = outcome.messages.iter().any(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, MessagePart::ToolCall { id, .. } if id == "c1"))
        });
        let has_result = outcome.messages.iter().any(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, MessagePart::ToolResult { call_id, .. } if call_id == "c1"))
        });
        assert!(
            !has_call || has_result,
            "module doc: the split adjustment avoids orphaning tool-call/result pairs — kept the call but dropped its result: {:?}",
            outcome.messages.len()
        );
    }

    #[tokio::test]
    async fn straddling_call_at_index_zero_reports_no_action() {
        // The call sits in the unconditionally-preserved first message
        // and its result lands in the recent slice, so the backward walk
        // pulls the split to 0 — nothing can be dropped, and the pass
        // must report no change (an unchanged list is classified
        // NoAction by the manager), not a compaction that reduced
        // nothing.
        let messages = vec![
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c1",
                    "Read",
                    json!({"path": "a.rs"}),
                )],
            ),
            Message::user("q1"),
            Message::assistant("a1"),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "c1",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::user("q2"),
            Message::assistant("a2"),
        ];
        let compactor = TruncatingCompactor::new()
            .with_min_messages(4)
            .with_preserve_recent(3);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages.clone(), 1, context).await;
        assert!(outcome.success);
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "a split of 0 keeps every message — the outcome must be the unchanged list"
        );
        assert_eq!(
            outcome.tokens_saved, 0,
            "no-action passes must claim zero savings"
        );
    }

    #[tokio::test]
    async fn pulled_result_message_does_not_strand_foreign_results() {
        // The result message for the first message's call also carries a
        // result for a call that stays dropped — the pull must not bring
        // that foreign result back without its call.
        let messages = vec![
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c1",
                    "Read",
                    json!({"path": "a.rs"}),
                )],
            ),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c2",
                    "Read",
                    json!({"path": "b.rs"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![
                    MessagePart::tool_result("c1", "Read", tool_text("ok"), false),
                    MessagePart::tool_result("c2", "Read", tool_text("ok"), false),
                ],
            ),
            Message::user("q2"),
            Message::assistant("a2"),
            Message::user("q3"),
            Message::assistant("a3"),
        ];
        let compactor = TruncatingCompactor::new().with_min_messages(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 1, context).await;
        assert!(outcome.success);

        let call_ids: Vec<&str> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let result_ids: Vec<&str> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        let orphaned_results: Vec<&str> = result_ids
            .iter()
            .filter(|id| !call_ids.contains(id))
            .copied()
            .collect();
        let orphaned_calls: Vec<&str> = call_ids
            .iter()
            .filter(|id| !result_ids.contains(id))
            .copied()
            .collect();
        assert!(
            orphaned_results.is_empty() && orphaned_calls.is_empty(),
            "module doc: tool-call/result pairs are never split — the compacted \
             list carries results without their calls {orphaned_results:?} and \
             calls without their results {orphaned_calls:?}"
        );
        assert!(
            call_ids.contains(&"c1") && result_ids.contains(&"c1"),
            "the pair the pull exists to repair must survive it"
        );
    }

    #[test]
    fn token_splitter_does_not_separate_a_call_from_its_result() {
        // The only boundary at or before the target sits between the
        // call and its result — a user message delivering tool results
        // continues the same turn, so the splitter must preserve the
        // whole conversation rather than split the pair.
        let messages = vec![
            Message::user("q1"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c1",
                    "Read",
                    json!({"path": "a.rs"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "c1",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::assistant("a2"),
            Message::user("q3"),
        ];
        let splitter = TokenSplitter::new()
            .with_min_messages(4)
            .with_preserve_recent(5);
        let split = splitter.split(&messages);
        let old_calls: Vec<&str> = split
            .to_compact
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let new_results: Vec<&str> = split
            .preserved
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        let separated: Vec<&str> = new_results
            .iter()
            .filter(|id| old_calls.contains(id))
            .copied()
            .collect();
        assert!(
            separated.is_empty(),
            "the split must never put a call into to_compact while its \
             result stays in preserved (split_index {}); the splitter \
             keeps the whole conversation instead",
            split.split_index
        );
    }

    #[test]
    fn splitter_skips_boundaries_that_straddle_a_later_result() {
        // The boundary candidate itself carries no tool results, but a
        // consecutive user message behind it delivers a result for a
        // call that would be split off — the boundary is not pair-safe.
        let messages = vec![
            Message::user("q1"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "c1",
                    "Read",
                    json!({"path": "a.rs"}),
                )],
            ),
            Message::user("ack"),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "c1",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::assistant("a2"),
        ];
        let splitter = TokenSplitter::new()
            .with_min_messages(4)
            .with_preserve_recent(5);
        let split = splitter.split(&messages);
        assert_eq!(
            split.split_index, 0,
            "a boundary that strands a call behind a result delivered in a \
             later message is not pair-safe — nothing is split"
        );
        assert!(
            split.to_compact.is_empty(),
            "the whole conversation stays preserved"
        );
    }

    fn part_counts(messages: &[Message]) -> Vec<(String, usize, usize)> {
        let mut calls: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut results: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for msg in messages {
            for part in &msg.parts {
                match part {
                    MessagePart::ToolCall { id, .. } => {
                        let counter = calls.entry(id.clone()).or_insert(0);
                        *counter = counter.saturating_add(1);
                    }
                    MessagePart::ToolResult { call_id, .. } => {
                        let counter = results.entry(call_id.clone()).or_insert(0);
                        *counter = counter.saturating_add(1);
                    }
                    _ => {}
                }
            }
        }
        let orphaned: Vec<(String, usize, usize)> = results
            .iter()
            .filter(|(id, _)| !calls.contains_key(id.as_str()))
            .map(|(id, r)| (id.clone(), 0, *r))
            .collect();
        calls
            .into_iter()
            .map(|(id, c)| {
                let r = results.get(&id).copied().unwrap_or(0);
                (id, c, r)
            })
            .chain(orphaned)
            .collect()
    }

    #[tokio::test]
    async fn reused_call_id_across_turns_keeps_pairs_distinct() {
        // Two completed turns reuse the id "x". A split between the
        // first turn's call and its result must not be fooled by the
        // second turn's call carrying the same id.
        let messages = vec![
            Message::user("q1"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("x", "Read", json!({"path": "x.rs"}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "x",
                    "Read",
                    json!({"path": "x2.rs"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("ok2"),
                    false,
                )],
            ),
            Message::assistant("a2"),
        ];
        let compactor = TruncatingCompactor::new()
            .with_min_messages(4)
            .with_preserve_recent(6);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 1, context).await;
        assert!(outcome.success);

        for (id, calls, results) in part_counts(&outcome.messages) {
            assert_eq!(
                calls, results,
                "each occurrence of reused call id {id:?} must keep its own \
                 pair — output carries {calls} calls and {results} results"
            );
        }
        let turn_one_call_kept = outcome.messages.iter().any(|m| {
            m.parts.iter().any(|p| {
                matches!(
                    p,
                    MessagePart::ToolCall { id, input, .. }
                        if id == "x" && input.get("path").is_some_and(|v| v == "x.rs")
                )
            })
        });
        let turn_one_result_kept = outcome.messages.iter().any(|m| {
            m.parts.iter().any(|p| {
                matches!(
                    p,
                    MessagePart::ToolResult { call_id, output, .. }
                        if call_id == "x" && output.to_string().contains("ok")
                )
            })
        });
        assert!(
            turn_one_call_kept && turn_one_result_kept,
            "the straddled first-turn pair is kept whole, not dropped to the \
             second turn's reused id"
        );
    }

    #[tokio::test]
    async fn first_message_pull_targets_its_own_result_occurrence() {
        // The preserved first message's call shares its id with a later
        // complete pair inside the recent slice; the pull must bring back
        // the first occurrence's own result, not accept the later one.
        let messages = vec![
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("x", "Read", json!({"path": "x.rs"}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("first"),
                    false,
                )],
            ),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("x", "Read", json!({"path": "y.rs"}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("second"),
                    false,
                )],
            ),
            Message::assistant("a2"),
            Message::user("q3"),
        ];
        let compactor = TruncatingCompactor::new()
            .with_min_messages(4)
            .with_preserve_recent(4);
        let context = make_context(&messages);
        let outcome = compactor.compact(messages, 1, context).await;
        assert!(outcome.success);

        for (id, calls, results) in part_counts(&outcome.messages) {
            assert_eq!(
                calls, results,
                "id {id:?}: every kept call occurrence must have its own \
                 result occurrence — {calls} calls vs {results} results"
            );
        }
        assert!(
            outcome
                .messages
                .iter()
                .any(|m| m.parts.iter().any(|p| matches!(
                    p,
                    MessagePart::ToolResult { output, .. } if output.to_string().contains("first")
                ))),
            "the first message's own result is pulled back alongside its call"
        );
    }

    #[test]
    fn splitter_allows_a_split_between_turns_reusing_one_call_id() {
        // Both turns are complete pairs; a boundary between them is
        // pair-safe even though the dropped turn's call id reappears in
        // the kept turn.
        let messages = vec![
            Message::user("q1"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call("x", "Read", json!({"path": "x.rs"}))],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("ok"),
                    false,
                )],
            ),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "x",
                    "Read",
                    json!({"path": "x2.rs"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::tool_result(
                    "x",
                    "Read",
                    tool_text("ok2"),
                    false,
                )],
            ),
            Message::assistant("a2"),
        ];
        let splitter = TokenSplitter::new()
            .with_min_messages(4)
            .with_preserve_recent(4);
        let split = splitter.split(&messages);
        assert!(
            split.split_index > 0,
            "a boundary between two complete turns is pair-safe even when \
             they reuse one call id — refusing it keeps the whole \
             conversation"
        );
        for (id, calls, results) in part_counts(&split.preserved) {
            assert_eq!(
                calls, results,
                "id {id:?}: the kept portion holds complete pairs — {calls} \
                 calls vs {results} results"
            );
        }
    }
}
