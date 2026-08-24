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
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

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
/// alongside it. If the conversation is shorter than `min_messages`, no
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
            // message containing that ToolCall.
            let split = Self::adjust_for_tool_pairs(&messages, initial_split);

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
    /// [`MessagePart::ToolResult`] whose matching
    /// [`MessagePart::ToolCall`] (identified by `call_id` == `id`) would
    /// be in the dropped portion (before `split`), the split is moved
    /// backward to include the message containing the orphaned call.
    fn adjust_for_tool_pairs(messages: &[Message], split: usize) -> usize {
        if split == 0 {
            return 0;
        }

        // Collect call IDs from the recent portion — those whose calls
        // are already preserved need no adjustment.
        let recent = messages.get(split..).unwrap_or_default();
        let recent_call_ids: HashSet<&String> = recent
            .iter()
            .flat_map(|msg| msg.parts.iter())
            .filter_map(|part| match part {
                MessagePart::ToolCall { id, .. } => Some(id),
                _ => None,
            })
            .collect();

        // Check each result in the recent portion: if its call_id is not
        // among the recent calls, the call is in the dropped portion.
        let orphaned_ids: Vec<&String> = recent
            .iter()
            .flat_map(|msg| msg.parts.iter())
            .filter_map(|part| match part {
                MessagePart::ToolResult { call_id, .. } => {
                    if recent_call_ids.contains(call_id) {
                        None
                    } else {
                        Some(call_id)
                    }
                }
                _ => None,
            })
            .collect();

        if orphaned_ids.is_empty() {
            return split;
        }

        // Walk backward from the split point to find the earliest message
        // that contains a ToolCall matching any orphaned ID.
        let mut new_split = split;
        for i in (0..split).rev() {
            let Some(msg) = messages.get(i) else {
                continue;
            };
            let has_orphaned_call = msg.parts.iter().any(|part| match part {
                MessagePart::ToolCall { id, .. } => orphaned_ids.contains(&id),
                _ => false,
            });
            if has_orphaned_call {
                new_split = i;
            }
        }

        new_split
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
    fn reattach_dropped_results(
        messages: &[Message],
        split: usize,
        kept: Vec<Message>,
    ) -> Vec<Message> {
        let first_call_ids: HashSet<&String> = kept
            .first()
            .into_iter()
            .flat_map(|msg| msg.parts.iter())
            .filter_map(|part| match part {
                MessagePart::ToolCall { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        if first_call_ids.is_empty() {
            return kept;
        }

        let kept_result_ids: HashSet<&String> = kept
            .iter()
            .flat_map(|msg| msg.parts.iter())
            .filter_map(|part| match part {
                MessagePart::ToolResult { call_id, .. } => Some(call_id),
                _ => None,
            })
            .collect();
        let missing: Vec<&String> = first_call_ids
            .iter()
            .filter(|id| !kept_result_ids.contains(*id))
            .copied()
            .collect();
        if missing.is_empty() {
            return kept;
        }

        let pulled: Vec<Message> = messages
            .get(1..split)
            .unwrap_or_default()
            .iter()
            .filter(|msg| {
                msg.parts.iter().any(|part| match part {
                    MessagePart::ToolResult { call_id, .. } => missing.contains(&call_id),
                    _ => false,
                })
            })
            .cloned()
            .collect();
        if pulled.is_empty() {
            return kept;
        }

        let mut kept_iter = kept.into_iter();
        let Some(first) = kept_iter.next() else {
            return pulled;
        };
        let mut out = Vec::with_capacity(pulled.len().saturating_add(kept_iter.len()));
        out.push(first);
        out.extend(pulled);
        out.extend(kept_iter);
        out
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
    /// Zero when no split was needed (the entire conversation was preserved).
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
    /// from assistant to user) as close as possible to leaving
    /// `preserve_recent` messages in the recent portion.
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

    /// Find the nearest turn boundary at or before the target index.
    ///
    /// A turn boundary is a position where the previous message is
    /// assistant-role and the next is user-role. This ensures we split
    /// at a coherent conversation boundary.
    fn find_turn_boundary(messages: &[Message], target: usize) -> usize {
        if target == 0 {
            return 0;
        }

        // Search backwards from target for an assistant→user transition.
        for i in (1..=target).rev() {
            if i < messages.len() {
                let Some(prev) = messages.get(i.saturating_sub(1)) else {
                    continue;
                };
                let Some(curr) = messages.get(i) else {
                    continue;
                };
                let prev_is_assistant = prev.role == Role::Assistant;
                let curr_is_user = curr.role == Role::User;
                if prev_is_assistant && curr_is_user {
                    return i;
                }
            }
        }

        // Fallback: no clean boundary found, split at target.
        target
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
}
