//! Truncating compactor and token splitter.
//!
//! Contents:
//!
//! - [`TruncatingCompactor`] — a simple compactor that drops the oldest messages.
//! - [`TokenSplitter`] — splits a conversation into "old" and "recent" at a turn boundary.
//! - [`SplitResult`] — result of splitting a conversation.

use crate::compact::types::{CompactionContext, CompactionOutcome};
use crate::compact::{ContextCompactor, ContextManager};
use crate::message::{Message, Role};
use std::future::Future;
use std::pin::Pin;

// ===================================================
// TruncatingCompactor
// ===================================================

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
/// behavior. If the conversation is shorter than `min_messages`, no
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
    /// Number of recent messages to always preserve.
    preserve_recent: usize,
    /// Minimum messages before compaction is considered.
    min_messages: usize,
}

impl TruncatingCompactor {
    /// Create a new truncating compactor with sensible defaults.
    ///
    /// Defaults:
    ///
    /// | Setting           | Default |
    /// |-------------------|---------|
    /// | `preserve_recent` | 4       |
    /// | `min_messages`    | 6       |
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

    /// Number of recent messages that will be preserved.
    #[must_use]
    pub fn preserve_recent(&self) -> usize {
        self.preserve_recent
    }

    /// Minimum messages before compaction is attempted.
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
            let split = total.saturating_sub(self.preserve_recent);
            let recent: Vec<Message> = messages.get(split..).unwrap_or_default().to_vec();

            // Always preserve the first message (typically the system prompt)
            // unless it is already included in the recent slice (split == 0).
            let preserved = if split > 0 {
                if let Some(first) = messages.first() {
                    let mut v = vec![first.clone()];
                    v.extend(recent);
                    v
                } else {
                    recent
                }
            } else {
                // split == 0 means recent already contains all messages.
                recent
            };

            let tokens_after = CompactionOutcome::estimate_tokens(&preserved);
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

// ===================================================
// TokenSplitter
// ===================================================

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
    /// Number of recent messages to always preserve.
    preserve_recent: usize,
    /// Minimum messages before considering a split.
    min_messages: usize,
}

/// Result of splitting a conversation into old and recent portions.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Messages to compact or summarize (the "old" part).
    pub to_compact: Vec<Message>,
    /// Messages to preserve as-is (the "recent" part).
    pub preserved: Vec<Message>,
    /// Estimated tokens in `to_compact`.
    pub compact_tokens: u64,
    /// Estimated tokens in `preserved`.
    pub preserved_tokens: u64,
    /// The index in the original message list where the split occurred.
    pub split_index: usize,
}

impl TokenSplitter {
    /// Create a new splitter with sensible defaults.
    ///
    /// Defaults:
    ///
    /// | Setting           | Default |
    /// |-------------------|---------|
    /// | `preserve_recent` | 4       |
    /// | `min_messages`    | 6       |
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_recent: 4,
            min_messages: 6,
        }
    }

    /// Set how many recent messages to preserve.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the minimum messages before splitting is considered.
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
                preserved_tokens: ContextManager::estimate_tokens(messages),
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
            compact_tokens: ContextManager::estimate_tokens(to_compact),
            preserved_tokens: ContextManager::estimate_tokens(preserved),
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
