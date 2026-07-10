//! Context compaction for agent conversations.
//!
//! When a [`ContextManager`](crate::compact::ContextManager) is configured,
//! checks token usage after each tool dispatch and triggers compaction if the
//! conversation exceeds the context window threshold.

use super::{ApiClient, BareLoop, Instant, LoopError};
#[cfg(feature = "hooks")]
use super::{CompactTrigger, PostCompactContext, PreCompactContext};
use crate::compact::EnsureContextResult;

use crate::capabilities::Compactable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
use crate::observer::CompactedContext;

impl<C: ApiClient> BareLoop<C> {
    /// Check if context compaction is needed and perform it if so.
    ///
    /// When a [`crate::compact::ContextManager`] is configured, this method:
    /// 1. Calls [`ContextManager::ensure_context_fits`](crate::compact::ContextManager::ensure_context_fits) to check token usage.
    /// 2. If compaction occurred, replaces `self.conversation` with the compacted messages.
    /// 3. Notifies observers via [`LoopObserver::on_compaction`](crate::observer::LoopObserver::on_compaction).
    ///
    /// When no `ContextManager` is set, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::ContextExceeded`] if compaction was needed but failed
    /// (i.e. the conversation exceeds the context window and the compactor
    /// could not reduce it sufficiently).
    pub(super) async fn maybe_compact_context(&mut self, turn: usize) -> Result<(), LoopError> {
        let Some(ctx_manager) = self.managers.context_manager() else {
            return Ok(());
        };

        if self.pre_compact_hook_aborts() {
            return Ok(());
        }

        let messages_before = self.conversation.len();
        let compact_start = Instant::now();
        let conversation = self.conversation.clone();
        let result = ctx_manager.ensure_context_fits(conversation, turn).await;

        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                let tokens_after = outcome.tokens_after;
                let tokens_saved = outcome.tokens_saved;
                self.conversation = outcome.messages;
                let tokens_before = tokens_after.saturating_add(tokens_saved);
                self.managers.observers().on_compaction(&CompactedContext {
                    tokens_before,
                    tokens_after,
                    tokens_saved,
                });
                self.notify_post_compact_hook(
                    messages_before,
                    tokens_after,
                    tokens_saved,
                    compact_start.elapsed(),
                );
                Ok(())
            }
            Ok(EnsureContextResult::NoAction(messages)) => {
                self.conversation = messages;
                Ok(())
            }
            Err(overflow) => Err(LoopError::ContextExceeded {
                used: overflow.tokens_used,
                limit: overflow.context_window,
            }),
        }
    }

    /// Run the pre-compact hook. Returns `true` if the hook aborts compaction.
    #[cfg(feature = "hooks")]
    fn pre_compact_hook_aborts(&self) -> bool {
        let Some(executor) = self.managers.hook_executor() else {
            return false;
        };
        let tokens_before = crate::compact::CompactionOutcome::estimate_tokens(&self.conversation);
        let ctx = PreCompactContext {
            trigger: CompactTrigger::Auto,
            custom_instructions: None,
            message_count: self.conversation.len(),
            tokens_before,
            context_window: self.config.context_window,
            session_id: self.config.session_id,
        };
        executor.check_pre_compact(&ctx).abort
    }

    #[cfg(not(feature = "hooks"))]
    fn pre_compact_hook_aborts(&self) -> bool {
        false
    }

    /// Notify the post-compact hook that compaction completed.
    #[cfg(feature = "hooks")]
    fn notify_post_compact_hook(
        &self,
        messages_before: usize,
        tokens_after: u64,
        tokens_saved: u64,
        duration: std::time::Duration,
    ) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let messages_after = self.conversation.len();
        let ctx = PostCompactContext {
            trigger: CompactTrigger::Auto,
            messages_compacted: messages_before.saturating_sub(messages_after),
            tokens_saved,
            tokens_after,
            duration_ms: u64::try_from(duration.as_millis()).unwrap_or(0),
            session_id: self.config.session_id,
        };
        executor.notify_post_compact(&ctx);
    }

    #[cfg(not(feature = "hooks"))]
    fn notify_post_compact_hook(
        &self,
        _messages_before: usize,
        _tokens_after: u64,
        _tokens_saved: u64,
        _duration: std::time::Duration,
    ) {
    }
}
