//! Context compaction phase — check and compact the conversation when needed.
//!
//! Extracted from [`BareLoop`] to isolate the compaction concern.
//! When a [`ContextManager`] is configured, checks token usage after each
//! tool dispatch and triggers compaction if usage exceeds the threshold.

use super::{ApiClient, BareLoop, EnsureContextResult, Instant, LoopError};
#[cfg(feature = "hooks")]
use super::{CompactTrigger, PostCompactContext, PreCompactContext};

use crate::observer::CompactedContext;

impl<C: ApiClient> BareLoop<C> {
    /// Check if context compaction is needed and perform it if so.
    ///
    /// When a [`ContextManager`] is configured, this method:
    /// 1. Calls [`ContextManager::ensure_context_fits()`] to check token usage.
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
        let Some(ref ctx_manager) = self.context_manager else {
            return Ok(());
        };

        #[cfg(feature = "hooks")]
        let messages_before = self.conversation.len();

        // Pre-compact hook check
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let tokens_before =
                crate::compact::CompactionOutcome::estimate_tokens(&self.conversation);
            let ctx = PreCompactContext {
                trigger: CompactTrigger::Auto,
                custom_instructions: None,
                message_count: messages_before,
                tokens_before,
                context_window: self.config.context_window,
                session_id: self.config.session_id,
            };
            let hook_result = executor.check_pre_compact(&ctx);
            if hook_result.abort {
                // Hook aborted compaction — return Ok, conversation unchanged.
                return Ok(());
            }
            // Note: hook_result.new_instructions and hook_result.additional_context
            // are available for future use with a hook-aware compactor.
        }

        let compact_start = Instant::now();
        let result = ctx_manager
            .ensure_context_fits(std::mem::take(&mut self.conversation), turn)
            .await;
        #[cfg(feature = "hooks")]
        let compact_duration_ms = u64::try_from(compact_start.elapsed().as_millis()).unwrap_or(0);
        #[cfg(not(feature = "hooks"))]
        let _ = compact_start;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                self.conversation = outcome.messages;
                #[cfg(feature = "hooks")]
                let messages_after = self.conversation.len();
                let tokens_before = outcome.tokens_after.saturating_add(outcome.tokens_saved);
                self.managers.observers().on_compaction(&CompactedContext {
                    tokens_before,
                    tokens_after: outcome.tokens_after,
                    tokens_saved: outcome.tokens_saved,
                });

                // Post-compact hook notification
                #[cfg(feature = "hooks")]
                if let Some(ref executor) = self.hook_executor {
                    let messages_compacted = messages_before.saturating_sub(messages_after);
                    let ctx = PostCompactContext {
                        trigger: CompactTrigger::Auto,
                        messages_compacted,
                        tokens_saved: outcome.tokens_saved,
                        tokens_after: outcome.tokens_after,
                        duration_ms: compact_duration_ms,
                        session_id: self.config.session_id,
                    };
                    executor.notify_post_compact(&ctx);
                }

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
}
