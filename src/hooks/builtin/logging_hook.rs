//! Logging hook — traces every pre/post tool use and compact event.
//!
//! Useful for development and debugging. Has no side effects.

use crate::hooks::Hook;
use crate::hooks::HookAction;
use crate::hooks::context::{
    CompactResult, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
};

/// A hook that logs every pre/post tool use and compact event via `tracing`.
///
/// Useful for development and debugging. Has no side effects.
///
/// # Example
///
/// ```
/// use loopctl::hooks::builtin::LoggingHook;
/// use loopctl::hooks::HookExecutor;
/// use std::sync::Arc;
///
/// let executor = HookExecutor::new()
///     .with_hook(Arc::new(LoggingHook));
///
/// assert_eq!(executor.hook_count(), 1);
/// ```
pub struct LoggingHook;

impl Hook for LoggingHook {
    fn name(&self) -> &'static str {
        "logging"
    }

    fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
        tracing::debug!(tool = %ctx.tool_name, turn = ctx.turn_number, "pre-tool-use");
        None
    }

    fn on_post_tool_use(&self, ctx: &PostToolUseContext) {
        tracing::debug!(
            tool = %ctx.tool_name,
            success = !ctx.is_error,
            duration_ms = ctx.duration_ms,
            "post-tool-use"
        );
    }

    fn on_pre_compact(&self, ctx: &PreCompactContext) -> Option<CompactResult> {
        tracing::debug!(
            trigger = ?ctx.trigger,
            tokens_before = ctx.tokens_before,
            messages = ctx.message_count,
            "pre-compact"
        );
        None
    }

    fn on_post_compact(&self, ctx: &PostCompactContext) {
        tracing::debug!(
            messages_compacted = ctx.messages_compacted,
            tokens_saved = ctx.tokens_saved,
            "post-compact"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::context::{
        CompactTrigger, PostCompactContext, PostToolUseContext, SessionEndContext,
        SessionEndReason, SessionStartContext,
    };
    use serde_json::json;

    #[test]
    fn logging_hook_allows_tool_use() {
        let hook = LoggingHook;
        let ctx = PreToolUseContext {
            tool_name: "test".to_string(),
            input: json!({}),
            session_id: uuid::Uuid::nil(),
            turn_number: 1,
        };
        assert!(hook.on_pre_tool_use(&ctx).is_none());
    }

    #[test]
    fn logging_hook_pre_compact_allows() {
        let hook = LoggingHook;
        let ctx = PreCompactContext {
            trigger: CompactTrigger::Auto,
            custom_instructions: None,
            message_count: 5,
            tokens_before: 1000,
            context_window: 2000,
            session_id: uuid::Uuid::nil(),
        };
        assert!(hook.on_pre_compact(&ctx).is_none());
    }

    #[test]
    fn logging_hook_post_tool_use_no_panic() {
        let hook = LoggingHook;
        let ctx = PostToolUseContext {
            tool_name: "test".to_string(),
            input: json!({}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        hook.on_post_tool_use(&ctx);
    }

    #[test]
    fn logging_hook_post_compact_no_panic() {
        let hook = LoggingHook;
        let ctx = PostCompactContext {
            trigger: CompactTrigger::Auto,
            messages_compacted: 5,
            tokens_saved: 100,
            tokens_after: 900,
            duration_ms: 50,
            session_id: uuid::Uuid::nil(),
        };
        hook.on_post_compact(&ctx);
    }

    #[test]
    fn logging_hook_session_start_no_panic() {
        let hook = LoggingHook;
        let ctx = SessionStartContext {
            session_id: uuid::Uuid::nil(),
            model: "test".to_string(),
            working_directory: "/tmp".to_string(),
        };
        hook.on_session_start(&ctx);
    }

    #[test]
    fn logging_hook_session_end_no_panic() {
        let hook = LoggingHook;
        let ctx = SessionEndContext {
            session_id: uuid::Uuid::nil(),
            reason: SessionEndReason::Complete,
            total_turns: 5,
            total_tokens: 1000,
            duration_secs: 30,
        };
        hook.on_session_end(&ctx);
    }
}
