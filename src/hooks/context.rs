//! Hook context types — data passed to hooks at each lifecycle point.
//!
//! Each context type captures the relevant state for its trigger point.
//! Contexts are read-only snapshots; hooks cannot mutate agent state
//! directly (they return `HookAction` or `CompactResult` to influence flow).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ===================================================
// Tool-use contexts
// ===================================================

/// Context provided to `on_pre_tool_use` hooks.
///
/// Contains everything a hook needs to decide whether to allow,
/// block, or ask about a tool invocation.
#[derive(Debug, Clone)]
pub struct PreToolUseContext {
    /// Name of the tool about to be invoked.
    pub tool_name: String,
    /// JSON input that will be passed to the tool.
    pub input: Value,
    /// Session ID of the running agent.
    pub session_id: uuid::Uuid,
    /// Current turn number within the session (0-indexed).
    pub turn_number: usize,
}

/// Context provided to `on_post_tool_use` hooks.
///
/// Captures the outcome of a tool invocation. Post-hooks are
/// notification-only and cannot alter the result.
#[derive(Debug, Clone)]
pub struct PostToolUseContext {
    /// Name of the tool that was invoked.
    pub tool_name: String,
    /// JSON input that was passed to the tool.
    pub input: Value,
    /// Output produced by the tool (stringified).
    pub output: String,
    /// Whether the tool reported an error.
    pub is_error: bool,
    /// Wall-clock execution time in milliseconds.
    pub duration_ms: u64,
    /// Session ID of the running agent.
    pub session_id: uuid::Uuid,
    /// Turn number within the session.
    pub turn_number: usize,
}

// ===================================================
// Compaction contexts
// ===================================================

/// Why compaction was triggered.
///
/// Distinguishes between automatic threshold-based compaction
/// and manual compaction requested by the agent or user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactTrigger {
    /// Automatic compaction triggered by token threshold.
    Auto,
    /// Manual compaction requested by the agent or user.
    Manual,
}

impl From<crate::compact::types::CompactReason> for CompactTrigger {
    /// Map the compaction pipeline's [`CompactReason`](crate::compact::types::CompactReason) into the hook-level
    /// [`CompactTrigger`].
    ///
    /// [`ThresholdExceeded`](crate::compact::types::CompactReason::ThresholdExceeded)
    /// and
    /// [`Emergency`](crate::compact::types::CompactReason::Emergency)
    /// are both automatic triggers, while
    /// [`Manual`](crate::compact::types::CompactReason::Manual) maps to
    /// [`Manual`](CompactTrigger::Manual).
    fn from(reason: crate::compact::types::CompactReason) -> Self {
        match reason {
            crate::compact::types::CompactReason::ThresholdExceeded
            | crate::compact::types::CompactReason::Emergency => CompactTrigger::Auto,
            crate::compact::types::CompactReason::Manual => CompactTrigger::Manual,
        }
    }
}

/// Context provided to `on_pre_compact` hooks.
///
/// Hooks can inspect the current state and decide to abort
/// compaction or inject additional instructions.
#[derive(Debug, Clone)]
pub struct PreCompactContext {
    /// Why compaction was triggered.
    pub trigger: CompactTrigger,
    /// Custom instructions from prior hooks (accumulated).
    pub custom_instructions: Option<String>,
    /// Number of messages in the conversation before compaction.
    pub message_count: usize,
    /// Estimated token count before compaction.
    pub tokens_before: u64,
    /// Model context window size in tokens.
    pub context_window: u64,
    /// Session ID of the running agent.
    pub session_id: uuid::Uuid,
}

/// Context provided to `on_post_compact` hooks.
///
/// Notification-only; cannot alter the compaction result.
#[derive(Debug, Clone)]
pub struct PostCompactContext {
    /// Why compaction was triggered.
    pub trigger: CompactTrigger,
    /// Number of messages removed by compaction.
    pub messages_compacted: usize,
    /// Tokens saved by compaction.
    pub tokens_saved: u64,
    /// Estimated tokens after compaction.
    pub tokens_after: u64,
    /// Compaction duration in milliseconds.
    pub duration_ms: u64,
    /// Session ID of the running agent.
    pub session_id: uuid::Uuid,
}

/// Result of a pre-compact hook check.
///
/// Unlike tool hooks (Allow/Block/Ask), compact hooks have richer semantics:
/// multiple hooks may want to inject context or instructions. The executor
/// accumulates results, with abort taking priority.
#[derive(Debug, Clone, Default)]
pub struct CompactResult {
    /// Whether to abort compaction.
    pub abort: bool,
    /// Reason for abort (shown to the user/agent).
    pub abort_reason: Option<String>,
    /// Override custom instructions for the summarizer.
    pub new_instructions: Option<String>,
    /// Additional context strings to include in the summary.
    pub additional_context: Vec<String>,
}

impl CompactResult {
    /// Allow compaction to proceed (default).
    #[must_use]
    pub fn allow() -> Self {
        Self::default()
    }

    /// Abort compaction with a reason.
    pub fn abort(reason: impl Into<String>) -> Self {
        Self {
            abort: true,
            abort_reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// Inject additional context into the summary.
    #[must_use]
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.additional_context.push(ctx.into());
        self
    }

    /// Override the summarization instructions.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.new_instructions = Some(instructions.into());
        self
    }
}

// ===================================================
// Session lifecycle contexts
// ===================================================

/// Why the session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionEndReason {
    /// Normal completion (agent converged).
    Complete,
    /// User interrupted (Ctrl+C / cancel signal).
    Cancelled,
    /// Agent error.
    Error,
    /// Maximum turns reached.
    MaxTurns,
    /// Context limit exceeded and compaction failed.
    ContextOverflow,
}

/// Context provided to `on_session_start` hooks.
///
/// Notification-only; fires once at session start.
#[derive(Debug, Clone)]
pub struct SessionStartContext {
    /// Session ID.
    pub session_id: uuid::Uuid,
    /// Model identifier.
    pub model: String,
    /// Working directory the agent was started in.
    pub working_directory: String,
}

/// Context provided to `on_session_end` hooks.
///
/// Notification-only; fires once at session end.
#[derive(Debug, Clone)]
pub struct SessionEndContext {
    /// Session ID.
    pub session_id: uuid::Uuid,
    /// Why the session ended.
    pub reason: SessionEndReason,
    /// Total turns executed.
    pub total_turns: usize,
    /// Total tokens consumed (input + output across all turns).
    pub total_tokens: u64,
    /// Wall-clock session duration in seconds.
    pub duration_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_result_allow_is_default() {
        let result = CompactResult::allow();
        assert!(!result.abort);
        assert!(result.abort_reason.is_none());
        assert!(result.new_instructions.is_none());
        assert!(result.additional_context.is_empty());
    }

    #[test]
    fn compact_result_abort_sets_reason() {
        let result = CompactResult::abort("too risky");
        assert!(result.abort);
        assert_eq!(result.abort_reason.as_deref(), Some("too risky"));
    }

    #[test]
    fn compact_result_builder_pattern() {
        let result = CompactResult::allow()
            .with_context("keep file X")
            .with_context("remember Y")
            .with_instructions("focus on Z");
        assert!(!result.abort);
        assert_eq!(result.additional_context.len(), 2);
        assert_eq!(result.new_instructions.as_deref(), Some("focus on Z"));
    }

    #[test]
    fn compact_trigger_serialization() {
        for trigger in [CompactTrigger::Auto, CompactTrigger::Manual] {
            let json = serde_json::to_string(&trigger).expect("serialize");
            let back: CompactTrigger = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, trigger);
        }
    }

    #[test]
    fn session_end_reason_serialization() {
        let reasons = [
            SessionEndReason::Complete,
            SessionEndReason::Cancelled,
            SessionEndReason::Error,
            SessionEndReason::MaxTurns,
            SessionEndReason::ContextOverflow,
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).expect("serialize");
            let back: SessionEndReason = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, reason);
        }
    }

    #[test]
    fn compact_result_abort_with_no_additional_fields() {
        let result = CompactResult::abort("reason");
        assert!(result.abort);
        assert_eq!(result.abort_reason.as_deref(), Some("reason"));
        assert!(
            result.new_instructions.is_none(),
            "should have no new_instructions"
        );
        assert!(
            result.additional_context.is_empty(),
            "should have empty additional_context"
        );
    }
}
