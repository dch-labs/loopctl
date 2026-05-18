//! Confirmation hook — requires interactive confirmation for specific tools.
//!
//! Delegates to a [`ConfirmationHandler`] trait that the agent provides
//! (e.g., a TUI prompt, a CLI yes/no, a webhook approval flow).

use std::sync::Arc;

use crate::hooks::Hook;
use crate::hooks::HookAction;
use crate::hooks::context::PreToolUseContext;

/// Trait for interactive confirmation UI.
///
/// The framework doesn't know how to prompt the user (CLI? TUI? web UI?).
/// Agents provide the handler. The framework provides the policy
/// (which tools to confirm).
pub trait ConfirmationHandler: Send + Sync {
    /// Ask the user whether to proceed. Returns `true` to allow.
    fn confirm(&self, message: &str) -> bool;
}

/// A hook that requires interactive confirmation for listed tools.
///
/// # Example
///
/// ```
/// use loopctl::hooks::builtin::{ConfirmationHook, ConfirmationHandler};
/// use loopctl::hooks::Hook;
/// use std::sync::Arc;
///
/// struct CliConfirmation;
/// impl ConfirmationHandler for CliConfirmation {
///     fn confirm(&self, _message: &str) -> bool {
///         true
///     }
/// }
///
/// let hook = ConfirmationHook::new(
///     vec!["delete_file".to_string(), "deploy".to_string()],
///     Arc::new(CliConfirmation),
/// );
/// assert_eq!(hook.name(), "confirmation");
/// ```
pub struct ConfirmationHook {
    /// Tool names that require confirmation.
    tools: Vec<String>,
    /// The confirmation UI (provided by the agent).
    handler: Arc<dyn ConfirmationHandler>,
}

impl ConfirmationHook {
    /// Create a new confirmation hook for the specified tools.
    pub fn new(tools: Vec<String>, handler: Arc<dyn ConfirmationHandler>) -> Self {
        Self { tools, handler }
    }
}

impl Hook for ConfirmationHook {
    fn name(&self) -> &'static str {
        "confirmation"
    }

    fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
        if self.tools.iter().any(|t| t == &ctx.tool_name) {
            let msg = format!("Allow tool '{}' to execute?", ctx.tool_name);
            if self.handler.confirm(&msg) {
                None // allow
            } else {
                Some(HookAction::block("User denied permission"))
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysAllow;
    impl ConfirmationHandler for AlwaysAllow {
        fn confirm(&self, _message: &str) -> bool {
            true
        }
    }

    struct AlwaysDeny;
    impl ConfirmationHandler for AlwaysDeny {
        fn confirm(&self, _message: &str) -> bool {
            false
        }
    }

    fn make_ctx(tool_name: &str) -> PreToolUseContext {
        PreToolUseContext {
            tool_name: tool_name.to_string(),
            input: serde_json::json!({}),
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        }
    }

    #[test]
    fn confirmation_allows_when_confirmed() {
        let hook = ConfirmationHook::new(vec!["deploy".to_string()], Arc::new(AlwaysAllow));
        assert!(hook.on_pre_tool_use(&make_ctx("deploy")).is_none());
    }

    #[test]
    fn confirmation_blocks_when_denied() {
        let hook = ConfirmationHook::new(vec!["deploy".to_string()], Arc::new(AlwaysDeny));
        let action = hook.on_pre_tool_use(&make_ctx("deploy")).unwrap();
        assert!(action.is_block());
    }

    #[test]
    fn confirmation_skips_unlisted_tools() {
        let hook = ConfirmationHook::new(vec!["deploy".to_string()], Arc::new(AlwaysDeny));
        assert!(hook.on_pre_tool_use(&make_ctx("read_file")).is_none());
    }
}
