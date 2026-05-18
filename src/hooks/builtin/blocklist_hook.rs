//! Blocklist hook — blocks or allows tools by name.
//!
//! Provides a simple denylist/allowlist mechanism for tool access control.

use crate::hooks::Hook;
use crate::hooks::HookAction;
use crate::hooks::context::PreToolUseContext;

/// A hook that blocks tools by name. Simple allowlist/denylist.
///
/// Useful for sandboxing agents to a subset of tools, or blocking
/// dangerous tools in production.
///
/// # Order of Precedence
///
/// The denylist takes precedence over the allowlist. If a tool appears in
/// both lists, it is blocked.
///
/// # Examples
///
/// ```
/// use loopctl::hooks::builtin::BlocklistHook;
/// use loopctl::hooks::Hook;
///
/// // Denylist: block specific dangerous tools
/// let hook = BlocklistHook::deny(vec!["rm_rf".to_string(), "format_disk".to_string()]);
/// assert_eq!(hook.name(), "blocklist");
///
/// // Allowlist: only permit specific tools
/// let hook = BlocklistHook::allow_only(vec!["read_file".to_string(), "search".to_string()]);
/// assert_eq!(hook.name(), "blocklist");
/// ```
pub struct BlocklistHook {
    /// Tool names to block (denylist). Takes precedence over allowlist.
    blocked: Vec<String>,
    /// If non-empty, only these tools are allowed (allowlist).
    allowed: Vec<String>,
}

impl BlocklistHook {
    /// Create a denylist hook — blocks the named tools.
    #[must_use]
    pub fn deny(blocked: Vec<String>) -> Self {
        Self {
            blocked,
            allowed: Vec::new(),
        }
    }

    /// Create an allowlist hook — only the named tools are permitted.
    #[must_use]
    pub fn allow_only(allowed: Vec<String>) -> Self {
        Self {
            blocked: Vec::new(),
            allowed,
        }
    }
}

impl Hook for BlocklistHook {
    fn name(&self) -> &'static str {
        "blocklist"
    }

    fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
        // Denylist takes precedence
        if self.blocked.iter().any(|b| b == &ctx.tool_name) {
            return Some(HookAction::block(format!(
                "Tool '{}' is blocked by policy",
                ctx.tool_name
            )));
        }
        // Allowlist: if non-empty, only listed tools are permitted
        if !self.allowed.is_empty() && !self.allowed.iter().any(|a| a == &ctx.tool_name) {
            return Some(HookAction::block(format!(
                "Tool '{}' is not in the allowlist",
                ctx.tool_name
            )));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_ctx(tool_name: &str) -> PreToolUseContext {
        PreToolUseContext {
            tool_name: tool_name.to_string(),
            input: json!({}),
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        }
    }

    #[test]
    fn deny_blocks_listed_tools() {
        let hook = BlocklistHook::deny(vec!["rm_rf".to_string()]);
        let action = hook.on_pre_tool_use(&make_ctx("rm_rf")).unwrap();
        assert!(action.is_block());
    }

    #[test]
    fn deny_allows_unlisted_tools() {
        let hook = BlocklistHook::deny(vec!["rm_rf".to_string()]);
        assert!(hook.on_pre_tool_use(&make_ctx("read_file")).is_none());
    }

    #[test]
    fn allow_only_permits_listed_tools() {
        let hook = BlocklistHook::allow_only(vec!["read_file".to_string()]);
        assert!(hook.on_pre_tool_use(&make_ctx("read_file")).is_none());
    }

    #[test]
    fn allow_only_blocks_unlisted_tools() {
        let hook = BlocklistHook::allow_only(vec!["read_file".to_string()]);
        let action = hook.on_pre_tool_use(&make_ctx("rm_rf")).unwrap();
        assert!(action.is_block());
    }

    #[test]
    fn deny_takes_precedence_over_allow() {
        let hook = BlocklistHook {
            blocked: vec!["rm_rf".to_string()],
            allowed: vec!["rm_rf".to_string()],
        };
        let action = hook.on_pre_tool_use(&make_ctx("rm_rf")).unwrap();
        assert!(action.is_block());
    }

    #[test]
    fn empty_allowlist_allows_everything() {
        let hook = BlocklistHook::allow_only(Vec::new());
        assert!(hook.on_pre_tool_use(&make_ctx("anything")).is_none());
    }

    #[test]
    fn block_reason_contains_tool_name() {
        let hook = BlocklistHook::deny(vec!["dangerous".to_string()]);
        match hook.on_pre_tool_use(&make_ctx("dangerous")) {
            Some(HookAction::Block { reason }) => {
                assert!(reason.contains("dangerous"));
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }
}
