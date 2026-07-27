//! Permission checking for tool invocations.
//!
//! [`PermissionCheck`] — the result type returned by
//! the agent loop's permission gate before a tool is allowed to execute.
//! See the [`PermissionCheck`] documentation for the full decision tree.

use serde_json::Value;

/// Result of a permission check before tool execution.
///
/// Before invoking [`Tool::call`](super::Tool::call), the agent loop can run a
/// permission gate that returns one of four outcomes: allow, deny, ask the user,
/// or modify the input. This lets host applications enforce safety
/// policies without modifying individual tool implementations.
///
/// # Example
///
/// ```rust,ignore
/// let check = PermissionCheck::deny("dangerous operation");
/// if check.is_deny() {
///     return Err(ToolError::Permission("blocked by policy".into()));
/// }
/// ```
#[derive(Debug, Clone)]
pub enum PermissionCheck {
    /// Allow the tool to execute unmodified.
    ///
    /// The agent loop proceeds with the original input and context.
    Allow,

    /// Deny execution with a human-readable reason.
    ///
    /// The agent loop should return
    /// [`ToolError::Permission`](super::ToolError::Permission) with the given
    /// `reason` so the LLM can react accordingly.
    Deny {
        /// Explanation forwarded to the LLM as part of the error.
        ///
        /// Surfaced inside [`ToolError::Permission`](super::ToolError::Permission)
        /// so the model can see *why* its call was rejected and adjust
        /// its next action. Keep it concrete and actionable — for
        /// example `"shell execution is disabled"` rather than just
        /// `"denied"`.
        reason: String,
    },

    /// Prompt the user for approval before proceeding.
    ///
    /// In interactive sessions the agent loop should present `prompt` to
    /// the user and then treat the response as either [`Allow`](PermissionCheck::Allow)
    /// or [`Deny`](PermissionCheck::Deny).
    Ask {
        /// Prompt text to present to the user for approval.
        ///
        /// Should clearly describe the action and its potential side
        /// effects so the user can make an informed decision — for
        /// example `"Allow write to /etc/config.yaml?"`. In
        /// non-interactive sessions the loop treats an `Ask` as a
        /// `Deny`, so this string is primarily for interactive hosts.
        prompt: String,
    },

    /// Modify the tool's input before execution.
    ///
    /// The agent loop should invoke [`Tool::call`](super::Tool::call) with
    /// `modified_input` instead of the original input. Useful for sanitising
    /// paths, redacting secrets, or injecting default values.
    Modify {
        /// Rewritten input to pass to [`Tool::call`](super::Tool::call)
        /// in place of the original.
        ///
        /// Must conform to the tool's
        /// [`ToolSchema::input_schema`](super::ToolSchema::input_schema)
        /// — the loop does not re-validate it. Use this to sanitise
        /// paths, redact secrets, or inject default values before the
        /// tool sees the input.
        modified_input: Value,
    },
}

impl PermissionCheck {
    /// Create an [`Allow`](PermissionCheck::Allow) result.
    ///
    /// Signals that the tool invocation may proceed without changes.
    /// The `#[must_use]` attribute reminds callers to check the result
    /// rather than silently discarding it.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation is within the configured safety policy — for example,
    /// a read-only tool invocation or an operation on an allowed path.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::allow();
    /// assert!(check.is_allow());
    /// ```
    #[must_use]
    pub fn allow() -> Self {
        Self::Allow
    }

    /// Create a [`Deny`](PermissionCheck::Deny) result with a reason.
    ///
    /// The `reason` string will be forwarded to the LLM as part of the
    /// error message, helping it understand why the invocation was
    /// rejected and adjust its next action.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation violates a hard safety rule — for example, executing
    /// a shell command when shell access is disabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::deny("shell execution is disabled");
    /// assert!(check.is_deny());
    /// ```
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Create an [`Ask`](PermissionCheck::Ask) result with a prompt.
    ///
    /// The agent loop should present the `prompt` to the user (in
    /// interactive mode) and then proceed based on the user's response.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant for operations that are
    /// potentially dangerous but not outright prohibited — for example,
    /// writing to a file for the first time. The user's decision is then
    /// converted to [`Allow`](PermissionCheck::Allow) or
    /// [`Deny`](PermissionCheck::Deny).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::ask("Allow write to /etc/config.yaml?");
    /// assert!(check.is_ask());
    /// ```
    pub fn ask(prompt: impl Into<String>) -> Self {
        Self::Ask {
            prompt: prompt.into(),
        }
    }

    /// Create a [`Modify`](PermissionCheck::Modify) result with rewritten input.
    ///
    /// The agent loop should replace the original tool input with the
    /// provided `modified_input` before invoking [`Tool::call`](super::Tool::call).
    /// Useful for sanitising paths, redacting secrets, or injecting default
    /// values.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation is acceptable but the input needs adjustment — for
    /// example, resolving a relative path to an absolute one within
    /// the allowed directory tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use serde_json::json;
    ///
    /// let check = PermissionCheck::modify(json!({"path": "/safe/dir/file.txt"}));
    /// assert!(check.is_modify());
    /// ```
    #[must_use]
    pub fn modify(modified_input: Value) -> Self {
        Self::Modify { modified_input }
    }

    /// Returns `true` if this is an [`Allow`](PermissionCheck::Allow).
    ///
    /// Convenience predicate for the most common happy-path check.
    /// Used by the agent loop to test whether to proceed with
    /// [`Tool::call`](super::Tool::call) without further processing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_allow() {
    ///     let result = tool.call(input, &ctx).await;
    /// }
    /// ```
    #[must_use]
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Returns `true` if this is a [`Deny`](PermissionCheck::Deny).
    ///
    /// When `true`, the agent loop should *not* invoke the tool and
    /// should instead return a permission error to the LLM. The denial
    /// reason can be extracted by destructuring the variant or by
    /// converting to [`ToolError::Permission`](super::ToolError::Permission).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_deny() {
    ///     return Err(ToolError::Permission("blocked by policy".into()));
    /// }
    /// ```
    #[must_use]
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    /// Returns `true` if this is an [`Ask`](PermissionCheck::Ask).
    ///
    /// When `true`, the agent loop should prompt the user before
    /// deciding whether to allow or deny the invocation. In
    /// non-interactive mode ([`ToolContext::is_non_interactive`](super::ToolContext::is_non_interactive)),
    /// the loop typically treats an [`Ask`](PermissionCheck::Ask) as a
    /// [`Deny`](PermissionCheck::Deny).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_ask() {
    ///     println!("Tool requests approval: {}", prompt);
    /// }
    /// ```
    #[must_use]
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    /// Returns `true` if this is a [`Modify`](PermissionCheck::Modify).
    ///
    /// When `true`, the agent loop should replace the original input
    /// with the modified version before calling the tool. The modified
    /// input can be extracted by matching the variant.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let PermissionCheck::Modify { modified_input } = check {
    ///     let result = tool.call(modified_input, &ctx).await;
    /// }
    /// ```
    #[must_use]
    pub fn is_modify(&self) -> bool {
        matches!(self, Self::Modify { .. })
    }
}
