//! Hook system — bidirectional lifecycle control for agent loops.
//!
//! Hooks differ from observers ([`crate::observer::LoopObserver`]) in two key ways:
//!
//! 1. **Return values matter.** Pre-hooks return [`HookAction`] (or [`CompactResult`])
//!    to control whether an action proceeds.
//! 2. **Short-circuit execution.** The executor stops at the first hook that returns
//!    a blocking result. Safety-critical hooks should be registered first.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::hooks::{HookExecutor, Hook, HookAction};
//! use loopctl::hooks::context::PreToolUseContext;
//! use std::sync::Arc;
//!
//! struct BlockRmRf;
//! impl Hook for BlockRmRf {
//!     fn name(&self) -> &str { "block_rmrf" }
//!     fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
//!         if ctx.tool_name == "rm_rf" {
//!             Some(HookAction::block("rm -rf is not allowed"))
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! let executor = HookExecutor::new()
//!     .with_hook(Arc::new(BlockRmRf));
//!
//! assert_eq!(executor.hook_count(), 1);
//! ```
//!
//! # Module Structure
//!
//! - **[`context`]** — Context types passed to hooks at each lifecycle point.
//! - **[`Hook`]** — The trait every hook implements.
//! - **[`HookExecutor`]** — Manages ordered hook execution with short-circuit semantics.
//! - **[`builtin`]** — Reference hook implementations (logging, blocklist, confirmation).

pub mod builtin;
pub mod context;
pub mod executor;

pub use executor::HookExecutor;

use std::fmt;

use context::{
    CompactResult, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
    RunEndContext, RunStartContext,
};

/// Hook trait for bidirectional lifecycle control.
///
/// Hooks differ from observers ([`crate::observer::LoopObserver`]) in two key ways:
///
/// 1. **Return values matter.** `on_pre_*` methods return `Option<HookAction>` (or
///    `Option<CompactResult>`). Returning `Some(Block{...})` prevents the action
///    from proceeding.
///
/// 2. **Short-circuit execution.** The executor stops at the first hook that returns
///    a non-`None` / non-Allow result. Safety-critical hooks should be registered first.
///
/// # Synchronous Design
///
/// Hook methods are synchronous because hooks should be fast (no I/O, no network
/// calls). A hook that blocks a tool call does so by inspecting the tool name and
/// input — a pattern match, not a database lookup. If a hook needs async work
/// (e.g., calling a policy server), it should do that work eagerly at registration
/// time or maintain a local cache.
///
/// # Default Implementations
///
/// All methods have default no-op implementations returning `None` (allow).
/// Hooks only override the methods they care about.
///
/// # Examples
///
/// ```
/// use loopctl::hooks::{Hook, HookAction};
/// use loopctl::hooks::context::PreToolUseContext;
///
/// struct BlockDangerousTools;
///
/// impl Hook for BlockDangerousTools {
///     fn name(&self) -> &str { "block_dangerous" }
///
///     fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
///         if ctx.tool_name == "RmRF" {
///             Some(HookAction::block("rm -rf is not allowed"))
///         } else {
///             None
///         }
///     }
/// }
///
/// let hook = BlockDangerousTools;
/// assert_eq!(hook.name(), "block_dangerous");
/// ```
pub trait Hook: Send + Sync {
    /// Human-readable name for this hook.
    ///
    /// Used in logging and diagnostics to identify which hook
    /// produced a given result.
    fn name(&self) -> &str;

    /// Called before a tool is executed.
    ///
    /// Return `Some(HookAction::Block{...})` to prevent execution.
    /// Return `Some(HookAction::Ask{...})` to request interactive confirmation.
    /// Return `None` to allow (default).
    fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
        None
    }

    /// Called before context compaction.
    ///
    /// Return `Some(CompactResult::abort(...))` to prevent compaction.
    /// Return `Some(CompactResult { new_instructions: Some(...), .. })` to
    /// inject custom instructions into the summarizer.
    /// Return `None` to allow (default).
    fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
        None
    }

    /// Called after a tool completes execution.
    ///
    /// Cannot block. Use this for audit logging, metric recording,
    /// or triggering side effects.
    fn on_post_tool_use(&self, _ctx: &PostToolUseContext) {}

    /// Called after context compaction completes.
    ///
    /// Notification-only; cannot alter the compaction result.
    fn on_post_compact(&self, _ctx: &PostCompactContext) {}

    /// Called when a run starts.
    ///
    /// Fires at the start of each `run()` call. Use for initialization,
    /// credential setup, or state restoration.
    fn on_run_start(&self, _ctx: &RunStartContext) {}

    /// Called when a run ends.
    ///
    /// Fires at the end of each `run()` call. Use for cleanup, teardown,
    /// or final metric emission.
    fn on_run_end(&self, _ctx: &RunEndContext) {}
}

/// Whether the session can interact with a human operator.
///
/// This controls how [`HookAction::Ask`] is handled by the
/// [`HookExecutor`]:
///
/// - [`Interactivity::Headless`] — there is no human in the loop, so
///   `Ask` is automatically downgraded to `Block` with a descriptive
///   reason. Default and correct mode for autonomous
///   / headless agents (e.g. `BareLoop`).
/// - [`Interactivity::Interactive`] — a human is available to respond
///   to prompts, so `Ask` passes through unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Interactivity {
    /// No human in the loop — `Ask` is downgraded to `Block`.
    #[default]
    Headless,
    /// Human is available — `Ask` passes through unchanged.
    Interactive,
}

/// Action returned by a pre-hook to control flow.
///
/// Pre-hooks return `Option<HookAction>`:
/// - `None` → allow (default)
/// - `Some(HookAction::Allow)` → explicitly allow
/// - `Some(HookAction::Block { reason })` → block with reason
/// - `Some(HookAction::Ask { message })` → request interactive confirmation
///
/// The executor short-circuits on the first non-`None` result that isn't `Allow`.
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Allow the action to proceed.
    ///
    /// No intervention — the executor continues to the next hook
    /// or proceeds with the original action.
    Allow,

    /// Block the action with a human-readable reason.
    ///
    /// The reason is returned to the model as a tool error, allowing it
    /// to adjust its approach.
    Block {
        /// Why the action was blocked.
        ///
        /// Returned to the model as a tool error, allowing it
        /// to adjust its approach.
        reason: String,
    },

    /// Request interactive confirmation before proceeding.
    ///
    /// The executor delegates to a [`builtin::ConfirmationHandler`] to
    /// resolve the confirmation. If denied, the action is treated as
    /// blocked.
    Ask {
        /// Message to present to the user for confirmation.
        ///
        /// The executor delegates to a [`builtin::ConfirmationHandler`]
        /// to resolve the confirmation.
        message: String,
    },
}

impl fmt::Display for HookAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => write!(f, "Allow"),
            Self::Block { reason } => write!(f, "Block({reason})"),
            Self::Ask { message } => write!(f, "Ask({message})"),
        }
    }
}

impl HookAction {
    /// Convenience constructor for [`HookAction::Block`].
    ///
    /// Accepts any `Into<String>` so string literals work directly.
    /// The carried `reason` is returned to the model as a tool error,
    /// giving it a chance to adjust its approach. Prefer this over
    /// struct-literal construction at call sites that block.
    pub fn block(reason: impl Into<String>) -> Self {
        Self::Block {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`HookAction::Ask`].
    ///
    /// Accepts any `Into<String>` so string literals work directly.
    /// The carried `message` is presented to the user for confirmation;
    /// the executor delegates to a
    /// [`ConfirmationHandler`](crate::hooks::builtin::ConfirmationHandler)
    /// to resolve it. In headless mode `Ask` is downgraded to `Block`
    /// by the [`HookExecutor`].
    pub fn ask(message: impl Into<String>) -> Self {
        Self::Ask {
            message: message.into(),
        }
    }

    /// Returns `true` if this action allows the operation to proceed.
    ///
    /// `true` only for [`HookAction::Allow`]. Use this in match arms
    /// where the default path is to proceed and only the block/ask
    /// branches need explicit handling.
    #[must_use]
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Returns `true` if this action blocks the operation.
    ///
    /// `true` only for [`HookAction::Block`]. Note that in headless
    /// mode an [`HookAction::Ask`] is downgraded to `Block`, so a
    /// post-executor action that reports `is_block` may have originated
    /// as an `Ask`.
    #[must_use]
    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    /// Returns `true` if this action requests interactive confirmation.
    ///
    /// `true` only for [`HookAction::Ask`]. Note this reflects the
    /// action as produced by the hook, before the executor applies its
    /// interactivity policy — an `Ask` returned by a hook may surface
    /// as `Block` after the executor downgrades it in headless mode.
    #[must_use]
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    /// Returns the block reason, if this is a [`HookAction::Block`].
    ///
    /// `Some(reason)` when this action is a `Block`, `None` for `Allow`
    /// and `Ask`. Useful for logging why an action was refused without
    /// a full match arm — pair with [`is_block`](Self::is_block) when
    /// you need to distinguish the variants.
    #[must_use]
    pub fn block_reason(&self) -> Option<&str> {
        match self {
            Self::Block { reason } => Some(reason),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_action_allow_is_allow() {
        assert!(HookAction::Allow.is_allow());
        assert!(!HookAction::Allow.is_block());
    }

    #[test]
    fn hook_action_block_factory() {
        let action = HookAction::block("dangerous");
        assert!(action.is_block());
        assert!(!action.is_allow());
        match action {
            HookAction::Block { reason } => assert_eq!(reason, "dangerous"),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[test]
    fn hook_action_ask_factory() {
        let action = HookAction::ask("confirm?");
        match action {
            HookAction::Ask { message } => assert_eq!(message, "confirm?"),
            other => panic!("expected Ask, got {other:?}"),
        }
    }

    #[test]
    fn hook_action_display() {
        assert_eq!(HookAction::Allow.to_string(), "Allow");
        assert_eq!(HookAction::block("nope").to_string(), "Block(nope)");
        assert_eq!(HookAction::ask("y/n").to_string(), "Ask(y/n)");
    }

    #[test]
    fn block_reason_returns_none_for_allow() {
        assert_eq!(HookAction::Allow.block_reason(), None);
    }

    #[test]
    fn block_reason_returns_none_for_ask() {
        assert_eq!(
            HookAction::Ask {
                message: "ok?".into()
            }
            .block_reason(),
            None
        );
    }

    #[test]
    fn block_reason_returns_some_for_block() {
        assert_eq!(
            HookAction::Block { reason: "x".into() }.block_reason(),
            Some("x")
        );
    }

    #[test]
    fn is_allow_returns_false_for_block() {
        assert!(!HookAction::Block { reason: "x".into() }.is_allow());
    }

    #[test]
    fn is_allow_returns_false_for_ask() {
        assert!(
            !HookAction::Ask {
                message: "x".into()
            }
            .is_allow()
        );
    }

    #[test]
    fn is_block_returns_false_for_allow() {
        assert!(!HookAction::Allow.is_block());
    }

    #[test]
    fn is_block_returns_false_for_ask() {
        assert!(
            !HookAction::Ask {
                message: "x".into()
            }
            .is_block()
        );
    }

    #[test]
    fn is_ask_true_for_ask() {
        let action = HookAction::ask("proceed?");
        assert!(action.is_ask());
    }

    #[test]
    fn is_ask_false_for_block() {
        let action = HookAction::block("nope");
        assert!(!action.is_ask());
    }

    #[test]
    fn is_ask_false_for_allow() {
        assert!(!HookAction::Allow.is_ask());
    }

    #[test]
    fn interactivity_default_is_headless() {
        assert_eq!(Interactivity::default(), Interactivity::Headless);
    }

    #[test]
    fn interactivity_eq_semantics() {
        assert_eq!(Interactivity::Headless, Interactivity::Headless);
        assert_eq!(Interactivity::Interactive, Interactivity::Interactive);
        assert_ne!(Interactivity::Headless, Interactivity::Interactive);
    }
}
