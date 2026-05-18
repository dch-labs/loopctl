//! Built-in hook implementations provided by the framework.
//!
//! These hooks are universally useful and have no external dependencies
//! beyond what the framework already uses (e.g., `tracing`).
//!
//! Domain-specific hooks (`SafetyShieldHook`, `PolicyServerHook`)
//! live in agent crates, not here.
//!
//! # Provided Hooks
//!
//! | Hook | Description |
//! |------|-------------|
//! | [`LoggingHook`] | Logs pre/post tool use and compact events via `tracing`. |
//! | [`BlocklistHook`] | Blocks or allows tools by name (denylist/allowlist). |
//! | [`ConfirmationHook`] | Requires interactive confirmation for specific tools. |
//! | [`AutoCommitHook`] | Tracks file modifications and auto-commits at session end. |

mod auto_commit;
mod blocklist_hook;
mod confirmation_hook;
mod logging_hook;

pub use auto_commit::{
    AutoCommitConfig, AutoCommitConfigBuilder, AutoCommitHook, AutoCommitResult, CommitMode,
    GitExecutor, GitExecutorError,
};
pub use blocklist_hook::BlocklistHook;
pub use confirmation_hook::{ConfirmationHandler, ConfirmationHook};
pub use logging_hook::LoggingHook;
