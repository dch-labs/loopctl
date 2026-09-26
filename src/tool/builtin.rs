//! Ready-made tools that are generally useful but never load-bearing.
//!
//! Every tool here is an ordinary [`Tool`](crate::tool::Tool) with no
//! framework coupling:
//! registering one is the only way it enters a session, and the
//! `builtin_tools` feature is the only way it compiles into the build.
//! The module exists so hosts opting into small-model or local-model
//! setups have a curated place to find these tools, not so the framework
//! grows implicit registrations.
//!
//! # Available tools
//!
//! - [`ThinkTool`] — a scratchpad the model reasons into before acting.
//! - [`ReadTool`] — line-aware reading over a pluggable
//!   [`ContentSource`], with loud truncation.
//! - With the `fs_tools` feature: the filesystem family —
//!   [`WriteTool`](fs::WriteTool), [`EditTool`](fs::EditTool),
//!   [`MultiEditTool`](fs::MultiEditTool),
//!   [`FileViewerTool`](fs::FileViewerTool), and the
//!   [`FileSource`](fs::FileSource) filesystem content source — over
//!   one shared [`FileSession`](fs::FileSession).
//!
//! # Example
//!
//! ```
//! use loopctl::tool::ToolRegistry;
//! use loopctl::tool::builtin::ThinkTool;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(ThinkTool::new());
//! assert!(registry.contains("think"));
//! ```

pub mod read;
pub mod think;

#[cfg(feature = "fs_tools")]
pub mod fs;

pub use read::{ContentSource, ReadTool};
pub use think::ThinkTool;
