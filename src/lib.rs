//! A trait-based framework for building agent loops with pluggable LLM
//! clients, tools, and memory.
//!
//! # Module Overview
//!
//! ## Foundational Types
//!
//! - **[`message`]** — Core conversation types: messages, parts, tool results.
//! - **[`error`]** — Central error enum ([`LoopError`](error::LoopError)) for all framework operations.
//! - **[`config`]** — Session configuration ([`SessionConfig`](config::SessionConfig)).
//! - **[`cancel`]** — Cooperative cancellation signal (`CancelSignal`).
//!
//! ## Subsystems
//!
//! - **[`observer`]** — Lifecycle event observation ([`LoopObserver`](observer::LoopObserver), [`ObserverHost`](observer::ObserverHost)).
//! - **[`memory`]** — Agent memory trait ([`LoopMemory`](memory::LoopMemory)) and entry types.
//! - **[`reflection`]** — Failure reflection and recovery strategies.
//! - **[`detection`]** — Loop and convergence detection ([`DetectionManager`](detection::DetectionManager)).
//! - **[`fallback`]** — Circuit breaker pattern for automatic API model fallback.
//! - **[`compact`]** — Context management and compaction (threshold detection, pluggable strategies).
//! - **[`stream`]** — Streaming event types for LLM API responses.
//! - **[`middleware`]** — Tool dispatch middleware pipeline (timeouts, permissions, output limits).
//! - **[`tool`]** — Tool trait, registry, and supporting types.
//! - **`tool::health`** — Per-tool health monitoring, circuit breakers, and self-healing routing. *Requires `tool_health` feature.*
//! - **[`mcp`]** — MCP client + server adapters ([`McpToolProvider`](mcp::McpToolProvider) adapts foreign MCP servers; [`McpServerAdapter`](mcp::McpServerAdapter) serves a `ToolRegistry` over MCP). *Requires `mcp` feature.*
//!
//! ## API Layer
//!
//! - **[`api`]** — LLM API client trait ([`ApiClient`](api::ApiClient)) and error types.
//!
//! ## Runtime & Capabilities
//!
//! - **[`capabilities`]** — Capability traits ([`Observable`](capabilities::Observable), [`Detectable`](capabilities::Detectable), etc.).
//! - **[`managers`]** — [`LoopManagers`](managers::LoopManagers) — the default infrastructure bundle.
//!
//! ## Engine
//!
//! - **[`engine`]** — The core agentic loop ([`BareLoop`](engine::BareLoop)) that orchestrates the full agent lifecycle.
//!
//! ## Support
//!
//! - **[`memory::builtin`]** — Reference [`InMemoryStore`](memory::builtin::InMemoryStore) implementation.
//! - **hooks** — Bidirectional lifecycle control (allow/block/ask before tool use, compaction). *Requires `hooks` feature.*
//! - **testing** — Test utilities and fixtures. *Requires `testing` feature.*

// Relax strict lints in test code. The crate enforces a strict no-panic /
// no-unwrap policy in production code, but test code legitimately uses
// assertions, unwrap, indexing, etc. for readability.
#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::missing_errors_doc,
        clippy::unnecessary_wraps,
        clippy::clone_on_ref_ptr,
        clippy::doc_markdown,
        clippy::field_reassign_with_default,
        clippy::used_underscore_items,
        clippy::wildcard_imports,
    )
)]

pub use tool::Tool;

/// Re-export for `#[derive(Tool)]`-generated code, so downstream
/// crates do not need a direct `serde_json` dependency.
///
/// Not part of the stable API surface; used only by the code the
/// derive macro emits.
#[cfg(feature = "derive")]
#[doc(hidden)]
pub mod __private {
    pub use serde_json;
}

#[cfg(feature = "derive")]
pub use loopctl_derive::Tool;

pub mod api;
pub mod cancel;
pub mod capabilities;
pub mod compact;
pub mod config;
pub mod contributor;
pub mod detection;
pub mod engine;
pub mod error;
pub mod fallback;
#[cfg(feature = "hooks")]
pub mod hooks;
pub mod managers;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod message;
pub mod middleware;
pub(crate) mod numeric;
pub mod observer;
pub mod presets;
#[cfg(feature = "providers")]
pub mod provider;
pub mod reflection;
pub mod stream;
pub mod structured;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tool;
