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
//! - **[`memory`]** — Agent memory trait ([`LoopMemory`](memory::LoopMemory)), entry types,
//!   trajectory capture, and learned-memory extraction
//!   ([`extractor`](memory::extractor)).
//! - **`memory::vector`** — Vector primitives for semantic retrieval ([`EmbeddingProvider`](memory::vector::EmbeddingProvider), [`VectorIndex`](memory::vector::VectorIndex)). *Requires `vector_index` feature.*
//! - **`memory::vector_memory`** — Semantic `LoopMemory` backend ([`VectorMemoryStore`](memory::vector_memory::VectorMemoryStore)) blending vector, keyword, and tag scores. *Requires `vector_memory` feature.*
//! - **[`reflection`]** — Failure reflection and recovery strategies.
//! - **[`detection`]** — Loop and convergence detection ([`DetectionManager`](detection::DetectionManager)).
//! - **[`fallback`]** — Circuit breaker pattern for automatic API model fallback.
//! - **[`compact`]** — Context management and compaction (threshold detection, pluggable strategies).
//! - **[`stream`]** — Streaming event types for LLM API responses.
//! - **[`middleware`]** — Tool dispatch middleware pipeline (timeouts, permissions, output limits).
//! - **[`tool`]** — Tool trait, registry, and supporting types.
//! - **`tool::health`** — Per-tool health monitoring, circuit breakers, and self-healing routing. *Requires `tool_health` feature.*
//! - **[`mcp`]** — MCP client + server adapters ([`McpToolProvider`](mcp::McpToolProvider) adapts foreign MCP servers; [`McpServerAdapter`](mcp::McpServerAdapter) serves a `ToolRegistry` over MCP). *Requires `mcp` feature.*
//! - **[`manifest`]** — The v1 `loop.yaml` manifest model: strict serde types, profile deep-merge, compose-style interpolation, span-bearing validation, and the JSON Schema export ([`ManifestDocument`](manifest::ManifestDocument) is the entry point). *Requires `manifest` feature.*
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
//!
//! # Quick Start
//!
//! A complete agent loop in one block — this is the fastest path from
//! nothing to a finished run. It uses the `testing` feature's mock
//! client, so it runs offline (the doctest executes it in CI):
//!
//! ```
//! # #[cfg(feature = "testing")]
//! # async fn example() {
//! use std::sync::Arc;
//! use loopctl::config::SessionConfig;
//! use loopctl::engine::core::Loop;
//! use loopctl::engine::BareLoop;
//! use loopctl::testing::MockApiClient;
//! use loopctl::tool::ToolRegistry;
//!
//! // Any `ApiClient` drives the loop; the mock serves canned text.
//! let client = MockApiClient::new("test-model").with_text_response("all done");
//! let registry = ToolRegistry::new();
//! let config = SessionConfig::default();
//!
//! let mut agent = BareLoop::new(
//!     Arc::new(client),
//!     registry,
//!     config,
//! );
//! let run = agent
//!     .run("say something", &loopctl::engine::RunConfig::default())
//!     .await
//!     .expect("the mock always answers");
//! assert_eq!(run.output.as_deref(), Some("all done"));
//! # }
//! # fn main() {
//! # #[cfg(feature = "testing")]
//! # tokio::runtime::Runtime::new().unwrap().block_on(example());
//! # }
//! ```
//!
//! Swap `MockApiClient` for a provider client (`loopctl::provider::openai`
//! and friends, behind their feature flags) and the same loop runs
//! against a live model.
//!
//! ## Support
//!
//! - **[`memory::builtin`]** — Reference [`InMemoryStore`](memory::builtin::InMemoryStore) implementation.
//! - **[`memory::consolidate`]** — Store-agnostic consolidation pass a backend runs under its own write lock
//!   ([`consolidate_entries`](memory::consolidate::consolidate_entries)).
//! - **[`memory::trajectory`]** — [`TrajectoryObserver`](memory::trajectory::TrajectoryObserver): captures each run as a serializable record.
//! - **[`memory::extractor`]** — Turns recorded trajectories into learned memories
//!   ([`extract_into`](memory::extractor::extract_into) for the reliable path,
//!   [`ExtractionObserver`](memory::extractor::ExtractionObserver) for automated best-effort at run end).
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
pub mod ask;
pub mod attachment;
pub mod budget;
pub mod cancel;
pub mod capabilities;
pub mod compact;
pub mod config;
pub mod contributor;
pub mod detection;
pub mod determinism;
pub mod engine;
pub mod error;
pub mod fallback;
#[cfg(feature = "hooks")]
pub mod hooks;
pub mod managers;
#[cfg(feature = "manifest")]
pub mod manifest;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod message;
pub mod middleware;
pub(crate) mod numeric;
pub mod observer;
pub mod presets;
pub mod pricing;
#[cfg(feature = "providers")]
pub mod provider;
pub mod reflection;
pub mod stream;
pub mod structured;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tool;
