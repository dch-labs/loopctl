//! A trait-based framework for building agent loops with pluggable LLM
//! clients, tools, and memory.
//!
//! # Module Overview
//!
//! ## Foundational Types
//!
//! - **[`message`]** — Core conversation types: messages, parts, tool results.
//! - **[`error`]** — Central error enum ([`LoopError`](error::LoopError)) for all framework operations.
//! - **[`config`]** — Session configuration ([`LoopConfig`](config::LoopConfig)).
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
//! - **[`tool::health`]** — Per-tool health monitoring, circuit breakers, and self-healing routing. *Requires `tool_health` feature.*
//!
//! ## API Layer
//!
//! - **[`api`]** — LLM API client trait ([`ApiClient`](api::ApiClient)) and error types.
//!
//! ## Runtime & Capabilities
//!
//! - **[`capabilities`]** — Capability traits ([`Observable`](capabilities::Observable), [`Detectable`](capabilities::Detectable), etc.).
//! - **[`runtime`]** — [`LoopRuntime`](runtime::LoopRuntime) — the default infrastructure bundle.
//!
//! ## Engine
//!
//! - **[`engine`]** — The core agentic loop ([`BareLoop`](engine::BareLoop)) that orchestrates the full agent lifecycle.
//!
//! ## Support
//!
//! - **[`memory::builtin`]** — Reference [`InMemoryStore`](memory::builtin::InMemoryStore) implementation.
//! - **[`hooks`]** — Bidirectional lifecycle control (allow/block/ask before tool use, compaction). *Requires `hooks` feature.*
//! - **[`testing`]** — Test utilities and fixtures. *Requires `testing` feature.*

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
    )
)]

pub mod api;
pub mod cancel;
pub mod capabilities;
pub mod compact;
pub mod config;
pub mod detection;
pub mod engine;
pub mod error;
pub mod fallback;
#[cfg(feature = "hooks")]
pub mod hooks;
pub mod memory;
pub mod message;
pub mod middleware;
pub mod observer;
#[cfg(feature = "providers")]
pub mod provider;
pub mod reflection;
pub mod runtime;
pub mod stream;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tool;
