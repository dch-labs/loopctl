//! A trait-based framework for building agent loops with pluggable LLM
//! clients, tools, and memory.
//!
//! # Module Overview
//!
//! - **[`message`]** — Core conversation types: messages, parts, tool results.
//! - **[`api_client`]** — Trait for LLM provider communication.
//! - **[`api_error`]** — API and infrastructure error types with classification.
//! - **[`builder`]** — Fluent builder API for constructing configured agents.
//! - **[`cancel`]** — Cooperative cancellation signal (`CancelSignal`).
//! - **[`compact`]** — Context management and compaction (threshold detection, pluggable strategies).
//! - **[`core`]** — Foundational traits (`AgentObserver`) and error types.
//! - **[`builtin`]** — Reference implementations of core traits ([`builtin::memory::InMemoryStore`], [`builtin::observer::LoggingObserver`], etc.).
//! - **[`loop_control`]** — Detection and intervention modules for agent loops.
//! - **[`engine`]** — The core agentic loop that orchestrates the full agent lifecycle.
//! - **[`observability`]** — Structured event streaming (`EventSink`, `ObserveEvent`, metrics).
//! - **[`stream`]** — Streaming event types for LLM API responses.
//! - **[`tool`]** — Tool trait, registry, and supporting types.

pub mod api_client;
pub mod api_error;
pub mod builder;
pub mod builtin;
pub mod cancel;
pub mod compact;
pub mod core;
pub mod engine;
pub mod loop_control;
pub mod message;
pub mod observability;
pub mod stream;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tool;
