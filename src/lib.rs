//! A trait-based framework for building agent loops with pluggable LLM
//! clients, tools, and memory.
//!
//! # Module Overview
//!
//! - **[`message`]** — Core conversation types: messages, parts, tool results.
//! - **[`api_error`]** — API and infrastructure error types with classification.
//! - **[`builder`]** — Fluent builder API for constructing configured agents.
//! - **[`core`]** — Foundational traits (`AgentObserver`) and error types.
//! - **[`loop_control`]** — Detection and intervention modules for agent loops.
//! - **[`stream`]** — Streaming event types for LLM API responses.

pub mod api_error;
pub mod builder;
pub mod core;
pub mod loop_control;
pub mod message;
pub mod stream;
