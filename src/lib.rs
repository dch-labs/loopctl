//! A trait-based framework for building agent loops with pluggable LLM
//! clients, tools, and memory.
//!
//! See the [`message`] module for the core conversation types,
//! the [`core`] module for foundational traits and error types,
//! and the [`api_error`] module for API and infrastructure error types.

pub mod api_error;
pub mod core;
pub mod message;
