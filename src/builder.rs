//! Builder module — fluent API for constructing configured agents.
//!
//! Provides a compile-time-safe builder that wires up all the components
//! an agent needs: `AgentCore`, memory, observers, managers, features,
//! and configuration. The builder uses **type-state generics** so that
//! missing required components are caught at compile time, not at runtime.
//!
//! # Provided Types
//!
//! - **[`BuildError`]** — Errors that can occur during builder validation.

pub mod error;

pub use error::BuildError;
