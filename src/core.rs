//! Core module — foundational traits and types for building agents.
//!
//! This module defines the core abstractions that every agent built with
//! the loopctl framework depends on. Implement these traits to plug in
//! your own agent logic, memory backend, or observability layer.
//!
//! # Traits
//!
//! | Trait             | Purpose                                               |
//! |-------------------|-------------------------------------------------------|
// TODO: to be added
//!
//! # Supporting Types
//!
//! | Type              | Purpose                                               |
//! |-------------------|-------------------------------------------------------|
//! | [`AgentError`]    | Unified error type for all framework operations       |

pub mod error;
// TODO: add remaining modules
// pub mod agent_core;
// pub mod agent_memory;
// pub mod agent_observer;
// pub mod types;

pub use error::*;
// TODO: add remaining modules
// pub use agent_core::*;
// pub use agent_memory::*;
// pub use agent_observer::*;
// pub use types::*;
