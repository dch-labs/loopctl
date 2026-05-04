//! Core module — foundational traits and types for building agents.
//!
//! This module defines the core abstractions that every agent built with
//! the loopctl framework depends on. Implement these traits to plug in
//! your own agent logic, memory backend, or observability layer.
//!
//! # Traits
//!
//! | Trait              | Purpose                                               |
//! |--------------------|-------------------------------------------------------|
//! | [`AgentObserver`]  | Lifecycle event hooks for monitoring agents            |
//!
//! # Supporting Types
//!
//! | Type              | Purpose                                               |
//! |-------------------|-------------------------------------------------------|
//! | [`AgentError`]    | Unified error type for all framework operations       |

pub mod agent_observer;
pub mod error;
// TODO: add remaining modules
// pub mod agent_core;
// pub mod agent_memory;
// pub mod types;

pub use agent_observer::*;
pub use error::*;
// TODO: add remaining pub use re-exports as modules are migrated
// pub use agent_core::*;
// pub use agent_memory::*;
// pub use types::*;
