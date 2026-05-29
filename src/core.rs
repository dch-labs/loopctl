//! Core module — foundational traits and types for building agents.
//!
//! This module defines the core abstractions that every agent built with
//! the loopctl framework depends on. Implement these traits to plug in
//! your own agent logic, memory backend, or observability layer.
//!
//! # Traits
//!
//! | Trait              | Purpose                                             |
//! |--------------------|-----------------------------------------------------|
//! | [`AgentCore`]      | Main lifecycle trait for all agent types            |
//! | [`AgentMemory`]    | Interface for agent memory backends                 |
//! | [`AgentObserver`]  | Lifecycle event hooks for monitoring agents         |
//!
//! # Supporting Types
//!
//! | Type                           | Purpose                                              |
//! |--------------------------------|------------------------------------------------------|
//! | [`AgentConfig`]                | Configuration for an agent session                   |
//! | [`AgentError`]                 | Unified error type for all framework operations      |
//! | [`AgentState`]                 | Lifecycle state machine for agents                   |
//! | [`Correction`]                 | Correction produced by the reflection system         |
//! | [`CompactReason`]              | Why context compaction was triggered                 |
//! | [`CorrectionResult`]           | Outcome of applying a correction                     |
//! | [`ConsolidationStats`]         | Statistics from a memory consolidation pass          |
//! | [`CorrectionType`]             | Category of fix strategy for a correction            |
//! | [`ExponentialBackoffRecovery`] | Retry strategy with exponential backoff              |   
//! | [`FailureAnalysis`]            | Analysis of a failed tool call                       |
//! | [`FailureSeverity`]            | How severe a failure is                              |
//! | [`MemoryCategory`]             | Category of a memory entry                           |
//! | [`MemoryEntry`]                | A single memory entry with metadata                  |
//! | [`NoopReflector`]              | Default reflector — marks everything non-recoverable |
//! | [`RecoveryAction`]             | What the framework should do after a failure         |
//! | [`ReflectionContext`]          | Retry state provided to the reflector                |
//! | [`StopReason`]                 | Why the API stopped generating                       |
//! | [`SessionResult`]              | Summary of a complete agent session                  |
//! | [`ToolCall`]                   | A tool call requested by the agent                   |
//! | [`ToolDispatchResult`]         | Result of a single tool execution                    |
//! | [`TurnResult`]                 | Result of a single agent turn                        |

pub mod agent_core;
pub mod agent_memory;
pub mod agent_observer;
pub mod error;
pub mod reflection;
pub mod types;

pub use agent_core::*;
pub use agent_memory::*;
pub use agent_observer::*;
pub use error::*;
pub use reflection::*;
pub use types::*;
