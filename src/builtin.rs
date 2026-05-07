//! Reference implementations of framework traits.
//!
//! This module provides ready-to-use implementations of the core traits so
//! consumers can get started without writing their own. Each implementation
//! is deliberately simple — suitable for prototyping and testing — and can
//! be replaced with domain-specific versions when needed.
//!
//! # Available Implementations
//!
//! | Implementation      | Trait                                         | Purpose                        |
//! |---------------------|-----------------------------------------------|--------------------------------|
//! | [`InMemoryStore`]   | [`AgentMemory`](crate::core::AgentMemory)     | `Vec`-backed memory store      |
//! | [`NoOpObserver`]    | [`AgentObserver`](crate::core::AgentObserver) | No-op (default) observer       |
//! | [`LoggingObserver`] | [`AgentObserver`](crate::core::AgentObserver) | Logs all events via `tracing`  |
//! | [`MultiObserver`]   | [`AgentObserver`](crate::core::AgentObserver) | Fans out to multiple observers |
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::builtin::{InMemoryStore, LoggingObserver, MultiObserver, NoOpObserver};
//! use loopctl::core::{AgentMemory, MemoryEntry, MemoryCategory};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! // In-memory store for agent memories
//! let mut store = InMemoryStore::new();
//! store.store(MemoryEntry::new(MemoryCategory::Fact, "PostgreSQL 15 is used")).await.unwrap();
//!
//! // Logging observer
//! let observer = LoggingObserver;
//!
//! // Compose multiple observers
//! let multi = MultiObserver::new()
//!     .with(LoggingObserver)
//!     .with(NoOpObserver);
//! assert_eq!(multi.len(), 2);
//! # });
//! ```

pub mod memory;
pub mod observer;

pub use memory::InMemoryStore;
pub use observer::{LoggingObserver, MultiObserver, NoOpObserver};
