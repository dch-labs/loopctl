//! Reference implementations of framework traits.
//!
//! This module provides ready-to-use implementations of the core traits so
//! consumers can get started without writing their own. Each implementation
//! is deliberately simple — suitable for prototyping and testing — and can
//! be replaced with domain-specific versions when needed.
//!
//! # Available Implementations
//!
//! | Implementation    | Trait                                         | Purpose                   |
//! |-------------------|-----------------------------------------------|---------------------------|
//! | [`InMemoryStore`] | [`AgentMemory`](crate::core::AgentMemory)     | `Vec`-backed memory store |
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::builtin::InMemoryStore;
//! use loopctl::core::{AgentMemory, MemoryEntry, MemoryCategory};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! // In-memory store for agent memories
//! let mut store = InMemoryStore::new();
//! store.store(MemoryEntry::new(MemoryCategory::Fact, "PostgreSQL 15 is used")).await.unwrap();
//! # });
//! ```

pub mod memory;

pub use memory::InMemoryStore;
