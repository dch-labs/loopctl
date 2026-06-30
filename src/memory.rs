//! Agent memory trait — interface for agent memory systems.
//!
//! Memory allows agents to learn from past interactions and retrieve
//! relevant context for future tasks. Defines the core
//! [`LoopMemory`] trait that all memory backends implement, along with
//! the [`MemoryEntry`] value type and supporting enumerations.
//!
//! # Provided Implementations
//!
//! - **[`InMemoryStore`]** — Records tool-execution trajectories and
//!   retrieves relevant past experiences.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
//! use loopctl::error::LoopError;
//! use std::future::Future;
//! use std::pin::Pin;
//! use std::sync::RwLock;
//!
//! struct MyStore {
//!     entries: RwLock<Vec<MemoryEntry>>,
//! }
//!
//! impl LoopMemory for MyStore {
//!     fn store(&self, entry: MemoryEntry)
//!         -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>
//!     {
//!         Box::pin(async move {
//!             self.entries.write().unwrap().push(entry);
//!             Ok(())
//!         })
//!     }
//!     fn retrieve(&self, query: &str, limit: usize)
//!         -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>>
//!     {
//!         let query = query.to_string();
//!         Box::pin(async move {
//!             let entries = self.entries.read().unwrap();
//!             Ok(entries.iter()
//!                 .filter(|e| e.memory.contains(&query))
//!                 .take(limit)
//!                 .cloned()
//!                 .collect())
//!         })
//!     }
//!     fn consolidate(&self)
//!         -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>
//!     {
//!         Box::pin(async move { Ok(ConsolidationStats::default()) })
//!     }
//!     fn len(&self) -> usize {
//!         self.entries.read().unwrap().len()
//!     }
//! }
//! ```

use crate::error::LoopError;
use std::future::Future;

pub use builtin::InMemoryStore;
pub use entry::{ConsolidationStats, MemoryCategory, MemoryEntry};
pub mod builtin;
pub mod entry;

/// A memory system for loops.
///
/// Implementations can store and retrieve entries using different
/// strategies (vector similarity, keyword matching, recency, etc.).
///
/// # Implementing
///
/// At a minimum you must provide [`store`](LoopMemory::store),
/// [`retrieve`](LoopMemory::retrieve), [`consolidate`](LoopMemory::consolidate),
/// and [`len`](LoopMemory::len). The trait supplies a default
/// [`is_empty`](LoopMemory::is_empty) implementation that delegates to `len`.
///
/// # Example
///
/// ```rust
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
/// use loopctl::error::LoopError;
/// use std::sync::RwLock;
///
/// struct MyStore {
///     entries: RwLock<Vec<MemoryEntry>>,
/// }
///
/// impl LoopMemory for MyStore {
///     fn store(&self, entry: MemoryEntry)
///         -> impl Future<Output = Result<(), LoopError>> + Send
///     {
///         async move {
///             self.entries.write().unwrap().push(entry);
///             Ok(())
///         }
///     }
///     fn retrieve(&self, query: &str, limit: usize)
///         -> impl Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send
///     {
///         let query = query.to_string();
///         async move {
///             let entries = self.entries.read().unwrap();
///             Ok(entries.iter()
///                 .filter(|e| e.memory.contains(&query))
///                 .take(limit)
///                 .cloned()
///                 .collect())
///         }
///     }
///     fn consolidate(&self)
///         -> impl Future<Output = Result<ConsolidationStats, LoopError>> + Send
///     {
///         async move {
///             let mut entries = self.entries.write().unwrap();
///             let before = entries.len();
///             entries.retain(|e| e.relevance > 0.1);
///             let after = entries.len();
///             Ok(ConsolidationStats {
///                 entries_before: before,
///                 entries_after: after,
///                 pruned: before - after,
///                 ..Default::default()
///             })
///         }
///     }
///     fn len(&self) -> usize {
///         self.entries.read().unwrap().len()
///     }
/// }
/// ```
pub trait LoopMemory: Send + Sync {
    /// Store a new memory entry.
    ///
    /// Called whenever the agent encounters information worth remembering —
    /// for example after a successful tool invocation, a resolved error, or
    /// an insight drawn from conversation. Implementations should persist the
    /// entry in whatever backing store they use.
    ///
    /// Takes `&self` so that memory stores can be shared via `Arc<impl LoopMemory>`.
    /// Implementations that need interior mutability (e.g. an in-memory `Vec`)
    /// should use `Mutex`, `RwLock`, or lock-free structures internally.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> impl Future<Output = Result<(), LoopError>> + Send;

    /// Retrieve memory entries relevant to the given query.
    ///
    /// Called before each turn (or on demand) to surface context the agent
    /// can use. Returns up to `limit` entries ordered by relevance. The
    /// definition of "relevance" is left to the implementation — common
    /// strategies include vector embedding similarity, keyword overlap,
    /// recency weighting, or a hybrid approach.
    ///
    /// Implementations that track [`MemoryEntry::access_count`] must use
    /// interior mutability (e.g. `AtomicUsize`, `Mutex`) since this method
    /// takes `&self`.
    fn retrieve(
        &self,
        query: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send;

    /// Consolidate memory (e.g. prune, summarize, compress).
    ///
    /// Called periodically to keep the memory store healthy. Implementations
    /// may remove low-relevance entries, merge duplicates, or produce
    /// compressed summaries. Returns [`ConsolidationStats`] describing what
    /// was done.
    ///
    /// Takes `&self` so that memory stores can be shared via `Arc<impl LoopMemory>`.
    /// Implementations should use interior mutability as needed.
    fn consolidate(
        &self,
    ) -> impl Future<Output = Result<ConsolidationStats, LoopError>> + Send;

    /// Number of entries currently stored.
    ///
    /// Used by the framework and by [`is_empty`](LoopMemory::is_empty).
    fn len(&self) -> usize;

    /// Whether the memory is empty.
    ///
    /// Defaults to `self.len() == 0`. Override only if you need a cheaper
    /// check than counting all entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
