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
//! ```
//! use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
//! use loopctl::error::LoopError;
//! use std::future::Future;
//! use std::pin::Pin;
//!
//! struct InMemoryStore {
//!     entries: Vec<MemoryEntry>,
//! }
//!
//! impl LoopMemory for InMemoryStore {
//!     fn store(&mut self, entry: MemoryEntry)
//!         -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>
//!     {
//!         Box::pin(async move { self.entries.push(entry); Ok(()) })
//!     }
//!     fn retrieve(&self, query: &str, limit: usize)
//!         -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>>
//!     {
//!         let query = query.to_string();
//!         Box::pin(async move {
//!             Ok(self.entries.iter()
//!                 .filter(|e| e.memory.contains(&query))
//!                 .take(limit)
//!                 .cloned()
//!                 .collect())
//!         })
//!     }
//!     fn consolidate(&mut self)
//!         -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>
//!     {
//!         Box::pin(async move { Ok(ConsolidationStats::default()) })
//!     }
//!     fn len(&self) -> usize {
//!         self.entries.len()
//!     }
//! }
//! ```

use crate::error::LoopError;
use std::future::Future;
use std::pin::Pin;

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
/// ```
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
/// use loopctl::error::LoopError;
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct InMemoryStore {
///     entries: Vec<MemoryEntry>,
/// }
///
/// impl LoopMemory for InMemoryStore {
///     fn store(&mut self, entry: MemoryEntry)
///         -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>
///     {
///         Box::pin(async move { self.entries.push(entry); Ok(()) })
///     }
///     fn retrieve(&self, query: &str, limit: usize)
///         -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>>
///     {
///         let query = query.to_string();
///         Box::pin(async move {
///             Ok(self.entries.iter()
///                 .filter(|e| e.memory.contains(&query))
///                 .take(limit)
///                 .cloned()
///                 .collect())
///         })
///     }
///     fn consolidate(&mut self)
///         -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>
///     {
///         Box::pin(async move {
///             let before = self.entries.len();
///             self.entries.retain(|e| e.relevance > 0.1);
///             let after = self.entries.len();
///             Ok(ConsolidationStats {
///                 entries_before: before,
///                 entries_after: after,
///                 pruned: before - after,
///                 ..Default::default()
///             })
///         })
///     }
///     fn len(&self) -> usize {
///         self.entries.len()
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
    fn store(
        &mut self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>;

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
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>>;

    /// Consolidate memory (e.g. prune, summarize, compress).
    ///
    /// Called periodically to keep the memory store healthy. Implementations
    /// may remove low-relevance entries, merge duplicates, or produce
    /// compressed summaries. Returns [`ConsolidationStats`] describing what
    /// was done.
    fn consolidate(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>;

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
