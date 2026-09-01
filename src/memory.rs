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
use std::pin::Pin;

pub use builtin::InMemoryStore;
pub use entry::{ConsolidationStats, MemoryCategory, MemoryEntry};
pub use trajectory::{
    TokenSummary, TrajectoryObserver, TrajectoryOutcome, TrajectoryRecord, TrajectoryToolCall,
    TrajectoryTurn,
};

pub mod builtin;
pub mod entry;
pub mod trajectory;

/// A memory system for agent loops.
///
/// The trait the engine uses to persist, recall, and consolidate knowledge
/// across turns and runs. When a memory store is attached via
/// [`BareLoop::set_memory`], the engine:
///
/// - **Stores** a [`MemoryEntry`] after every successful tool call,
///   recording what tool ran, with what input, and what it returned.
/// - **Retrieves** up to a few relevant entries before each turn and
///   injects them as a system message so the model sees prior experience.
/// - **Consolidates** the store at the end of each successful run,
///   pruning low-relevance entries.
///
/// All three hooks are no-ops when no store is attached, so memory is
/// purely opt-in.
///
/// # Object safety
///
/// Async methods return [`Pin<Box<dyn Future>>`] so the trait is object-safe
/// and a store can be held behind `Arc<dyn LoopMemory>` on
/// [`LoopManagers`] — the same convention used by [`Reflector`] and
/// [`RecoveryStrategy`]. Implementations wrap each method body in
/// `Box::pin(async move { ... })`; see [`InMemoryStore`] for a reference.
///
/// [`BareLoop::set_memory`]: crate::engine::BareLoop::set_memory
/// [`LoopManagers`]: crate::managers::LoopManagers
/// [`Reflector`]: crate::reflection::Reflector
/// [`RecoveryStrategy`]: crate::reflection::RecoveryStrategy
///
/// # Example
///
/// ```rust
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
/// use loopctl::error::LoopError;
/// use std::future::Future;
/// use std::pin::Pin;
/// use std::sync::RwLock;
///
/// struct MyStore {
///     entries: RwLock<Vec<MemoryEntry>>,
/// }
///
/// impl LoopMemory for MyStore {
///     fn store(&self, entry: MemoryEntry)
///         -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>
///     {
///         Box::pin(async move {
///             self.entries.write().unwrap().push(entry);
///             Ok(())
///         })
///     }
///     fn retrieve(&self, query: &str, limit: usize)
///         -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>>
///     {
///         let query = query.to_string();
///         Box::pin(async move {
///             let entries = self.entries.read().unwrap();
///             Ok(entries.iter()
///                 .filter(|e| e.memory.contains(&query))
///                 .take(limit)
///                 .cloned()
///                 .collect())
///         })
///     }
///     fn consolidate(&self)
///         -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>
///     {
///         Box::pin(async move {
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
///         })
///     }
///     fn len(&self) -> usize {
///         self.entries.read().unwrap().len()
///     }
/// }
/// ```
pub trait LoopMemory: Send + Sync {
    /// Store a new memory entry.
    ///
    /// Called by the engine after every successful tool call to record the
    /// trajectory — the tool name, its input, and its result. Implementations
    /// should persist the entry in whatever backing store they use.
    ///
    /// Takes `&self` so that memory stores can be shared via
    /// `Arc<dyn LoopMemory>`. Implementations that need interior mutability
    /// (e.g. an in-memory `Vec`) should use `Mutex`, `RwLock`, or lock-free
    /// structures internally.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>;

    /// Retrieve memory entries relevant to the given query.
    ///
    /// Called by the engine before each turn to surface context the agent
    /// can use. Returns up to `limit` entries ordered by relevance. The
    /// definition of "relevance" is left to the implementation — common
    /// strategies include vector embedding similarity, keyword overlap,
    /// recency weighting, or a hybrid approach.
    ///
    /// Implementations that track [`MemoryEntry::access_count`] must use
    /// interior mutability (e.g. `AtomicUsize`, `Mutex`) since this method
    /// takes `&self`.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + 'a>>;

    /// Consolidate memory (prune, summarize, compress).
    ///
    /// Called by the engine at the end of each successful run to keep the
    /// store healthy. Implementations may remove low-relevance entries,
    /// merge duplicates, or produce compressed summaries. Returns
    /// [`ConsolidationStats`] describing what was done.
    ///
    /// Takes `&self` so that memory stores can be shared via
    /// `Arc<dyn LoopMemory>`. Implementations should use interior mutability
    /// as needed.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>>;

    /// Number of entries currently stored.
    ///
    /// Used by [`is_empty`](Self::is_empty) and reported by the engine's
    /// consolidate hook after each run.
    fn len(&self) -> usize;

    /// Whether the memory is empty.
    ///
    /// Defaults to `self.len() == 0`. Override only if you need a cheaper
    /// check than counting all entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
