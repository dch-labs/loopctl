//! Agent memory trait — interface for agent memory systems.
//!
//! Memory allows agents to learn from past interactions and retrieve
//! relevant context for future tasks. This module defines the core
//! [`AgentMemory`] trait that all memory backends implement, along with
//! the [`MemoryEntry`] value type and supporting enumerations.
//!
//! # Provided Implementations
//!
//! - **`TrajectoryMemory`** — Records tool-execution trajectories and
//!   retrieves relevant past experiences.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::core::agent_memory::{AgentMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
//! use loopctl::core::error::AgentError;
//!
//! struct InMemoryStore {
//!     entries: Vec<MemoryEntry>,
//! }
//!
//! impl AgentMemory for InMemoryStore {
//!     async fn store(&mut self, entry: MemoryEntry) -> Result<(), AgentError> {
//!         self.entries.push(entry);
//!         Ok(())
//!     }
//!     async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, AgentError> {
//!         Ok(self.entries.iter()
//!             .filter(|e| e.memory.contains(query))
//!             .take(limit)
//!             .cloned()
//!             .collect())
//!     }
//!     async fn consolidate(&mut self) -> Result<ConsolidationStats, AgentError> {
//!         Ok(ConsolidationStats::default())
//!     }
//!     fn len(&self) -> usize {
//!         self.entries.len()
//!     }
//! }
//! ```

use crate::core::error::AgentError;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A memory system for agents.
///
/// Implementations can store and retrieve entries using different
/// strategies (vector similarity, keyword matching, recency, etc.).
///
/// # Lifecycle
///
/// ```text
/// store(entry)                    [called whenever the agent learns something]
///   → retrieve(query, limit)      [called before each turn to gather context]
///   → retrieve(query, limit)
///   → ...
///   → consolidate()               [called periodically to prune / compress]
/// ```
///
/// # Implementing
///
/// At a minimum you must provide [`store`](AgentMemory::store),
/// [`retrieve`](AgentMemory::retrieve), [`consolidate`](AgentMemory::consolidate),
/// and [`len`](AgentMemory::len). The trait supplies a default
/// [`is_empty`](AgentMemory::is_empty) implementation that delegates to `len`.
///
/// # Example
///
/// ```
/// use loopctl::core::agent_memory::{AgentMemory, MemoryEntry, MemoryCategory, ConsolidationStats};
/// use loopctl::core::error::AgentError;
///
/// struct InMemoryStore {
///     entries: Vec<MemoryEntry>,
/// }
///
/// impl AgentMemory for InMemoryStore {
///     async fn store(&mut self, entry: MemoryEntry) -> Result<(), AgentError> {
///         self.entries.push(entry);
///         Ok(())
///     }
///     async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, AgentError> {
///         Ok(self.entries.iter()
///             .filter(|e| e.memory.contains(query))
///             .take(limit)
///             .cloned()
///             .collect())
///     }
///     async fn consolidate(&mut self) -> Result<ConsolidationStats, AgentError> {
///         let before = self.entries.len();
///         self.entries.retain(|e| e.relevance > 0.1);
///         let after = self.entries.len();
///         Ok(ConsolidationStats {
///             entries_before: before,
///             entries_after: after,
///             pruned: before - after,
///             ..Default::default()
///         })
///     }
///     fn len(&self) -> usize {
///         self.entries.len()
///     }
/// }
/// ```
#[allow(async_fn_in_trait)]
pub trait AgentMemory: Send + Sync {
    /// Store a new memory entry.
    ///
    /// Called whenever the agent encounters information worth remembering —
    /// for example after a successful tool invocation, a resolved error, or
    /// an insight drawn from conversation. Implementations should persist the
    /// entry in whatever backing store they use.
    async fn store(&mut self, entry: MemoryEntry) -> Result<(), AgentError>;

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
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, AgentError>;

    /// Consolidate memory (e.g. prune, summarize, compress).
    ///
    /// Called periodically to keep the memory store healthy. Implementations
    /// may remove low-relevance entries, merge duplicates, or produce
    /// compressed summaries. Returns [`ConsolidationStats`] describing what
    /// was done.
    async fn consolidate(&mut self) -> Result<ConsolidationStats, AgentError>;

    /// Number of entries currently stored.
    ///
    /// Used by the framework and by [`is_empty`](AgentMemory::is_empty).
    fn len(&self) -> usize;

    /// Whether the memory is empty.
    ///
    /// Defaults to `self.len() == 0`. Override only if you need a cheaper
    /// check than counting all entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A single memory entry.
///
/// Each entry represents one discrete piece of information the agent has
/// learned. Entries carry metadata — category, tags, relevance score,
/// access count, and a validated flag — that implementations can use to
/// rank, filter, and consolidate the store.
///
/// # Construction
///
/// Prefer the builder-style API starting from [`MemoryEntry::new`]:
///
/// ```
/// use loopctl::core::agent_memory::{MemoryEntry, MemoryCategory};
///
/// let entry = MemoryEntry::new(MemoryCategory::Insight, "Prefer concurrent requests when possible")
///     .with_tag("performance")
///     .validated();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier for this entry.
    ///
    /// Typically a UUID v4 string generated at creation time. Used for
    /// deduplication and as a stable reference when merging entries during
    /// consolidation.
    pub id: String,

    /// The category of this memory.
    ///
    /// See [`MemoryCategory`] for the full set of categories. The category
    /// influences how entries are ranked during retrieval and which
    /// consolidation rules apply.
    pub category: MemoryCategory,

    /// What the agent learned — the core payload of this entry.
    ///
    /// Free-form text describing the learned information. Implementations may
    /// apply NLP techniques (embedding, tokenisation) to this field for
    /// similarity search.
    pub memory: String,

    /// Tags for categorization and retrieval.
    ///
    /// Arbitrary strings that act as lightweight indexes. Useful for
    /// broad queries like "all performance-related memories".
    pub tags: Vec<String>,

    /// When this memory was created.
    ///
    /// Set to `SystemTime::now()` by [`MemoryEntry::new`]. Recency-based
    /// retrieval strategies use this field directly.
    pub created_at: SystemTime,

    /// A relevance score (0.0–1.0) for ranking during retrieval.
    ///
    /// Starts at 1.0 for new entries. Implementations may decay this
    /// value over time or boost it when the entry is accessed frequently.
    pub relevance: f32,

    /// How many times this memory has been accessed.
    ///
    /// Implementations should increment this counter each time the entry
    /// is returned from [`retrieve`](AgentMemory::retrieve). Since
    /// [`retrieve`](AgentMemory::retrieve) takes `&self`, implementations
    /// must use interior mutability (e.g. `AtomicUsize` or `Mutex`) to
    /// update this field. A high access count signals that the memory is
    /// broadly useful and may be a candidate for pinning or promotion to
    /// a long-term store.
    pub access_count: usize,

    /// Whether this memory has been validated by successful outcomes.
    ///
    /// Set to `true` via the [`validated`](MemoryEntry::validated) builder
    /// method or manually. Consolidation algorithms may treat validated
    /// entries as higher-confidence and prefer to keep them when pruning.
    pub validated: bool,
}

impl Default for MemoryEntry {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            category: MemoryCategory::Working,
            memory: String::new(),
            tags: Vec::new(),
            created_at: SystemTime::now(),
            relevance: 0.5,
            access_count: 0,
            validated: false,
        }
    }
}

impl MemoryEntry {
    /// Create a new memory entry with a fresh UUID and the current time.
    ///
    /// The entry starts with `relevance = 1.0`, `access_count = 0`, and
    /// `validated = false`. Use the builder methods ([`with_tag`], [`validated`])
    /// to customise further.
    ///
    /// [`with_tag`]: MemoryEntry::with_tag
    /// [`validated`]: MemoryEntry::validated
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::agent_memory::{MemoryEntry, MemoryCategory};
    ///
    /// let entry = MemoryEntry::new(
    ///     MemoryCategory::ErrorPattern,
    ///     "Timeout on external API — retry with exponential back-off",
    /// );
    /// ```
    #[must_use]
    pub fn new(category: MemoryCategory, memory: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            category,
            memory: memory.into(),
            tags: Vec::new(),
            created_at: SystemTime::now(),
            relevance: 1.0,
            access_count: 0,
            validated: false,
        }
    }

    /// Add a tag to this entry (builder style).
    ///
    /// Tags are lightweight, human-readable labels that speed up broad
    /// queries. Call chain-style:
    ///
    /// ```
    /// use loopctl::core::agent_memory::{MemoryEntry, MemoryCategory};
    ///
    /// let entry = MemoryEntry::new(MemoryCategory::Fact, "Rust 1.75 stabilised async fn in trait")
    ///     .with_tag("rust")
    ///     .with_tag("async");
    /// ```
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Mark this entry as validated (builder style).
    ///
    /// Validated entries are treated as higher-confidence by consolidation
    /// algorithms and are less likely to be pruned.
    ///
    /// ```
    /// use loopctl::core::agent_memory::{MemoryEntry, MemoryCategory};
    ///
    /// let entry = MemoryEntry::new(MemoryCategory::Strategy, "Use parallel tool calls when independent")
    ///     .validated();
    /// ```
    #[must_use]
    pub fn validated(mut self) -> Self {
        self.validated = true;
        self
    }
}

/// Category of a memory entry.
///
/// Each category represents a distinct *kind* of knowledge. Retrieval
/// strategies may weight categories differently (e.g. preferring
/// [`ErrorPattern`](MemoryCategory::ErrorPattern) when debugging), and
/// consolidation rules may vary by category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    /// A recorded trajectory of tool executions.
    ///
    /// Captures the sequence of tool calls, their inputs, and outcomes for
    /// a particular task. Useful for replaying successful strategies.
    Trajectory,

    /// A pattern or insight learned from experience.
    ///
    /// Generalised knowledge that transcends a single interaction — e.g.
    /// "users prefer concise summaries over verbose explanations".
    Insight,

    /// A pattern of errors and how they were resolved.
    ///
    /// Pairs an observed error signature with the fix that resolved it,
    /// allowing the agent to avoid repeating the same mistake.
    ErrorPattern,

    /// A strategy that was proven effective.
    ///
    /// High-level plans or heuristics that led to good outcomes, such as
    /// "when facing a large refactoring, start with tests".
    Strategy,

    /// A fact or piece of knowledge.
    ///
    /// Static information the agent has learned — e.g. "the project uses
    /// `PostgreSQL` 15". Facts are not derived from the agent's own reasoning
    /// but are still valuable context.
    Fact,

    /// Short-term working memory for the current session.
    ///
    /// Ephemeral entries that are typically discarded at the end of a
    /// session. Useful for tracking intermediate state such as "the user
    /// asked about file X in the previous turn".
    Working,
}

/// Statistics from a memory consolidation pass.
///
/// Returned by [`AgentMemory::consolidate`] so callers can monitor the
/// health of the memory store over time.
///
/// # Example
///
/// ```
/// use loopctl::core::agent_memory::ConsolidationStats;
///
/// let stats = ConsolidationStats {
///     entries_before: 100,
///     entries_after: 80,
///     pruned: 15,
///     merged: 5,
///     ..Default::default()
/// };
/// println!(
///     "Consolidated: {} → {} entries (pruned {}, merged {})",
///     stats.entries_before, stats.entries_after, stats.pruned, stats.merged,
/// );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of entries before consolidation.
    ///
    /// Captured at the start of the [`AgentMemory::consolidate`] call.
    /// Together with [`entries_after`](ConsolidationStats::entries_after)
    /// it gives a quick measure of how aggressively the pass pruned or
    /// merged data: `entries_before - entries_after` equals the net
    /// reduction.
    pub entries_before: usize,

    /// Number of entries after consolidation.
    ///
    /// Captured at the end of the [`AgentMemory::consolidate`] call.
    /// Compare with [`entries_before`](ConsolidationStats::entries_before)
    /// to determine the net reduction: `entries_before - entries_after`.
    /// Note this reflects the count *after* both pruning and merging,
    /// so it already accounts for entries that were merged into existing
    /// ones (which reduce the count by at most one per merge).
    pub entries_after: usize,

    /// Number of entries that were pruned (removed entirely).
    ///
    /// Pruned entries are those the implementation judged no longer
    /// worth keeping — typically because their
    /// [`relevance`](MemoryEntry::relevance) decayed below a threshold,
    /// they are stale, or they have been superseded by newer data.
    /// This count is a subset of the total reduction:
    /// `entries_before - entries_after == pruned + merged` (since each
    /// merge removes at least one entry).
    pub pruned: usize,

    /// Number of entries that were merged into existing ones.
    ///
    /// Merging combines duplicate or near-duplicate entries into a single
    /// representative entry, preserving the highest
    /// [`relevance`](MemoryEntry::relevance) score and unioning the
    /// [`tags`](MemoryEntry::tags). This is preferable to pruning when
    /// the information is still valuable but is redundantly stored.
    /// Each merge typically reduces the entry count by one (the source is
    /// absorbed into the target).
    pub merged: usize,

    /// Approximate bytes saved by consolidation.
    ///
    /// A best-effort estimate of the storage reclaimed, useful for
    /// logging dashboards and capacity planning. Implementations should
    /// sum the serialized size of pruned entries plus the source entries
    /// that were absorbed during merges. The value is approximate because
    /// exact byte accounting depends on the backing store's encoding and
    /// overhead (e.g. index entries, padding).
    pub bytes_saved: usize,
}
