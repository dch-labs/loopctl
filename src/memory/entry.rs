//! Memory entry types — [`MemoryEntry`], [`MemoryCategory`], [`ConsolidationStats`].
//!
//! These value types support the [`LoopMemory`](super::LoopMemory) trait.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use uuid::Uuid;

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
/// use loopctl::memory::{MemoryEntry, MemoryCategory};
///
/// let entry = MemoryEntry::new(MemoryCategory::Insight, "Prefer concurrent requests when possible")
///     .with_tag("performance")
///     .validated();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// UUID v4 for deduplication and stable reference during consolidation.
    pub id: Uuid,
    /// Entry category, influencing ranking and consolidation rules.
    pub category: MemoryCategory,
    /// What the agent learned — free-form text.
    pub memory: String,
    /// Arbitrary labels for categorization and retrieval (e.g. `"performance"`, `"security"`).
    pub tags: Vec<String>,
    /// Timestamp set by [`MemoryEntry::new`]. Recency-based strategies use this.
    pub created_at: SystemTime,
    /// Relevance score (0.0–1.0). Starts at 1.0; implementations may decay over time.
    pub relevance: f32,
    /// Number of times this entry has been retrieved.
    pub access_count: usize,
    /// Whether this entry has been validated. Consolidation prefers keeping validated entries.
    pub validated: bool,
}

impl Default for MemoryEntry {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
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
    /// use loopctl::memory::{MemoryEntry, MemoryCategory};
    ///
    /// let entry = MemoryEntry::new(
    ///     MemoryCategory::ErrorPattern,
    ///     "Timeout on external API — retry with exponential back-off",
    /// );
    /// ```
    #[must_use]
    pub fn new(category: MemoryCategory, memory: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
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
    /// use loopctl::memory::{MemoryEntry, MemoryCategory};
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
    /// use loopctl::memory::{MemoryEntry, MemoryCategory};
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
/// Returned by [`LoopMemory::consolidate`](super::LoopMemory::consolidate) so callers can monitor the
/// health of the memory store over time.
///
/// # Example
///
/// ```
/// use loopctl::memory::ConsolidationStats;
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
    pub entries_before: usize,
    /// Number of entries after consolidation.
    pub entries_after: usize,
    /// Entries removed (low relevance, stale, superseded).
    pub pruned: usize,
    /// Entries merged (duplicates combined).
    pub merged: usize,
    /// Estimated storage reclaimed in bytes.
    pub bytes_saved: usize,
}
