//! Reference memory implementation — in-memory [`LoopMemory`] backend.
//!
//! [`InMemoryStore`], a simple `Vec`-backed implementation of the
//! [`LoopMemory`] trait. Intended for testing, prototyping, and as a
//! reference for building more sophisticated memory backends (e.g. vector similarity stores).
//!
//! # Provided Implementations
//!
//! - **[`InMemoryStore`]** — Stores [`MemoryEntry`] values in a `Vec` and
//!   retrieves them via weighted keyword + tag scoring. Supports
//!   [`consolidate`](LoopMemory::consolidate) by pruning entries whose
//!   [`relevance`](MemoryEntry::relevance) drops below 0.05.
//!
//! # When to Use
//!
//! Use this backend when you need a zero-dependency, deterministic memory
//! store — for example in unit tests, benchmarks, or single-session agents
//! that don't require persistence across restarts. For production agents
//! that need durable or distributed memory, implement [`LoopMemory`] on
//! top of a database or vector store instead.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::memory::builtin::InMemoryStore;
//! use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let mut store = InMemoryStore::new();
//!
//! store.store(
//!     MemoryEntry::new(MemoryCategory::Insight, "Prefer Glob over manual file search")
//! ).await.unwrap();
//!
//! let results = store.retrieve("file search", 5).await.unwrap();
//! assert_eq!(results.len(), 1);
//! # });
//! ```

use crate::error::LoopError;
use crate::memory::{ConsolidationStats, LoopMemory, MemoryEntry};
use std::future::Future;
use std::pin::Pin;

/// A simple in-memory store for loop memory entries.
///
/// Stores [`MemoryEntry`] values in a flat `Vec` and retrieves them using
/// a weighted scoring function that combines the entry's base
/// [`relevance`](MemoryEntry::relevance), word-overlap with the query,
/// and tag matching. This scoring strategy provides reasonable results
/// without requiring an embedding model.
///
/// **Not suitable for production** — entries are held in process memory
/// and lost on crash. Use this for unit tests, integration tests, and
/// as a reference when implementing a real backend (e.g. one backed by
/// a vector database).
///
/// # Scoring Formula
///
/// Each candidate entry is scored during [`retrieve`](LoopMemory::retrieve)
/// using a weighted blend of three signals:
///
/// ```text
/// final_score = relevance × 0.5
///             + word_overlap_ratio × 0.4
///             + tag_bonus (0.3 if any tag matches)
///             + 0.1  (baseline)
/// ```
///
/// The baseline term ensures that every entry has a non-zero score so
/// that even entries with no word overlap can still be returned when the
/// store is sparse.
///
/// # Thread Safety
///
/// [`InMemoryStore`] is `Send + Sync` because all mutation goes through
/// `&mut self` in the [`LoopMemory`] trait. If you need shared mutable
/// access from multiple tasks, wrap it in `Arc<Mutex<_>>`.
///
/// # Construction
///
/// ```
/// use loopctl::memory::builtin::InMemoryStore;
/// use loopctl::memory::{MemoryEntry, MemoryCategory};
///
/// // Empty store:
/// let store = InMemoryStore::new();
///
/// // Pre-populated:
/// let store = InMemoryStore::new().with_entries(vec![
///     MemoryEntry::new(MemoryCategory::Fact, "The project uses Rust 1.95"),
/// ]);
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::memory::builtin::InMemoryStore;
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let mut store = InMemoryStore::new();
///
/// store.store(MemoryEntry::new(MemoryCategory::Insight, "Prefer Glob over manual file search")).await.unwrap();
///
/// let results = store.retrieve("file search", 5).await.unwrap();
/// assert_eq!(results.len(), 1);
/// # });
/// ```
///
/// # Unbounded Growth
///
/// `InMemoryStore` accumulates entries in a `Vec` with no automatic
/// eviction. The [`consolidate()`](InMemoryStore::consolidate) method
/// prunes entries with `relevance < 0.05`, but it must be called
/// explicitly. A long-running session that never calls `consolidate()`
/// will accumulate memory indefinitely. For production use, consider
/// calling `consolidate()` periodically or implementing a custom
/// [`LoopMemory`] with bounded capacity.
pub struct InMemoryStore {
    entries: Vec<MemoryEntry>,
}

// ===================================================
// Construction
// ===================================================

impl InMemoryStore {
    /// Create a new empty store.
    ///
    /// Returns a fresh [`InMemoryStore`] whose [`len`](LoopMemory::len) is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::memory::builtin::InMemoryStore;
    /// use loopctl::memory::LoopMemory;
    ///
    /// let store = InMemoryStore::new();
    /// assert!(store.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a store pre-populated with the given entries.
    ///
    /// Useful for setting up test fixtures or seeding an agent with
    /// initial context.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::memory::builtin::InMemoryStore;
    /// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
    ///
    /// let store = InMemoryStore::new().with_entries(vec![
    ///     MemoryEntry::new(MemoryCategory::Fact, "Rust 1.75 stabilised async fn in trait"),
    ///     MemoryEntry::new(MemoryCategory::Strategy, "Start refactors with tests"),
    /// ]);
    /// assert_eq!(store.len(), 2);
    /// ```
    #[must_use]
    pub fn with_entries(mut self, entries: Vec<MemoryEntry>) -> Self {
        self.entries = entries;
        self
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================
// LoopMemory implementation
// ===================================================

impl LoopMemory for InMemoryStore {
    /// Store a new memory entry by appending it to the backing list.
    ///
    /// Called whenever the agent encounters information worth remembering —
    /// for example after a successful tool invocation, a resolved error, or
    /// an insight drawn from conversation.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error.
    fn store(
        &mut self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            self.entries.push(entry);
            Ok(())
        })
    }

    /// Retrieve memory entries relevant to the given query.
    ///
    /// Called before each turn (or on demand) to surface context the agent
    /// can use. Returns up to `limit` entries ordered by a composite score
    /// that blends:
    ///
    /// - **Base relevance** (50%) — the entry's [`relevance`](MemoryEntry::relevance) field.
    /// - **Word overlap** (40%) — fraction of query words found in the entry memory.
    /// - **Tag match** (30% flat bonus) — whether any tag contains the full query.
    /// - **Baseline** (10%) — ensures every entry has a non-zero score.
    ///
    /// The query is matched case-insensitively against both the entry
    /// [`memory`](MemoryEntry::memory) and [`tags`](MemoryEntry::tags).
    ///
    /// # Returns
    ///
    /// A `Vec<MemoryEntry>` of at most `limit` entries, sorted by descending
    /// composite score. May be empty if no entries match or the store is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::memory::builtin::InMemoryStore;
    /// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let mut store = InMemoryStore::new();
    /// store.store(MemoryEntry::new(MemoryCategory::Fact, "file search uses Glob")).await.unwrap();
    ///
    /// let results = store.retrieve("file search", 5).await.unwrap();
    /// for entry in &results {
    ///     println!("{:?}", entry.category);
    /// }
    /// # });
    /// ```
    fn retrieve(
        &self,
        query: &str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + '_>> {
        let query = query.to_string();
        Box::pin(async move {
            let query_lower = query.to_lowercase();
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();

            let mut scored: Vec<(f32, MemoryEntry)> = self
                .entries
                .iter()
                .map(|entry| {
                    let memory_lower = entry.memory.to_lowercase();
                    let tag_match = entry
                        .tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower));
                    let word_matches = query_words
                        .iter()
                        .filter(|w| memory_lower.contains(*w))
                        .count();
                    let base_score = entry.relevance;
                    #[allow(clippy::cast_precision_loss)]
                    let query_bonus = if word_matches > 0 {
                        word_matches as f32 / query_words.len().max(1) as f32
                    } else {
                        0.0
                    };
                    let tag_bonus = if tag_match { 0.3 } else { 0.0 };
                    (
                        base_score * 0.5 + query_bonus * 0.4 + tag_bonus + 0.1,
                        entry.clone(),
                    )
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            Ok(scored.into_iter().take(limit).map(|(_, e)| e).collect())
        })
    }

    /// Consolidate memory by pruning low-relevance entries.
    ///
    /// Called periodically by the framework to keep the memory store healthy.
    /// This implementation removes entries whose
    /// [`relevance`](MemoryEntry::relevance) score has decayed below 0.05.
    /// It does **not** perform merging — [`merged`](ConsolidationStats::merged)
    /// and [`bytes_saved`](ConsolidationStats::bytes_saved) are always zero.
    ///
    /// # Returns
    ///
    /// A [`ConsolidationStats`] describing the number of entries before and
    /// after pruning, and how many were removed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::memory::builtin::InMemoryStore;
    /// use loopctl::memory::LoopMemory;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let mut store = InMemoryStore::new();
    /// let stats = store.consolidate().await.unwrap();
    /// println!("Pruned {} entries", stats.pruned);
    /// # });
    /// ```
    fn consolidate(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let entries_before = self.entries.len();
            self.entries.retain(|e| e.relevance >= 0.05);
            let pruned = entries_before.saturating_sub(self.entries.len());
            Ok(ConsolidationStats {
                entries_before,
                entries_after: self.entries.len(),
                pruned,
                merged: 0,
                bytes_saved: 0,
            })
        })
    }

    /// Number of entries currently stored.
    ///
    /// Used by the framework to monitor memory usage and by the
    /// [`is_empty`](LoopMemory::is_empty) provided method.
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryCategory;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let mut store = InMemoryStore::new();

        store
            .store(MemoryEntry::new(
                MemoryCategory::Insight,
                "Prefer Glob over manual file search",
            ))
            .await
            .unwrap();

        store
            .store(MemoryEntry::new(
                MemoryCategory::ErrorPattern,
                "Edit failures often caused by stale file content",
            ))
            .await
            .unwrap();

        let results = store.retrieve("Glob manual file search", 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].category, MemoryCategory::Insight);
    }

    #[tokio::test]
    async fn test_retrieve_respects_limit() {
        let mut store = InMemoryStore::new();

        for i in 0..10 {
            store
                .store(MemoryEntry::new(
                    MemoryCategory::Fact,
                    format!("Fact number {i} about testing"),
                ))
                .await
                .unwrap();
        }

        let results = store.retrieve("testing", 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_retrieve_empty_store() {
        let store = InMemoryStore::new();
        let results = store.retrieve("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_len_and_is_empty() {
        let store = InMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn test_consolidate_prunes_low_relevance() {
        let mut store = InMemoryStore::new();

        let mut good_entry = MemoryEntry::new(MemoryCategory::Insight, "useful insight");
        good_entry.relevance = 0.9;
        store.store(good_entry).await.unwrap();

        let mut bad_entry = MemoryEntry::new(MemoryCategory::Working, "temporary data");
        bad_entry.relevance = 0.01;
        store.store(bad_entry).await.unwrap();

        assert_eq!(store.len(), 2);

        let stats = store.consolidate().await.unwrap();

        assert_eq!(stats.entries_before, 2);
        assert_eq!(stats.pruned, 1);
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn test_with_entries() {
        let entries = vec![
            MemoryEntry::new(MemoryCategory::Fact, "fact 1"),
            MemoryEntry::new(MemoryCategory::Fact, "fact 2"),
        ];
        let store = InMemoryStore::new().with_entries(entries);
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn test_tag_matching_boosts_relevance() {
        let mut store = InMemoryStore::new();

        let tagged =
            MemoryEntry::new(MemoryCategory::Strategy, "use iterators for loops").with_tag("rust");
        store.store(tagged).await.unwrap();

        store
            .store(MemoryEntry::new(
                MemoryCategory::Strategy,
                "use caching for performance",
            ))
            .await
            .unwrap();

        let results = store.retrieve("rust iterators", 2).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].memory.contains("iterators"));
    }

    #[tokio::test]
    async fn test_default_is_empty() {
        let store = InMemoryStore::default();
        assert!(store.is_empty());
    }
}
