//! The semantic [`LoopMemory`] backend — hybrid vector +
//! keyword retrieval.
//!
//! [`VectorMemoryStore`] implements the existing [`LoopMemory`]
//! trait unchanged: no trait change, no new required method. It embeds every
//! entry through an [`EmbeddingProvider`] into a [`VectorIndex`], then
//! retrieves by blending **semantic** similarity, **lexical** word overlap,
//! and a **tag** bonus, so "the agent did something like this before" is
//! findable by meaning while exact wording still wins when it applies.
//! [`consolidate`](LoopMemory::consolidate) prunes low-relevance entries and
//! their vectors and optionally merges near-duplicates, keeping the index in
//! sync with the payload map.
//!
//! One deliberate divergence from the flat stores: a `retrieve` here always
//! returns and stamps up to `limit` entries, whatever the query — the
//! semantic channel ranks nearest neighbours for any input — while
//! [`InMemoryStore`](super::builtin::InMemoryStore) never stamps the
//! baseline entries it hands out, so access-based consolidation on the
//! vector side always has signal to read.
//!
//! The two collaborator slots are object-safe trait objects, so a consumer
//! swaps backends at construction: the bundled
//! [`HashingEmbedder`](super::vector::HashingEmbedder) +
//! [`LinearVectorIndex`](super::vector::LinearVectorIndex) for a
//! dependency-free start, or the `loopctl-hnsw`
//! companion crate's `HnswIndex` for approximate search at scale — both wire
//! in with one `Box::new` each and nothing else changes.
//!
//! # Example
//!
//! ```
//! use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
//! use loopctl::memory::vector_memory::VectorMemoryStore;
//! use loopctl::memory::{LoopMemory, MemoryCategory, MemoryEntry};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let store = VectorMemoryStore::new(
//!     Box::new(HashingEmbedder::new(128)),
//!     Box::new(LinearVectorIndex::new(128)),
//! );
//!
//! store
//!     .store(MemoryEntry::new(
//!         MemoryCategory::Strategy,
//!         "prefer glob over manual file search",
//!     ))
//!     .await
//!     .unwrap();
//!
//! let hits = store.retrieve("how do I list files quickly?", 3).await.unwrap();
//! assert!(!hits.is_empty());
//! # });
//! ```

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;
use std::time::SystemTime;

use uuid::Uuid;

use crate::error::{LoopError, recover_guard};
use crate::memory::entry::PROVIDER_DERIVED_TAG;
use crate::memory::vector::{Embedding, EmbeddingProvider, VectorIndex};
use crate::memory::{ConsolidationStats, LoopMemory, MemoryEntry};
use crate::numeric::unit_ratio;

/// Tuning knobs for [`VectorMemoryStore`].
///
/// Every field has a deliberate default; the `Default` shape is a reasonable
/// starting point for both the deterministic test embedder and real models.
/// The three blend weights are clamped at zero, non-finite values are
/// treated as zero, and the triple is **normalized to sum to 1** at
/// construction, so the blended score always stays in `0.0..=1.0` and
/// re-weighting one channel never silently rescales the others. Zeroing all
/// three falls back to the defaults rather than producing a store that
/// ranks every entry equally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorMemoryConfig {
    /// Weight of the semantic (cosine) channel in the retrieval blend.
    ///
    /// A normalized share in `0.0..=1.0`; the stored value is the clamped,
    /// normalized weight, not the raw input. The default gives the vector
    /// channel the majority share because it is the only channel that can
    /// match paraphrases.
    pub semantic_weight: f32,

    /// Weight of the lexical (word-overlap) channel in the retrieval blend.
    ///
    /// A normalized share in `0.0..=1.0`. The default keeps exact keyword
    /// hits competitive with semantic near-misses, which is what makes the
    /// store robust when the embedder's geometry misses.
    pub lexical_weight: f32,

    /// Weight of the tag bonus in the retrieval blend.
    ///
    /// A normalized share in `0.0..=1.0`, awarded in full when any
    /// lowercased entry tag contains any lowercased query word. The default
    /// keeps tags as a tiebreaker rather than a primary signal.
    pub tag_weight: f32,

    /// How many semantic candidates to fetch per requested result.
    ///
    /// The index is asked for `limit × overfetch` neighbours before the
    /// lexical and tag channels re-rank them, so the semantic pre-filter
    /// never starves a lexically exact entry out of the candidate set.
    /// Values below 1 clamp to 1.
    pub overfetch: usize,

    /// Relevance floor below which
    /// [`consolidate`](LoopMemory::consolidate) prunes an entry.
    ///
    /// A sanitized value in `0.0..=1.0` (non-finite inputs fall back to the
    /// default). Entries flagged [`validated`](MemoryEntry::validated) are
    /// exempt from this floor.
    pub prune_threshold: f32,

    /// Pairwise cosine above which two surviving entries are
    /// near-duplicates.
    ///
    /// A sanitized value in `0.0..=1.0`. The comparison is strict, so the
    /// default `0.95` merges near-identical embeddings while setting `1.0`
    /// disables merging entirely — only mathematically identical vectors
    /// exceed it.
    pub dedup_threshold: f32,

    /// How many nearest survivors each survivor is compared against while
    /// planning near-duplicate merges.
    ///
    /// Consolidation asks the index for this many neighbours per survivor
    /// instead of comparing every pair, so planning cost stays near-linear
    /// in the entry count; the merge set is therefore top-k-bounded — a
    /// duplicate cluster wider than this window merges across repeated
    /// passes rather than in one. Values below 1 clamp to 1.
    pub dedup_candidates: usize,
}

impl Default for VectorMemoryConfig {
    /// Returns the default blend: semantic-majority with active lexical and
    /// tag channels.
    ///
    /// The weights normalize to exactly `0.6` / `0.3` / `0.1`; the threshold
    /// and overfetch values are chosen so the first consolidation of a
    /// fresh store is a no-op and retrieval over-fetches broadly enough for
    /// the re-rank to matter.
    fn default() -> Self {
        Self {
            semantic_weight: 0.6,
            lexical_weight: 0.3,
            tag_weight: 0.1,
            overfetch: 4,
            prune_threshold: 0.1,
            dedup_threshold: 0.95,
            dedup_candidates: 16,
        }
    }
}

impl VectorMemoryConfig {
    /// Clamp, sanitize, and normalize the weights for internal use.
    ///
    /// Non-finite weights become zero and negative weights clamp to zero;
    /// if every weight sanitizes to zero the defaults are restored so the
    /// store can never end up with no active channel. `overfetch` and
    /// `dedup_candidates` clamp to at least 1 and the two thresholds to
    /// `0.0..=1.0` (non-finite thresholds fall back to their defaults).
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.semantic_weight = sanitize_weight(self.semantic_weight);
        self.lexical_weight = sanitize_weight(self.lexical_weight);
        self.tag_weight = sanitize_weight(self.tag_weight);
        let sum = self.semantic_weight + self.lexical_weight + self.tag_weight;
        if sum <= 0.0 {
            let fallback = Self::default();
            self.semantic_weight = fallback.semantic_weight;
            self.lexical_weight = fallback.lexical_weight;
            self.tag_weight = fallback.tag_weight;
        } else {
            self.semantic_weight /= sum;
            self.lexical_weight /= sum;
            self.tag_weight /= sum;
        }
        self.overfetch = self.overfetch.max(1);
        self.dedup_candidates = self.dedup_candidates.max(1);
        self.prune_threshold = sanitize_threshold(self.prune_threshold, 0.1);
        self.dedup_threshold = sanitize_threshold(self.dedup_threshold, 0.95);
        self
    }
}

/// Clamp a blend weight to a finite, non-negative value.
///
/// Non-finite inputs become zero — a poisoned channel goes silent rather
/// than borrowing a share it did not earn — while otherwise negative values
/// clamp to `0.0` and values above 1.0 clamp to 1.0, keeping every channel
/// share inside the range the normalization step expects.
fn sanitize_weight(weight: f32) -> f32 {
    if !weight.is_finite() {
        return 0.0;
    }
    weight.clamp(0.0, 1.0)
}

/// Clamp a threshold to a finite value in `0.0..=1.0`.
///
/// Non-finite inputs fall back to `fallback`, so a hand-poisoned config
/// cannot push consolidation into pruning everything or merging everything.
fn sanitize_threshold(threshold: f32, fallback: f32) -> f32 {
    if !threshold.is_finite() {
        return fallback;
    }
    threshold.clamp(0.0, 1.0)
}

/// Sanitize an entry's relevance the same way [`crate::memory::score`] does,
/// so every store agrees on what a poisoned value is worth.
///
/// Values outside `0.0..=1.0` — including non-finite ones — are worth zero,
/// which makes consolidation prune a hand-poisoned entry rather than
/// promote it.
fn sanitize_relevance(relevance: f32) -> f32 {
    if (0.0..=1.0).contains(&relevance) {
        relevance
    } else {
        0.0
    }
}

/// One near-duplicate fold decided during consolidation: fold `victim`'s
/// topic `tags` into `survivor` once the victim's removal is confirmed.
///
/// The indirection exists because removals are snapshot-bound — a fold
/// only applies when its victim was actually removed, never when the
/// live entry changed after the snapshot (today: a concurrent
/// `retrieve`'s access stamp raced the victim into survival).
struct MergeFold {
    /// The entry that survives the merge and receives the tags.
    ///
    /// consolidate's write-locked apply phase applies the fold, and only
    /// after the victim's removal is confirmed against the live map.
    survivor: Uuid,

    /// The entry folded away; its removal must be confirmed first.
    ///
    /// Confirmation compares the live entry against the consolidation
    /// snapshot, so any post-snapshot change to the live entry (today: a
    /// concurrent `retrieve`'s access stamp) cancels its own fold.
    victim: Uuid,

    /// The victim's topic tags, provenance tag already stripped.
    ///
    /// Stripping happens at decision time — `plan_merges` filters the
    /// `PROVIDER_DERIVED_TAG` constant — because provenance belongs to
    /// the entry that earned it, never to the survivor absorbing it.
    tags: Vec<String>,
}

/// A semantic [`LoopMemory`] store: entries retrieved by
/// blended vector, keyword, and tag scores.
///
/// The store is a drop-in `impl LoopMemory` over the [`super::vector`]
/// primitives — it holds an [`EmbeddingProvider`] and a [`VectorIndex`] as
/// boxed trait objects, embeds on [`store`](LoopMemory::store), and blends
/// semantic similarity with lexical overlap and a tag bonus on
/// [`retrieve`](LoopMemory::retrieve). [`consolidate`](LoopMemory::consolidate)
/// prunes entries whose relevance fell below the configured floor (validated
/// entries are exempt), optionally merges near-duplicates, and removes every
/// evicted entry's vector from the index.
///
/// Besides the index's copy, the store keeps the embedding it computed at
/// store time — one `Vec<f32>` per entry, the same order of memory the index
/// already holds — so consolidation can compare and evict vectors without
/// re-embedding through a possibly-remote provider.
///
/// # Example
///
/// See the [module docs](self) for a construction example; the
/// [`InMemoryStore`](super::builtin::InMemoryStore) contract carries over
/// unchanged — both satisfy [`LoopMemory`] through
/// `Arc<dyn LoopMemory>`.
pub struct VectorMemoryStore {
    /// The embedder every entry and query passes through.
    ///
    /// One embedder for the store's lifetime: entries and queries are only
    /// comparable when they come from the same embedding space, and the
    /// store's dimensionality checks inherit from the index it feeds.
    embedder: Box<dyn EmbeddingProvider>,

    /// The vector index holding every entry's embedding under its id.
    ///
    /// Any [`VectorIndex`] implementation works — the bundled
    /// [`LinearVectorIndex`] for exact search, an approximate index for
    /// scale — because the trait's scoring contract fixes the metric.
    index: Box<dyn VectorIndex>,

    /// The payload map: entry id to the entry itself.
    ///
    /// Guarded by an [`RwLock`] recovered per operation on poison (the
    /// crate's Category-1 `recover_guard` policy for single-operation
    /// data), so a panicked writer never permanently blinds readers.
    entries: RwLock<HashMap<Uuid, MemoryEntry>>,

    /// The store's own copy of each entry's embedding, keyed by entry id.
    ///
    /// Consolidation reads this map to score near-duplicate pairs and to
    /// evict pruned vectors without re-embedding; guarded by the same
    /// poison-recovery policy as the payload map.
    vectors: RwLock<HashMap<Uuid, Embedding>>,

    /// The clamped, normalized configuration the store was built with.
    ///
    /// Constructed through [`VectorMemoryConfig::normalized`], so every
    /// read of these values can assume the invariants documented there.
    config: VectorMemoryConfig,

    /// The store↔consolidation sequencing lock.
    ///
    /// [`store`](LoopMemory::store) embeds first and then holds this lock
    /// across its index insert and map writes, and
    /// [`consolidate`](LoopMemory::consolidate) across its whole pass, so
    /// the two write phases can never interleave: a pass observes either
    /// the full pre-state or the full post-state of any concurrent store,
    /// which is what makes the snapshot-bound removals sound — while no
    /// lock at all spans either side's embedder latency.
    /// [`retrieve`](LoopMemory::retrieve) takes no part in it and stays
    /// lock-free.
    pass_lock: tokio::sync::Mutex<()>,
}

impl VectorMemoryStore {
    /// Create a store with the default [`VectorMemoryConfig`].
    ///
    /// The embedder's [`dim`](EmbeddingProvider::dim) should match the
    /// index's dimensionality — a mismatch surfaces as a
    /// [`LoopError::Memory`] on the first store or retrieve, so align them
    /// at construction.
    #[must_use]
    pub fn new(embedder: Box<dyn EmbeddingProvider>, index: Box<dyn VectorIndex>) -> Self {
        Self::with_config(embedder, index, VectorMemoryConfig::default())
    }

    /// Create a store with an explicit configuration.
    ///
    /// The config is passed through
    /// [`VectorMemoryConfig::normalized`], so the stored copy is always the
    /// clamped, normalized shape regardless of what the caller handed in.
    #[must_use]
    pub fn with_config(
        embedder: Box<dyn EmbeddingProvider>,
        index: Box<dyn VectorIndex>,
        config: VectorMemoryConfig,
    ) -> Self {
        Self {
            embedder,
            index,
            entries: RwLock::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            config: config.normalized(),
            pass_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Blend the three retrieval channels for one candidate entry.
    ///
    /// The semantic term is the index's cosine clamped to `0.0..=1.0`, the
    /// lexical term is the share of query words contained in the entry text
    /// (lowercase substring semantics, matching the flat stores' scorer),
    /// and the tag term is 1.0 when any tag contains any query word. The
    /// normalized weights keep the result in `0.0..=1.0`.
    fn blend(&self, entry: &MemoryEntry, semantic: f32, query_words: &[String]) -> f32 {
        let semantic = semantic.clamp(0.0, 1.0);
        let memory_lower = entry.memory.to_lowercase();
        let matched = query_words
            .iter()
            .filter(|word| memory_lower.contains(word.as_str()))
            .count();
        let overlap = unit_ratio(matched, query_words.len());
        let tag_bonus = f32::from(u8::from(entry.tags.iter().any(|tag| {
            let tag_lower = tag.to_lowercase();
            query_words
                .iter()
                .any(|word| tag_lower.contains(word.as_str()))
        })));
        self.config.semantic_weight * semantic
            + self.config.lexical_weight * overlap
            + self.config.tag_weight * tag_bonus
    }
}

impl LoopMemory for VectorMemoryStore {
    /// Embed and index one entry.
    ///
    /// The embedding is awaited **before the store↔consolidation pass lock
    /// is taken** — no lock at all is held across the embedder call, so
    /// overlapping stores interleave their embeds and a slow or remote
    /// embedder never serializes concurrent stores behind its own latency.
    /// The pass lock then covers only the index insert and the two map
    /// writes, keeping a store's write phase atomic against consolidation
    /// passes — the guarantee the pass's snapshot-bound removals rest on —
    /// while [`retrieve`](LoopMemory::retrieve) stays lock-free throughout.
    /// The vector lands in the index first and the payload maps second,
    /// keeping the index and the store's vector copy in lockstep with the
    /// entry they describe. Embedding and index errors propagate as
    /// [`LoopError`] unchanged — the embedding provider picks the variant
    /// that fits its failure.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let id = entry.id;
            let embedding = self.embedder.embed(&entry.memory).await?;
            let _pass = self.pass_lock.lock().await;
            self.index.add(id, embedding.clone()).await?;
            recover_guard(self.vectors.write()).insert(id, embedding);
            recover_guard(self.entries.write()).insert(id, entry);
            Ok(())
        })
    }

    /// Retrieve up to `limit` entries by blended semantic, lexical, and tag
    /// scores.
    ///
    /// The query is embedded first (no lock held), the index is asked for
    /// `limit × overfetch` semantic neighbours, and each candidate present
    /// in the payload map is re-ranked by the three-channel blend. Results
    /// sort by descending blend with an id tiebreak so equal scores stay
    /// deterministic; every returned entry has its
    /// [`access_count`](MemoryEntry::access_count) incremented and its
    /// [`last_accessed`](MemoryEntry::last_accessed) stamped, and entries
    /// that were not returned are left untouched.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            if limit == 0 {
                return Ok(Vec::new());
            }
            let query_embedding = self.embedder.embed(query).await?;
            let overfetch = limit.saturating_mul(self.config.overfetch);
            let matches = self.index.search(&query_embedding, overfetch).await?;
            let query_words: Vec<String> = query
                .trim()
                .to_lowercase()
                .split_whitespace()
                .map(str::to_owned)
                .collect();
            let mut scored: Vec<(f32, Uuid, MemoryEntry)> = Vec::with_capacity(matches.len());
            {
                let entries = recover_guard(self.entries.read());
                for m in &matches {
                    let Some(entry) = entries.get(&m.id) else {
                        continue;
                    };
                    let blend = self.blend(entry, m.score, &query_words);
                    scored.push((blend, m.id, entry.clone()));
                }
            }
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            let top_score = scored.first().map_or(0.0, |(score, _, _)| *score);
            let selected: Vec<MemoryEntry> = scored
                .into_iter()
                .take(limit)
                .map(|(_, _, entry)| entry)
                .collect();
            if !selected.is_empty() {
                let mut entries = recover_guard(self.entries.write());
                for entry in &selected {
                    if let Some(stored) = entries.get_mut(&entry.id) {
                        stored.access_count = stored.access_count.saturating_add(1);
                        stored.last_accessed = Some(SystemTime::now());
                    }
                }
            }
            tracing::debug!(
                target: "loopctl::metrics",
                span = "memory.retrieve",
                k_requested = limit,
                k_returned = selected.len(),
                top_score = %top_score,
                "vector memory retrieve complete"
            );
            Ok(selected)
        })
    }

    /// Prune low-relevance entries, optionally merge near-duplicates, and
    /// evict every removed entry's vector.
    ///
    /// The pass runs in three phases: snapshot, decide, apply. The snapshot
    /// copies the payload map under a short read lock; deciding (which
    /// entries fall below
    /// [`prune_threshold`](VectorMemoryConfig::prune_threshold) — validated
    /// entries are exempt — and which survivors merge, via one bounded
    /// index search per survivor over the store's own embedding copy,
    /// keeping matches strictly above
    /// [`dedup_threshold`](VectorMemoryConfig::dedup_threshold)) is
    /// computation over the store's vector copy and never a re-embed;
    /// applying awaits
    /// the index removals **before** taking the write locks, then removes
    /// payloads, folds merged victims' tags into their survivors (a
    /// `provider-derived` provenance tag never folds — provenance belongs
    /// to the entry that earned it), and stamps the stats. A duplicate
    /// pair's survivor is the entry with the higher sanitized relevance,
    /// then access count, then creation time, then id — deterministic in
    /// every tie — and an absorber is never itself a victim within a
    /// pass, so every fold's survivor is live at apply time; transitive
    /// chains complete across passes, each folded tag reaching the chain's
    /// final survivor.
    ///
    /// A fresh store returns default stats without touching the embedder
    /// or the index.
    ///
    /// Removals are **snapshot-bound** and the whole pass is
    /// **serialized against stores** by the pass lock: a concurrent
    /// `store` runs either entirely before the snapshot or entirely after
    /// the apply, so the equality re-check below (a candidate is only
    /// deleted if the live entry still equals the snapshot it was decided
    /// on) is defense-in-depth rather than the primary guarantee — a
    /// `store` racing the pass cannot be swallowed by a prune that
    /// predated it, and no replacement can be left indexed-or-mapped
    /// half-written. A raced candidate — reachable today when a
    /// concurrent `retrieve` stamps
    /// [`access_count`](MemoryEntry::access_count)/[`last_accessed`](MemoryEntry::last_accessed)
    /// on it after the snapshot, since `retrieve` takes no part in the
    /// pass lock — is kept and its tombstoned index slot is re-linked
    /// before the pass returns.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let _pass = self.pass_lock.lock().await;
            let entries_before = recover_guard(self.entries.read()).len();
            if entries_before == 0 {
                return Ok(ConsolidationStats::default());
            }
            let threshold = self.config.prune_threshold;
            let snapshot: Vec<MemoryEntry> = {
                let entries = recover_guard(self.entries.read());
                entries.values().cloned().collect()
            };
            let pruned_ids: Vec<Uuid> = snapshot
                .iter()
                .filter(|entry| !entry.validated && sanitize_relevance(entry.relevance) < threshold)
                .map(|entry| entry.id)
                .collect();
            let survivors: Vec<MemoryEntry> = snapshot
                .iter()
                .filter(|entry| entry.validated || sanitize_relevance(entry.relevance) >= threshold)
                .cloned()
                .collect();
            let mut live: Vec<(Uuid, Vec<f32>)> = {
                let vectors = recover_guard(self.vectors.read());
                survivors
                    .iter()
                    .filter_map(|entry| {
                        vectors
                            .get(&entry.id)
                            .map(|vector| (entry.id, vector.as_slice().to_vec()))
                    })
                    .collect()
            };
            live.sort_by_key(|entry| entry.0);
            let (merged_ids, tag_folds) = self.plan_merges(&survivors, &live).await?;
            let by_id: HashMap<Uuid, &MemoryEntry> =
                snapshot.iter().map(|entry| (entry.id, entry)).collect();
            let mut candidates: Vec<(Uuid, MemoryEntry)> = Vec::new();
            for id in pruned_ids.iter().chain(merged_ids.iter()) {
                if let Some(snapshot_entry) = by_id.get(id) {
                    candidates.push((*id, (*snapshot_entry).clone()));
                }
            }
            for (id, _) in &candidates {
                self.index.remove(*id).await?;
            }
            let (removed_ids, raced_ids): (Vec<Uuid>, Vec<Uuid>) = {
                let entries = recover_guard(self.entries.read());
                let mut removed = Vec::new();
                let mut raced = Vec::new();
                for (id, snapshot_entry) in &candidates {
                    match entries.get(id) {
                        Some(current) if current == snapshot_entry => removed.push(*id),
                        _ => raced.push(*id),
                    }
                }
                (removed, raced)
            };
            let removed_set: HashSet<Uuid> = removed_ids.iter().copied().collect();
            for id in &raced_ids {
                let replacement = recover_guard(self.vectors.read()).get(id).cloned();
                if let Some(vector) = replacement {
                    self.index.add(*id, vector).await?;
                }
            }
            {
                let mut entries = recover_guard(self.entries.write());
                for id in &removed_ids {
                    entries.remove(id);
                }
                for fold in &tag_folds {
                    if removed_set.contains(&fold.victim)
                        && let Some(survivor) = entries.get_mut(&fold.survivor)
                    {
                        for tag in &fold.tags {
                            if !survivor.tags.contains(tag) {
                                survivor.tags.push(tag.clone());
                            }
                        }
                    }
                }
            }
            {
                let mut vectors = recover_guard(self.vectors.write());
                for id in &removed_ids {
                    vectors.remove(id);
                }
            }
            let entries_after = recover_guard(self.entries.read()).len();
            let removed_total = removed_ids.len();
            let bytes_saved = removed_total.saturating_mul(self.embedder.dim().saturating_mul(4));
            let pruned_count = pruned_ids
                .iter()
                .filter(|id| removed_set.contains(id))
                .count();
            let merged_count = merged_ids
                .iter()
                .filter(|id| removed_set.contains(id))
                .count();
            tracing::debug!(
                target: "loopctl::metrics",
                span = "memory.consolidate",
                removed = removed_total,
                entries_after = entries_after,
                "vector memory consolidation complete"
            );
            Ok(ConsolidationStats {
                entries_before,
                entries_after,
                pruned: pruned_count,
                merged: merged_count,
                bytes_saved,
            })
        })
    }

    /// Returns the number of entries currently stored.
    ///
    /// Counts payload entries, which is the
    /// [`LoopMemory`] contract's basis; the index holds
    /// exactly one vector per stored entry between consolidation passes.
    fn len(&self) -> usize {
        recover_guard(self.entries.read()).len()
    }
}

impl VectorMemoryStore {
    /// Decide which near-duplicate survivors merge, without re-embedding.
    ///
    /// Survivors are visited in id order over the id-sorted `live`
    /// vectors; each survivor's candidates come from one
    /// [`VectorIndex::search`] over the store's own copy of its embedding
    /// — the top [`dedup_candidates`](VectorMemoryConfig::dedup_candidates)
    /// matches — keeping only candidates that are survivors, are not
    /// already absorbed, and score strictly above
    /// [`dedup_threshold`](VectorMemoryConfig::dedup_threshold), so
    /// planning costs one bounded index search per survivor instead of an
    /// all-pairs scan. The merge set is therefore top-k-bounded: a
    /// duplicate cluster wider than the window — or chained through an
    /// intermediate absorber — merges across repeated passes rather than
    /// in one. A matching pair folds the lower-ranked
    /// entry into the higher-ranked one (relevance, then access count,
    /// then creation time, then id); already-absorbed entries leave the
    /// candidate scan, so each entry merges at most once per pass, and
    /// an entry that has already won a merge cannot become a victim in
    /// the same pass — a transitive pair defers to a later pass, keeping
    /// every fold's survivor live when the folds apply. Returns
    /// the victim ids and the folds — the `provider-derived` provenance
    /// tag is never folded, because provenance belongs to the entry that
    /// earned it. The pass lock is held throughout, and the index still
    /// contains every survivor at plan time (removals are apply-phase), so
    /// the searches run against the full live set.
    ///
    /// # Errors
    ///
    /// [`LoopError`] unchanged from the per-survivor
    /// [`VectorIndex::search`] calls — an index failure mid-planning
    /// aborts the pass before any removal or fold is applied, leaving
    /// the store exactly as the snapshot saw it.
    async fn plan_merges(
        &self,
        survivors: &[MemoryEntry],
        live: &[(Uuid, Vec<f32>)],
    ) -> Result<(Vec<Uuid>, Vec<MergeFold>), LoopError> {
        let mut merged_ids: Vec<Uuid> = Vec::new();
        let mut tag_folds: Vec<MergeFold> = Vec::new();
        if self.config.dedup_threshold >= 1.0 {
            return Ok((merged_ids, tag_folds));
        }
        let by_survivor: HashMap<Uuid, &MemoryEntry> =
            survivors.iter().map(|entry| (entry.id, entry)).collect();
        let mut absorbed: HashSet<Uuid> = HashSet::new();
        let mut absorbers: HashSet<Uuid> = HashSet::new();
        for (survivor_id, vector) in live {
            if absorbed.contains(survivor_id) {
                continue;
            }
            let Some(first) = by_survivor.get(survivor_id) else {
                continue;
            };
            let candidates = self
                .index
                .search(
                    &Embedding::new(vector.clone()),
                    self.config.dedup_candidates,
                )
                .await?;
            for candidate in candidates {
                if candidate.id == *survivor_id
                    || candidate.score <= self.config.dedup_threshold
                    || absorbed.contains(&candidate.id)
                {
                    continue;
                }
                let Some(second) = by_survivor.get(&candidate.id) else {
                    continue;
                };
                let ordering = sanitize_relevance(first.relevance)
                    .partial_cmp(&sanitize_relevance(second.relevance))
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| first.access_count.cmp(&second.access_count))
                    .then_with(|| first.created_at.cmp(&second.created_at))
                    .then_with(|| first.id.cmp(&second.id));
                let (survivor_entry, victim) = if ordering.is_ge() {
                    (first, second)
                } else {
                    (second, first)
                };
                if absorbers.contains(&victim.id) {
                    continue;
                }
                let folded: Vec<String> = victim
                    .tags
                    .iter()
                    .filter(|tag| tag.as_str() != PROVIDER_DERIVED_TAG)
                    .cloned()
                    .collect();
                tag_folds.push(MergeFold {
                    survivor: survivor_entry.id,
                    victim: victim.id,
                    tags: folded,
                });
                merged_ids.push(victim.id);
                absorbed.insert(victim.id);
                absorbers.insert(survivor_entry.id);
                if victim.id == *survivor_id {
                    break;
                }
            }
            tokio::task::yield_now().await;
        }
        Ok((merged_ids, tag_folds))
    }
}

/// A labelled retrieval fixture shared by every semantic-precision
/// measurement.
///
/// Fifty [`MemoryEntry`] values spanning four of the six
/// [`MemoryCategory`](crate::memory::MemoryCategory) variants —
/// [`Strategy`](crate::memory::MemoryCategory::Strategy),
/// [`Fact`](crate::memory::MemoryCategory::Fact),
/// [`Insight`](crate::memory::MemoryCategory::Insight), and
/// [`ErrorPattern`](crate::memory::MemoryCategory::ErrorPattern);
/// [`Trajectory`](crate::memory::MemoryCategory::Trajectory) and
/// [`Working`](crate::memory::MemoryCategory::Working) are absent because
/// the fixture targets retrieval behaviour, not category coverage —
/// organized as ten topics of five entries each, plus twenty queries whose
/// relevant entries are hand-labelled by fixture index. The wording is
/// engineered for deterministic embedders: a query's tokens appear verbatim
/// in its relevant entries and in few distractors, so the fixture measures
/// the *store* (candidate generation, blending, ranking) rather than a
/// particular model's geometry.
///
/// This is the crate's shared precision fixture: the in-crate gate and the
/// `loopctl-hnsw` companion's drop-in gate both measure against this exact
/// set, so their precision numbers stay comparable across backends.
#[derive(Debug, Clone)]
#[cfg(feature = "testing")]
pub struct GoldenSet {
    /// The fixture entries, in fixture order.
    ///
    /// Query labels reference these by index; store them into the store
    /// under test in this order.
    pub entries: Vec<MemoryEntry>,

    /// The fixture queries with their hand-labelled relevant indices.
    ///
    /// Indices point into [`entries`](GoldenSet::entries) and always name
    /// entries whose text shares the query's content words.
    pub queries: Vec<GoldenQuery>,
}

/// One labelled query in the [`GoldenSet`] fixture.
///
/// `relevant_indices` names the fixture entries a correct retrieval should
/// surface; a measurement counts the query as answered when any of them
/// appears in the returned window.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "testing")]
pub struct GoldenQuery {
    /// The query text, phrased the way a caller would ask.
    ///
    /// Content words are chosen from the relevant entries' own vocabulary,
    /// with ordinary interrogative filler that matches nothing.
    pub text: &'static str,

    /// Fixture indices of the entries a correct answer surfaces.
    ///
    /// At least one labelled entry shares every content word of
    /// [`text`](GoldenQuery::text); the rest share a strict subset.
    pub relevant_indices: &'static [usize],
}

#[cfg(feature = "testing")]
impl GoldenSet {
    /// The entry ids a query's labels resolve to.
    ///
    /// Callers store [`entries`](GoldenSet::entries) into their store under
    /// test, run [`retrieve`](LoopMemory::retrieve) with
    /// [`text`](GoldenQuery::text), and count a hit when any returned id
    /// appears here — the shared definition of "answered correctly" across
    /// every backend that measures against this fixture. An out-of-range
    /// `query_index` yields an empty vector rather than a panic.
    #[must_use]
    pub fn relevant_ids(&self, query_index: usize) -> Vec<Uuid> {
        self.queries
            .get(query_index)
            .map_or_else(Vec::new, |query| {
                query
                    .relevant_indices
                    .iter()
                    .filter_map(|index| self.entries.get(*index))
                    .map(|entry| entry.id)
                    .collect()
            })
    }

    /// The share of queries answered correctly at the given window.
    ///
    /// Runs nothing — callers hand in the per-query hit flags — so the
    /// arithmetic (hits over queries, saturating) is identical for every
    /// consumer. An empty `hits` slice scores `0.0`.
    #[must_use]
    pub fn precision_ratio(hits: &[bool]) -> f32 {
        let answered = hits.iter().filter(|hit| **hit).count();
        unit_ratio(answered, hits.len())
    }
}

/// The fixture entries: ten topics of five, `(category, text, topic tag)`.
///
/// The first entry of each topic is the topic's hub — it contains every
/// content word the topic's queries use — and the remaining four share
/// subsets, which is what lets a query's top window reliably contain a
/// labelled entry regardless of hash collisions in a deterministic
/// embedder.
#[cfg(feature = "testing")]
static GOLDEN_ENTRIES: [(crate::memory::MemoryCategory, &str, &str); 50] = [
    (
        crate::memory::MemoryCategory::Strategy,
        "glob expands patterns into a fast listing of files and paths",
        "files",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "prefer glob over opening every directory when listing many files",
        "files",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "recursive paths can be filtered after the glob listing returns",
        "files",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "a glob that matches no files still returns an empty listing",
        "files",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "large paths slow the listing, so narrow the glob first",
        "files",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "on a 429 response retry with exponential backoff and respect the rate headers",
        "reliability",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "the retry ladder doubles its backoff each attempt after a 429",
        "reliability",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "a shared rate budget means the retry must wait for the window",
        "reliability",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "retries without backoff hammer a rate-limited endpoint into worse 429s",
        "reliability",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "cap the retry count so backoff cannot stall the run forever",
        "reliability",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "hashing folds tokens into buckets to build deterministic test embeddings",
        "embeddings",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "each embedding buckets similar words together when the hashing collides",
        "embeddings",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "deterministic embeddings make retrieval checks reproducible without a model",
        "embeddings",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "a hashing embedder needs a wide bucket space or distinct words blur",
        "embeddings",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "normalize embeddings before any scoring pass",
        "embeddings",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "compaction triggers when the context window crosses its threshold and summarizes older history",
        "context",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "a smaller window reaches the compaction threshold sooner",
        "context",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "summarize before compaction so the newest turns survive verbatim",
        "context",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "skipping compaction at the threshold floods the window with stale turns",
        "context",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "compaction quality is bounded by what the summary keeps",
        "context",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "cosine similarity scores the angle between two vectors and ignores their length",
        "retrieval",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "two parallel vectors carry a similarity of one under cosine",
        "retrieval",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "orthogonal vectors score zero similarity no matter their magnitudes",
        "retrieval",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "rank by similarity descending when searching vectors",
        "retrieval",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "an angle near zero means maximum similarity between vectors",
        "retrieval",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "memoize tool results in a cache and evict on invalidation after writes",
        "performance",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "a cached result is reused until an invalidation event evicts the key",
        "performance",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "the memoize layer short-circuits repeated calls before they reach the tool",
        "performance",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "forgetting invalidation serves stale cache entries after a write",
        "performance",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "scope the cache key narrowly so memoize cannot collapse distinct calls",
        "performance",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "sqlite with wal journaling keeps memory durable across restarts through transactions",
        "persistence",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "a wal checkpoint bounds how much the durable log replays",
        "persistence",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "wrap multi-row updates in transactions so sqlite stays consistent",
        "persistence",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "durability here means process crashes, not power loss, under wal",
        "persistence",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "open the sqlite database before the loop starts to fail fast",
        "persistence",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "land every feature on its branch, rebase onto master, and note it in the changelog before the commit",
        "process",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "one commit per task keeps the changelog honest",
        "process",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "a stale branch needs a rebase before the commit lands",
        "process",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "the changelog entry names what changed and why the commit exists",
        "process",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "never amend a pushed commit on a shared branch",
        "process",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "cassettes replay recorded wire responses and golden bytes pin what the engine sends",
        "testing",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "a red cassette means the provider or client drifted on the wire",
        "testing",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "regenerating a golden file is a deliberate, reviewed act",
        "testing",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "replay hermetically so cassette checks never touch the network",
        "testing",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "array order in a golden body is pinned on purpose",
        "testing",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "redact secrets from tool output by scrubbing bearer headers and high-entropy tokens",
        "security",
    ),
    (
        crate::memory::MemoryCategory::Fact,
        "the scrub pass replaces each secret with a redaction marker",
        "security",
    ),
    (
        crate::memory::MemoryCategory::Insight,
        "an entropy heuristic catches secrets the curated patterns miss",
        "security",
    ),
    (
        crate::memory::MemoryCategory::ErrorPattern,
        "logging raw output leaks secrets that a redact pass would have caught",
        "security",
    ),
    (
        crate::memory::MemoryCategory::Strategy,
        "keep redaction before persistence so stored traces hold no secrets",
        "security",
    ),
];

/// The fixture queries: two per topic, each labelled with the indices of
/// entries that share its content words.
///
/// Labels are fixture positions resolved against [`GOLDEN_ENTRIES`] by
/// [`golden_set()`], so a measurement's relevant ids always come from the
/// same build of the fixture.
#[cfg(feature = "testing")]
static GOLDEN_QUERIES: [GoldenQuery; 20] = [
    GoldenQuery {
        text: "how do I list files quickly with glob",
        relevant_indices: &[0, 1, 2, 4],
    },
    GoldenQuery {
        text: "find files by their paths",
        relevant_indices: &[0, 2, 4],
    },
    GoldenQuery {
        text: "when should I retry with backoff",
        relevant_indices: &[5, 6, 9],
    },
    GoldenQuery {
        text: "what does a 429 rate response mean for my budget",
        relevant_indices: &[5, 6, 7, 8],
    },
    GoldenQuery {
        text: "how are test embeddings built from hashing",
        relevant_indices: &[10, 11, 13],
    },
    GoldenQuery {
        text: "why do deterministic embeddings help reproducible checks",
        relevant_indices: &[10, 11, 12],
    },
    GoldenQuery {
        text: "when does compaction trigger on the context window",
        relevant_indices: &[15, 16, 17, 18],
    },
    GoldenQuery {
        text: "how do I summarize older turns before compaction",
        relevant_indices: &[15, 17, 19],
    },
    GoldenQuery {
        text: "how is cosine similarity computed for vectors",
        relevant_indices: &[20, 21, 22, 23, 24],
    },
    GoldenQuery {
        text: "what similarity do orthogonal vectors score",
        relevant_indices: &[20, 22, 23, 24],
    },
    GoldenQuery {
        text: "when does the memoize cache evict a key",
        relevant_indices: &[25, 26, 28],
    },
    GoldenQuery {
        text: "how does memoize handle invalidation after writes",
        relevant_indices: &[25, 26, 28],
    },
    GoldenQuery {
        text: "is sqlite durable across restarts with wal",
        relevant_indices: &[30, 31, 33],
    },
    GoldenQuery {
        text: "why wrap updates in transactions with sqlite",
        relevant_indices: &[30, 32],
    },
    GoldenQuery {
        text: "what belongs in the changelog of a commit",
        relevant_indices: &[35, 36, 38],
    },
    GoldenQuery {
        text: "when does a branch need a rebase before landing",
        relevant_indices: &[35, 37],
    },
    GoldenQuery {
        text: "what does a red cassette mean on the wire",
        relevant_indices: &[40, 41, 43],
    },
    GoldenQuery {
        text: "why are golden files regenerated deliberately",
        relevant_indices: &[40, 42],
    },
    GoldenQuery {
        text: "how do I redact secrets from tool output",
        relevant_indices: &[45, 46, 48, 49],
    },
    GoldenQuery {
        text: "which heuristic catches missed secrets",
        relevant_indices: &[45, 47, 49],
    },
];

/// Build the shared golden-set fixture.
///
/// Entries are freshly constructed on every call — new `Uuid`s, current
/// timestamps — so each measurement gets its own independent store
/// population; the labels in [`GoldenQuery`] reference fixture positions,
/// and [`GoldenSet::relevant_ids`] resolves them to the ids of *this*
/// build. Gate this behind `testing` in downstream code: it is test
/// tooling, not a runtime data source.
#[cfg(feature = "testing")]
#[must_use]
pub fn golden_set() -> GoldenSet {
    let entries: Vec<MemoryEntry> = GOLDEN_ENTRIES
        .iter()
        .map(|(category, text, tag)| MemoryEntry::new(*category, *text).with_tag(*tag))
        .collect();
    GoldenSet {
        entries,
        queries: GOLDEN_QUERIES.to_vec(),
    }
}
