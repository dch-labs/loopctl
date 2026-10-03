//! HNSW-backed [`VectorIndex`] for loopctl's semantic memory.
//!
//! [`HnswIndex`] implements loopctl's vector-index trait over a
//! hand-rolled, incremental Hierarchical Navigable Small World graph
//! (Malkov & Yashunin, 2016): `add` links a new vector into the layered
//! graph immediately, `remove` tombstones without disturbing traversal,
//! and compaction clears tombstones into a fresh graph — automatically
//! past the configured tombstone ratio, or on demand through `rebuild`. Distances are
//! computed with loopctl's own [`cosine_similarity`], so results are
//! directly comparable with the reference [`LinearVectorIndex`] — the
//! recall gate in this crate's tests pins that comparability at
//! recall@10 ≥ 90% against brute force on random unit vectors.
//!
//! Construction is deterministic: node levels are drawn from a
//! seeded [`fastrand::Rng`] (the seed is a parameter with a fixed
//! default), and every construction tie-break sorts by slot, so the same
//! insert sequence builds an index that answers identically — the pinned,
//! result-level contract that keeps the recall and precision gates
//! reproducible.
//!
//! Add this crate as a direct dependency alongside `loopctl`; no feature
//! on `loopctl` itself is required:
//!
//! ```toml
//! [dependencies]
//! loopctl = { version = "0.3", features = ["vector_memory"] }
//! loopctl-hnsw = "0.3"
//! ```
//!
//! [`VectorIndex`]: loopctl::memory::vector::VectorIndex
//! [`LinearVectorIndex`]: loopctl::memory::vector::LinearVectorIndex
//! [`cosine_similarity`]: loopctl::memory::vector::cosine_similarity

#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::missing_errors_doc,
        clippy::unnecessary_wraps,
        clippy::clone_on_ref_ptr,
        clippy::doc_markdown,
        clippy::field_reassign_with_default,
        clippy::used_underscore_items,
        clippy::wildcard_imports,
    )
)]

mod graph;
mod map;

use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use loopctl::error::{LoopError, recover_guard};
use loopctl::memory::vector::{Embedding, VectorIndex, VectorMatch};
use uuid::Uuid;

use crate::graph as hnsw_graph;
use crate::map::IdMap;

/// The seed every index builds with unless
/// [`with_params`](HnswIndex::with_params) overrides it.
///
/// A fixed constant — never entropy — so two indexes built from the same
/// insert sequence answer identically, which is what keeps the recall and
/// precision gates reproducible. The pinned contract is result-level:
/// identical search results, pinned at build time by
/// `same_seed_and_insertion_sequence_build_identical_indexes` and across
/// rebuilds by `a_rebuild_replays_the_same_live_set_identically`.
pub const DEFAULT_SEED: u64 = 0x5EED_1A18;

/// Tuning knobs for [`HnswIndex`].
///
/// The defaults follow the paper's typical range and are what the recall
/// gate in this crate's tests runs at; raise `ef_search` (or
/// `ef_construction`) to trade search latency for recall on harder data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// Links per node on the upper layers (`M` in the paper).
    ///
    /// Layer 0 carries double this, per the standard construction. Values
    /// below 2 clamp to 2, the smallest value that keeps a graph
    /// navigable.
    pub m: usize,

    /// Candidate list size while *building* (`efConstruction`).
    ///
    /// Larger values explore more neighbours when linking a new node and
    /// build a higher-quality graph at insert cost.
    pub ef_construction: usize,

    /// Candidate list size while *searching* (`efSearch`).
    ///
    /// The effective value at query time never drops below the requested
    /// `k` plus the live tombstone count — deleted slots must be
    /// traversed past — and never exceeds the total slot count, live
    /// plus tombstoned: the whole traversable graph.
    pub ef_search: usize,

    /// The seed for the level-assignment RNG.
    ///
    /// Fixed by default ([`DEFAULT_SEED`]); two indexes sharing a seed and
    /// an insert sequence produce identical search results — the pinned,
    /// result-level contract.
    pub seed: u64,

    /// The tombstone-to-live ratio past which `add` and `remove` compact
    /// the graph automatically.
    ///
    /// Compaction is a full graph replay under the write lock, so it is
    /// amortized behind a debt threshold rather than run per removal: it
    /// fires once tombstones exceed `len() × max_tombstone_ratio`. The
    /// default `1.0` tolerates one tombstone per live vector — bounded
    /// over-fetch in the meantime; a lower ratio trades replay cost for a
    /// leaner graph sooner. Non-finite and non-positive values sanitize
    /// to the default.
    pub max_tombstone_ratio: f32,
}

impl Default for HnswParams {
    /// The defaults the recall gate runs at: `M = 16`,
    /// `efConstruction = 200`, `efSearch = 128`, fixed seed, compaction
    /// past one tombstone per live vector.
    ///
    /// Every constructor and both quality gates run at these values;
    /// raise `ef_search` (or `ef_construction`) through [`HnswParams`]
    /// to trade search latency for recall on harder data.
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 128,
            seed: DEFAULT_SEED,
            max_tombstone_ratio: 1.0,
        }
    }
}

impl HnswParams {
    /// Clamp the parameters to their minimum viable values.
    ///
    /// `m` clamps to at least 2; the two `ef` values clamp to at least 1,
    /// so a hand-built parameter set cannot produce a degenerate search.
    /// The tombstone ratio must be finite and positive — anything else
    /// would either never trigger compaction or trigger it on every
    /// removal — so degenerate values restore the default `1.0`.
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        self.m = self.m.max(2);
        self.ef_construction = self.ef_construction.max(1);
        self.ef_search = self.ef_search.max(1);
        if !self.max_tombstone_ratio.is_finite() || self.max_tombstone_ratio <= 0.0 {
            self.max_tombstone_ratio = 1.0;
        }
        self
    }
}

/// The mutable interior of [`HnswIndex`]: the slot map, the graph, and
/// the level-assignment RNG.
///
/// One write lock guards all three so an [`add`](VectorIndex::add) is
/// atomic — the fresh slot, its links, and the consumed level draw land
/// together or not at all — and a `rebuild` swaps the map and graph in
/// one assignment batch, so a reader never observes a half-replayed
/// index. The map is the single source of truth for which ids are live;
/// the graph holds adjacency only, and every distance either operation
/// computes reads the map's vectors. Poisoned locks recover per
/// operation, matching the crate's Category-1 policy for
/// single-operation data.
struct Inner {
    /// The `Uuid ↔ u32` slot map with its tombstone set.
    ///
    /// The single source of truth for which ids are live — the graph's
    /// adjacency lists are indexes into it — and the storage every
    /// distance computation reads its vectors from.
    map: IdMap,

    /// The layered graph over the map's slots.
    ///
    /// Adjacency only: no coordinates live here, so a rebuild can swap
    /// the graph wholesale without touching a single stored vector.
    graph: hnsw_graph::Graph,

    /// The seeded RNG node levels are drawn from, persisted across adds.
    ///
    /// Because the generator is never rewound during inserts, the same
    /// insert sequence against the same seed consumes the same draw
    /// sequence — the determinism the pinned result-level contract rests
    /// on, and what `rebuild` deliberately resets away.
    rng: fastrand::Rng,
}

/// An approximate-nearest-neighbour [`VectorIndex`] over a pure-Rust HNSW
/// graph.
///
/// The production-grade backend for loopctl's `VectorMemoryStore`
/// (`loopctl::memory::vector_memory`, feature `vector_memory`):
/// sublinear search over millions of in-RAM vectors where the brute-force
/// linear index would rescan everything. Up to a few thousand vectors the
/// linear index is simpler and exactly equal in recall — the trait's
/// scoring contract makes the two interchangeable.
///
/// # Removal model
///
/// HNSW graphs have no native deletion: [`remove`](VectorIndex::remove)
/// tombstones the slot (still traversable, never returned) and a
/// compaction clears the tombstones into a fresh graph — explicitly
/// through [`rebuild`](HnswIndex::rebuild), or automatically at the end
/// of `add`/`remove` once tombstones exceed
/// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio) live slots.
/// `search` over-fetches by the tombstone count so a deleted
/// neighbour cannot crowd a live one out of the top `k`.
///
/// # Thread safety
///
/// `Send + Sync`; all operations take `&self` over an internal `RwLock`
/// recovered per operation on poison, matching the trait's
/// shareable-via-`Arc` contract.
///
/// # Example
///
/// ```
/// use loopctl::memory::vector::{Embedding, VectorIndex};
/// use loopctl_hnsw::HnswIndex;
/// use uuid::Uuid;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let index = HnswIndex::new(4);
/// index
///     .add(Uuid::new_v4(), Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0]))
///     .await
///     .unwrap();
/// let hits = index
///     .search(&Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0]), 5)
///     .await
///     .unwrap();
/// assert_eq!(hits.len(), 1);
/// # });
/// ```
pub struct HnswIndex {
    /// The dimensionality every stored vector must have.
    ///
    /// Checked on `add` and `search` exactly like the linear index, so a
    /// mismatched vector is rejected with [`LoopError::Memory`].
    dim: usize,

    /// The sanitized graph parameters.
    ///
    /// Stored post-[`sanitized`](HnswParams::sanitized), so every read —
    /// the insert path's capacities, search's effective `ef` — can assume
    /// the documented minimums without re-checking.
    params: HnswParams,

    /// The map, graph, and RNG behind the write lock.
    ///
    /// One lock for the three of them keeps an add atomic: the new
    /// slot, its links, and the consumed RNG draw all land together.
    inner: RwLock<Inner>,
}

impl HnswIndex {
    /// Create an empty index that only accepts `dim`-dimensional vectors,
    /// with the default [`HnswParams`].
    ///
    /// The defaults are the parameters the recall gate runs at; use
    /// [`with_params`](Self::with_params) to trade search latency for
    /// recall or to pin a different level-draw seed.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, HnswParams::default())
    }

    /// Create an empty index with explicit parameters.
    ///
    /// The parameters are passed through
    /// [`HnswParams::sanitized`], so stored copies always satisfy their
    /// documented minimums.
    #[must_use]
    pub fn with_params(dim: usize, params: HnswParams) -> Self {
        let sanitized = params.sanitized();
        Self {
            dim,
            params: sanitized,
            inner: RwLock::new(Inner {
                map: IdMap::new(),
                graph: hnsw_graph::Graph::new(),
                rng: fastrand::Rng::with_seed(sanitized.seed),
            }),
        }
    }

    /// Compact tombstoned slots into a fresh graph.
    ///
    /// Live vectors are replayed into a new map and graph in allocation
    /// order with the seed reset to the configured value, so the same
    /// live set always rebuilds to an index with identical search results
    /// (pinned by `a_rebuild_replays_the_same_live_set_identically`).
    /// Recall is unaffected —
    /// same vectors, same metric, fresh links. This is also the engine
    /// behind automatic compaction: `add` and `remove` run the same
    /// replay once tombstones pass
    /// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio), so an
    /// index left to its own devices never accumulates unbounded
    /// tombstone debt.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the rebuilt id space overflows `u32` —
    /// unreachable in practice, refused rather than wrapped.
    pub fn rebuild(&self) -> Result<(), LoopError> {
        let mut inner = recover_guard(self.inner.write());
        Self::compact_forced(&mut inner, self.params)
    }

    /// Replay the live set into a fresh interior, deterministically.
    ///
    /// The one compaction engine: live vectors are re-inserted in
    /// allocation order into a new map and graph with the level RNG reset
    /// to the configured seed — the same live set always replays to an
    /// index with identical search results, whether the replay was
    /// requested through [`rebuild`](HnswIndex::rebuild) or triggered by
    /// the tombstone ratio.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the replayed id space overflows `u32` —
    /// unreachable in practice, refused rather than wrapped.
    fn replay_live(live: Vec<(Uuid, Vec<f32>)>, params: HnswParams) -> Result<Inner, LoopError> {
        let mut map = IdMap::new();
        let mut graph = hnsw_graph::Graph::new();
        let mut rng = fastrand::Rng::with_seed(params.seed);
        for (id, vector) in live {
            let slot = map.insert(id, vector.clone())?;
            let level = hnsw_graph::random_level(&mut rng, params.m);
            hnsw_graph::insert(&map, &mut graph, params, slot, &vector, level);
        }
        Ok(Inner { map, graph, rng })
    }

    /// Replay the live set into `inner`, unconditionally.
    ///
    /// What [`rebuild`](HnswIndex::rebuild) reduces to once the write
    /// lock is held: the live set is collected from the map and handed to
    /// [`replay_live`](Self::replay_live), and the fresh interior replaces
    /// the old in one assignment batch.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the replayed id space overflows `u32` —
    /// unreachable in practice, refused rather than wrapped.
    fn compact_forced(inner: &mut Inner, params: HnswParams) -> Result<(), LoopError> {
        let live: Vec<(Uuid, Vec<f32>)> = inner
            .map
            .live()
            .into_iter()
            .map(|(_, id, vector)| (id, vector.to_vec()))
            .collect();
        *inner = Self::replay_live(live, params)?;
        Ok(())
    }

    /// Compact past the configured tombstone ratio, under the held write
    /// lock.
    ///
    /// The threshold check both mutation paths run after their write
    /// lands: once tombstones exceed the live count scaled by
    /// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio), the
    /// live set replays through [`compact_forced`](Self::compact_forced)
    /// — the same engine [`rebuild`](HnswIndex::rebuild) drives — so
    /// tombstone debt stays bounded without any host intervention.
    /// Below the ratio this is a no-op, keeping the replay amortized.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the replayed id space overflows `u32` —
    /// unreachable in practice, refused rather than wrapped.
    fn compact_if_past_ratio(inner: &mut Inner, params: HnswParams) -> Result<(), LoopError> {
        if !tombstones_exceed_ratio(
            inner.map.tombstone_count(),
            inner.map.len(),
            params.max_tombstone_ratio,
        ) {
            return Ok(());
        }
        Self::compact_forced(inner, params)
    }

    /// The number of tombstoned slots the graph still carries.
    ///
    /// Slots removed since the last compaction keep their vectors so
    /// traversal can walk past them; this count is how a host observes
    /// that debt — against the configured
    /// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio), past
    /// which `add` and `remove` compact automatically — or when deciding
    /// on an explicit [`rebuild`](HnswIndex::rebuild).
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        recover_guard(self.inner.read()).map.tombstone_count()
    }
}

impl VectorIndex for HnswIndex {
    /// Returns the fixed dimension every vector in this index carries.
    ///
    /// Set at construction; a vector of any other dimension is rejected
    /// by [`add`](VectorIndex::add) rather than truncated or padded.
    fn dim(&self) -> usize {
        self.dim
    }

    /// Link one vector into the graph under `id`.
    ///
    /// Upsert semantics: an id that is already live is tombstoned first
    /// (its old slot stays traversable until a compaction) and a fresh
    /// slot is linked, so [`len`](VectorIndex::len) counts distinct live
    /// ids. The new node's level is drawn from the index's seeded RNG,
    /// making an insert sequence reproducible end to end. Once the
    /// tombstone debt passes
    /// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio), the
    /// insert finishes by compacting the graph — the same replay
    /// [`rebuild`](HnswIndex::rebuild) performs — so replace-heavy
    /// workloads cannot grow the graph without bound.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the vector's dimensionality differs
    /// from the index's, and when the id space is exhausted (more than
    /// `u32::MAX` slots — unreachable in practice). The capacity check
    /// runs before the previous slot is tombstoned, so a failed upsert
    /// leaves the caller's existing entry live.
    fn add(
        &self,
        id: Uuid,
        vector: Embedding,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            if vector.dim() != self.dim {
                return Err(LoopError::Memory(format!(
                    "vector dimension mismatch: index is {}, got {}",
                    self.dim,
                    vector.dim()
                )));
            }
            let mut inner = recover_guard(self.inner.write());
            {
                let Inner { map, graph, rng } = &mut *inner;
                map.ensure_capacity()?;
                map.remove(id);
                let stored = vector.as_slice().to_vec();
                let slot = map.insert(id, stored.clone())?;
                let level = hnsw_graph::random_level(rng, self.params.m);
                hnsw_graph::insert(map, graph, self.params, slot, &stored, level);
            }
            Self::compact_if_past_ratio(&mut inner, self.params)
        })
    }

    /// Return the `k` nearest live vectors to `query`, most-similar first.
    ///
    /// The search descends greedily through the upper layers, runs a
    /// best-first search on layer 0 sized to
    /// `max(ef_search, k + tombstones)` and capped at the total slot
    /// count, filters tombstoned slots, and
    /// scores results with loopctl's
    /// [`cosine_similarity`](loopctl::memory::vector::cosine_similarity) —
    /// the same
    /// metric, clamped to `-1.0..=1.0`, with the same descending-score,
    /// id-tiebreak ordering the linear index guarantees. Emits one
    /// `vector.index.search` metric event per successful call.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the query's dimensionality differs from
    /// the index's.
    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<VectorMatch>, LoopError>> + Send + '_>> {
        let query = query.as_slice().to_vec();
        Box::pin(async move {
            if query.len() != self.dim {
                return Err(LoopError::Memory(format!(
                    "query dimension mismatch: index is {}, got {}",
                    self.dim,
                    query.len()
                )));
            }
            let started = std::time::Instant::now();
            if k == 0 {
                emit_search_metric(k, &[], started);
                return Ok(Vec::new());
            }
            let inner = recover_guard(self.inner.read());
            if inner.map.is_empty() {
                emit_search_metric(k, &[], started);
                return Ok(Vec::new());
            }
            let Some((entry_slot, top_level)) = inner.graph.entry() else {
                emit_search_metric(k, &[], started);
                return Ok(Vec::new());
            };
            let ef = inner
                .map
                .tombstone_count()
                .saturating_add(k)
                .max(self.params.ef_search)
                .min(inner.map.len().saturating_add(inner.map.tombstone_count()));
            let mut entry_point = entry_slot;
            for layer in (1..=top_level).rev() {
                entry_point =
                    hnsw_graph::greedy(&inner.map, &inner.graph, &query, entry_point, layer);
            }
            let entry_points: Vec<(f32, u32)> = vec![(
                hnsw_graph::distance_to(&inner.map, &query, entry_point),
                entry_point,
            )];
            let found =
                hnsw_graph::search_layer(&inner.map, &inner.graph, &query, &entry_points, ef, 0);
            let mut matches: Vec<VectorMatch> = found
                .into_iter()
                .filter_map(|(distance, slot)| {
                    let id = inner.map.id(slot)?;
                    if inner.map.is_tombstoned(slot) {
                        return None;
                    }
                    Some(VectorMatch {
                        id,
                        score: (1.0 - distance).clamp(-1.0, 1.0),
                    })
                })
                .collect();
            matches.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            let matches: Vec<VectorMatch> = matches.into_iter().take(k).collect();
            emit_search_metric(k, &matches, started);
            Ok(matches)
        })
    }

    /// Tombstone the vector stored under `id`, if any.
    ///
    /// Idempotent like the linear index: a missing id is a no-op. The
    /// slot stays traversable so the graph never loses connectivity, but
    /// it can no longer be returned; a compaction drops it for real —
    /// explicitly through [`rebuild`](HnswIndex::rebuild), or
    /// automatically at the end of this very removal once the tombstone
    /// debt passes
    /// [`max_tombstone_ratio`](HnswParams::max_tombstone_ratio).
    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let mut inner = recover_guard(self.inner.write());
            inner.map.remove(id);
            Self::compact_if_past_ratio(&mut inner, self.params)
        })
    }

    /// Returns the number of live vectors, excluding tombstoned slots.
    ///
    /// Paired with the trait's provided
    /// [`is_empty`](VectorIndex::is_empty) default.
    fn len(&self) -> usize {
        recover_guard(self.inner.read()).map.len()
    }
}

/// Whether tombstone debt has passed the configured ratio of live slots.
///
/// Counts convert through `u32` into `f64` — lossless at both steps, and
/// the lint-clean path the repo-wide no-cast-allow ruling permits — so
/// the comparison needs no raw `usize as f32` narrowing; a count above
/// the `u32` slot space cannot exist, making the saturating conversion
/// exact in every reachable state.
fn tombstones_exceed_ratio(tombstones: usize, live: usize, ratio: f32) -> bool {
    let tombstones = f64::from(u32::try_from(tombstones).unwrap_or(u32::MAX));
    let live = f64::from(u32::try_from(live).unwrap_or(u32::MAX));
    tombstones > live * f64::from(ratio)
}

/// Emit the `vector.index.search` metric event for one completed search.
///
/// Called on every successful return — the degenerate successes (`k = 0`,
/// an empty map, an empty graph) included — so the event stream matches
/// the linear index's one-event-per-success contract exactly and the two
/// backends stay indistinguishable in metrics.
fn emit_search_metric(k: usize, matches: &[VectorMatch], started: std::time::Instant) {
    let top_score = matches.first().map_or(0.0, |m| m.score);
    tracing::debug!(
        target: "loopctl::metrics",
        span = "vector.index.search",
        provider = "hnsw",
        k,
        returned = matches.len(),
        top_score = %top_score,
        duration_ms = %started.elapsed().as_millis(),
        "vector index search complete"
    );
}

#[cfg(test)]
mod tests {
    use super::HnswParams;

    /// Degenerate tombstone ratios sanitize to the default.
    ///
    /// Non-finite, zero, and negative ratios all mean "not a usable
    /// threshold" — the first never triggers compaction, the others fire
    /// it on every removal — so the sanitized parameter set must carry
    /// the default `1.0` instead.
    #[test]
    fn a_degenerate_tombstone_ratio_sanitizes_to_the_default() {
        for degenerate in [f32::NAN, f32::INFINITY, 0.0, -0.5] {
            let params = HnswParams {
                max_tombstone_ratio: degenerate,
                ..HnswParams::default()
            }
            .sanitized();
            assert!(
                (params.max_tombstone_ratio - 1.0).abs() < f32::EPSILON,
                "a degenerate ratio ({degenerate}) must sanitize to the default 1.0"
            );
        }
    }
}
