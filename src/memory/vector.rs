//! Vector primitives for semantic-retrieval memory.
//!
//! [`EmbeddingProvider`] turns text into an [`Embedding`]; [`VectorIndex`]
//! stores embeddings keyed by `Uuid` and answers nearest-neighbour
//! [`search`](VectorIndex::search)es. This module ships two reference
//! implementations — [`HashingEmbedder`] and [`LinearVectorIndex`] — so the
//! traits are usable end-to-end with no external model and no network.
//!
//! These are the building blocks a semantic `LoopMemory` store composes
//! from: hold an [`EmbeddingProvider`] and a [`VectorIndex`], embed on
//! store, search on retrieve. Nothing here implements
//! [`LoopMemory`](super::LoopMemory) directly.
//!
//! # Example
//!
//! ```
//! use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex, VectorIndex};
//!
//! let embedder = HashingEmbedder::new(64);
//! let index = LinearVectorIndex::new(64);
//! assert_eq!(index.dim(), 64);
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::{PoisonError, RwLock};
use std::time::Instant;

use uuid::Uuid;

use crate::error::LoopError;

/// A dense vector embedding of a piece of text.
///
/// Produced by [`EmbeddingProvider::embed`], consumed by [`VectorIndex`].
/// Carries no metadata beyond its coordinates; the originating text and any
/// [`MemoryEntry`](super::MemoryEntry) live elsewhere (the index is keyed by
/// `Uuid`, so the caller joins matches back to payloads).
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Wrap an owned vector of components.
    ///
    /// The embedding takes ownership of `vec` without copying, so this is
    /// the cheapest way to build one from a buffer that is already on hand.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::memory::vector::Embedding;
    ///
    /// let embedding = Embedding::new(vec![0.5, 0.25]);
    /// assert_eq!(embedding.as_slice(), &[0.5, 0.25]);
    /// assert_eq!(embedding.dim(), 2);
    /// ```
    #[must_use]
    pub fn new(vec: Vec<f32>) -> Self {
        Self(vec)
    }

    /// Copy a slice into a fresh embedding.
    ///
    /// The components are cloned, so the new [`Embedding`] owns its storage
    /// and never aliases `slice`. Convenient at call sites that already
    /// hold the components as a fixed-size array or borrowed buffer.
    #[must_use]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self(slice.to_vec())
    }

    /// Dimensionality — the number of components.
    ///
    /// Every embedding an [`EmbeddingProvider`] produces has the
    /// dimensionality its [`dim`](EmbeddingProvider::dim) reports; a
    /// [`VectorIndex`] compares this count against its own and rejects
    /// mismatched vectors at [`add`](VectorIndex::add) time.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// The raw components, for distance computation.
    ///
    /// Exposed read-only so similarity functions such as
    /// [`cosine_similarity`] can score against the vector without copying
    /// it. There is deliberately no mutable accessor — an indexed
    /// embedding is only ever replaced wholesale via
    /// [`VectorIndex::add`].
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns a value in `-1.0..=1.0` (higher = more similar), clamped so
/// float overshoot cannot escape the range. Returns `0.0` when either
/// vector is zero-length, the lengths differ, or any component is not
/// finite — the index guards the length case at `add` time, so in normal
/// use that path is unreachable and the fallback is purely defensive.
///
/// Robust to component magnitude: both inputs are rescaled by the
/// largest absolute component before accumulation, so extreme values
/// cannot overflow the products to infinity and tiny values cannot
/// underflow the norms to zero — a plain dot-product implementation
/// returns NaN or a false `0.0` in exactly those cases. Non-finite
/// components (a model emitting NaN or infinity) are treated as no
/// similarity at all for the same reason: they would poison the result
/// to NaN and, through the sort, let a garbage vector outrank a real
/// match.
///
/// Made `pub` so a custom [`VectorIndex`] implementation (or any
/// approximate-index companion) can score candidates with the same metric
/// the reference index uses.
///
/// # Example
///
/// ```
/// use loopctl::memory::vector::cosine_similarity;
///
/// assert!((cosine_similarity(&[3.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
/// assert_eq!(cosine_similarity(&[f32::NAN, 1.0], &[1.0, 0.5]), 0.0);
/// ```
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut all_finite = true;
    let mut scale = 0.0_f32;
    for component in a.iter().chain(b.iter()) {
        if !component.is_finite() {
            all_finite = false;
        }
        scale = scale.max(component.abs());
    }
    if !all_finite || scale == 0.0 {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (x / scale, y / scale);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        (dot / denominator).clamp(-1.0, 1.0)
    }
}

/// A single nearest-neighbour match from [`VectorIndex::search`].
///
/// `score` is a cosine similarity in `-1.0..=1.0` — higher is closer — and
/// [`VectorIndex::search`] returns matches sorted by descending score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorMatch {
    /// The `Uuid` the vector was added under.
    ///
    /// The same id the caller passed to
    /// [`VectorIndex::add`]; a memory store joins it back to the
    /// originating payload.
    pub id: Uuid,

    /// Cosine similarity to the query, clamped to `-1.0..=1.0`.
    ///
    /// Higher is closer; `1.0` means parallel, `0.0` orthogonal or
    /// unmatched, `-1.0` opposed. Search results are sorted by this
    /// field descending.
    pub score: f32,
}

/// Turns text into a fixed-dimensional [`Embedding`].
///
/// Implementations wrap an embedding model — a hosted API (OpenAI
/// `text-embedding-3-*`, Cohere `embed-v*`), a local ONNX or
/// sentence-transformers model, or, for tests, the bundled
/// [`HashingEmbedder`]. All implementations report the same
/// [`dim`](EmbeddingProvider::dim) for every embedding they produce; a
/// [`VectorIndex`] is constructed against that dimensionality.
///
/// # Async shape
///
/// The async methods return boxed futures (`Pin<Box<dyn Future .. + Send +
/// '_>>`), exactly like [`LoopMemory`](super::LoopMemory) — which is what
/// makes the traits object-safe: a store can hold a
/// `Box<dyn EmbeddingProvider>` / `Box<dyn VectorIndex>` and swap backends
/// without touching its own contract. The `'_` bound lets an
/// implementation borrow its arguments instead of cloning them.
///
/// # Errors
///
/// Returns [`LoopError`]. A hosted-model failure typically maps to
/// `LoopError::Api`; an index or wiring failure maps to
/// [`LoopError::Memory`]. The trait does
/// not mandate which — the implementation picks the variant that fits, since
/// an embedder need not be HTTP-backed.
pub trait EmbeddingProvider: Send + Sync {
    /// The dimensionality of every embedding this provider produces.
    ///
    /// The value is fixed for the provider's lifetime and matches
    /// [`Embedding::dim`] on every embedding it returns; construct the
    /// receiving [`VectorIndex`] against this same number.
    #[must_use]
    fn dim(&self) -> usize;

    /// Embed a single piece of text.
    ///
    /// The returned future may borrow `text`, so implementations can
    /// build request bodies without cloning the input.
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, LoopError>> + Send + 'a>>;

    /// Embed many texts.
    ///
    /// The default runs [`embed`](EmbeddingProvider::embed) once per text,
    /// sequentially, preserving input order. Override it when the backing
    /// model has a genuine batch endpoint — a batched call is materially
    /// cheaper per text on most hosted APIs. Trivial implementations (and
    /// test stubs) get batching for free. Like every method here, the
    /// default owns its inputs before pinning the future (the
    /// [`LoopMemory`](super::LoopMemory) idiom), so implementations that
    /// override it may borrow or consume their inputs freely.
    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Embedding>, LoopError>> + Send + '_>> {
        let owned: Vec<String> = texts.iter().map(|text| (*text).to_string()).collect();
        Box::pin(async move {
            let mut out = Vec::with_capacity(owned.len());
            for text in &owned {
                out.push(self.embed(text).await?);
            }
            Ok(out)
        })
    }
}

/// A nearest-neighbour store of [`Embedding`]s keyed by `Uuid`.
///
/// Implementations range from the bundled brute-force
/// [`LinearVectorIndex`] (O(n) cosine scan, fine up to a few thousand
/// vectors) to an approximate index (HNSW) to a remote vector database.
/// The trait is the seam a semantic memory store holds as
/// `Box<dyn VectorIndex>`, so a memory store can swap its index without
/// touching its own contract.
///
/// # Keying
///
/// Keys are `Uuid` — the same type as
/// [`MemoryEntry::id`](super::MemoryEntry::id) — so a memory store joins
/// index matches back to payloads with no conversion.
///
/// # Scoring contract
///
/// [`search`](VectorIndex::search) returns [`VectorMatch`]es whose `score`
/// is cosine similarity in `-1.0..=1.0`, higher = closer, sorted
/// descending. An implementation whose native metric is L2 must convert
/// before returning (rescoring the candidate set with
/// [`cosine_similarity`] is the cheapest correct form).
///
/// # Thread safety
///
/// `Send + Sync`; mutators take `&self` and use interior mutability, so an
/// index is shareable via `Arc<dyn VectorIndex>` — matching
/// [`LoopMemory`](super::LoopMemory).
pub trait VectorIndex: Send + Sync {
    /// The dimensionality every stored vector must have.
    ///
    /// Fixed at construction; [`add`](VectorIndex::add) rejects vectors
    /// whose [`dim`](Embedding::dim) differs with [`LoopError::Memory`].
    #[must_use]
    fn dim(&self) -> usize;

    /// Insert (or replace) `vector` under `id`.
    ///
    /// Upsert semantics: if `id` is already present its vector is
    /// overwritten in place and [`len`](VectorIndex::len) is unchanged; if
    /// not, a new entry is added and `len` grows by one. This lets a memory
    /// store re-embed an updated [`MemoryEntry`](super::MemoryEntry) under
    /// the same id without leaking stale vectors.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when
    /// `vector.dim() != self.dim()`.
    fn add(
        &self,
        id: Uuid,
        vector: Embedding,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>;

    /// Return the `k` nearest vectors to `query`, most-similar first.
    ///
    /// Returns up to `k` matches — fewer when the store holds fewer
    /// vectors, zero when it is empty.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the
    /// query's dimensionality differs from the index's (a wrong-dimensional
    /// vector cannot be scored).
    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<VectorMatch>, LoopError>> + Send + '_>>;

    /// Remove the vector stored under `id`, if any.
    ///
    /// Idempotent: removing a missing id is `Ok(())` (a no-op), not an
    /// error — matching set semantics and keeping consolidation loops
    /// simple.
    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>>;

    /// Number of vectors currently stored (after dedup by id).
    ///
    /// Replacing a vector through [`add`](VectorIndex::add) leaves the
    /// count unchanged, so `len` reports distinct ids, not cumulative
    /// inserts.
    #[must_use]
    fn len(&self) -> usize;

    /// Whether the index holds no vectors.
    ///
    /// The provided default derives the answer from
    /// [`len`](VectorIndex::len); implementations rarely need to override
    /// it.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Brute-force [`VectorIndex`] — a flat list scanned with cosine similarity.
///
/// The reference implementation: every operation is O(n) — `add` runs a
/// linear upsert scan under the write lock, `search` and `remove` scan
/// the whole list.
/// Fine for tests, prototypes, and single-session agents up to a few
/// thousand vectors. **Not suitable for production at scale** — every
/// `search` re-scores the whole list. Use it as the correctness oracle a
/// faster index is tested against, and as the zero-dependency fallback.
///
/// # Thread safety
///
/// `Send + Sync`. Interior mutability via an internal [`RwLock`], so `add`
/// and `remove` take only `&self` and the index is shareable via `Arc`.
///
/// # Example
///
/// ```
/// use loopctl::memory::vector::{Embedding, LinearVectorIndex, VectorIndex};
/// use uuid::Uuid;
///
/// futures::executor::block_on(async {
///     let index = LinearVectorIndex::new(3);
///     index
///         .add(Uuid::new_v4(), Embedding::from_slice(&[0.1, 0.9, 0.0]))
///         .await
///         .unwrap();
///     let hits = index
///         .search(&Embedding::from_slice(&[0.0, 1.0, 0.0]), 5)
///         .await
///         .unwrap();
///     println!("{} match(es)", hits.len());
/// });
/// ```
pub struct LinearVectorIndex {
    /// Dimensionality every stored vector must have; fixed at
    /// construction and checked on `add` and `search`.
    dim: usize,

    /// The stored vectors under their ids, in insertion order.
    ///
    /// Guarded by an internal `RwLock` (see the type docs); `search`
    /// copies matches out under the read lock and sorts the copy, so no
    /// lock is held across sorting or awaiting.
    rows: RwLock<Vec<(Uuid, Embedding)>>,
}

impl LinearVectorIndex {
    /// Create an empty index that only accepts `dim`-dimensional vectors.
    ///
    /// The dimension is fixed for the index's lifetime: later
    /// [`add`](VectorIndex::add) or [`search`](VectorIndex::search) calls
    /// with a different dimensionality fail with [`LoopError::Memory`].
    /// Match it to the [`dim`](EmbeddingProvider::dim) of the embedder
    /// feeding the index.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: RwLock::new(Vec::new()),
        }
    }
}

impl VectorIndex for LinearVectorIndex {
    fn dim(&self) -> usize {
        self.dim
    }

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
            let mut rows = self.rows.write().unwrap_or_else(PoisonError::into_inner);
            match rows.iter_mut().find(|(existing, _)| *existing == id) {
                Some(slot) => slot.1 = vector,
                None => rows.push((id, vector)),
            }
            Ok(())
        })
    }

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
            let started = Instant::now();
            let rows = self.rows.read().unwrap_or_else(PoisonError::into_inner);
            let mut scored: Vec<VectorMatch> = rows
                .iter()
                .map(|(id, vector)| VectorMatch {
                    id: *id,
                    score: cosine_similarity(&query, vector.as_slice()),
                })
                .collect();
            drop(rows);
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            let matches: Vec<VectorMatch> = scored.into_iter().take(k).collect();
            let top_score = matches.first().map_or(0.0, |m| m.score);
            tracing::debug!(
                target: "loopctl::metrics",
                span = "vector.index.search",
                k,
                returned = matches.len(),
                top_score = %top_score,
                duration_ms = %started.elapsed().as_millis(),
                "vector index search complete"
            );
            Ok(matches)
        })
    }

    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let mut rows = self.rows.write().unwrap_or_else(PoisonError::into_inner);
            rows.retain(|(existing, _)| *existing != id);
            Ok(())
        })
    }

    fn len(&self) -> usize {
        self.rows
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Deterministic, dependency-free [`EmbeddingProvider`] for tests and
/// demos.
///
/// Maps text to a fixed-size vector by hashing each whitespace token into
/// a bucket and folding a `+1`/`-1` sign into it, then L2-normalising.
/// **Retrieval quality is unsuitable for real use** — the mapping has no
/// semantics beyond "identical text produces an identical vector, and
/// shared tokens produce non-zero cosine". It exists so the traits and the
/// index are exercisable with no network, no API key, and no model
/// download: two texts that share tokens score higher under cosine than
/// two texts that share none, which is enough to assert "the relevant
/// memory was retrieved first" in tests and to demo the index end-to-end.
///
/// The exact hash is unspecified and must not be relied upon across
/// versions — assert relative ordering and self-consistency, never a fixed
/// bit pattern. Zero-token input yields the zero vector, which
/// [`cosine_similarity`] scores `0.0` against everything; that is
/// documented behavior, not an error.
///
/// # Example
///
/// ```
/// use loopctl::memory::vector::{EmbeddingProvider, HashingEmbedder};
///
/// let embedder = HashingEmbedder::new(64);
/// let embedding = embedder.embed_sync("prefer glob over manual search");
/// assert_eq!(embedding.dim(), 64);
/// ```
pub struct HashingEmbedder {
    /// The dimensionality of every vector produced; each token hashes
    /// into one of `dim` buckets, so small values fold many tokens onto
    /// the same component.
    dim: usize,
}

impl HashingEmbedder {
    /// Create a hasher producing `dim`-dimensional vectors.
    ///
    /// Each token hashes into one of `dim` buckets, so very small values
    /// fold many tokens onto the same component and blur the embedding;
    /// pick a size comfortably larger than the vocabulary of the texts
    /// under test, and use the same `dim` for the index that will store
    /// the embeddings.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Embed `text` synchronously.
    ///
    /// The hashing is pure computation, so this needs no executor; the
    /// trait's async [`embed`](EmbeddingProvider::embed) wraps it. Each
    /// whitespace token is hashed into a bucket with a `+1`/`-1` sign, and
    /// the result is L2-normalised — unless the fold produced the zero
    /// vector (no tokens, or every sign cancelled), which is returned
    /// as-is.
    #[must_use]
    pub fn embed_sync(&self, text: &str) -> Embedding {
        let mut vector = vec![0.0_f32; self.dim];
        for token in text.split_whitespace() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&token, &mut hasher);
            let token_hash = std::hash::Hasher::finish(&hasher);
            let bucket = u64::checked_rem(token_hash, self.dim as u64)
                .and_then(|slot| usize::try_from(slot).ok())
                .unwrap_or(0);
            let sign = if token_hash & (1_u64 << 63) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            if let Some(slot) = vector.get_mut(bucket) {
                *slot += sign;
            }
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        Embedding::new(vector)
    }
}

impl EmbeddingProvider for HashingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, LoopError>> + Send + 'a>> {
        let started = Instant::now();
        Box::pin(async move {
            let embedding = self.embed_sync(text);
            tracing::debug!(
                target: "loopctl::metrics",
                span = "vector.embed",
                provider = "hashing",
                model = "hashing",
                dim = self.dim,
                inputs = 1,
                duration_ms = %started.elapsed().as_millis(),
                "embedding complete"
            );
            Ok(embedding)
        })
    }
}
