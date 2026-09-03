//! Vector primitives: `EmbeddingProvider`, `VectorIndex`, and the
//! reference `HashingEmbedder` / `LinearVectorIndex` implementations.
//!
//! Run: `cargo test --features vector_index --test vector_index -- --nocapture`
//!
//! Requires the `vector_index` feature. No network, no API keys — every
//! test uses the hashing embedder and the linear index only.

#![cfg(feature = "vector_index")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc
)]

use loopctl::error::LoopError;
use loopctl::memory::vector::{
    Embedding, EmbeddingProvider, HashingEmbedder, LinearVectorIndex, VectorIndex, VectorMatch,
    cosine_similarity,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

fn block_on<F: Future>(future: F) -> F::Output {
    futures::executor::block_on(future)
}

// A deterministic linear-congruential generator: seeded so the oracle
// test's vectors are reproducible without a `rand` dependency.
struct Seeded(u64);

impl Seeded {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / (1_u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Embedding {
        Embedding::from_slice(&(0..dim).map(|_| self.next_f32()).collect::<Vec<_>>())
    }
}

#[test]
fn hashing_embedder_is_deterministic_and_honours_its_dim() {
    let embedder = HashingEmbedder::new(64);
    let first = embedder.embed_sync("prefer glob over manual search");
    let second = embedder.embed_sync("prefer glob over manual search");
    assert_eq!(first, second, "identical text embeds identically");
    assert_eq!(first.dim(), 64);

    for dim in [8_usize, 64, 256] {
        assert_eq!(
            HashingEmbedder::new(dim)
                .embed_sync("anything at all")
                .dim(),
            dim,
            "dim() matches the constructor value"
        );
    }
}

#[test]
fn shared_tokens_score_higher_than_disjoint_tokens() {
    let embedder = HashingEmbedder::new(64);
    let query = embedder.embed_sync("prefer glob over manual search");
    let overlapping = embedder.embed_sync("always prefer glob for file search");
    let disjoint = embedder.embed_sync("compaction shrinks the context window");

    let to_overlapping = cosine_similarity(query.as_slice(), overlapping.as_slice());
    let to_disjoint = cosine_similarity(query.as_slice(), disjoint.as_slice());
    assert!(
        to_overlapping > to_disjoint,
        "shared tokens score higher: {to_overlapping} vs {to_disjoint}"
    );
}

#[test]
fn cosine_similarity_basics() {
    assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    assert!(
        (cosine_similarity(&[3.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6,
        "cosine is magnitude-invariant: 3x parallel still scores 1.0"
    );
    assert!(
        (cosine_similarity(&[0.0, 2.0], &[0.0, 5.0]) - 1.0).abs() < 1e-6,
        "different nonzero magnitudes of parallel vectors still score 1.0"
    );
    assert_eq!(
        cosine_similarity(&[f32::NAN, 1.0], &[1.0, 0.5]),
        0.0,
        "a NaN component cannot poison the similarity"
    );
    assert_eq!(
        cosine_similarity(&[f32::INFINITY, 0.0], &[1.0, 0.0]),
        0.0,
        "an infinite component cannot poison the similarity"
    );
    assert!(
        (cosine_similarity(&[-1.0, 0.0], &[-1.0, 0.0]) - 1.0).abs() < 1e-6,
        "the negative orthant scales by absolute magnitude: parallel negatives score 1.0"
    );
    assert!(
        (cosine_similarity(&[-1.0, -1.0], &[1.0, 1.0]) + 1.0).abs() < 1e-6,
        "opposite-sign vectors of equal shape score -1.0"
    );
    assert_eq!(cosine_similarity(&[], &[1.0]), 0.0, "zero-length is 0.0");
    assert_eq!(
        cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]),
        0.0,
        "length mismatch is a defensive 0.0"
    );
}

/// A provider that keeps the trait's default `embed_batch`.
struct DefaultBatchProvider {
    dim: usize,
}

impl EmbeddingProvider for DefaultBatchProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, LoopError>> + Send + '_>> {
        let embedding = Embedding::from_slice(&[text.len() as f32, 1.0]);
        Box::pin(std::future::ready(Ok(embedding)))
    }
}

/// A provider whose `embed_batch` override counts its invocations.
struct CountingBatchProvider {
    batch_calls: AtomicUsize,
}

impl EmbeddingProvider for CountingBatchProvider {
    fn dim(&self) -> usize {
        2
    }

    fn embed(
        &self,
        text: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, LoopError>> + Send + '_>> {
        let embedding = Embedding::from_slice(&[text.len() as f32, 1.0]);
        Box::pin(std::future::ready(Ok(embedding)))
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Embedding>, LoopError>> + Send + '_>> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        let embeddings = texts
            .iter()
            .map(|text| Embedding::from_slice(&[-1.0, text.len() as f32]))
            .collect();
        Box::pin(std::future::ready(Ok(embeddings)))
    }
}

#[test]
fn embed_batch_default_matches_the_embed_loop() {
    let provider = DefaultBatchProvider { dim: 2 };
    let batched = block_on(provider.embed_batch(&["a", "bb", "ccc"])).expect("default batch");
    let sequential: Vec<Embedding> = ["a", "bb", "ccc"]
        .iter()
        .map(|text| block_on(provider.embed(text)).expect("single embed"))
        .collect();
    assert_eq!(batched, sequential, "the default batches by looping embed");
}

#[test]
fn embed_batch_override_is_honoured() {
    let provider = CountingBatchProvider {
        batch_calls: AtomicUsize::new(0),
    };
    let batched = block_on(provider.embed_batch(&["a", "bb"])).expect("override batch");
    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0].as_slice()[0], -1.0, "the override's own output");
    assert_eq!(
        provider.batch_calls.load(Ordering::Relaxed),
        1,
        "the override is called, not shadowed by the default"
    );
}

#[tokio::test]
async fn add_len_and_is_empty_track_the_index() {
    let index = LinearVectorIndex::new(4);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.dim(), 4, "dim matches the constructor");

    for id in [Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()] {
        index
            .add(id, Embedding::from_slice(&[0.0; 4]))
            .await
            .expect("dimension matches");
    }
    assert_eq!(index.len(), 3);
    assert!(!index.is_empty());
}

#[tokio::test]
async fn add_dimension_mismatch_is_a_memory_error() {
    let index = LinearVectorIndex::new(4);
    let err = index
        .add(Uuid::new_v4(), Embedding::from_slice(&[0.0; 3]))
        .await
        .expect_err("a 3-dim vector cannot enter a 4-dim index");
    match err {
        LoopError::Memory(message) => {
            assert!(
                message.contains('4') && message.contains('3'),
                "the error names both dimensions: {message}"
            );
        }
        other => panic!("expected Memory, got {other:?}"),
    }
}

#[tokio::test]
async fn search_ranks_by_cosine_similarity() {
    let index = LinearVectorIndex::new(2);
    let id_a = Uuid::new_v4();
    let id_b = Uuid::new_v4();
    let id_c = Uuid::new_v4();
    index
        .add(id_a, Embedding::from_slice(&[1.0, 0.0]))
        .await
        .unwrap();
    index
        .add(id_b, Embedding::from_slice(&[0.8, 0.6]))
        .await
        .unwrap();
    index
        .add(id_c, Embedding::from_slice(&[0.0, 1.0]))
        .await
        .unwrap();

    let hits: Vec<VectorMatch> = index
        .search(&Embedding::from_slice(&[1.0, 0.0]), 3)
        .await
        .unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![id_a, id_b, id_c],
        "most-similar first"
    );
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert!((hits[1].score - 0.8).abs() < 1e-6);
    assert!(hits[2].score.abs() < 1e-6);
}

#[tokio::test]
async fn search_respects_k() {
    let index = LinearVectorIndex::new(2);
    for _ in 0..10 {
        index
            .add(Uuid::new_v4(), Embedding::from_slice(&[1.0, 0.0]))
            .await
            .unwrap();
    }
    assert_eq!(
        index
            .search(&Embedding::from_slice(&[1.0, 0.0]), 3)
            .await
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        index
            .search(&Embedding::from_slice(&[1.0, 0.0]), 0)
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        index
            .search(&Embedding::from_slice(&[1.0, 0.0]), 100)
            .await
            .unwrap()
            .len(),
        10,
        "k above the store size returns everything, not an error"
    );
}

#[tokio::test]
async fn search_on_empty_index_and_mismatched_query() {
    let index = LinearVectorIndex::new(4);
    let hits = index
        .search(&Embedding::from_slice(&[0.0; 4]), 5)
        .await
        .unwrap();
    assert!(hits.is_empty(), "an empty index searches to empty");

    let err = index
        .search(&Embedding::from_slice(&[0.0; 3]), 5)
        .await
        .expect_err("a wrong-dim query cannot be scored");
    assert!(matches!(err, LoopError::Memory(_)));
}

#[tokio::test]
async fn add_is_an_upsert() {
    let index = LinearVectorIndex::new(2);
    let id = Uuid::new_v4();
    index
        .add(id, Embedding::from_slice(&[1.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(index.len(), 1);

    index
        .add(id, Embedding::from_slice(&[0.0, 1.0]))
        .await
        .unwrap();
    assert_eq!(index.len(), 1, "the same id replaces in place");

    let hits = index
        .search(&Embedding::from_slice(&[0.0, 1.0]), 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6,
        "the replacement vector is what is stored"
    );
}

#[tokio::test]
async fn remove_is_idempotent_and_effective() {
    let index = LinearVectorIndex::new(2);
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    index
        .add(id1, Embedding::from_slice(&[1.0, 0.0]))
        .await
        .unwrap();
    index
        .add(id2, Embedding::from_slice(&[0.0, 1.0]))
        .await
        .unwrap();

    index.remove(id1).await.expect("removal succeeds");
    assert_eq!(index.len(), 1);
    let hits = index
        .search(&Embedding::from_slice(&[1.0, 0.0]), 5)
        .await
        .unwrap();
    assert!(
        hits.iter().all(|hit| hit.id != id1),
        "the removed id is no longer returned"
    );

    index.remove(id1).await.expect("re-removal is a no-op");
    index
        .remove(Uuid::new_v4())
        .await
        .expect("never-added is a no-op");
    assert_eq!(index.len(), 1);
}

#[tokio::test]
async fn linear_index_matches_a_brute_force_oracle() {
    let mut seeded = Seeded(0x5EED_5EED_5EED_5EED);
    let dim = 8;
    let index = LinearVectorIndex::new(dim);
    let mut corpus: Vec<(Uuid, Embedding)> = Vec::new();
    for _ in 0..50 {
        let id = Uuid::new_v4();
        let vector = seeded.vector(dim);
        index.add(id, vector.clone()).await.unwrap();
        corpus.push((id, vector));
    }

    for _ in 0..5 {
        let query = seeded.vector(dim);
        let mut expected: Vec<VectorMatch> = corpus
            .iter()
            .map(|(id, vector)| VectorMatch {
                id: *id,
                score: cosine_similarity(query.as_slice(), vector.as_slice()),
            })
            .collect();
        expected.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        let expected_ids: Vec<Uuid> = expected.into_iter().take(5).map(|hit| hit.id).collect();

        let hits = index.search(&query, 5).await.unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            expected_ids,
            "the linear index is its own correctness oracle on seeded vectors"
        );
    }
}

#[tokio::test]
async fn zero_text_embeds_to_a_handled_zero_vector() {
    let embedder = HashingEmbedder::new(16);
    let embedding = embedder.embed_sync("");
    assert_eq!(embedding.dim(), 16);
    assert!(
        embedding
            .as_slice()
            .iter()
            .all(|component| component == &0.0),
        "zero-token input yields the zero vector"
    );
    assert_eq!(
        cosine_similarity(embedding.as_slice(), &[0.5; 16]),
        0.0,
        "the zero vector scores 0.0 against everything"
    );

    let index = LinearVectorIndex::new(16);
    let id = Uuid::new_v4();
    index.add(id, embedding).await.expect("dim matches");
    let hits = index
        .search(&Embedding::from_slice(&[0.0; 16]), 5)
        .await
        .unwrap();
    assert!(
        hits.iter().all(|hit| hit.score == 0.0),
        "the stored zero vector scores 0.0, not a panic"
    );
}

#[tokio::test]
async fn a_non_finite_vector_never_outranks_a_finite_match() {
    let index = LinearVectorIndex::new(2);
    let garbage = Uuid::nil();
    let good = Uuid::new_v4();
    // The provider emitted a NaN component for `garbage` and a perfect
    // match for `good`; the NaN vector must score 0.0, not poison the
    // ranking into putting garbage first.
    index
        .add(garbage, Embedding::from_slice(&[f32::NAN, 1.0]))
        .await
        .unwrap();
    index
        .add(good, Embedding::from_slice(&[1.0, 0.5]))
        .await
        .unwrap();

    let hits = index
        .search(&Embedding::from_slice(&[1.0, 0.5]), 5)
        .await
        .unwrap();
    assert_eq!(
        hits.first().expect("the finite match is returned").id,
        good,
        "the perfect finite match outranks the non-finite vector"
    );
    assert!(
        (hits.first().expect("present").score - 1.0).abs() < 1e-6,
        "the finite match keeps its perfect score"
    );
}

#[tokio::test]
async fn concurrent_adds_and_searches_keep_the_index_consistent() {
    let index = Arc::new(LinearVectorIndex::new(8));
    let mut handles = Vec::new();
    for writer in 0..4_usize {
        let index = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            let mut seeded = Seeded(1_000 + writer as u64);
            for _ in 0..100 {
                let id = Uuid::new_v4();
                let vector = seeded.vector(8);
                block_on(index.add(id, vector)).expect("adds succeed under contention");
            }
        }));
    }
    for reader in 0..2_usize {
        let index = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            for _ in 0..50 {
                let query = Seeded(9_000 + reader as u64).vector(8);
                block_on(index.search(&query, 5)).expect("searches succeed under contention");
            }
        }));
    }
    for handle in handles {
        handle.join().expect("no task panics under contention");
    }
    assert_eq!(index.len(), 400, "every concurrent add landed exactly once");
}

#[test]
fn the_traits_are_object_safe() {
    // A semantic memory store holds `Box<dyn EmbeddingProvider>` and
    // `Box<dyn VectorIndex>`; this compile-time pin keeps the boxed-future
    // shape that makes both traits usable as trait objects.
    let _provider: Box<dyn EmbeddingProvider> = Box::new(HashingEmbedder::new(16));
    let _index: Box<dyn VectorIndex> = Box::new(LinearVectorIndex::new(16));
}
