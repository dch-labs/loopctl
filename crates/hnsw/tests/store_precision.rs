//! Golden-set precision with the HNSW backend — the drop-in proof with no
//! feature flag on `loopctl`: the companion crate's `HnswIndex` wires into
//! `VectorMemoryStore` exactly the way a consumer would, and the shared
//! fixture applies the same ≥ 80% precision bar the linear index meets.
//!
//! Run: `cargo test -p loopctl-hnsw --test store_precision`

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects,
    clippy::float_cmp
)]

use loopctl::memory::LoopMemory;
use loopctl::memory::vector::HashingEmbedder;
use loopctl::memory::vector_memory::{GoldenSet, VectorMemoryStore, golden_set};
use loopctl_hnsw::HnswIndex;

#[tokio::test]
async fn golden_set_precision_survives_the_hnsw_backend() {
    let set = golden_set();
    let store = VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(128)),
        Box::new(HnswIndex::new(128)),
    );
    for fixture_entry in &set.entries {
        store.store(fixture_entry.clone()).await.unwrap();
    }
    let mut hits: Vec<bool> = Vec::with_capacity(set.queries.len());
    for (index, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(index);
        let returned = store.retrieve(query.text, 3).await.unwrap();
        hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let precision = GoldenSet::precision_ratio(&hits);
    assert!(
        precision >= 0.8,
        "golden-set precision {precision:.2} over the HNSW backend is below \
         the 80% gate; missed queries: {:?}",
        hits.iter()
            .zip(set.queries.iter())
            .filter(|(hit, _)| !**hit)
            .map(|(_, query)| query.text)
            .collect::<Vec<_>>()
    );
}
