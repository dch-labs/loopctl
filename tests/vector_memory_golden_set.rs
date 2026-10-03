//! The golden-set precision gate and fixture characterization for
//! [`VectorMemoryStore`](loopctl::memory::vector_memory::VectorMemoryStore).
//!
//! Split from `tests/vector_memory.rs` because the shared fixture is
//! `testing`-gated in the library while the store suite is not: gating the
//! whole store file on both features would hide every store contract test
//! from a `--features vector_memory`-only run — exactly the feature-alone
//! signal that caught the suite's one-time failure to compile without
//! `testing`.

#![cfg(all(feature = "vector_memory", feature = "testing"))]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects
)]

use loopctl::memory::LoopMemory;
use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
use loopctl::memory::vector_memory::{GoldenSet, VectorMemoryStore, golden_set};

const DIM: usize = 128;

fn store() -> VectorMemoryStore {
    VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(LinearVectorIndex::new(DIM)),
    )
}

#[test]
fn the_golden_fixture_spans_four_of_the_six_categories() {
    let set = golden_set();
    assert_eq!(
        set.entries.len(),
        50,
        "the fixture holds fifty entries — the documented size"
    );
    let mut categories: Vec<String> = Vec::new();
    for fixture_entry in &set.entries {
        let name = format!("{:?}", fixture_entry.category);
        if !categories.contains(&name) {
            categories.push(name);
        }
    }
    categories.sort_unstable();
    assert_eq!(
        categories,
        vec![
            "ErrorPattern".to_string(),
            "Fact".to_string(),
            "Insight".to_string(),
            "Strategy".to_string(),
        ],
        "the fixture spans exactly the four documented categories — widen \
         the fixture and the GoldenSet doc together"
    );
}

#[tokio::test]
async fn golden_set_queries_hit_a_relevant_memory_eighty_percent_of_the_time() {
    let set = golden_set();
    let store = store();
    for fixture_entry in &set.entries {
        store.store(fixture_entry.clone()).await.unwrap();
    }
    assert_eq!(store.len(), set.entries.len(), "every fixture entry stored");
    let mut hits: Vec<bool> = Vec::with_capacity(set.queries.len());
    for (index, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(index);
        let returned = store.retrieve(query.text, 3).await.unwrap();
        hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let precision = GoldenSet::precision_ratio(&hits);
    assert!(
        precision >= 0.8,
        "semantic precision {precision:.2} is below the 80% acceptance gate; \
         missed queries: {:?}",
        hits.iter()
            .zip(set.queries.iter())
            .filter(|(hit, _)| !**hit)
            .map(|(_, query)| query.text)
            .collect::<Vec<_>>()
    );
}
