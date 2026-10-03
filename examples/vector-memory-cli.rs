//! Vector-memory demo: store a small fixture through
//! `VectorMemoryStore` — the semantic `LoopMemory` backend — with the
//! dependency-free `HashingEmbedder` and `LinearVectorIndex`, then run
//! semantic and keyword queries through the hybrid blend. No LLM, no
//! network, no API key.
//!
//! Run: `cargo run --features vector_memory --example vector-memory-cli`

use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
use loopctl::memory::vector_memory::VectorMemoryStore;
use loopctl::memory::{LoopMemory, MemoryCategory, MemoryEntry};

fn main() -> Result<(), loopctl::error::LoopError> {
    futures::executor::block_on(demo())
}

/// # Errors
///
/// Propagates any [`LoopError`] from storing, retrieving, or consolidating
/// — this demo surfaces store errors instead of swallowing them.
async fn demo() -> Result<(), loopctl::error::LoopError> {
    let store = VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(128)),
        Box::new(LinearVectorIndex::new(128)),
    );

    let corpus = [
        (
            MemoryCategory::Strategy,
            "prefer glob over manual file search when listing many paths",
        ),
        (
            MemoryCategory::Fact,
            "the retry ladder backs off exponentially on 429 responses",
        ),
        (
            MemoryCategory::Insight,
            "hash tokens into buckets to build a deterministic test embedding",
        ),
        (
            MemoryCategory::Fact,
            "compaction triggers when the context window crosses its threshold",
        ),
        (
            MemoryCategory::Fact,
            "cosine similarity ranks vectors by angle, not magnitude",
        ),
        (
            MemoryCategory::Strategy,
            "memoize repeated tool calls and evict the cache on writes",
        ),
    ];

    for (category, text) in &corpus {
        store.store(MemoryEntry::new(*category, *text)).await?;
    }
    println!("stored {} entries\n", corpus.len());

    let queries = [
        "how do I find files quickly",
        "what happens when the provider rate-limits me",
        "summarize the oldest turns to fit the window",
    ];
    for query in queries {
        println!("query: {query}");
        for (rank, hit) in store.retrieve(query, 2).await?.iter().enumerate() {
            println!(
                "  {}. relevance {:.2} — {}",
                rank.saturating_add(1),
                hit.relevance,
                hit.memory
            );
        }
        println!();
    }

    let stats = store.consolidate().await?;
    println!(
        "consolidated: {} -> {} entries (pruned {}, merged {}, {} bytes of vectors reclaimed)",
        stats.entries_before, stats.entries_after, stats.pruned, stats.merged, stats.bytes_saved
    );
    Ok(())
}
