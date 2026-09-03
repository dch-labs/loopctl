//! Vector-index demo: embed a few strings with the dependency-free
//! `HashingEmbedder`, index them, and run a ranked nearest-neighbour
//! search — no LLM, no network, no API key.
//!
//! Run: `cargo run --features vector_index --example vector-index-demo`

use loopctl::memory::vector::{Embedding, HashingEmbedder, LinearVectorIndex, VectorIndex};
use uuid::Uuid;

fn main() -> Result<(), loopctl::error::LoopError> {
    let embedder = HashingEmbedder::new(64);
    let index = LinearVectorIndex::new(64);

    let corpus = [
        "prefer glob over manual search when listing files",
        "the retry ladder backs off exponentially on 429 responses",
        "hash tokens into buckets to build a test embedding",
        "compaction triggers when the context window crosses the threshold",
        "cosine similarity ranks vectors by angle, not magnitude",
    ];

    let mut ids = Vec::new();
    for (position, text) in corpus.iter().enumerate() {
        let id = Uuid::new_v4();
        let embedding: Embedding = embedder.embed_sync(text);
        futures::executor::block_on(index.add(id, embedding))?;
        println!("{position}: {id} <- {text}");
        ids.push((id, text));
    }

    let query = "how do I embed text for a similarity search?";
    let query_embedding = embedder.embed_sync(query);
    let matches = futures::executor::block_on(index.search(&query_embedding, 3))?;

    println!("\nquery: {query}\n");
    for (rank, hit) in matches.iter().enumerate() {
        let source = ids
            .iter()
            .find(|(id, _)| *id == hit.id)
            .map_or("unknown", |(_, text)| *text);
        println!(
            "{}. score {:.3} — {source}",
            rank.saturating_add(1),
            hit.score
        );
    }

    Ok(())
}
