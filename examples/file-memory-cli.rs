//! `FileMemoryStore` demo — learned memory that survives a restart.
//!
//! Stores three entries, drops the store (a stand-in for the process
//! exiting), reopens the same file, and retrieves: the entries written
//! before the drop come back.
//!
//! ```sh
//! cargo run --example file-memory-cli --features file_memory
//! ```

#![allow(clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use loopctl::memory::FileMemoryStore;
use loopctl::memory::{LoopMemory, MemoryCategory, MemoryEntry};

fn main() {
    let dir = std::env::temp_dir().join("loopctl-file-memory-cli");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("agent-memory.jsonl");
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let store = Arc::new(FileMemoryStore::open(&path).expect("store opens"));
        for text in [
            "prefer Glob over manual file listing",
            "read the whole file before editing it",
            "verify the saved diff compiles",
        ] {
            store
                .store(MemoryEntry::new(MemoryCategory::Insight, text))
                .await
                .expect("store");
        }
        println!("stored {} entries; dropping the store", store.len());
        drop(store);

        let reopened = FileMemoryStore::open(&path).expect("reopen");
        println!("reopened with {} entries:", reopened.len());
        for entry in reopened.retrieve("file", 3).await.expect("retrieve") {
            println!("  - {}", entry.memory);
        }
    });

    std::fs::remove_dir_all(&dir).expect("cleanup");
}
