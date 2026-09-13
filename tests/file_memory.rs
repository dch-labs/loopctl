//! `FileMemoryStore` integration suite — persistence, restart-survival,
//! corruption recovery, atomic rewrites, and ranking parity with the
//! in-memory store.
//!
//! Run: `cargo test --features file_memory --test file_memory -- --nocapture`
//!
//! Requires the `file_memory` feature. No network; every test works in a
//! fresh temp directory it creates and removes itself.

#![cfg(feature = "file_memory")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use loopctl::memory::builtin::InMemoryStore;
use loopctl::memory::{FileMemoryStore, LoopMemory, MemoryCategory, MemoryEntry};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static DIR_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Create a fresh temp directory unique to this test invocation.
///
/// Uniqueness comes from the test process's id plus a per-process
/// counter, so parallel test binaries never share a directory.
fn temp_dir(tag: &str) -> PathBuf {
    let seq = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "loopctl-file-memory-{}-{tag}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn stored_entries_round_trip_through_retrieve() {
    let dir = temp_dir("round-trip");
    let store = FileMemoryStore::open(dir.join("memory.jsonl")).unwrap();

    for text in [
        "prefer Glob for file search",
        "read the whole file before editing",
        "file paths must be absolute",
    ] {
        store
            .store(MemoryEntry::new(MemoryCategory::Insight, text))
            .await
            .unwrap();
    }

    assert_eq!(store.len(), 3, "every stored entry is held");
    let hits = store.retrieve("file", 5).await.unwrap();
    assert_eq!(hits.len(), 3, "all three entries match the query word");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn memory_survives_a_process_restart() {
    let dir = temp_dir("restart");
    let path = dir.join("memory.jsonl");

    let store = FileMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Strategy, "verify before editing a file");
    entry.tags.push("editing".to_string());
    entry.relevance = 0.8;
    store.store(entry.clone()).await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(reopened.len(), 1, "the entry came back from disk");
    let hits = reopened.retrieve("verify editing", 3).await.unwrap();
    let returned = hits.first().expect("the reopened entry is retrievable");
    assert_eq!(returned.id, entry.id, "identity survives the round trip");
    assert_eq!(
        returned.memory, entry.memory,
        "text survives the round trip"
    );
    assert_eq!(returned.category, entry.category, "category survives");
    assert_eq!(returned.tags, entry.tags, "tags survive");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_missing_or_empty_file_opens_as_an_empty_store() {
    let dir = temp_dir("empty");
    let fresh = FileMemoryStore::open(dir.join("absent.jsonl")).unwrap();
    assert!(fresh.is_empty(), "a missing path opens empty");

    let empty_path = dir.join("empty.jsonl");
    std::fs::write(&empty_path, b"").unwrap();
    let empty = FileMemoryStore::open(&empty_path).unwrap();
    assert!(empty.is_empty(), "a zero-byte file opens empty");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn consolidate_prunes_and_rewrites_the_file() {
    let dir = temp_dir("consolidate");
    let path = dir.join("memory.jsonl");

    let store = FileMemoryStore::open(&path).unwrap();
    let mut keep = MemoryEntry::new(MemoryCategory::Fact, "a durable lesson");
    keep.relevance = 0.9;
    let mut drop_entry = MemoryEntry::new(MemoryCategory::Fact, "a spent lesson");
    drop_entry.relevance = 0.01;
    store.store(keep.clone()).await.unwrap();
    store.store(drop_entry).await.unwrap();

    let stats = store.consolidate().await.unwrap();
    assert_eq!(stats.pruned, 1, "the below-floor entry is pruned");
    assert_eq!(store.len(), 1, "only the durable lesson remains");
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "the rewrite removed the pruned entry from disk"
    );
    assert_eq!(
        reopened
            .retrieve("durable lesson", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(keep.id),
        "the survivor is the durable lesson"
    );
    assert!(
        !dir.join("memory.jsonl.tmp").try_exists().unwrap(),
        "a successful rewrite leaves no temp file behind"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_failed_rewrite_leaves_the_original_file_intact() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = temp_dir("atomic");
    let path = dir.join("memory.jsonl");

    let store = FileMemoryStore::open(&path).unwrap();
    let entry = MemoryEntry::new(MemoryCategory::Insight, "must survive the failed rewrite");
    store.store(entry.clone()).await.unwrap();

    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o500);
    std::fs::set_permissions(&dir, perms).unwrap();
    let flush_result = store.flush();
    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(&dir, perms).unwrap();

    assert!(
        flush_result.is_err(),
        "the rewrite cannot create its temp file"
    );
    assert!(
        !dir.join("memory.jsonl.tmp").try_exists().unwrap(),
        "the failed rewrite cleaned up its temp file"
    );
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(reopened.len(), 1, "the original file is untouched");
    assert_eq!(
        reopened
            .retrieve("survive", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(entry.id),
        "the entry is intact"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_torn_final_line_is_dropped_on_open() {
    let dir = temp_dir("torn");
    let path = dir.join("memory.jsonl");

    let entry = MemoryEntry::new(MemoryCategory::Fact, "the whole line that made it");
    let mut contents = serde_json::to_string(&entry).unwrap();
    contents.push('\n');
    contents.push_str("{\"id\":\"torn");
    std::fs::write(&path, contents).unwrap();

    let store = FileMemoryStore::open(&path).unwrap();
    assert_eq!(store.len(), 1, "the torn trailing fragment is skipped");
    assert_eq!(
        store
            .retrieve("whole line", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(entry.id),
        "the intact line loads"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_torn_tail_is_repaired_so_later_stores_survive() {
    let dir = temp_dir("torn-repair");
    let path = dir.join("memory.jsonl");

    let first = MemoryEntry::new(MemoryCategory::Fact, "the whole line that made it");
    let mut contents = serde_json::to_string(&first).unwrap();
    contents.push('\n');
    contents.push_str("{\"id\":\"torn");
    std::fs::write(&path, contents).unwrap();

    let store = FileMemoryStore::open(&path).unwrap();
    assert_eq!(store.len(), 1, "the torn trailing fragment is skipped");
    let second = MemoryEntry::new(MemoryCategory::Fact, "stored after the repair");
    store.store(second.clone()).await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the repair removed the fragment, so the next append is its own line and \
        both entries survive the reopen"
    );
    assert_eq!(
        reopened
            .retrieve("whole line", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(first.id),
        "the pre-crash entry survives"
    );
    assert_eq!(
        reopened
            .retrieve("after the repair", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(second.id),
        "the post-repair entry survives"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn concurrent_consolidations_leave_a_reopenable_store() {
    let dir = temp_dir("concurrent-consolidate");
    let path = dir.join("memory.jsonl");
    let store = Arc::new(FileMemoryStore::open(&path).unwrap());

    let mut durable = MemoryEntry::new(MemoryCategory::Fact, "nightly deploys pause the world");
    durable.relevance = 0.9;
    let mut durable_two =
        MemoryEntry::new(MemoryCategory::Fact, "grip large files with both hands");
    durable_two.relevance = 0.9;
    store.store(durable).await.unwrap();
    store.store(durable_two).await.unwrap();

    let first = Arc::clone(&store);
    let second = Arc::clone(&store);
    let (a, b) = tokio::join!(first.consolidate(), second.consolidate());
    a.unwrap();
    b.unwrap();
    assert_eq!(store.len(), 2, "both durable entries survive the pass");
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the serialized rewrites leave a clean file — no interleaved temp content"
    );
    assert!(
        !dir.join("memory.jsonl.tmp").try_exists().unwrap(),
        "no temp file is left behind"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn mid_file_corruption_fails_the_open() {
    let dir = temp_dir("corrupt");
    let path = dir.join("memory.jsonl");

    let entry = MemoryEntry::new(MemoryCategory::Fact, "a valid entry");
    let mut contents = String::from("this line is not json\n");
    contents.push_str(&serde_json::to_string(&entry).unwrap());
    contents.push('\n');
    std::fs::write(&path, contents).unwrap();

    let result = FileMemoryStore::open(&path);
    assert!(
        matches!(result, Err(loopctl::error::LoopError::Memory(_))),
        "corruption before the last line is surfaced, not skipped"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn concurrent_stores_never_interleave_lines() {
    const WRITERS: usize = 8;
    let dir = temp_dir("concurrent");
    let path = dir.join("memory.jsonl");
    let store = Arc::new(FileMemoryStore::open(&path).unwrap());

    let mut tasks = Vec::new();
    for index in 0..WRITERS {
        let store = Arc::clone(&store);
        tasks.push(tokio::task::spawn(async move {
            store
                .store(MemoryEntry::new(
                    MemoryCategory::Fact,
                    format!("writer {index} stored a whole line"),
                ))
                .await
                .unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }

    assert_eq!(store.len(), WRITERS, "every writer's entry is held");
    drop(store);
    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        WRITERS,
        "the file holds one parseable line per writer — no interleaving"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn ranking_matches_the_in_memory_store() {
    let dir = temp_dir("parity");
    let file_store = FileMemoryStore::open(dir.join("memory.jsonl")).unwrap();
    let memory_store = InMemoryStore::new();

    let mut first = MemoryEntry::new(MemoryCategory::Insight, "deploy checklist nightly");
    first.relevance = 0.4;
    let mut second = MemoryEntry::new(MemoryCategory::Fact, "deploy scripts live in ops");
    second.relevance = 0.9;
    second.tags.push("deploy".to_string());
    let mut third = MemoryEntry::new(MemoryCategory::ErrorPattern, "unrelated rust compiler note");
    third.relevance = 1.0;
    for entry in [first, second, third] {
        file_store.store(entry.clone()).await.unwrap();
        memory_store.store(entry).await.unwrap();
    }

    let from_file = file_store.retrieve("deploy", 3).await.unwrap();
    let from_memory = memory_store.retrieve("deploy", 3).await.unwrap();
    let file_ids: Vec<_> = from_file.iter().map(|e| e.id).collect();
    let memory_ids: Vec<_> = from_memory.iter().map(|e| e.id).collect();
    assert_eq!(
        file_ids, memory_ids,
        "both stores rank the same entry set identically for the same query"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
