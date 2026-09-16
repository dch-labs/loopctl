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

/// Assert no temp sibling remains in `dir` for any of its files.
///
/// Scans the directory rather than probing a literal name, so it stays
/// valid under the rewrite's unpredictable temp names.
fn assert_no_temp_siblings(dir: &std::path::Path, context: &str) {
    let leftovers: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .filter(|name| {
            std::path::Path::new(name)
                .extension()
                .is_some_and(|ext| ext == "tmp")
        })
        .collect();
    assert!(leftovers.is_empty(), "{context}: {leftovers:?}");
}

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
    assert_no_temp_siblings(&dir, "a successful rewrite leaves no temp sibling");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn a_failed_rewrite_leaves_the_original_file_intact() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("atomic");
    let path = dir.join("memory.jsonl");

    let store = FileMemoryStore::open(&path).unwrap();
    let entry = MemoryEntry::new(MemoryCategory::Insight, "must survive the failed rewrite");
    store.store(entry.clone()).await.unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::TempCreate);
    let flush_result = store.flush();

    assert!(
        flush_result.is_err(),
        "the rewrite cannot create its temp file"
    );
    assert_no_temp_siblings(&dir, "the failed rewrite cleaned up its temp");
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

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn a_fault_registered_before_creation_through_a_symlinked_parent_fires() {
    use loopctl::memory::file::RewriteFaultStage;
    use std::os::unix::fs::symlink;

    let dir = temp_dir("fault-key");
    let real = dir.join("real");
    std::fs::create_dir_all(&real).unwrap();
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let through_link = link.join("memory.jsonl");
    let store = FileMemoryStore::new(&through_link);

    loopctl::memory::file::fail_next_rewrite_at(&through_link, RewriteFaultStage::TempCreate);
    let flush_result = store.flush();

    assert!(
        flush_result.is_err(),
        "a registration keys exactly as the store keys the path it rewrites, \
        so the fault registered through a symlinked parent before the file \
        existed still strikes the rewrite"
    );
    assert!(
        !through_link.exists(),
        "the fault fires before the first rewrite creates anything through \
        the link"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn each_registration_injects_exactly_one_fault() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("fault-count");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::TempCreate);
    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::TempCreate);
    assert!(
        store.flush().is_err(),
        "the first registration strikes the first rewrite"
    );
    assert!(
        store.flush().is_err(),
        "the second registration strikes the second rewrite — one hit \
        consumes exactly one registration"
    );
    assert!(
        store.flush().is_ok(),
        "with both registrations consumed, the rewrite path is clean again"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn a_fault_for_a_later_stage_survives_an_earlier_stage_strike() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("fault-stages");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    store
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "must survive two staged injected faults",
        ))
        .await
        .unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::TempCreate);
    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::DirectorySync);
    assert!(
        store.flush().is_err(),
        "the TempCreate registration strikes the first rewrite before \
        anything is written"
    );
    assert!(
        store.flush().is_err(),
        "the DirectorySync registration survives the TempCreate strike and \
        fires at its own stage"
    );
    let rewritten = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        rewritten.lines().count(),
        1,
        "the second failure is post-rename — the entry is on disk and only \
        its durability is unconfirmed: {rewritten}"
    );
    assert!(
        store.flush().is_ok(),
        "with both registrations consumed, the rewrite path is clean again"
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
async fn a_valid_line_without_its_newline_is_repaired_on_open() {
    let dir = temp_dir("unterminated");
    let path = dir.join("memory.jsonl");

    let first = MemoryEntry::new(MemoryCategory::Fact, "survived the crash mid-newline");
    let contents = serde_json::to_string(&first).unwrap();
    std::fs::write(&path, contents).unwrap();

    let store = FileMemoryStore::open(&path).unwrap();
    assert_eq!(store.len(), 1, "the unterminated-but-valid line loads");
    let second = MemoryEntry::new(MemoryCategory::Fact, "stored after the repair");
    store.store(second.clone()).await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the repair appended the missing terminator, so the next append is \
        its own line — neither entry is lost to a welded line"
    );
    assert_eq!(
        reopened
            .retrieve("survived the crash", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(first.id),
        "the pre-crash entry survives"
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
async fn a_torn_utf8_fragment_is_dropped_and_repaired() {
    let dir = temp_dir("torn-utf8");
    let path = dir.join("memory.jsonl");

    let entry = MemoryEntry::new(MemoryCategory::Fact, "a whole multi-byte line survived");
    let mut contents = serde_json::to_string(&entry).unwrap().into_bytes();
    contents.push(b'\n');
    contents.extend_from_slice(b"{\"id\":\"torn \xFF\xFE");
    std::fs::write(&path, &contents).unwrap();

    let store = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        store.len(),
        1,
        "the invalid-UTF-8 tail fragment is dropped, the complete lines load"
    );
    let second = MemoryEntry::new(MemoryCategory::Fact, "stored after the utf8 repair");
    store.store(second.clone()).await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the repair removed the fragment before the append"
    );
    assert_eq!(
        reopened
            .retrieve("whole multi-byte", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(entry.id),
        "the pre-crash entry survives"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn invalid_utf8_in_a_complete_line_fails_the_open() {
    let dir = temp_dir("utf8-corrupt");
    let path = dir.join("memory.jsonl");

    let entry = MemoryEntry::new(MemoryCategory::Fact, "a valid entry");
    let mut contents = b"not json \xFF\n".to_vec();
    contents.extend_from_slice(serde_json::to_string(&entry).unwrap().as_bytes());
    contents.push(b'\n');
    std::fs::write(&path, &contents).unwrap();

    let result = FileMemoryStore::open(&path);
    assert!(
        matches!(result, Err(loopctl::error::LoopError::Memory(_))),
        "invalid UTF-8 in a complete (newline-terminated) line is corruption, not a torn tail"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn access_stamps_survive_a_failed_consolidation() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("stamps-survive");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "grip large files with both hands");
    entry.relevance = 0.9;
    store.store(entry.clone()).await.unwrap();
    store.retrieve("grip large files", 3).await.unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::TempCreate);
    let failed = store.consolidate().await;
    assert!(failed.is_err(), "the rewrite cannot create its temp file");

    store.consolidate().await.unwrap();
    let reloaded = store.retrieve("grip large files", 3).await.unwrap();
    let stamped = reloaded
        .first()
        .expect("the entry is retrievable after the successful pass");
    assert_eq!(
        stamped.access_count, 1,
        "the access stamp from before the failed pass was preserved and re-folded, not lost"
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
    assert_no_temp_siblings(&dir, "no temp sibling is left behind");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn an_unrecoverable_append_failure_marks_the_store_unusable() {
    let store = FileMemoryStore::new("/dev/full");
    let first = store
        .store(MemoryEntry::new(MemoryCategory::Fact, "first"))
        .await;
    assert!(
        first.is_err(),
        "writing to /dev/full always fails with ENOSPC"
    );
    let second = store
        .store(MemoryEntry::new(MemoryCategory::Fact, "second"))
        .await;
    let message = second.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        message.contains("unusable"),
        "when even the truncation fails, later appends are rejected until \
        the store is reopened: {message}"
    );
}

#[tokio::test]
async fn two_handles_on_one_path_lose_no_entries() {
    let dir = temp_dir("two-handles");
    let path = dir.join("memory.jsonl");

    let first = FileMemoryStore::open(&path).unwrap();
    let second = FileMemoryStore::open(&path).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    first.consolidate().await.unwrap();
    assert_eq!(first.len(), 2, "both handles see the shared mirror");
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the consolidate rewrote from the shared mirror — no handle's \
        entries are silently dropped from disk"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn concurrent_stores_through_two_handles_never_weld() {
    const WRITERS: usize = 8;
    let dir = temp_dir("two-handles-concurrent");
    let path = dir.join("memory.jsonl");
    let first = Arc::new(FileMemoryStore::open(&path).unwrap());
    let second = Arc::new(FileMemoryStore::open(&path).unwrap());

    let mut tasks = Vec::new();
    for index in 0..WRITERS {
        let store = if index % 2 == 0 {
            Arc::clone(&first)
        } else {
            Arc::clone(&second)
        };
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
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        WRITERS,
        "appends through both handles serialize behind the shared lock — \
        one parseable line per store, no welded lines"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unconverged_spellings_concurrent_stores_never_weld_a_line() {
    use std::os::unix::fs::symlink;

    const WRITERS_PER_SPELLING: usize = 8;
    let dir = temp_dir("two-spellings-concurrent");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = Arc::new(FileMemoryStore::new(link1.join("memory.jsonl")));
    let second = Arc::new(FileMemoryStore::new(link2.join("memory.jsonl")));

    std::fs::create_dir_all(&real).unwrap();
    let mut tasks = Vec::new();
    for index in 0..(WRITERS_PER_SPELLING * 2) {
        let store = if index % 2 == 0 {
            Arc::clone(&first)
        } else {
            Arc::clone(&second)
        };
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
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        WRITERS_PER_SPELLING * 2,
        "each store converges the spellings before appending, so every \
        writer serializes behind the one shared lock — one parseable \
        line per store, never a corrupt mid-file line"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_non_finite_relevance_is_refused_before_it_bricks_the_file() {
    let dir = temp_dir("nan-relevance");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let healthy = MemoryEntry::new(MemoryCategory::Fact, "alpha survives the poisoned sibling");
    store.store(healthy.clone()).await.unwrap();

    let mut poisoned = MemoryEntry::new(MemoryCategory::Fact, "relevance from a broken scorer");
    poisoned.relevance = f32::NAN;
    assert!(
        store.store(poisoned).await.is_err(),
        "a non-finite relevance is refused loudly at store time — \
        serialized it would be a null the loader treats as fatal \
        mid-file corruption"
    );
    let mut infinite = MemoryEntry::new(MemoryCategory::Fact, "infinite relevance");
    infinite.relevance = f32::INFINITY;
    assert!(
        store.store(infinite).await.is_err(),
        "infinity is the same class — refused before the append"
    );

    drop(store);
    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "the file reopens with its healthy entry — no null line ever \
        reached it"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn divergent_copies_under_one_id_survive_a_convergence_flush() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("duplicate-content-flush");
    let real = dir.join("realdir");
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let store = FileMemoryStore::new(link.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let mut first_copy = MemoryEntry::new(MemoryCategory::Fact, "alpha version one content");
    first_copy.relevance = 0.9;
    let mut second_copy = first_copy.clone();
    second_copy.memory = "alpha version two content".to_string();
    second_copy.relevance = 0.1;
    store.store(first_copy.clone()).await.unwrap();
    store.store(second_copy.clone()).await.unwrap();

    store.flush().unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    let hits = reopened.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.memory.clone(), hit.relevance))
            .collect::<Vec<_>>(),
        vec![
            ("alpha version one content".to_string(), 0.9),
            ("alpha version two content".to_string(), 0.1),
        ],
        "two divergent copies stored under one id keep both contents \
        in file order across a convergence rewrite — the count was \
        never the contract, the content is"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn divergent_copies_under_one_id_survive_a_unification() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("duplicate-content-unify");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let mut first_copy = MemoryEntry::new(MemoryCategory::Fact, "alpha version one content");
    first_copy.relevance = 0.9;
    let mut second_copy = first_copy.clone();
    second_copy.memory = "alpha version two content".to_string();
    second_copy.relevance = 0.1;
    first.store(first_copy.clone()).await.unwrap();
    second.store(second_copy.clone()).await.unwrap();

    let third = FileMemoryStore::open(link1.join("memory.jsonl")).unwrap();
    drop(first);
    drop(second);
    drop(third);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    let hits = reopened.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.memory.clone(), hit.relevance))
            .collect::<Vec<_>>(),
        vec![
            ("alpha version one content".to_string(), 0.9),
            ("alpha version two content".to_string(), 0.1),
        ],
        "a unification of two spellings preserves both contents under \
        one id — each copy pairs with its own file occurrence"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_provisional_consolidate_does_not_double_count_prior_accesses() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("fold-order");
    let real = dir.join("realdir");
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let store = FileMemoryStore::new(link.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "alpha accessed twice");
    entry.relevance = 0.9;
    store.store(entry).await.unwrap();

    store.retrieve("alpha", 3).await.unwrap();
    store.consolidate().await.unwrap();
    store.retrieve("alpha", 3).await.unwrap();
    store.consolidate().await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    let hits = reopened.retrieve("alpha", 3).await.unwrap();
    assert_eq!(hits.len(), 1, "one entry, not a durable and a stamped copy");
    assert_eq!(
        hits.first().map(|hit| hit.access_count),
        Some(2),
        "two retrieves mean two counted accesses — the stamped mirror \
        copy must pair with its durable occurrence before the fold \
        changes its metadata, or the pass merges and double-counts \
        the prior history"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_provisional_consolidate_with_merging_disabled_keeps_one_copy() {
    use loopctl::memory::consolidate::ConsolidationConfig;
    use std::os::unix::fs::symlink;

    let dir = temp_dir("fold-order-nomerge");
    let real = dir.join("realdir");
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let store =
        FileMemoryStore::new(link.join("memory.jsonl")).with_consolidation(ConsolidationConfig {
            merge: false,
            ..ConsolidationConfig::default()
        });

    std::fs::create_dir_all(&real).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "alpha never merged");
    entry.relevance = 0.9;
    store.store(entry).await.unwrap();

    store.retrieve("alpha", 3).await.unwrap();
    store.consolidate().await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "with clustering disabled there is nothing to fold the stray \
        stamped copy into — the pre-fold pairing is the only thing \
        keeping one entry on disk instead of a durable and a stamped \
        duplicate"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_new_store_discards_entries_already_in_the_file() {
    let dir = temp_dir("new-discards");
    let path = dir.join("memory.jsonl");
    let previous = FileMemoryStore::open(&path).unwrap();
    let old = MemoryEntry::new(MemoryCategory::Fact, "written by an earlier process");
    previous.store(old).await.unwrap();
    drop(previous);

    let replacement = FileMemoryStore::new(&path);
    let fresh = MemoryEntry::new(MemoryCategory::Fact, "written by the fresh store");
    replacement.store(fresh.clone()).await.unwrap();
    replacement.flush().unwrap();
    drop(replacement);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "a fresh new() neither loads nor preserves an existing file's \
        entries — the rewrite persists only what the live handles stored"
    );
    assert_eq!(
        reopened
            .retrieve("fresh", 3)
            .await
            .unwrap()
            .first()
            .map(|hit| hit.id),
        Some(fresh.id),
        "the surviving entry is the new store's, not the file's old one"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_custom_consolidation_config_shapes_the_pass() {
    use loopctl::memory::consolidate::ConsolidationConfig;

    let dir = temp_dir("config-floor");
    let default_store = FileMemoryStore::open(dir.join("default.jsonl")).unwrap();
    let tuned_store = FileMemoryStore::open(dir.join("tuned.jsonl"))
        .unwrap()
        .with_consolidation(ConsolidationConfig {
            prune_floor: 0.0,
            ..ConsolidationConfig::default()
        });
    let mut spent = MemoryEntry::new(MemoryCategory::Fact, "a spent lesson");
    spent.relevance = 0.01;
    default_store.store(spent.clone()).await.unwrap();
    tuned_store.store(spent).await.unwrap();
    default_store.consolidate().await.unwrap();
    tuned_store.consolidate().await.unwrap();

    assert_eq!(
        default_store.len(),
        0,
        "the default floor prunes the spent entry"
    );
    assert_eq!(
        tuned_store.len(),
        1,
        "the per-handle config shapes this handle's pass over the shared \
        mirror — a zero prune floor keeps what the default prunes"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn new_attaches_to_a_live_store_for_the_same_path() {
    let dir = temp_dir("new-attaches");
    let path = dir.join("memory.jsonl");

    let opened = FileMemoryStore::open(&path).unwrap();
    opened
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "stored through open",
        ))
        .await
        .unwrap();
    let attached = FileMemoryStore::new(&path);
    assert_eq!(
        attached.len(),
        1,
        "new attaches to the live shared state for the path instead of \
        starting a private empty mirror"
    );
    attached
        .store(MemoryEntry::new(MemoryCategory::Fact, "stored through new"))
        .await
        .unwrap();
    assert_eq!(
        opened.len(),
        2,
        "the open handle sees the new handle's entry"
    );
    drop(opened);
    drop(attached);

    let fresh = FileMemoryStore::new(dir.join("fresh.jsonl"));
    fresh
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "stored before any open",
        ))
        .await
        .unwrap();
    let reopened = FileMemoryStore::open(dir.join("fresh.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "a new-first store persists through its first store, and a later \
        open attaches to it"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn new_and_open_share_state_through_a_symlinked_directory() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("symlink-key");
    let real = dir.join("real");
    std::fs::create_dir_all(&real).unwrap();
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let through_link = link.join("memory.jsonl");
    let first = FileMemoryStore::new(&through_link);
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();
    let second = FileMemoryStore::open(&through_link).unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    first.consolidate().await.unwrap();
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "new and open derive the same registry key through a symlinked \
        parent, so both handles share state and no entry is lost — \
        verified by reopening through the real path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_link_handles_share_state_and_the_link_survives() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("dangling-link");
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let target = elsewhere.join("target.jsonl");
    let link = dir.join("mem-link.jsonl");
    symlink(&target, &link).unwrap();

    let first = FileMemoryStore::new(&link);
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();
    let second = FileMemoryStore::open(&link).unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    first.consolidate().await.unwrap();
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(&link).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "new and open agree on the dangling link's target, so both \
        handles share state and no entry is lost"
    );
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the consolidation rewrote the target, not the link — the symlink \
        itself survives"
    );
    assert!(
        target.is_file(),
        "the physical data lives at the target as a regular file"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_parent_handles_share_state_once_the_path_resolves() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("dangling-parent");
    let real = dir.join("realdir");
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let through_link = link.join("memory.jsonl");
    let first = FileMemoryStore::new(&through_link);

    std::fs::create_dir_all(&real).unwrap();
    let second = FileMemoryStore::open(&through_link).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();
    assert_eq!(
        first.len(),
        2,
        "the handle born while the parent link was still dangling \
        attaches to the same shared state as the handle constructed \
        after the path resolved"
    );
    assert_eq!(second.len(), 2, "both handles observe both entries");
    first.consolidate().await.unwrap();
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "one shared mirror means the consolidation persisted both \
        handles' entries — verified by reopening through the real path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn two_provisional_handles_unify_when_their_paths_resolve_to_one_file() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("unify");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let through_link1 = link1.join("memory.jsonl");
    let through_link2 = link2.join("memory.jsonl");
    let first = FileMemoryStore::new(&through_link1);
    let second = FileMemoryStore::new(&through_link2);

    std::fs::create_dir_all(&real).unwrap();
    let third = FileMemoryStore::open(&through_link1).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();
    third
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "read the whole file before editing",
        ))
        .await
        .unwrap();
    assert_eq!(
        first.len(),
        3,
        "the link1 handle observes the unified store, not a private \
        mirror"
    );
    assert_eq!(
        second.len(),
        3,
        "the link2 handle — whose registration collided — is re-pointed \
        at the same shared state as the other two"
    );
    assert_eq!(third.len(), 3, "the resolving handle shares it too");
    second.consolidate().await.unwrap();
    drop(first);
    drop(second);
    drop(third);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        3,
        "one unified mirror means the consolidation persisted every \
        handle's entry — verified by reopening through the real path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn mirrors_written_before_unification_merge_into_the_surviving_store() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("unify-merge");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let through_link1 = link1.join("memory.jsonl");
    let through_link2 = link2.join("memory.jsonl");
    let first = FileMemoryStore::new(&through_link1);
    let second = FileMemoryStore::new(&through_link2);

    std::fs::create_dir_all(&real).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();

    let third = FileMemoryStore::open(&through_link1).unwrap();
    assert_eq!(
        first.len(),
        2,
        "entries stored through each pre-resolution spelling reach the \
        unified mirror"
    );
    assert_eq!(second.len(), 2, "the collided handle observes the union");
    assert_eq!(third.len(), 2, "the resolving handle observes the union");
    third.consolidate().await.unwrap();
    drop(first);
    drop(second);
    drop(third);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the union survived the consolidation — no handle's entry was \
        discarded by the rewrite"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn an_existing_handle_converges_before_it_rewrites() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("converge-consolidate");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();

    first.consolidate().await.unwrap();
    assert_eq!(
        second.len(),
        2,
        "the consolidation converged the registry first — the stale \
        side is re-pointed at the unified mirror"
    );
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the consolidation rewrote from the unified mirror — no \
        construction ran between the stores and the rewrite, and both \
        handles' entries persist through the real path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_flush_on_a_converged_handle_rewrites_from_the_unified_mirror() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("converge-flush");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "nightly deploys pause the world",
        ))
        .await
        .unwrap();

    first.flush().unwrap();
    assert_eq!(
        second.len(),
        2,
        "the flush converged the registry first — the stale side is \
        re-pointed at the unified mirror"
    );
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "the flush rewrote from the unified mirror — both handles' \
        entries persist through the real path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn convergence_preserves_the_backing_files_append_order() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("converge-order");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let a1 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha grip large files with both hands",
    );
    let b = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha nightly deploys pause the world",
    );
    let a2 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha read the whole file before editing",
    );
    first.store(a1.clone()).await.unwrap();
    second.store(b.clone()).await.unwrap();
    first.store(a2.clone()).await.unwrap();

    let third = FileMemoryStore::open(link1.join("memory.jsonl")).unwrap();
    let hits = third.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![a1.id, b.id, a2.id],
        "convergence resets the merged mirror against the file's append \
        order — the interleaved stores rule out either side winning the \
        collision, so retrieval's insertion-order tie-break matches the \
        file under any assignment"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_convergence_then_consolidation_preserves_the_files_append_order() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("converge-order-rewrite");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let a1 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha grip large files with both hands",
    );
    let b = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha nightly deploys pause the world",
    );
    let a2 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha read the whole file before editing",
    );
    first.store(a1.clone()).await.unwrap();
    second.store(b.clone()).await.unwrap();
    first.store(a2.clone()).await.unwrap();

    first.consolidate().await.unwrap();
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    let hits = reopened.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![a1.id, b.id, a2.id],
        "the converged rewrite keeps the file's append order durable — \
        the consolidation rewrote from the file-ordered merged mirror, \
        so the reopened store delivers the original sequence"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_dropped_sibling_s_entries_survive_convergence() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("converge-adopt");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    let link3 = dir.join("link3");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    symlink(&real, &link3).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));
    let third = FileMemoryStore::new(link3.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let a1 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha grip large files with both hands",
    );
    let b = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha nightly deploys pause the world",
    );
    let c = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha read the whole file before editing",
    );
    first.store(a1.clone()).await.unwrap();
    second.store(b.clone()).await.unwrap();
    third.store(c.clone()).await.unwrap();
    drop(third);

    first.flush().unwrap();
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    let hits = reopened.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![a1.id, b.id, c.id],
        "the convergence rewrite adopts the dropped sibling's entries \
        from the file — the file is the truth for content, not just \
        order, so every append survives in file order"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_flush_from_a_lone_provisional_spelling_adopts_the_file() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("lone-move-adopt");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let a1 = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha grip large files with both hands",
    );
    let x = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha nightly deploys pause the world",
    );
    first.store(a1.clone()).await.unwrap();
    second.store(x.clone()).await.unwrap();
    drop(second);

    first.flush().unwrap();
    drop(first);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "a rewrite from a store that was ever provisionally keyed \
        re-syncs against the file first — the dropped sibling's \
        durable append is adopted, not stranded on the replaced inode"
    );
    let hits = reopened.retrieve("alpha", 5).await.unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![a1.id, x.id],
        "the adopted entries land in the file's append order"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_store_through_a_second_spelling_converges_before_it_appends() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("store-converges");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let first = FileMemoryStore::new(link1.join("memory.jsonl"));
    let second = FileMemoryStore::new(link2.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let a = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha stored through the first spelling",
    );
    let x = MemoryEntry::new(
        MemoryCategory::Fact,
        "alpha stored through the second spelling",
    );
    first.store(a.clone()).await.unwrap();
    second.store(x.clone()).await.unwrap();

    assert_eq!(
        first.len(),
        2,
        "the second store converges the spellings before appending — both \
        handles share one mirror from the first append on"
    );
    assert_eq!(
        second.len(),
        2,
        "the converging handle observes the union, not just its own appends"
    );
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened
            .retrieve("alpha", 5)
            .await
            .unwrap()
            .iter()
            .map(|hit| hit.id)
            .collect::<Vec<_>>(),
        vec![a.id, x.id],
        "the file holds both appends in the order they were stored"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_lone_spellings_flush_keeps_a_files_earlier_duplicate_content() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("lone-duplicate");
    let real = dir.join("realdir");
    let link = dir.join("link");
    symlink(&real, &link).unwrap();
    let store = FileMemoryStore::new(link.join("memory.jsonl"));

    std::fs::create_dir_all(&real).unwrap();
    let path = real.join("memory.jsonl");
    let mut first_copy = MemoryEntry::new(MemoryCategory::Fact, "alpha version one content");
    first_copy.relevance = 0.9;
    let mut second_copy = first_copy.clone();
    second_copy.memory = "alpha version two content".to_string();
    second_copy.relevance = 0.1;
    std::fs::write(
        &path,
        format!("{}\n", serde_json::to_string(&first_copy).unwrap()).as_bytes(),
    )
    .unwrap();

    store.store(second_copy.clone()).await.unwrap();
    store.flush().unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened
            .retrieve("alpha", 5)
            .await
            .unwrap()
            .iter()
            .map(|hit| (hit.memory.clone(), hit.relevance))
            .collect::<Vec<_>>(),
        vec![
            ("alpha version one content".to_string(), 0.9),
            ("alpha version two content".to_string(), 0.1),
        ],
        "a duplicate occurrence without an equal mirror copy adopts its \
        own durable content — the re-sync cannot swap a divergent copy \
        into an occurrence it did not come from"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stores_racing_a_flush_never_lose_an_append() {
    use std::os::unix::fs::symlink;

    const WRITERS: usize = 8;
    const FLUSHERS: usize = 2;
    let dir = temp_dir("store-vs-flush");
    let real = dir.join("realdir");
    let link1 = dir.join("link1");
    let link2 = dir.join("link2");
    symlink(&real, &link1).unwrap();
    symlink(&real, &link2).unwrap();
    let writers = Arc::new(FileMemoryStore::new(link1.join("memory.jsonl")));
    let flushers = Arc::new(FileMemoryStore::new(link2.join("memory.jsonl")));
    std::fs::create_dir_all(&real).unwrap();

    let mut tasks = Vec::new();
    for index in 0..WRITERS {
        let store = Arc::clone(&writers);
        tasks.push(tokio::task::spawn(async move {
            store
                .store(MemoryEntry::new(
                    MemoryCategory::Fact,
                    format!("alpha writer {index} appended a whole line"),
                ))
                .await
                .unwrap();
        }));
    }
    for _ in 0..FLUSHERS {
        let store = Arc::clone(&flushers);
        tasks.push(tokio::task::spawn(async move {
            store.flush().unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    drop(writers);
    drop(flushers);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        WRITERS,
        "every append either precedes the flush's re-sync read — and is \
        adopted — or queues behind the rewrite on the shared lock; none \
        is stranded on the replaced inode"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn exhausted_chain_handles_share_state_once_it_resolves() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("chain-resolve");
    let real = dir.join("realdir");
    let mut head = dir.join("link-8.jsonl");
    symlink(real.join("memory.jsonl"), &head).unwrap();
    for step in (0..8).rev() {
        let link = dir.join(format!("link-{step}.jsonl"));
        symlink(&head, &link).unwrap();
        head = link;
    }
    let first = FileMemoryStore::new(&head);

    std::fs::create_dir_all(&real).unwrap();
    let second = FileMemoryStore::open(&head).unwrap();
    first
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "alpha grip large files with both hands",
        ))
        .await
        .unwrap();
    second
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "alpha nightly deploys pause the world",
        ))
        .await
        .unwrap();
    assert_eq!(
        first.len(),
        2,
        "the chain-head handle attaches to the resolved store — an \
        exhausted chain's provisional key re-derives once the chain \
        resolves"
    );
    assert_eq!(second.len(), 2, "the resolving handle shares it");
    drop(first);
    drop(second);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened.len(),
        2,
        "every entry persists through the real path — the appends went \
        through the chain to one file"
    );
    reopened.consolidate().await.unwrap();
    drop(reopened);
    let after = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        after.len(),
        2,
        "a rewrite through the real path — the only spelling whose final \
        component is not a symlink — keeps both entries"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_rewrite_refuses_to_replace_a_final_symlink() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("chain-refuse");
    let real = dir.join("realdir");
    let mut head = dir.join("link-8.jsonl");
    symlink(real.join("memory.jsonl"), &head).unwrap();
    for step in (0..8).rev() {
        let link = dir.join(format!("link-{step}.jsonl"));
        symlink(&head, &link).unwrap();
        head = link;
    }
    let store = FileMemoryStore::new(&head);

    let flush_result = store.flush();
    let message = flush_result
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        message.contains("refusing to replace the link"),
        "the rewrite must refuse to rename over a final-component \
        symlink: {message}"
    );
    assert!(
        std::fs::symlink_metadata(&head).is_ok_and(|metadata| metadata.file_type().is_symlink()),
        "the head of the chain is still a symlink — the refusal left it \
        untouched"
    );
    assert_no_temp_siblings(&dir, "the refused rewrite cleaned up its temp");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_live_exhausted_handle_checkpoints_once_the_chain_resolves() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("chain-live");
    let real = dir.join("realdir");
    let mut head = dir.join("link-8.jsonl");
    symlink(real.join("memory.jsonl"), &head).unwrap();
    for step in (0..8).rev() {
        let link = dir.join(format!("link-{step}.jsonl"));
        symlink(&head, &link).unwrap();
        head = link;
    }
    let store = FileMemoryStore::new(&head);

    std::fs::create_dir_all(&real).unwrap();
    let entry = MemoryEntry::new(MemoryCategory::Fact, "checkpointed through the chain head");
    store.store(entry.clone()).await.unwrap();

    store.flush().unwrap();
    store.consolidate().await.unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(real.join("memory.jsonl")).unwrap();
    assert_eq!(
        reopened
            .retrieve("checkpointed", 3)
            .await
            .unwrap()
            .first()
            .map(|hit| hit.id),
        Some(entry.id),
        "the live handle checkpointed through the converged path — the \
        chain-head spelling a provisional registration moved off of no \
        longer reaches the rewrite"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn a_planted_symlink_at_a_predictable_temp_path_is_not_followed() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir("temp-symlink");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    store
        .store(MemoryEntry::new(MemoryCategory::Fact, "legitimate entry"))
        .await
        .unwrap();

    let victim = dir.join("victim.txt");
    std::fs::write(&victim, "precious contents").unwrap();
    symlink(&victim, dir.join("memory.jsonl.tmp")).unwrap();

    store.flush().unwrap();
    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "precious contents",
        "the rewrite's exclusive, unpredictable temp create never follows a \
        planted symlink — the victim is untouched"
    );
    assert_eq!(
        store.len(),
        1,
        "the store itself is unaffected by the planted link"
    );
    drop(std::fs::remove_file(dir.join("memory.jsonl.tmp")));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn an_unopenable_parent_reports_unconfirmed_durability_after_the_rename() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("dir-sync");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let entry = MemoryEntry::new(MemoryCategory::Fact, "checkpointed before the sync failed");
    store.store(entry.clone()).await.unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::DirectorySync);
    let flush_result = store.flush();

    let error = flush_result.expect_err("the directory cannot be opened for the sync");
    assert!(
        error.to_string().contains("cannot sync directory"),
        "the failure is the post-rename directory sync: {error}"
    );
    assert_eq!(
        store.len(),
        1,
        "the rename itself already happened — the new file is in place"
    );
    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened
            .retrieve("checkpointed", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(entry.id),
        "the checkpoint content is the post-rename state — only its \
        durability is unconfirmed"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(all(unix, feature = "testing"))]
#[tokio::test]
async fn a_durability_unconfirmed_consolidation_still_commits_the_pass() {
    use loopctl::memory::file::RewriteFaultStage;

    let dir = temp_dir("commit-on-not-durable");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let mut durable = MemoryEntry::new(MemoryCategory::Fact, "nightly deploys pause the world");
    durable.relevance = 0.9;
    let mut spent = MemoryEntry::new(MemoryCategory::Fact, "a spent lesson");
    spent.relevance = 0.01;
    store.store(durable.clone()).await.unwrap();
    store.store(spent).await.unwrap();

    loopctl::memory::file::fail_next_rewrite_at(&path, RewriteFaultStage::DirectorySync);
    let result = store.consolidate().await;
    let error = result.expect_err("the directory sync after the rename fails");
    assert!(
        error.to_string().contains("cannot sync directory"),
        "the failure is the durability-unconfirmed class: {error}"
    );

    assert_eq!(
        store.len(),
        1,
        "the pass committed to the live mirror despite the error — the \
        rename had already landed"
    );
    store.flush().unwrap();
    drop(store);

    let reopened = FileMemoryStore::open(&path).unwrap();
    assert_eq!(
        reopened.len(),
        1,
        "a later flush from the committed mirror does not resurrect the \
        pruned entry on disk"
    );
    assert_eq!(
        reopened
            .retrieve("nightly deploys", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(durable.id),
        "the survivor is the durable entry"
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
async fn a_foreign_infinite_relevance_line_fails_the_open() {
    let dir = temp_dir("ingress-inf");
    let path = dir.join("memory.jsonl");

    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "a foreign hand edit");
    entry.relevance = 0.5;
    let mut value = serde_json::to_value(&entry).unwrap();
    value["relevance"] = serde_json::json!(1e40_f64);
    let spliced = format!("{}\n", serde_json::to_string(&value).unwrap());
    std::fs::write(&path, spliced.as_bytes()).unwrap();

    let result = FileMemoryStore::open(&path);
    let message = result
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        message.contains("non-finite relevance") || message.contains("out of range"),
        "a valid-JSON number that overflows f32 must never reach the \
        mirror — the loader refuses it at the open, whichever layer \
        catches it (the store's own guard, or serde_json's range \
        check under dependency configurations that enable it), or \
        the first flush bricks the reopen: {message}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        spliced.as_bytes(),
        "the refused open leaves the foreign bytes untouched for the \
        operator to repair"
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
