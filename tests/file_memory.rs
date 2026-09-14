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

#[cfg(unix)]
#[tokio::test]
async fn access_stamps_survive_a_failed_consolidation() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = temp_dir("stamps-survive");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "grip large files with both hands");
    entry.relevance = 0.9;
    store.store(entry.clone()).await.unwrap();
    store.retrieve("grip large files", 3).await.unwrap();

    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o500);
    std::fs::set_permissions(&dir, perms).unwrap();
    let failed = store.consolidate().await;
    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(&dir, perms).unwrap();
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

#[cfg(unix)]
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

#[cfg(unix)]
#[tokio::test]
async fn an_unopenable_parent_reports_unconfirmed_durability_after_the_rename() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = temp_dir("dir-sync");
    let path = dir.join("memory.jsonl");
    let store = FileMemoryStore::open(&path).unwrap();
    let entry = MemoryEntry::new(MemoryCategory::Fact, "checkpointed before the sync failed");
    store.store(entry.clone()).await.unwrap();

    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o300);
    std::fs::set_permissions(&dir, perms).unwrap();
    let flush_result = store.flush();
    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(&dir, perms).unwrap();

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
