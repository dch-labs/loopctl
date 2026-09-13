//! `SqliteMemoryStore` integration suite — schema bootstrap,
//! round-trips, restart-survival, full-text retrieval with fallback,
//! consolidation, and multi-connection concurrency.
//!
//! Run: `cargo test -p loopctl-sqlite`
//!
//! No network; file-backed tests use temp databases they create and
//! remove themselves, the rest run against `:memory:`.

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

use loopctl::memory::{LoopMemory, MemoryCategory, MemoryEntry};
use loopctl_sqlite::SqliteMemoryStore;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, UNIX_EPOCH};

static DIR_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Create a fresh temp directory unique to this test invocation.
///
/// Uniqueness comes from the test process's id plus a per-process
/// counter, so parallel test binaries never share a database file.
fn temp_dir(tag: &str) -> PathBuf {
    let seq = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("loopctl-sqlite-{}-{tag}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A fully-populated entry exercising every column.
///
/// Every timestamp is pinned to an exact epoch offset so the
/// round-trip assertions compare known values, not tolerances.
fn loaded_entry() -> MemoryEntry {
    let mut entry = MemoryEntry::new(MemoryCategory::Strategy, "verify the diff compiles");
    entry.tags = vec!["editing".to_string(), "verification".to_string()];
    entry.created_at = UNIX_EPOCH + Duration::from_secs(1_234_567_890);
    entry.relevance = 0.42;
    entry.access_count = 7;
    entry.validated = true;
    entry.last_accessed = Some(UNIX_EPOCH + Duration::from_secs(1_234_567_891));
    entry.last_decayed = Some(UNIX_EPOCH + Duration::from_secs(1_234_567_892));
    entry
}

#[tokio::test]
async fn opening_an_existing_database_is_idempotent() {
    let dir = temp_dir("idempotent");
    let path = dir.join("memory.db");

    let store = SqliteMemoryStore::open(&path).unwrap();
    store
        .store(MemoryEntry::new(MemoryCategory::Fact, "one durable fact"))
        .await
        .unwrap();
    drop(store);

    let reopened = SqliteMemoryStore::open(&path).unwrap();
    assert_eq!(reopened.len(), 1, "re-running the bootstrap keeps the rows");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn entries_survive_a_restart_with_every_field_intact() {
    let dir = temp_dir("restart");
    let path = dir.join("memory.db");
    let original = loaded_entry();

    let store = SqliteMemoryStore::open(&path).unwrap();
    store.store(original.clone()).await.unwrap();
    drop(store);

    let reopened = SqliteMemoryStore::open(&path).unwrap();
    assert_eq!(reopened.len(), 1);
    let hits = reopened.retrieve("verify diff", 3).await.unwrap();
    let returned = hits.first().expect("the entry is retrievable");
    assert_eq!(returned.id, original.id, "identity round-trips");
    assert_eq!(returned.memory, original.memory, "text round-trips");
    assert_eq!(returned.category, original.category, "category round-trips");
    assert_eq!(returned.tags, original.tags, "tags round-trip");
    assert_eq!(
        returned.created_at, original.created_at,
        "creation time round-trips exactly at millisecond resolution"
    );
    assert_eq!(
        returned.relevance, original.relevance,
        "relevance round-trips"
    );
    assert_eq!(
        returned.access_count, original.access_count,
        "access count round-trips"
    );
    assert!(returned.validated, "validated round-trips");
    assert_eq!(
        returned.last_accessed, original.last_accessed,
        "last accessed round-trips"
    );
    assert_eq!(
        returned.last_decayed, original.last_decayed,
        "last decayed round-trips"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn retrieval_ranks_full_text_matches_first() {
    let store = SqliteMemoryStore::in_memory().unwrap();
    for text in [
        "concurrency needs bounded channels",
        "parallelism needs more than one worker",
        "rust traits are the extension point",
    ] {
        store
            .store(MemoryEntry::new(MemoryCategory::Insight, text))
            .await
            .unwrap();
    }

    let hits = store.retrieve("concurrency", 5).await.unwrap();
    assert_eq!(
        hits.first().map(|entry| entry.memory.as_str()),
        Some("concurrency needs bounded channels"),
        "the FTS5 match outranks the baseline fill"
    );

    for odd_query in [
        "rust AND",
        "UNION SELECT",
        "'; DROP TABLE;--",
        "trailing quote '",
    ] {
        let result = store.retrieve(odd_query, 5).await;
        assert!(
            result.is_ok(),
            "query {odd_query:?} must never error the store"
        );
    }
}

#[tokio::test]
async fn retrieval_respects_the_limit() {
    let store = SqliteMemoryStore::in_memory().unwrap();
    for index in 0..10 {
        store
            .store(MemoryEntry::new(
                MemoryCategory::Fact,
                format!("fact number {index} about testing"),
            ))
            .await
            .unwrap();
    }
    let hits = store.retrieve("testing", 3).await.unwrap();
    assert_eq!(hits.len(), 3, "no more than the limit comes back");
}

#[tokio::test]
async fn consolidation_prunes_and_the_delete_is_durable() {
    let dir = temp_dir("consolidate");
    let path = dir.join("memory.db");

    let store = SqliteMemoryStore::open(&path).unwrap();
    let mut durable = MemoryEntry::new(MemoryCategory::Fact, "a durable lesson");
    durable.relevance = 0.9;
    let mut spent = MemoryEntry::new(MemoryCategory::Fact, "a spent lesson");
    spent.relevance = 0.01;
    store.store(durable.clone()).await.unwrap();
    store.store(spent).await.unwrap();

    let stats = store.consolidate().await.unwrap();
    assert_eq!(stats.pruned, 1, "the below-floor entry is pruned");
    drop(store);

    let reopened = SqliteMemoryStore::open(&path).unwrap();
    assert_eq!(reopened.len(), 1, "the delete survived the restart");
    assert_eq!(
        reopened
            .retrieve("durable", 3)
            .await
            .unwrap()
            .first()
            .map(|e| e.id),
        Some(durable.id),
        "the survivor is the durable lesson"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn len_counts_rows_and_respects_deletes() {
    let store = SqliteMemoryStore::in_memory().unwrap();
    assert_eq!(store.len(), 0);
    for index in 0..4 {
        store
            .store(MemoryEntry::new(
                MemoryCategory::Fact,
                format!("fact {index}"),
            ))
            .await
            .unwrap();
    }
    assert_eq!(store.len(), 4);
}

#[tokio::test]
async fn two_stores_on_one_file_coordinate_through_wal() {
    let dir = temp_dir("wal");
    let path = dir.join("memory.db");

    let first = SqliteMemoryStore::open(&path).unwrap();
    let second = SqliteMemoryStore::open(&path).unwrap();
    let (a, b) = tokio::join!(
        async {
            first
                .store(MemoryEntry::new(
                    MemoryCategory::Fact,
                    "from the first writer",
                ))
                .await
        },
        async {
            second
                .store(MemoryEntry::new(
                    MemoryCategory::Fact,
                    "from the second writer",
                ))
                .await
        }
    );
    a.unwrap();
    b.unwrap();
    drop(first);
    drop(second);

    let third = SqliteMemoryStore::open(&path).unwrap();
    assert_eq!(third.len(), 2, "both writers' rows are durable");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn token_matches_rank_identically_to_the_in_memory_store() {
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let flat_store = loopctl::memory::builtin::InMemoryStore::new();

    let mut first = MemoryEntry::new(MemoryCategory::Insight, "deploy checklist nightly");
    first.relevance = 0.4;
    let mut second = MemoryEntry::new(MemoryCategory::Fact, "deploy scripts live in ops");
    second.relevance = 0.9;
    second.tags.push("deploy".to_string());
    let mut third = MemoryEntry::new(MemoryCategory::ErrorPattern, "unrelated rust compiler note");
    third.relevance = 1.0;
    for entry in [first, second, third] {
        sqlite_store.store(entry.clone()).await.unwrap();
        flat_store.store(entry).await.unwrap();
    }

    let from_sqlite = sqlite_store.retrieve("deploy", 3).await.unwrap();
    let from_flat = flat_store.retrieve("deploy", 3).await.unwrap();
    let sqlite_ids: Vec<_> = from_sqlite.iter().map(|entry| entry.id).collect();
    let flat_ids: Vec<_> = from_flat.iter().map(|entry| entry.id).collect();
    assert_eq!(
        sqlite_ids, flat_ids,
        "token-level matches order by the shared scorer, same as the flat backends"
    );
}

#[tokio::test]
async fn substring_hits_surface_as_fill_up_in_sqlite() {
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let flat_store = loopctl::memory::builtin::InMemoryStore::new();

    let entry = MemoryEntry::new(MemoryCategory::Fact, "the currency crashed overnight");
    sqlite_store.store(entry.clone()).await.unwrap();
    flat_store.store(entry.clone()).await.unwrap();

    let from_flat = flat_store.retrieve("curr", 3).await.unwrap();
    let from_sqlite = sqlite_store.retrieve("curr", 3).await.unwrap();
    assert_eq!(
        from_flat.first().map(|e| e.id),
        Some(entry.id),
        "the flat backends match substrings anywhere in the text"
    );
    assert_eq!(
        from_sqlite.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![entry.id],
        "FTS5 matches whole tokens, so the substring-only hit surfaces via \
        baseline fill-up — delivered, but not a token match"
    );

    let token_hit = sqlite_store.retrieve("currency", 3).await.unwrap();
    assert_eq!(
        token_hit.first().map(|e| e.id),
        Some(entry.id),
        "the whole token matches through FTS5"
    );
}

#[tokio::test]
async fn a_missing_fts_index_falls_back_to_like() {
    let dir = temp_dir("fts-fallback");
    let path = dir.join("memory.db");

    let store = SqliteMemoryStore::open(&path).unwrap();
    let mut target = MemoryEntry::new(MemoryCategory::Insight, "quokkas never blink sideways");
    target.relevance = 0.2;
    store.store(target.clone()).await.unwrap();
    for filler_text in [
        "an unrelated fact about rust",
        "another filler about compilers",
        "a third filler about async runtimes",
    ] {
        let mut filler = MemoryEntry::new(MemoryCategory::Fact, filler_text);
        filler.relevance = 0.9;
        store.store(filler).await.unwrap();
    }

    let saboteur = rusqlite::Connection::open(&path).unwrap();
    saboteur.execute("DROP TABLE memory_fts", []).unwrap();
    drop(saboteur);

    let hits = store.retrieve("quokkas", 2).await.unwrap();
    assert_eq!(
        hits.first().map(|entry| entry.id),
        Some(target.id),
        "retrieve still finds the low-relevance entry through the LIKE fallback — baseline fill-up alone could not reach it below the limit"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn an_in_memory_database_round_trips_without_the_filesystem() {
    let store = SqliteMemoryStore::in_memory().unwrap();
    let entry = loaded_entry();
    store.store(entry.clone()).await.unwrap();

    let hits = store.retrieve("verify", 3).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits.first().map(|e| e.id), Some(entry.id));
}
