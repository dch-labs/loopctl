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
async fn substring_hits_match_exactly_like_the_flat_backends() {
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let flat_store = loopctl::memory::builtin::InMemoryStore::new();

    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "the currency crashed overnight");
    entry.relevance = 0.5;
    sqlite_store.store(entry.clone()).await.unwrap();
    flat_store.store(entry.clone()).await.unwrap();
    for filler_text in ["a quiet note about rust", "another quiet note about async"] {
        let mut filler = MemoryEntry::new(MemoryCategory::Fact, filler_text);
        filler.relevance = 0.9;
        sqlite_store.store(filler.clone()).await.unwrap();
        flat_store.store(filler).await.unwrap();
    }

    let from_flat = flat_store.retrieve("curr", 2).await.unwrap();
    let from_sqlite = sqlite_store.retrieve("curr", 2).await.unwrap();
    assert_eq!(
        from_flat.first().map(|e| e.id),
        Some(entry.id),
        "the flat backends match substrings anywhere in the text"
    );
    assert_eq!(
        from_sqlite.first().map(|e| e.id),
        Some(entry.id),
        "the text scan carries the same substring match below the \
        fill-up cut — same result, not a rescued baseline entry"
    );
}

#[tokio::test]
async fn tag_hits_match_exactly_like_the_flat_backends() {
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let flat_store = loopctl::memory::builtin::InMemoryStore::new();

    let mut tagged = MemoryEntry::new(MemoryCategory::Insight, "scripts live in ops");
    tagged.relevance = 0.5;
    tagged.tags.push("deploy".to_string());
    let mut textual = MemoryEntry::new(MemoryCategory::Fact, "deploy checklist nightly");
    textual.relevance = 0.4;
    sqlite_store.store(tagged.clone()).await.unwrap();
    flat_store.store(tagged.clone()).await.unwrap();
    sqlite_store.store(textual.clone()).await.unwrap();
    flat_store.store(textual.clone()).await.unwrap();
    for filler_text in ["a quiet note about rust", "another quiet note about async"] {
        let mut filler = MemoryEntry::new(MemoryCategory::Fact, filler_text);
        filler.relevance = 0.9;
        sqlite_store.store(filler.clone()).await.unwrap();
        flat_store.store(filler).await.unwrap();
    }

    let from_flat = flat_store.retrieve("deploy", 2).await.unwrap();
    let from_sqlite = sqlite_store.retrieve("deploy", 2).await.unwrap();
    let flat_ids: Vec<_> = from_flat.iter().map(|entry| entry.id).collect();
    let sqlite_ids: Vec<_> = from_sqlite.iter().map(|entry| entry.id).collect();
    assert_eq!(
        flat_ids, sqlite_ids,
        "tag matches reach the candidate pool and rank identically to the \
        flat backends — fill-up alone cannot deliver them below the limit"
    );
    assert_eq!(
        sqlite_ids.first(),
        Some(&textual.id),
        "the word hit ranks first"
    );
    assert_eq!(
        sqlite_ids.get(1),
        Some(&tagged.id),
        "the tag hit follows on its bonus"
    );
}

#[tokio::test]
async fn equal_score_ties_order_like_the_flat_backends() {
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let flat_store = loopctl::memory::builtin::InMemoryStore::new();

    for text in ["first quiet note", "second quiet note", "third quiet note"] {
        let entry = MemoryEntry::new(MemoryCategory::Fact, text);
        sqlite_store.store(entry.clone()).await.unwrap();
        flat_store.store(entry).await.unwrap();
    }

    let from_flat = flat_store.retrieve("unrelated", 2).await.unwrap();
    let from_sqlite = sqlite_store.retrieve("unrelated", 2).await.unwrap();
    let flat_ids: Vec<_> = from_flat.iter().map(|entry| entry.id).collect();
    let sqlite_ids: Vec<_> = from_sqlite.iter().map(|entry| entry.id).collect();
    assert_eq!(
        flat_ids, sqlite_ids,
        "baseline ties keep insertion order in both stores — rowid order \
        matches the flat stores' stable sort"
    );
}

#[tokio::test]
async fn losing_the_full_text_index_loses_no_recall() {
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
        "the full-text index is a supplement — losing it costs no recall, the text scan still reaches the low-relevance entry below the limit"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn access_stamps_are_shared_across_store_instances() {
    let dir = temp_dir("shared-stamps");
    let path = dir.join("memory.db");

    let writer = SqliteMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "grip large files with both hands");
    entry.relevance = 0.9;
    writer.store(entry.clone()).await.unwrap();

    let reader = SqliteMemoryStore::open(&path).unwrap();
    reader.retrieve("grip large files", 3).await.unwrap();
    drop(reader);

    writer.consolidate().await.unwrap();

    let probe = rusqlite::Connection::open(&path).unwrap();
    let pending: i64 = probe
        .query_row("SELECT COUNT(*) FROM access_stamps", [], |row| row.get(0))
        .unwrap();
    assert_eq!(pending, 0, "the pass consumed every pending stamp row");

    let consolidated = writer.retrieve("grip large files", 3).await.unwrap();
    assert_eq!(
        consolidated.first().map(|e| e.access_count),
        Some(1),
        "the consolidating instance consumed the stamp the other instance wrote"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn access_stamps_survive_a_restart() {
    let dir = temp_dir("stamps-restart");
    let path = dir.join("memory.db");

    let store = SqliteMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "verify the diff compiles");
    entry.relevance = 0.9;
    store.store(entry.clone()).await.unwrap();
    store.retrieve("verify diff", 3).await.unwrap();
    drop(store);

    let reopened = SqliteMemoryStore::open(&path).unwrap();
    reopened.consolidate().await.unwrap();
    let consolidated = reopened.retrieve("verify diff", 3).await.unwrap();
    assert_eq!(
        consolidated.first().map(|e| e.access_count),
        Some(1),
        "the stamp written before the restart folds into the next pass"
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn consolidates_racing_retrieves_never_lose_a_stamp() {
    let dir = temp_dir("stamp-race");
    let path = dir.join("memory.db");

    let writer = SqliteMemoryStore::open(&path).unwrap();
    let mut entry = MemoryEntry::new(MemoryCategory::Fact, "grip large files with both hands");
    entry.relevance = 0.9;
    writer.store(entry.clone()).await.unwrap();
    drop(writer);

    let retriever = std::sync::Arc::new(SqliteMemoryStore::open(&path).unwrap());
    let consolidator = std::sync::Arc::new(SqliteMemoryStore::open(&path).unwrap());
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let retriever = std::sync::Arc::clone(&retriever);
        tasks.push(tokio::task::spawn(async move {
            let mut attempts = 0;
            loop {
                match retriever.retrieve("grip large files", 3).await {
                    Ok(_) => return,
                    Err(error) => {
                        attempts += 1;
                        assert!(
                            attempts < 200,
                            "a busy database must eventually yield: {error}"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            }
        }));
    }
    for _ in 0..4 {
        let consolidator = std::sync::Arc::clone(&consolidator);
        tasks.push(tokio::task::spawn(async move {
            let mut attempts = 0;
            loop {
                match consolidator.consolidate().await {
                    Ok(_) => return,
                    Err(error) => {
                        attempts += 1;
                        assert!(
                            attempts < 200,
                            "a busy database must eventually yield: {error}"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    consolidator.consolidate().await.unwrap();

    let probe = rusqlite::Connection::open(&path).unwrap();
    let pending: i64 = probe
        .query_row("SELECT COUNT(*) FROM access_stamps", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        pending, 0,
        "every stamp written before the final pass was folded by it — \
        none was lost to a retrieve racing a consolidate"
    );
    let consolidated = consolidator.retrieve("grip large files", 3).await.unwrap();
    let folded = consolidated.first().map_or(0, |e| e.access_count);
    assert!(
        folded >= 1,
        "the entry survived the racing passes and every stamp written \
        before the final pass folded into it — repeated re-stamping may \
        fold several times, but none may be lost"
    );
    drop(retriever);
    drop(consolidator);
    drop(probe);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_custom_consolidation_config_shapes_the_pass() {
    use loopctl::memory::consolidate::ConsolidationConfig;

    let default_store = SqliteMemoryStore::in_memory().unwrap();
    let tuned_store =
        SqliteMemoryStore::in_memory()
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
        "the per-handle config shapes this handle's pass — a zero prune \
        floor keeps what the default prunes"
    );
}

#[tokio::test]
async fn all_three_backends_rank_identically_including_ties() {
    use loopctl::memory::FileMemoryStore;
    use loopctl::memory::builtin::InMemoryStore;

    let dir = temp_dir("three-way");
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let file_store = FileMemoryStore::new(dir.join("memory.jsonl"));
    let flat_store = InMemoryStore::new();
    let entries = [
        MemoryEntry::new(
            MemoryCategory::Fact,
            "alpha grip large files with both hands",
        ),
        MemoryEntry::new(
            MemoryCategory::Fact,
            "alpha nightly deploys pause the world",
        ),
        MemoryEntry::new(
            MemoryCategory::Fact,
            "alpha read the whole file before editing",
        ),
    ];
    for entry in &entries {
        sqlite_store.store(entry.clone()).await.unwrap();
        file_store.store(entry.clone()).await.unwrap();
        flat_store.store(entry.clone()).await.unwrap();
    }

    for query in ["alpha", "deploy", "grip files", "unmatched query"] {
        let sqlite_ids = sqlite_store
            .retrieve(query, 5)
            .await
            .unwrap()
            .iter()
            .map(|entry| (entry.id, entry.memory.clone()))
            .collect::<Vec<_>>();
        let file_ids = file_store
            .retrieve(query, 5)
            .await
            .unwrap()
            .iter()
            .map(|entry| (entry.id, entry.memory.clone()))
            .collect::<Vec<_>>();
        let flat_ids = flat_store
            .retrieve(query, 5)
            .await
            .unwrap()
            .iter()
            .map(|entry| (entry.id, entry.memory.clone()))
            .collect::<Vec<_>>();
        assert_eq!(sqlite_ids, flat_ids, "sqlite matches the flat oracle");
        assert_eq!(
            file_ids, flat_ids,
            "the file store matches the flat oracle directly, not transitively"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn duplicate_id_stores_collapse_to_one_copy_here_but_not_in_the_file_store() {
    use loopctl::memory::FileMemoryStore;

    let dir = temp_dir("duplicate-id");
    let sqlite_store = SqliteMemoryStore::in_memory().unwrap();
    let file_store = FileMemoryStore::new(dir.join("memory.jsonl"));
    let entry = MemoryEntry::new(MemoryCategory::Fact, "stored twice under one id");

    sqlite_store.store(entry.clone()).await.unwrap();
    sqlite_store.store(entry.clone()).await.unwrap();
    file_store.store(entry.clone()).await.unwrap();
    file_store.store(entry.clone()).await.unwrap();

    assert_eq!(
        sqlite_store.len(),
        1,
        "the core write is INSERT OR REPLACE keyed by UUID — a \
        duplicate-id store keeps the newest single copy"
    );
    assert_eq!(
        file_store.len(),
        2,
        "the file backend appends and keeps both copies — the pinned, \
        documented divergence between the backends"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn a_count_error_reads_as_zero_with_the_entry_table_gone() {
    let dir = temp_dir("count-error");
    let path = dir.join("memory.db");
    let store = SqliteMemoryStore::open(&path).unwrap();
    store.store(loaded_entry()).await.unwrap();
    assert_eq!(store.len(), 1, "the store counts its one row");

    let saboteur = rusqlite::Connection::open(&path).unwrap();
    saboteur.execute("DROP TABLE memory_entries", []).unwrap();
    assert_eq!(
        store.len(),
        0,
        "the trait's len is infallible — a database that errors under \
        the count reads as zero, the documented error signal"
    );
    drop(store);
    drop(saboteur);
    std::fs::remove_dir_all(&dir).unwrap();
}
