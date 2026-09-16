//! SQLite-backed [`LoopMemory`] — durable, indexed, concurrent.
//!
//! [`SqliteMemoryStore`] persists each
//! [`MemoryEntry`](loopctl::memory::MemoryEntry) as a row in a local
//! SQLite database (WAL mode, `rusqlite` with the `bundled` feature so
//! no system SQLite is required) and retrieves by loading every entry
//! and ranking it in Rust with the shared
//! [`score_entry`](loopctl::memory::score::score_entry) — the same
//! entries match, in the same order with the same tie-breaking, as the
//! other backends. An FTS5 index is maintained on every write; see
//! [`schema`](crate::schema) for its role.
//!
//! Add `loopctl-sqlite` as a direct dependency; do not enable any
//! feature on `loopctl` itself.

use crate::error::SqliteMemoryError;
use crate::schema::{CREATE_CORE_TABLE, CREATE_FTS_TABLE, CREATE_STAMPS_TABLE};
use loopctl::error::{LoopError, recover_guard};
use loopctl::memory::LoopMemory;
use loopctl::memory::consolidate::{ConsolidationConfig, consolidate_entries};
use loopctl::memory::entry::{ConsolidationStats, MemoryCategory, MemoryEntry};
use loopctl::memory::score::score_entry;
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// A [`LoopMemory`] backend backed by a SQLite database file.
///
/// Holds a single connection (WAL mode, five-second busy timeout) behind
/// a mutex, so the store is `Send + Sync` and safe to share via
/// `Arc<SqliteMemoryStore>` within one process. For multi-connection or
/// multi-process concurrency, open several stores against the same file
/// — WAL coordinates them — rather than pooling inside one.
///
/// # Retrieval shape
///
/// Retrieval loads every entry and ranks it with the shared
/// [`score_entry`](loopctl::memory::score::score_entry) — the same
/// results, ordering, and tie-breaking as the flat backends (loading in
/// rowid order matches their insertion-order stable sort, so even
/// baseline ties deliver identically). The FTS5 index is maintained on
/// every write but not consulted by retrieval: ranking every entry is
/// what guarantees the cross-backend parity contract.
///
/// # Example
///
/// ```rust
/// use loopctl_sqlite::SqliteMemoryStore;
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let store = SqliteMemoryStore::in_memory().unwrap();
/// store
///     .store(MemoryEntry::new(MemoryCategory::Fact, "grip the file with both hands"))
///     .await
///     .unwrap();
/// assert_eq!(store.len(), 1);
/// # });
/// ```
pub struct SqliteMemoryStore {
    /// The single SQLite connection, WAL mode, behind a mutex.
    ///
    /// `rusqlite::Connection` is `Send` but not `Sync`; the mutex is
    /// what makes the store shareable. It is the only lock the store
    /// holds — access stamps live in the database itself, shared with
    /// every other store on the same file.
    conn: Mutex<Connection>,

    /// The configuration driving this store's consolidation pass.
    ///
    /// Defaults to the same configuration
    /// [`InMemoryStore`](loopctl::memory::builtin::InMemoryStore) uses;
    /// set per store with
    /// [`with_consolidation`](Self::with_consolidation).
    consolidation: ConsolidationConfig,
}

/// The column list every entry SELECT uses, in decode order.
///
/// Kept in one place so the raw-row reader and every loader agree on
/// column indexes; `read_entry_row` maps positionally onto it.
const ENTRY_COLUMNS: &str = "id, category, memory, tags, created_at, relevance, access_count, validated, last_accessed, last_decayed";

/// One raw database row, before decoding into a [`MemoryEntry`].
///
/// Tags and category are still JSON text here; decoding is fallible
/// (bad JSON, unknown category, malformed UUID) and happens in
/// [`decode_entry`].
struct EntryRow {
    /// The entry's UUID, as text.
    ///
    /// Decoded back into a `Uuid` after the row loads; a value that
    /// does not parse is an `InvalidEntry`.
    id: String,

    /// The category's serde name (`snake_case`).
    ///
    /// Round-trips through the same serde renaming the enum serializes
    /// with, so no second name mapping can drift.
    category: String,

    /// The memory text.
    ///
    /// Duplicated into the FTS5 index row for retrieval; the core
    /// column remains the source of truth.
    memory: String,

    /// The tags as a compact JSON array.
    ///
    /// One `serde_json` round trip per store/load; tag matching itself
    /// happens in Rust during re-scoring.
    tags: String,

    /// Creation time, milliseconds since the Unix epoch.
    ///
    /// The lossless-if-millisecond storage form for `SystemTime`;
    /// pre-epoch stamps clamp to zero on the way in.
    created_at: i64,

    /// The relevance score.
    ///
    /// Rides SQLite's `REAL` column; the 0.0–1.0 range round-trips
    /// through the wider float type without loss.
    relevance: f32,

    /// How often the entry was surfaced by retrieval.
    ///
    /// Widened from `usize` on the way in and saturated back on the
    /// way out, so a foreign writer's absurd value cannot panic a load.
    access_count: i64,

    /// Whether the entry is validated, 0 or 1.
    ///
    /// SQLite has no boolean affinity; the mapping is the conventional
    /// nonzero-is-true.
    validated: i64,

    /// Last retrieval time, milliseconds, or `NULL`.
    ///
    /// `NULL` is the entry's never-retrieved state, matching the
    /// `Option<SystemTime>` it decodes into.
    last_accessed: Option<i64>,

    /// Last decay time, milliseconds, or `NULL`.
    ///
    /// Stamped by the shared consolidation pass; `NULL` means decay
    /// has not run since the field existed.
    last_decayed: Option<i64>,
}

impl SqliteMemoryStore {
    /// Open (or create) the database at `path`, bootstrap the schema.
    ///
    /// Idempotent: every DDL statement is `IF NOT EXISTS`, so opening an
    /// existing database is safe and leaves its rows untouched. Enables
    /// WAL journaling (concurrent readers with one writer), `NORMAL`
    /// sync, and a five-second busy timeout so a contended writer waits
    /// rather than failing immediately.
    ///
    /// # Errors
    ///
    /// [`SqliteMemoryError`] if the database cannot be opened or the
    /// schema bootstrap fails — the concrete cause before the
    /// `LoopError` boundary; the trait methods convert there.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteMemoryError> {
        let conn = Connection::open(path)?;
        Self::configure(&conn)?;
        Ok(Self::from_connection(conn))
    }

    /// Open a private in-memory database.
    ///
    /// Useful for tests and throwaway stores; nothing touches the
    /// filesystem, and the data dies with the store.
    ///
    /// # Errors
    ///
    /// [`SqliteMemoryError`] if the database or its schema cannot be
    /// created (in practice, only on allocation failure) — the concrete
    /// cause before the `LoopError` boundary.
    pub fn in_memory() -> Result<Self, SqliteMemoryError> {
        let conn = Connection::open_in_memory()?;
        Self::configure(&conn)?;
        Ok(Self::from_connection(conn))
    }

    /// Set the configuration driving this store's consolidation pass.
    ///
    /// Builder-style, mirroring
    /// [`InMemoryStore::with_consolidation`](loopctl::memory::builtin::InMemoryStore::with_consolidation);
    /// the config is read-only after construction.
    #[must_use]
    pub fn with_consolidation(mut self, config: ConsolidationConfig) -> Self {
        self.consolidation = config;
        self
    }

    /// Assemble a store around an already-configured connection.
    ///
    /// Both constructors differ only in how they obtain the connection;
    /// everything after it is this one path.
    fn from_connection(conn: Connection) -> Self {
        Self {
            conn: Mutex::new(conn),
            consolidation: ConsolidationConfig::default(),
        }
    }

    /// Apply pragmas and create the schema on a fresh connection.
    ///
    /// WAL on an in-memory database reports `memory` instead — the
    /// pragma statement still succeeds, and journaling is irrelevant
    /// without a file.
    ///
    /// # Errors
    ///
    /// Any pragma statement or DDL batch failure, mapped from
    /// [`rusqlite`](rusqlite).
    fn configure(conn: &Connection) -> Result<(), SqliteMemoryError> {
        let _journal: String = conn
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .map_err(SqliteMemoryError::from)?;
        conn.busy_timeout(Duration::from_millis(5000))
            .map_err(SqliteMemoryError::from)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(SqliteMemoryError::from)?;
        conn.execute_batch(CREATE_CORE_TABLE)
            .map_err(SqliteMemoryError::from)?;
        conn.execute_batch(CREATE_FTS_TABLE)
            .map_err(SqliteMemoryError::from)?;
        conn.execute_batch(CREATE_STAMPS_TABLE)
            .map_err(SqliteMemoryError::from)?;
        Ok(())
    }

    /// Load every entry, in insertion (rowid) order.
    ///
    /// Insertion order is what the flat backends' stable sort ties break
    /// on, so loading in rowid order keeps equal-score ordering
    /// identical across backends.
    ///
    /// # Errors
    ///
    /// The SELECT fails, or any stored row does not decode into a valid
    /// entry (first malformed row wins).
    fn load_all_entries(conn: &Connection) -> Result<Vec<MemoryEntry>, SqliteMemoryError> {
        let sql = format!("SELECT {ENTRY_COLUMNS} FROM memory_entries ORDER BY rowid");
        let mut statement = conn.prepare(&sql).map_err(SqliteMemoryError::from)?;
        let rows = statement
            .query_map([], read_entry_row)
            .map_err(SqliteMemoryError::from)?;
        collect_decoded(rows)
    }

    /// Record retrieval access stamps as pending rows in the database.
    ///
    /// One row per entry id, keeping the newest stamp — the same
    /// max-rule the in-memory side log used. Because the rows live in
    /// the database, stamps are visible to every `SqliteMemoryStore` on
    /// the same file and survive a process restart.
    ///
    /// # Errors
    ///
    /// The upsert statement fails.
    fn record_access_stamps(
        conn: &Mutex<Connection>,
        ids: &[Uuid],
    ) -> Result<(), SqliteMemoryError> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = recover_guard(conn.lock());
        let mut statement = conn
            .prepare(
                "INSERT INTO access_stamps (id, stamped_at) VALUES (?1, ?2) \
                 ON CONFLICT(id) DO UPDATE SET stamped_at = max(stamped_at, excluded.stamped_at)",
            )
            .map_err(SqliteMemoryError::from)?;
        let now = time_to_millis(SystemTime::now());
        for id in ids {
            statement
                .execute(params![id.to_string(), now])
                .map_err(SqliteMemoryError::from)?;
        }
        Ok(())
    }

    /// Load all pending access stamps within the consolidation transaction.
    ///
    /// The caller holds the transaction, so the load and the consume
    /// delete see the same snapshot: a stamp committed before this pass
    /// is folded and deleted, one committed after survives as a row for
    /// the next pass — no stamp can be lost in between.
    ///
    /// # Errors
    ///
    /// The SELECT fails, or a stamp row holds an unreadable id.
    fn load_access_stamps(tx: &Connection) -> Result<HashMap<Uuid, SystemTime>, SqliteMemoryError> {
        let mut statement = tx
            .prepare("SELECT id, stamped_at FROM access_stamps")
            .map_err(SqliteMemoryError::from)?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(SqliteMemoryError::from)?;
        let mut stamps = HashMap::new();
        for row in rows {
            let (id, stamped_at) = row.map_err(SqliteMemoryError::from)?;
            let id = Uuid::parse_str(&id)
                .map_err(|e| SqliteMemoryError::InvalidEntry(format!("stamp id: {e}")))?;
            stamps.insert(id, millis_to_time(stamped_at));
        }
        Ok(stamps)
    }

    /// Write one entry's core row and FTS row inside the given transaction.
    ///
    /// The core write is `INSERT OR REPLACE` keyed by the entry's UUID,
    /// so re-storing an entry (post-consolidation rewrite, dedup) keeps
    /// one row — the last stored copy, at the end of insertion order.
    /// Storing the same id twice is therefore a documented divergence
    /// from the flat backends, which append and keep both copies; a
    /// duplicate-id store here collapses to the newest copy. The FTS
    /// row is deleted-then-inserted by the same id —
    /// the standalone index is keyed by UUID, not by the core table's
    /// rowid, which `INSERT OR REPLACE` would change.
    ///
    /// # Errors
    ///
    /// Either statement fails, or the entry's tags or category cannot
    /// be serialized.
    fn insert_entry(tx: &Connection, entry: &MemoryEntry) -> Result<(), SqliteMemoryError> {
        let tags = serde_json::to_string(&entry.tags).map_err(SqliteMemoryError::from)?;
        let category = category_name(entry.category)?;
        tx.execute(
            "INSERT OR REPLACE INTO memory_entries \
             (id, category, memory, tags, created_at, relevance, access_count, validated, \
              last_accessed, last_decayed) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                entry.id.to_string(),
                category,
                entry.memory,
                tags,
                time_to_millis(entry.created_at),
                entry.relevance,
                access_count_column(entry.access_count),
                i64::from(entry.validated),
                entry.last_accessed.map(time_to_millis),
                entry.last_decayed.map(time_to_millis),
            ],
        )
        .map_err(SqliteMemoryError::from)?;
        tx.execute(
            "DELETE FROM memory_fts WHERE id = ?1",
            params![entry.id.to_string()],
        )
        .map_err(SqliteMemoryError::from)?;
        tx.execute(
            "INSERT INTO memory_fts (memory, id) VALUES (?1, ?2)",
            params![entry.memory, entry.id.to_string()],
        )
        .map_err(SqliteMemoryError::from)?;
        Ok(())
    }
}

/// Read one row of the entry table into its raw form.
///
/// # Errors
///
/// Any column is of the wrong type or the row is incomplete — the
/// mapping layer's failure, not the data's.
fn read_entry_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<EntryRow> {
    Ok(EntryRow {
        id: row.get(0)?,
        category: row.get(1)?,
        memory: row.get(2)?,
        tags: row.get(3)?,
        created_at: row.get(4)?,
        relevance: row.get(5)?,
        access_count: row.get(6)?,
        validated: row.get(7)?,
        last_accessed: row.get(8)?,
        last_decayed: row.get(9)?,
    })
}

/// Decode raw rows into entries, surfacing the first malformed row.
///
/// # Errors
///
/// The row iteration fails, or any row does not decode into a valid
/// entry.
fn collect_decoded<F>(
    rows: rusqlite::MappedRows<'_, F>,
) -> Result<Vec<MemoryEntry>, SqliteMemoryError>
where
    F: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<EntryRow>,
{
    let raw = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(SqliteMemoryError::from)?;
    raw.iter().map(decode_entry).collect()
}

/// Decode one raw row into a [`MemoryEntry`].
///
/// # Errors
///
/// The stored id is not a UUID, the category name is unknown, or the
/// tags column is not a JSON array of strings.
fn decode_entry(row: &EntryRow) -> Result<MemoryEntry, SqliteMemoryError> {
    let id = Uuid::parse_str(&row.id)
        .map_err(|e| SqliteMemoryError::InvalidEntry(format!("stored id is not a UUID: {e}")))?;
    let category = category_from_name(&row.category)?;
    let tags: Vec<String> = serde_json::from_str(&row.tags).map_err(SqliteMemoryError::from)?;
    Ok(MemoryEntry {
        id,
        category,
        memory: row.memory.clone(),
        tags,
        created_at: millis_to_time(row.created_at),
        relevance: row.relevance,
        access_count: usize::try_from(row.access_count).unwrap_or(usize::MAX),
        validated: row.validated != 0,
        last_accessed: row.last_accessed.map(millis_to_time),
        last_decayed: row.last_decayed.map(millis_to_time),
    })
}

/// Serialize a category to its `snake_case` serde name.
///
/// # Errors
///
/// Serialization fails, or the category did not serialize to a string
/// (unreachable for the closed enum, guarded anyway).
fn category_name(category: MemoryCategory) -> Result<String, SqliteMemoryError> {
    match serde_json::to_value(category).map_err(SqliteMemoryError::from)? {
        serde_json::Value::String(name) => Ok(name),
        other => Err(SqliteMemoryError::InvalidEntry(format!(
            "category did not serialize to a string: {other}"
        ))),
    }
}

/// Parse a category from its `snake_case` serde name.
///
/// # Errors
///
/// The name is not one of the known categories.
fn category_from_name(name: &str) -> Result<MemoryCategory, SqliteMemoryError> {
    serde_json::from_value(serde_json::Value::String(name.to_string()))
        .map_err(SqliteMemoryError::from)
}

/// Wall-clock time to milliseconds since the Unix epoch.
///
/// Times before the epoch clamp to zero — SQLite stores millis as a
/// signed integer, and a pre-epoch stamp has no meaning worth
/// preserving. Values beyond `i64` saturate.
fn time_to_millis(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

/// Milliseconds since the Unix epoch back to wall-clock time.
///
/// Negative values (not written by this store) clamp to the epoch; the
/// sub-millisecond remainder of a round trip is at most one millisecond.
fn millis_to_time(millis: i64) -> SystemTime {
    u64::try_from(millis).map_or(UNIX_EPOCH, |m| {
        UNIX_EPOCH
            .checked_add(Duration::from_millis(m))
            .unwrap_or(UNIX_EPOCH)
    })
}

/// Access count to its INTEGER column value, saturating at `i64::MAX`.
///
/// The saturation is invisible in practice — a count that large needs
/// more retrievals than the universe has time for — and keeps the
/// conversion total so the write path never panics.
fn access_count_column(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}

impl LoopMemory for SqliteMemoryStore {
    /// Upsert the entry and its full-text row in one transaction.
    ///
    /// Both writes commit or roll back together, so the index can never
    /// reference a row the table does not hold.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let mut conn = recover_guard(self.conn.lock());
            let tx = conn
                .transaction()
                .map_err(SqliteMemoryError::from)
                .map_err(LoopError::from)?;
            Self::insert_entry(&tx, &entry)?;
            tx.commit().map_err(SqliteMemoryError::from)?;
            Ok(())
        })
    }

    /// Rank the whole store with the shared scorer, identical to the
    /// flat backends.
    ///
    /// Loads every entry in rowid order, scores each with
    /// [`score_entry`](loopctl::memory::score::score_entry), sorts by
    /// descending composite score, and takes `limit` — so results,
    /// ordering, and tie-breaking all match
    /// [`InMemoryStore`](loopctl::memory::builtin::InMemoryStore)
    /// (and, under loopctl's `file_memory` feature,
    /// `FileMemoryStore`) — then stamps the access log for delivered
    /// query-matched entries.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + 'a>> {
        let query = query.to_string();
        Box::pin(async move {
            let query_lower = query.to_lowercase();
            let trimmed = query_lower.trim().to_string();
            let words: Vec<String> = query_lower.split_whitespace().map(str::to_string).collect();
            let entries = {
                let conn = recover_guard(self.conn.lock());
                Self::load_all_entries(&conn)?
            };
            let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();
            let mut scored: Vec<(f32, bool, MemoryEntry)> = entries
                .into_iter()
                .map(|entry| {
                    let (score, matched) = score_entry(&entry, &trimmed, &word_refs);
                    (score, matched, entry)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let selected: Vec<(bool, MemoryEntry)> = scored
                .into_iter()
                .take(limit)
                .map(|(_, matched, entry)| (matched, entry))
                .collect();
            let stamped: Vec<Uuid> = selected
                .iter()
                .filter(|(matched, _)| *matched)
                .map(|(_, entry)| entry.id)
                .collect();
            if let Err(error) = Self::record_access_stamps(&self.conn, &stamped) {
                tracing::warn!(%error, "recording access stamps failed; best-effort");
            }
            Ok(selected.into_iter().map(|(_, entry)| entry).collect())
        })
    }

    /// Run the shared consolidation pass and rewrite the survivors.
    ///
    /// Consumes the pending access stamps — every row, from any store
    /// instance on this file and from before any restart — inside the
    /// transaction, folds them into
    /// [`last_accessed`](MemoryEntry::last_accessed) and
    /// [`access_count`](MemoryEntry::access_count), then delegates to
    /// [`consolidate_entries`](loopctl::memory::consolidate::consolidate_entries)
    /// — the same decay, merge, and prune pass every backend runs — and
    /// rewrites the table and the full-text index in the same
    /// transaction, so pruned entries stay gone after a restart. The
    /// transaction opens before any read: a concurrent writer's commit
    /// either precedes this pass's snapshot or fails its write upgrade,
    /// so a stamp can never be lost to a retrieve racing this pass — it
    /// is either folded here or survives as a row for the next pass.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let now = SystemTime::now();
            let mut conn = recover_guard(self.conn.lock());
            let tx = conn
                .transaction()
                .map_err(SqliteMemoryError::from)
                .map_err(LoopError::from)?;
            let mut entries = Self::load_all_entries(&tx)?;
            let stamps = Self::load_access_stamps(&tx)?;
            for entry in &mut entries {
                if let Some(stamp) = stamps.get(&entry.id) {
                    entry.last_accessed = entry.last_accessed.max(Some(*stamp));
                    entry.access_count = entry.access_count.saturating_add(1);
                }
            }
            tx.execute("DELETE FROM access_stamps", [])
                .map_err(SqliteMemoryError::from)?;
            let stats = consolidate_entries(&mut entries, &self.consolidation, now);
            tx.execute("DELETE FROM memory_entries", [])
                .map_err(SqliteMemoryError::from)?;
            tx.execute("DELETE FROM memory_fts", [])
                .map_err(SqliteMemoryError::from)?;
            for entry in &entries {
                Self::insert_entry(&tx, entry)?;
            }
            tx.commit().map_err(SqliteMemoryError::from)?;
            Ok(stats)
        })
    }

    /// Number of entries currently in the table.
    ///
    /// A single `COUNT(*)` under the connection lock — no rows are
    /// materialized. The trait's `len` is infallible, so a database that
    /// errors under the count (busy past the timeout, corruption) reads
    /// as zero with a warning — treat a sudden zero on a known-loaded
    /// store as an error signal, not as emptiness.
    fn len(&self) -> usize {
        let conn = recover_guard(self.conn.lock());
        match conn.query_row("SELECT COUNT(*) FROM memory_entries", [], |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(count) => usize::try_from(count).unwrap_or(usize::MAX),
            Err(error) => {
                tracing::warn!(%error, "memory count failed; reporting zero");
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EntryRow, access_count_column, category_from_name, category_name, decode_entry,
        millis_to_time, time_to_millis,
    };
    use crate::error::SqliteMemoryError;
    use loopctl::memory::MemoryCategory;
    use std::time::{Duration, UNIX_EPOCH};

    fn valid_row() -> EntryRow {
        EntryRow {
            id: "0b9a2c78-6d1e-4a3f-8f2c-9c5a6b7d8e9f".to_string(),
            category: "strategy".to_string(),
            memory: "verify the diff compiles".to_string(),
            tags: "[\"editing\"]".to_string(),
            created_at: 1_234_567_890_000,
            relevance: 0.42,
            access_count: 7,
            validated: 1,
            last_accessed: Some(1_234_567_891_000),
            last_decayed: None,
        }
    }

    #[test]
    fn decode_entry_rebuilds_every_field() {
        let entry = decode_entry(&valid_row()).unwrap();
        assert_eq!(entry.memory, "verify the diff compiles");
        assert_eq!(entry.category, MemoryCategory::Strategy);
        assert_eq!(entry.tags, vec!["editing".to_string()]);
        assert!((entry.relevance - 0.42).abs() < 1e-6);
        assert_eq!(entry.access_count, 7);
        assert!(entry.validated);
        assert_eq!(
            entry.created_at,
            UNIX_EPOCH + Duration::from_secs(1_234_567_890)
        );
        assert_eq!(entry.last_decayed, None);
    }

    #[test]
    fn decode_entry_rejects_a_non_uuid_id() {
        let mut row = valid_row();
        row.id = "not-a-uuid".to_string();
        assert!(matches!(
            decode_entry(&row),
            Err(SqliteMemoryError::InvalidEntry(_))
        ));
    }

    #[test]
    fn decode_entry_rejects_an_unknown_category() {
        let mut row = valid_row();
        row.category = "no_such_category".to_string();
        assert!(decode_entry(&row).is_err());
    }

    #[test]
    fn decode_entry_rejects_malformed_tags_json() {
        let mut row = valid_row();
        row.tags = "not json".to_string();
        assert!(decode_entry(&row).is_err());
    }

    #[test]
    fn decode_entry_saturates_an_absurd_access_count() {
        let mut row = valid_row();
        row.access_count = -5;
        let entry = decode_entry(&row).unwrap();
        assert_eq!(
            entry.access_count,
            usize::MAX,
            "a foreign writer's absurd count saturates on the way out — \
            it cannot panic a load and does not read as an honest zero"
        );
    }

    #[test]
    fn category_names_round_trip_through_the_serde_form() {
        for category in [
            MemoryCategory::Fact,
            MemoryCategory::Strategy,
            MemoryCategory::ErrorPattern,
        ] {
            let name = category_name(category).unwrap();
            assert_eq!(category_from_name(&name).unwrap(), category);
        }
        assert!(category_from_name("unknown").is_err());
    }

    #[test]
    fn time_conversions_round_trip_at_millisecond_resolution() {
        let stamp = UNIX_EPOCH + Duration::from_millis(1_724_000_000_123);
        assert_eq!(millis_to_time(time_to_millis(stamp)), stamp);
    }

    #[test]
    fn pre_epoch_times_clamp_to_zero_and_negatives_to_the_epoch() {
        let before = UNIX_EPOCH - Duration::from_secs(10);
        assert_eq!(time_to_millis(before), 0);
        assert_eq!(millis_to_time(-1), UNIX_EPOCH);
    }

    #[test]
    fn huge_durations_saturate_instead_of_panicking() {
        let enormous = UNIX_EPOCH + Duration::from_millis(u64::MAX);
        assert_eq!(time_to_millis(enormous), i64::MAX);
        assert_eq!(access_count_column(usize::MAX), i64::MAX);
    }
}
