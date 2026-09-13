//! SQLite-backed [`LoopMemory`] — durable, indexed, concurrent.
//!
//! [`SqliteMemoryStore`] persists each
//! [`MemoryEntry`](loopctl::memory::MemoryEntry) as a row in a local
//! SQLite database (WAL mode, `rusqlite` with the `bundled` feature so
//! no system SQLite is required) and retrieves through an FTS5
//! full-text index with a `LIKE` fallback, re-ranked in Rust with the
//! shared [`score_entry`](loopctl::memory::score::score_entry) so
//! matched candidates order by the same formula as the other backends
//! (recall differs: FTS5 matches stemmed whole tokens, so a query that
//! appears only as a substring surfaces through baseline fill-up rather
//! than as a match).
//!
//! Add `loopctl-sqlite` as a direct dependency; do not enable any
//! feature on `loopctl` itself.

use crate::error::SqliteMemoryError;
use crate::schema::{CREATE_CORE_TABLE, CREATE_FTS_TABLE};
use loopctl::error::{LoopError, recover_guard};
use loopctl::memory::LoopMemory;
use loopctl::memory::consolidate::{ConsolidationConfig, consolidate_entries};
use loopctl::memory::entry::{ConsolidationStats, MemoryCategory, MemoryEntry};
use loopctl::memory::score::score_entry;
use rusqlite::{Connection, params, params_from_iter};
use std::collections::{HashMap, HashSet};
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
/// Candidates come from the FTS5 index (or the `LIKE` fallback when the
/// query is not valid FTS5 syntax, e.g. odd punctuation), are re-scored
/// with [`score_entry`](loopctl::memory::score::score_entry), and — when
/// fewer than `limit` matched — are topped up from the
/// highest-relevance rows, mirroring the flat backends' behavior of
/// delivering baseline-ranked entries when the store is sparser than
/// the limit. Recall is token-level: FTS5 matches stemmed whole words,
/// so a query that appears only as a substring inside a word (the flat
/// backends' substring matching) surfaces through fill-up, not as a
/// match.
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
    /// `rusqlite::Connection` is `Send` but not `Sync`; the mutex is what
    /// makes the store shareable. Lock order is always `conn` before
    /// `access_log` — `retrieve` takes both, and nothing takes them in
    /// the opposite order.
    conn: Mutex<Connection>,

    /// Access stamps recorded by `retrieve()`, keyed by entry id.
    ///
    /// In-memory only, following the reference store's side-log pattern:
    /// matched entries that were actually delivered are stamped, and the
    /// next consolidation pass folds the stamps into
    /// [`last_accessed`](MemoryEntry::last_accessed) and
    /// [`access_count`](MemoryEntry::access_count) before clearing the
    /// log. Not persisted — a process that exits mid-pass loses at most
    /// one pass of access accounting.
    access_log: Mutex<HashMap<Uuid, SystemTime>>,

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
            access_log: Mutex::new(HashMap::new()),
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

    /// Load the entries with the given ids, preserving the input order.
    ///
    /// # Errors
    ///
    /// The SELECT fails, or any stored row does not decode into a valid
    /// entry.
    fn load_entries_by_ids(
        conn: &Connection,
        ids: &[String],
    ) -> Result<Vec<MemoryEntry>, SqliteMemoryError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql =
            format!("SELECT {ENTRY_COLUMNS} FROM memory_entries WHERE id IN ({placeholders})");
        let mut statement = conn.prepare(&sql).map_err(SqliteMemoryError::from)?;
        let rows = statement
            .query_map(params_from_iter(ids.iter()), read_entry_row)
            .map_err(SqliteMemoryError::from)?;
        let loaded = collect_decoded(rows)?;
        let by_id: HashMap<String, MemoryEntry> = loaded
            .into_iter()
            .map(|entry| (entry.id.to_string(), entry))
            .collect();
        Ok(ids.iter().filter_map(|id| by_id.get(id).cloned()).collect())
    }

    /// Load up to `limit` highest-relevance entries not in `exclude`.
    ///
    /// The retrieval fill-up: when fewer entries than `limit` matched
    /// the query, the flat backends still deliver baseline-ranked
    /// entries, so this tops the candidate set up the same way.
    ///
    /// # Errors
    ///
    /// The SELECT fails, or any stored row does not decode into a valid
    /// entry.
    fn top_relevance_excluding(
        conn: &Connection,
        exclude: &HashSet<String>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, SqliteMemoryError> {
        let sql =
            format!("SELECT {ENTRY_COLUMNS} FROM memory_entries ORDER BY relevance DESC, rowid");
        let mut statement = conn.prepare(&sql).map_err(SqliteMemoryError::from)?;
        let rows = statement
            .query_map([], read_entry_row)
            .map_err(SqliteMemoryError::from)?;
        let decoded = collect_decoded(rows)?;
        Ok(decoded
            .into_iter()
            .filter(|entry| !exclude.contains(&entry.id.to_string()))
            .take(limit)
            .collect())
    }

    /// Ids of entries the FTS5 index matches, or `None` to fall back.
    ///
    /// Any FTS5 failure — unsupported syntax, index trouble — yields
    /// `None` after a warning, and the caller retries with `LIKE`; a
    /// query that is valid but matches nothing yields `Some(vec![])`.
    fn fts_candidate_ids(conn: &Connection, expression: &str, limit: usize) -> Option<Vec<String>> {
        let result = (|| -> Result<Vec<String>, rusqlite::Error> {
            let mut statement = conn.prepare(
                "SELECT id FROM memory_fts WHERE memory MATCH ?1 \
                     ORDER BY bm25(memory_fts) LIMIT ?2",
            )?;
            let rows = statement.query_map(
                params![expression, i64::try_from(limit).unwrap_or(i64::MAX)],
                |row| row.get::<_, String>(0),
            )?;
            rows.collect()
        })();
        match result {
            Ok(ids) => Some(ids),
            Err(error) => {
                tracing::warn!(%error, "fts query failed; falling back to LIKE");
                None
            }
        }
    }

    /// Ids of entries whose memory text contains any query word.
    ///
    /// The fallback path when FTS5 rejects the query; `%` and `_` in the
    /// query act as their LIKE selves, which can only over-match.
    ///
    /// # Errors
    ///
    /// The SELECT fails.
    fn like_candidate_ids(
        conn: &Connection,
        words: &[String],
    ) -> Result<Vec<String>, SqliteMemoryError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let clauses = words
            .iter()
            .enumerate()
            .map(|(index, _)| format!("memory LIKE ?{}", index.saturating_add(1)))
            .collect::<Vec<_>>()
            .join(" OR ");
        let patterns = words
            .iter()
            .map(|word| format!("%{word}%"))
            .collect::<Vec<_>>();
        let sql = format!("SELECT id FROM memory_entries WHERE {clauses}");
        let mut statement = conn.prepare(&sql).map_err(SqliteMemoryError::from)?;
        let rows = statement
            .query_map(params_from_iter(patterns), |row| row.get::<_, String>(0))
            .map_err(SqliteMemoryError::from)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(SqliteMemoryError::from)
    }

    /// Write one entry's core row and FTS row inside the given transaction.
    ///
    /// The core write is `INSERT OR REPLACE` keyed by the entry's UUID,
    /// so re-storing an entry (post-consolidation rewrite, dedup) keeps
    /// one row. The FTS row is deleted-then-inserted by the same id —
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
        access_count: usize::try_from(row.access_count).unwrap_or(0),
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

/// Quote each query term for FTS5 and join with `OR`.
///
/// Single quotes inside a term are doubled per FTS5 string syntax, so
/// arbitrary user text can never produce invalid match syntax; `None`
/// for a whitespace-only query, which matches nothing by construction.
fn fts_match_expression(query: &str) -> Option<String> {
    let terms = query
        .split_whitespace()
        .map(|term| format!("'{}'", term.replace('\'', "''")))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
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

    /// Full-text candidates, re-ranked with the shared scorer.
    ///
    /// Pulls candidates from FTS5 (or `LIKE` on any FTS5 failure), tops
    /// the set up from the highest-relevance rows when fewer than
    /// `limit` matched — the flat backends deliver baseline-ranked
    /// entries on sparse stores — then sorts with
    /// [`score_entry`](loopctl::memory::score::score_entry) and stamps
    /// the access log for delivered query-matched entries.
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
            let mut entries = {
                let conn = recover_guard(self.conn.lock());
                let mut candidates = if trimmed.is_empty() {
                    Self::load_all_entries(&conn)?
                } else {
                    let ids = fts_match_expression(&trimmed)
                        .and_then(|expression| {
                            Self::fts_candidate_ids(&conn, &expression, limit.saturating_mul(5))
                        })
                        .map_or_else(|| Self::like_candidate_ids(&conn, &words), Ok)?;
                    Self::load_entries_by_ids(&conn, &ids)?
                };
                if candidates.len() < limit {
                    let present: HashSet<String> = candidates
                        .iter()
                        .map(|entry| entry.id.to_string())
                        .collect();
                    let needed = limit.saturating_sub(candidates.len());
                    let fill = Self::top_relevance_excluding(&conn, &present, needed)?;
                    candidates.extend(fill);
                }
                candidates
            };
            let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();
            let mut scored: Vec<(f32, bool, MemoryEntry)> = entries
                .drain(..)
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
            let now = SystemTime::now();
            let mut access_log = recover_guard(self.access_log.lock());
            for (matched, entry) in &selected {
                if !matched {
                    continue;
                }
                let stamp = access_log.get(&entry.id).copied();
                access_log.insert(entry.id, stamp.map_or(now, |existing| existing.max(now)));
            }
            Ok(selected.into_iter().map(|(_, entry)| entry).collect())
        })
    }

    /// Run the shared consolidation pass and rewrite the survivors.
    ///
    /// Folds the access log into the entries, delegates to
    /// [`consolidate_entries`](loopctl::memory::consolidate::consolidate_entries)
    /// — the same decay, merge, and prune pass every backend runs — then
    /// rewrites the table and the full-text index in one transaction, so
    /// pruned entries stay gone after a restart.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let now = SystemTime::now();
            let mut conn = recover_guard(self.conn.lock());
            let mut entries = Self::load_all_entries(&conn)?;
            {
                let mut access_log = recover_guard(self.access_log.lock());
                for entry in &mut entries {
                    if let Some(stamp) = access_log.get(&entry.id) {
                        entry.last_accessed = entry.last_accessed.max(Some(*stamp));
                        entry.access_count = entry.access_count.saturating_add(1);
                    }
                }
                access_log.clear();
            }
            let stats = consolidate_entries(&mut entries, &self.consolidation, now);
            let tx = conn
                .transaction()
                .map_err(SqliteMemoryError::from)
                .map_err(LoopError::from)?;
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
