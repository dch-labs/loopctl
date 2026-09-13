//! SQL schema for the SQLite memory store.
//!
//! One core table holding one row per
//! [`MemoryEntry`](loopctl::memory::MemoryEntry), plus a full-text index
//! over the memory text. All statements are idempotent
//! (`IF NOT EXISTS`), so opening an existing database is safe.

/// Creates the core entry table and its secondary indexes.
///
/// Columns map one-to-one onto [`MemoryEntry`](loopctl::memory::MemoryEntry)
/// fields: `tags` is a compact JSON array (one `serde_json` round trip —
/// the shared scorer handles tag matching in Rust, so a join table buys
/// nothing), `created_at` / `last_accessed` / `last_decayed` are
/// milliseconds since the Unix epoch (`SystemTime` has no SQLite
/// affinity), `validated` is 0/1, and `relevance` rides the REAL column
/// as an `f32`.
pub const CREATE_CORE_TABLE: &str = "CREATE TABLE IF NOT EXISTS memory_entries (
    id            TEXT PRIMARY KEY,
    category      TEXT NOT NULL,
    memory        TEXT NOT NULL,
    tags          TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    relevance     REAL NOT NULL,
    access_count  INTEGER NOT NULL,
    validated     INTEGER NOT NULL,
    last_accessed INTEGER,
    last_decayed  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memory_relevance ON memory_entries(relevance);
CREATE INDEX IF NOT EXISTS idx_memory_category  ON memory_entries(category);";

/// Creates the full-text index over the memory text.
///
/// A standalone FTS5 table carrying the owning entry's `id` as an
/// unindexed column, keyed by that id rather than by the core table's
/// rowid — `INSERT OR REPLACE` on the core table assigns a fresh rowid,
/// which would silently desynchronize a rowid-linked external-content
/// index. The porter unicode61 tokenizer gives stemming and
/// case-folding. Retrieval currently loads every entry and ranks it in
/// Rust with the shared [`score_entry`](loopctl::memory::score::score_entry)
/// for exact parity with the flat backends, so this index is maintained
/// on every write but not yet consulted — it is reserved for a future
/// indexed path that preserves that ranking contract.
pub const CREATE_FTS_TABLE: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(memory, id UNINDEXED, tokenize='porter unicode61')";
