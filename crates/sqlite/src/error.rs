//! Error type for the SQLite memory store.

use loopctl::error::LoopError;

/// Errors raised by [`SqliteMemoryStore`](crate::SqliteMemoryStore).
///
/// The constructors ([`open`](crate::SqliteMemoryStore::open),
/// [`in_memory`](crate::SqliteMemoryStore::in_memory)) return this type
/// directly, so a host setting up its store sees the concrete cause; the
/// `LoopMemory` trait methods convert to [`LoopError::Memory`] at the
/// boundary, where the trait's error type is fixed.
#[derive(Debug, thiserror::Error)]
pub enum SqliteMemoryError {
    /// A SQLite operation failed.
    ///
    /// Wraps the `rusqlite` error verbatim — schema bootstrap, statement
    /// execution, or a row conversion failure. Retriable conditions
    /// (`SQLITE_BUSY` after the configured busy timeout) surface here
    /// too, since the store holds no background retry state.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Serializing or deserializing an entry failed.
    ///
    /// Entry tags and categories round-trip through JSON text columns;
    /// a malformed value there surfaces as this variant rather than a
    /// raw SQLite type error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A stored row does not decode into a valid entry.
    ///
    /// The row's `id` is not a UUID, or its `category` is not one of
    /// the known names — damage only an external writer can introduce,
    /// since this store writes only valid values.
    #[error("invalid stored entry: {0}")]
    InvalidEntry(String),
}

impl From<SqliteMemoryError> for LoopError {
    fn from(error: SqliteMemoryError) -> Self {
        LoopError::Memory(format!("sqlite memory store: {error}"))
    }
}
