//! SQLite-backed [`LoopMemory`] for loopctl — durable, indexed,
//! concurrent.
//!
//! [`SqliteMemoryStore`] implements loopctl's memory trait over a local
//! SQLite database (`rusqlite` with the `bundled` feature, so no system
//! SQLite is required): WAL journaling for concurrent readers with one
//! writer, and retrieval that loads every entry and ranks it in Rust
//! with loopctl's shared scorer — the same entries match, in the same
//! order with the same tie-breaking, as the in-memory and file
//! backends. An FTS5 index is maintained on every write; retrieval
//! does not consult it — ranking every entry is what guarantees the
//! parity contract.
//!
//! Add this crate as a direct dependency; no feature on `loopctl`
//! itself is required:
//!
//! ```toml
//! [dependencies]
//! loopctl = "0.3"
//! loopctl-sqlite = "0.3"
//! ```
//!
//! ```no_run
//! use loopctl_sqlite::SqliteMemoryStore;
//! # fn main() -> Result<(), loopctl_sqlite::SqliteMemoryError> {
//! let store = SqliteMemoryStore::open("agent-memory.db")?;
//! # Ok(())
//! # }
//! ```
//!
//! [`LoopMemory`]: loopctl::memory::LoopMemory

#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::missing_errors_doc,
        clippy::unnecessary_wraps,
        clippy::clone_on_ref_ptr,
        clippy::doc_markdown,
        clippy::field_reassign_with_default,
        clippy::used_underscore_items,
        clippy::wildcard_imports,
    )
)]

mod error;
mod schema;
mod store;

pub use error::SqliteMemoryError;
pub use store::SqliteMemoryStore;
