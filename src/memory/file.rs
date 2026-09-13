//! File-backed [`LoopMemory`] — JSONL persistence with no new dependencies.
//!
//! [`FileMemoryStore`] mirrors [`InMemoryStore`](super::builtin::InMemoryStore)
//! in memory and appends each stored entry to a JSONL file — one
//! [`MemoryEntry`] per line — so an agent's
//! learned memory survives a process restart. Retrieval ranks with the
//! shared [`score_entry`] formula, so results
//! order identically to the in-memory store. Enable with the
//! `file_memory` feature.
//!
//! The store is single-process: two processes writing the same file will
//! interleave and corrupt it. For multi-process durability use a
//! database-backed companion such as `loopctl-sqlite`.

use crate::error::{LoopError, recover_guard};
use crate::memory::LoopMemory;
use crate::memory::consolidate::{ConsolidationConfig, consolidate_entries};
use crate::memory::entry::{ConsolidationStats, MemoryEntry};
use crate::memory::score::score_entry;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;
use uuid::Uuid;

/// A [`LoopMemory`] backend that persists entries to a JSONL file.
///
/// Holds an in-memory mirror of the file (identical in shape to
/// [`InMemoryStore`](super::builtin::InMemoryStore)) and appends one
/// compact-JSON line per stored entry, so `store` stays O(1) regardless
/// of store size. Construction via [`open`](Self::open) loads an
/// existing file — memory written by a previous process comes back on
/// the next launch; [`new`](Self::new) starts fresh without reading.
///
/// # On-disk format
///
/// One [`MemoryEntry`] as compact JSON per
/// line, UTF-8, `\n`-terminated. [`store`](LoopMemory::store) appends a
/// single line; [`consolidate`](LoopMemory::consolidate) and
/// [`flush`](Self::flush) rewrite the whole file atomically (write to a
/// sibling temp file, then `rename`). A torn final line — a crash
/// mid-append — is dropped with a warning and repaired (its tail
/// rewritten away) on the next `open`, so a later append cannot weld
/// itself onto the fragment; a malformed line anywhere earlier is
/// surfaced as an error, since it indicates real corruption rather than
/// an interrupted write.
///
/// # Concurrency
///
/// Safe to share via `Arc<FileMemoryStore>` within one process: the
/// internal `RwLock` serializes writers, and every file-mutating path
/// (`store`'s append, `flush`'s and `consolidate`'s rewrite) holds the
/// write lock across its I/O, so concurrent stores can never interleave
/// partial lines and concurrent rewrites can never collide on the temp
/// file or strand an append on a replaced file. Multi-process access to
/// the same file is not supported — there is no file locking, and two
/// processes will corrupt the file.
///
/// # Example
///
/// ```rust
/// use loopctl::memory::FileMemoryStore;
/// use loopctl::memory::{LoopMemory, MemoryEntry, MemoryCategory};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let dir = std::env::temp_dir().join(format!(
///     "loopctl-file-memory-doctest-{}",
///     std::process::id()
/// ));
/// std::fs::create_dir_all(&dir).unwrap();
/// let path = dir.join("agent-memory.jsonl");
/// let store = FileMemoryStore::open(&path).unwrap();
/// store
///     .store(MemoryEntry::new(MemoryCategory::Insight, "prefer Glob for file search"))
///     .await
///     .unwrap();
/// drop(store);
///
/// let reopened = FileMemoryStore::open(&path).unwrap();
/// assert_eq!(reopened.len(), 1, "memory survives the restart");
/// # });
/// ```
pub struct FileMemoryStore {
    /// Path of the JSONL file backing the store.
    ///
    /// Created empty by [`open`](Self::open) when missing, and rewritten
    /// atomically by [`flush`](Self::flush); the store never removes it.
    path: PathBuf,

    /// The stored entries — the single source of truth while the store is open.
    ///
    /// Mirrors the file: `store` appends to the file first, then pushes
    /// here under the same write lock, so the two can never disagree on
    /// ordering. Retrieval snapshots under a read guard that is dropped
    /// before the access log is touched.
    entries: RwLock<Vec<MemoryEntry>>,

    /// Access stamps recorded by `retrieve()`, keyed by entry id.
    ///
    /// A side log under its own lock, following the
    /// [`InMemoryStore`](super::builtin::InMemoryStore) pattern: only
    /// query-matched entries that were actually delivered are stamped,
    /// and the next consolidation pass folds the stamps into
    /// [`last_accessed`](MemoryEntry::last_accessed) and
    /// [`access_count`](MemoryEntry::access_count) before clearing the
    /// log.
    access_log: Mutex<HashMap<Uuid, SystemTime>>,

    /// The configuration driving this store's consolidation pass.
    ///
    /// Defaults to the same configuration
    /// [`InMemoryStore`](super::builtin::InMemoryStore) uses; set per
    /// store with [`with_consolidation`](Self::with_consolidation).
    consolidation: ConsolidationConfig,
}

impl FileMemoryStore {
    /// Open (or create) a store backed by `path`, loading existing entries.
    ///
    /// If `path` does not exist it is created empty. If it exists, each
    /// line is deserialized as one [`MemoryEntry`]: a malformed final
    /// line is a torn write and is dropped with a `tracing` warning —
    /// and the file is repaired, its tail rewritten away, so a later
    /// `store` cannot weld itself onto the fragment — while a malformed
    /// line anywhere earlier is real corruption and fails the open.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Memory`] if the path cannot be created or
    /// read, if any non-final line fails to deserialize, or if the
    /// torn-tail repair rewrite fails.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoopError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            std::fs::File::create(&path)
                .map_err(|e| LoopError::Memory(format!("cannot create memory file: {e}")))?;
        }
        let (entries, torn_tail) = Self::load_entries(&path)?;
        if torn_tail {
            Self::rewrite(&path, &entries)?;
        }
        Ok(Self {
            path,
            entries: RwLock::new(entries),
            access_log: Mutex::new(HashMap::new()),
            consolidation: ConsolidationConfig::default(),
        })
    }

    /// Create a fresh empty store that persists to `path` on first `store`.
    ///
    /// Does not read or create the file — use [`open`](Self::open) to
    /// load an existing one. The first [`store`](LoopMemory::store) or
    /// [`flush`](Self::flush) creates it.
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            entries: RwLock::new(Vec::new()),
            access_log: Mutex::new(HashMap::new()),
            consolidation: ConsolidationConfig::default(),
        }
    }

    /// Set the configuration driving this store's consolidation pass.
    ///
    /// Builder-style, mirroring
    /// [`InMemoryStore::with_consolidation`](super::builtin::InMemoryStore::with_consolidation);
    /// the config is read-only after construction.
    #[must_use]
    pub fn with_consolidation(mut self, config: ConsolidationConfig) -> Self {
        self.consolidation = config;
        self
    }

    /// Persist the full in-memory state, rewriting the file atomically.
    ///
    /// Used internally by [`consolidate`](LoopMemory::consolidate); also
    /// exposed for callers that want a checkpoint after bulk mutation.
    /// The rewrite is atomic — a sibling temp file is written, synced,
    /// and renamed over the target — so a crash mid-rewrite leaves the
    /// previous file intact. The entries write lock is held across the
    /// whole rewrite, so concurrent `flush`es, `consolidate`s, and
    /// `store`s are serialized: two rewrites can never interleave on the
    /// temp file, and an append can never land on the unlinked old file
    /// after the rename.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] on any I/O or serialization failure. On
    /// failure the original file is left untouched.
    pub fn flush(&self) -> Result<(), LoopError> {
        let entries = recover_guard(self.entries.write());
        Self::rewrite(&self.path, &entries)
    }

    /// Load a JSONL file into a vec of entries, applying the corruption rules.
    ///
    /// A missing file yields an empty store. Blank lines are skipped.
    /// A malformed final line is a torn write — skipped with a warning,
    /// and reported in the returned flag so the caller repairs the file
    /// — while a malformed earlier line is surfaced as an error.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be read, or if any
    /// non-final line fails to deserialize.
    fn load_entries(path: &Path) -> Result<(Vec<MemoryEntry>, bool), LoopError> {
        let contents = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), false)),
            Err(e) => return Err(LoopError::Memory(format!("cannot read memory file: {e}"))),
        };
        let mut entries = Vec::new();
        let mut torn_tail = false;
        let lines: Vec<&str> = contents.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    let is_last = index.saturating_add(1) == lines.len();
                    if is_last {
                        torn_tail = true;
                        tracing::warn!(error = %e, "dropping a torn final memory line");
                    } else {
                        return Err(LoopError::Memory(format!(
                            "corrupt memory line {}: {e}",
                            index.saturating_add(1)
                        )));
                    }
                }
            }
        }
        Ok((entries, torn_tail))
    }

    /// Append one serialized entry as a line to the file.
    ///
    /// The caller holds the entries write lock, so appends are serialized
    /// and lines can never interleave. The line is flushed to the OS —
    /// visible to other processes and safe against a process crash, but
    /// not fsynced; callers needing power-loss durability call
    /// [`flush`](Self::flush).
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be opened for append or
    /// the line cannot be written and flushed.
    fn append_line(path: &Path, line: &str) -> Result<(), LoopError> {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| LoopError::Memory(format!("cannot open memory file for append: {e}")))?;
        file.write_all(line.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .and_then(|()| file.flush())
            .map_err(|e| LoopError::Memory(format!("cannot append to memory file: {e}")))
    }

    /// Rewrite `path` with `entries`, atomically.
    ///
    /// Writes a sibling temp file (target path plus a `.tmp` suffix),
    /// syncs it, and renames it over the target — readers observe either
    /// the old or the new file, never a partial one. The temp file is
    /// removed on failure.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if any entry cannot be serialized, or the
    /// temp file cannot be written, synced, or renamed over the target.
    /// The original file is left untouched.
    fn rewrite(path: &Path, entries: &[MemoryEntry]) -> Result<(), LoopError> {
        use std::io::Write as _;
        let mut temp_name = path.as_os_str().to_os_string();
        temp_name.push(".tmp");
        let temp_path = PathBuf::from(temp_name);
        let write_result = (|| -> Result<(), LoopError> {
            let mut buffer = String::new();
            for entry in entries {
                let line = serde_json::to_string(entry).map_err(|e| {
                    LoopError::Memory(format!("memory entry serialization failed: {e}"))
                })?;
                buffer.push_str(&line);
                buffer.push('\n');
            }
            let mut file = std::fs::File::create(&temp_path)
                .map_err(|e| LoopError::Memory(format!("cannot create rewrite temp file: {e}")))?;
            file.write_all(buffer.as_bytes())
                .and_then(|()| file.sync_all())
                .map_err(|e| LoopError::Memory(format!("cannot write rewrite temp file: {e}")))?;
            std::fs::rename(&temp_path, path)
                .map_err(|e| LoopError::Memory(format!("cannot finalize memory rewrite: {e}")))
        })();
        if write_result.is_err() {
            drop(std::fs::remove_file(&temp_path));
        }
        write_result
    }
}

impl LoopMemory for FileMemoryStore {
    /// Append `entry` to the file and the in-memory mirror.
    ///
    /// The file append happens first, under the entries write lock: on
    /// success the mirror is updated to match; on failure neither side
    /// changes, so the two can never disagree.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let line = serde_json::to_string(&entry).map_err(|e| {
                LoopError::Memory(format!("memory entry serialization failed: {e}"))
            })?;
            let mut entries = recover_guard(self.entries.write());
            Self::append_line(&self.path, &line)?;
            entries.push(entry);
            Ok(())
        })
    }

    /// Rank with the shared scorer, identical to [`InMemoryStore`](super::builtin::InMemoryStore).
    ///
    /// Snapshots the mirror under a read guard, scores every entry with
    /// [`score_entry`], sorts by descending
    /// composite score (stable, so equal scores keep insertion order),
    /// and stamps the access log for delivered query-matched entries —
    /// baseline-only returns are delivered but never stamped.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + 'a>> {
        let query = query.to_string();
        Box::pin(async move {
            let query_lower = query.to_lowercase();
            let query_trimmed = query_lower.trim();
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            let snapshot: Vec<MemoryEntry> =
                recover_guard(self.entries.read()).iter().cloned().collect();
            let mut scored: Vec<(f32, bool, MemoryEntry)> = snapshot
                .into_iter()
                .map(|entry| {
                    let (score, matched) = score_entry(&entry, query_trimmed, &query_words);
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

    /// Run the shared consolidation pass and rewrite the file atomically.
    ///
    /// Folds the access log into a copy of the entries (stamping
    /// [`last_accessed`](MemoryEntry::last_accessed) and bumping
    /// [`access_count`](MemoryEntry::access_count)), then delegates to
    /// [`consolidate_entries`]
    /// — the same decay, merge, and prune pass
    /// [`InMemoryStore`](super::builtin::InMemoryStore) runs — and
    /// persists the survivors with an atomic rewrite before swapping
    /// them into the mirror, so pruned entries do not come back on the
    /// next restart. On a rewrite failure the method returns `Err` with
    /// both the file and the mirror still in their pre-consolidation
    /// state — the two sides never diverge.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let now = SystemTime::now();
            let mut guard = recover_guard(self.entries.write());
            let mut next = guard.clone();
            let stats = {
                let mut access_log = recover_guard(self.access_log.lock());
                for entry in &mut next {
                    if let Some(stamp) = access_log.get(&entry.id) {
                        entry.last_accessed = entry.last_accessed.max(Some(*stamp));
                        entry.access_count = entry.access_count.saturating_add(1);
                    }
                }
                access_log.clear();
                consolidate_entries(&mut next, &self.consolidation, now)
            };
            Self::rewrite(&self.path, &next)?;
            *guard = next;
            Ok(stats)
        })
    }

    /// Number of entries currently in the mirror.
    ///
    /// Used by [`is_empty`](LoopMemory::is_empty) and reported by the
    /// engine's consolidate hook after each run.
    fn len(&self) -> usize {
        recover_guard(self.entries.read()).len()
    }
}
