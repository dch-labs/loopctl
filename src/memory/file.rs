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
//! Handles on one path within a process share state — opening the same
//! file twice is safe. Cross-process access is unsupported: two
//! processes writing the same file will interleave and corrupt it; for
//! that, use a database-backed companion such as `loopctl-sqlite`.

use crate::error::{LoopError, recover_guard};
use crate::memory::LoopMemory;
use crate::memory::consolidate::{ConsolidationConfig, consolidate_entries};
use crate::memory::entry::{ConsolidationStats, MemoryEntry};
use crate::memory::score::score_entry;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock, Weak};
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
/// mid-append, or between the line and its newline — is repaired (its
/// tail rewritten with the terminator) on the next `open`, so a later
/// append cannot weld itself onto it; an undecodable fragment is
/// additionally dropped with a warning, and a malformed complete line
/// anywhere earlier is surfaced as an error, since it indicates real
/// corruption rather than an interrupted write.
///
/// # Concurrency
///
/// Within one process this is safe at every level: sharing a handle
/// via `Arc`, and opening or constructing several handles for the same
/// path — they all attach to one shared state, so every file-mutating
/// path (`store`'s append, `flush`'s and `consolidate`'s rewrite) goes
/// through the same locks: concurrent stores can never interleave
/// partial lines, concurrent rewrites can never collide on the temp
/// file or strand an append on a replaced file, and no handle can
/// rewrite the file without the others' entries. Multi-process access
/// to the same file is not supported — there is no file locking, and
/// two processes will corrupt the file.
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
    /// The state shared by every live handle on this file path.
    ///
    /// Appends, rewrites, and the access log all serialize behind the
    /// shared locks inside, so two handles constructed for one path —
    /// `open`, `new`, or a mix — observe and mutate the same store.
    inner: Arc<Inner>,

    /// The configuration driving this handle's consolidation pass.
    ///
    /// Defaults to the same configuration
    /// [`InMemoryStore`](super::builtin::InMemoryStore) uses; set per
    /// handle with [`with_consolidation`](Self::with_consolidation). A
    /// pass runs with the consolidating handle's config over the shared
    /// entries.
    consolidation: ConsolidationConfig,
}

/// The mutable state shared by every live [`FileMemoryStore`] on one path.
///
/// One `Inner` exists per live path per process (the registry in
/// [`live_stores`] enforces that), so every file-mutating path through
/// any handle serializes behind these locks.
struct Inner {
    /// Path of the JSONL file backing the store.
    ///
    /// Created empty by [`open`](FileMemoryStore::open) when missing,
    /// and rewritten atomically by [`flush`](FileMemoryStore::flush);
    /// the store never removes it.
    path: PathBuf,

    /// The stored entries — the single source of truth while any handle is live.
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

    /// Whether an append failure could not even be truncated back.
    ///
    /// Set when a partial line was written and restoring the original
    /// file length failed — the file may now hold a partial line, so
    /// every further append through any handle is rejected until a
    /// fresh [`open`](FileMemoryStore::open) loads and repairs the
    /// file.
    append_unusable: AtomicBool,
}

impl Inner {
    /// Assemble the shared state for one file path.
    ///
    /// `entries` is whatever the constructing call loaded (or nothing,
    /// for a fresh `new`); the registry call site is responsible for
    /// registering the result.
    fn new(path: PathBuf, entries: Vec<MemoryEntry>) -> Self {
        Self {
            path,
            entries: RwLock::new(entries),
            access_log: Mutex::new(HashMap::new()),
            append_unusable: AtomicBool::new(false),
        }
    }
}

/// The process-wide registry of live shared store states, by path.
///
/// Constructors look a path up here and attach to the live state when
/// one exists, so two handles on one file share locks and mirror
/// instead of racing each other into data loss. Dead entries are
/// pruned opportunistically on lookup.
fn live_stores() -> &'static Mutex<HashMap<PathBuf, Weak<Inner>>> {
    static LIVE: OnceLock<Mutex<HashMap<PathBuf, Weak<Inner>>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How many final-component symlinks registry key derivation will follow.
///
/// Bounds the chain walk below the kernel's own `ELOOP` threshold; a
/// loop of links simply exhausts the budget and keys by the last
/// resolved spelling, which every handle through the loop derives
/// identically.
const SYMLINK_FOLLOW_LIMIT: usize = 8;

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
    /// A live handle for this path is attached to instead of loading —
    /// both handles then share one mirror and one set of locks, so
    /// neither can rewrite the file without the other's entries. The
    /// load, the torn-tail check, and the repair below run only when no
    /// live handle exists.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Memory`] if the path cannot be created,
    /// resolved, or read, if any non-final line fails to deserialize,
    /// or if the torn-tail repair rewrite fails.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoopError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            std::fs::File::create(&path)
                .map_err(|e| LoopError::Memory(format!("cannot create memory file: {e}")))?;
        }
        let canonical = std::fs::canonicalize(&path)
            .map_err(|e| LoopError::Memory(format!("cannot resolve memory file path: {e}")))?;
        let inner = {
            let mut registry = recover_guard(live_stores().lock());
            let attached = registry.get(&canonical).and_then(Weak::upgrade);
            if let Some(inner) = attached {
                inner
            } else {
                registry.remove(&canonical);
                let (entries, torn_tail) = Self::load_entries(&canonical)?;
                if torn_tail {
                    Self::rewrite(&canonical, &entries)?;
                }
                let inner = Arc::new(Inner::new(canonical.clone(), entries));
                registry.insert(canonical, Arc::downgrade(&inner));
                inner
            }
        };
        Ok(Self {
            inner,
            consolidation: ConsolidationConfig::default(),
        })
    }

    /// Create a fresh empty store that persists to `path` on first `store`.
    ///
    /// Attaches to the live shared store for the path when one exists
    /// (both handles then observe the same entries); otherwise it
    /// starts empty without reading or creating the file — the first
    /// [`store`](LoopMemory::store) or [`flush`](Self::flush) creates
    /// it. Entries already in an existing file are neither loaded nor
    /// preserved: the next `flush` or `consolidate` persists only what
    /// the live handles stored. Use [`open`](Self::open) to load an
    /// existing file when no handle is live.
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        let key = Self::registry_key(path.as_ref());
        let inner = {
            let mut registry = recover_guard(live_stores().lock());
            let attached = registry.get(&key).and_then(Weak::upgrade);
            if let Some(inner) = attached {
                inner
            } else {
                registry.remove(&key);
                let inner = Arc::new(Inner::new(key.clone(), Vec::new()));
                registry.insert(key, Arc::downgrade(&inner));
                inner
            }
        };
        Self {
            inner,
            consolidation: ConsolidationConfig::default(),
        }
    }

    /// Resolve a dangling symlink's final component to its target.
    ///
    /// Returns `Some` only when `path` itself is a symlink that does
    /// not resolve to an existing file (the caller has already failed
    /// to canonicalize it): the target is read, and a relative target
    /// is anchored at the canonicalized parent of the link. A `None`
    /// covers every other shape — an ordinary missing path, or a
    /// symlink whose metadata cannot be read.
    fn dangling_final_symlink(path: &Path) -> Option<PathBuf> {
        let metadata = std::fs::symlink_metadata(path).ok()?;
        if !metadata.file_type().is_symlink() {
            return None;
        }
        let target = std::fs::read_link(path).ok()?;
        if target.is_absolute() {
            Some(target)
        } else {
            let parent = path.parent()?.canonicalize().ok()?;
            Some(parent.join(target))
        }
    }

    /// Derive the registry key for a path that may not exist yet.
    ///
    /// An existing file canonicalizes outright — including through
    /// symlinked directories and symlinked final components. A dangling
    /// final symlink is resolved to its target first (following chains,
    /// bounded by [`SYMLINK_FOLLOW_LIMIT`]), so a store created through
    /// the link keys by where its data will really live — matching what
    /// [`open`](Self::open) derives once the first `store` creates the
    /// target through the link. What remains is a genuinely missing
    /// final component, which carries no symlink ambiguity of its own:
    /// canonicalizing the parent and re-joining the file name matches
    /// what `open` will derive once it creates the file. Only when the
    /// parent itself cannot be resolved (missing as well, or gone
    /// between the two lookups) does the key fall back to the lexical
    /// `absolute` form.
    fn registry_key(path: &Path) -> PathBuf {
        let mut current = path.to_path_buf();
        for _ in 0..SYMLINK_FOLLOW_LIMIT {
            if let Ok(canonical) = std::fs::canonicalize(&current) {
                return canonical;
            }
            match Self::dangling_final_symlink(&current) {
                Some(target) => current = target,
                None => break,
            }
        }
        if let Some(key) = current
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .and_then(|parent| current.file_name().map(|name| parent.join(name)))
        {
            return key;
        }
        std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
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
        let entries = recover_guard(self.inner.entries.write());
        Self::rewrite(&self.inner.path, &entries)
    }

    /// Load a JSONL file into a vec of entries, applying the corruption rules.
    ///
    /// The file is read as raw bytes and split at newline boundaries, so
    /// a torn final fragment that ends mid–multi-byte character is
    /// droppable like any other torn write — a strict UTF-8 read of the
    /// whole file would reject valid complete lines over one broken
    /// tail byte. A missing file yields an empty store. Blank lines are
    /// skipped. A trailing fragment — bytes after the last newline —
    /// means the file lost its terminating newline, whatever those bytes
    /// hold: an undecodable fragment is a torn write (dropped with a
    /// warning) and a decodable one loads, but either way the returned
    /// flag tells the caller to repair, since the next append would
    /// weld onto an unterminated line. A malformed complete line
    /// anywhere is surfaced as an error.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be read, or if any
    /// complete line is not valid UTF-8 or fails to deserialize.
    fn load_entries(path: &Path) -> Result<(Vec<MemoryEntry>, bool), LoopError> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), false)),
            Err(e) => return Err(LoopError::Memory(format!("cannot read memory file: {e}"))),
        };
        let (complete, trailing) = split_complete_lines(&bytes);
        let mut entries = Vec::new();
        let mut torn_tail = false;
        for (index, line_bytes) in complete.iter().enumerate() {
            let line = match std::str::from_utf8(line_bytes) {
                Ok(line) => line,
                Err(e) => {
                    return Err(LoopError::Memory(format!(
                        "corrupt memory line {}: not valid UTF-8: {e}",
                        index.saturating_add(1)
                    )));
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    return Err(LoopError::Memory(format!(
                        "corrupt memory line {}: {e}",
                        index.saturating_add(1)
                    )));
                }
            }
        }
        if let Some(fragment) = trailing {
            torn_tail = true;
            let decodable = std::str::from_utf8(fragment)
                .ok()
                .and_then(|text| serde_json::from_str(text).ok());
            if let Some(entry) = decodable {
                entries.push(entry);
            } else {
                tracing::warn!("dropping a torn final memory line");
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
    /// [`flush`](Self::flush). If the write or flush fails partway, the
    /// file is truncated back to its pre-append length so a partial line
    /// cannot sit in later appends' way; if even the truncation fails,
    /// the store marks itself unusable and every further append is
    /// rejected until a fresh [`open`](Self::open) repairs the file.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be opened for append,
    /// the line cannot be written and flushed, or the store is unusable
    /// after an earlier unrecoverable append failure.
    fn append_line(&self, line: &str) -> Result<(), LoopError> {
        use std::io::{Seek as _, SeekFrom, Write as _};
        use std::sync::atomic::Ordering as AtomicOrdering;
        if self.inner.append_unusable.load(AtomicOrdering::SeqCst) {
            return Err(LoopError::Memory(
                "the memory file is unusable after an unrecoverable append failure; drop every live handle and reopen to repair it"
                    .to_string(),
            ));
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.inner.path)
            .map_err(|e| LoopError::Memory(format!("cannot open memory file for append: {e}")))?;
        let original_len = file
            .seek(SeekFrom::End(0))
            .map_err(|e| LoopError::Memory(format!("cannot size the memory file: {e}")))?;
        let write_result = file
            .write_all(line.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .and_then(|()| file.flush());
        if let Err(e) = write_result {
            let truncation = file.set_len(original_len).and_then(|()| file.flush());
            if truncation.is_err() {
                self.inner
                    .append_unusable
                    .store(true, AtomicOrdering::SeqCst);
            }
            return Err(LoopError::Memory(format!(
                "cannot append to memory file: {e}"
            )));
        }
        Ok(())
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

/// Split raw file bytes into complete lines and the trailing fragment.
///
/// Complete lines are the newline-terminated segments; the fragment is
/// whatever follows the last newline (`None` when the file ends with a
/// newline or is empty). A torn write's partial bytes can only ever be
/// the fragment, so decoding it may fail without condemning the
/// complete lines before it.
fn split_complete_lines(bytes: &[u8]) -> (Vec<&[u8]>, Option<&[u8]>) {
    let split_point = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at.saturating_add(1));
    let (complete_region, fragment) = bytes.split_at(split_point);
    let complete = complete_region
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect();
    let trailing = if fragment.is_empty() {
        None
    } else {
        Some(fragment)
    };
    (complete, trailing)
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
            let mut entries = recover_guard(self.inner.entries.write());
            self.append_line(&line)?;
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
            let snapshot: Vec<MemoryEntry> = recover_guard(self.inner.entries.read())
                .iter()
                .cloned()
                .collect();
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
            let mut access_log = recover_guard(self.inner.access_log.lock());
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
    /// the file, the mirror, and the pending access stamps all still in
    /// their pre-consolidation state — the stamps are only cleared once
    /// the rewrite has succeeded, held under the access-log lock across
    /// it, so the next pass re-folds them instead of losing them.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        Box::pin(async move {
            let now = SystemTime::now();
            let mut guard = recover_guard(self.inner.entries.write());
            let mut next = guard.clone();
            let mut access_log = recover_guard(self.inner.access_log.lock());
            for entry in &mut next {
                if let Some(stamp) = access_log.get(&entry.id) {
                    entry.last_accessed = entry.last_accessed.max(Some(*stamp));
                    entry.access_count = entry.access_count.saturating_add(1);
                }
            }
            let stats = consolidate_entries(&mut next, &self.consolidation, now);
            Self::rewrite(&self.inner.path, &next)?;
            access_log.clear();
            drop(access_log);
            *guard = next;
            Ok(stats)
        })
    }

    /// Number of entries currently in the mirror.
    ///
    /// Used by [`is_empty`](LoopMemory::is_empty) and reported by the
    /// engine's consolidate hook after each run.
    fn len(&self) -> usize {
        recover_guard(self.inner.entries.read()).len()
    }
}

#[cfg(test)]
mod tests {
    use super::FileMemoryStore;
    use super::split_complete_lines;

    fn unit_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-key-unit-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    #[test]
    fn an_existing_file_keys_by_its_canonical_location() {
        use std::os::unix::fs::symlink;

        let dir = unit_dir("existing");
        let real = dir.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = dir.join("link");
        symlink(&real, &link).unwrap();
        let through_link = link.join("memory.jsonl");
        std::fs::File::create(&through_link).unwrap();

        assert_eq!(
            FileMemoryStore::registry_key(&through_link),
            real.join("memory.jsonl"),
            "an existing file keys by where it really lives, resolving the \
            symlinked directory"
        );

        let file_link = dir.join("mem-link.jsonl");
        symlink(real.join("memory.jsonl"), &file_link).unwrap();
        assert_eq!(
            FileMemoryStore::registry_key(&file_link),
            real.join("memory.jsonl"),
            "a symlinked file keys by its target — only the outright \
            canonicalization resolves a symlink on the file itself"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_missing_file_keys_by_its_canonical_parent() {
        use std::os::unix::fs::symlink;

        let dir = unit_dir("missing");
        let real = dir.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = dir.join("link");
        symlink(&real, &link).unwrap();
        let absent_through_link = link.join("memory.jsonl");

        assert_eq!(
            FileMemoryStore::registry_key(&absent_through_link),
            real.join("memory.jsonl"),
            "a missing final component carries no ambiguity of its own — \
            the key is the canonical parent plus the name, exactly what \
            open derives once it creates the file"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_final_symlink_keys_by_its_target() {
        use std::os::unix::fs::symlink;

        let dir = unit_dir("dangling");
        let elsewhere = dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();

        let absolute_link = dir.join("abs-link.jsonl");
        symlink(elsewhere.join("target.jsonl"), &absolute_link).unwrap();
        assert_eq!(
            FileMemoryStore::registry_key(&absolute_link),
            elsewhere.join("target.jsonl"),
            "a dangling final symlink keys by its absolute target — where \
            the first store will create the data"
        );

        let relative_link = dir.join("rel-link.jsonl");
        symlink(
            std::path::Path::new("elsewhere/rel-target.jsonl"),
            &relative_link,
        )
        .unwrap();
        assert_eq!(
            FileMemoryStore::registry_key(&relative_link),
            elsewhere.join("rel-target.jsonl"),
            "a relative target is anchored at the canonicalized parent of \
            the link before keying"
        );

        let chain_end = elsewhere.join("chain-end.jsonl");
        let chain_mid = elsewhere.join("chain-mid.jsonl");
        symlink(&chain_end, &chain_mid).unwrap();
        let chain_link = dir.join("chain-link.jsonl");
        symlink(&chain_mid, &chain_link).unwrap();
        assert_eq!(
            FileMemoryStore::registry_key(&chain_link),
            elsewhere.join("chain-end.jsonl"),
            "a chain of dangling links is followed to its end"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unresolvable_path_falls_back_to_the_lexical_absolute_form() {
        let absent = std::path::Path::new("/nonexistent-file-memory-unit/parent/mem.jsonl");
        assert_eq!(
            FileMemoryStore::registry_key(absent),
            absent.to_path_buf(),
            "with neither the file nor its parent resolvable, the already \
            absolute spelling is returned as-is"
        );
    }

    #[test]
    fn an_empty_path_falls_back_to_the_raw_spelling() {
        let empty = std::path::Path::new("");
        assert_eq!(
            FileMemoryStore::registry_key(empty),
            empty.to_path_buf(),
            "the empty path resolves nowhere at any stage — the final \
            fallback returns the raw spelling instead of panicking"
        );
    }

    #[test]
    fn a_trailing_dotdot_still_yields_a_key() {
        let edge = std::path::Path::new("/nonexistent-file-memory-unit/..");
        assert_eq!(
            FileMemoryStore::registry_key(edge),
            edge.to_path_buf(),
            "a path with no file name component skips the parent-join arm \
            without panicking and lands in the lexical fallback"
        );
    }

    #[test]
    fn a_file_ending_in_a_newline_has_no_trailing_fragment() {
        let (complete, trailing) = split_complete_lines(b"one\ntwo\n");
        assert_eq!(complete, vec![b"one".as_slice(), b"two".as_slice()]);
        assert!(trailing.is_none(), "nothing follows the last newline");
    }

    #[test]
    fn bytes_after_the_last_newline_form_the_fragment() {
        let (complete, trailing) = split_complete_lines(b"one\ntw");
        assert_eq!(complete, vec![b"one".as_slice()]);
        assert_eq!(trailing, Some(b"tw".as_slice()));
    }

    #[test]
    fn a_file_with_no_newline_is_all_fragment() {
        let (complete, trailing) = split_complete_lines(b"torn");
        assert!(complete.is_empty(), "no newline means no complete line");
        assert_eq!(trailing, Some(b"torn".as_slice()));
    }

    #[test]
    fn an_empty_file_yields_no_lines_and_no_fragment() {
        let (complete, trailing) = split_complete_lines(b"");
        assert!(complete.is_empty());
        assert!(trailing.is_none());
    }

    #[test]
    fn append_line_writes_the_line_and_a_trailing_newline() {
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-append-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("append.jsonl");
        assert!(
            !path.try_exists().unwrap(),
            "new() does not touch the disk, so the file starts absent"
        );

        let store = super::FileMemoryStore::new(&path);
        store.append_line("{\"line\":1}").unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"{\"line\":1}\n",
            "an append is exactly the line plus its terminating newline"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rewrite_emits_one_line_per_entry_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-rewrite-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rewrite.jsonl");

        let entries = vec![
            crate::memory::MemoryEntry::new(crate::memory::MemoryCategory::Fact, "first"),
            crate::memory::MemoryEntry::new(crate::memory::MemoryCategory::Fact, "second"),
        ];
        super::FileMemoryStore::rewrite(&path, &entries).unwrap();

        let rewritten = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = rewritten.lines().collect();
        assert_eq!(lines.len(), 2, "one line per entry: {rewritten}");
        assert!(
            rewritten.ends_with('\n'),
            "the file is newline-terminated: {rewritten:?}"
        );
        assert!(
            !dir.join("rewrite.jsonl.tmp").try_exists().unwrap(),
            "a successful rename consumes the temp file"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn flush_creates_the_file_from_new_and_persists_the_mirror() {
        use crate::memory::LoopMemory as _;

        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-flush-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flush.jsonl");

        let store = super::FileMemoryStore::new(&path);
        store.flush().unwrap();
        assert!(
            path.try_exists().unwrap() && std::fs::read_to_string(&path).unwrap().is_empty(),
            "flushing an empty mirror creates an empty file"
        );

        store
            .store(crate::memory::MemoryEntry::new(
                crate::memory::MemoryCategory::Fact,
                "flushed",
            ))
            .await
            .unwrap();
        store.flush().unwrap();
        let flushed = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            flushed.lines().count(),
            1,
            "the mirror is on disk: {flushed}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_entries_reports_a_missing_file_as_empty_and_intact() {
        let (entries, torn) = super::FileMemoryStore::load_entries(std::path::Path::new(
            "/nonexistent-file-memory-probe",
        ))
        .unwrap();
        assert!(entries.is_empty());
        assert!(!torn, "nothing was dropped, so there is nothing to repair");
    }

    #[test]
    fn load_entries_flags_a_undecodable_trailing_fragment() {
        let dir =
            std::env::temp_dir().join(format!("loopctl-file-memory-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flag.jsonl");
        std::fs::write(&path, b"{\"torn\"").unwrap();
        let (entries, torn) = super::FileMemoryStore::load_entries(&path).unwrap();
        assert!(entries.is_empty());
        assert!(torn, "the caller must repair a torn tail");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_entries_keeps_a_complete_undecodable_line_loud() {
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-loud-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("loud.jsonl");
        std::fs::write(&path, b"not json\nvalid later\n").unwrap();
        assert!(
            super::FileMemoryStore::load_entries(&path).is_err(),
            "a complete malformed line is corruption, not a torn tail"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn blank_lines_do_not_reach_the_complete_list() {
        let (complete, trailing) = split_complete_lines(b"\n\none\n\n");
        assert_eq!(complete, vec![b"one".as_slice()]);
        assert!(trailing.is_none());
    }
}
