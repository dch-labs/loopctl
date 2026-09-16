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
use std::collections::{HashMap, HashSet};
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
/// rewrite the file without the others' entries. A handle constructed
/// while a parent symlink was still dangling keys provisionally by
/// the given spelling; the next construction re-derives that key and
/// moves the registration — and spellings that collide, having
/// resolved to one file, unify: their mirrors merge in the file's
/// append order and the stale handle is re-pointed, so every handle
/// shares one state as soon as any construction or rewrite observes
/// the paths converged — a flush or consolidation on a converged
/// handle unifies first, then rewrites from the one mirror.
/// Multi-process access to the same file is not supported — there is
/// no file locking, and two processes will corrupt the file.
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
    /// The redirectable state shared by every live handle on this file path.
    ///
    /// Operations pin the current shared state for their whole duration
    /// under a read guard (see [`SharedInner`]), so handles — however
    /// constructed, and re-pointed by a spelling unification — always
    /// observe one store.
    inner: Arc<SharedInner>,

    /// The configuration driving this handle's consolidation pass.
    ///
    /// Defaults to the same configuration
    /// [`InMemoryStore`](super::builtin::InMemoryStore) uses; set per
    /// handle with [`with_consolidation`](Self::with_consolidation). A
    /// pass runs with the consolidating handle's config over the shared
    /// entries.
    consolidation: ConsolidationConfig,
}

/// The mutable state of one store, held as the current value of a
/// [`SharedInner`].
///
/// One `Inner` serves every handle that converged on one registry key;
/// a unification of two pre-resolution spellings retires the stale one
/// and re-points its holders, so every file-mutating path through any
/// handle serializes behind these locks.
struct Inner {
    /// Path of the JSONL file backing the store.
    ///
    /// Created empty by [`open`](FileMemoryStore::open) when missing,
    /// and rewritten atomically by [`flush`](FileMemoryStore::flush);
    /// the store never removes it. A store registered under a
    /// provisional spelling keeps that spelling only until migration
    /// converges it onto a re-derived key — a write that happens only
    /// inside migration, at most once per convergence event (a target
    /// unified by several stale spellings takes one identical write
    /// each), which is why readers take a brief read guard
    /// and clone (see [`backing_path`](Inner::backing_path)) instead
    /// of holding the guard across file operations.
    path: RwLock<PathBuf>,

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

    /// Whether an append failed and left the tail suspect.
    ///
    /// Set when a write or flush fails — the file may hold a partial
    /// line, and no rollback is attempted because restoring the
    /// pre-append length could delete a concurrent writer's line that
    /// landed in the window, so every further append through a handle
    /// sharing this state is rejected until a repair: a fresh
    /// [`open`](FileMemoryStore::open) loads and fixes the file, or a
    /// [`flush`](FileMemoryStore::flush) or consolidation rewrites it
    /// whole from the intact mirror — both clear the latch. A
    /// sibling handle on a second,
    /// not-yet-converged spelling shares no state with this one, so
    /// its appends are not blocked here; one landing after such a
    /// partial write turns the repairable tail into a mid-file line
    /// the next `open` reports loudly. Any construction or rewrite
    /// converges the spellings and closes that window.
    append_unusable: AtomicBool,

    /// Whether this store was registered under a provisional key.
    ///
    /// Set when construction fell back to the lexical spelling because
    /// nothing on the path resolved yet — a parent symlink still
    /// dangling — or when a symlink chain outran the follow budget and
    /// the deepest spelling reached may be resolvable in principle;
    /// later constructions re-derive the key and move or unify the
    /// registration once the world resolves. Never cleared —
    /// re-derivation is idempotent, so the flag only marks the entry
    /// as worth re-checking.
    is_fallback: bool,
}

impl Inner {
    /// Assemble the shared state for one file path.
    ///
    /// `entries` is whatever the constructing call loaded (or nothing,
    /// for a fresh `new`); `is_fallback` mirrors the key derivation's
    /// provisional flag; the registry call site is responsible for
    /// registering the result.
    fn new(path: PathBuf, entries: Vec<MemoryEntry>, is_fallback: bool) -> Self {
        Self {
            path: RwLock::new(path),
            entries: RwLock::new(entries),
            access_log: Mutex::new(HashMap::new()),
            append_unusable: AtomicBool::new(false),
            is_fallback,
        }
    }

    /// The current backing-file path, cloned under a brief read guard.
    ///
    /// The path converges at most once per store — when a provisional
    /// registration moves or unifies onto a re-derived key — so the
    /// clone is stable for the duration of any single operation that
    /// holds it.
    fn backing_path(&self) -> PathBuf {
        recover_guard(self.path.read()).clone()
    }

    /// Append one serialized entry as a line to the file.
    ///
    /// The caller holds the entries write lock, so appends through one
    /// shared state are serialized and their lines can never
    /// interleave; the line and its terminator go out as a single
    /// write, so even appenders through a not-yet-converged second
    /// spelling of the same file — a second lock domain the caller
    /// cannot cover — can never weld a line mid-file. The line is
    /// flushed to the OS — visible to other processes and safe against
    /// a process crash, but not fsynced; callers needing power-loss
    /// durability call [`FileMemoryStore::flush`]. If the write or
    /// flush fails, the failed append never removes bytes: a rollback
    /// to the pre-append length could delete a concurrent writer's
    /// line that landed in the window, so any partial bytes stay for
    /// the reopen's torn-tail repair, and the store marks itself
    /// unusable — every further append through a handle sharing this
    /// state is rejected until a fresh [`FileMemoryStore::open`] loads
    /// and repairs the file.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be opened for append,
    /// the line cannot be written and flushed, or the store is unusable
    /// after an earlier unrecoverable append failure.
    fn append_line(&self, line: &str) -> Result<(), LoopError> {
        use std::io::Write as _;
        use std::sync::atomic::Ordering as AtomicOrdering;
        if self.append_unusable.load(AtomicOrdering::SeqCst) {
            return Err(LoopError::Memory(
                "the memory file is unusable after an unrecoverable append failure; drop every live handle and reopen to repair it"
                    .to_string(),
            ));
        }
        let backing_path = self.backing_path();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&backing_path)
            .map_err(|e| LoopError::Memory(format!("cannot open memory file for append: {e}")))?;
        let mut line_bytes = line.as_bytes().to_vec();
        line_bytes.push(b'\n');
        let write_result = file.write_all(&line_bytes).and_then(|()| file.flush());
        if let Err(e) = write_result {
            self.append_unusable.store(true, AtomicOrdering::SeqCst);
            return Err(LoopError::Memory(format!(
                "cannot append to memory file: {e}"
            )));
        }
        Ok(())
    }
}

/// The redirectable cell holding the current [`Inner`] of one store.
///
/// Every handle holds one of these instead of the [`Inner`] itself, so
/// a unification of two pre-resolution spellings of one file can
/// re-point the stale cell's holders at the unified state. Operations
/// pin the current `Inner` for their whole duration under a read guard
/// — the airtightness rule: a redirect happens under a write guard, so
/// it can never interleave with an operation that already observed the
/// old state, and no in-flight append can land in a mirror the redirect
/// is about to retire; the retired `Inner` drops once the last
/// in-flight operation releases its guard.
struct SharedInner {
    /// The live shared state, replaceable only by a unification.
    ///
    /// Readers hold a read guard for their entire operation; a redirect
    /// installs the unified store's [`Inner`] here under a write guard,
    /// after which every handle on this cell observes the unified
    /// store.
    current: RwLock<Arc<Inner>>,
}

impl SharedInner {
    /// Wrap one shared state as the current state of a fresh cell.
    ///
    /// Constructors call this once per new store; the cell is only
    /// re-pointed afterwards, never re-wrapped.
    fn new(inner: Arc<Inner>) -> Self {
        Self {
            current: RwLock::new(inner),
        }
    }
}

/// The process-wide registry of live shared store states, by path.
///
/// Constructors look a path up here and attach to the live state when
/// one exists, so two handles on one file share locks and mirror
/// instead of racing each other into data loss. Dead entries are
/// pruned opportunistically on lookup.
fn live_stores() -> &'static Mutex<HashMap<PathBuf, Weak<SharedInner>>> {
    static LIVE: OnceLock<Mutex<HashMap<PathBuf, Weak<SharedInner>>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Which rewrite stage an injected fault should strike.
///
/// Exists for test determinism on privileged runners, where directory
/// permission bits cannot force rewrite failures; absent from normal
/// builds.
#[cfg(feature = "testing")]
#[derive(Clone, Copy)]
pub enum RewriteFaultStage {
    /// Fail the temp file's exclusive create (pre-commit).
    ///
    /// The rewrite returns before anything is written, exactly as a
    /// real create failure would.
    TempCreate,

    /// Fail the post-rename parent-directory sync (not durable).
    ///
    /// The rewrite returns after the rename landed, so the new file is
    /// in place and only its durability is unconfirmed — the ordering
    /// the durability pins assert.
    DirectorySync,
}

/// Register a fault for the next rewrite of exactly one path.
///
/// The registration is keyed exactly as the store keys the path it
/// rewrites — an existing file by its canonical location, a missing one
/// by its canonical parent plus name, a dangling final symlink by its
/// resolved target — so a registration made before the file exists, or
/// through a symlinked spelling, still matches the rewrite that later
/// strikes it. Each registration injects exactly one fault and is
/// consumed by the first matching rewrite — register N times for N
/// consecutive failures; parallel tests registering against their own
/// unique temp paths cannot interfere. Test-only surface for
/// determinism on privileged runners; absent from normal builds.
#[cfg(feature = "testing")]
pub fn fail_next_rewrite_at(path: &Path, stage: RewriteFaultStage) {
    let key = FileMemoryStore::registry_key(path).0;
    recover_guard(registered_faults().lock()).push((key, stage));
}

/// The registered per-path rewrite faults.
///
/// Static plumbing behind [`fail_next_rewrite_at`]; never touched
/// outside this module.
#[cfg(feature = "testing")]
fn registered_faults() -> &'static Mutex<Vec<(PathBuf, RewriteFaultStage)>> {
    static FAULTS: OnceLock<Mutex<Vec<(PathBuf, RewriteFaultStage)>>> = OnceLock::new();
    FAULTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Consume one fault registered for `path` when it strikes `stage`.
///
/// [`rewrite`](FileMemoryStore::rewrite) calls this at each stage it
/// can inject; exactly the first matching registration is removed, so
/// further registrations for the same path and stage stay queued for
/// later rewrites, and a registration for the other stage stays queued
/// for its own site.
#[cfg(feature = "testing")]
fn injected_fault_at(path: &Path, stage: RewriteFaultStage) -> bool {
    let mut faults = recover_guard(registered_faults().lock());
    let wanted = std::mem::discriminant(&stage);
    match faults
        .iter()
        .position(|(faulted, at)| faulted == path && std::mem::discriminant(at) == wanted)
    {
        Some(index) => {
            faults.remove(index);
            true
        }
        None => false,
    }
}

/// How many final-component symlinks registry key derivation will follow.
///
/// Bounds the chain walk below the kernel's own `ELOOP` threshold; a
/// loop of links simply exhausts the budget, and the deepest spelling
/// reached keys the store — provisionally, since the world may
/// resolve past it, so every handle through the loop derives the same
/// re-derivable key.
const SYMLINK_FOLLOW_LIMIT: usize = 8;

/// Why a file-store rewrite failed, split by how far it got.
///
/// The two classes call for opposite caller reactions: a pre-commit
/// failure leaves the file untouched, so a caller proposing a next
/// state should discard it; a committed-but-not-durable failure has
/// already replaced the file, so the proposal must be committed —
/// discarding it would leave the caller's state behind the disk.
#[derive(Debug)]
enum RewriteError {
    /// The rewrite never reached the rename; the file is untouched.
    ///
    /// Serialization, temp-file creation, writing, syncing, or the
    /// rename itself failed — the caller's proposed next state should
    /// be discarded, since the disk never saw it.
    PreCommit(LoopError),

    /// The rename landed; the parent-directory sync failed. The new
    /// file is in place — only its durability is unconfirmed.
    ///
    /// The file already holds the rewritten entries; a caller holding
    /// a proposed next state must commit it to stay level with the
    /// disk, and may surface the inner error to report the unconfirmed
    /// durability.
    NotDurable(LoopError),
}

impl From<RewriteError> for LoopError {
    fn from(error: RewriteError) -> Self {
        match error {
            RewriteError::PreCommit(error) | RewriteError::NotDurable(error) => error,
        }
    }
}

impl FileMemoryStore {
    /// Open (or create) a store backed by `path`, loading existing entries.
    ///
    /// If `path` does not exist it is created empty (created without
    /// truncating, so a concurrent open cannot discard a writer's
    /// lines). If it exists, each line is deserialized as one
    /// [`MemoryEntry`]: a malformed final line is a torn write and is
    /// dropped with a `tracing` warning — and the file is repaired,
    /// its tail rewritten away, so a later `store` cannot weld itself
    /// onto the fragment — while a malformed line anywhere earlier is
    /// real corruption and fails the open.
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
        Self::ensure_file(&path)?;
        let canonical = std::fs::canonicalize(&path)
            .map_err(|e| LoopError::Memory(format!("cannot resolve memory file path: {e}")))?;
        let shared = {
            let mut registry = recover_guard(live_stores().lock());
            Self::migrate_provisional_entries(&mut registry);
            let attached = registry.get(&canonical).and_then(Weak::upgrade);
            if let Some(shared) = attached {
                shared
            } else {
                registry.remove(&canonical);
                let (entries, torn_tail) = Self::load_entries(&canonical)?;
                if torn_tail {
                    Self::rewrite(&canonical, &entries).map_err(LoopError::from)?;
                }
                let shared = Arc::new(SharedInner::new(Arc::new(Inner::new(
                    canonical.clone(),
                    entries,
                    false,
                ))));
                registry.insert(canonical, Arc::downgrade(&shared));
                shared
            }
        };
        Ok(Self {
            inner: shared,
            consolidation: ConsolidationConfig::default(),
        })
    }

    /// Ensure the memory file exists, creating it if it is missing.
    ///
    /// An existing file returns immediately, whatever its permission
    /// state — a read-only file opens read-only, as loading a frozen
    /// memory requires. Only a missing path is created, via append
    /// mode: a create that cannot truncate, so when two opens race on
    /// one missing path the loser's create discards none of the
    /// winner's already-appended lines. A probe that cannot decide
    /// (an inaccessible parent) is treated as missing, letting the
    /// append-create surface the underlying error itself. The handle
    /// is dropped — the file only needs to exist for the caller's
    /// next step.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the file cannot be created or opened
    /// for append.
    fn ensure_file(path: &Path) -> Result<(), LoopError> {
        if path.try_exists().unwrap_or(false) {
            return Ok(());
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| LoopError::Memory(format!("cannot create memory file: {e}")))?;
        Ok(())
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
        let (key, is_fallback) = Self::registry_key(path.as_ref());
        let shared = {
            let mut registry = recover_guard(live_stores().lock());
            Self::migrate_provisional_entries(&mut registry);
            let attached = registry.get(&key).and_then(Weak::upgrade);
            if let Some(shared) = attached {
                shared
            } else {
                registry.remove(&key);
                let shared = Arc::new(SharedInner::new(Arc::new(Inner::new(
                    key.clone(),
                    Vec::new(),
                    is_fallback,
                ))));
                registry.insert(key, Arc::downgrade(&shared));
                shared
            }
        };
        Self {
            inner: shared,
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
    /// what `open` will derive once it creates the file — unless the
    /// chain budget ran out first, in which case the key is the deepest
    /// spelling reached, flagged provisional exactly like a fallback,
    /// because the world may resolve past it. Only when the parent
    /// itself cannot be resolved (missing or dangling, or gone between
    /// the two lookups) does the key fall back to the lexical
    /// `absolute` form — also flagged provisional, since that spelling
    /// can stop matching the world once the parent resolves; later
    /// constructions re-derive flagged keys and move the registration
    /// (see [`migrate_provisional_entries`](Self::migrate_provisional_entries)).
    /// A relative spelling is anchored by `absolute` at the process's
    /// working directory, so constructions from different working
    /// directories key differently while the path stays unresolved —
    /// absolute spellings are the portable form.
    ///
    /// Returns the derived key and whether it is provisional — either
    /// from the lexical fallback arm or from chain-budget exhaustion.
    fn registry_key(path: &Path) -> (PathBuf, bool) {
        let mut current = path.to_path_buf();
        let mut exhausted = true;
        for _ in 0..SYMLINK_FOLLOW_LIMIT {
            if let Ok(canonical) = std::fs::canonicalize(&current) {
                return (canonical, false);
            }
            if let Some(target) = Self::dangling_final_symlink(&current) {
                current = target;
            } else {
                exhausted = false;
                break;
            }
        }
        if let Some(key) = current
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .and_then(|parent| current.file_name().map(|name| parent.join(name)))
        {
            return (key, exhausted);
        }
        (
            std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
            true,
        )
    }

    /// Move or unify provisional registrations onto the keys the world
    /// now yields.
    ///
    /// Runs inside the constructors' registry critical section, before
    /// their attach-or-create lookup, so handles constructed while a
    /// parent symlink was dangling are found once the path resolves;
    /// and before every mirror-driven rewrite, via
    /// [`converge_with_registry`](Self::converge_with_registry), so a
    /// rewrite from one mirror cannot delete a converged sibling's
    /// entries. Every live cell flagged as a fallback has its key
    /// re-derived; a
    /// registration whose derived key is unoccupied moves there, and
    /// one whose derived key is already occupied unifies into the
    /// occupant — mirrors merge
    /// (see [`merge_mirrors`](Self::merge_mirrors)), the stale cell is
    /// re-pointed at the occupant's [`Inner`]
    /// (see [`unify_shared_states`](Self::unify_shared_states)), and
    /// the stale registration key is re-registered against the
    /// occupant, so a late construction through either spelling
    /// attaches to the one unified store. [`Inner::path`] converges
    /// with the key on both paths — a move writes the re-derived key
    /// into the inner it moves, and a unification writes it into the
    /// target — because a frozen provisional spelling ending in a
    /// symlink could never survive a rename once the rewrite refuses
    /// to replace links; a re-derived key's final component is never
    /// a symlink. Lock order for a unification: the registry mutex
    /// (held by the caller), both `current` write locks in
    /// deterministic pointer order, then the target's path write
    /// guard, then the mirrors' locks as documented on
    /// [`merge_mirrors`] — nothing in this module takes the registry
    /// mutex while holding any of the others, so the order cannot
    /// cycle.
    fn migrate_provisional_entries(registry: &mut HashMap<PathBuf, Weak<SharedInner>>) {
        let mut candidates = Vec::new();
        for (registered, weak) in registry.iter() {
            let Some(shared) = weak.upgrade() else {
                continue;
            };
            let inner = recover_guard(shared.current.read()).clone();
            if !inner.is_fallback {
                continue;
            }
            let derived = Self::registry_key(&inner.backing_path()).0;
            if &derived != registered {
                candidates.push((registered.clone(), derived, shared));
            }
        }
        candidates.sort_by_key(|(_, _, shared)| Arc::as_ptr(shared) as usize);
        for (registered, derived, shared) in candidates {
            let occupant = registry.get(&derived).and_then(Weak::upgrade);
            match occupant {
                None => {
                    if let Some(weak) = registry.remove(&registered) {
                        registry.insert(derived.clone(), weak);
                    }
                    let inner = recover_guard(shared.current.read()).clone();
                    *recover_guard(inner.path.write()) = derived;
                }
                Some(target) if Arc::ptr_eq(&shared, &target) => {}
                Some(target) => {
                    Self::unify_shared_states(&shared, &target, &derived);
                    registry.insert(registered, Arc::downgrade(&target));
                }
            }
        }
    }

    /// Re-point one cell at another and fold the stale mirror in.
    ///
    /// Takes both `current` write locks in deterministic pointer order
    /// (the caller holds the registry mutex; nothing here or in an
    /// operation takes that mutex while holding any of these locks, so
    /// the order cannot cycle), converges the target's backing path
    /// onto `derived` — a re-derived key, whose final component is
    /// never a symlink — merges the stale mirror into the target
    /// (see [`merge_mirrors`](Self::merge_mirrors)), then installs the
    /// target's [`Inner`] as the stale cell's current state — in-flight
    /// operations on the stale cell finish against the retired
    /// `Inner`, and every later operation observes the unified store.
    fn unify_shared_states(stale: &Arc<SharedInner>, target: &Arc<SharedInner>, derived: &Path) {
        let mut cells = [stale, target];
        cells.sort_by_key(|cell| Arc::as_ptr(*cell) as usize);
        let [lesser, greater] = cells;
        let mut first_current = recover_guard(lesser.current.write());
        let mut second_current = recover_guard(greater.current.write());
        let (stale_current, target_current) = if Arc::ptr_eq(stale, lesser) {
            (&mut first_current, &mut second_current)
        } else {
            (&mut second_current, &mut first_current)
        };
        let stale_inner = stale_current.clone();
        let target_inner = target_current.clone();
        *recover_guard(target_inner.path.write()) = derived.to_path_buf();
        Self::merge_mirrors(&stale_inner, &target_inner);
        **stale_current = target_inner;
    }

    /// Fold a retired mirror into the store that survives it.
    ///
    /// Both mirrors' appends already reached the same physical file —
    /// the mirror is the divergence — so the union restores
    /// completeness: entries present only in the stale mirror (matched
    /// by id) are appended to the target mirror, the merged order is
    /// then reset against the backing file (see
    /// [`reorder_merged_entries`](Self::reorder_merged_entries)) so
    /// the arbitrary target/stale assignment cannot flip retrieval's
    /// insertion order, and the access logs union with the same
    /// max-stamp rule retrieval applies. Locks, in order: the stale
    /// mirror's entries read, the target mirror's entries write, then
    /// both access logs — safe because the caller pins both cells
    /// under their `current` write locks, so no operation can hold
    /// either mirror's locks concurrently.
    fn merge_mirrors(stale: &Inner, target: &Inner) {
        let stale_entries = recover_guard(stale.entries.read());
        let mut target_entries = recover_guard(target.entries.write());
        let present: HashSet<Uuid> = target_entries.iter().map(|entry| entry.id).collect();
        for entry in stale_entries.iter() {
            if !present.contains(&entry.id) {
                target_entries.push(entry.clone());
            }
        }
        if let Some(reordered) =
            Self::reorder_merged_entries(&target_entries, &target.backing_path())
        {
            *target_entries = reordered;
        }
        drop(stale_entries);
        drop(target_entries);
        let stale_log = recover_guard(stale.access_log.lock());
        let mut target_log = recover_guard(target.access_log.lock());
        for (id, stamp) in stale_log.iter() {
            match target_log.get_mut(id) {
                Some(existing) => {
                    *existing = (*existing).max(*stamp);
                }
                None => {
                    target_log.insert(*id, *stamp);
                }
            }
        }
    }

    /// Reset a merged or provisional mirror against the backing file's
    /// entry order.
    ///
    /// Called from a unification's merge and, before every rewrite
    /// out of a store that was ever provisionally keyed, from
    /// [`flush`](Self::flush) and
    /// [`consolidate`](LoopMemory::consolidate) — in both roles the
    /// file is the durable record of append order — the order
    /// retrieval's stable tie-break names as insertion order — and
    /// which side of a convergence a mirror sat on is arbitrary, so
    /// the mirror is reordered to the file's sequence. The file is
    /// the truth for content too: an entry found in both keeps the
    /// mirror copy (fresher access state), while a file-only entry is
    /// adopted from the file itself — mirrors never deliberately drop
    /// entries outside consolidation, which rewrites the file in the
    /// same pass, so file-only content can only be a dropped
    /// sibling's un-converged appends, exactly what must survive the
    /// convergence rewrite. Mirror-only entries (not expected —
    /// `store` appends before mirroring) keep their merged order at
    /// the end. The file read runs under the registry mutex the
    /// caller holds — convergence is rare, so a read-only lookup
    /// there is a deliberate trade-off. A file that cannot be loaded
    /// (mid-file corruption) returns `None` and leaves the merged
    /// mirror as-is; the next `open` surfaces that corruption loudly —
    /// ordering is best-effort, completeness is not.
    fn reorder_merged_entries(merged: &[MemoryEntry], path: &Path) -> Option<Vec<MemoryEntry>> {
        let file_entries = Self::load_entries(path).ok()?.0;
        let file_ids: HashSet<Uuid> = file_entries.iter().map(|entry| entry.id).collect();
        let mut by_id: HashMap<Uuid, MemoryEntry> = merged
            .iter()
            .cloned()
            .map(|entry| (entry.id, entry))
            .collect();
        let mut reordered = Vec::with_capacity(merged.len());
        for entry in &file_entries {
            match by_id.remove(&entry.id) {
                Some(matched) => reordered.push(matched),
                None => reordered.push(entry.clone()),
            }
        }
        for entry in merged {
            if !file_ids.contains(&entry.id) {
                reordered.push(entry.clone());
            }
        }
        Some(reordered)
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

    /// Re-derive provisional registrations before a mirror-driven rewrite.
    ///
    /// [`flush`](Self::flush) and
    /// [`consolidate`](LoopMemory::consolidate) rewrite the file from
    /// a mirror, so a handle whose path has converged with another's
    /// since construction must unify first — otherwise the rewrite
    /// would persist one mirror and silently delete the other's
    /// entries from disk. Takes the registry mutex as a leaf —
    /// acquired and released before any other lock, since nothing in
    /// this module takes it while holding one — and must run before
    /// the caller's whole-operation `current` guard, so the rewrite
    /// then executes entirely against the post-redirect [`Inner`]. If
    /// this handle is the stale side of a collision, the migration
    /// merges its mirror into the target before redirecting, so
    /// nothing it stored before converging is lost by the redirect.
    fn converge_with_registry() {
        let mut registry = recover_guard(live_stores().lock());
        Self::migrate_provisional_entries(&mut registry);
    }

    /// Persist the full in-memory state, rewriting the file atomically.
    ///
    /// Used internally by [`consolidate`](LoopMemory::consolidate); also
    /// exposed for callers that want a checkpoint after bulk mutation.
    /// The rewrite is atomic and durable — a sibling temp file under an
    /// unpredictable name is written and synced, renamed over the
    /// target, and the containing directory is synced so the rename
    /// itself survives power loss — so a crash mid-rewrite leaves the
    /// previous file intact and a completed flush is on disk. The shared
    /// state stays pinned and the entries write lock held across the
    /// whole rewrite, so concurrent `flush`es, `consolidate`s, and
    /// `store`s are serialized: two rewrites can never interleave on the
    /// temp file, and an append can never land on the unlinked old file
    /// after the rename. A store that was ever provisionally keyed
    /// re-syncs its mirror against the file before rewriting, so a
    /// sibling's durable append that this mirror never observed —
    /// through a spelling whose convergence happened outside this
    /// call, or a sibling dropped before it — is adopted rather than
    /// stranded on the replaced inode; a canonically-keyed store skips
    /// the read and keeps its documented discard of an existing file's
    /// old entries. A successful rewrite repairs the tail, so it also
    /// re-arms appends after an earlier append failure latched the
    /// store unusable.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] on any I/O or serialization failure. On
    /// failure the original file is left untouched.
    pub fn flush(&self) -> Result<(), LoopError> {
        use std::sync::atomic::Ordering as AtomicOrdering;
        Self::converge_with_registry();
        let shared = recover_guard(self.inner.current.read());
        let mut entries = recover_guard(shared.entries.write());
        let backing_path = shared.backing_path();
        if shared.is_fallback
            && let Some(adopted) = Self::reorder_merged_entries(&entries, &backing_path)
        {
            *entries = adopted;
        }
        let rewrite_result = Self::rewrite(&backing_path, &entries).map_err(LoopError::from);
        if rewrite_result.is_ok() {
            shared.append_unusable.store(false, AtomicOrdering::SeqCst);
        }
        rewrite_result
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

    /// Rewrite `path` with `entries`, atomically.
    ///
    /// Writes a sibling temp file under an unpredictable name
    /// (`<name>.<uuid>.tmp`) created exclusively — `create_new` fails if
    /// anything exists at the path, symlinks included, so a planted
    /// link can never redirect the rewrite's contents into another
    /// file — syncs it, renames it over the target, and then syncs the
    /// containing directory so the rename itself is durable. The rename
    /// is refused when the target's final component is a symlink — an
    /// unresolved spelling: a rename never follows it, so replacing
    /// the link would fork the chain's data. The refusal is
    /// pre-commit, the link survives, and the caller keeps its mirror.
    /// A rewriting thread converges the registration and the live path
    /// onto the re-derived key before pinning the store, and a
    /// re-derived key's final component is never a symlink, so no
    /// rewrite of a resolved store is refused; a concurrent
    /// constructor may still swap the path under an operation that
    /// already converged — the stale spelling then meets the refusal,
    /// a retryable error, never corruption.
    /// Readers
    /// observe either the old or the new file, never a partial one; the
    /// temp file is removed on failure, and a successful rewrite also
    /// sweeps `<name>.<uuid>.tmp` siblings left behind by earlier
    /// crashed rewrites (random names never collide with a live one, so
    /// a crash between create and rename leaves at most an orphan the
    /// next successful pass removes).
    ///
    /// # Errors
    ///
    /// [`RewriteError::PreCommit`](RewriteError::PreCommit) if
    /// any entry cannot be serialized, the temp file cannot be
    /// created, written, synced, or renamed over the target, or the
    /// target's final component is a symlink — the original file is
    /// untouched.
    /// [`RewriteError::NotDurable`](RewriteError::NotDurable) if
    /// the post-rename directory sync fails: the new file is already
    /// in place, only its durability is unconfirmed.
    fn rewrite(path: &Path, entries: &[MemoryEntry]) -> Result<(), RewriteError> {
        use std::io::Write as _;
        let temp_path = Self::temp_sibling_path(path);
        let write_result = (|| -> Result<(), RewriteError> {
            #[cfg(feature = "testing")]
            if injected_fault_at(path, RewriteFaultStage::TempCreate) {
                return Err(RewriteError::PreCommit(LoopError::Memory(
                    "cannot create rewrite temp file: injected fault".to_string(),
                )));
            }
            let mut buffer = String::new();
            for entry in entries {
                let line = serde_json::to_string(entry).map_err(|e| {
                    RewriteError::PreCommit(LoopError::Memory(format!(
                        "memory entry serialization failed: {e}"
                    )))
                })?;
                buffer.push_str(&line);
                buffer.push('\n');
            }
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
                .map_err(|e| {
                    RewriteError::PreCommit(LoopError::Memory(format!(
                        "cannot create rewrite temp file: {e}"
                    )))
                })?;
            file.write_all(buffer.as_bytes())
                .and_then(|()| file.sync_all())
                .map_err(|e| {
                    RewriteError::PreCommit(LoopError::Memory(format!(
                        "cannot write rewrite temp file: {e}"
                    )))
                })?;
            let path_is_symlink = std::fs::symlink_metadata(path)
                .is_ok_and(|metadata| metadata.file_type().is_symlink());
            if path_is_symlink {
                return Err(RewriteError::PreCommit(LoopError::Memory(
                    "memory file path resolves to a symlink; refusing to replace the link"
                        .to_string(),
                )));
            }
            std::fs::rename(&temp_path, path).map_err(|e| {
                RewriteError::PreCommit(LoopError::Memory(format!(
                    "cannot finalize memory rewrite: {e}"
                )))
            })?;
            #[cfg(feature = "testing")]
            if injected_fault_at(path, RewriteFaultStage::DirectorySync) {
                return Err(RewriteError::NotDurable(LoopError::Memory(
                    "cannot sync directory after memory rewrite: injected fault".to_string(),
                )));
            }
            Self::sync_parent_directory(path).map_err(RewriteError::NotDurable)?;
            Self::drop_stale_temp_siblings(path);
            Ok(())
        })();
        if write_result.is_err() {
            drop(std::fs::remove_file(&temp_path));
        }
        write_result
    }

    /// Build the unpredictable temp-sibling path for a rewrite of `path`.
    ///
    /// `<name>.<uuid>.tmp` next to the target: unpredictable, so a
    /// local attacker cannot pre-plant it, and distinct per rewrite, so
    /// leftovers from crashed passes never collide with a live one.
    fn temp_sibling_path(path: &Path) -> PathBuf {
        let mut temp_name = path.as_os_str().to_os_string();
        temp_name.push(format!(".{}.tmp", Uuid::new_v4()));
        PathBuf::from(temp_name)
    }

    /// Sync the directory holding `path` so a completed rename is durable.
    ///
    /// A rename changes a directory entry, and on Unix that change is
    /// not durable until the containing directory itself is synced —
    /// without this step a power loss can lose a rewrite the caller
    /// already saw succeed. A missing or empty parent skips the sync
    /// rather than failing the rewrite.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the parent directory cannot be opened
    /// or synced. Called after a successful rename, so this failure
    /// means the new file is in place but its durability is
    /// unconfirmed.
    #[cfg(unix)]
    fn sync_parent_directory(path: &Path) -> Result<(), LoopError> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        match parent {
            Some(parent) => std::fs::File::open(parent)
                .and_then(|dir| dir.sync_all())
                .map_err(|e| {
                    LoopError::Memory(format!("cannot sync directory after memory rewrite: {e}"))
                }),
            None => Ok(()),
        }
    }

    /// No-op parent sync where directory handles cannot be synced.
    ///
    /// Non-Unix platforms keep the rewrite path byte-identical to
    /// before the directory-sync change.
    #[cfg(not(unix))]
    fn sync_parent_directory(_path: &Path) -> Result<(), LoopError> {
        Ok(())
    }

    /// Remove `<name>.<uuid>.tmp` siblings left by earlier crashed rewrites.
    ///
    /// Called only from a successful rewrite, which already holds the
    /// entries write lock, so no live temp file of this store can
    /// exist. Only UUID-shaped middles are removed — a sibling like
    /// `<name>.backup.tmp` does not match a real temp's
    /// `<name>.<uuid>.tmp` shape and is left alone. Best-effort
    /// janitorial work: removal errors are ignored, and a missing or
    /// empty parent sweeps the current directory.
    fn drop_stale_temp_siblings(path: &Path) {
        let directory = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let stem = path
            .file_name()
            .map_or_else(String::new, |name| format!("{}.", name.to_string_lossy()));
        if stem.is_empty() {
            return;
        }
        if let Ok(siblings) = std::fs::read_dir(&directory) {
            for sibling in siblings.flatten() {
                let name = sibling.file_name().to_string_lossy().to_string();
                let is_our_temp = name
                    .strip_prefix(&stem)
                    .and_then(|rest| rest.strip_suffix(".tmp"))
                    .is_some_and(|middle| Uuid::parse_str(middle).is_ok());
                if is_our_temp {
                    drop(std::fs::remove_file(sibling.path()));
                }
            }
        }
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
    /// The file append happens first, with the shared state pinned and
    /// the entries write lock held: on success the mirror is updated to
    /// match; on failure neither side changes, so the two can never
    /// disagree.
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let line = serde_json::to_string(&entry).map_err(|e| {
                LoopError::Memory(format!("memory entry serialization failed: {e}"))
            })?;
            let shared = recover_guard(self.inner.current.read());
            let mut entries = recover_guard(shared.entries.write());
            shared.append_line(&line)?;
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
            let shared = recover_guard(self.inner.current.read());
            let query_lower = query.to_lowercase();
            let query_trimmed = query_lower.trim();
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            let snapshot: Vec<MemoryEntry> = recover_guard(shared.entries.read())
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
            let mut access_log = recover_guard(shared.access_log.lock());
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
    /// [`access_count`](MemoryEntry::access_count)), re-syncs a store
    /// that was ever provisionally keyed against the file first (so a
    /// sibling's unobserved durable append is adopted into the pass
    /// rather than stranded by the rewrite), then delegates to
    /// [`consolidate_entries`]
    /// — the same decay, merge, and prune pass
    /// [`InMemoryStore`](super::builtin::InMemoryStore) runs — and
    /// persists the survivors with an atomic rewrite before swapping
    /// them into the mirror, so pruned entries do not come back on the
    /// next restart. On a rewrite failure the method returns `Err` with
    /// the file, the mirror, and the pending access stamps all still in
    /// their pre-consolidation state — the stamps are only cleared once
    /// the rewrite has succeeded, held under the access-log lock across
    /// it, so the next pass re-folds them instead of losing them. A
    /// durability-unconfirmed failure (the rename landed, the
    /// directory sync did not) still commits the pass to the mirror
    /// and the file before returning the error — only fsync-durability
    /// is unconfirmed, never the pass itself. A rewrite that lands —
    /// durable or not — repairs the tail and re-arms appends after an
    /// earlier append failure latched the store unusable.
    fn consolidate(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidationStats, LoopError>> + Send + '_>> {
        use std::sync::atomic::Ordering as AtomicOrdering;
        Box::pin(async move {
            Self::converge_with_registry();
            let shared = recover_guard(self.inner.current.read());
            let now = SystemTime::now();
            let mut guard = recover_guard(shared.entries.write());
            let mut next = guard.clone();
            let mut access_log = recover_guard(shared.access_log.lock());
            for entry in &mut next {
                if let Some(stamp) = access_log.get(&entry.id) {
                    entry.last_accessed = entry.last_accessed.max(Some(*stamp));
                    entry.access_count = entry.access_count.saturating_add(1);
                }
            }
            let backing_path = shared.backing_path();
            if shared.is_fallback
                && let Some(adopted) = Self::reorder_merged_entries(&next, &backing_path)
            {
                next = adopted;
            }
            let stats = consolidate_entries(&mut next, &self.consolidation, now);
            match Self::rewrite(&backing_path, &next) {
                Ok(()) => {
                    shared.append_unusable.store(false, AtomicOrdering::SeqCst);
                }
                Err(RewriteError::NotDurable(error)) => {
                    access_log.clear();
                    drop(access_log);
                    *guard = next;
                    shared.append_unusable.store(false, AtomicOrdering::SeqCst);
                    return Err(error);
                }
                Err(RewriteError::PreCommit(error)) => return Err(error),
            }
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
        let shared = recover_guard(self.inner.current.read());
        recover_guard(shared.entries.read()).len()
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

        let (key, is_fallback) = FileMemoryStore::registry_key(&through_link);
        assert_eq!(
            key,
            real.join("memory.jsonl"),
            "an existing file keys by where it really lives, resolving the \
            symlinked directory"
        );
        assert!(
            !is_fallback,
            "an outright canonical resolution is never provisional — no \
            later construction needs to migrate it"
        );

        let file_link = dir.join("mem-link.jsonl");
        symlink(real.join("memory.jsonl"), &file_link).unwrap();
        let (file_key, _) = FileMemoryStore::registry_key(&file_link);
        assert_eq!(
            file_key,
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

        let (key, is_fallback) = FileMemoryStore::registry_key(&absent_through_link);
        assert_eq!(
            key,
            real.join("memory.jsonl"),
            "a missing final component carries no ambiguity of its own — \
            the key is the canonical parent plus the name, exactly what \
            open derives once it creates the file"
        );
        assert!(
            !is_fallback,
            "the canonical-parent arm resolves the world, so its key is \
            never provisional"
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
        let (key, is_fallback) = FileMemoryStore::registry_key(&absolute_link);
        assert_eq!(
            key,
            elsewhere.join("target.jsonl"),
            "a dangling final symlink keys by its absolute target — where \
            the first store will create the data"
        );
        assert!(
            !is_fallback,
            "a dangling final symlink resolves to a definite target, so \
            its key is never provisional"
        );

        let relative_link = dir.join("rel-link.jsonl");
        symlink(
            std::path::Path::new("elsewhere/rel-target.jsonl"),
            &relative_link,
        )
        .unwrap();
        let (relative_key, _) = FileMemoryStore::registry_key(&relative_link);
        assert_eq!(
            relative_key,
            elsewhere.join("rel-target.jsonl"),
            "a relative target is anchored at the canonicalized parent of \
            the link before keying"
        );

        let chain_end = elsewhere.join("chain-end.jsonl");
        let chain_mid = elsewhere.join("chain-mid.jsonl");
        symlink(&chain_end, &chain_mid).unwrap();
        let chain_link = dir.join("chain-link.jsonl");
        symlink(&chain_mid, &chain_link).unwrap();
        let (chain_key, _) = FileMemoryStore::registry_key(&chain_link);
        assert_eq!(
            chain_key,
            elsewhere.join("chain-end.jsonl"),
            "a chain of dangling links is followed to its end"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn an_exhausted_symlink_chain_keys_provisionally() {
        use std::os::unix::fs::symlink;

        let dir = unit_dir("exhausted");
        let mut deepest = dir.join(format!("link-{}.jsonl", super::SYMLINK_FOLLOW_LIMIT));
        symlink(dir.join("absent-target.jsonl"), &deepest).unwrap();
        for step in (0..super::SYMLINK_FOLLOW_LIMIT).rev() {
            let link = dir.join(format!("link-{step}.jsonl"));
            symlink(&deepest, &link).unwrap();
            deepest = link;
        }

        let (key, is_fallback) = FileMemoryStore::registry_key(&deepest);
        assert_eq!(
            key,
            dir.join(format!("link-{}.jsonl", super::SYMLINK_FOLLOW_LIMIT)),
            "budget exhaustion keys by the deepest spelling reached"
        );
        assert!(
            is_fallback,
            "the world may resolve past the deepest spelling, so an \
            exhausted chain's key is provisional — later constructions \
            re-derive it"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unresolvable_path_falls_back_to_the_lexical_absolute_form() {
        let absent = std::path::Path::new("/nonexistent-file-memory-unit/parent/mem.jsonl");
        let (key, is_fallback) = FileMemoryStore::registry_key(absent);
        assert_eq!(
            key,
            absent.to_path_buf(),
            "with neither the file nor its parent resolvable, the already \
            absolute spelling is returned as-is"
        );
        assert!(
            is_fallback,
            "only the unresolved lexical arm marks the key provisional — \
            later constructions re-derive it"
        );
    }

    #[test]
    fn an_empty_path_falls_back_to_the_raw_spelling() {
        let empty = std::path::Path::new("");
        let (key, is_fallback) = FileMemoryStore::registry_key(empty);
        assert_eq!(
            key,
            empty.to_path_buf(),
            "the empty path resolves nowhere at any stage — the final \
            fallback returns the raw spelling instead of panicking"
        );
        assert!(
            is_fallback,
            "a key that resolves nothing is provisional by construction"
        );
    }

    #[test]
    fn a_trailing_dotdot_still_yields_a_key() {
        let edge = std::path::Path::new("/nonexistent-file-memory-unit/..");
        let (key, is_fallback) = FileMemoryStore::registry_key(edge);
        assert_eq!(
            key,
            edge.to_path_buf(),
            "a path with no file name component skips the parent-join arm \
            without panicking and lands in the lexical fallback"
        );
        assert!(
            is_fallback,
            "the lexical fallback marks its keys provisional so a later \
            construction can move the registration"
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
    fn ensuring_an_existing_file_never_truncates() {
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-ensure-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ensure.jsonl");
        let entry = crate::memory::MemoryEntry::new(
            crate::memory::MemoryCategory::Fact,
            "must survive an interleaved open",
        );
        let line = format!("{}\n", serde_json::to_string(&entry).unwrap());
        std::fs::write(&path, line.as_bytes()).unwrap();

        FileMemoryStore::ensure_file(&path).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            line.as_bytes(),
            "ensuring an existing file leaves its bytes untouched — a \
            concurrent writer's appends survive an interleaved open"
        );

        let missing = dir.join("missing.jsonl");
        FileMemoryStore::ensure_file(&missing).unwrap();
        assert_eq!(
            std::fs::read(&missing).unwrap(),
            b"",
            "ensuring a missing path creates the file empty"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_read_only_memory_file_still_opens() {
        use crate::memory::LoopMemory as _;
        use std::os::unix::fs::PermissionsExt as _;

        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-readonly-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("readonly.jsonl");
        let entry = crate::memory::MemoryEntry::new(
            crate::memory::MemoryCategory::Fact,
            "frozen memory opens read-only",
        );
        let line = format!("{}\n", serde_json::to_string(&entry).unwrap());
        std::fs::write(&path, line.as_bytes()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();

        let store = FileMemoryStore::open(&path).unwrap();
        assert_eq!(store.len(), 1, "the read-only file loads its entry");
        assert_eq!(
            store
                .retrieve("frozen", 3)
                .await
                .unwrap()
                .first()
                .map(|hit| hit.id),
            Some(entry.id),
            "retrieval works over the read-only store"
        );
        drop(store);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
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
        let inner = super::recover_guard(store.inner.current.read());
        inner.append_line("{\"line\":1}").unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"{\"line\":1}\n",
            "an append is exactly the line plus its terminating newline"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn appends_through_two_spellings_of_one_file_never_weld() {
        use std::os::unix::fs::symlink;

        const LINES_PER_THREAD: usize = 2000;
        let dir = std::env::temp_dir().join(format!(
            "loopctl-file-memory-unit-weld-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link1 = dir.join("link1");
        let link2 = dir.join("link2");
        symlink(&real, &link1).unwrap();
        symlink(&real, &link2).unwrap();

        let first = super::Inner::new(link1.join("memory.jsonl"), Vec::new(), true);
        let second = super::Inner::new(link2.join("memory.jsonl"), Vec::new(), true);
        let entry =
            crate::memory::MemoryEntry::new(crate::memory::MemoryCategory::Fact, "x".repeat(4096));
        let line = serde_json::to_string(&entry).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let thread_line = line.clone();
        let start = std::sync::Arc::clone(&barrier);
        let first_writer = std::thread::spawn(move || {
            start.wait();
            for _ in 0..LINES_PER_THREAD {
                first.append_line(&thread_line).unwrap();
            }
        });
        let start = std::sync::Arc::clone(&barrier);
        let second_writer = std::thread::spawn(move || {
            start.wait();
            for _ in 0..LINES_PER_THREAD {
                second.append_line(&line).unwrap();
            }
        });
        first_writer.join().unwrap();
        second_writer.join().unwrap();

        let (entries, torn) =
            super::FileMemoryStore::load_entries(&real.join("memory.jsonl")).unwrap();
        assert!(
            !torn,
            "every line ends in its own newline: no trailing fragment"
        );
        assert_eq!(
            entries.len(),
            LINES_PER_THREAD * 2,
            "two unconverged spellings share no lock — the weld guard is \
            the single-write append itself, so every line parses whole"
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
        std::fs::write(
            dir.join(format!("rewrite.jsonl.{}.tmp", uuid::Uuid::new_v4())),
            b"stale",
        )
        .unwrap();
        std::fs::write(dir.join("rewrite.jsonl.backup.tmp"), b"user data").unwrap();
        super::FileMemoryStore::rewrite(&path, &entries).unwrap();

        let rewritten = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = rewritten.lines().collect();
        assert_eq!(lines.len(), 2, "one line per entry: {rewritten}");
        assert!(
            rewritten.ends_with('\n'),
            "the file is newline-terminated: {rewritten:?}"
        );
        let temp_leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .filter(|name| {
                std::path::Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext == "tmp")
            })
            .collect();
        assert_eq!(
            temp_leftovers,
            vec!["rewrite.jsonl.backup.tmp".to_string()],
            "a successful rename consumes the temp file and sweeps stale \
            UUID-shaped siblings — a user file sharing the prefix and \
            suffix is preserved"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("rewrite.jsonl.backup.tmp")).unwrap(),
            "user data",
            "the preserved sibling keeps its contents"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn temp_sibling_paths_are_unpredictable_and_well_shaped() {
        let dir = std::path::Path::new("/store");
        let first = super::FileMemoryStore::temp_sibling_path(&dir.join("memory.jsonl"));
        let second = super::FileMemoryStore::temp_sibling_path(&dir.join("memory.jsonl"));
        assert_ne!(
            first, second,
            "each rewrite gets its own unpredictable temp"
        );
        let first = first.to_string_lossy();
        assert!(
            first.starts_with("/store/memory.jsonl.") && first.ends_with(".tmp"),
            "the temp sibling carries the target name, a unique middle, and \
            the tmp suffix: {first}"
        );
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
