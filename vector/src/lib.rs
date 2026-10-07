//! External vector-store backends for loopctl's semantic memory.
//!
//! Two [`VectorIndex`](loopctl::memory::vector::VectorIndex) implementations that take semantic memory past
//! a single process — a shared server tier and the Postgres a team
//! already runs — each behind its own feature (`qdrant`, `pgvector`;
//! `default = []` pulls nothing):
//!
//! | Store | Tier | The one-line story |
//! |---|---|---|
//! | [`LinearVectorIndex`](loopctl::memory::vector::LinearVectorIndex) | in-process, exact | The loopctl default: brute-force cosine, zero services, always right. |
//! | [`HnswIndex`](https://docs.rs/loopctl-hnsw) | in-process, ANN | Millions of vectors in RAM at ~90%+ recall; the `loopctl-hnsw` companion. |
//! | [`QdrantIndex`](crate::qdrant::QdrantIndex) | server | The dedicated vector DB — memory shared across sessions and hosts, self-hosted or cloud. |
//! | [`PgVectorIndex`](crate::pgvector::PgVectorIndex) | your Postgres | One table in the database most teams already run and back up; zero new infrastructure. |
//!
//! Every backend implements the same trait, so a
//! `VectorMemoryStore` (the memory feature's store)
//! swaps tiers with one constructor call and nothing else changes —
//! same hybrid retrieval, same [`LoopMemory`](loopctl::memory::LoopMemory)
//! contract, same golden-set quality bar. Each backend auto-provisions
//! its target on first use (collection, table, or on-disk dataset,
//! create-if-absent, idempotent — and validates an existing target's
//! vector shape before adopting it, so a mismatched reuse rejects at
//! connect instead of failing on use), maps every failure to
//! [`LoopError::Memory`](loopctl::error::LoopError::Memory) with the
//! backend's message preserved but bounded, and emits the same
//! `loopctl.vector.index.search` metric event the in-process indexes
//! emit — one uniform stream across all four tiers.
//!
//! Add this crate as a direct dependency alongside `loopctl` — the
//! backends need no feature beyond the memory surface the host already
//! enables (`vector_memory` here, because the store is what swaps the
//! index in):
//!
//! ```toml
//! [dependencies]
//! loopctl = { version = "0.3", features = ["vector_memory"] }
//! loopctl-vector = { version = "0.3", features = ["qdrant"] }
//! ```
//!
//! The vector *entries* (text, category, metadata) still live in the
//! memory store's own persistence — these backends make the **index**
//! external, scalable, and durable.

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

#[cfg(feature = "testing")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]
pub mod contract;
#[cfg(feature = "pgvector")]
pub mod pgvector;
#[cfg(feature = "qdrant")]
pub mod qdrant;

use uuid::Uuid;

/// Characters of a backend error message kept in a mapped error.
///
/// Backend clients produce arbitrarily long error chains (gRPC towers,
/// Postgres hints, IO dumps); the mapped [`LoopError::Memory`] string
/// carries enough to diagnose without flooding logs and observer
/// surfaces.
const MAX_ERROR_CHARS: usize = 512;

/// Map a backend failure into the one error surface hosts see.
///
/// Every backend funnels here: the message keeps the backend's name and
/// its own diagnosis (bounded at [`MAX_ERROR_CHARS`] characters, cut on
/// a character boundary), so no backend error type ever leaks past
/// `LoopError::Memory` and every failure names where it came from.
fn memory_error(backend: &str, error: impl std::fmt::Display) -> loopctl::error::LoopError {
    let text = error.to_string();
    let bounded: String = text.chars().take(MAX_ERROR_CHARS).collect();
    loopctl::error::LoopError::Memory(format!("{backend}: {bounded}"))
}

/// Reject a blank name or URL the way every builder does.
///
/// One shared validation so both backends answer a blank endpoint,
/// table, or collection identically: an error naming the field and the
/// backend, before any client or connection is built.
///
/// # Errors
///
/// Returns [`LoopError::Memory`] naming the field when blank.
fn require_non_blank(
    backend: &str,
    field: &'static str,
    value: &str,
) -> Result<(), loopctl::error::LoopError> {
    if value.trim().is_empty() {
        return Err(loopctl::error::LoopError::Memory(format!(
            "{backend}: {field} must not be blank"
        )));
    }
    Ok(())
}

/// Validate a store target name as a strict identifier.
///
/// A SQL table name (pgvector — it is interpolated into DDL that
/// cannot be parameterized) must be `[A-Za-z_][A-Za-z0-9_]*`; the
/// validated name returns as owned so callers never re-handle the raw
/// input.
///
/// # Errors
///
/// Returns [`LoopError::Memory`](loopctl::error::LoopError) naming the
/// rule when the name is not a strict identifier.
fn strict_identifier(backend: &str, name: &str) -> Result<String, loopctl::error::LoopError> {
    let mut chars = name.chars();
    let head_ok = chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
    let tail_ok = chars.all(|rest| rest.is_ascii_alphanumeric() || rest == '_');
    if head_ok && tail_ok {
        return Ok(name.to_string());
    }
    Err(loopctl::error::LoopError::Memory(format!(
        "{backend}: table name must be a strict identifier ([A-Za-z_][A-Za-z0-9_]*), got {name:?}"
    )))
}

/// Reject a zero dimension the way every builder does.
///
/// A zero-dimensional vector space cannot carry meaning and every
/// backend's schema machinery rejects it in its own words later; the
/// builders reject it up front in the crate's one voice.
///
/// # Errors
///
/// Returns [`LoopError::Memory`] when the dimension is zero.
fn require_dim(backend: &str, dim: usize) -> Result<(), loopctl::error::LoopError> {
    if dim == 0 {
        return Err(loopctl::error::LoopError::Memory(format!(
            "{backend}: dim must be greater than zero"
        )));
    }
    Ok(())
}

/// Emit the first-use provisioning event for one target.
///
/// One INFO event per target per process — creating a collection,
/// table, or dataset is a side effect an operator should see, and the
/// idempotence contract means repeats stay silent.
fn emit_provisioned(backend: &str, target: &str) {
    tracing::info!(
        target: "loopctl::metrics",
        metric = "loopctl.vector.provision",
        backend,
        target_name = target,
        "vector store target provisioned on first use"
    );
}

/// Emit one provisioning-failure counter event.
///
/// Paired with the provision event so a store that cannot reach or
/// create its target is countable next to the healthy tier.
fn emit_provision_error(backend: &str) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.vector.provision.errors",
        backend,
        "vector store provisioning failed"
    );
}

/// The external backends' client-side answer to the trait's sync `len`.
///
/// The [`VectorIndex`](loopctl::memory::vector::VectorIndex) trait's
/// `len` is a synchronous accessor — an in-process shape — while an
/// external store's true count is a server round-trip. Each backend
/// seeds this bookkeeping with one async count fetch inside its
/// already-async constructor, then maintains it with its own
/// successful `add`/`remove` operations, so [`len`](RemoteCount::len)
/// answers without IO and stays exact while this process is the only
/// writer since construction. Writes from other processes (or before
/// this construction) are invisible to it by design; when that
/// approximation matters, re-construct the index.
///
/// Bookkeeping rules: `add` of an id this client has already added is a
/// replace (no delta); `add` of a re-added-after-remove id counts
/// again; `remove` of an id this client never saw decrements the base
/// exactly when the backend confirmed a row actually left (a foreign
/// write this client removed), and is remembered so a repeated remove
/// of the same absent id cannot double-decrement. Backends gate the
/// call on that confirmation — pgvector on `rows_affected`, qdrant on
/// a point lookup — so a phantom remove never reaches these rules.
#[derive(Debug, Default)]
struct RemoteCount {
    /// The server's count at construction time.
    ///
    /// Seeded by one async fetch inside the constructor; every later
    /// answer is this figure plus this client's own deltas.
    base: std::sync::atomic::AtomicUsize,

    /// Ids this client has added and not removed, plus ids it removed
    /// without ever adding them (foreign removes), each tagged.
    known: std::sync::Mutex<std::collections::HashMap<Uuid, Known>>,
}

/// One id's client-side bookkeeping state.
///
/// The tag decides whether an `add`/`remove` under the id changes the
/// count — the replacements and no-op removes the trait's semantics
/// produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Known {
    /// Added by this client and present on the server.
    ///
    /// A later `add` under the same id is a replace — no delta.
    Added,

    /// Absent — this client removed it (or never added it); re-adding
    /// counts anew.
    Removed,
}

impl RemoteCount {
    /// Seed the bookkeeping with the constructor-time server count.
    ///
    /// Called exactly once, from the async constructor after
    /// provisioning; the figure is this client's count baseline.
    fn seed(&self, count: usize) {
        self.base.store(count, std::sync::atomic::Ordering::Release);
    }

    /// Record one successful upsert under `id`.
    ///
    /// A first-time id (or a re-add after remove) counts one; an id
    /// this client already added is a replace and changes nothing.
    fn note_add(&self, id: Uuid) {
        let mut known = self.lock_known();
        if !matches!(known.get(&id), Some(Known::Added)) {
            let _ignored = known.insert(id, Known::Added);
            self.base.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// Record one remove under `id`, saturating at zero.
    ///
    /// Removes this client's own id or a foreign row one time each;
    /// repeating a remove of an already-removed id is the no-op the
    /// trait promises, and callers only invoke this after their
    /// backend confirmed the removal touched a row — a phantom remove
    /// never reaches here. The saturation guard (never below empty) is
    /// defense in depth behind that confirmation. The known-set lock
    /// serializes the check-then-sub against every other note, so the
    /// saturation read is race-free within this client.
    fn note_remove(&self, id: Uuid) {
        let mut known = self.lock_known();
        if matches!(known.get(&id), Some(Known::Removed)) {
            return;
        }
        let _ignored = known.insert(id, Known::Removed);
        if self.base.load(std::sync::atomic::Ordering::Acquire) > 0 {
            self.base.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// The client-side count — no IO, ever.
    ///
    /// One atomic load of the seed plus this client's deltas; the
    /// exactness contract is the type's own docs.
    fn len(&self) -> usize {
        self.base.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The known-set under its lock, poisoning recovered.
    ///
    /// The bookkeeping is advisory arithmetic, so a poisoned lock
    /// continues instead of spreading the panic into every `len`.
    fn lock_known(&self) -> std::sync::MutexGuard<'_, std::collections::HashMap<Uuid, Known>> {
        recover_known(self.known.lock())
    }
}

impl Clone for RemoteCount {
    fn clone(&self) -> Self {
        let known = recover_known(self.known.lock()).clone();
        Self {
            base: std::sync::atomic::AtomicUsize::new(self.len()),
            known: std::sync::Mutex::new(known),
        }
    }
}

/// Recover a poisoned known-set lock the way loopctl's own mutexes do.
///
/// The bookkeeping is advisory count arithmetic — a poisoned lock means
/// a panicking thread mid-update, whose worst effect is a stale count;
/// continuing beats poisoning every future `len`.
fn recover_known(
    guard: std::sync::LockResult<std::sync::MutexGuard<'_, std::collections::HashMap<Uuid, Known>>>,
) -> std::sync::MutexGuard<'_, std::collections::HashMap<Uuid, Known>> {
    guard.unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_errors_map_to_bounded_memory_strings() {
        let long = "x".repeat(2_000);
        let error = memory_error("backend-under-test", &long);
        let loopctl::error::LoopError::Memory(text) = &error else {
            panic!("every backend error maps to the Memory variant: {error:?}");
        };
        assert!(
            text.starts_with("backend-under-test: "),
            "the message names its backend: {text}"
        );
        assert!(
            text.chars().count() <= "backend-under-test: ".len() + MAX_ERROR_CHARS,
            "the message is bounded: {} chars",
            text.chars().count()
        );
    }

    #[test]
    fn shared_validation_rejects_blank_names_and_zero_dims() {
        assert!(require_non_blank("b", "url", "   ").is_err());
        assert!(require_non_blank("b", "url", "http://x").is_ok());
        assert!(require_dim("b", 0).is_err());
        assert!(require_dim("b", 4).is_ok());
    }

    #[test]
    fn remote_count_saturates_on_removes_of_never_added_ids() {
        let count = RemoteCount::default();
        count.seed(0);
        count.note_remove(Uuid::new_v4());
        assert_eq!(
            count.len(),
            0,
            "removing an id over an empty target leaves the count at zero, never wrapped"
        );
        count.note_remove(Uuid::new_v4());
        assert_eq!(count.len(), 0, "a second absent-id remove still saturates");
    }

    #[test]
    fn remote_count_tracks_adds_replaces_and_removes_without_io() {
        let count = RemoteCount::default();
        count.seed(10);
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        count.note_add(first);
        count.note_add(first);
        assert_eq!(
            count.len(),
            11,
            "a replace under the same id never counts twice"
        );
        count.note_remove(first);
        assert_eq!(count.len(), 10, "removing this client's id decrements");
        count.note_add(first);
        assert_eq!(count.len(), 11, "re-adding after remove counts anew");
        count.note_remove(second);
        count.note_remove(second);
        assert_eq!(
            count.len(),
            10,
            "removing a foreign id decrements once; a repeated remove of the absent id is a no-op"
        );
    }
}
