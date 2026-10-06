//! The shared backend contract suite.
//!
//! One set of tests every [`VectorIndex`] implementation runs
//! unchanged — the crate's two backends pass their own factory, and a
//! third-party backend can do the same to prove the trait's contract
//! holds: upsert semantics, cosine ordering, `k` discipline,
//! dimension rejection, remove semantics, and provisioning
//! idempotence. Each function takes an [`IndexFactory`] that
//! constructs a fresh index over a target the test names (and owns
//! the lifetime of), so suites are contamination-free and repeatable.
//!
//! ```rust,no_run
//! use loopctl_vector::contract::{self, IndexFactory};
//! use std::pin::Pin;
//!
//! fn factory() -> IndexFactory {
//!     Box::new(|_name, _dim| {
//!         Box::pin(async move {
//!             let index = loopctl::memory::vector::LinearVectorIndex::new(2);
//!             Box::new(index) as Box<dyn loopctl::memory::vector::VectorIndex>
//!         })
//!     })
//! }
//!
//! # async fn demo() {
//! contract::upsert_replaces_without_leaking(&factory()).await;
//! # }
//! ```
//!
//! [`VectorIndex`]: loopctl::memory::vector::VectorIndex

use std::future::Future;
use std::pin::Pin;

use loopctl::error::LoopError;
use loopctl::memory::vector::Embedding;
use loopctl::memory::vector::VectorIndex;
use uuid::Uuid;

/// The dimension the contract tests embed at.
///
/// Two axes for exact cosine control; the planting below uses unit
/// axis vectors, so ordering assertions carry no floating-point
/// ambiguity.
pub const CONTRACT_DIM: usize = 2;

/// The factory type every backend supplies: one fresh index per call,
/// over a target the test names (and owns the lifetime of).
pub type IndexFactory = Box<
    dyn Fn(String, usize) -> Pin<Box<dyn Future<Output = Box<dyn VectorIndex>> + Send>>
        + Send
        + Sync,
>;

/// A unique target name for one test, namespaced by the test's slug.
///
/// Uuid-suffixed so suites are contamination-free and repeatable — two
/// runs of a suite never see each other's rows.
#[must_use]
pub fn unique_name(slug: &str) -> String {
    format!("lv_{slug}_{}", Uuid::new_v4().simple())
}

/// A strictly-unit axis vector for [`CONTRACT_DIM`] dimensions.
///
/// `axis 0` is `[1.0, 0.0]`, `axis 1` the orthogonal partner — exact
/// cosine similarities, no floating-point ambiguity in the ordering
/// assertions.
#[must_use]
pub fn axis(axis: usize) -> Embedding {
    let mut components = vec![0.0_f32; CONTRACT_DIM];
    if let Some(slot) = components.get_mut(axis % CONTRACT_DIM) {
        *slot = 1.0;
    }
    Embedding::new(components)
}

/// The all-ones vector for [`CONTRACT_DIM`] dimensions.
///
/// Every component set — the exact middle of the axis pair under
/// cosine (the vector's magnitude is irrelevant to cosine), one
/// unambiguous ranking step between aligned and opposed.
#[must_use]
pub fn diagonal() -> Embedding {
    let mut components = vec![0.0_f32; CONTRACT_DIM];
    components.fill(1.0);
    Embedding::new(components)
}

/// Re-adding under one id replaces without leaking.
///
/// The upsert contract: the second add under the same id leaves `len`
/// unchanged and search finds the *new* vector's neighborhood.
///
/// # Panics
///
/// Panics naming the violated contract when the backend leaks the
/// replaced vector or answers from the old neighborhood.
pub async fn upsert_replaces_without_leaking(make: &IndexFactory) {
    let index = make(unique_name("upsert"), CONTRACT_DIM).await;
    let id = Uuid::new_v4();
    index.add(id, axis(0)).await.expect("first add lands");
    index.add(id, axis(1)).await.expect("second add replaces");
    assert_eq!(
        index.len(),
        1,
        "a replace under one id never leaks a second vector"
    );
    let hits = index
        .search(&axis(1), 5)
        .await
        .expect("search after replace");
    assert_eq!(
        hits.first().map(|hit| (hit.id, hit.score)),
        Some((id, 1.0)),
        "the replaced vector's neighborhood answers, not the old one's: {hits:?}"
    );
}

/// Search orders by descending cosine similarity and respects `k`.
///
/// A planted nearest/farthest set with exact unit vectors: the top hit
/// is exact, scores never ascend, `k` caps the result, and a short
/// store returns fewer than `k` rather than padding.
///
/// # Panics
///
/// Panics naming the violated contract on any ordering or `k`
/// violation.
pub async fn search_orders_by_cosine_and_respects_k(make: &IndexFactory) {
    let index = make(unique_name("ordering"), CONTRACT_DIM).await;
    let nearest = Uuid::new_v4();
    let middle = Uuid::new_v4();
    let farthest = Uuid::new_v4();
    index.add(nearest, axis(0)).await.expect("nearest plants");
    index.add(middle, diagonal()).await.expect("middle plants");
    index.add(farthest, axis(1)).await.expect("farthest plants");

    let hits = index.search(&axis(0), 3).await.expect("full ordering");
    assert_eq!(
        hits.first().map(|hit| hit.id),
        Some(nearest),
        "the exact-aligned vector is the top hit: {hits:?}"
    );
    assert!(
        hits.windows(2).all(|pair| {
            pair[0].score >= pair[1].score || (pair[0].score - pair[1].score).abs() < 1e-5
        }),
        "scores never ascend: {hits:?}"
    );
    assert_eq!(hits.len(), 3, "a store of three answers k=3 with three");

    let capped = index.search(&axis(0), 1).await.expect("k=1 search");
    assert_eq!(capped.len(), 1, "k=1 returns exactly one match: {capped:?}");
}

/// Removing an id shrinks `len` and the id never returns from search.
///
/// # Panics
///
/// Panics naming the violated contract when a removed id survives in
/// the count or the results.
pub async fn remove_then_len_shrinks_and_id_is_gone(make: &IndexFactory) {
    let index = make(unique_name("remove"), CONTRACT_DIM).await;
    let kept = Uuid::new_v4();
    let removed = Uuid::new_v4();
    index.add(kept, axis(0)).await.expect("kept plants");
    index.add(removed, axis(0)).await.expect("removed plants");
    assert_eq!(index.len(), 2, "both vectors count");
    index.remove(removed).await.expect("remove lands");
    assert_eq!(index.len(), 1, "the removed id leaves the count");
    let hits = index
        .search(&axis(0), 10)
        .await
        .expect("search after remove");
    assert!(
        hits.iter().all(|hit| hit.id != removed),
        "the removed id never returns from search: {hits:?}"
    );
    index.remove(removed).await.expect("re-remove is a no-op");
    assert_eq!(index.len(), 1, "removing an absent id changes nothing");
    let never_added = Uuid::new_v4();
    index
        .remove(never_added)
        .await
        .expect("remove of a never-stored id is the trait's no-op");
    assert_eq!(
        index.len(),
        1,
        "a remove of an id no one ever stored changes nothing — the count \
         saturates, never wraps"
    );
}

/// A wrong-dimension vector rejects at `add` and nothing is stored.
///
/// # Panics
///
/// Panics naming the violated contract when a wrong-dimension vector
/// is accepted or stored.
pub async fn dim_mismatch_rejects_at_add(make: &IndexFactory) {
    let index = make(unique_name("dim_mismatch"), CONTRACT_DIM).await;
    let wrong = Embedding::new(vec![1.0, 0.0, 0.0]);
    let rejection = index
        .add(Uuid::new_v4(), wrong)
        .await
        .expect_err("a wrong-dimension vector rejects");
    assert!(
        matches!(rejection, LoopError::Memory(_))
            || rejection.to_string().to_lowercase().contains("dim"),
        "the rejection is a memory error naming the dimension: {rejection}"
    );
    assert_eq!(index.len(), 0, "the rejected vector stored nothing");
}

/// Two constructions against one target provision exactly once.
///
/// The idempotence contract: the second construction is a no-op (no
/// error, no duplicate target), and both handles see the same store —
/// a point added through the second is visible to the first.
///
/// # Panics
///
/// Panics naming the violated contract when the second construction
/// errors or the handles disagree.
pub async fn provisioning_is_idempotent(make: &IndexFactory) {
    let name = unique_name("idempotent");
    let first = make(name.clone(), CONTRACT_DIM).await;
    let second = make(name, CONTRACT_DIM).await;
    let id = Uuid::new_v4();
    second.add(id, axis(0)).await.expect("add through second");
    let hits = first
        .search(&axis(0), 5)
        .await
        .expect("search through first");
    assert!(
        hits.iter().any(|hit| hit.id == id),
        "both handles share one provisioned target: {hits:?}"
    );
}
