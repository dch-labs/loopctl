//! The shared backend contract suite.
//!
//! One set of checks every [`VectorIndex`] implementation runs
//! unchanged — the crate's two backends pass their own factory, and a
//! third-party backend can do the same to prove the trait's contract
//! holds: upsert semantics, cosine ordering, `k` discipline,
//! dimension rejection, remove semantics, and provisioning
//! idempotence. Each function takes an [`IndexFactory`] that
//! constructs a fresh index over a target the check names (and owns
//! the lifetime of), so suites are contamination-free and repeatable.
//! Every check returns [`Err`] naming the violated contract — unwrap
//! it from test code, where panicking on a violation is the point.
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
//! contract::upsert_replaces_without_leaking(&factory()).await.unwrap();
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

/// The dimension the contract checks embed at.
///
/// Two axes for exact cosine control; the planting below uses unit
/// axis vectors, so ordering assertions carry no floating-point
/// ambiguity.
pub const CONTRACT_DIM: usize = 2;

/// The factory type every backend supplies: one fresh index per call,
/// over a target the check names (and owns the lifetime of).
pub type IndexFactory = Box<
    dyn Fn(String, usize) -> Pin<Box<dyn Future<Output = Box<dyn VectorIndex>> + Send>>
        + Send
        + Sync,
>;

/// A unique target name for one check, namespaced by its slug.
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
/// # Errors
///
/// Returns an error naming the violated contract when the backend
/// leaks the replaced vector or answers from the old neighborhood.
pub async fn upsert_replaces_without_leaking(make: &IndexFactory) -> Result<(), String> {
    let index = make(unique_name("upsert"), CONTRACT_DIM).await;
    let id = Uuid::new_v4();
    index
        .add(id, axis(0))
        .await
        .map_err(|error| format!("the first add must land: {error}"))?;
    index
        .add(id, axis(1))
        .await
        .map_err(|error| format!("the second add must replace: {error}"))?;
    if index.len() != 1 {
        return Err(format!(
            "a replace under one id never leaks a second vector: len={}",
            index.len()
        ));
    }
    let hits = index
        .search(&axis(1), 5)
        .await
        .map_err(|error| format!("search after replace must run: {error}"))?;
    if hits.first().map(|hit| (hit.id, hit.score)) != Some((id, 1.0)) {
        return Err(format!(
            "the replaced vector's neighborhood answers, not the old one's: {hits:?}"
        ));
    }
    Ok(())
}

/// Search orders by descending cosine similarity and respects `k`.
///
/// A planted nearest/farthest set with exact unit vectors: the top hit
/// is exact, scores never ascend, `k` caps the result, and a short
/// store returns fewer than `k` rather than padding.
///
/// # Errors
///
/// Returns an error naming the violated contract on any ordering or
/// `k` violation.
pub async fn search_orders_by_cosine_and_respects_k(make: &IndexFactory) -> Result<(), String> {
    let index = make(unique_name("ordering"), CONTRACT_DIM).await;
    let nearest = Uuid::new_v4();
    let middle = Uuid::new_v4();
    let farthest = Uuid::new_v4();
    index
        .add(nearest, axis(0))
        .await
        .map_err(|error| format!("the nearest plant must land: {error}"))?;
    index
        .add(middle, diagonal())
        .await
        .map_err(|error| format!("the middle plant must land: {error}"))?;
    index
        .add(farthest, axis(1))
        .await
        .map_err(|error| format!("the farthest plant must land: {error}"))?;

    let hits = index
        .search(&axis(0), 3)
        .await
        .map_err(|error| format!("the full ordering search must run: {error}"))?;
    if hits.first().map(|hit| hit.id) != Some(nearest) {
        return Err(format!("the exact-aligned vector is the top hit: {hits:?}"));
    }
    let ascends = hits.windows(2).any(|pair| {
        let [first, second] = pair else {
            return false;
        };
        first.score < second.score && (second.score - first.score).abs() >= 1e-5
    });
    if ascends {
        return Err(format!("scores never ascend: {hits:?}"));
    }
    if hits.len() != 3 {
        return Err(format!(
            "a store of three answers k=3 with three: {} hits",
            hits.len()
        ));
    }

    let capped = index
        .search(&axis(0), 1)
        .await
        .map_err(|error| format!("the k=1 search must run: {error}"))?;
    if capped.len() != 1 {
        return Err(format!("k=1 returns exactly one match: {capped:?}"));
    }
    Ok(())
}

/// A `k` of zero returns an empty result on every backend.
///
/// The `k` discipline's zero edge: no backend may error on it (the
/// server-side `limit` validators reject `0`), and no backend may
/// ignore it and return rows — the answer is exactly no matches.
///
/// # Errors
///
/// Returns an error naming the violated contract when a zero `k`
/// errors or returns rows.
pub async fn zero_k_returns_empty(make: &IndexFactory) -> Result<(), String> {
    let index = make(unique_name("zero_k"), CONTRACT_DIM).await;
    let planted = Uuid::new_v4();
    index
        .add(planted, axis(0))
        .await
        .map_err(|error| format!("one vector must plant: {error}"))?;
    let hits = index
        .search(&axis(0), 0)
        .await
        .map_err(|error| format!("a zero k must succeed, never error: {error}"))?;
    if !hits.is_empty() {
        return Err(format!(
            "a zero k returns exactly no matches, never rows: {hits:?}"
        ));
    }
    Ok(())
}

/// Removing an id shrinks `len` and the id never returns from search.
///
/// # Errors
///
/// Returns an error naming the violated contract when a removed id
/// survives in the count or the results.
pub async fn remove_then_len_shrinks_and_id_is_gone(make: &IndexFactory) -> Result<(), String> {
    let index = make(unique_name("remove"), CONTRACT_DIM).await;
    let kept = Uuid::new_v4();
    let removed = Uuid::new_v4();
    index
        .add(kept, axis(0))
        .await
        .map_err(|error| format!("the kept plant must land: {error}"))?;
    index
        .add(removed, axis(0))
        .await
        .map_err(|error| format!("the removed plant must land: {error}"))?;
    if index.len() != 2 {
        return Err(format!("both vectors count: len={}", index.len()));
    }
    index
        .remove(removed)
        .await
        .map_err(|error| format!("the remove must land: {error}"))?;
    if index.len() != 1 {
        return Err(format!(
            "the removed id leaves the count: len={}",
            index.len()
        ));
    }
    let hits = index
        .search(&axis(0), 10)
        .await
        .map_err(|error| format!("search after remove must run: {error}"))?;
    if hits.iter().any(|hit| hit.id == removed) {
        return Err(format!(
            "the removed id never returns from search: {hits:?}"
        ));
    }
    index
        .remove(removed)
        .await
        .map_err(|error| format!("the re-remove must be a no-op: {error}"))?;
    if index.len() != 1 {
        return Err(format!(
            "removing an absent id changes nothing: len={}",
            index.len()
        ));
    }
    let never_added = Uuid::new_v4();
    index
        .remove(never_added)
        .await
        .map_err(|error| format!("a never-stored id's remove is the trait's no-op: {error}"))?;
    if index.len() != 1 {
        return Err(format!(
            "a remove of an id no one ever stored changes nothing — the count \
             saturates, never wraps: len={}",
            index.len()
        ));
    }
    Ok(())
}

/// A wrong-dimension vector rejects at `add` and nothing is stored.
///
/// # Errors
///
/// Returns an error naming the violated contract when a
/// wrong-dimension vector is accepted or stored.
pub async fn dim_mismatch_rejects_at_add(make: &IndexFactory) -> Result<(), String> {
    let index = make(unique_name("dim_mismatch"), CONTRACT_DIM).await;
    let wrong = Embedding::new(vec![1.0, 0.0, 0.0]);
    let rejection = match index.add(Uuid::new_v4(), wrong).await {
        Ok(()) => {
            return Err("a wrong-dimension vector must reject, never store".to_string());
        }
        Err(rejection) => rejection,
    };
    if !names_the_dimension(&rejection) {
        return Err(format!(
            "the rejection is a memory error naming the dimension: {rejection}"
        ));
    }
    match index.len() {
        0 => {}
        stored => {
            return Err(format!("the rejected vector stored nothing: len={stored}"));
        }
    }
    Ok(())
}

/// Two constructions against one target provision exactly once.
///
/// The idempotence contract: the second construction is a no-op (no
/// error, no duplicate target), and both handles see the same store —
/// a point added through the second is visible to the first.
///
/// # Errors
///
/// Returns an error naming the violated contract when the second
/// construction errors or the handles disagree.
pub async fn provisioning_is_idempotent(make: &IndexFactory) -> Result<(), String> {
    let name = unique_name("idempotent");
    let first = make(name.clone(), CONTRACT_DIM).await;
    let second = make(name, CONTRACT_DIM).await;
    let id = Uuid::new_v4();
    second
        .add(id, axis(0))
        .await
        .map_err(|error| format!("the add through the second handle must land: {error}"))?;
    let hits = first
        .search(&axis(0), 5)
        .await
        .map_err(|error| format!("the search through the first handle must run: {error}"))?;
    if !hits.iter().any(|hit| hit.id == id) {
        return Err(format!(
            "both handles share one provisioned target: {hits:?}"
        ));
    }
    Ok(())
}

/// The rejection shape [`dim_mismatch_rejects_at_add`] accepts.
///
/// A backend may reject through [`LoopError::Memory`] or any error
/// whose text names the dimension — one shared predicate so the
/// contract's tolerance is stated once.
#[must_use]
pub fn names_the_dimension(rejection: &LoopError) -> bool {
    matches!(rejection, LoopError::Memory(_))
        || rejection.to_string().to_lowercase().contains("dim")
}
