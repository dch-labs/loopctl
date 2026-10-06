//! pgvector contract suite — the shared harness plus the golden-set bar
//! over a live Postgres.
//!
//! Gated on `LOOPCTL_VECTOR_E2E=1`: the suite needs a reachable
//! Postgres with the `pgvector` extension at `PG_VECTOR_URL` (default
//! `postgres://postgres:postgres@localhost:5432/postgres`). Without
//! the gate the binary skips with a printed reason instead of failing
//! hermetic runs.
//!
//! Run: `LOOPCTL_VECTOR_E2E=1 cargo test -p loopctl-vector --features
//! pgvector --test pgvector`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use loopctl::memory::vector::VectorIndex as _;
use loopctl_vector::contract;
use loopctl_vector::contract::IndexFactory;
use loopctl_vector::pgvector::PgVectorIndexBuilder;
use std::sync::Arc;

/// The suite's connection profile.
///
/// `PG_VECTOR_URL` when set (CI service, docker run), the local
/// default otherwise; the table is always unique per test.
fn url() -> String {
    std::env::var("PG_VECTOR_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/postgres".to_string())
}

/// The backend's index factory for the shared suite.
fn factory() -> IndexFactory {
    let url = url();
    Box::new(move |table, dim| {
        let url = url.clone();
        Box::pin(async move {
            let index = PgVectorIndexBuilder::new(url, table, dim)
                .connect()
                .await
                .expect("the pgvector server is reachable under the e2e gate");
            Box::new(index) as Box<dyn loopctl::memory::vector::VectorIndex>
        })
    })
}

/// Skip the suite unless the e2e gate is set.
fn gated() -> bool {
    if std::env::var("LOOPCTL_VECTOR_E2E").is_ok_and(|value| value == "1") {
        return true;
    }
    println!("skipping: set LOOPCTL_VECTOR_E2E=1 with a reachable Postgres at PG_VECTOR_URL");
    false
}

#[tokio::test]
async fn upsert_replaces_without_leaking() {
    if gated() {
        contract::upsert_replaces_without_leaking(&factory()).await;
    }
}

#[tokio::test]
async fn search_orders_by_cosine_and_respects_k() {
    if gated() {
        contract::search_orders_by_cosine_and_respects_k(&factory()).await;
    }
}

#[tokio::test]
async fn remove_then_len_shrinks_and_id_is_gone() {
    if gated() {
        contract::remove_then_len_shrinks_and_id_is_gone(&factory()).await;
    }
}

#[tokio::test]
async fn dim_mismatch_rejects_at_add() {
    if gated() {
        contract::dim_mismatch_rejects_at_add(&factory()).await;
    }
}

#[tokio::test]
async fn provisioning_is_idempotent() {
    if gated() {
        contract::provisioning_is_idempotent(&factory()).await;
    }
}

/// The golden-set recall bar through a full `VectorMemoryStore`.
///
/// The out-of-the-box trust claim: loopctl's shared golden fixture
/// through the hashing embedder over this backend meets the same
/// ≥ 80% precision the in-process indexes meet.
#[tokio::test]
async fn recall_parities_against_the_golden_set() {
    use loopctl::memory::LoopMemory as _;
    if !gated() {
        return;
    }
    let set = loopctl::memory::vector_memory::golden_set();
    let index = PgVectorIndexBuilder::new(url(), contract::unique_name("golden"), 128)
        .connect()
        .await
        .expect("the pgvector server is reachable under the e2e gate");
    let store = loopctl::memory::vector_memory::VectorMemoryStore::new(
        Box::new(loopctl::memory::vector::HashingEmbedder::new(128)),
        Box::new(index),
    );
    for fixture_entry in &set.entries {
        store
            .store(fixture_entry.clone())
            .await
            .expect("entry stores");
    }
    let mut hits = Vec::with_capacity(set.queries.len());
    for (position, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(position);
        let returned = store.retrieve(query.text, 3).await.expect("query runs");
        hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let precision = loopctl::memory::vector_memory::GoldenSet::precision_ratio(&hits);
    assert!(
        precision >= 0.8,
        "golden-set precision {precision:.2} is below the 80% gate; missed queries: {:?}",
        hits.iter()
            .zip(set.queries.iter())
            .filter(|(hit, _)| !**hit)
            .map(|(_, query)| query.text)
            .collect::<Vec<_>>()
    );
}

/// The provision event fires once per target per process.
///
/// Creation emits; every later construction against the existing
/// target stays silent — the once-per-target claim, asserted by
/// counting through a global capturing subscriber that filters on the
/// test's unique target name (parallel tests construct their own
/// uniquely named targets, so their emissions never count).
#[tokio::test]
async fn provision_event_fires_once_per_target() {
    if !gated() {
        return;
    }
    let name = contract::unique_name("provision");
    let counter = Arc::new(ProvisionCounter::new(name.clone()));
    let _ignored = tracing::subscriber::set_global_default(CounterSubscriber {
        counter: Arc::clone(&counter),
    });
    let first = PgVectorIndexBuilder::new(url(), name.clone(), contract::CONTRACT_DIM)
        .connect()
        .await
        .expect("the first construction provisions the table");
    drop(first);
    let _second = PgVectorIndexBuilder::new(url(), name, contract::CONTRACT_DIM)
        .connect()
        .await
        .expect("the second construction reuses the target");
    assert_eq!(
        counter.count(),
        1,
        "creation emits exactly one provision event; the reuse emits none"
    );
}

/// A minimal event counter scoped to one target name.
///
/// Counts `loopctl.vector.provision` events whose `target_name` label
/// matches the name this counter was built for — the filter that makes
/// a global subscriber safe beside parallel tests constructing their
/// own uniquely named targets.
struct ProvisionCounter {
    /// The only target name whose events count.
    target: String,
    /// How many matching provision events have been seen.
    count: std::sync::atomic::AtomicUsize,
}

impl ProvisionCounter {
    /// Build a counter scoped to `target`.
    fn new(target: String) -> Self {
        Self {
            target,
            count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The number of matching events observed so far.
    fn count(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// The subscriber shell around [`ProvisionCounter`].
struct CounterSubscriber {
    /// The shared counter the visitor increments.
    counter: Arc<ProvisionCounter>,
}

impl tracing::Subscriber for CounterSubscriber {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target() == "loopctl::metrics"
    }

    fn event(&self, event: &tracing::Event<'_>) {
        let mut grab = EventGrabber::default();
        event.record(&mut grab);
        if grab.metric.as_deref() == Some("loopctl.vector.provision")
            && grab.target_name.as_deref() == Some(self.counter.target.as_str())
        {
            self.counter
                .count
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// The field visitor pulling the `metric` and `target_name` labels.
#[derive(Default)]
struct EventGrabber {
    /// The grabbed metric name, when the event carried one.
    metric: Option<String>,
    /// The grabbed target name, when the event carried one.
    target_name: Option<String>,
}

impl tracing::field::Visit for EventGrabber {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "metric" => self.metric = Some(value.to_string()),
            "target_name" => self.target_name = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "metric" => self.metric = Some(format!("{value:?}")),
            "target_name" => self.target_name = Some(format!("{value:?}")),
            _ => {}
        }
    }
}

#[tokio::test]
async fn pgvector_cosine_distance_maps_to_similarity_order() {
    if !gated() {
        return;
    }
    let index = PgVectorIndexBuilder::new(
        url(),
        contract::unique_name("mapping"),
        contract::CONTRACT_DIM,
    )
    .connect()
    .await
    .expect("the pgvector server is reachable under the e2e gate");
    let aligned = uuid::Uuid::new_v4();
    let diagonal = uuid::Uuid::new_v4();
    let opposed = uuid::Uuid::new_v4();
    index
        .add(aligned, contract::axis(0))
        .await
        .expect("aligned");
    let mut both = vec![1.0_f32; contract::CONTRACT_DIM];
    both.fill(1.0);
    index
        .add(diagonal, loopctl::memory::vector::Embedding::new(both))
        .await
        .expect("diagonal");
    index
        .add(opposed, contract::axis(1))
        .await
        .expect("opposed");
    let hits = index
        .search(&contract::axis(0), 3)
        .await
        .expect("the mapping search runs");
    let order: Vec<uuid::Uuid> = hits.iter().map(|hit| hit.id).collect();
    assert_eq!(
        order,
        vec![aligned, diagonal, opposed],
        "cosine distance maps to descending similarity exactly: {hits:?}"
    );
    let aligned_score = hits.first().map(|hit| hit.score);
    assert!(
        aligned_score.is_some_and(|score| (score - 1.0).abs() < 1e-4),
        "the aligned vector's score is cosine similarity 1.0: {hits:?}"
    );
    let opposed_score = hits.last().map(|hit| hit.score);
    assert!(
        opposed_score.is_some_and(|score| score.abs() < 1e-4),
        "the opposed vector's score is cosine similarity 0.0: {hits:?}"
    );
}
