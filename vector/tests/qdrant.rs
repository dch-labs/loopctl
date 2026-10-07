//! Qdrant contract suite — the shared harness plus the golden-set bar
//! over a live server.
//!
//! Gated on `LOOPCTL_VECTOR_E2E=1` (the recorded-exchange convention):
//! the suite needs a reachable Qdrant at `QDRANT_URL` (default
//! `http://localhost:6334`, gRPC). Without the gate the binary skips
//! with a printed reason instead of failing hermetic runs.
//!
//! Run: `LOOPCTL_VECTOR_E2E=1 cargo test -p loopctl-vector --features
//! qdrant --test qdrant`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use loopctl::memory::vector::VectorIndex;
use loopctl_vector::contract;
use loopctl_vector::contract::IndexFactory;
use loopctl_vector::qdrant::QdrantIndex;
use std::sync::Arc;

/// The suite's connection profile.
///
/// `QDRANT_URL` when set (CI service, docker run), the local default
/// otherwise; the collection is always unique per test.
fn url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string())
}

/// The backend's index factory for the shared suite.
fn factory() -> IndexFactory {
    let url = url();
    Box::new(move |collection, dim| {
        let url = url.clone();
        Box::pin(async move {
            let index = QdrantIndex::builder(url, collection, dim)
                .connect()
                .await
                .expect("the qdrant server is reachable under the e2e gate");
            Box::new(index) as Box<dyn loopctl::memory::vector::VectorIndex>
        })
    })
}

/// Skip the suite unless the e2e gate is set.
fn gated() -> bool {
    if std::env::var("LOOPCTL_VECTOR_E2E").is_ok_and(|value| value == "1") {
        return true;
    }
    println!("skipping: set LOOPCTL_VECTOR_E2E=1 with a reachable Qdrant at QDRANT_URL");
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
    let index = QdrantIndex::builder(url(), contract::unique_name("golden"), 128)
        .connect()
        .await
        .expect("the qdrant server is reachable under the e2e gate");
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

/// Concurrent constructions against a fresh collection both succeed.
///
/// The idempotence contract under its hardest shape: two connects race
/// the create; the loser must see the winner's collection and proceed,
/// not error with already-exists.
#[tokio::test]
async fn concurrent_construction_against_a_fresh_collection_both_succeed() {
    if !gated() {
        return;
    }
    for round in 0..3 {
        let name = contract::unique_name(&format!("race{round}"));
        let url = url();
        let (first, second) = tokio::join!(
            QdrantIndex::builder(url.clone(), name.clone(), contract::CONTRACT_DIM).connect(),
            QdrantIndex::builder(url, name, contract::CONTRACT_DIM).connect(),
        );
        assert!(
            first.is_ok() && second.is_ok(),
            "both racing constructions succeed against one collection:              {first:?} {second:?}"
        );
    }
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
    let first = QdrantIndex::builder(url(), name.clone(), contract::CONTRACT_DIM)
        .connect()
        .await
        .expect("the first construction provisions");
    drop(first);
    let _second = QdrantIndex::builder(url(), name, contract::CONTRACT_DIM)
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

/// A raw client for fixture collections the index must refuse.
///
/// Fixture collections are created outside the index under test so
/// its connect-time validation judges shapes it would never itself
/// produce.
fn fixture_client() -> qdrant_client::Qdrant {
    qdrant_client::Qdrant::from_url(&url())
        .skip_compatibility_check()
        .build()
        .expect("the lazy channel builds without a server")
}

#[tokio::test]
async fn a_dimension_mismatched_target_rejects_at_connect() {
    if !gated() {
        return;
    }
    let collection = contract::unique_name("dim_mismatch");
    let _first = QdrantIndex::builder(url(), collection.clone(), 4)
        .connect()
        .await
        .expect("the dim-4 construction provisions the collection");
    let rejection = QdrantIndex::builder(url(), collection, 8).connect().await;
    let text = rejection
        .expect_err("a mismatched dim rejects at connect, not on use")
        .to_string();
    assert!(
        text.contains("holds 4-dimensional vectors")
            && text.contains("configured for 8 dimensions"),
        "the rejection names the collection's dimension and the requested one: {text}"
    );
}

#[tokio::test]
async fn a_named_vector_collection_rejects_at_connect() {
    if !gated() {
        return;
    }
    let collection = contract::unique_name("named_vectors");
    let client = fixture_client();
    let named = qdrant_client::qdrant::vectors_config::Config::ParamsMap(
        qdrant_client::qdrant::VectorParamsMap {
            map: [(
                "default".to_string(),
                qdrant_client::qdrant::VectorParams {
                    size: 4,
                    distance: i32::from(qdrant_client::qdrant::Distance::Cosine),
                    ..Default::default()
                },
            )]
            .into(),
        },
    );
    client
        .create_collection(
            qdrant_client::qdrant::CreateCollectionBuilder::new(collection.clone())
                .vectors_config(named),
        )
        .await
        .expect("the named-vector fixture collection creates");
    let rejection = QdrantIndex::builder(url(), collection, 4).connect().await;
    assert!(
        rejection
            .as_ref()
            .is_err_and(|error| error.to_string().contains("named")),
        "the rejection names the unsupported layout: {rejection:?}"
    );
}

#[tokio::test]
async fn a_non_cosine_collection_rejects_at_connect() {
    if !gated() {
        return;
    }
    let collection = contract::unique_name("euclid");
    let client = fixture_client();
    client
        .create_collection(
            qdrant_client::qdrant::CreateCollectionBuilder::new(collection.clone()).vectors_config(
                qdrant_client::qdrant::VectorParamsBuilder::new(
                    4,
                    qdrant_client::qdrant::Distance::Euclid,
                ),
            ),
        )
        .await
        .expect("the euclid fixture collection creates");
    let rejection = QdrantIndex::builder(url(), collection, 4).connect().await;
    assert!(
        rejection
            .as_ref()
            .is_err_and(|error| error.to_string().to_lowercase().contains("cosine")),
        "the rejection names the cosine requirement: {rejection:?}"
    );
}

#[tokio::test]
#[ignore = "live cloud credentials required: QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY"]
async fn qdrant_cloud_smoke() {
    let index = QdrantIndex::from_env(2)
        .await
        .expect("the cloud profile connects");
    let id = uuid::Uuid::new_v4();
    index
        .add(id, contract::axis(0))
        .await
        .expect("the cloud collection accepts an upsert");
    let hits = index
        .search(&contract::axis(0), 1)
        .await
        .expect("the cloud collection answers a search");
    assert_eq!(hits.first().map(|hit| hit.id), Some(id));
    index
        .remove(id)
        .await
        .expect("the cloud collection deletes");
}
