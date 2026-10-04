//! Pins the HNSW search's per-call metric events through a process-global
//! tracing capture.
//!
//! The capture lives in its own test binary so it can claim the process
//! global subscriber — the one tracing lane that observes a crate-internal
//! emission deterministically, because every callsite's cached interest is
//! computed against the global from the moment it is registered and no
//! sibling test can race the cache.
//!
//! Run: `cargo test -p loopctl-hnsw --test search_metrics`

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use std::sync::Mutex;

use loopctl::memory::vector::Embedding;
use loopctl::memory::vector::VectorIndex;
use loopctl_hnsw::HnswIndex;
use uuid::Uuid;

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the search path's metric events are asserted rather than
/// trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    ///
    /// One row per `loopctl::`-targeted event the subscriber accepted, so
    /// the test asserts on emission counts and field values alike.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// Only the vector-index search metric events, oldest first.
    ///
    /// Filters the full capture down to the `metric=loopctl.vector.index.search`
    /// rows, so counting them is counting searches that reported.
    fn search_events(&self) -> Vec<String> {
        self.events()
            .into_iter()
            .filter(|event| event.contains("metric=loopctl.vector.index.search"))
            .collect()
    }

    /// The captured events, oldest first.
    ///
    /// Cloned out under the capture lock; tests read this only after the
    /// searches under test have completed.
    fn events(&self) -> Vec<String> {
        self.events.lock().expect("capture lock").clone()
    }
}

impl tracing::Subscriber for MetricCapture {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target().starts_with("loopctl::")
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::Id {
        tracing::Id::from_u64(1)
    }
    fn record(&self, _id: &tracing::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct FieldVisitor {
            fields: Vec<String>,
        }
        impl tracing::field::Visit for FieldVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.fields.push(format!("{}={:?}", field.name(), value));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.fields.push(format!("{}={}", field.name(), value));
            }
        }
        let mut visitor = FieldVisitor { fields: Vec::new() };
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture lock")
            .push(visitor.fields.join(" "));
    }
    fn enter(&self, _id: &tracing::Id) {}
    fn exit(&self, _id: &tracing::Id) {}
}

/// The numeric value of `name=` within a captured event line.
///
/// Splits on whitespace so the joined `field=value` render stays
/// parseable, then parses the first field carrying the requested prefix.
fn captured_number(event: &str, name: &str) -> Option<u64> {
    event
        .split_whitespace()
        .find(|field| field.starts_with(name))
        .and_then(|field| field.trim_start_matches(name).parse().ok())
}

/// Every successful search emits exactly one metric event — the degenerate
/// successes included.
///
/// `k = 0`, an empty index, and a populated search must each land one
/// `loopctl.vector.index.search` event with their own `returned` count, matching
/// the linear index's one-event-per-success contract; before the fix only
/// the populated path emitted, so the two backends disagreed observably.
#[tokio::test]
async fn every_successful_search_emits_one_metric_event() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let hnsw = HnswIndex::new(4);
    let query = Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let empty_hits = hnsw.search(&query, 3).await.unwrap();
    assert!(empty_hits.is_empty(), "an empty index returns nothing");

    for index in 0..2_u128 {
        hnsw.add(
            Uuid::from_u128(index),
            Embedding::from_slice(&[1.0, index as f32, 0.0, 0.0]),
        )
        .await
        .unwrap();
    }
    let zero_k_hits = hnsw.search(&query, 0).await.unwrap();
    assert!(zero_k_hits.is_empty(), "k = 0 returns nothing");
    let populated_hits = hnsw.search(&query, 2).await.unwrap();
    assert_eq!(
        populated_hits.len(),
        2,
        "the populated search returns both live vectors"
    );

    let search_events = capture.search_events();
    assert_eq!(
        search_events.len(),
        3,
        "one event per successful search — the degenerate successes \
         included: {:?}",
        capture.events()
    );
    assert!(
        search_events
            .iter()
            .all(|event| event.contains("provider=hnsw")),
        "every search event names its backend: {search_events:?}"
    );
    let returned_counts: Vec<Option<u64>> = search_events
        .iter()
        .map(|event| captured_number(event, "returned="))
        .collect();
    assert_eq!(
        returned_counts,
        vec![Some(0), Some(0), Some(2)],
        "each event carries its own call's returned count — empty, k = 0, \
         then the populated search"
    );
}
