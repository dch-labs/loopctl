//! Pins the embedders' per-batch metric events through a process-global
//! tracing capture.
//!
//! The capture lives in its own test binary so it can claim the process
//! global subscriber — the same one-test, one-binary discipline as
//! `loopctl-hnsw`'s `search_metrics.rs`, because a global subscriber
//! cannot be shared across concurrently running tests and every
//! callsite's interest is cached against the global from registration.
//!
//! Each scenario's assertions live in their own `assert_*_settle` helper, so
//! the three settle contracts read as three named claims rather than one
//! long body: a failure names which settle broke.
//!
//! Run: `cargo test --all-features --test embeddings_metrics`

#![cfg(all(feature = "openai", feature = "ollama", feature = "vector_index"))]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects
)]

use std::sync::Arc;
use std::sync::Mutex;

use loopctl::provider::embeddings::{OllamaEmbedder, OpenAiEmbedder};

/// The metric name every scenario settles exactly once, per batch.
const BATCH: &str = "loopctl.embed.batch.duration_ms";

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the embedding paths' metric events are asserted rather
/// than trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    ///
    /// One row per `loopctl::`-targeted event the subscriber accepted;
    /// the test snapshots the length between scenarios and asserts on
    /// the delta, so cross-scenario leakage is impossible to miss.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// The captured events, oldest first.
    ///
    /// Cloned out under the capture lock; the single owning test reads
    /// this only after the embed call under test has settled.
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

/// The captured events naming `wanted` in their `metric=` field, oldest
/// first.
///
/// The match is on the whole `field=value` token rather than a substring, so
/// a row's name cannot be satisfied by an event carrying a longer name that
/// merely starts with it.
fn named<'a>(events: &'a [String], wanted: &str) -> Vec<&'a String> {
    events
        .iter()
        .filter(|event| {
            event
                .split_whitespace()
                .any(|field| field == format!("metric={wanted}"))
        })
        .collect()
}

/// How many captured events name `wanted`.
///
/// The per-name count is what a duplicate or a vanished signal trips on;
/// see [`assert_success_settle`] for why the total is not the invariant.
fn count_named(events: &[String], wanted: &str) -> usize {
    named(events, wanted).len()
}

/// Pin the successful batch's settle: the three declared signals once each,
/// and the batch event's whole fingerprint.
///
/// One batch settles one duration event — and the same settle also carries
/// the two counters the catalog declares for an embedding call
/// (`embed.requests`, `embed.inputs`). The invariant a duplicate would break
/// is therefore per-name, not "one event total": counting by metric still
/// fails a doubled batch event, and it fails a signal that stopped riding the
/// path, which an absolute count cannot tell apart.
fn assert_success_settle(delta: &[String]) {
    assert_eq!(
        count_named(delta, BATCH),
        1,
        "one batch, one settle event — the success path settles exactly one: {delta:?}"
    );
    assert_eq!(
        count_named(delta, "loopctl.embed.requests"),
        1,
        "one request settles as exactly one request counter: {delta:?}"
    );
    assert_eq!(
        count_named(delta, "loopctl.embed.inputs"),
        1,
        "one request settles as exactly one input counter: {delta:?}"
    );
    assert_eq!(
        delta.len(),
        3,
        "the settle carries exactly the three declared signals and nothing else: {delta:?}"
    );
    let batch = delta
        .iter()
        .find(|event| event.contains("metric=loopctl.embed.batch.duration_ms"))
        .expect("the batch settles its duration event");
    assert!(
        batch.contains("metric=loopctl.embed.batch.duration_ms")
            && !batch.contains("metric=loopctl.embed.batch ")
            && batch.contains("provider=openai")
            && batch.contains("model=text-embedding-3-small")
            && batch.contains("dim=256")
            && batch.contains("inputs=2")
            && batch.contains("outcome=ok")
            && batch.contains("prompt_tokens=7")
            && batch.contains("total_tokens=9"),
        "the event carries the whole batch fingerprint: {delta:?}"
    );
}

/// Pin a failed batch's settle, and the two ways its wire failure narrates
/// itself.
///
/// The failed wire request speaks twice over: one forensic event classifying
/// the 500 — the only event in the delta that carries no `metric=` field —
/// and one counter event for the catalog's `provider.wire_failures` row. The
/// batch metric and both of these are separate concerns, so the settle
/// invariant counts only the batch event by name.
fn assert_failed_settle(delta: &[String]) {
    let settled = named(delta, BATCH);
    assert_eq!(
        settled.len(),
        1,
        "one batch, one settle event — the failure path settles exactly one: {delta:?}"
    );
    assert!(
        settled[0].contains("metric=loopctl.embed.batch.duration_ms")
            && settled[0].contains("provider=ollama")
            && settled[0].contains("outcome=error"),
        "the failed batch's event names its metric, provider, and outcome: {delta:?}"
    );
    let wire: Vec<&String> = delta
        .iter()
        .filter(|event| !event.contains(" metric="))
        .collect();
    assert_eq!(
        wire.len(),
        1,
        "the 500 answers with exactly one wire-failure forensic: {delta:?}"
    );
    assert!(
        wire[0].contains("status=500") && wire[0].contains("class=server"),
        "the wire-failure forensic names the status and its class: {wire:?}"
    );
    let counted = named(delta, "loopctl.provider.wire_failures");
    assert_eq!(
        counted.len(),
        1,
        "the same 500 settles exactly one wire-failure counter: {delta:?}"
    );
    assert!(
        counted[0].contains("class=server"),
        "the wire-failure counter names its class: {counted:?}"
    );
}

/// Pin the dimension-mismatch settle: a failed batch, its two counters, and
/// the WARN that names both lengths.
fn assert_mismatch_settle(delta: &[String]) {
    // The failed settle carries the same three signals as a successful one
    // (batch duration with `outcome=error`, the request counter, the input
    // counter) plus the dimension-mismatch WARN — four events, one per name.
    assert_eq!(
        delta.len(),
        4,
        "a mismatch settles its failed batch event, its two counters, and the WARN: {delta:?}"
    );
    let warn = delta
        .iter()
        .find(|event| event.contains("model may have been swapped"))
        .expect("the dimension mismatch emits its WARN metric event");
    assert!(
        warn.contains("provider=openai")
            && warn.contains("expected=256")
            && warn.contains("received=2"),
        "the WARN names the provider and both lengths: {warn}"
    );
}

#[tokio::test]
async fn every_batch_settles_one_event_and_a_mismatch_warns() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let ok_server = httpmock::MockServer::start_async().await;
    let row = |lead: f32| format!("[{lead},{}]", vec!["0.0"; 255].join(","));
    ok_server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(format!(
                "{{\"data\":[{{\"index\":0,\"embedding\":{}}},{{\"index\":1,\"embedding\":{}}}],\
                 \"usage\":{{\"prompt_tokens\":7,\"total_tokens\":9}}}}",
                row(1.0),
                row(2.0)
            ));
        })
        .await;
    let openai = OpenAiEmbedder::builder()
        .with_api_key("sk-metrics-test")
        .with_base_url(ok_server.base_url())
        .with_dimensions(256)
        .build()
        .unwrap();
    let before = capture.events().len();
    let embedded = openai.embed_texts(&["first", "second"]).await.unwrap();
    assert_eq!(embedded.len(), 2, "the success scenario embeds both inputs");
    assert_success_settle(&capture.events()[before..]);

    let error_server = httpmock::MockServer::start_async().await;
    error_server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(500).body("{\"error\":\"boom\"}");
        })
        .await;
    let ollama = OllamaEmbedder::builder()
        .with_model("metrics-test-embed")
        .with_dim(3)
        .with_base_url(error_server.base_url())
        .build()
        .unwrap();
    let before = capture.events().len();
    let failed = ollama.embed_texts(&["one text"]).await;
    assert!(failed.is_err(), "the failure scenario must fail the embed");
    assert_failed_settle(&capture.events()[before..]);

    let mismatch_server = httpmock::MockServer::start_async().await;
    mismatch_server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .body("{\"data\":[{\"index\":0,\"embedding\":[1.0,2.0]}]}");
        })
        .await;
    let swapped = OpenAiEmbedder::builder()
        .with_api_key("sk-metrics-test")
        .with_base_url(mismatch_server.base_url())
        .with_dimensions(256)
        .build()
        .unwrap();
    let before = capture.events().len();
    let mismatch = swapped.embed_texts(&["one text"]).await;
    assert!(
        mismatch.is_err(),
        "the swapped-dimension scenario must fail the embed"
    );
    assert_mismatch_settle(&capture.events()[before..]);
}
