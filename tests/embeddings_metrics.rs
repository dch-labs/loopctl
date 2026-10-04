//! Pins the embedders' per-batch metric events through a process-global
//! tracing capture.
//!
//! The capture lives in its own test binary so it can claim the process
//! global subscriber — the same one-test, one-binary discipline as
//! `loopctl-hnsw`'s `search_metrics.rs`, because a global subscriber
//! cannot be shared across concurrently running tests and every
//! callsite's interest is cached against the global from registration.
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
    let delta = capture.events()[before..].to_vec();
    assert_eq!(
        delta.len(),
        1,
        "one batch, one event — the success path settles exactly one: {delta:?}"
    );
    assert!(
        delta[0].contains("metric=loopctl.embed.batch")
            && delta[0].contains("provider=openai")
            && delta[0].contains("model=text-embedding-3-small")
            && delta[0].contains("dim=256")
            && delta[0].contains("inputs=2")
            && delta[0].contains("outcome=ok")
            && delta[0].contains("prompt_tokens=7")
            && delta[0].contains("total_tokens=9"),
        "the event carries the whole batch fingerprint: {delta:?}"
    );

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
    let delta = capture.events()[before..].to_vec();
    assert_eq!(
        delta.len(),
        1,
        "one batch, one event — the failure path settles exactly one: {delta:?}"
    );
    assert!(
        delta[0].contains("metric=loopctl.embed.batch")
            && delta[0].contains("provider=ollama")
            && delta[0].contains("outcome=error"),
        "the failed batch's event names its provider and outcome: {delta:?}"
    );

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
    let delta = capture.events()[before..].to_vec();
    assert_eq!(
        delta.len(),
        2,
        "a mismatch settles its failed batch event plus the WARN: {delta:?}"
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
