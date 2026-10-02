//! Pins the fallback chain's pass and degradation counters through a
//! process-global tracing capture.
//!
//! The capture lives in its own test binary so it can claim the process
//! global subscriber — the one tracing lane that observes a crate-internal
//! emission deterministically, because every callsite's cached interest is
//! computed against the global from the moment it is registered and no
//! sibling test can race the cache.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use loopctl::api::ApiClient;
use loopctl::api::NonStreamingResponse;
use loopctl::api::StreamRequest;
use loopctl::api::error::ApiError;
use loopctl::compact::FallbackCompactor;
use loopctl::compact::types::CompactionContext;
use loopctl::compact::{CompactReason, CompactionOutcome, ContextCompactor, HeuristicTokenCounter};
use loopctl::message::Message;
use loopctl::stream::StreamEvent;

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the chain's metric events are asserted rather than trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// Only the chain's two counter events, oldest first.
    fn chain_events(&self) -> Vec<String> {
        self.events
            .lock()
            .expect("capture lock")
            .iter()
            .filter(|event| {
                event.contains("metric=loopctl.compaction.passes")
                    || event.contains("metric=loopctl.compaction.degradations")
            })
            .cloned()
            .collect()
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

/// An `ApiClient` double whose every call fails, driving both LLM stages
/// of the default chain to decline.
struct FailingClient;

impl ApiClient for FailingClient {
    fn model(&self) -> String {
        "fallback-metrics".to_string()
    }

    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }

    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        Box::pin(std::future::ready(Err(ApiError::api(
            "the scripted provider is unreachable",
        ))))
    }
}

/// Six user/assistant pairs, long enough to clear the terminal stage's
/// minimum-message guard.
fn conversation() -> Vec<Message> {
    let mut messages = Vec::new();
    for turn in 0..6 {
        messages.push(Message::user(format!(
            "user turn {turn} asks about topic-{turn}"
        )));
        messages.push(Message::assistant(format!(
            "assistant turn {turn} decided fact-{turn}"
        )));
    }
    messages
}

#[tokio::test]
async fn passes_and_degradations_counters_report_the_chain_s_shape() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
    let messages = conversation();
    let context = CompactionContext::new(
        CompactionOutcome::estimate_tokens(&messages),
        CompactReason::ThresholdExceeded,
        1_000_000,
        3,
        Arc::new(HeuristicTokenCounter),
    );
    let outcome = chain.compact(messages, 40_000, context).await;
    assert!(outcome.success, "the terminal truncate carried the pass");

    let events = capture.chain_events();
    let passes: Vec<&String> = events
        .iter()
        .filter(|event| event.contains("metric=loopctl.compaction.passes"))
        .collect();
    assert_eq!(
        passes.len(),
        1,
        "one pass event per succeeded run, at the winning stage: {events:?}"
    );
    assert!(
        passes[0].contains("stage=TruncatingCompactor"),
        "the pass event names the stage that served it: {}",
        passes[0]
    );
    let degradations: Vec<&String> = events
        .iter()
        .filter(|event| event.contains("metric=loopctl.compaction.degradations"))
        .collect();
    assert_eq!(
        degradations.len(),
        2,
        "one degradation event per failed stage the winner declined through: {degradations:?}"
    );
    let declined_from: Vec<&str> = degradations
        .iter()
        .filter_map(|event| event.split_whitespace().find(|f| f.starts_with("from=")))
        .collect();
    assert!(
        declined_from.contains(&"from=QaSummarizer")
            && declined_from.contains(&"from=StructuredSummarizer"),
        "both LLM declines are named: {degradations:?}"
    );
    assert!(
        degradations
            .iter()
            .all(|event| event.contains("to=TruncatingCompactor")),
        "the truncate that carried the pass is the destination: {degradations:?}"
    );
}
