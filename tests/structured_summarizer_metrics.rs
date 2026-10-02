//! Pins the structured summarizer's sections counter through a
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

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use futures::Stream;
use loopctl::api::ApiClient;
use loopctl::api::NonStreamingResponse;
use loopctl::api::StreamRequest;
use loopctl::api::error::ApiError;
use loopctl::compact::StructuredSummarizer;
use loopctl::compact::StructuredSummaryConfig;
use loopctl::compact::{CompactReason, CompactionContext, CompactionOutcome, ContextCompactor};
use loopctl::message::Message;
use loopctl::stream::{StreamEvent, StreamStopReason};

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the summarizer's metric events are asserted rather than
/// trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// Only the structured summarizer's sections-counter events, oldest
    /// first.
    fn section_events(&self) -> Vec<String> {
        self.events
            .lock()
            .expect("capture lock")
            .iter()
            .filter(|event| event.contains("metric=loopctl.compaction.structured.sections"))
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

/// An `ApiClient` double serving the scripted responses in order.
struct ScriptedClient {
    /// The responses the double serves front-first, one per call.
    script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        "structured-metrics".to_string()
    }

    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }

    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        let next = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .expect("every scripted call gets a response");
        Box::pin(std::future::ready(next))
    }
}

fn ok(text: &str) -> Result<NonStreamingResponse, ApiError> {
    Ok(NonStreamingResponse {
        message: Message::assistant(text),
        stop_reason: StreamStopReason::EndTurn,
        usage: None,
    })
}

/// Six user/assistant pairs, long enough to clear the summarizer's
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

fn context_for(messages: &[Message]) -> CompactionContext {
    CompactionContext::new(
        CompactionOutcome::estimate_tokens(messages),
        CompactReason::ThresholdExceeded,
        1_000_000,
        3,
        Arc::new(loopctl::compact::HeuristicTokenCounter),
    )
}

#[tokio::test]
async fn the_sections_counter_reports_populated_and_empty() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let client = Arc::new(ScriptedClient {
        script: Mutex::new(
            vec![ok(
                "### Key facts\n- src/api.rs uses reqwest\n- the window is 200k",
            )]
            .into(),
        ),
    });
    let summarizer = StructuredSummarizer::new(client, StructuredSummaryConfig::default());
    let messages = conversation();
    let outcome = summarizer
        .compact(messages, 40_000, context_for(&conversation()))
        .await;
    assert!(outcome.success, "the one-section pass completes");
    let events = capture.section_events();
    assert_eq!(
        events.len(),
        4,
        "a pass emits one counter event per configured section, populated or not: {events:?}"
    );
    assert!(
        events[0].contains("section=facts") && events[0].contains("state=populated"),
        "the populated section reports itself: {}",
        events[0]
    );
    for (event, slug) in events
        .iter()
        .zip(["facts", "decisions", "todos", "questions"])
    {
        assert!(
            event.contains(&format!("section={slug}")),
            "every event carries its section slug: {event}"
        );
    }
    assert!(
        events[1..]
            .iter()
            .all(|event| event.contains("state=empty")),
        "an absent section reports empty, never silence — a climbing empty rate is the \
         earliest degradation signal: {:?}",
        &events[1..]
    );
}
