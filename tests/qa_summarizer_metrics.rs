//! Pins the QA summarizer's pass-level metric events through a
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
use loopctl::compact::qa_summarizer::QaSummarizer;
use loopctl::compact::qa_summarizer::QaSummarizerConfig;
use loopctl::compact::{CompactReason, CompactionContext, CompactionOutcome, ContextCompactor};
use loopctl::message::Message;
use loopctl::stream::{StreamEvent, StreamStopReason, Usage};

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the summarizer's metric events are asserted rather than
/// trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// The captured events, oldest first.
    fn events(&self) -> Vec<String> {
        self.events.lock().expect("capture lock").clone()
    }

    /// Only the summarizer's token-metric events, oldest first.
    fn metric_events(&self) -> Vec<String> {
        self.events()
            .into_iter()
            .filter(|event| event.contains("metric=loopctl.compaction.summarizer.tokens"))
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

/// The numeric value of `name=` within a captured event line.
fn captured_number(event: &str, name: &str) -> Option<u64> {
    event
        .split_whitespace()
        .find(|field| field.starts_with(name))
        .and_then(|field| field.strip_prefix(name))
        .and_then(|value| value.parse().ok())
}

/// An `ApiClient` double serving the scripted responses in order.
struct ScriptedClient {
    script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
}

impl ScriptedClient {
    /// A double whose calls consume the script front-first.
    fn new(script: Vec<Result<NonStreamingResponse, ApiError>>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into()),
        })
    }
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        "qa-metrics".to_string()
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

fn ok_with_usage(
    text: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> Result<NonStreamingResponse, ApiError> {
    Ok(NonStreamingResponse {
        message: Message::assistant(text),
        stop_reason: StreamStopReason::EndTurn,
        usage: Some(Usage::new(input_tokens, output_tokens)),
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
async fn summarizer_metric_events_carry_direction_and_the_sticky_estimate_flag() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let client = ScriptedClient::new(vec![
        ok_with_usage("S", 100, 10),
        ok_with_usage("[]", 200, 20),
    ]);
    let summarizer = QaSummarizer::new(client, QaSummarizerConfig::default());
    let messages = conversation();
    let outcome = summarizer
        .compact(messages, 40_000, context_for(&conversation()))
        .await;
    assert!(outcome.success, "the fully-reported pass completes");
    let metric_events = capture.metric_events();
    assert_eq!(
        metric_events.len(),
        2,
        "a pass emits one metric event per direction: {:?}",
        capture.events()
    );
    let input_event = metric_events
        .iter()
        .find(|event| event.contains("direction=in"))
        .expect("the input direction is reported");
    assert_eq!(
        captured_number(input_event, "tokens="),
        Some(300),
        "a fully-reported pass sums the two calls' reported input usage"
    );
    assert!(
        input_event.contains("estimated=false"),
        "a fully-reported pass reads as billed, not estimated: {input_event}"
    );
    let output_event = metric_events
        .iter()
        .find(|event| event.contains("direction=out"))
        .expect("the output direction is reported");
    assert_eq!(
        captured_number(output_event, "tokens="),
        Some(30),
        "the output direction carries its own reported total"
    );

    let mixed_client = ScriptedClient::new(vec![ok_with_usage("MIXED", 40, 4), ok("[]")]);
    let mixed = QaSummarizer::new(mixed_client, QaSummarizerConfig::default());
    let mixed_messages = conversation();
    let outcome = mixed
        .compact(mixed_messages, 40_000, context_for(&conversation()))
        .await;
    assert!(outcome.success, "the mixed pass completes");
    let metric_events = capture.metric_events();
    assert_eq!(
        metric_events.len(),
        4,
        "the second pass adds its own two events: {metric_events:?}"
    );
    let second_pass: Vec<&String> = metric_events.iter().rev().take(2).collect();
    assert!(
        second_pass
            .iter()
            .all(|event| event.contains("estimated=true")),
        "one estimated call latches the sticky flag across the whole pass: {second_pass:?}"
    );
    let input_event = second_pass
        .iter()
        .find(|event| event.contains("direction=in"))
        .expect("the mixed pass reports its input direction");
    assert!(
        captured_number(input_event, "tokens=").is_some_and(|tokens| tokens > 40),
        "the mixed total folds the reported 40 together with the estimated call: {input_event}"
    );
}
