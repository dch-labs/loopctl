//! Prompted structured output: schema-in-prompt, lenient parse, one
//! corrective retry.
//!
//! Run: `cargo test --all-features --test prompted_structured -- --nocapture`
//!
//! Requires the `testing` feature for the recording mock client.

#![cfg(feature = "testing")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::float_cmp
)]

use loopctl::message::Message;
use loopctl::structured::{StructuredError, StructuredOutput};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RouterDecision {
    route: String,
    confidence: f64,
}

impl StructuredOutput for RouterDecision {
    fn name() -> &'static str {
        "router_decision"
    }
    fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "route": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["route", "confidence"],
            "additionalProperties": false
        })
    }
}

fn response(text: &str) -> MockResponse {
    MockResponse {
        text: text.to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

#[tokio::test]
async fn a_prose_wrapped_fenced_answer_parses_on_the_first_try() {
    let client = MockApiClient::new("local-model").with_responses(vec![response(
        "Sure! ```json\n{\"route\": \"local\", \"confidence\": 0.9}\n```",
    )]);
    let decision: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("the lenient scanner recovers fenced, prose-wrapped JSON");
    assert_eq!(decision.route, "local");
    assert_eq!(decision.confidence, 0.9);
    assert_eq!(
        client.create_message_calls(),
        1,
        "a first-try success makes exactly one call"
    );
    assert_eq!(
        client.with_options_calls(),
        0,
        "the prompted path never sends request options"
    );
}

#[tokio::test]
async fn a_corrective_retry_recovers_a_malformed_first_answer() {
    let client = MockApiClient::new("local-model").with_responses(vec![
        response("I cannot answer in JSON."),
        response("{\"route\": \"local\", \"confidence\": 0.9}"),
    ]);
    let decision: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("the corrective retry recovers a recoverable failure");
    assert_eq!(decision.route, "local");
    assert_eq!(
        client.create_message_calls(),
        2,
        "one corrective retry means exactly two calls"
    );
    assert_eq!(
        client.with_options_calls(),
        0,
        "the prompted path never sends request options"
    );
    let captured = client.captured_requests();
    assert_eq!(captured.len(), 2);
    let retry_conversation = &captured[1].messages;
    assert_eq!(
        retry_conversation.len(),
        3,
        "the retry carries the original turn, the failed answer, and the correction"
    );
    let correction = &retry_conversation[2];
    let correction_text = correction.text_content();
    assert!(
        correction_text.contains("not valid JSON for the schema"),
        "the corrective turn names the failure: {correction_text}"
    );
    assert!(
        correction_text.contains("invalid type"),
        "the concrete serde error rides the correction verbatim: {correction_text}"
    );
    assert!(
        retry_conversation[1]
            .text_content()
            .contains("I cannot answer in JSON."),
        "the failed answer stays in the conversation the model conditions on"
    );
}

#[tokio::test]
async fn an_oversized_first_failure_rides_the_retry_bounded() {
    let oversized_prose = format!(
        "I refuse, and here is a long essay. {}",
        "x".repeat(100_000)
    );
    let client = MockApiClient::new("local-model").with_responses(vec![
        response(&oversized_prose),
        response("{\"route\": \"local\", \"confidence\": 0.9}"),
    ]);
    let decision: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("the corrective retry recovers");
    assert_eq!(decision.route, "local");
    let captured = client.captured_requests();
    let correction = &captured[1].messages[2];
    let correction_text = correction.text_content();
    assert!(
        correction_text.contains("not valid JSON for the schema"),
        "the correction still names the failure"
    );
    assert!(
        correction_text.contains("…[truncated]"),
        "the oversized serde error is truncated before riding the retry wire"
    );
    assert!(
        correction_text.chars().count() < 2_400,
        "the corrective turn stays within a small model's budget, got {} chars",
        correction_text.chars().count()
    );
}

#[tokio::test]
async fn a_tool_call_only_failure_still_reports_its_input() {
    let tool_call = MockToolCall {
        id: "call_tc".to_string(),
        name: "echo".to_string(),
        input: json!({"wrong": "shape"}),
    };
    let client = MockApiClient::new("local-model").with_responses(vec![
        MockResponse {
            text: String::new(),
            tool_call: Some(tool_call.clone()),
            stop_reason: "end_turn".to_string(),
        },
        MockResponse {
            text: String::new(),
            tool_call: Some(tool_call),
            stop_reason: "end_turn".to_string(),
        },
    ]);
    let err = loopctl::structured::request_structured_prompted::<RouterDecision>(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect_err("both attempts fail on the schema-invalid tool-call input");
    match err {
        StructuredError::Deserialize(source) => {
            let message = source.to_string();
            assert!(
                message.contains("Last output:") && message.contains("[tool call echo input:"),
                "the offending tool-call input rides the error instead of empty text: {message}"
            );
            assert!(
                message.contains("\"wrong\""),
                "the input value itself stays visible: {message}"
            );
        }
        other @ StructuredError::Api(_) => panic!("expected Deserialize, got {other:?}"),
    }
}

#[tokio::test]
async fn a_tool_call_failure_replays_as_a_text_only_answer() {
    let tool_call = MockToolCall {
        id: "call_retry".to_string(),
        name: "echo".to_string(),
        input: json!({"wrong": "shape"}),
    };
    let client = MockApiClient::new("local-model").with_responses(vec![
        MockResponse {
            text: String::new(),
            tool_call: Some(tool_call),
            stop_reason: "end_turn".to_string(),
        },
        response("{\"route\": \"local\", \"confidence\": 0.9}"),
    ]);
    let decision: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("the corrective retry recovers after a tool-call-shaped failure");
    assert_eq!(decision.route, "local");
    let captured = client.captured_requests();
    let replayed = &captured[1].messages[1];
    assert!(
        replayed.tool_call_parts().is_empty(),
        "the replayed answer is text-only, so the retry turn cannot trip a provider's tool-call validation"
    );
    assert!(
        replayed
            .text_content()
            .contains("[tool call echo input: {\"wrong\":\"shape\"}]"),
        "the replayed text renders the tool-call input the model actually produced"
    );
}

#[tokio::test]
async fn an_empty_first_answer_replays_as_a_non_empty_turn() {
    let client = MockApiClient::new("local-model").with_responses(vec![
        response(""),
        response("{\"route\": \"local\", \"confidence\": 0.9}"),
    ]);
    let decision: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("the corrective retry recovers an empty first answer");
    assert_eq!(decision.route, "local");
    let replayed = &client.captured_requests()[1].messages[1];
    assert_eq!(
        replayed.text_content(),
        "(empty reply)",
        "an empty rendering becomes a non-empty assistant turn a provider will accept"
    );
}

#[tokio::test]
async fn a_second_failure_returns_the_reason_and_bounded_last_output() {
    let oversized = format!("Still no JSON here. {}", "x".repeat(6_000));
    let client = MockApiClient::new("local-model")
        .with_responses(vec![response(&oversized), response(&oversized)]);
    let err = loopctl::structured::request_structured_prompted::<RouterDecision>(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect_err("both attempts fail");
    match err {
        StructuredError::Deserialize(source) => {
            let message = source.to_string();
            assert!(
                message.contains("the corrective retry also failed"),
                "the error says the retry ran: {message}"
            );
            assert!(
                message.contains("Last output:"),
                "the error carries the output"
            );
            assert!(
                message.contains("…[truncated]"),
                "an oversized output is truncated inside the error"
            );
            assert!(
                message.chars().count() < 3_000,
                "the embedded output is bounded, not unbounded"
            );
        }
        other @ StructuredError::Api(_) => panic!("expected Deserialize, got {other:?}"),
    }
}

#[tokio::test]
async fn the_prompted_wire_carries_no_tools_and_no_options() {
    let client = MockApiClient::new("local-model").with_responses(vec![response(
        "{\"route\": \"local\", \"confidence\": 0.9}",
    )]);
    let _: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect("succeeds");
    assert_eq!(
        client.with_options_calls(),
        0,
        "the prompted path never sends request options"
    );
    for request in client.captured_requests() {
        assert!(
            request.tools.is_none(),
            "the prompted path never sends a tools payload"
        );
        assert!(
            request
                .system
                .as_ref()
                .is_some_and(|system| system.contains("You will answer with JSON only.")),
            "every request carries the prompted envelope in the system slot"
        );
    }
}

#[tokio::test]
async fn a_caller_system_prompt_is_preceded_not_replaced() {
    let client = MockApiClient::new("local-model").with_responses(vec![response(
        "{\"route\": \"local\", \"confidence\": 0.9}",
    )]);
    let _: RouterDecision = loopctl::structured::request_structured_prompted(
        &client,
        vec![Message::user("route this")],
        Some("Be terse. You are a router.".to_string()),
    )
    .await
    .expect("succeeds");
    let captured = client.captured_requests();
    let system = captured[0]
        .system
        .as_ref()
        .expect("the composed system prompt is present");
    let user_position = system
        .find("Be terse. You are a router.")
        .expect("the caller's system prompt survives composition");
    let prefix_position = system
        .find("You will answer with JSON only.")
        .expect("the prompted envelope is present");
    assert!(
        user_position < prefix_position,
        "the caller's system prompt is prepended, not replaced"
    );
}

#[tokio::test]
async fn an_api_failure_fails_fast_without_a_retry() {
    let client = MockApiClient::new("local-model").with_error("boom");
    let err = loopctl::structured::request_structured_prompted::<RouterDecision>(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect_err("the provider failure surfaces as the api error");
    match err {
        StructuredError::Api(_) => {}
        other @ StructuredError::Deserialize(_) => panic!("expected Api, got {other:?}"),
    }
    assert_eq!(
        client.create_message_calls(),
        1,
        "an API failure never triggers the corrective retry"
    );
}

#[tokio::test]
async fn an_api_failure_on_the_retry_itself_surfaces_as_the_api_error() {
    let client = MockApiClient::new("local-model")
        .with_responses(vec![response("no json here")])
        .with_errors(vec![None, Some("retry request failed".to_string())]);
    let err = loopctl::structured::request_structured_prompted::<RouterDecision>(
        &client,
        vec![Message::user("route this")],
        None,
    )
    .await
    .expect_err("a retry that itself fails surfaces as the api error");
    match err {
        StructuredError::Api(_) => {}
        other @ StructuredError::Deserialize(_) => panic!("expected Api, got {other:?}"),
    }
    assert_eq!(
        client.create_message_calls(),
        2,
        "both attempts reached the provider before the failure"
    );
    assert_eq!(
        client.captured_requests().len(),
        2,
        "the recovery request reached the wire before the retry failed"
    );
}

/// A subscriber that captures every metric-targeted event as one flat
/// `target field=value …` string and every span with its attributes and
/// recorded fields, so the documented telemetry names are pinned rather
/// than trusted.
struct Capture {
    events: Mutex<Vec<String>>,

    /// Spans by id, as `name field=value …` with recorded fields
    /// appended.
    spans: Mutex<std::collections::HashMap<u64, String>>,

    /// Id source for [`new_span`](tracing::Subscriber::new_span).
    next_span_id: std::sync::atomic::AtomicU64,
}

impl Capture {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            spans: Mutex::new(std::collections::HashMap::new()),
            next_span_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn events_containing(&self, needle: &str) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.contains(needle))
            .count()
    }

    fn spans_containing(&self, needle: &str) -> usize {
        self.spans
            .lock()
            .unwrap()
            .values()
            .filter(|span| span.contains(needle))
            .count()
    }
}

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

fn capture_fields(record: impl FnOnce(&mut dyn tracing::field::Visit)) -> Vec<String> {
    let mut visitor = FieldVisitor { fields: Vec::new() };
    record(&mut visitor);
    visitor.fields
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _meta: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::Id {
        let id = self
            .next_span_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let fields = capture_fields(|visit| span.record(visit));
        let line = format!("name={} {}", span.metadata().name(), fields.join(" "));
        self.spans.lock().unwrap().insert(id, line);
        tracing::Id::from_u64(id)
    }

    fn record(&self, id: &tracing::Id, values: &tracing::span::Record<'_>) {
        let recorded = capture_fields(|visit| values.record(visit));
        if let Some(span) = self.spans.lock().unwrap().get_mut(&id.into_u64()) {
            for field in recorded {
                span.push(' ');
                span.push_str(&field);
            }
        }
    }

    fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let fields = capture_fields(|visit| event.record(visit));
        let line = format!("{} {}", event.metadata().target(), fields.join(" "));
        self.events.lock().unwrap().push(line);
    }

    fn enter(&self, _id: &tracing::Id) {}

    fn exit(&self, _id: &tracing::Id) {}
}

#[tokio::test]
async fn telemetry_names_and_outcomes_match_the_documented_contract() {
    let capture = std::sync::Arc::new(Capture::new());
    assert!(
        tracing::subscriber::set_global_default(std::sync::Arc::clone(&capture)).is_ok(),
        "this test owns the process-global subscriber"
    );

    let happy = MockApiClient::new("local-model").with_responses(vec![response(
        "{\"route\": \"local\", \"confidence\": 0.9}",
    )]);
    let _outcome: Result<RouterDecision, _> = loopctl::structured::request_structured_prompted(
        &happy,
        vec![Message::user("route this")],
        None,
    )
    .await;

    let retried = MockApiClient::new("local-model").with_responses(vec![
        response("no json here"),
        response("{\"route\": \"local\", \"confidence\": 0.9}"),
    ]);
    let _outcome: Result<RouterDecision, _> = loopctl::structured::request_structured_prompted(
        &retried,
        vec![Message::user("route this")],
        None,
    )
    .await;

    let failed = MockApiClient::new("local-model")
        .with_responses(vec![response("no json here"), response("still no json")]);
    let _outcome: Result<RouterDecision, _> = loopctl::structured::request_structured_prompted(
        &failed,
        vec![Message::user("route this")],
        None,
    )
    .await;

    let errored = MockApiClient::new("local-model").with_error("boom");
    let _outcome: Result<RouterDecision, _> = loopctl::structured::request_structured_prompted(
        &errored,
        vec![Message::user("route this")],
        None,
    )
    .await;

    let retry_errored = MockApiClient::new("local-model")
        .with_responses(vec![response("no json here")])
        .with_errors(vec![None, Some("boom".to_string())]);
    let _outcome: Result<RouterDecision, _> = loopctl::structured::request_structured_prompted(
        &retry_errored,
        vec![Message::user("route this")],
        None,
    )
    .await;

    assert!(
        capture.events_containing("metric=loopctl.structured.prompted outcome=first_try") >= 1,
        "a first-try success settles with the documented outcome"
    );
    assert!(
        capture.events_containing("metric=loopctl.structured.prompted outcome=after_retry") >= 1,
        "a recovered retry settles with the after_retry outcome"
    );
    assert!(
        capture.events_containing("metric=loopctl.structured.prompted outcome=failed") >= 1,
        "a double failure settles with the failed outcome"
    );
    assert!(
        capture.events_containing("metric=loopctl.structured.prompted outcome=api_error") >= 2,
        "an api failure settles with the api_error outcome, whether the first attempt or the retry failed"
    );
    assert!(
        capture.events_containing("metric=loopctl.structured.prompted.attempts_count value=1") >= 1,
        "the attempts counter carries single-attempt extractions"
    );
    assert!(
        capture.events_containing("metric=loopctl.structured.prompted.attempts_count value=2") >= 1,
        "the attempts counter carries retried extractions"
    );
    assert!(
        capture
            .events_containing("metric=loopctl.structured.prompted.tokens direction=in value=50")
            >= 1,
        "the in-token counter carries the usage figure"
    );
    assert!(
        capture
            .events_containing("metric=loopctl.structured.prompted.tokens direction=out value=25")
            >= 1,
        "the out-token counter carries the usage figure"
    );

    assert!(
        capture.spans_containing("name=structured.extract mode=prompted attempt=0") >= 1,
        "the first attempt's span carries the documented name, mode, and attempt"
    );
    assert!(
        capture.spans_containing("name=structured.extract mode=prompted attempt=1") >= 1,
        "the retry's span carries the documented name, mode, and attempt"
    );
    assert!(
        capture.spans_containing("gen_ai.operation.name=structured_extract") >= 1,
        "the OTel-convention operation field rides the span"
    );
    assert!(
        capture.spans_containing("ok=true") >= 1,
        "successful attempts record ok before the span closes"
    );
    assert!(
        capture.spans_containing("ok=false") >= 1,
        "failed attempts record ok before the span closes"
    );
}
