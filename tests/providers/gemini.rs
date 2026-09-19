//! Cassette replay for the Gemini client.
//!
//! Hermetic: the mock server serves the recorded cloud exchange, the
//! client is built with a dummy key, and replay's byte-exact matching
//! is the assertion — if the request the client builds today differs
//! by one byte from what the real Gemini API accepted when the
//! cassette was recorded (path version, query, body shape), the
//! request misses and this test fails.

#![cfg(feature = "gemini")]
#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use crate::cassette;
use crate::cassette::{CassetteSession, Scenario, client_base_url};
use crate::collect_events;
use futures::StreamExt;
use loopctl::api::ApiClient;

fn scenario() -> Scenario {
    cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "gemini" && s.name == "minimal_text")
        .expect("the gemini minimal_text scenario exists")
}

#[tokio::test]
async fn minimal_text_replays_from_cassette() {
    let scenario = scenario();
    let server = httpmock::MockServer::start_async().await;
    let session = CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(client_base_url(scenario.provider, &session.base_url()))
        .with_model(scenario.model)
        .build()
        .unwrap();
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        events.push(result.expect("replayed events arrive as recorded"));
    }
    session.finish().await;

    let mut text = String::new();
    for event in &events {
        if let loopctl::stream::StreamEvent::IndexedDelta(delta) = event
            && let loopctl::stream::DeltaPart::Text { text: chunk } = &delta.delta
        {
            text.push_str(chunk);
        }
    }
    assert_eq!(
        text, "Hello to you.",
        "the accumulated text must reconstruct the recorded response exactly"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop)),
        "the recorded stream ends with MessageStop"
    );
    let usage = events.iter().find_map(|e| match e {
        loopctl::stream::StreamEvent::MessageDelta(md) => md.usage,
        _ => None,
    });
    assert!(
        usage.is_some_and(|u| u.input_tokens > 0 && u.output_tokens > 0),
        "the recorded usageMetadata must reach the terminal MessageDelta"
    );
}

#[tokio::test]
async fn tool_call_lifecycle_replays_from_cassette() {
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "gemini" && s.name == "tool_call_lifecycle")
        .expect("the gemini tool_call_lifecycle scenario exists");
    let server = httpmock::MockServer::start_async().await;
    let session = CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(client_base_url(scenario.provider, &session.base_url()))
        .with_model(scenario.model)
        .build()
        .unwrap();
    let events = collect_events(&client, &scenario).await;
    session.finish().await;

    let mut tool_input = String::new();
    for event in &events {
        if let loopctl::stream::StreamEvent::IndexedDelta(delta) = event {
            match &delta.delta {
                loopctl::stream::DeltaPart::ToolCall { partial_json } => {
                    tool_input.push_str(&partial_json.to_string());
                }
                loopctl::stream::DeltaPart::InputJson { partial_json } => {
                    tool_input.push_str(partial_json);
                }
                _ => {}
            }
        }
    }
    let parsed: serde_json::Value = serde_json::from_str(&tool_input)
        .unwrap_or_else(|e| panic!("the concatenated tool input must parse: {e} ({tool_input})"));
    assert_eq!(
        parsed.get("city").and_then(serde_json::Value::as_str),
        Some("Paris"),
        "the Gemini functionCall stream must accumulate into the \
        recorded call — the third tool-calling dialect in the corpus"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop)),
        "the recorded stream ends with MessageStop after the tool call"
    );
}
