//! Cassette replay for the OpenAI client against the real OpenAI wire.
//!
//! Hermetic: the mock server serves the recorded cloud exchange, the
//! client is built with a dummy key, and replay's byte-exact matching
//! is the assertion — if the request the client builds today differs
//! by one byte from what the real OpenAI API accepted when the
//! cassette was recorded, the request misses and this test fails.

#![cfg(feature = "openai")]
#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use crate::cassette;
use crate::{collect_events, text_of, usage_of};

fn scenario(name: &str) -> cassette::Scenario {
    cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai" && s.name == name)
        .unwrap_or_else(|| panic!("unknown openai scenario {name}"))
}

async fn replay(name: &str) -> Vec<loopctl::stream::StreamEvent> {
    let scenario = scenario(name);
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model(scenario.model)
        .with_stream_usage(scenario.stream_usage)
        .build()
        .unwrap();
    let events = collect_events(&client, &scenario).await;
    session.finish().await;
    events
}

#[tokio::test]
async fn minimal_text_replays_from_cassette() {
    let events = replay("minimal_text").await;
    assert_eq!(
        text_of(&events),
        "Hello there, friend!",
        "the accumulated text must reconstruct the recorded response exactly"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop)),
        "the recorded stream ends with MessageStop"
    );
    assert!(
        usage_of(&events).is_some_and(|u| u.input_tokens > 0 && u.output_tokens > 0),
        "the recorded include_usage final chunk must reach the terminal MessageDelta"
    );
}

#[tokio::test]
async fn tool_call_lifecycle_replays_from_cassette() {
    let events = replay("tool_call_lifecycle").await;

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
        "the real-OpenAI fragmented tool-argument stream must accumulate \
        into the recorded call — the lifecycle Ollama emits as one delta"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop)),
        "the recorded stream ends with MessageStop after the tool call"
    );
}
