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
use futures::StreamExt;
use loopctl::api::ApiClient;

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

#[tokio::test]
#[ignore = "the cassette lands when a recording attempt actually meets a 429"]
async fn rate_limit_error_replays_from_cassette() {
    let scenario = scenario("rate_limit_error");
    let expected_retry_after = cassette::load_interactions(scenario.provider, scenario.name)
        .first()
        .and_then(|interaction| interaction.then.header.as_ref())
        .and_then(|headers| {
            headers
                .iter()
                .find(|pair| pair.name == "retry-after")
                .map(|pair| pair.value.clone())
        })
        .and_then(|value| value.parse::<u64>().ok())
        .map(std::time::Duration::from_secs);

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
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut rate_limit = None;
    while let Some(result) = stream.next().await {
        match result {
            Err(loopctl::api::error::ApiError::RateLimit { retry_after, .. }) => {
                rate_limit = Some(retry_after);
            }
            other => panic!(
                "a replayed 429 must surface as the structured rate-limit error, got {other:?}"
            ),
        }
    }
    session.finish().await;

    assert!(
        rate_limit.is_some(),
        "the replayed rate-limit exchange must reach the caller as the structured error"
    );
    assert_eq!(
        rate_limit.flatten(),
        expected_retry_after,
        "the recorded Retry-After must reach the caller parsed"
    );
}

#[tokio::test]
async fn model_override_replays_from_cassette() {
    let scenario = scenario("model_override");
    // The recorded body must carry the override model, not the client
    // default — computed from the cassette, never hardcoded.
    let expected_model = cassette::load_interactions(scenario.provider, scenario.name)
        .first()
        .and_then(|interaction| interaction.when.body.as_ref())
        .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
        .and_then(|body| {
            body.get("model")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .expect("the cassette holds a JSON body with a model field");
    assert_ne!(
        expected_model, "gpt-4o-mini",
        "the override, not the client default, is what was recorded"
    );

    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model("gpt-4o-mini")
        .with_stream_usage(scenario.stream_usage)
        .build()
        .unwrap();
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let options = loopctl::structured::RequestOptions::new().with_model(&expected_model);
    let stream = client.stream_messages_with_options(&request, options);
    let mut stream = std::pin::pin!(stream);
    while let Some(result) = stream.next().await {
        result.expect("replayed events arrive as recorded");
    }
    session.finish().await;
}
