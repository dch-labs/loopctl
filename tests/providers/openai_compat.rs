//! Cassette replay for the OpenAI-compat client.
//!
//! Hermetic: the mock server serves the recorded Ollama exchange,
//! the client is built with a dummy key, and replay's byte-exact
//! matching is the assertion — if the request the client builds today
//! differs by one byte from what the real server accepted when the
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

#[path = "../cassette.rs"]
mod cassette;

use cassette::{CassetteSession, client_base_url, get_weather_tool};
use futures::StreamExt;
use loopctl::api::ApiClient;

fn scenario(name: &str) -> cassette::Scenario {
    cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai-compat" && s.name == name)
        .unwrap_or_else(|| panic!("unknown openai-compat scenario {name}"))
}

async fn replay(name: &str) -> (cassette::Scenario, Vec<loopctl::stream::StreamEvent>) {
    let scenario = scenario(name);
    let server = httpmock::MockServer::start_async().await;
    let session = CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(client_base_url(scenario.provider, &session.base_url()))
        .with_model(scenario.model)
        .with_stream_usage(scenario.stream_usage)
        .build()
        .unwrap();
    let mut request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    if scenario.tools {
        request = request.with_tools(Some(vec![get_weather_tool()]));
    }
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        events.push(result.expect("replayed events arrive as recorded"));
    }
    session.finish().await;
    (scenario, events)
}

fn text_of(events: &[loopctl::stream::StreamEvent]) -> String {
    let mut text = String::new();
    for event in events {
        if let loopctl::stream::StreamEvent::IndexedDelta(delta) = event
            && let loopctl::stream::DeltaPart::Text { text: chunk } = &delta.delta
        {
            text.push_str(chunk);
        }
    }
    text
}

#[tokio::test]
async fn minimal_text_replays_from_cassette() {
    let (_, events) = replay("minimal_text").await;
    assert_eq!(
        text_of(&events),
        "Hello there!",
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
        "the recorded usage-on-final-chunk must reach the terminal MessageDelta"
    );
}

#[tokio::test]
async fn tool_call_lifecycle_replays_from_cassette() {
    let (_, events) = replay("tool_call_lifecycle").await;

    let mut tool_input = String::new();
    let mut saw_tool_delta = false;
    for event in &events {
        if let loopctl::stream::StreamEvent::IndexedDelta(delta) = event {
            match &delta.delta {
                loopctl::stream::DeltaPart::ToolCall { partial_json } => {
                    saw_tool_delta = true;
                    tool_input.push_str(&partial_json.to_string());
                }
                loopctl::stream::DeltaPart::InputJson { partial_json } => {
                    saw_tool_delta = true;
                    tool_input.push_str(partial_json);
                }
                _ => {}
            }
        }
    }
    assert!(
        saw_tool_delta,
        "the recorded tool-call lifecycle must surface tool deltas to the consumer"
    );
    let parsed: serde_json::Value = serde_json::from_str(&tool_input)
        .unwrap_or_else(|e| panic!("the concatenated tool input must parse: {e} ({tool_input})"));
    assert_eq!(
        parsed.get("city").and_then(serde_json::Value::as_str),
        Some("Paris"),
        "the accumulated tool arguments must reconstruct the recorded call"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop)),
        "the recorded stream ends with MessageStop after the tool call"
    );

    let tool_part_start = events
        .iter()
        .position(|event| match event {
            loopctl::stream::StreamEvent::PartStart(start) => {
                start.index == 0
                    && matches!(
                        &start.part,
                        Some(loopctl::message::MessagePart::ToolCall { id, name, .. })
                            if id == "call_cassette_2" && name == "get_weather"
                    )
            }
            _ => false,
        })
        .expect("the tool lane opens with a PartStart carrying the recorded call identity");
    let tool_delta = events
        .iter()
        .position(|event| match event {
            loopctl::stream::StreamEvent::IndexedDelta(delta) => {
                delta.index == 0
                    && matches!(
                        delta.delta,
                        loopctl::stream::DeltaPart::ToolCall { .. }
                            | loopctl::stream::DeltaPart::InputJson { .. }
                    )
            }
            _ => false,
        })
        .expect("the tool lane carries its argument delta on the same index");
    let tool_part_stop = events
        .iter()
        .position(|event| {
            matches!(
                event,
                loopctl::stream::StreamEvent::PartStop { index: Some(0) }
            )
        })
        .expect("the tool lane closes with a PartStop on the same index");
    assert!(
        tool_part_start < tool_delta && tool_delta < tool_part_stop,
        "the tool lane is a PartStart → IndexedDelta → PartStop sequence on \
        lane index 0 — a client that drops either boundary event fails here"
    );
}
