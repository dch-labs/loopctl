//! Cassette replay suites, one module per provider.
//!
//! Each suite replays that provider's committed cassettes hermetically
//! — dummy keys, no network — and the byte-exact matching is the
//! outbound-drift guard. The helpers here hold the parts every suite
//! shares; client construction stays per suite, behind its feature.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

mod cassette;

#[path = "providers/anthropic.rs"]
mod anthropic;

#[path = "providers/deepseek.rs"]
mod deepseek;

#[path = "providers/gemini.rs"]
mod gemini;

#[path = "providers/grok.rs"]
mod grok;

#[path = "providers/openai.rs"]
mod openai;

#[path = "providers/openai_compat.rs"]
mod openai_compat;

#[path = "providers/zai.rs"]
mod zai;

use cassette::Scenario;
use futures::StreamExt;
use loopctl::api::ApiClient;

/// Drive one scenario's request against a client and collect the
/// event stream to its end.
///
/// Builds the request from the scenario's fixed prompt (plus the
/// shared weather tool for tool scenarios), polls the stream to
/// completion, and fails on any errored event — the caller asserts
/// on the collected shape.
pub(crate) async fn collect_events(
    client: &dyn ApiClient,
    scenario: &Scenario,
) -> Vec<loopctl::stream::StreamEvent> {
    let mut request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    if scenario.tools {
        request = request.with_tools(Some(vec![cassette::get_weather_tool()]));
    }
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        events.push(result.expect("replayed events arrive as recorded"));
    }
    events
}

/// The concatenated text deltas of an event stream.
///
/// The replay suites' exact-text assertions build on this: the sum of
/// the text lane's deltas is what a consumer would render.
pub(crate) fn text_of(events: &[loopctl::stream::StreamEvent]) -> String {
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

/// The usage carried by the terminal `MessageDelta`, if any.
///
/// Every provider reports usage differently (final chunk, latch from
/// message_start); this reads the engine's normalized terminal value
/// so suites assert one shape.
pub(crate) fn usage_of(events: &[loopctl::stream::StreamEvent]) -> Option<loopctl::stream::Usage> {
    events.iter().find_map(|event| match event {
        loopctl::stream::StreamEvent::MessageDelta(md) => md.usage,
        _ => None,
    })
}
