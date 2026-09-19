//! Cassette replay for the Z.ai client.
//!
//! Hermetic: the mock server serves the recorded cloud exchange, the
//! client is built with a dummy key, and replay's byte-exact matching
//! is the assertion — if the request the client builds today differs
//! by one byte from what the real Z.ai API accepted when the
//! cassette was recorded, the request misses and this test fails.

#![cfg(feature = "zai")]
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

#[tokio::test]
async fn minimal_text_replays_from_cassette() {
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "zai" && s.name == "minimal_text")
        .expect("the zai minimal_text scenario exists");
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, "", &server).await;
    let client = loopctl::provider::AnthropicClient::builder()
        .with_api_key(cassette::CASSETTE_KEY)
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model(scenario.model)
        .build()
        .unwrap();
    let events = collect_events(&client, &scenario).await;
    session.finish().await;

    assert_eq!(
        text_of(&events),
        "Hello to you!",
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
        "the recorded usage must reach the terminal MessageDelta"
    );
}
