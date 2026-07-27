//! Minimal BareLoop CLI — no tools, single-turn.
//!
//! Demonstrates the absolute simplest way to run a [`BareLoop`]:
//! create a mock client, build the loop, and call [`run`].

#![allow(clippy::expect_used, clippy::doc_markdown)]
//!
//! ```sh
//! cargo run --example hello-cli --features testing
//! ```
//!
//! [`run`]: loopctl::engine::bare::BareLoop::run

use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::RunConfig;
use loopctl::engine::core::Loop;
use loopctl::testing::MockApiClient;
use loopctl::tool::ToolRegistry;

#[tokio::main]
async fn main() {
    // 1. Create a mock API client with a canned response.
    let client = MockApiClient::new("hello-model").with_text_response("Hello, world!");

    // 2. Build the components.
    let tools = ToolRegistry::new();
    let config = SessionConfig::default();

    // 3. Construct the loop.
    let mut agent = BareLoop::new(Arc::new(client), tools, config);

    // Ctrl-C interrupts the in-flight turn via loopctl's CancelSignal.
    let cancel_signal = agent.cancel_signal();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_signal.cancel();
        }
    });

    // 4. Run and print the result.
    let result = agent
        .run("Say hello!", &RunConfig::default())
        .await
        .expect("session should succeed");

    println!("Turns:  {}", result.turn_count());
    println!("Output: {}", result.output.unwrap_or_default());
}
