//! Minimal BareLoop CLI — no tools, single-turn.
//!
//! Demonstrates the absolute simplest way to run a [`BareLoop`]:
//! create a mock client, build the loop, and call [`run`].
//!
//! ```sh
//! cargo run --example hello-cli --features testing
//! ```
//!
//! [`run`]: loopctl::engine::bare::BareLoop::run

use std::sync::Arc;

use loopctl::config::LoopConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::loop_core::Loop;
use loopctl::testing::MockApiClient;
use loopctl::tool::ToolRegistry;

#[tokio::main]
async fn main() {
    // 1. Create a mock API client with a canned response.
    let client = MockApiClient::new("hello-model").with_text_response("Hello, world!");

    // 2. Build the components.
    let tools = ToolRegistry::new();
    let config = LoopConfig::default();

    // 3. Construct the loop.
    let mut agent = BareLoop::new(Arc::new(client), tools, config);

    // 4. Run and print the result.
    let result = agent
        .run("Say hello!")
        .await
        .expect("session should succeed");

    println!("Turns:  {}", result.total_turns);
    println!("Output: {}", result.final_output.unwrap_or_default());
}
