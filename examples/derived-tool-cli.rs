//! A minimal agent loop with one derived tool.
//!
//! The side-by-side with `echo-tool-cli.rs` is the derive's pitch: the
//! struct below replaces the hand-written `name`/`description`/`schema`
//! and the `Value` extraction.
//!
//! Run: `cargo run --features derive,testing --example derived-tool-cli`

#![allow(clippy::expect_used, clippy::pedantic)]

use std::sync::Arc;

use loopctl::Tool;
use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::testing::MockApiClient;
use loopctl::tool::ToolRegistry;
use serde::Deserialize;

/// Echo a message back to the caller.
#[derive(Tool, Deserialize)]
#[tool(name = "echo", description = "Echo back the provided message")]
struct EchoInput {
    /// The text to echo.
    message: String,
}

impl EchoInput {
    async fn run(
        &self,
        input: EchoInput,
        _ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        Ok(loopctl::tool::ToolOutput::text(input.message))
    }
}

#[tokio::main]
async fn main() {
    let client = MockApiClient::new("echo-model").with_responses(vec![
        loopctl::testing::MockResponse {
            text: "one go".to_string(),
            tool_call: Some(loopctl::testing::MockToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "derived!"}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        loopctl::testing::MockResponse {
            text: "all done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(EchoInput {
        message: String::new(),
    });
    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    let outcome = agent.run("say it", &RunConfig::default()).await;
    println!("{outcome:?}");
}
