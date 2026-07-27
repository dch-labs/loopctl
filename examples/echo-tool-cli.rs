//! BareLoop CLI with a custom tool — multi-turn tool dispatch.
//!
//! Registers an `echo` tool, configures the mock client to request it
//! on the first turn, then prints the final response.
//!
//! ```sh
//! cargo run --example echo-tool-cli --features testing
//! ```

#![allow(clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::RunConfig;
use loopctl::engine::core::Loop;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{FnTool, ToolOutput, ToolRegistry};
use serde_json::json;

/// The async function backing the `echo` tool.
fn echo_fn(
    input: serde_json::Value,
    _ctx: &loopctl::tool::ToolContext,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<ToolOutput, loopctl::tool::ToolError>>
            + Send
            + 'static,
    >,
> {
    Box::pin(async move {
        let text = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)");
        Ok(ToolOutput::text(format!("echo: {text}")))
    })
}

#[tokio::main]
async fn main() {
    // 1. Create a mock client that requests the echo tool on turn 1,
    //    then gives a final text response on turn 2.
    let client = MockApiClient::new("echo-model").with_responses(vec![
        MockResponse {
            text: String::new(),
            tool_call: Some(MockToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: json!({"message": "Hello from the model!"}),
            }),
            stop_reason: "tool_use".into(),
        },
        MockResponse {
            text: "I echoed your message. Done!".into(),
            tool_call: None,
            stop_reason: "end_turn".into(),
        },
    ]);

    // 2. Build the tool registry with a single `echo` tool.
    let mut tools = ToolRegistry::new();
    tools.register(
        FnTool::new(
            "echo".into(),
            "Echo back the provided message.".into(),
            json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The text to echo"}
                },
                "required": ["message"]
            }),
            echo_fn,
        )
        .read_only(),
    );

    // 3. Construct and run the loop.
    let mut agent = BareLoop::new(Arc::new(client), tools, SessionConfig::default());

    // Ctrl-C interrupts the in-flight turn via loopctl's CancelSignal.
    let cancel_signal = agent.cancel_signal();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_signal.cancel();
        }
    });

    let result = agent
        .run("Please echo something.", &RunConfig::default())
        .await
        .expect("session should succeed");

    println!("Turns:      {}", result.turn_count());
    println!("Tool calls: {}", result.tool_call_count());
    println!("Output:     {}", result.output.unwrap_or_default());
}
