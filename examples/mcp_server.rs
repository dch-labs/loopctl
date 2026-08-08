//! Serve a loopctl `ToolRegistry` over MCP stdio — the MCP client smoke test.
//!
//! Builds a registry with an echo tool, wraps it in `McpServerAdapter`, and
//! serves over stdio. Point MCP client at this binary:
//!
//! ```json,ignore
//! { "mcpServers": { "loopctl": { "command": "/path/to/mcp_server" } } }
//! ```
//!
//! ```sh
//! cargo run --example mcp_server --features mcp
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    dead_code
)]

use std::future::Future;
use std::pin::Pin;

use loopctl::mcp::McpServerAdapter;
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use serde_json::json;

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Echo back the message field"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "echo".into(),
            description: "Echo back the message field".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "message": { "type": "string" } },
                "required": ["message"]
            }),
        }
    }
    fn call(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let msg = input
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        Box::pin(async move { Ok(ToolOutput::text(msg)) })
    }
}

#[tokio::main]
async fn main() {
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let adapter = McpServerAdapter::new(
        registry,
        ToolContext::default(),
        "loopctl-example".into(),
        env!("CARGO_PKG_VERSION").into(),
    );
    let service = adapter.serve_stdio().await.expect("stdio serve");
    service.waiting().await.expect("waiting");
}
