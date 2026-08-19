//! Serve a loopctl `ToolRegistry` over MCP stdio — the end-to-end smoke test.
//!
//! Builds a registry with an echo tool and a failing tool (so both the success
//! and the error path are reachable), wraps it in `McpServerAdapter`, and
//! serves over stdio. Run it directly:
//!
//! ```sh
//! cargo run --example mcp_server --features mcp
//! ```
//!
//! To use it from an MCP client, register the built binary with the client as
//! a stdio server (each client documents its own configuration). The piped
//! handshake below exercises the same code path with no client at all.
//!
//! The process serves until the client closes the pipe (stdin EOF) or Ctrl-C
//! cancels the service — the canonical shutdown shape for a binary embedding
//! the adapter.
//!
//! # Acceptance check without an MCP client
//!
//! Pipe the MCP handshake and a few requests into the binary and read the
//! JSON-RPC responses off stdout:
//!
//! ```sh
//! printf '%s\n' \
//!   '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
//!   '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
//!   '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
//!   '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hi"}}}' \
//!   '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fail"}}' \
//!   | cargo run --example mcp_server --features mcp
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

/// A tool that returns its `message` argument unchanged.
///
/// The success-path fixture of this smoke test: it round-trips a string
/// through `Tool::call` → MCP `tools/call` → back to the client. Input
/// whose `message` is missing or not a string is rejected with
/// [`ToolError::InvalidInput`], which the adapter surfaces as a tool-level
/// error result the caller can read.
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
        Box::pin(async move {
            let Some(message) = input.get("message").and_then(serde_json::Value::as_str) else {
                return Err(ToolError::InvalidInput(
                    "expected a string field `message`".into(),
                ));
            };
            Ok(ToolOutput::text(message.to_string()))
        })
    }
}

/// A tool that always fails with a hard execution error.
///
/// The error-path fixture: it proves a failing tool surfaces as a
/// tool-level `is_error: true` result the caller can read, not a protocol
/// error that would tear down the session.
struct FailTool;

impl Tool for FailTool {
    fn name(&self) -> &'static str {
        "fail"
    }
    fn description(&self) -> &'static str {
        "Always fails with a hard tool error"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "fail".into(),
            description: "Always fails with a hard tool error".into(),
            input_schema: json!({"type": "object"}),
        }
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Err(ToolError::Execution("this tool always fails".into())) })
    }
}

#[tokio::main]
async fn main() {
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    registry.register(FailTool);
    let adapter = McpServerAdapter::new(
        registry,
        ToolContext::default(),
        "loopctl-example".into(),
        env!("CARGO_PKG_VERSION").into(),
    );
    let service = adapter.serve_stdio().await.expect("stdio serve");
    let cancel = service.cancellation_token();
    tokio::select! {
        quit = service.waiting() => {
            quit.expect("server loop");
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("shutting down");
            cancel.cancel();
        }
    }
}
