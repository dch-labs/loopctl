//! Adapt an in-process MCP server's tools as loopctl `Tool` impls.
//!
//! Builds a tiny rmcp server (with the `#[tool_router]` / `#[tool]` macros),
//! connects an [`McpToolProvider`] to it over a `tokio::io::duplex`, discovers
//! its tools, registers them into a [`ToolRegistry`], and calls one to prove
//! the round-trip works end-to-end.
//!
//! ```sh
//! cargo run --example mcp-adapter --features mcp
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    dead_code
)]

use loopctl::mcp::{McpClient, McpToolProvider};
use loopctl::tool::{ToolContext, ToolRegistry};
use rmcp::handler::server::ServerHandler;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::{ServiceExt, tool, tool_handler, tool_router};
use serde_json::json;

/// An rmcp server exposing one `greet` tool.
#[derive(Clone)]
struct GreetServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl GreetServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Return a friendly greeting")]
    async fn greet(&self) -> String {
        "hello, world!".to_string()
    }
}

#[tool_handler]
impl ServerHandler for GreetServer {}

#[tokio::main]
async fn main() {
    // 1. Start the server on one end of a duplex, a pure rmcp client on the
    //    other, and wrap the client as an McpClient.
    let (server_end, client_end) = tokio::io::duplex(4096);
    tokio::spawn(async move {
        let running = GreetServer::new()
            .serve(server_end)
            .await
            .expect("server initialize");
        running.waiting().await.ok();
    });
    let client =
        ().serve(client_end)
            .await
            .map(McpClient::from_service)
            .expect("client initialize");

    // 2. Discover the server's tools and register them.
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect + list_tools");
    let mut registry = ToolRegistry::new();
    provider.register_into(&mut registry);
    println!("discovered {} tool(s):", registry.len());
    for schema in registry.all_schemas() {
        println!("  - {} : {}", schema.tool, schema.description);
    }

    // 3. Call the adapted `greet` tool through the registry, exactly as the
    //    agent loop would call any native loopctl tool.
    let greet = registry.get("greet").expect("greet registered");
    let ctx = ToolContext::default();
    let out = greet.call(json!({}), &ctx).await.expect("greet call");
    println!("greet -> {}", out.text_content());
    assert!(!out.is_error);
    assert_eq!(out.text_content(), "hello, world!");
    println!("OK");
}
