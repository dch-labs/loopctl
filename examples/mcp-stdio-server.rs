//! A minimal MCP server over stdio — the subprocess target for the L-13 stdio
//! transport tests.
//!
//! Built with rmcp's `#[tool_router]` / `#[tool]` macros, exposing one `greet`
//! tool. Runs the server on stdin/stdout (the MCP stdio transport); stderr is
//! inherited so server logs surface during tests. Exits when the client
//! disconnects.
//!
//! ```sh
//! cargo run --example mcp-stdio-server --features mcp
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    dead_code
)]

use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::{tool, tool_handler, tool_router};

/// A server exposing one `greet` tool.
#[derive(Clone)]
struct StdioServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl StdioServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Return a friendly greeting")]
    async fn greet(&self) -> String {
        "hello from stdio".to_string()
    }
}

#[tool_handler]
impl ServerHandler for StdioServer {}

#[tokio::main]
async fn main() {
    let server = StdioServer::new();
    let transport = rmcp::transport::io::stdio();
    let running = server
        .serve(transport)
        .await
        .expect("stdio server initialize");
    running.waiting().await.ok();
}
