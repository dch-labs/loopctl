//! MCP server adapter — serve a loopctl `ToolRegistry` over MCP.
//!
//! [`McpServerAdapter`] wraps a `ToolRegistry` and
//! implements rmcp's [`ServerHandler`] so that any MCP-speaking client (Claude
//! Desktop being the named acceptance target) can discover and invoke the
//! registry's tools as if they were native MCP tools.
//!
//! # Transports
//!
//! The adapter's own API never names a transport. [`McpServerAdapter::serve_stdio`]
//! is a convenience for the canonical MCP-client case (stdin/stdout JSON-RPC);
//! a host that wants another transport (post-L-13) calls
//! `adapter.clone().serve(transport)` directly. This makes L-13's HTTP/SSE
//! client transports a pure-additive follow-up — no adapter changes needed.
//!
//! # Non-goals
//!
//! - **No resources / prompts / sampling / logging / completions / tasks** —
//!   only the MCP **tools** capability is advertised. loopctl has no model for
//!   the other primitives.
//! - **No loopctl `Hook` / permission integration** — `call_tool` dispatches
//!   through [`Tool::call`](crate::tool::Tool::call) directly. A future task
//!   could wrap the dispatch in loopctl's `HookExecutor`.
//! - **No per-chunk streaming** — MCP delivers a tool result as one
//!   `CallToolResult`.
//! - **No live tool-list mutation** — the adapter serves a snapshot; rebuild
//!   for a different tool set.
//!
//! # MCP client
//!
//! Build the example (`cargo build --example mcp_server --features mcp`) and add
//! it to `MCP client config`:
//!
//! ```json,ignore
//! {
//!   "mcpServers": {
//!     "loopctl": {
//!       "command": "/path/to/loopctl-example-mcp-server"
//!     }
//!   }
//! }
//! ```

use std::sync::Arc;

use rmcp::ErrorData as McpError;
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::CallToolRequestParams;
use rmcp::model::PaginatedRequestParams;
use rmcp::model::ServerInfo;
use rmcp::model::ToolsCapability;
use rmcp::service::RequestContext;
use rmcp::service::RoleServer;

use crate::mcp::convert;
use crate::tool::ToolContext;
use crate::tool::ToolRegistry;

/// An MCP server backed by a loopctl [`ToolRegistry`].
///
/// Wraps a registry (and the [`ToolContext`] passed to every dispatch) and
/// exposes it over MCP via the [`ServerHandler`] trait: `list_tools` returns
/// every registered tool's schema, `call_tool` dispatches to
/// [`Tool::call`](crate::tool::Tool::call).
///
/// Construct with [`McpServerAdapter::new`], then drive it with
/// [`serve_stdio`](Self::serve_stdio) (or `ServiceExt::serve` with any transport).
/// The adapter is `Clone` (it holds only `Arc`s) because `ServerHandler` requires
/// `Clone + Send + Sync + 'static`.
///
/// # What it advertises
///
/// Only the MCP **tools** capability. Resources, prompts, sampling, logging,
/// completions, and subscriptions are not implemented — loopctl has no model
/// for them — and their `ServerHandler` defaults (all no-ops / `None`) apply.
#[derive(Clone)]
pub struct McpServerAdapter {
    /// The registry being served.
    ///
    /// Shared immutably across handler clones (the agent-loop usage pattern:
    /// built once, dispatched read-only). Every `list_tools` call reads
    /// `all_schemas()`; every `call_tool` reads `get(name)`. No mutation after
    /// construction.
    registry: Arc<ToolRegistry>,
    /// The per-session context handed to every [`Tool::call`](crate::tool::Tool::call).
    ///
    /// `ToolContext: Clone`; held in `Arc` so every clone of the adapter shares
    /// one context — same `session_id`, `cwd`, extensions. One MCP server = one
    /// logical session.
    context: Arc<ToolContext>,
    /// Server name reported in `initialize` → `ServerInfo`.
    ///
    /// Populates the `name` field of the rmcp `Implementation` struct. What the
    /// client shows as the server identity (e.g. in MCP client's server list).
    server_name: String,
    /// Server version reported in `initialize` → `ServerInfo`.
    ///
    /// Populates the `version` field of the rmcp `Implementation` struct.
    server_version: String,
}

impl McpServerAdapter {
    /// Build an adapter wrapping `registry`, dispatching every tool call with `context`.
    ///
    /// `server_name` / `server_version` populate the MCP `ServerInfo` returned
    /// during the `initialize` handshake (what the client shows as the server
    /// identity, e.g. in MCP client's server list). Use your application's
    /// name and version.
    ///
    /// The registry and context are moved into `Arc`s and shared by every clone
    /// of the adapter. The registry is **not** mutated after construction —
    /// rebuild the adapter if you need a different tool set.
    #[must_use]
    pub fn new(
        registry: ToolRegistry,
        context: ToolContext,
        server_name: String,
        server_version: String,
    ) -> Self {
        Self {
            registry: Arc::new(registry),
            context: Arc::new(context),
            server_name,
            server_version,
        }
    }

    /// Serve over stdio and return the running service handle.
    ///
    /// Convenience for the canonical MCP-client case: stdin/stdout JSON-RPC.
    /// Equivalent to `self.clone().serve(rmcp::transport::io::stdio()).await`.
    /// The returned service has completed the `initialize`/`initialized`
    /// handshake; call `.waiting()` to block until the client disconnects
    /// (stdin EOF) or `.cancel()` to shut down proactively.
    ///
    /// # Errors
    ///
    /// Returns `rmcp::service::ServerInitializeError` if the handshake fails
    /// (e.g. the client sends a malformed `initialize`).
    pub async fn serve_stdio(
        self,
    ) -> Result<rmcp::service::RunningService<RoleServer, Self>, rmcp::service::ServerInitializeError>
    {
        let transport = rmcp::transport::io::stdio();
        self.serve(transport).await
    }
}

impl ServerHandler for McpServerAdapter {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities.tools = Some(ToolsCapability::default());
        info.server_info.name.clone_from(&self.server_name);
        info.server_info.version.clone_from(&self.server_version);
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, McpError> {
        let tools = self
            .registry
            .all_schemas()
            .into_iter()
            .map(convert::tool_schema_to_mcp)
            .collect();
        Ok(rmcp::model::ListToolsResult {
            tools,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        let name = request.name.to_string();
        let Some(tool) = self.registry.get(&name) else {
            let names = self.registry.tool_names();
            let available: Vec<&str> = names.iter().map(String::as_str).collect();
            return Err(convert::not_found_error(&name, &available));
        };
        let input = request
            .arguments
            .map_or(serde_json::Value::Null, serde_json::Value::Object);
        let result = tool.call(input, &self.context).await;
        Ok(convert::dispatch_result_to_call_tool(&name, result).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echo"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "echo".into(),
                description: "Echo".into(),
                input_schema: json!({"type": "object"}),
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

    struct FailTool;

    impl Tool for FailTool {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn description(&self) -> &'static str {
            "Always fails"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "fail".into(),
                description: "Always fails".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Err(ToolError::Execution("always fails".into())) })
        }
    }

    #[test]
    fn get_info_advertises_tools_only() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let info = adapter.get_info();
        assert!(info.capabilities.tools.is_some());
        assert!(info.capabilities.prompts.is_none());
        assert!(info.capabilities.resources.is_none());
        assert_eq!(info.server_info.name, "test");
        assert_eq!(info.server_info.version, "0.0.1");
    }

    #[test]
    fn adapter_is_clone_send_sync() {
        fn _assert_clone_send_sync<T: Clone + Send + Sync>() {}
        _assert_clone_send_sync::<McpServerAdapter>();
    }

    async fn discover_via_client(adapter: McpServerAdapter) -> Vec<String> {
        let (server_end, client_end) = tokio::io::duplex(4096);
        let cloned = adapter.clone();
        tokio::spawn(async move {
            let running = cloned.serve(server_end).await.expect("server serve");
            let _ = running.waiting().await.ok();
        });
        let client =
            ().serve(client_end)
                .await
                .map(crate::mcp::McpClient::from_service)
                .expect("client connect");
        let tools = client
            .service
            .list_all_tools()
            .await
            .expect("list_all_tools");
        drop(client);
        tools.into_iter().map(|t| t.name.to_string()).collect()
    }

    #[tokio::test]
    async fn list_tools_serves_registry_via_mcp() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(FailTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let names = discover_via_client(adapter).await;
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"fail".to_string()));
    }

    #[tokio::test]
    async fn call_tool_round_trips_through_mcp() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let (server_end, client_end) = tokio::io::duplex(4096);
        let cloned = adapter.clone();
        tokio::spawn(async move {
            let running = cloned.serve(server_end).await.expect("serve");
            let _ = running.waiting().await.ok();
        });
        let client =
            ().serve(client_end)
                .await
                .map(crate::mcp::McpClient::from_service)
                .expect("client");
        let result = client
            .service
            .call_tool(
                rmcp::model::CallToolRequestParams::new("echo")
                    .with_arguments(json!({"message": "hello"}).as_object().unwrap().clone()),
            )
            .await
            .expect("call_tool");
        let text = result
            .content
            .first()
            .and_then(rmcp::model::ContentBlock::as_text)
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("hello"));
    }
}
