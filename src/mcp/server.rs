//! MCP server adapter — serve a loopctl `ToolRegistry` over MCP.
//!
//! [`McpServerAdapter`] wraps a `ToolRegistry` and
//! implements rmcp's [`ServerHandler`] so that any MCP-speaking client can
//! discover and invoke the registry's tools as if they were native MCP tools.
//!
//! # Transports
//!
//! The adapter's own API never names a transport. [`McpServerAdapter::serve_stdio`]
//! is a convenience for the canonical interactive case (stdin/stdout JSON-RPC);
//! a host that wants any other transport rmcp supports calls
//! `adapter.clone().serve(transport)` directly — the handler is
//! transport-independent by construction.
//!
//! # Tool names
//!
//! Tool names are forwarded to the client exactly as the registry knows them.
//! The MCP specification recommends names matching `^[a-zA-Z0-9_-]{1,64}$`, and
//! strict clients reject tool lists containing non-conforming names. loopctl
//! imposes no naming rules of its own, so a registry whose tool names carry
//! dots, unicode, or exceed 64 characters may be refused by such a client —
//! conforming names are the embedding application's responsibility.
//!
//! # Non-goals
//!
//! - **No resources / prompts / sampling / logging / completions / tasks** —
//!   only the MCP **tools** capability is advertised. loopctl has no model for
//!   the other primitives.
//! - **No loopctl `Hook` / permission integration** — `call_tool` dispatches
//!   through [`Tool::call`] directly. A future task
//!   could wrap the dispatch in loopctl's `HookExecutor`.
//! - **No per-chunk streaming** — MCP delivers a tool result as one
//!   `CallToolResult`.
//! - **No live tool-list mutation** — the adapter serves a snapshot; rebuild
//!   for a different tool set.
//!
//! # Wiring it into an MCP client
//!
//! Build the example (`cargo build --example mcp_server --features mcp`) and
//! register the resulting binary with an MCP client that spawns stdio servers;
//! each client documents how to point it at a command. The piped JSON-RPC
//! handshake in the example's module doc exercises the same code path with no
//! client at all.

use std::sync::Arc;

use rmcp::ErrorData;
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::CallToolRequestParams;
use rmcp::model::PaginatedRequestParams;
use rmcp::model::ServerInfo;
use rmcp::model::ToolsCapability;
use rmcp::service::RequestContext;
use rmcp::service::RoleServer;

use crate::mcp::convert;
use crate::tool::Tool;
use crate::tool::ToolContext;
use crate::tool::ToolError;
use crate::tool::ToolOutput;
use crate::tool::ToolRegistry;

/// An MCP server backed by a loopctl [`ToolRegistry`].
///
/// Wraps a registry (and the [`ToolContext`] passed to every dispatch) and
/// exposes it over MCP via the [`ServerHandler`] trait: `list_tools` returns
/// every registered tool's schema, `call_tool` dispatches to
/// [`Tool::call`].
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

    /// The per-session context handed to every [`Tool::call`].
    ///
    /// `ToolContext: Clone`; held in `Arc` so every clone of the adapter shares
    /// one context — same `session_id`, `cwd`, extensions. One MCP server = one
    /// logical session.
    context: Arc<ToolContext>,

    /// Server name reported in `initialize` → `ServerInfo`.
    ///
    /// Populates the `name` field of the rmcp `Implementation` struct. What the
    /// client shows as the server identity (e.g. in the client's server list).
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
    /// identity, e.g. in the client's server list). Use your application's
    /// name and version.
    ///
    /// The registry and context are moved into `Arc`s and shared by every clone
    /// of the adapter. The registry is **not** mutated after construction —
    /// rebuild the adapter if you need a different tool set.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::mcp::McpServerAdapter;
    /// use loopctl::tool::{ToolContext, ToolRegistry};
    ///
    /// let registry = ToolRegistry::new();
    /// let adapter = McpServerAdapter::new(
    ///     registry,
    ///     ToolContext::default(),
    ///     "my-server".into(),
    ///     "0.1.0".into(),
    /// );
    /// let _serving_clone = adapter.clone();
    /// ```
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
    /// Convenience for the canonical interactive case: stdin/stdout JSON-RPC.
    /// Equivalent to serving on `rmcp::transport::io::stdio()`. The returned
    /// service has completed the `initialize`/`initialized` handshake; call
    /// `.waiting()` to block until the client disconnects (stdin EOF), or cancel
    /// its `cancellation_token()` to shut down proactively. Process signals are
    /// the embedding binary's responsibility, not the adapter's — see the
    /// `mcp_server` example for the canonical Ctrl-C → cancel shape.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::mcp::McpServerAdapter;
    /// use loopctl::tool::{ToolContext, ToolRegistry};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let adapter = McpServerAdapter::new(
    ///     ToolRegistry::new(),
    ///     ToolContext::default(),
    ///     "my-server".into(),
    ///     "0.1.0".into(),
    /// );
    /// let service = adapter.serve_stdio().await?;
    /// service.waiting().await?;
    /// # Ok(())
    /// # }
    /// ```
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

/// Dispatch one tool call, guarded against request cancellation.
///
/// The cancellation token is polled first (`biased` selection), and the tool
/// future is only constructed when its branch is polled — so a request that is
/// already cancelled when it reaches dispatch resolves to
/// [`ToolError::Cancelled`] without invoking the tool at all. Once the tool is
/// running, cancellation drops its in-flight future; tools served over MCP must
/// therefore be cancellation-safe, the same contract the engine's dispatch
/// path imposes.
///
/// # Errors
///
/// [`ToolError::Cancelled`] when the token is already cancelled or fires while
/// the tool is running; otherwise whatever the tool itself reports.
async fn dispatch_guarded(
    tool: &dyn Tool,
    input: serde_json::Value,
    context: &ToolContext,
    ct: &tokio_util::sync::CancellationToken,
) -> Result<ToolOutput, ToolError> {
    tokio::select! {
        biased;
        () = ct.cancelled() => Err(ToolError::Cancelled),
        result = async { tool.call(input, context).await } => result,
    }
}

impl ServerHandler for McpServerAdapter {
    /// Advertise the server identity and the **tools** capability only.
    ///
    /// Resources, prompts, sampling, logging, completions, and subscriptions
    /// are neither implemented nor advertised; their [`ServerHandler`]
    /// defaults (no-ops / `None`) apply. `listChanged` is not advertised —
    /// the adapter serves the snapshot taken at construction.
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities.tools = Some(ToolsCapability::default());
        info.server_info.name.clone_from(&self.server_name);
        info.server_info.version.clone_from(&self.server_version);
        info
    }

    /// `tools/list` → every registered tool's schema, mapped to MCP tools.
    ///
    /// Pagination is ignored: loopctl registries are small, and the MCP
    /// `nextCursor` mechanism exists for tool lists with thousands of entries.
    /// The whole list is returned in one page regardless of any client cursor.
    /// A tool whose input schema is not an object-typed JSON Schema is omitted
    /// from the listing (with a warning) rather than advertised with a
    /// malformed schema.
    ///
    /// Each tool's [`is_read_only`](crate::tool::Tool::is_read_only) flag is
    /// forwarded as the MCP `annotations.readOnlyHint` so clients can apply
    /// their own read-only policy.
    ///
    /// # Errors
    ///
    /// Never returns `Err` — listing local schemas cannot fail — but the
    /// [`ServerHandler`] signature is `Result`, so the error type remains.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, ErrorData> {
        let tools = self
            .registry
            .all_schemas()
            .into_iter()
            .filter_map(|schema| {
                let is_read_only = self
                    .registry
                    .get(&schema.tool)
                    .is_some_and(Tool::is_read_only);
                convert::tool_schema_to_mcp(schema, is_read_only)
            })
            .collect();
        Ok(rmcp::model::ListToolsResult {
            tools,
            ..Default::default()
        })
    }

    /// `tools/call` → registry lookup → [`Tool::call`].
    ///
    /// - **Unknown tool name** → a `METHOD_NOT_FOUND` protocol error, not a
    ///   tool-level result: the client asked for a tool the server does not
    ///   have, which is a request-routing failure. The message names the
    ///   registered tools for clients and logs that surface protocol-error
    ///   text.
    /// - **Tool found, `Ok`** → content from the [`ToolOutput`]
    ///   payload with `is_error` mirroring its soft-error flag.
    /// - **Tool found, `Err`** → `is_error: true` with the error text as
    ///   content — the model should see "permission denied: …" and react, not
    ///   a transport error.
    ///
    /// The call is raced against the request's cancellation token: a request
    /// that is already cancelled when it reaches dispatch never invokes the
    /// tool, and when the client cancels mid-flight (`notifications/cancelled`)
    /// or the connection drops, the in-flight future is **dropped** and the
    /// call resolves to a cancelled tool-level result. Tools served over MCP
    /// must therefore be cancellation-safe — the same contract the engine's
    /// dispatch path imposes on tools it runs.
    ///
    /// # Errors
    ///
    /// Returns `Err` only for an unknown tool name (`METHOD_NOT_FOUND`, with
    /// the registered names in the message). Everything the tool itself
    /// reports — soft error, hard failure, cancellation — is a tool-level
    /// `Ok` result with `is_error: true`.
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, ErrorData> {
        let name = request.name.to_string();
        let Some(tool) = self.registry.get(&name) else {
            let names = self.registry.tool_names();
            let available: Vec<&str> = names.iter().map(String::as_str).collect();
            return Err(convert::not_found_error(&name, &available));
        };
        let input = request
            .arguments
            .map_or(serde_json::Value::Null, serde_json::Value::Object);
        let result = dispatch_guarded(tool, input, &self.context, &context.ct).await;
        Ok(convert::dispatch_result_to_call_tool(&name, result).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
    use rmcp::service::PeerRequestOptions;
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

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

    struct PeekTool;

    impl Tool for PeekTool {
        fn name(&self) -> &'static str {
            "peek"
        }
        fn description(&self) -> &'static str {
            "Reads state without modifying it"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "peek".into(),
                description: "Reads state without modifying it".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolOutput::text("state")) })
        }
        fn is_read_only(&self) -> bool {
            true
        }
    }

    struct RecordingTool {
        received: Arc<Mutex<Option<serde_json::Value>>>,
    }

    impl Tool for RecordingTool {
        fn name(&self) -> &'static str {
            "record"
        }
        fn description(&self) -> &'static str {
            "Records the input value it received"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "record".into(),
                description: "Records the input value it received".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            *self.received.lock().unwrap() = Some(input);
            Box::pin(async { Ok(ToolOutput::text("recorded")) })
        }
    }

    struct HangingTool {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<tokio::sync::Notify>,
    }

    impl Tool for HangingTool {
        fn name(&self) -> &'static str {
            "hanging"
        }
        fn description(&self) -> &'static str {
            "Never completes"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "hanging".into(),
                description: "Never completes".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(NeverCompletingCall {
                started: Arc::clone(&self.started),
                dropped: Arc::clone(&self.dropped),
            })
        }
    }

    /// A tool-call future that signals its every poll, never resolves, and
    /// signals its drop — the observable ends of the cancellation contract.
    struct NeverCompletingCall {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<tokio::sync::Notify>,
    }

    impl Future for NeverCompletingCall {
        type Output = Result<ToolOutput, ToolError>;

        fn poll(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            self.started.notify_one();
            std::task::Poll::Pending
        }
    }

    impl Drop for NeverCompletingCall {
        fn drop(&mut self) {
            self.dropped.notify_one();
        }
    }

    /// Records invocation at the moment `call` runs — before any awaiting —
    /// so a test can tell whether dispatch ever started the tool.
    struct CallCountingTool {
        invoked: Arc<AtomicBool>,
    }

    impl Tool for CallCountingTool {
        fn name(&self) -> &'static str {
            "counted"
        }
        fn description(&self) -> &'static str {
            "Records whether it was invoked"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "counted".into(),
                description: "Records whether it was invoked".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            self.invoked.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(ToolOutput::text("ran")) })
        }
    }

    fn echo_fail_adapter() -> McpServerAdapter {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(FailTool);
        McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        )
    }

    /// Serve `adapter` over an in-process duplex and connect a real MCP client
    /// to it — the shared harness for every dispatch-level test.
    async fn serve_and_connect(adapter: &McpServerAdapter) -> crate::mcp::McpClient {
        let (server_end, client_end) = tokio::io::duplex(4096);
        let server = adapter.clone();
        tokio::spawn(async move {
            let running = server.serve(server_end).await.expect("server serve");
            let _ = running.waiting().await.ok();
        });
        ().serve(client_end)
            .await
            .map(crate::mcp::McpClient::from_service)
            .expect("client connect")
    }

    #[test]
    fn get_info_advertises_tools_only() {
        let adapter = echo_fail_adapter();
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

    #[tokio::test]
    async fn list_tools_serves_registry_via_mcp() {
        let adapter = echo_fail_adapter();
        let client = serve_and_connect(&adapter).await;
        let tools = client
            .service
            .list_all_tools()
            .await
            .expect("list_all_tools");
        drop(client);
        let names: Vec<String> = tools.into_iter().map(|t| t.name.to_string()).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"fail".to_string()));
    }

    #[tokio::test]
    async fn list_tools_forwards_read_only_hint() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(PeekTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;
        let tools = client
            .service
            .list_all_tools()
            .await
            .expect("list_all_tools");
        drop(client);
        let peek = tools
            .iter()
            .find(|t| t.name.as_ref() == "peek")
            .expect("peek is listed");
        let annotations = peek
            .annotations
            .as_ref()
            .expect("read-only hints forwarded");
        assert_eq!(annotations.read_only_hint, Some(true));
        assert_eq!(annotations.destructive_hint, Some(false));
        let echo = tools
            .iter()
            .find(|t| t.name.as_ref() == "echo")
            .expect("echo is listed");
        assert!(echo.annotations.is_none());
    }

    #[tokio::test]
    async fn call_tool_round_trips_through_mcp() {
        let adapter = echo_fail_adapter();
        let client = serve_and_connect(&adapter).await;
        let result = client
            .service
            .call_tool(
                rmcp::model::CallToolRequestParams::new("echo")
                    .with_arguments(json!({"message": "hello"}).as_object().unwrap().clone()),
            )
            .await
            .expect("call_tool");
        drop(client);
        let text = result
            .content
            .first()
            .and_then(rmcp::model::ContentBlock::as_text)
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("hello"));
    }

    #[tokio::test]
    async fn call_tool_unknown_tool_is_a_method_not_found_protocol_error() {
        let adapter = echo_fail_adapter();
        let client = serve_and_connect(&adapter).await;
        let err = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("grep"))
            .await
            .expect_err("unknown tool is a protocol error");
        drop(client);
        let msg = err.to_string();
        assert!(
            msg.contains("Tool not found") && msg.contains("grep"),
            "error names the missing tool: {msg}"
        );
        assert!(msg.contains("echo"), "error lists available tools: {msg}");
    }

    #[tokio::test]
    async fn call_tool_hard_tool_failure_is_a_tool_level_error_result() {
        let adapter = echo_fail_adapter();
        let client = serve_and_connect(&adapter).await;
        let result = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("fail"))
            .await
            .expect("hard tool failure is Ok, not a protocol error");
        drop(client);
        assert_eq!(result.is_error, Some(true));
        let text = result
            .content
            .first()
            .and_then(rmcp::model::ContentBlock::as_text)
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("Execution error: always fails"));
    }

    #[tokio::test]
    async fn call_tool_without_arguments_passes_null_to_the_tool() {
        let received = Arc::new(Mutex::new(None));
        let mut registry = ToolRegistry::new();
        registry.register(RecordingTool {
            received: Arc::clone(&received),
        });
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;
        let result = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("record"))
            .await
            .expect("argument-less call succeeds");
        drop(client);
        assert_eq!(result.is_error, Some(false));
        assert_eq!(*received.lock().unwrap(), Some(serde_json::Value::Null));
    }

    #[tokio::test]
    async fn call_tool_in_flight_call_is_cancelled_when_client_cancels() {
        let started = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(tokio::sync::Notify::new());
        let mut registry = ToolRegistry::new();
        registry.register(HangingTool {
            started: Arc::clone(&started),
            dropped: Arc::clone(&dropped),
        });
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;
        // Both waiters are registered before the request is sent, so neither
        // signal can be missed — the notified futures exist before the events.
        let started_wait = started.notified();
        let dropped_wait = dropped.notified();
        tokio::pin!(started_wait, dropped_wait);
        let request =
            rmcp::model::CallToolRequest::new(rmcp::model::CallToolRequestParams::new("hanging"));
        let handle = client
            .service
            .send_cancellable_request(
                rmcp::model::ClientRequest::CallToolRequest(request),
                PeerRequestOptions::no_options(),
            )
            .await
            .expect("send request");
        started_wait.await;
        handle
            .cancel(Some("test cancellation".into()))
            .await
            .expect("cancel notification sent");
        let guard = std::time::Duration::from_secs(10);
        tokio::time::timeout(guard, dropped_wait)
            .await
            .expect("cancelled call's tool future was dropped");
    }

    #[tokio::test]
    async fn dispatch_guarded_already_cancelled_token_never_invokes_the_tool() {
        let invoked = Arc::new(AtomicBool::new(false));
        let tool = CallCountingTool {
            invoked: Arc::clone(&invoked),
        };
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let result = dispatch_guarded(
            &tool,
            serde_json::Value::Null,
            &ToolContext::default(),
            &token,
        )
        .await;
        assert!(
            matches!(result, Err(ToolError::Cancelled)),
            "already-cancelled request resolves to Cancelled, got {result:?}"
        );
        assert!(
            !invoked.load(Ordering::SeqCst),
            "already-cancelled request must not invoke the tool"
        );
    }

    #[tokio::test]
    async fn dispatch_guarded_live_token_invokes_the_tool() {
        let invoked = Arc::new(AtomicBool::new(false));
        let tool = CallCountingTool {
            invoked: Arc::clone(&invoked),
        };
        let token = tokio_util::sync::CancellationToken::new();
        let result = dispatch_guarded(
            &tool,
            serde_json::Value::Null,
            &ToolContext::default(),
            &token,
        )
        .await;
        assert!(
            result.is_ok(),
            "live request dispatches normally: {result:?}"
        );
        assert!(
            invoked.load(Ordering::SeqCst),
            "live request must invoke the tool"
        );
    }
}
