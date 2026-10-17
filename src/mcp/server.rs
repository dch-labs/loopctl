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

use std::future::Future;
use std::sync::Arc;

use rmcp::ErrorData;
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::CallToolRequestParams;
use rmcp::model::PaginatedRequestParams;
use rmcp::model::ServerConfig;
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
    /// Shared across handler clones; read live on every request —
    /// each `list_tools` reads `all_schemas()` and each `call_tool`
    /// reads `get(name)`, so a registry mutated at runtime is served
    /// exactly as it stands, with no adapter rebuild.
    registry: Arc<ToolRegistry>,

    /// The per-session context handed to every [`Tool::call`].
    ///
    /// `ToolContext: Clone`; held in `Arc` so every clone of the adapter shares
    /// one context — same `session_id`, `cwd`, extensions. One MCP server = one
    /// logical session.
    context: Arc<ToolContext>,

    /// Server name reported in `initialize` → `ServerConfig`.
    ///
    /// Populates the `name` field of the rmcp `Implementation` struct. What the
    /// client shows as the server identity (e.g. in the client's server list).
    server_name: String,

    /// Server version reported in `initialize` → `ServerConfig`.
    ///
    /// Populates the `version` field of the rmcp `Implementation` struct.
    server_version: String,
}

impl McpServerAdapter {
    /// Build an adapter wrapping `registry`, dispatching every tool call with `context`.
    ///
    /// `server_name` / `server_version` populate the MCP `ServerConfig` returned
    /// during the `initialize` handshake (what the client shows as the server
    /// identity, e.g. in the client's server list). Use your application's
    /// name and version.
    ///
    /// The registry and context are moved into `Arc`s and shared by every clone
    /// of the adapter. The registry is read live on every `tools/list` and
    /// `tools/call` — a registry mutated at runtime (see
    /// [`ToolRegistry::register`](crate::tool::ToolRegistry::register)) is
    /// served as it stands, with no adapter rebuild.
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

    /// The shared handle to the served registry.
    ///
    /// The same [`Arc<ToolRegistry>`](std::sync::Arc) every `tools/list`
    /// and `tools/call` reads — mutate through it and the next request
    /// from any connected client serves the mutated surface, with no
    /// adapter rebuild (the registry is read live, never snapshotted).
    #[must_use]
    pub fn registry(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.registry)
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
    /// Returns a boxed `rmcp::service::ServerInitializeError` if the handshake
    /// fails (e.g. the client sends a malformed `initialize`).
    pub async fn serve_stdio(
        self,
    ) -> Result<
        rmcp::service::RunningService<RoleServer, Self>,
        Box<rmcp::service::ServerInitializeError>,
    > {
        let transport = rmcp::transport::io::stdio();
        self.serve(transport).await.map_err(Box::new)
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
    /// the adapter serves its registry live (mutations surface on the next
    /// request), so connected clients that want change awareness must
    /// re-list on their own schedule rather than wait for a notification
    /// this server never sends.
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::default();
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
    /// A tool whose input schema is not a compilable, object-typed JSON Schema
    /// is omitted from the listing (with a warning) rather than advertised
    /// with a schema strict clients may reject.
    ///
    /// Each tool's [`is_read_only`](crate::tool::Tool::is_read_only) flag is
    /// forwarded as the MCP `annotations.readOnlyHint` so clients can apply
    /// their own read-only policy.
    ///
    /// # Errors
    ///
    /// Never returns `Err` — listing local schemas cannot fail — but the
    /// [`ServerHandler`] signature is `Result`, so the error type remains.
    /// The listing is built synchronously; the returned future is immediately
    /// ready.
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<rmcp::model::ListToolsResult, ErrorData>> {
        let tools = self
            .registry
            .all_schemas()
            .into_iter()
            .filter_map(|schema| {
                let is_read_only = self
                    .registry
                    .get(&schema.tool)
                    .is_some_and(|tool| tool.is_read_only());
                convert::tool_schema_to_mcp(schema, is_read_only)
            })
            .collect();
        std::future::ready(Ok(rmcp::model::ListToolsResult {
            tools,
            ..Default::default()
        }))
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
        let result = dispatch_guarded(tool.as_ref(), input, &self.context, &context.ct).await;
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

    /// The `peek` tool re-registered under its name with the
    /// read-only hint flipped and nothing else moved.
    ///
    /// The metadata-only survivor of the sync pins: name, description,
    /// and schema are byte-identical to [`PeekTool`]'s — only the
    /// policy-bearing `read_only` classification flips — so a sync
    /// that compares schemas alone keeps the stale handle and the
    /// registry keeps advertising a mutable tool the server just made
    /// read-only, or vice versa.
    struct HintFlippedPeek;

    impl Tool for HintFlippedPeek {
        fn name(&self) -> &'static str {
            "peek"
        }
        fn description(&self) -> &'static str {
            PeekTool.description()
        }
        fn schema(&self) -> ToolSchema {
            PeekTool.schema()
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolOutput::text("state")) })
        }
    }

    /// An `echo` tool from a different server: same exposed name,
    /// different description.
    ///
    /// The foreign registration of the cross-provider pin: when this
    /// tool overwrites the first provider's `echo` in a shared
    /// registry, only a schema-identity check can tell whose entry the
    /// registry holds — and the first provider's sync must not remove
    /// it when its own server drops the name.
    struct ForeignEcho;

    impl Tool for ForeignEcho {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "A different server's echo"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "echo".into(),
                description: "a different server's echo".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolOutput::text("foreign")) })
        }
    }

    /// The `echo` tool re-registered under its name with a changed
    /// description.
    ///
    /// The changed survivor of the foreign-overwrite pin: the first
    /// provider's server mutates its `echo` while a second provider
    /// owns the shared registry's entry under the name — the sync must
    /// skip the refresh rather than take ownership of a foreign
    /// registration.
    struct ReDescribedEcho;

    impl Tool for ReDescribedEcho {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echo with a fresher description"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "echo".into(),
                description: "echo with a fresher description".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolOutput::text("changed")) })
        }
    }

    /// The `peek` tool re-registered under its name with a changed
    /// description.
    ///
    /// The schema-changed survivor of the sync pins: registering this
    /// over [`PeekTool`] flips the served tool's description without
    /// changing its name, so the sync that follows must refresh the
    /// survivor in place — one registry bump, no addition, no removal
    /// — while the identical `echo` survivor is not touched at all.
    struct ReDescribedPeek;

    impl Tool for ReDescribedPeek {
        fn name(&self) -> &'static str {
            "peek"
        }
        fn description(&self) -> &'static str {
            "Peek at the session with a fresher description"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "peek".into(),
                description: "peek at the session with a fresher description".into(),
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
    }

    /// A host tool whose result is a JSON object string marked with
    /// the JSON display hint — the emission fixture: serving it over
    /// the adapter must populate `structuredContent` beside the text
    /// lane.
    struct ShapedTool;

    impl Tool for ShapedTool {
        fn name(&self) -> &'static str {
            "shaped"
        }
        fn description(&self) -> &'static str {
            "Returns a shaped object"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "shaped".into(),
                description: self.description().into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async {
                Ok(ToolOutput::text(r#"{"city":"Tromsø"}"#.to_string())
                    .with_hint(crate::tool::DisplayHint::Json))
            })
        }
    }

    /// A host tool whose result is prose followed by a JSON object
    /// part, marked with the JSON display hint.
    ///
    /// The multipart emission fixture: exactly the shape the client
    /// bridge produces for a prose-plus-structured result — the JSON
    /// lane rides as the *last* text part, so an emission arm that
    /// parses the joined text instead of the last part loses it.
    struct ProseAndJsonTool;

    impl Tool for ProseAndJsonTool {
        fn name(&self) -> &'static str {
            "prose_and_json"
        }
        fn description(&self) -> &'static str {
            "Prose then a shaped object"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "prose_and_json".into(),
                description: self.description().into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async {
                Ok(
                    ToolOutput::success(crate::message::ToolContent::from_multipart(vec![
                        crate::message::ToolContentPart::text("the shaped result follows"),
                        crate::message::ToolContentPart::text(r#"{"city":"Tromsø"}"#),
                    ]))
                    .with_hint(crate::tool::DisplayHint::Json),
                )
            })
        }
    }

    /// A host tool whose JSON hint lies — the payload is prose, not
    /// JSON — so the emission arm must keep the text lane and warn
    /// instead of emitting garbage.
    struct LyingHintTool;

    impl Tool for LyingHintTool {
        fn name(&self) -> &'static str {
            "lying_hint"
        }
        fn description(&self) -> &'static str {
            "Hints JSON but returns prose"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "lying_hint".into(),
                description: self.description().into(),
                input_schema: json!({"type": "object"}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async {
                Ok(ToolOutput::text("just words".to_string())
                    .with_hint(crate::tool::DisplayHint::Json))
            })
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
        let registry = ToolRegistry::new();
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
    async fn a_dropped_name_never_removes_another_provider_s_registration() {
        let first = McpServerAdapter::new(
            {
                let registry = ToolRegistry::new();
                registry.register(EchoTool);
                registry.register(FailTool);
                registry
            },
            ToolContext::default(),
            "first".into(),
            "0.0.1".into(),
        );
        let second = McpServerAdapter::new(
            {
                let registry = ToolRegistry::new();
                registry.register(ForeignEcho);
                registry
            },
            ToolContext::default(),
            "second".into(),
            "0.0.1".into(),
        );
        let first_client = serve_and_connect(&first).await;
        let second_client = serve_and_connect(&second).await;

        let mut first_provider = crate::mcp::McpToolProvider::connect(first_client, None)
            .await
            .expect("discover first");
        let second_provider = crate::mcp::McpToolProvider::connect(second_client, None)
            .await
            .expect("discover second");
        let shared = ToolRegistry::new();
        first_provider.register_into(&shared);
        second_provider.register_into(&shared);

        first.registry().remove("echo");
        let report = first_provider.sync_into(&shared).await.expect("sync");

        assert!(
            shared.contains("echo"),
            "the second provider's registration survives the first provider's \
             drop — the baseline never owned the registry's current entry"
        );
        assert!(
            report.removed.is_empty(),
            "the only dropped name is foreign-held — nothing is reported removed: \
             {:?}",
            report.removed
        );
        assert!(
            shared.contains("fail"),
            "this provider's own survivor stays registered"
        );
        assert_eq!(
            report.untouched,
            ["fail"],
            "the survivor this target still holds is reported untouched: {report:?}"
        );
    }

    #[tokio::test]
    async fn a_changed_survivor_never_overwrites_a_foreign_registration() {
        let first = McpServerAdapter::new(
            {
                let registry = ToolRegistry::new();
                registry.register(EchoTool);
                registry
            },
            ToolContext::default(),
            "first".into(),
            "0.0.1".into(),
        );
        let second = McpServerAdapter::new(
            {
                let registry = ToolRegistry::new();
                registry.register(ForeignEcho);
                registry
            },
            ToolContext::default(),
            "second".into(),
            "0.0.1".into(),
        );
        let first_client = serve_and_connect(&first).await;
        let second_client = serve_and_connect(&second).await;

        let mut first_provider = crate::mcp::McpToolProvider::connect(first_client, None)
            .await
            .expect("discover first");
        let second_provider = crate::mcp::McpToolProvider::connect(second_client, None)
            .await
            .expect("discover second");
        let shared = ToolRegistry::new();
        first_provider.register_into(&shared);
        second_provider.register_into(&shared);

        first.registry().register(ReDescribedEcho);
        let report = first_provider.sync_into(&shared).await.expect("sync");

        let described: Vec<String> = shared
            .all_schemas()
            .into_iter()
            .filter(|schema| schema.tool == "echo")
            .map(|schema| schema.description)
            .collect();
        assert_eq!(
            described,
            ["a different server's echo"],
            "the foreign registration survives the changed survivor's sync \
             byte-equal — ownership never crosses by refresh"
        );
        assert!(
            !report.added.contains(&"echo".to_string()),
            "a skipped foreign-held name is not claimed as added: {report:?}"
        );
        assert!(
            !report.untouched.contains(&"echo".to_string()),
            "nor claimed as untouched — the report blames nothing for a name \
             this provider does not own in the target: {report:?}"
        );
    }

    #[tokio::test]
    async fn a_sync_into_a_second_registry_registers_its_survivors() {
        let adapter = McpServerAdapter::new(
            {
                let registry = ToolRegistry::new();
                registry.register(EchoTool);
                registry.register(PeekTool);
                registry
            },
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let mut provider = crate::mcp::McpToolProvider::connect(client, None)
            .await
            .expect("discover");
        let first = ToolRegistry::new();
        provider.register_into(&first);

        let second = ToolRegistry::new();
        let report = provider.sync_into(&second).await.expect("sync");

        assert_eq!(
            second.tool_names(),
            ["echo", "peek"],
            "a target that never saw the provider gains its survivors — \
             presence is reconciled against the target, not the provider map"
        );
        assert_eq!(
            report.added,
            ["echo", "peek"],
            "the report counts what this target gained: {report:?}"
        );
        assert!(
            report.untouched.is_empty(),
            "nothing was untouched in a target that held nothing"
        );
    }

    #[tokio::test]
    async fn a_refresh_after_a_server_side_change_still_refreshes_the_handle() {
        let server_registry = ToolRegistry::new();
        server_registry.register(EchoTool);
        server_registry.register(PeekTool);
        let adapter = McpServerAdapter::new(
            server_registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let mut provider = crate::mcp::McpToolProvider::connect(client, None)
            .await
            .expect("discover");
        let live = ToolRegistry::new();
        provider.register_into(&live);
        let before = live.version();

        let served = adapter.registry();
        served.register(ReDescribedPeek);
        provider.refresh().await.expect("refresh");
        let report = provider.sync_into(&live).await.expect("sync");

        assert!(
            report.added.is_empty() && report.removed.is_empty(),
            "a changed survivor is neither added nor removed: {report:?}"
        );
        assert!(
            live.version() > before,
            "the change the refresh observed still re-registers the handle — \
             the registry cannot keep serving the stale contract"
        );
        let described: Vec<String> = live
            .all_schemas()
            .into_iter()
            .filter(|schema| schema.tool == "peek")
            .map(|schema| schema.description)
            .collect();
        assert_eq!(
            described,
            ["peek at the session with a fresher description"],
            "the registry reflects the server's changed description, not the \
             one captured at setup"
        );
    }

    #[tokio::test]
    async fn a_refresh_between_setup_and_sync_still_sheds_dropped_tools() {
        let server_registry = ToolRegistry::new();
        server_registry.register(EchoTool);
        server_registry.register(FailTool);
        let adapter = McpServerAdapter::new(
            server_registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let mut provider = crate::mcp::McpToolProvider::connect(client, None)
            .await
            .expect("discover");
        let live = ToolRegistry::new();
        provider.register_into(&live);

        let served = adapter.registry();
        served.remove("fail");
        served.register(PeekTool);
        provider.refresh().await.expect("refresh");

        let report = provider.sync_into(&live).await.expect("sync");
        assert!(
            report.removed.contains(&"fail".to_string()),
            "the tool the server dropped reaches the report's removals: {:?}",
            report.removed
        );
        assert!(
            !live.contains("fail"),
            "the orphan leaves the registry — the sync mirrors the server, \
             not the refreshed snapshot"
        );
    }

    #[tokio::test]
    async fn a_read_only_hint_flip_refreshes_the_surviving_handle() {
        let server_registry = ToolRegistry::new();
        server_registry.register(EchoTool);
        server_registry.register(PeekTool);
        let adapter = McpServerAdapter::new(
            server_registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let mut provider = crate::mcp::McpToolProvider::connect(client, None)
            .await
            .expect("discover");
        let live = ToolRegistry::new();
        provider.register_into(&live);
        let before = live.version();
        assert!(
            live.get("peek").is_some_and(|tool| tool.is_read_only()),
            "the fixture starts with the read-only classification"
        );

        let served = adapter.registry();
        served.register(HintFlippedPeek);
        let report = provider.sync_into(&live).await.expect("sync");

        assert!(
            report.added.is_empty() && report.removed.is_empty(),
            "a hint flip is neither: {report:?}"
        );
        assert!(
            live.version() > before,
            "the policy-bearing contract moved — the survivor's handle is replaced"
        );
        assert!(
            !live.get("peek").is_some_and(|tool| tool.is_read_only()),
            "the registry reflects the server's flipped classification"
        );
    }

    #[tokio::test]
    async fn server_emits_structured_content_for_json_payloads() {
        let registry = ToolRegistry::new();
        registry.register(ShapedTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let result = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("shaped"))
            .await
            .expect("call shaped");
        let structured = result
            .structured_content
            .as_ref()
            .expect("the JSON-hinted payload rides the structured lane");
        assert_eq!(
            structured.get("city").and_then(serde_json::Value::as_str),
            Some("Tromsø"),
            "the parsed object rides verbatim"
        );
        assert!(
            result
                .content
                .iter()
                .any(|block| matches!(block, rmcp::model::ContentBlock::Text(_))),
            "the text lane survives for clients that ignore structure"
        );
    }

    #[tokio::test]
    async fn server_emits_structured_content_from_a_multipart_payload() {
        let registry = ToolRegistry::new();
        registry.register(ProseAndJsonTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let result = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("prose_and_json"))
            .await
            .expect("call prose_and_json");
        let structured = result
            .structured_content
            .as_ref()
            .expect("the JSON lane rides the structured lane even beside prose");
        assert_eq!(
            structured.get("city").and_then(serde_json::Value::as_str),
            Some("Tromsø"),
            "the parsed object is the last text part, not the joined text"
        );
    }

    #[tokio::test]
    async fn a_lying_json_hint_keeps_the_text_lane_only() {
        let registry = ToolRegistry::new();
        registry.register(LyingHintTool);
        let adapter = McpServerAdapter::new(
            registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let result = client
            .service
            .call_tool(rmcp::model::CallToolRequestParams::new("lying_hint"))
            .await
            .expect("call lying_hint");
        assert!(
            result.structured_content.is_none(),
            "prose that fails to parse never becomes structuredContent"
        );
        assert!(
            result
                .content
                .iter()
                .any(|block| matches!(block, rmcp::model::ContentBlock::Text(_))),
            "the prose rides the text lane unchanged"
        );
    }

    #[tokio::test]
    async fn sync_into_diffs_the_live_registry() {
        let server_registry = ToolRegistry::new();
        server_registry.register(EchoTool);
        server_registry.register(FailTool);
        let adapter = McpServerAdapter::new(
            server_registry,
            ToolContext::default(),
            "test".into(),
            "0.0.1".into(),
        );
        let client = serve_and_connect(&adapter).await;

        let mut provider = crate::mcp::McpToolProvider::connect(client, None)
            .await
            .expect("discover");
        let live = ToolRegistry::new();
        provider.register_into(&live);
        let baseline = live.version();
        assert_eq!(
            live.tool_names(),
            ["echo", "fail"],
            "the initial wiring registers the discovered set"
        );

        let served = adapter.registry();
        served.remove("fail");
        served.register(PeekTool);

        let report = provider.sync_into(&live).await.expect("sync");
        assert_eq!(
            report.added,
            ["peek"],
            "the server's new tool is the sync's addition"
        );
        assert_eq!(
            report.removed,
            ["fail"],
            "the server's dropped tool is the sync's removal"
        );
        assert_eq!(
            report.untouched,
            ["echo"],
            "a name present in both snapshots survives"
        );
        assert_eq!(
            live.version(),
            baseline + 2,
            "one registry bump per added and per removed name — the survivor never moved"
        );
        let order: Vec<String> = live
            .all_schemas()
            .into_iter()
            .map(|schema| schema.tool)
            .collect();
        assert_eq!(
            order,
            ["echo", "peek"],
            "the survivor keeps its registration position; the mirror is the server's list"
        );

        let settled = live.version();
        served.register(ReDescribedPeek);
        let refreshed = provider.sync_into(&live).await.expect("second sync");
        assert_eq!(
            refreshed.added,
            Vec::<String>::new(),
            "a re-described survivor is not an addition"
        );
        assert_eq!(refreshed.removed, Vec::<String>::new(), "nor a removal");
        assert_eq!(
            refreshed.untouched,
            ["echo", "peek"],
            "both names survived the second sync"
        );
        assert_eq!(
            live.version(),
            settled + 1,
            "only the schema-changed survivor re-registered — the identical one did not move"
        );
        let described: Vec<String> = live
            .all_schemas()
            .into_iter()
            .filter(|schema| schema.tool == "peek")
            .map(|schema| schema.description)
            .collect();
        assert_eq!(
            described,
            ["peek at the session with a fresher description"],
            "the refreshed schema reads on the registry the engine serves"
        );
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
        let registry = ToolRegistry::new();
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
        let registry = ToolRegistry::new();
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
        let registry = ToolRegistry::new();
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
        let guard = std::time::Duration::from_secs(10);
        tokio::time::timeout(guard, started_wait)
            .await
            .expect("hanging tool was polled at least once");
        handle
            .cancel(Some("test cancellation".into()))
            .await
            .expect("cancel notification sent");
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
