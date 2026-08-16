//! Integration tests for the MCP client adapter.
//!
//! Every test spins up a tiny in-process rmcp server (built with
//! `#[tool_router]` / `#[tool]`) and connects an `McpToolProvider` to it over a
//! `tokio::io::duplex`. No subprocess, no network.

#![cfg(feature = "mcp")]
// `ToolRouter<T>` fields are read by rmcp's `#[tool_handler]`-generated dispatch
// methods; the dead-code analysis can't see macro-generated reads.
#![allow(dead_code)]
// Integration tests are a separate crate and do not inherit `lib.rs`'s
// `cfg_attr(test, allow(...))`. Apply the same test-code relaxations the lib
// uses: assertions legitimately `unwrap`/`expect`/`panic`/index for clarity.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use std::time::Duration;

use loopctl::mcp::{McpClient, McpToolProvider};
use loopctl::message::ToolContent as MessageToolContent;
use loopctl::message::ToolContentPart;
use loopctl::tool::{Tool, ToolContext, ToolRegistry};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::model::CallToolResult;
use rmcp::model::ContentBlock;
use rmcp::model::Tool as RmcpTool;
use rmcp::model::ToolAnnotations;
use rmcp::{ErrorData as McpErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde_json::Value;
use serde_json::json;

/// Buffer size matching the adapter's own duplex.
const DUPLEX_BUFFER: usize = 4096;

/// Connect a pure client to `server` over an in-memory duplex, returning the
/// client and the server's background join handle (so tests can assert on
/// shutdown).
async fn connect_in_process<S>(server: S) -> (McpClient, tokio::task::JoinHandle<()>)
where
    S: ServerHandler + Clone + Send + 'static,
{
    let (server_end, client_end) = tokio::io::duplex(DUPLEX_BUFFER);
    let server_handle = tokio::spawn(async move {
        let running = server.serve(server_end).await.expect("server serve");
        let _ = running.waiting().await.ok();
    });
    let client = ().serve(client_end).await.map(McpClient::from_service).expect("client serve");
    (client, server_handle)
}

/// Echo server: an `echo` tool and a `status` tool, both argument-free so the
/// test servers need no `schemars::JsonSchema` derive (rmcp's `#[tool]` macro
/// would otherwise require it for typed `Parameters<T>`).
#[derive(Clone)]
struct EchoServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl EchoServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Echo a fixed message back")]
    async fn echo(&self) -> String {
        "hi".to_string()
    }

    #[tool(description = "Report server status")]
    async fn status(&self) -> String {
        "ok".to_string()
    }
}

#[tool_handler]
impl ServerHandler for EchoServer {}

#[tokio::test]
async fn discovery_lists_server_tools() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let tools = provider.tools();
    assert_eq!(tools.len(), 2, "two server tools discovered");
    let names: Vec<&str> = tools.iter().map(Tool::name).collect();
    assert!(names.contains(&"echo"), "echo present: {names:?}");
    assert!(names.contains(&"status"), "status present: {names:?}");
}

#[tokio::test]
async fn echo_round_trip_returns_text() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let echo = provider
        .tools()
        .iter()
        .find(|t| t.name() == "echo")
        .expect("echo tool");
    let ctx = ToolContext::default();
    let out = echo.call(json!({}), &ctx).await.expect("call ok");
    assert!(!out.is_error, "not a soft error");
    assert_eq!(out.text_content(), "hi");
    assert!(matches!(out.payload, MessageToolContent::Text(_)));
}

#[tokio::test]
async fn name_prefix_namespaces_exposed_name_only() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, Some("git".into()))
        .await
        .expect("connect");
    let status = provider
        .tools()
        .iter()
        .find(|t| t.name() == "git__status")
        .expect("prefixed status tool");
    assert_eq!(status.schema().tool, "git__status");
    // The forwarded call uses the un-prefixed server name; the status tool
    // answers "ok", proving the call reached the right server-side tool.
    let ctx = ToolContext::default();
    let out = status.call(json!({}), &ctx).await.expect("call ok");
    assert_eq!(out.text_content(), "ok");
}

#[tokio::test]
async fn register_into_populates_registry() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let mut registry = ToolRegistry::new();
    provider.register_into(&mut registry);
    assert_eq!(registry.len(), 2);
    assert!(registry.contains("echo"));
    assert!(registry.contains("status"));
    assert_eq!(registry.all_schemas().len(), 2);
}

#[tokio::test]
async fn is_read_only_defaults_false_and_never_concurrency_safe() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    for tool in provider.tools() {
        assert!(
            !tool.is_read_only(),
            "unannotated tool {} should not be read-only",
            tool.name()
        );
        assert!(
            !tool.is_concurrency_safe(),
            "MCP tool {} should never claim concurrency-safe",
            tool.name()
        );
    }
}

/// A server whose tool returns a soft error (`isError = true`) with text.
#[derive(Clone)]
struct SoftErrorServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl SoftErrorServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Always reports a soft error")]
    async fn fail(&self) -> Result<CallToolResult, McpErrorData> {
        Ok(CallToolResult::error(vec![ContentBlock::text("boom")]))
    }
}

#[tool_handler]
impl ServerHandler for SoftErrorServer {}

#[tokio::test]
async fn soft_error_returns_ok_with_is_error_set() {
    let (client, _server) = connect_in_process(SoftErrorServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let fail = provider
        .tools()
        .iter()
        .find(|t| t.name() == "fail")
        .expect("fail tool");
    let ctx = ToolContext::default();
    let out = fail.call(json!({}), &ctx).await.expect("soft error is Ok");
    assert!(out.is_error, "is_error flag set");
    assert_eq!(out.text_content(), "boom");
}

/// A server whose tool returns an empty error (`isError = true`, no content).
#[derive(Clone)]
struct EmptyErrorServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl EmptyErrorServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Reports an error with no content")]
    async fn fail(&self) -> Result<CallToolResult, McpErrorData> {
        Ok(CallToolResult::error(vec![]))
    }
}

#[tool_handler]
impl ServerHandler for EmptyErrorServer {}

/// A server whose tool rejects the call at the JSON-RPC level, returning
/// `Err(ErrorData)` from `call_tool` — exercises the protocol-error bridge
/// (`ServiceError` → `ToolError::Execution`), distinct from the soft `isError`
/// path and the empty-error path.
#[derive(Clone)]
struct ProtocolErrorServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl ProtocolErrorServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Rejects the call with a JSON-RPC error")]
    async fn reject(&self) -> Result<CallToolResult, McpErrorData> {
        Err(McpErrorData::invalid_params("not allowed", None))
    }
}

#[tool_handler]
impl ServerHandler for ProtocolErrorServer {}

#[tokio::test]
async fn empty_error_becomes_hard_toolerror() {
    let (client, _server) = connect_in_process(EmptyErrorServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let fail = provider
        .tools()
        .iter()
        .find(|t| t.name() == "fail")
        .expect("fail tool");
    let ctx = ToolContext::default();
    let err = fail
        .call(json!({}), &ctx)
        .await
        .expect_err("empty error must be a hard Err");
    match err {
        loopctl::tool::ToolError::Execution(msg) => {
            assert!(
                msg.contains("fail"),
                "empty-error message should name the tool: {msg}"
            );
        }
        other => panic!("expected ToolError::Execution, got {other:?}"),
    }
}

#[tokio::test]
async fn protocol_error_rejection_becomes_hard_toolerror() {
    let (client, _server) = connect_in_process(ProtocolErrorServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let reject = provider
        .tools()
        .iter()
        .find(|t| t.name() == "reject")
        .expect("reject tool");
    let ctx = ToolContext::default();
    let err = reject
        .call(json!({}), &ctx)
        .await
        .expect_err("a JSON-RPC rejection must be a hard Err");
    match err {
        loopctl::tool::ToolError::Execution(msg) => {
            assert!(
                msg.contains("MCP tools/call failed"),
                "error should carry the tools/call failure context: {msg}"
            );
        }
        other => panic!("expected ToolError::Execution, got {other:?}"),
    }
}

/// A server with multipart-returning tools.
#[derive(Clone)]
struct MultipartServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl MultipartServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Returns two text parts")]
    async fn two(&self) -> Result<CallToolResult, McpErrorData> {
        Ok(CallToolResult::success(vec![
            ContentBlock::text("a"),
            ContentBlock::text("b"),
        ]))
    }

    #[tool(description = "Returns text plus image")]
    async fn mixed(&self) -> Result<CallToolResult, McpErrorData> {
        Ok(CallToolResult::success(vec![
            ContentBlock::text("caption"),
            ContentBlock::image("Zm9v", "image/png"),
        ]))
    }
}

#[tool_handler]
impl ServerHandler for MultipartServer {}

#[tokio::test]
async fn multipart_text_joins_with_newline() {
    let (client, _server) = connect_in_process(MultipartServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let two = provider
        .tools()
        .iter()
        .find(|t| t.name() == "two")
        .expect("two tool");
    let ctx = ToolContext::default();
    let out = two.call(json!({}), &ctx).await.expect("call ok");
    assert!(
        matches!(out.payload, MessageToolContent::Multipart(_)),
        "expected multipart"
    );
    assert_eq!(out.text_content(), "a\nb");
}

#[tokio::test]
async fn multipart_mixed_preserves_part_kinds_in_order() {
    let (client, _server) = connect_in_process(MultipartServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let mixed = provider
        .tools()
        .iter()
        .find(|t| t.name() == "mixed")
        .expect("mixed tool");
    let ctx = ToolContext::default();
    let out = mixed.call(json!({}), &ctx).await.expect("call ok");
    let MessageToolContent::Multipart(parts) = out.payload else {
        panic!("expected multipart, got {:?}", out.payload);
    };
    assert_eq!(parts.len(), 2, "two parts");
    assert!(
        matches!(parts.first(), Some(ToolContentPart::Text { text }) if text == "caption"),
        "first is the caption text"
    );
    assert!(
        matches!(parts.get(1), Some(ToolContentPart::Image { .. })),
        "second is the image"
    );
}

/// A server whose tool returns an audio block — the "stringify, don't drop"
/// fallback.
#[derive(Clone)]
struct AudioServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl AudioServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Returns an audio block")]
    async fn beep(&self) -> Result<CallToolResult, McpErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::audio(
            "AAAA",
            "audio/wav",
        )]))
    }
}

#[tool_handler]
impl ServerHandler for AudioServer {}

#[tokio::test]
async fn unsupported_content_type_does_not_vanish() {
    let (client, _server) = connect_in_process(AudioServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let beep = provider
        .tools()
        .iter()
        .find(|t| t.name() == "beep")
        .expect("beep tool");
    let ctx = ToolContext::default();
    let out = beep.call(json!({}), &ctx).await.expect("call ok");
    let text = out.text_content();
    assert!(
        text.contains("unsupported MCP content type"),
        "audio must surface as a note, got {text:?}"
    );
}

#[tokio::test]
async fn refresh_replaces_snapshot_in_place() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let mut provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let before = provider.tools().len();

    let mut registry = ToolRegistry::new();
    provider.register_into(&mut registry);
    let registry_len_before = registry.len();

    provider.refresh().await.expect("refresh");
    assert_eq!(
        provider.tools().len(),
        before,
        "same server, same count after refresh"
    );
    assert_eq!(
        registry.len(),
        registry_len_before,
        "refresh must not mutate an already-populated registry"
    );
}

/// A server with two tools whose server-side names collide after prefixing.
#[derive(Clone)]
struct CollisionServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl CollisionServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(name = "status", description = "first status")]
    async fn status_a(&self) -> String {
        "a".to_string()
    }

    #[tool(name = "status", description = "second status")]
    async fn status_b(&self) -> String {
        "b".to_string()
    }
}

#[tool_handler]
impl ServerHandler for CollisionServer {}

#[tokio::test]
async fn intra_batch_name_collision_keeps_one_no_panic() {
    let (client, _server) = connect_in_process(CollisionServer::new()).await;
    let provider = McpToolProvider::connect(client, Some("git".into()))
        .await
        .expect("connect");
    let names: Vec<&str> = provider.tools().iter().map(Tool::name).collect();
    assert_eq!(
        names.iter().filter(|&&n| n == "git__status").count(),
        1,
        "exactly one colliding name kept: {names:?}"
    );
}

#[tokio::test]
async fn drop_provider_cancels_background_server() {
    let (client, server_handle) = connect_in_process(EchoServer::new()).await;
    {
        let provider = McpToolProvider::connect(client, None)
            .await
            .expect("connect");
        assert!(!provider.tools().is_empty());
    }
    let resolved = tokio::time::timeout(Duration::from_secs(2), server_handle)
        .await
        .expect("server shuts down within 2s of provider drop");
    resolved.expect("server task did not panic");
}

#[tokio::test]
async fn re_register_with_overlapping_name_overwrites() {
    let (client, _server) = connect_in_process(EchoServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let mut registry = ToolRegistry::new();
    provider.register_into(&mut registry);
    let len_after_first = registry.len();
    provider.register_into(&mut registry);
    assert_eq!(registry.len(), len_after_first);
}

#[tokio::test]
async fn schema_passed_through_preserves_properties() {
    // `plain` is advertised by `AnnotatedServer` with a non-trivial input
    // schema (type/properties/required); the bridge must carry it verbatim.
    let (client, _server) = connect_in_process(AnnotatedServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let plain = provider
        .tools()
        .iter()
        .find(|t| t.name() == "plain")
        .expect("plain tool");
    let adapted = plain.schema().input_schema;
    assert!(
        adapted.get("properties").is_some(),
        "adapted schema preserves properties: {adapted}"
    );
    let required = adapted.get("required").and_then(Value::as_array);
    assert!(
        required.is_some_and(|arr| arr.iter().any(|v| v == "q")),
        "required field `q` carried through: {adapted}"
    );
}

#[tokio::test]
async fn output_schema_carried_through_accessor() {
    // `plain` declares an outputSchema; the adapter carries it verbatim and
    // exposes it via McpTool::output_schema(). Pins the round-1 accessor that
    // had no coverage.
    let (client, _server) = connect_in_process(AnnotatedServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let plain = provider
        .tools()
        .iter()
        .find(|t| t.name() == "plain")
        .expect("plain tool");
    let carried = plain.output_schema().expect("outputSchema carried through");
    assert_eq!(
        carried.get("type").and_then(Value::as_str),
        Some("string"),
        "outputSchema carried verbatim: {carried}"
    );
    // The annotated tool declares no outputSchema → None.
    let annotated = provider
        .tools()
        .iter()
        .find(|t| t.name() == "annotated")
        .expect("annotated tool");
    assert!(
        annotated.output_schema().is_none(),
        "absent outputSchema → None"
    );
}

/// An `RmcpTool` with explicit annotations, used to exercise the annotation
/// bridge. Built via rmcp's constructors because `Tool`/`ToolAnnotations` are
/// `#[non_exhaustive]`.
fn annotated_tool(read_only: bool) -> RmcpTool {
    let annotations = ToolAnnotations::default()
        .read_only(read_only)
        .destructive(false);
    RmcpTool::new("annotated", "annotated tool", serde_json::Map::new())
        .with_annotations(annotations)
}

/// An `RmcpTool` with **no** annotations, a non-trivial input schema, and a
/// declared `outputSchema` — exercises the spec-default path (absent
/// `readOnlyHint` → false; absent `destructiveHint` → true), verbatim schema
/// passthrough, and `outputSchema` carriage.
fn plain_tool() -> RmcpTool {
    let mut schema = serde_json::Map::new();
    schema.insert(
        "type".to_string(),
        serde_json::Value::String("object".to_string()),
    );
    schema.insert(
        "properties".to_string(),
        serde_json::json!({"q": {"type": "string"}}),
    );
    schema.insert(
        "required".to_string(),
        serde_json::Value::Array(vec![serde_json::Value::String("q".to_string())]),
    );
    let mut output_schema = serde_json::Map::new();
    output_schema.insert(
        "type".to_string(),
        serde_json::Value::String("string".to_string()),
    );
    RmcpTool::new("plain", "an unannotated tool", schema)
        .with_raw_output_schema(std::sync::Arc::new(output_schema))
}

/// A server that advertises an annotated tool and an unannotated tool by
/// overriding `list_tools` (the `#[tool]` macro does not annotate, so a manual
/// override is the way to exercise the annotation bridge). The `#[tool_handler]`
/// macro sees the manual `list_tools` and skips generating its own, so the
/// override wins; it still generates `call_tool`/`get_info`.
#[derive(Clone)]
struct AnnotatedServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl AnnotatedServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }
}

impl ServerHandler for AnnotatedServer {
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _ctx: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, McpErrorData> {
        Ok(rmcp::model::ListToolsResult {
            next_cursor: None,
            tools: vec![annotated_tool(true), plain_tool()],
            ..Default::default()
        })
    }
}

#[tokio::test]
async fn read_only_hint_honored_from_live_server() {
    let (client, _server) = connect_in_process(AnnotatedServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let lookup = provider
        .tools()
        .iter()
        .find(|t| t.name() == "annotated")
        .expect("annotated tool discovered");
    assert!(lookup.is_read_only(), "annotated read-only honored");
}

#[tokio::test]
async fn absent_destructive_hint_defaults_to_destructive_per_spec() {
    // The MCP spec: an absent destructiveHint means "assume destructive".
    // `plain` carries no annotations, so is_destructive_hint() must be true.
    let (client, _server) = connect_in_process(AnnotatedServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let plain = provider
        .tools()
        .iter()
        .find(|t| t.name() == "plain")
        .expect("plain tool discovered");
    assert!(
        plain.is_destructive_hint(),
        "absent destructiveHint must default to destructive (true) per spec"
    );
    assert!(
        !plain.is_read_only(),
        "absent readOnlyHint must default to non-read-only (false)"
    );
    // And the explicitly-annotated tool keeps its explicit value.
    let annotated = provider
        .tools()
        .iter()
        .find(|t| t.name() == "annotated")
        .expect("annotated tool discovered");
    assert!(
        !annotated.is_destructive_hint(),
        "explicit destructiveHint=false must be honored"
    );
}

/// A server whose only tool sleeps before answering — long relative to the
/// test's shortened call timeout, short relative to the default.
#[derive(Clone)]
struct SlowServer {
    router: ToolRouter<Self>,
}

#[tool_router]
impl SlowServer {
    fn new() -> Self {
        Self {
            router: Self::tool_router(),
        }
    }

    #[tool(description = "Sleeps before answering")]
    async fn slow(&self) -> String {
        tokio::time::sleep(Duration::from_millis(150)).await;
        "finally".to_string()
    }
}

#[tool_handler]
impl ServerHandler for SlowServer {}

#[tokio::test]
async fn call_timeout_cuts_a_slow_tool_with_a_soft_error() {
    let (client, _server) = connect_in_process(SlowServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect")
        .with_call_timeout(Duration::from_millis(10));
    let slow = provider
        .tools()
        .iter()
        .find(|t| t.name() == "slow")
        .expect("slow tool");
    let ctx = ToolContext::default();

    let out = slow
        .call(json!({}), &ctx)
        .await
        .expect("timeout is a soft error");
    assert!(out.is_error, "the timeout must surface as is_error");
    let text = out.text_content();
    assert!(
        text.contains("slow") && text.contains("timed out"),
        "the soft error must name the tool and the timeout: {text:?}"
    );
}

#[tokio::test]
async fn default_timeout_lets_a_quick_tool_through() {
    let (client, _server) = connect_in_process(SlowServer::new()).await;
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect");
    let slow = provider
        .tools()
        .iter()
        .find(|t| t.name() == "slow")
        .expect("slow tool");
    let ctx = ToolContext::default();

    let out = slow
        .call(json!({}), &ctx)
        .await
        .expect("call ok under the default timeout");
    assert!(!out.is_error, "a 150ms call must survive the 60s default");
    assert_eq!(out.text_content(), "finally");
}

#[tokio::test]
async fn refresh_preserves_the_overridden_call_timeout() {
    let (client, _server) = connect_in_process(SlowServer::new()).await;
    let mut provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect")
        .with_call_timeout(Duration::from_millis(10));

    // Rebuild the tool snapshot: the refreshed tools must inherit the
    // provider's overridden timeout, not the constructor default.
    provider.refresh().await.expect("refresh");
    let slow = provider
        .tools()
        .iter()
        .find(|t| t.name() == "slow")
        .expect("slow tool after refresh");
    let ctx = ToolContext::default();

    let out = slow.call(json!({}), &ctx).await.expect("call resolves");
    assert!(
        out.is_error && out.text_content().contains("timed out"),
        "the refreshed tool must still be bounded by the override: {}",
        out.text_content()
    );
}
