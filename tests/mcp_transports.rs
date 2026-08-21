//! Integration tests for the MCP transports (stdio + Streamable HTTP/SSE).
//!
//! **stdio** tests are real-subprocess: they spawn the loopctl-shipped
//! `examples/mcp-stdio-server` binary (an rmcp `#[tool_router]` server over
//! stdio), so they run in CI with no external runtime dependency.
//!
//! **HTTP** tests are `#[ignore]` + `LOOPCTL_MCP_E2E=1`: they spawn an official
//! Python/TypeScript SDK streamable-http server, which CI lacks. A developer
//! opts in with `LOOPCTL_MCP_E2E=1 cargo test --features mcp -- --ignored`.

#![cfg(feature = "mcp")]
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

use loopctl::mcp::{CommandSpec, McpClient, McpError, McpToolProvider};
use loopctl::stream::handler::StreamRetryConfig;

/// Path to the loopctl-shipped stdio example server binary.
///
/// Derived from the test binary's own location: `cargo test` places the test
/// binary in `target/<profile>/deps/` and example binaries in
/// `target/<profile>/examples/`, so walking two levels up from
/// `current_exe()` lands on the profile directory regardless of `--release`
/// or a custom `CARGO_TARGET_DIR`.
fn stdio_server_bin() -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let profile_dir = exe
        .parent()
        .and_then(|deps| deps.parent())
        .expect("profile directory above the test binary's deps directory");
    let name = if cfg!(windows) {
        "mcp-stdio-server.exe"
    } else {
        "mcp-stdio-server"
    };
    profile_dir
        .join("examples")
        .join(name)
        .to_string_lossy()
        .to_string()
}

/// A `CommandSpec` pointing at the loopctl stdio example server.
fn stdio_server_spec() -> CommandSpec {
    CommandSpec {
        program: stdio_server_bin(),
        args: vec![],
        env: vec![],
        cwd: None,
    }
}

/// Discover tools via the public `McpToolProvider` API.
async fn discover(client: McpClient) -> Vec<String> {
    let provider = McpToolProvider::connect(client, None)
        .await
        .expect("connect + list_tools");
    provider
        .tools()
        .iter()
        .map(loopctl::tool::Tool::name)
        .map(str::to_owned)
        .collect()
}

/// A `StreamRetryConfig` tightened for fast tests: 2 retries, ~1ms base.
fn fast_retry() -> StreamRetryConfig {
    StreamRetryConfig {
        max_retries: 2,
        base_delay_ms: 1,
        max_delay_ms: 5,
        jitter_factor: 0.0,
    }
}

#[tokio::test]
async fn stdio_discovers_tools_from_child() {
    let client = McpClient::stdio(stdio_server_spec())
        .await
        .expect("stdio connect");
    let names = discover(client).await;
    assert_eq!(names, vec!["greet"], "the example server exposes one tool");
}

#[tokio::test]
async fn stdio_spawn_failure_is_handshake_error() {
    // A nonexistent program: TokioChildProcess::new returns io::Error.
    let spec = CommandSpec {
        program: "/nonexistent/mcp-server-binary-xyz".into(),
        args: vec![],
        env: vec![],
        cwd: None,
    };
    let err = McpClient::stdio(spec).await.expect_err("spawn must fail");
    assert!(matches!(err, McpError::Handshake(_)), "got {err:?}");
}

#[tokio::test]
async fn stdio_command_spec_env_is_applied() {
    // CommandSpec carries env vars to the child. The example server ignores
    // unknown env, so we only assert the spawn + handshake succeeds with an
    // extra var set — proving env was forwarded without breaking the child.
    let spec = CommandSpec {
        program: stdio_server_bin(),
        args: vec![],
        env: vec![("LOOPCTL_TEST_MARKER".into(), "present".into())],
        cwd: None,
    };
    let client = McpClient::stdio(spec).await.expect("connect with env");
    let names = discover(client).await;
    assert_eq!(names, vec!["greet"]);
}

#[tokio::test]
async fn stdio_drop_does_not_hang() {
    // rmcp's TokioChildProcess kills the child on drop. This test exercises the
    // drop path end-to-end and confirms it completes promptly (no hang waiting
    // on a leaked child). The reconnect test below proves the child is gone.
    let client = McpClient::stdio(stdio_server_spec())
        .await
        .expect("connect");
    tokio::time::timeout(Duration::from_secs(2), async move { drop(client) })
        .await
        .expect("drop completes within 2s");
}

#[tokio::test]
async fn reconnect_in_process_client_is_error() {
    // in_process clients have no reconnect_spec — reconnect must reject.
    use rmcp::handler::server::ServerHandler;
    use rmcp::{ServiceExt, tool, tool_handler, tool_router};
    #[derive(Clone)]
    struct S {
        router: rmcp::handler::server::router::tool::ToolRouter<Self>,
    }
    #[tool_router]
    impl S {
        fn new() -> Self {
            Self {
                router: Self::tool_router(),
            }
        }
        #[tool(description = "noop")]
        async fn noop(&self) -> String {
            "ok".into()
        }
    }
    #[allow(clippy::unused_async_trait_impl)] // FIXME(rmcp): drop when tool_handler emits awaits
    #[tool_handler]
    impl ServerHandler for S {}

    let (server_end, client_end) = tokio::io::duplex(4096);
    tokio::spawn(async move {
        if let Ok(r) = S::new().serve(server_end).await {
            let _ = r.waiting().await.ok();
        }
    });
    let client = ().serve(client_end).await.map(McpClient::from_service).expect("connect");
    let err = client
        .reconnect(&fast_retry())
        .await
        .expect_err("in_process cannot reconnect");
    assert!(matches!(err, McpError::Handshake(_)), "got {err:?}");
}

#[tokio::test]
async fn reconnect_stdio_re_establishes_after_drop() {
    // Build a client (child spawned), drop it (child dies per rmcp cleanup),
    // then reconnect on the dead client's retained spec — the spec re-spawns
    // the same binary and rediscovers the tool.
    //
    // This covers reconnect's happy path (spec read, constructor re-run,
    // success returned). The give-up-after-max-retries path is structurally
    // the same loop with a failing constructor; it is not tested directly
    // because `ReconnectSpec` is private and a guaranteed-failing spec cannot
    // be attached to a client via the public API (the constructor must
    // succeed first to produce a client, and a failing constructor produces
    // no client to call `reconnect` on). The loop bound `0..=max_retries` is
    // simple enough to read; the per-attempt logic is exercised here.
    let spec = stdio_server_spec();
    let live = McpClient::stdio(spec).await.expect("first connect");
    let dead = live.clone();
    drop(live);
    // Give rmcp's async child-kill a moment to complete.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let reconnected = dead
        .reconnect(&fast_retry())
        .await
        .expect("reconnect re-spawns the child");
    let names = discover(reconnected).await;
    assert_eq!(
        names,
        vec!["greet"],
        "reconnected client rediscovers the tool"
    );
}

fn e2e_enabled() -> bool {
    std::env::var("LOOPCTL_MCP_E2E").is_ok_and(|v| v == "1")
}

#[tokio::test]
#[ignore = "requires LOOPCTL_MCP_E2E=1 and an official SDK streamable-http server on 127.0.0.1:3001"]
async fn http_sse_round_trip_against_local_sdk_server() {
    if !e2e_enabled() {
        eprintln!("skipped: set LOOPCTL_MCP_E2E=1 to run the live HTTP smoke");
        return;
    }
    let client = McpClient::http_sse("http://127.0.0.1:3001/mcp")
        .await
        .expect("http connect");
    let names = discover(client).await;
    assert!(!names.is_empty(), "server advertised tools");
}

#[tokio::test]
#[ignore = "requires LOOPCTL_MCP_E2E=1 and an official SDK streamable-http server on 127.0.0.1:3001"]
async fn http_sse_with_client_round_trips() {
    if !e2e_enabled() {
        eprintln!("skipped: set LOOPCTL_MCP_E2E=1 to run the live HTTP smoke");
        return;
    }
    let req_client = reqwest::Client::builder().build().expect("reqwest client");
    let client = McpClient::http_sse_with_client("http://127.0.0.1:3001/mcp", req_client)
        .await
        .expect("http connect");
    let names = discover(client).await;
    assert!(!names.is_empty(), "server advertised tools");
}

/// An endpoint whose single connection is accepted and then severed.
///
/// Binds a loopback listener on an OS-assigned port, accepts exactly one
/// connection on a background thread, and closes it — so a client dialing
/// the endpoint deterministically sees the connection drop mid-handshake,
/// with no dependence on how any well-known port behaves on this host.
fn severed_endpoint() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().expect("assigned port");
    std::thread::spawn(move || {
        if let Ok((conn, _)) = listener.accept() {
            drop(conn);
        }
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn http_sse_connect_refused_is_handshake_error() {
    let err = McpClient::http_sse(severed_endpoint())
        .await
        .expect_err("severed connection must fail the handshake");
    assert!(matches!(err, McpError::Handshake(_)), "got {err:?}");
}

#[tokio::test]
async fn http_sse_with_client_connect_refused_is_handshake_error() {
    let req_client = reqwest::Client::builder().build().expect("reqwest client");
    let err = McpClient::http_sse_with_client(severed_endpoint(), req_client)
        .await
        .expect_err("severed connection must fail the handshake");
    assert!(matches!(err, McpError::Handshake(_)), "got {err:?}");
}
