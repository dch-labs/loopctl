//! MCP client adapter — adapt MCP servers as loopctl [`Tool`] implementations.
//!
//! [Model Context Protocol][mcp] servers expose callable *tools*. This module
//! connects one server, discovers its tools (`tools/list`), and wraps each one
//! as an ordinary loopctl [`Tool`] whose [`call`](Tool::call) forwards to the
//! server (`tools/call`). The agent loop, the registry, the middleware
//! pipeline, permission gates, and observers never learn a tool is remote —
//! they see a `Box<dyn Tool>`.
//!
//! # The adapter surface
//!
//! - [`McpClient`] — a connected, initialized client handle. The
//!   transport-agnostic boundary: obtain one from [`McpClient::stdio`],
//!   [`McpClient::http_sse`], or [`McpClient::in_process`].
//! - [`McpToolProvider`] — owns a [`McpClient`] and a snapshot of the server's
//!   tool list; [`McpToolProvider::connect`] discovers,
//!   [`register_into`](McpToolProvider::register_into) registers the batch into
//!   a [`ToolRegistry`].
//! - [`McpTool`] — one server tool as a [`Tool`].
//! - [`McpError`] — adapter errors.
//!
//! No rmcp type appears in any of these public signatures; an rmcp upgrade is
//! a one-file change (this one).
//!
//! # Transports
//!
//! Three ways to obtain an [`McpClient`], all yielding the same type so
//! [`McpToolProvider`] is indifferent to how the client was built:
//!
//! - **stdio** — [`McpClient::stdio`] spawns an MCP server as a child process
//!   (NDJSON JSON-RPC over the child's stdin/stdout). The common local
//!   deployment shape (MCP client, Cursor, the reference clients).
//! - **Streamable HTTP/SSE** — [`McpClient::http_sse`] (rmcp's default HTTP
//!   client) or [`McpClient::http_sse_with_client`] (caller-supplied
//!   `reqwest::Client`) connect to a remote server. rmcp handles
//!   `Mcp-Session-Id`, JSON-vs-SSE response splitting, and DELETE-on-close.
//! - **in-process** — [`McpClient::in_process`] for tests and bundled servers.
//!
//! A dropped connection can be re-established via [`McpClient::reconnect`],
//! which reuses loopctl's [`StreamRetryConfig`](crate::stream::handler::StreamRetryConfig)
//! backoff — the one retry strategy for the crate, not a second one for MCP.
//!
//! [mcp]: https://modelcontextprotocol.io
//!
//! # Example
//!
//! See `examples/mcp-adapter.rs` for a runnable end-to-end demo (an in-process
//! server, discovery, registration, and a call). In short:
//!
//! ```rust,ignore
//! use loopctl::mcp::{McpClient, McpToolProvider};
//! use loopctl::tool::ToolRegistry;
//!
//! # async fn run(server: impl rmcp::handler::server::ServerHandler) -> Result<(), loopctl::mcp::McpError> {
//! let client = McpClient::in_process(server).await?;
//! let provider = McpToolProvider::connect(client, None).await?;
//! let mut registry = ToolRegistry::new();
//! provider.register_into(&mut registry);
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Default per-call budget for a forwarded `tools/call` round-trip.
///
/// Generous enough for interactive server tools (searches, reads, short
/// builds) while still bounding a wedged server so the agent loop cannot
/// hang indefinitely. Override per provider with
/// [`McpToolProvider::with_call_timeout`](McpToolProvider::with_call_timeout).
const DEFAULT_MCP_CALL_TIMEOUT: Duration = Duration::from_mins(1);

use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::ContentBlock;
use rmcp::service::RoleClient;
use rmcp::service::RunningService;
use rmcp::transport::IntoTransport;
#[cfg(feature = "mcp")]
use rmcp::transport::StreamableHttpClientTransport;
#[cfg(feature = "mcp")]
use rmcp::transport::TokioChildProcess;
#[cfg(feature = "mcp")]
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

use crate::message::ToolContent as MessageToolContent;
use crate::message::ToolContentPart;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// Buffer size for the in-process duplex channel ([`McpClient::in_process`]).
const DUPLEX_BUFFER: usize = 4096;

/// A live, initialized connection to an MCP server.
///
/// This is the transport-agnostic boundary for the adapter. [`Self::in_process`]
/// is the only constructor shipped here: it connects a client to a server over
/// an in-memory channel. Transport constructors for real servers (a stdio
/// child process, HTTP/SSE) arrive in a later release and will build the same
/// handle via rmcp's transport APIs.
///
/// The rmcp running client service is held behind an [`Arc`]: rmcp's
/// [`RunningService`] is not itself [`Clone`] (it owns a background task and a
/// cancellation guard), so the cheap sharing the provider↔tool split needs goes
/// through the [`Arc`]. Tool calls reach the server via [`RunningService`]'s
/// [`Deref`](std::ops::Deref) to rmcp's `Peer`, which holds the channel sender.
/// Each [`McpTool`] clones the [`Arc`] rather than borrowing from the provider,
/// keeping `Tool::call(&self) -> Pin<Box<Future + '_>>` free of lifetime
/// entanglement with the provider's lifetime.
///
/// Dropping the last clone drops the [`RunningService`], whose cancellation
/// guard cancels the background task — there is no leaked runtime work.
#[derive(Clone, Debug)]
pub struct McpClient {
    /// The running rmcp client service backing this connection.
    ///
    /// Held behind an [`Arc`] so every [`McpTool`](crate::mcp::McpTool) clone
    /// shares one connection cheaply; calls reach the server via
    /// [`RunningService`]'s `Deref` to rmcp's `Peer`. The handler type is fixed
    /// to `()` (the pure client), so a server's `sampling`/`roots` requests get
    /// default empty answers — a host that wants to honour those constructs its
    /// own client with a richer handler (out of scope for this module). Dropping
    /// the last clone drops the [`RunningService`], whose cancellation guard
    /// ends the background task.
    service: Arc<RunningService<RoleClient, ()>>,

    /// How to rebuild this connection on [`Self::reconnect`], or `None`.
    ///
    /// Set by [`Self::stdio`] (a [`CommandSpec`]) and [`Self::http_sse`] (an
    /// endpoint), so a dropped connection can re-spawn the child or re-connect
    /// to the server. `None` for [`Self::in_process`] and
    /// [`Self::from_service`]: those clients have no way to reconstruct their
    /// transport, so [`Self::reconnect`] returns [`McpError::Handshake`] for
    /// them. Private because the concrete [`ReconnectSpec`] enum is an
    /// implementation detail; callers drive reconnect through the method, not
    /// the field.
    reconnect_spec: Option<ReconnectSpec>,
}

impl McpClient {
    /// Connect a client to `server` over an in-memory channel and run the MCP
    /// `initialize` handshake to completion.
    ///
    /// Opens one [`tokio::io::duplex`], drives `server.serve(server_end)` on a
    /// spawned background task, and `().serve(client_end)` (a pure client) on
    /// this future, awaiting the client's `initialize` round-trip. rmcp splits
    /// each combined read+write duplex end internally, so one duplex cross-wires
    /// the two sides — no manual split, no second duplex.
    ///
    /// Intended for the test suite and for callers who bundle an rmcp server
    /// in-process. It is **not** the path for real-world servers — use a
    /// transport constructor for those (a later release).
    ///
    /// # Runtime requirement
    ///
    /// Spawns the server's `serve` future via [`tokio::spawn`], so this must be
    /// called from within a running multi-threaded or current-thread tokio
    /// runtime (it panics with "no reactor running" otherwise). The spawned
    /// server task is **detached**: it runs until the returned [`McpClient`] (and
    /// all its clones) are dropped, at which point the client side of the duplex
    /// closes and the server's `serve` future sees EOF and ends. If the
    /// [`McpClient`] is leaked, the server task runs indefinitely — keep the
    /// handle's lifetime bounded.
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] if the client's `serve`/`initialize` fails.
    pub async fn in_process<S>(server: S) -> Result<Self, McpError>
    where
        S: ServerHandler,
    {
        let (server_end, client_end) = tokio::io::duplex(DUPLEX_BUFFER);
        tokio::spawn(async move {
            match server.serve(server_end).await {
                Ok(running) => {
                    let _ = running.waiting().await.ok();
                }
                Err(e) => tracing::error!(
                    error = %e,
                    "in-process MCP server failed to initialize (the client side reports this via McpError::Handshake)"
                ),
            }
        });
        let client = ().serve(client_end).await.map_err(|e| McpError::Handshake(e.to_string()))?;
        Ok(Self {
            service: Arc::new(client),
            reconnect_spec: None,
        })
    }

    /// Wrap an already-running rmcp client service as an [`McpClient`].
    ///
    /// For the common case use [`Self::in_process`], which handles the duplex
    /// and handshake. This constructor is for callers (and tests) that drive
    /// `().serve(transport)` themselves — e.g. to attach a custom client
    /// handler, or to share a transport set up out-of-band. The returned client
    /// has no [`Self::reconnect`] spec (returns [`McpError::Handshake`]).
    #[must_use]
    pub fn from_service(service: RunningService<RoleClient, ()>) -> Self {
        Self {
            service: Arc::new(service),
            reconnect_spec: None,
        }
    }

    /// Build an [`McpClient`] from a running service + the spec to rebuild it.
    ///
    /// Shared tail of the three constructors: [`Self::stdio`] and
    /// [`Self::http_sse`] (via [`Self::http_connect`]) pass `Some(spec)` so
    /// [`Self::reconnect`] can rebuild the transport; [`Self::in_process`] and
    /// [`Self::from_service`] pass `None` (no rebuildable transport). Wraps the
    /// service in an [`Arc`] so each [`McpTool`](crate::mcp::McpTool) clone
    /// shares one connection cheaply.
    fn wrap(service: RunningService<RoleClient, ()>, spec: Option<ReconnectSpec>) -> Self {
        Self {
            service: Arc::new(service),
            reconnect_spec: spec,
        }
    }

    /// Bridge a `tools/call` round-trip into a loopctl [`ToolOutput`].
    ///
    /// Builds the rmcp request with the given tool `name` and, when `input` is a
    /// JSON object, attaches its fields as the call's `arguments`. rmcp's
    /// high-level `call_tool` drives SEP-2322 `input_required` rounds up to its
    /// built-in cap (10) using the local client handler — which is the pure
    /// `()` here, so it cannot actually answer elicitation; a server that never
    /// completes surfaces as `ServiceError::InputRequiredRoundsExceeded`. A
    /// protocol-level failure (RPC error, or that rounds-exceeded condition)
    /// becomes [`ToolError::Execution`]; a server-reported tool error (`isError`)
    /// is surfaced as a *soft* [`ToolOutput`] with `is_error` set (see
    /// [`bridge_result`]).
    ///
    /// # Errors
    ///
    /// [`ToolError::Execution`] if the rmcp `call_tool` RPC fails (transport,
    /// protocol `ErrorData`, or `input_required` rounds exceeded), or the result
    /// bridges to [`McpError`] (an empty tool error).
    async fn call_tool_forward(
        &self,
        server_name: &str,
        input: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let mut params = rmcp::model::CallToolRequestParams::new(server_name.to_string());
        if let serde_json::Value::Object(map) = input {
            params = params.with_arguments(map);
        }
        let result = self
            .service
            .call_tool(params)
            .await
            .map_err(|e| ToolError::Execution(format!("MCP tools/call failed: {e}")))?;
        bridge_result(server_name, result).map_err(|e| ToolError::Execution(e.to_string()))
    }

    /// Connect to an MCP server running as a child process over stdio.
    ///
    /// `command` is spawned via rmcp's [`TokioChildProcess`]
    /// (`transport-child-process` cargo-feature): it pipes the child's
    /// stdin/stdout (NDJSON JSON-RPC) and **inherits stderr** so server logs
    /// surface during development. The transport implements rmcp's `Transport`
    /// directly, so `().serve(transport)` (a pure client, handler `()`) drives
    /// the MCP `initialize` handshake before returning.
    ///
    /// **Lifecycle:** rmcp kills the child on drop via its `ChildWithCleanup`
    /// (force-kill after a 3s graceful timeout). Callers do **not** set
    /// `kill_on_drop` themselves — it is redundant with rmcp's cleanup.
    /// Dropping the returned [`McpClient`] (and all its clones) terminates the
    /// server process.
    ///
    /// **Runtime:** spawns the child and drives the handshake from the caller's
    /// async context, so this must be called from within a running tokio
    /// runtime (it panics with "no reactor running" otherwise).
    ///
    /// The [`CommandSpec`] is retained so [`Self::reconnect`] can re-spawn the
    /// same server.
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] if the child fails to spawn (`io::Error` from
    /// [`TokioChildProcess::new`]), exits before handshake, or the `initialize`
    /// round-trip fails (`ClientInitializeError`).
    pub async fn stdio(command: CommandSpec) -> Result<Self, McpError> {
        let transport = TokioChildProcess::new(command.as_tokio_command())
            .map_err(|e| McpError::Handshake(e.to_string()))?;
        let service = ().serve(transport).await.map_err(|e| McpError::Handshake(e.to_string()))?;
        Ok(Self::wrap(service, Some(ReconnectSpec::Stdio(command))))
    }

    /// Connect to a remote MCP server via Streamable HTTP, using rmcp's default
    /// HTTP client.
    ///
    /// JSON-RPC 2.0 over a single endpoint: `POST` for requests (the server
    /// replies `application/json` or `text/event-stream`), optional `GET` for a
    /// long-lived SSE notification stream, `DELETE` for session termination.
    /// Session identity is the `Mcp-Session-Id` response header, echoed back.
    ///
    /// rmcp's [`StreamableHttpClientTransport`] handles **all** of
    /// `Mcp-Session-Id` capture/echo, JSON-vs-SSE response splitting, the
    /// optional GET-opened SSE stream, DELETE-on-close, and transparent session
    /// re-init on HTTP 404 — loopctl does none of it. This constructor builds
    /// the transport via `from_uri` and runs `().serve(transport)` to handshake.
    ///
    /// The endpoint is retained so [`Self::reconnect`] can re-connect. rmcp's
    /// default client deliberately disables connection pooling
    /// (`pool_max_idle_per_host(0)`) to avoid ~40ms TCP Delayed-ACK stalls; for
    /// pooling/TLS/timeouts use [`Self::http_sse_with_client`].
    ///
    /// **Runtime:** rmcp's Streamable HTTP transport spawns a background worker
    /// task, so this must be called from within a running tokio runtime.
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] if the HTTP connection cannot be established or
    /// `initialize` fails (`ClientInitializeError`).
    pub async fn http_sse(endpoint: impl Into<Arc<str>>) -> Result<Self, McpError> {
        let endpoint = endpoint.into();
        let transport = StreamableHttpClientTransport::from_uri(Arc::clone(&endpoint));
        Self::http_connect(transport, endpoint).await
    }

    /// Connect via Streamable HTTP with a caller-supplied [`reqwest::Client`].
    ///
    /// Like [`Self::http_sse`], but the caller provides its own HTTP client —
    /// for connection pooling, custom TLS, or timeouts. Built via
    /// [`StreamableHttpClientTransport::with_client`]. Prefer the plain
    /// [`Self::http_sse`] unless you need pooling (rmcp's default disables it to
    /// avoid TCP Delayed-ACK stalls).
    ///
    /// The endpoint (not the client) is retained for [`Self::reconnect`]; a
    /// reconnect re-connects with rmcp's default client, not the originally
    /// supplied one.
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] if the HTTP connection cannot be established or
    /// `initialize` fails.
    pub async fn http_sse_with_client(
        endpoint: impl Into<Arc<str>>,
        client: reqwest::Client,
    ) -> Result<Self, McpError> {
        let endpoint = endpoint.into();
        let transport = StreamableHttpClientTransport::with_client(
            client,
            StreamableHttpClientTransportConfig::with_uri(Arc::clone(&endpoint)),
        );
        Self::http_connect(transport, endpoint).await
    }

    /// Drive `().serve(transport)` for an HTTP transport and wrap the result.
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] if the rmcp `serve`/`initialize` fails
    /// (`ClientInitializeError`).
    async fn http_connect<T, E, A>(transport: T, endpoint: Arc<str>) -> Result<Self, McpError>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ().serve(transport).await.map_err(|e| McpError::Handshake(e.to_string()))?;
        Ok(Self::wrap(service, Some(ReconnectSpec::HttpSse(endpoint))))
    }

    /// Re-establish a dropped connection using [`StreamRetryConfig`](crate::stream::handler::StreamRetryConfig)
    /// backoff.
    ///
    /// After a transient failure (child exited, HTTP 5xx, broken pipe), call
    /// this on the dead [`McpClient`]: it backs off per `retry` and re-runs the
    /// constructor that originally built this client (a spec stored on the
    /// client at construction time), returning a **new** [`McpClient`]. The old
    /// client's transport is dead; the caller re-issues `tools/list` via a fresh
    /// [`McpToolProvider`] on the returned client.
    ///
    /// Returns [`McpError::Handshake`] if this client is an [`Self::in_process`]
    /// client (no reconnect spec — an in-process server cannot be rebuilt) or
    /// after the retry config's max retries.
    ///
    /// **Reuse, don't reinvent:** loopctl already ships exponential-backoff-
    /// with-jitter retry ([`StreamRetryConfig`](crate::stream::handler::StreamRetryConfig)).
    /// This method consumes it directly — there is one retry strategy for the
    /// whole crate, not a second one for MCP. (rmcp ships no transport-level
    /// reconnect, only an in-stream SSE resume policy — verified.)
    ///
    /// # Errors
    ///
    /// [`McpError::Handshake`] for `in_process` clients, or after
    /// `retry.max_retries` failed attempts (the last error is carried). The
    /// loop always runs at least one attempt before giving up, so a returned
    /// `Handshake` carries a real connection failure, not a placeholder.
    pub async fn reconnect(
        &self,
        retry: &crate::stream::handler::StreamRetryConfig,
    ) -> Result<Self, McpError> {
        let Some(spec) = self.reconnect_spec.clone() else {
            return Err(McpError::Handshake(
                "this McpClient cannot be reconnected (in-process)".to_string(),
            ));
        };
        let mut last_err: Option<McpError> = None;
        for attempt in 0..=retry.max_retries {
            if attempt > 0 {
                let delay = retry.jittered_base_delay(attempt.saturating_sub(1));
                tokio::time::sleep(delay).await;
            }
            match spec.connect().await {
                Ok(client) => return Ok(client),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            McpError::Handshake("reconnect made no attempts (max_retries overflow)".to_string())
        }))
    }
}

/// What to spawn for [`McpClient::stdio`]. [`Clone`] so [`McpClient::reconnect`]
/// can re-establish the child.
///
/// Build with struct-literal syntax or [`CommandSpec::default`] then set the
/// fields you need; only [`program`](Self::program) is required for a working
/// spawn.
///
/// # Example
///
/// ```
/// use loopctl::mcp::CommandSpec;
///
/// let spec = CommandSpec {
///     program: "npx".into(),
///     args: vec!["-y".into(), "@modelcontextprotocol/server-everything".into()],
///     ..Default::default()
/// };
/// assert_eq!(spec.program, "npx");
/// assert_eq!(spec.args.len(), 2);
/// ```
#[derive(Clone, Debug, Default)]
pub struct CommandSpec {
    /// Executable path or name resolvable on `PATH`.
    ///
    /// Passed verbatim to [`tokio::process::Command::new`]; resolution follows
    /// the platform's usual `PATH` search. Required for a working spawn.
    pub program: String,

    /// Arguments after [`program`](Self::program).
    ///
    /// Forwarded to the child in order via `Command::args`. Empty by default.
    pub args: Vec<String>,

    /// Extra environment variables for the child.
    ///
    /// Each `(key, value)` pair is added via `Command::env` on top of the
    /// parent's environment; existing keys are overwritten. Empty by default.
    pub env: Vec<(String, String)>,

    /// Working directory, or inherit the parent's.
    ///
    /// `None` (the default) inherits the calling process's cwd; `Some(path)`
    /// sets it via `Command::current_dir`.
    pub cwd: Option<String>,
}

impl CommandSpec {
    /// Build a [`tokio::process::Command`] from this spec.
    ///
    /// Maps the four fields onto the equivalent `tokio::process::Command`
    /// calls. Does **not** set `kill_on_drop`: rmcp's `TokioChildProcess` kills
    /// the child on drop via its own cleanup guard, so a caller-set
    /// `kill_on_drop` would be redundant.
    fn as_tokio_command(&self) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new(&self.program);
        cmd.args(&self.args);
        for (key, value) in &self.env {
            cmd.env(key, value);
        }
        if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        cmd
    }
}

/// How to rebuild a connection on [`McpClient::reconnect`]. Stored on
/// [`McpClient`] by [`McpClient::stdio`] / [`McpClient::http_sse`]; `None` for
/// [`McpClient::in_process`].
#[derive(Clone, Debug)]
enum ReconnectSpec {
    /// Re-spawn this command.
    ///
    /// Carries the full [`CommandSpec`] so a reconnect reproduces the original
    /// program, arguments, environment, and working directory.
    Stdio(CommandSpec),

    /// Re-connect to this endpoint.
    ///
    /// The `Arc<str>` endpoint is reused; the reconnect uses rmcp's default
    /// HTTP client (not any caller-supplied client from the original
    /// `http_sse_with_client` call — that client is not retained).
    HttpSse(Arc<str>),
}

impl ReconnectSpec {
    /// Re-run the constructor that originally built the client.
    ///
    /// # Errors
    ///
    /// Propagates [`McpError::Handshake`] from the underlying constructor
    /// (`stdio`/`http_sse`).
    async fn connect(&self) -> Result<McpClient, McpError> {
        match self {
            Self::Stdio(command) => McpClient::stdio(command.clone()).await,
            Self::HttpSse(endpoint) => McpClient::http_sse(Arc::clone(endpoint)).await,
        }
    }
}

/// Adapts one MCP server's tools as loopctl [`Tool`] implementations.
///
/// A provider owns a connected [`McpClient`] and a snapshot of the server's
/// tool list, producing one [`McpTool`] per tool discovered via MCP's
/// `tools/list`. Each adapted tool forwards `tools/call` to the server and
/// bridges the result back into a loopctl [`ToolOutput`], so the agent loop,
/// registry, middleware pipeline, and observers see ordinary `Box<dyn Tool>`
/// values — they never learn a tool is remote.
///
/// # Construction
///
/// Build with [`McpToolProvider::connect`], which runs the MCP `initialize`
/// handshake (via the supplied [`McpClient`]) followed by `tools/list`, then
/// snapshots the result. From there either:
/// - call [`McpToolProvider::tools`] to take the adapted [`McpTool`] instances
///   and register them yourself, or
/// - call [`McpToolProvider::register_into`] to clone-and-register the whole
///   batch into a [`ToolRegistry`] in one shot.
///
/// # Tool-name collisions
///
/// MCP tools are named only within their server. If two providers each expose a
/// tool named `search`, registering both into one [`ToolRegistry`] would
/// collide — the second silently overwrites the first (the registry's own
/// behaviour). Pass a `name_prefix` to [`McpToolProvider::connect`] to
/// namespace every tool from this server: `name_prefix = Some("git".into())`
/// yields `git__status`, `git__log`, and so on. The un-prefixed name is still
/// what the provider sends to the server in each `tools/call` request, so
/// namespacing is purely a client-side registry concern.
///
/// # Static vs. dynamic discovery
///
/// [`McpToolProvider::connect`] takes a **static snapshot** of the tool list at
/// handshake time. A server may later emit `notifications/tools/list_changed`;
/// this adapter does not auto-refresh (auto-refresh would require a background
/// task per provider and a thread-safe mutable registry, neither of which the
/// current registry model supports). Call [`McpToolProvider::refresh`] to re-run
/// `tools/list` and rebuild the snapshot on demand. Tools already registered
/// into a [`ToolRegistry`] under stale names are **not** updated by a refresh —
/// the caller decides whether to re-register.
///
/// # Thread safety
///
/// `McpToolProvider` is `Send + Sync` when the underlying [`McpClient`] is (it
/// is — the rmcp handle is `Arc`-backed). The snapshot is immutable between
/// [`McpToolProvider::connect`] / [`McpToolProvider::refresh`] calls; the only
/// interior mutation is [`McpToolProvider::refresh`], which takes `&mut self`,
/// so concurrent reads of [`McpToolProvider::tools`] during a run are safe.
#[derive(Debug)]
pub struct McpToolProvider {
    /// The connected client the adapted tools forward through.
    ///
    /// Held by reference (cheaply cloneable — see [`McpClient`]); each
    /// [`McpTool`] in `tools` carries its own clone of this handle so calls are
    /// free of lifetime entanglement with the provider.
    client: McpClient,

    /// The discovered tools, frozen at the last snapshot.
    ///
    /// Populated by [`Self::connect`] and replaced wholesale by
    /// [`Self::refresh`]; never mutated in place between those calls. The slice
    /// returned by [`Self::tools`] borrows this field.
    tools: Vec<McpTool>,

    /// The prefix applied to every adapted tool name, or `None` for unprefixed.
    ///
    /// Captured at [`Self::connect`] time and re-applied by [`Self::refresh`]
    /// so a refresh preserves the original namespacing without the caller
    /// having to pass the prefix again.
    prefix: Option<String>,

    /// The per-call timeout applied to every adapted tool's `tools/call`.
    ///
    /// Bounds each forwarded call so a wedged server cannot hang the agent
    /// loop indefinitely — a call that exceeds it resolves to a *soft* error
    /// result the model can see and adapt to. Defaults to
    /// [`DEFAULT_MCP_CALL_TIMEOUT`]; overridden via
    /// [`with_call_timeout`](Self::with_call_timeout), which also updates
    /// already-discovered tools and applies to later refreshes.
    call_timeout: Duration,
}

impl McpToolProvider {
    /// Connect to a server and snapshot its tool list.
    ///
    /// The primary constructor. Runs `tools/list` against the already-connected
    /// `client` (the `initialize` handshake is *not* done here — it completed
    /// when the [`McpClient`] was built via [`McpClient::in_process`] or
    /// [`McpClient::from_service`]). rmcp auto-paginates `tools/list` by
    /// following `nextCursor` to exhaustion. The returned provider holds one
    /// [`McpTool`] per server-declared tool, each sharing a cheap clone of the
    /// client handle.
    ///
    /// `name_prefix`, when `Some`, namespaces every adapted tool: it is
    /// prepended to each tool's name as `"{prefix}__{tool_name}"` for both
    /// [`Tool::name`] and the [`ToolSchema::tool`] field sent to the LLM. The
    /// original un-prefixed name is what the provider forwards to the server in
    /// each `tools/call` request, so namespacing is purely a client-side
    /// registry concern and never confuses the server. The prefix is retained
    /// and re-applied by any later [`Self::refresh`].
    ///
    /// # Errors
    ///
    /// [`McpError::Protocol`] if the `tools/list` RPC fails (transport,
    /// protocol `ErrorData`, or pagination error). Handshake failures surface
    /// earlier, from the construction of the supplied [`McpClient`].
    pub async fn connect(client: McpClient, name_prefix: Option<String>) -> Result<Self, McpError> {
        let mut tools = Vec::new();
        bridge_tool_list(
            &client,
            name_prefix.as_deref(),
            DEFAULT_MCP_CALL_TIMEOUT,
            &mut tools,
        )
        .await?;
        Ok(Self {
            client,
            tools,
            prefix: name_prefix,
            call_timeout: DEFAULT_MCP_CALL_TIMEOUT,
        })
    }

    /// Set the per-call timeout for every adapted tool's `tools/call`.
    ///
    /// Updates both already-discovered tools and the value applied to later
    /// [`refresh`](Self::refresh) snapshots, so the knob takes effect
    /// immediately regardless of when it is called. A call that exceeds the
    /// timeout resolves to a soft error result naming the tool and the
    /// budget — the run continues and the model decides how to adapt.
    #[must_use]
    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = timeout;
        for tool in &mut self.tools {
            tool.call_timeout = timeout;
        }
        self
    }

    /// Re-run `tools/list` and rebuild the tool snapshot in place.
    ///
    /// Replaces `self.tools` wholesale with a fresh discovery, re-applying the
    /// `name_prefix` captured at [`Self::connect`] time so a refresh preserves
    /// the original namespacing without the caller re-passing it. Intended for
    /// picking up newly-added tools after a server signals
    /// `notifications/tools/list_changed`.
    ///
    /// Tools already handed to a [`ToolRegistry`] (via [`Self::register_into`])
    /// under stale names are **not** updated by this call — the registry still
    /// holds the old [`McpTool`] clones. The caller decides whether to
    /// re-register, and how to handle names that vanished from the new snapshot.
    ///
    /// # Errors
    ///
    /// [`McpError::Protocol`] if the re-list RPC fails. On error the prior
    /// snapshot is left untouched (the new list is built in a local `Vec` and
    /// only assigned on success).
    pub async fn refresh(&mut self) -> Result<(), McpError> {
        let mut tools = Vec::new();
        bridge_tool_list(
            &self.client,
            self.prefix.as_deref(),
            self.call_timeout,
            &mut tools,
        )
        .await?;
        self.tools = tools;
        Ok(())
    }

    /// The adapted tools from the current snapshot.
    ///
    /// Returns a borrow of the [`McpTool`] instances produced by the last
    /// [`Self::connect`] or [`Self::refresh`]. The slice is immutable between
    /// those calls; iterate it to register tools selectively, or use
    /// [`Self::register_into`] to register the whole batch.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = McpToolProvider::connect(client, None).await?;
    /// for tool in provider.tools() {
    ///     println!("{}: {}", tool.name(), tool.description());
    /// }
    /// ```
    #[must_use]
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    /// Clone-and-register every tool from the snapshot into `registry`.
    ///
    /// Convenience for the common "register everything" path: each [`McpTool`]
    /// is [`Clone`] (the underlying client handle is cheaply cloneable), so
    /// this hands the registry owned copies while the provider keeps its
    /// snapshot. Intra-batch duplicate names — two server tools that collide
    /// after prefixing — were already collapsed to one at [`Self::connect`]
    /// time, so the registry sees unique names; an overlap with a tool already
    /// in the registry follows [`ToolRegistry::register`]'s own overwrite-with
    /// -`warn` behaviour.
    ///
    /// Use [`Self::tools`] instead when you need finer control over which tools
    /// to register.
    pub fn register_into(&self, registry: &mut ToolRegistry) {
        for tool in &self.tools {
            registry.register(tool.clone());
        }
    }

    /// Borrow the underlying client.
    ///
    /// Exposed for advanced callers — e.g. to issue raw rmcp requests the
    /// adapter does not wrap, or to feed the same live connection into other
    /// machinery. Future transport constructors will also build on this seam.
    #[must_use]
    pub fn client(&self) -> &McpClient {
        &self.client
    }
}

/// A single MCP server tool exposed as a loopctl [`Tool`].
///
/// One `McpTool` adapts a single server-declared tool — captured at discovery
/// time — as an ordinary loopctl [`Tool`] that forwards `tools/call` to the
/// server and bridges the result back into [`ToolOutput`]. Produced by
/// [`McpToolProvider::connect`] (and rebuilt by [`McpToolProvider::refresh`]);
/// not constructed directly by callers.
///
/// The adapter stores a snapshot of the server-side `name`, `description`,
/// `inputSchema` (and optional `outputSchema`) plus a cheap clone of the shared
/// [`McpClient`] handle, so a call has everything it needs without borrowing
/// from the provider. See [`Tool::call`] for the round-trip details.
///
/// # Concurrency
///
/// [`Clone`] is cheap: it clones the [`Arc`]-backed client handle and the small
/// schema snapshot, nothing more. Each clone drives the same underlying
/// connection, so the adapter conservatively reports
/// [`Tool::is_concurrency_safe`] as `false` — see that method for why.
///
/// # Annotations
///
/// The server's `annotations` block is distilled into two booleans carried on
/// the struct and surfaced via [`Tool::is_read_only`] and
/// [`McpTool::is_destructive_hint`]. Both apply the MCP-spec defaults for an
/// absent hint (read-only defaults to `false`; destructive defaults to `true`),
/// so a consumer can read them directly without re-applying the spec.
#[derive(Clone, Debug)]
pub struct McpTool {
    /// The original server-side name, sent verbatim in each `tools/call`.
    ///
    /// Distinct from `exposed_name` when the provider was constructed with a
    /// `name_prefix`: the prefix is a client-side registry concern only, so the
    /// server always sees the name it declared.
    server_name: String,

    /// The loopctl-facing name (prefixed if the provider was given a prefix).
    ///
    /// Returned by [`Tool::name`] and used as [`ToolSchema::tool`] for the LLM.
    /// Equals `server_name` when no prefix was supplied.
    exposed_name: String,

    /// Human-readable description copied from the server tool at discovery.
    ///
    /// Empty string when the server omitted a description. Returned verbatim by
    /// [`Tool::description`] and embedded in [`ToolSchema::description`].
    description: String,

    /// The server's `inputSchema`, carried verbatim as a JSON value.
    ///
    /// MCP permits any JSON-Schema draft; the adapter does not normalize it
    /// (loopctl forwards the schema to the LLM and never validates against it).
    /// Embedded unchanged in [`ToolSchema::input_schema`].
    input_schema: serde_json::Value,

    /// The server's `outputSchema`, if it declared one.
    ///
    /// Carried for forward-compatibility only — the adapter does not validate
    /// call results against it. Exposed via [`McpTool::output_schema`].
    output_schema: Option<serde_json::Value>,

    /// A cheap clone of the shared client handle.
    ///
    /// Cloned per-tool at discovery so each [`Tool::call`] owns its connection
    /// without borrowing from the provider (the future is `'_`-bounded but
    /// self-contained).
    client: McpClient,

    /// Whether the server annotated the tool as read-only.
    ///
    /// Mirrors `annotations.readOnlyHint` when present, else `false` (the
    /// spec default). Drives [`Tool::is_read_only`].
    read_only_hint: bool,

    /// Whether the tool should be treated as destructive.
    ///
    /// Mirrors `annotations.destructiveHint` when present, else `true` — the
    /// spec default for an absent hint is "assume destructive" (the opposite
    /// polarity from `read_only_hint`). Exposed via
    /// [`McpTool::is_destructive_hint`] for a future permission gate.
    destructive_hint: bool,

    /// The per-call timeout for this tool's `tools/call` round-trip.
    ///
    /// Copied from the provider at discovery (or updated by
    /// [`with_call_timeout`](McpToolProvider::with_call_timeout)). A call that
    /// exceeds it resolves to a soft error result rather than hanging the
    /// agent loop on a wedged server.
    call_timeout: Duration,
}

impl McpTool {
    /// The server's `outputSchema`, if it declared one.
    ///
    /// Carried verbatim for forward-compatibility; the adapter does not validate
    /// call results against it. A future release may enforce it.
    #[must_use]
    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        self.output_schema.as_ref()
    }

    /// Whether the tool should be treated as destructive.
    ///
    /// Mirrors `annotations.destructiveHint` when the server set it, and applies
    /// the MCP-spec default for an absent hint: **absent means destructive**
    /// (`true`). This is the opposite polarity from
    /// [`is_read_only`](McpTool::is_read_only), whose absent hint defaults to
    /// non-destructive (`false`). A permission gate can read this directly
    /// without re-applying the spec default.
    #[must_use]
    pub fn is_destructive_hint(&self) -> bool {
        self.destructive_hint
    }
}

impl Tool for McpTool {
    /// The loopctl-facing (possibly prefixed) tool name.
    ///
    /// Returns `exposed_name`, which equals the server-declared name when the
    /// provider was constructed without a `name_prefix`, or
    /// `"{prefix}__{name}"` otherwise. The LLM and the [`ToolRegistry`] see
    /// this value; the server sees the un-prefixed `server_name` (see
    /// [`Tool::call`]).
    fn name(&self) -> &str {
        &self.exposed_name
    }

    /// The server-supplied description (empty string if the server omitted it).
    ///
    /// Copied verbatim from the server tool at discovery time and embedded in
    /// [`ToolSchema::description`] for the LLM.
    fn description(&self) -> &str {
        &self.description
    }

    /// Build the [`ToolSchema`] sent to the LLM for this tool.
    ///
    /// Carries the (possibly prefixed) name, the description, and the server's
    /// `inputSchema` verbatim — no normalization, no draft rewriting. Built
    /// fresh on each call so a caller can mutate the snapshot without affecting
    /// previously-returned schemas.
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.exposed_name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    /// Forward a `tools/call` round-trip to the server and bridge the result.
    ///
    /// Sends the `input` (when it is a JSON object) as the call's `arguments`
    /// under the **un-prefixed** `server_name`, so the server always sees the
    /// name it declared regardless of any client-side namespacing. The cheap
    /// client handle is cloned before the future is constructed so the future
    /// owns its connection and is bounded only by `'_` (no borrow of `self`
    /// survives the `await`). The result-bridging and error-mapping rules live
    /// on [`McpClient`].
    ///
    /// # Errors
    ///
    /// [`ToolError::Execution`] for any protocol failure, transport error, or
    /// server-reported empty error — mapped at the [`McpClient`] boundary. A
    /// server-reported tool error (`isError: true`) with content is surfaced as
    /// a *soft* [`ToolOutput`] with `is_error` set, not as an `Err`. A call
    /// that exceeds the tool's [`call_timeout`](McpTool) likewise resolves to
    /// a soft error result naming the tool and the budget, so a wedged server
    /// costs one tool result instead of the whole run.
    fn call(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let client = self.client.clone();
        let server_name = self.server_name.clone();
        let exposed_name = self.exposed_name.clone();
        let call_timeout = self.call_timeout;
        Box::pin(async move {
            match tokio::time::timeout(call_timeout, client.call_tool_forward(&server_name, input))
                .await
            {
                Ok(result) => result,
                Err(_) => Ok(ToolOutput::error_text(format!(
                    "MCP tool '{exposed_name}' timed out after {call_timeout:?} without a response"
                ))),
            }
        })
    }

    /// Conservatively `false` for every MCP tool.
    ///
    /// A remote server may serialize calls internally, mutate shared state per
    /// call, or rate-limit, and loopctl cannot tell. The parallel dispatcher
    /// therefore never overlaps calls into the same server unless a caller that
    /// trusts a specific server overrides this. The MCP `annotations` block
    /// carries no concurrency hint in the current spec, so there is nothing
    /// finer to honour today.
    fn is_concurrency_safe(&self) -> bool {
        false
    }

    /// Honours the server's `annotations.readOnlyHint`, defaulting to `false`.
    ///
    /// Returns `true` only when the server explicitly annotated the tool as
    /// read-only; an absent hint follows the [`Tool`] trait default of `false`
    /// (matching the MCP spec). A permission gate may auto-approve tools for
    /// which this returns `true`.
    fn is_read_only(&self) -> bool {
        self.read_only_hint
    }
}

/// Errors from the MCP client adapter.
///
/// All variants carry the underlying rmcp failure as a string, so the type is
/// cheap, `Send + 'static`, and stable across rmcp version bumps (rmcp's own
/// error types are not uniformly `Send + 'static`).
///
/// # Example
///
/// ```
/// use loopctl::mcp::McpError;
///
/// let err = McpError::Protocol("unknown tool 'search'".into());
/// assert!(err.to_string().contains("unknown tool"));
/// assert!(format!("{}", McpError::Handshake("connect refused".into()))
///     .contains("handshake/transport error"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The `initialize` handshake or underlying transport failed before any
    /// tools could be listed. Carries the rmcp service/transport error as a
    /// string (the rmcp error types are not uniformly `Send + 'static`; this
    /// keeps [`McpError`] cheap and stable across rmcp version bumps).
    #[error("MCP handshake/transport error: {0}")]
    Handshake(String),

    /// A JSON-RPC-level protocol error from the server, e.g. `tools/list`
    /// rejected or `tools/call` for an unknown tool.
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    /// The server returned a tool result with `isError = true` and no textual
    /// content to surface. (When there *is* text content, the bridge returns a
    /// soft-error [`ToolOutput`] instead of raising this.)
    #[error("MCP tool '{0}' reported an error with no content")]
    EmptyToolError(String),
}

/// Discover the server's tools and append one [`McpTool`] per result.
///
/// The single home for `tools/list` pagination: rmcp's `list_all_tools` follows
/// `nextCursor` to exhaustion, so if a future rmcp version drops that helper
/// only this function changes.
///
/// # Errors
///
/// [`McpError::Protocol`] if the `list_all_tools` RPC fails.
async fn bridge_tool_list(
    client: &McpClient,
    prefix: Option<&str>,
    call_timeout: Duration,
    out: &mut Vec<McpTool>,
) -> Result<(), McpError> {
    let server_tools = client
        .service
        .list_all_tools()
        .await
        .map_err(|e| McpError::Protocol(e.to_string()))?;
    let mut seen = std::collections::HashSet::new();
    for server_tool in server_tools {
        let Some(adapted) = bridge_tool(&server_tool, prefix, client, call_timeout) else {
            continue;
        };
        if !seen.insert(adapted.exposed_name.clone()) {
            tracing::warn!(
                tool = %adapted.exposed_name,
                "duplicate MCP tool name after prefixing; keeping the first"
            );
            continue;
        }
        out.push(adapted);
    }
    Ok(())
}

/// Build one [`McpTool`] from a server-declared tool.
///
/// The per-tool half of discovery (the list-driving half is [`bridge_tool_list`]).
/// Copies the server's `name`, `description`, `inputSchema`, and optional
/// `outputSchema` into a fresh [`McpTool`], hands the tool a cheap clone of the
/// shared `client` handle, and distils the server's `annotations` block into the
/// two booleans the adapter carries.
///
/// # Naming
///
/// `prefix`, when `Some`, yields an `exposed_name` of `"{prefix}__{name}"`
/// (loopctl-facing, sent to the LLM and used as the registry key) while
/// `server_name` keeps the original value the server declared — the latter is
/// what `tools/call` forwards, so namespacing is purely a client-side concern.
///
/// # Schema fidelity
///
/// `inputSchema` and `outputSchema` are coerced from rmcp's `Arc<JsonObject>`
/// into `serde_json::Value::Object(..)` and carried **verbatim** — no draft
/// normalization, no field rewriting. loopctl forwards the schema to the LLM and
/// never validates against it, so any JSON-Schema draft passes through
/// losslessly.
///
/// # Annotations
///
/// `annotations` is read with the MCP-spec defaults for an absent hint:
/// `readOnlyHint` defaults to `false`, `destructiveHint` defaults to `true`
/// (the opposite polarity — an unannotated tool is assumed destructive). An
/// entirely absent `annotations` block yields `(false, true)`.
///
/// # Returns
///
/// `None` only for a malformed discovery entry whose `name` is empty — the
/// caller ([`bridge_tool_list`]) skips `None` rather than panicking, since the
/// no-panic lint forbids indexing/`unwrap` and an empty name is a server bug
/// worth dropping silently with a warning rather than aborting discovery.
fn bridge_tool(
    server_tool: &rmcp::model::Tool,
    prefix: Option<&str>,
    client: &McpClient,
    call_timeout: Duration,
) -> Option<McpTool> {
    let server_name = server_tool.name.to_string();
    if server_name.is_empty() {
        tracing::warn!("MCP server declared a tool with an empty name; skipping");
        return None;
    }
    let exposed_name =
        prefix.map_or_else(|| server_name.clone(), |p| format!("{p}__{server_name}"));
    let description = server_tool
        .description
        .as_deref()
        .unwrap_or_default()
        .to_string();
    let input_schema = serde_json::Value::Object(server_tool.input_schema.as_ref().clone());
    let output_schema = server_tool
        .output_schema
        .as_ref()
        .map(|schema| serde_json::Value::Object(schema.as_ref().clone()));
    let (read_only_hint, destructive_hint) =
        server_tool
            .annotations
            .as_ref()
            .map_or((false, true), |annotations| {
                (
                    annotations.read_only_hint.unwrap_or(false),
                    // MCP spec: an absent destructiveHint means "assume
                    // destructive" (default true) — the opposite of read-only.
                    annotations.destructive_hint.unwrap_or(true),
                )
            });
    Some(McpTool {
        server_name,
        exposed_name,
        description,
        input_schema,
        output_schema,
        client: client.clone(),
        read_only_hint,
        destructive_hint,
        call_timeout,
    })
}

/// Bridge a rmcp `CallToolResult` into a loopctl [`ToolOutput`].
///
/// Maps the result's content blocks into [`MessageToolContent`]: a single text
/// block becomes [`MessageToolContent::Text`]; any other shape (multiple blocks,
/// an image) becomes [`MessageToolContent::Multipart`]. `isError` becomes
/// [`ToolOutput::is_error`] — a server-reported tool error is a *soft* failure,
/// matching how native loopctl tools report recoverable errors. `structuredContent`,
/// when present, is appended as one extra JSON-stringified text part (carried,
/// not parsed). An error with no content at all becomes
/// [`McpError::EmptyToolError`] carrying `tool_name`.
///
/// A successful result with zero content blocks yields an empty successful
/// [`ToolOutput`] (no error) — the server ran the tool and returned nothing.
///
/// # Errors
///
/// [`McpError::EmptyToolError`] when the server reported an error but supplied
/// no content to surface.
fn bridge_result(
    tool_name: &str,
    res: rmcp::model::CallToolResult,
) -> Result<ToolOutput, McpError> {
    let is_error = res.is_error.unwrap_or(false);
    let mut parts: Vec<ToolContentPart> = res.content.iter().map(bridge_content).collect();
    if let Some(structured) = res.structured_content {
        let note = serde_json::to_string(&structured)
            .unwrap_or_else(|_| "<unserializable structuredContent>".to_string());
        parts.push(ToolContentPart::text(note));
    }
    if parts.is_empty() {
        return if is_error {
            Err(McpError::EmptyToolError(tool_name.to_string()))
        } else {
            Ok(ToolOutput::text(String::new()))
        };
    }
    let payload = if parts.len() == 1 {
        match parts.pop() {
            Some(ToolContentPart::Text { text }) => MessageToolContent::Text(text),
            Some(single) => MessageToolContent::Multipart(vec![single]),
            None => MessageToolContent::Text(String::new()),
        }
    } else {
        MessageToolContent::Multipart(parts)
    };
    let output = if is_error {
        ToolOutput::error(payload)
    } else {
        ToolOutput::success(payload)
    };
    Ok(output)
}

/// Map one rmcp content block to a loopctl [`ToolContentPart`].
///
/// Text and image carry through. Audio falls back to a short text note. Embedded
/// resources and resource links are stringified with their identifying payload
/// (uri, text/name) so the model sees *what* the server returned, not just that
/// it returned something. Any future block kind falls back to a generic note.
fn bridge_content(block: &ContentBlock) -> ToolContentPart {
    match block {
        ContentBlock::Text(text) => ToolContentPart::text(&text.text),
        ContentBlock::Image(image) => ToolContentPart::image(
            crate::message::ImageSource::new_base64(&image.mime_type, &image.data),
        ),
        ContentBlock::Resource(resource) => match &resource.resource {
            rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                ToolContentPart::text(format!("MCP resource {uri}: {text}"))
            }
            rmcp::model::ResourceContents::BlobResourceContents { uri, .. } => {
                ToolContentPart::text(format!("MCP resource {uri}: (blob)"))
            }
            _ => ToolContentPart::text("unsupported MCP content type: embedded resource"),
        },
        ContentBlock::ResourceLink(link) => {
            ToolContentPart::text(format!("MCP resource link: {} ({})", link.name, link.uri))
        }
        _ => ToolContentPart::text("unsupported MCP content type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolResult, ContentBlock};

    #[test]
    fn bridge_result_single_text_becomes_text_payload() {
        let res = CallToolResult::success(vec![ContentBlock::text("hi")]);
        let out = bridge_result("t", res).expect("success bridges");
        assert!(!out.is_error);
        assert!(matches!(out.payload, MessageToolContent::Text(_)));
        assert_eq!(out.text_content(), "hi");
    }

    #[test]
    fn bridge_result_single_image_becomes_single_part_multipart() {
        let res = CallToolResult::success(vec![ContentBlock::image("Zm9v", "image/png")]);
        let out = bridge_result("t", res).expect("success bridges");
        assert!(!out.is_error);
        match out.payload {
            MessageToolContent::Multipart(parts) => {
                assert_eq!(parts.len(), 1, "single image → one-element multipart");
                assert!(matches!(parts.first(), Some(ToolContentPart::Image { .. })));
            }
            other @ MessageToolContent::Text(_) => {
                panic!("expected Multipart, got {other:?}")
            }
        }
    }

    #[test]
    fn bridge_result_multiple_blocks_become_multipart_in_order() {
        let res = CallToolResult::success(vec![
            ContentBlock::text("a"),
            ContentBlock::image("Zg==", "image/jpeg"),
            ContentBlock::text("b"),
        ]);
        let out = bridge_result("t", res).expect("success bridges");
        let MessageToolContent::Multipart(parts) = out.payload else {
            panic!("expected Multipart");
        };
        assert_eq!(parts.len(), 3);
        assert!(matches!(parts.first(), Some(ToolContentPart::Text { text }) if text == "a"));
        assert!(matches!(parts.get(1), Some(ToolContentPart::Image { .. })));
        assert!(matches!(parts.get(2), Some(ToolContentPart::Text { text }) if text == "b"));
    }

    #[test]
    fn bridge_result_soft_error_returns_ok_with_is_error() {
        let res = CallToolResult::error(vec![ContentBlock::text("boom")]);
        let out = bridge_result("t", res).expect("soft error is Ok");
        assert!(out.is_error);
        assert_eq!(out.text_content(), "boom");
    }

    #[test]
    fn bridge_result_empty_error_is_hard_empty_tool_error_with_name() {
        let res = CallToolResult::error(vec![]);
        let err = bridge_result("search", res).expect_err("empty error is hard Err");
        match err {
            McpError::EmptyToolError(name) => assert_eq!(name, "search"),
            other => panic!("expected EmptyToolError, got {other:?}"),
        }
    }

    #[test]
    fn bridge_result_empty_success_yields_empty_text_output() {
        let res = CallToolResult::success(vec![]);
        let out = bridge_result("t", res).expect("empty success is Ok");
        assert!(!out.is_error);
        assert_eq!(out.text_content(), "");
    }

    #[test]
    fn bridge_result_structured_content_appended_as_text_part() {
        let mut res = CallToolResult::success(vec![ContentBlock::text("body")]);
        res.structured_content = Some(serde_json::json!({"count": 7}));
        let out = bridge_result("t", res).expect("success bridges");
        let MessageToolContent::Multipart(parts) = out.payload else {
            panic!("text + structured must be multipart");
        };
        assert_eq!(parts.len(), 2);
        // The structured part is JSON-stringified text appended after the body.
        let structured_text = &parts
            .last()
            .and_then(|p| match p {
                ToolContentPart::Text { text } => Some(text.as_str()),
                ToolContentPart::Image { .. } => None,
            })
            .expect("structured part is text");
        assert!(
            structured_text.contains("count"),
            "carries the structured json"
        );
        assert!(structured_text.contains('7'));
    }

    #[test]
    fn bridge_result_error_with_structured_content_is_soft_error() {
        let mut res = CallToolResult::error(vec![]);
        res.structured_content = Some(serde_json::json!({"reason": "denied"}));
        let out = bridge_result("search", res).expect("structured error is soft Ok");
        assert!(out.is_error, "is_error flag set");
        let text = out.text_content();
        assert!(
            text.contains("denied"),
            "structured payload surfaces as the error text: {text}"
        );
    }

    #[test]
    fn bridge_content_unsupported_kinds_surface_as_text_notes() {
        // Audio carries no payload loopctl can render, so it falls through to
        // the generic unsupported-note arm (the `_` fallback). It must still
        // surface as a Text part — never silently dropped.
        let audio = bridge_content(&ContentBlock::audio("AAAA", "audio/wav"));
        assert!(
            matches!(audio, ToolContentPart::Text { .. }),
            "audio → text note (not dropped), got {audio:?}"
        );

        let resource = bridge_content(&ContentBlock::Resource(rmcp::model::EmbeddedResource::new(
            rmcp::model::ResourceContents::text("body", "mem://x"),
        )));
        assert!(
            matches!(resource, ToolContentPart::Text { ref text } if text.contains("mem://x") && text.contains("body")),
            "embedded text resource surfaces its uri and text, got {resource:?}"
        );

        let link = rmcp::model::Resource::new("file:///a", "thing");
        let link_part = bridge_content(&ContentBlock::ResourceLink(link));
        assert!(
            matches!(link_part, ToolContentPart::Text { ref text } if text.contains("thing") && text.contains("file:///a")),
            "resource link surfaces name and uri, got {link_part:?}"
        );
    }

    #[test]
    fn bridge_content_text_and_image_carry_through() {
        let text = bridge_content(&ContentBlock::text("hello"));
        assert!(
            matches!(&text, ToolContentPart::Text { text } if text == "hello"),
            "got {text:?}"
        );
        let image = bridge_content(&ContentBlock::image("Zm9v", "image/png"));
        match image {
            ToolContentPart::Image { source } => {
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "Zm9v");
            }
            other @ ToolContentPart::Text { .. } => {
                panic!("expected Image, got {other:?}")
            }
        }
    }
}
