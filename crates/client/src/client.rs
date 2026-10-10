//! The daemon client: connect, spawn, and the typed verb layer.
//!
//! [`DaemonClient`] is the socket-side implementation of the
//! [`LoopctlClient`](crate::LoopctlClient) surface: it performs the
//! versioned handshake (refusing a family member outside the
//! negotiated floor), optionally spawns the daemon when the socket is
//! absent (a bounded, visible retry — the host supplies the spawn
//! action), and speaks the typed verbs over one generic request path.

use std::fmt::Write as _;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;

use crate::ClientError;
use crate::GatePending;
use crate::LoopSummary;
use crate::LoopctlClient;
use crate::RunHandle;
use crate::RunState;
use crate::ScheduleSummary;
use crate::events::EventStream;
use crate::http::HttpRequest;
use crate::http::open;
use crate::http::send;

/// The family name every handshake participant must carry.
///
/// The handshake refuses any peer whose family differs — the
/// socket belongs to this product or to nobody.
const FAMILY: &str = "loopctl";

/// The protocol version this client speaks.
///
/// Bumps only with an additive-but-distinguishable change; the
/// handshake floor keeps old pairs honest across a mismatch.
const PROTOCOL_VERSION: u64 = 1;

/// The oldest server this client will talk to.
///
/// A server reporting a lower version is refused; so is a server
/// whose own floor is above this client's version — a mismatch is
/// never silently absorbed in either direction.
const PROTOCOL_FLOOR: u64 = 1;

/// The handshake document `GET /v1/hello` answers with.
///
/// The negotiation triple every family member leads with.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Hello {
    /// The family name, e.g. `loopctl`.
    ///
    /// Must match this client's family exactly.
    family: String,

    /// The server's protocol version.
    ///
    /// Compared against this client's floor from below.
    protocol: u64,

    /// The oldest client the server will talk to.
    ///
    /// Compared against this client's version from above — the
    /// refusal direction old clients need to survive.
    floor: u64,
}

/// How a [`DaemonClient`] may reach the daemon.
///
/// The socket, the spawn policy for an absent socket, and the
/// bounded window a spawn waits inside.
#[derive(Debug, Clone)]
pub struct ConnectOptions {
    /// The daemon's unix socket path.
    ///
    /// A missing file is the typed absence, not a generic IO
    /// error — the degrade logic keys on the distinction.
    pub socket_path: PathBuf,

    /// Whether and how to spawn a daemon when the socket is absent.
    ///
    /// [`Never`](SpawnPolicy::Never) keeps the constructor pure;
    /// an action carries the host's spawn recipe.
    pub spawn: SpawnPolicy,

    /// The bounded retry window an auto-spawn waits inside.
    ///
    /// Applies only under [`SpawnPolicy::Spawn`].
    pub retry: RetryWindow,
}

impl ConnectOptions {
    /// Options for one socket, no spawning.
    ///
    /// The pure embedder entry: an absent socket is the typed
    /// [`SocketAbsent`](ClientError::SocketAbsent) the host's
    /// in-process degrade keys on.
    #[must_use]
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            spawn: SpawnPolicy::Never,
            retry: RetryWindow::default(),
        }
    }

    /// Attach a spawn action and retry window, chaining.
    ///
    /// The action runs once per connect; the window bounds how long
    /// the handshake retries after it.
    #[must_use]
    pub fn with_spawn(mut self, action: SpawnAction, retry: RetryWindow) -> Self {
        self.spawn = SpawnPolicy::Spawn(action);
        self.retry = retry;
        self
    }
}

/// The host-supplied daemon spawn action.
///
/// The CLI forks its own binary detached; a test forks a mock. The
/// action is a plain sync closure because spawning a detached process
/// is a fork-and-forget — the client's retry window, not the action,
/// waits for readiness.
pub type SpawnAction = Arc<dyn Fn() -> Result<(), std::io::Error> + Send + Sync>;

/// Whether an absent socket triggers a daemon spawn.
///
/// The embedder's choice: a pure probe or the CLI's fork-and-wait.
#[derive(Clone)]
pub enum SpawnPolicy {
    /// Never spawn — surface the typed absence.
    ///
    /// The default; tests and embedders get the clean signal.
    Never,

    /// Run the action, then retry the connect inside the window.
    ///
    /// One action run, bounded retries, a typed timeout on
    /// failure — never an unbounded wait.
    Spawn(SpawnAction),
}

impl std::fmt::Debug for SpawnPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Never => formatter.write_str("Never"),
            Self::Spawn(_) => formatter.write_str("Spawn(<action>)"),
        }
    }
}

/// The bounded window an auto-spawn waits for the socket in.
///
/// The wait spans the delays between attempts — the first attempt
/// follows the spawn immediately, so `attempts` tries are separated
/// by `attempts − 1` delays and the reported span counts exactly
/// those; the default matches the family's bounded-retry guidance
/// (~5 s).
#[derive(Debug, Clone, Copy)]
pub struct RetryWindow {
    /// How many connect attempts the window spans.
    ///
    /// The first attempt follows the spawn immediately; the delay
    /// separates the rest.
    pub attempts: u16,

    /// The delay between attempts.
    ///
    /// Short — daemon startup dominates, not the poll cadence.
    pub delay: Duration,
}

impl Default for RetryWindow {
    fn default() -> Self {
        Self {
            attempts: 50,
            delay: Duration::from_millis(100),
        }
    }
}

/// Negotiate the handshake document against this client's constants.
///
/// # Errors
///
/// Refuses a foreign family, a server below this client's floor, and
/// a server whose own floor is above this client's version — each
/// naming both sides.
fn negotiate(hello: &Hello) -> Result<(), ClientError> {
    if hello.family != FAMILY {
        return Err(ClientError::FamilyMismatch {
            daemon: hello.family.clone(),
            client: FAMILY.to_string(),
        });
    }
    if hello.protocol < PROTOCOL_FLOOR {
        return Err(ClientError::ServerTooOld {
            daemon_protocol: hello.protocol,
            client_floor: PROTOCOL_FLOOR,
        });
    }
    if hello.floor > PROTOCOL_VERSION {
        return Err(ClientError::ServerTooNew {
            daemon_floor: hello.floor,
            client_protocol: PROTOCOL_VERSION,
        });
    }
    Ok(())
}

/// The socket-side client.
///
/// Constructed through [`connect`](DaemonClient::connect) (or
/// [`connect_default`](DaemonClient::connect_default), which resolves
/// the standard socket path and honors `LOOPCTL_NO_AUTOSPAWN`); every
/// verb is one HTTP request over the socket, and the event stream is
/// one long-lived SSE connection.
#[derive(Debug, Clone)]
pub struct DaemonClient {
    /// The daemon's socket path every verb addresses.
    ///
    /// Kept from [`ConnectOptions`] at construction so each request
    /// reconnects by path — the connection-per-request shape.
    socket: PathBuf,

    /// The negotiated handshake snapshot.
    ///
    /// What the daemon answered at connect: its family, protocol
    /// version, and floor. Kept for [`handshake`](Self::handshake)
    /// and never re-negotiated within one client's lifetime.
    hello: Hello,
}

impl DaemonClient {
    /// Connect to the daemon under `options`, performing the
    /// handshake.
    ///
    /// With [`SpawnPolicy::Spawn`], an absent socket runs the spawn
    /// action and retries inside the window — a daemon that never
    /// appears surfaces as the window's typed timeout naming what was
    /// tried.
    /// # Errors
    ///
    /// [`ClientError::SocketAbsent`] when spawning is off and no
    /// daemon listens; [`ClientError::SpawnFailed`] and
    /// [`ClientError::ConnectTimeout`] for the spawn path; transport,
    /// malformed-response, and the three handshake refusals
    /// (`NotADaemon`, family mismatch, floor violation) otherwise.
    pub async fn connect(options: ConnectOptions) -> Result<Self, ClientError> {
        if let Some(client) = Self::try_handshake(&options.socket_path).await? {
            return Ok(client);
        }
        let SpawnPolicy::Spawn(action) = &options.spawn else {
            return Err(ClientError::SocketAbsent);
        };
        action().map_err(|source| ClientError::SpawnFailed {
            detail: source.to_string(),
        })?;
        for attempt in 0..options.retry.attempts {
            if attempt > 0 {
                tokio::time::sleep(options.retry.delay).await;
            }
            if let Some(client) = Self::try_handshake(&options.socket_path).await? {
                return Ok(client);
            }
        }
        Err(ClientError::ConnectTimeout {
            window_ms: u64::from(options.retry.attempts.saturating_sub(1)).saturating_mul(
                options
                    .retry
                    .delay
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
            ),
        })
    }

    /// Connect on the standard per-user socket.
    ///
    /// The socket path resolves from `$XDG_RUNTIME_DIR/loopctl/loopctl.sock`;
    /// a spawn action may be attached and is suppressed entirely when
    /// `LOOPCTL_NO_AUTOSPAWN` is set to `1`, mirroring the CLI's
    /// opt-out.
    ///
    /// # Errors
    ///
    /// [`ClientError::SocketPathUnavailable`] when the runtime dir is
    /// unset; otherwise the [`DaemonClient::connect`] set.
    pub async fn connect_default(spawn: Option<SpawnAction>) -> Result<Self, ClientError> {
        let runtime_dir =
            std::env::var("XDG_RUNTIME_DIR").map_err(|_| ClientError::SocketPathUnavailable)?;
        let mut options =
            ConnectOptions::new(Path::new(&runtime_dir).join("loopctl").join("loopctl.sock"));
        if let Some(action) = spawn
            && std::env::var("LOOPCTL_NO_AUTOSPAWN").as_deref() != Ok("1")
        {
            options = options.with_spawn(action, RetryWindow::default());
        }
        Self::connect(options).await
    }

    /// One handshake attempt.
    ///
    /// `Ok(None)` means the socket is absent right now — the caller
    /// decides whether that is terminal or a spawn-and-retry signal.
    ///
    /// # Errors
    ///
    /// Propagates transport and malformed-response failures; refuses
    /// non-family or floor-violating peers through [`negotiate`].
    async fn try_handshake(socket: &Path) -> Result<Option<Self>, ClientError> {
        let request = HttpRequest {
            method: "GET",
            path: "/v1/hello".to_string(),
            body: None,
        };
        let response = match send(socket, request).await {
            Ok(response) => response,
            Err(ClientError::SocketAbsent) => return Ok(None),
            Err(error) => return Err(error),
        };
        if response.status == 404 {
            return Err(ClientError::NotADaemon {
                path: socket.to_path_buf(),
            });
        }
        if !status_ok(response.status) {
            return Err(map_api_error(response.status, &response.body));
        }
        let hello: Hello = decode_json(&response.body)?;
        negotiate(&hello)?;
        Ok(Some(Self {
            socket: socket.to_path_buf(),
            hello,
        }))
    }

    /// The negotiated handshake snapshot.
    ///
    /// What the connected daemon reported — family, protocol, and
    /// its floor — for diagnostics and version display.
    #[must_use]
    pub fn handshake(&self) -> (String, u64, u64) {
        (
            self.hello.family.clone(),
            self.hello.protocol,
            self.hello.floor,
        )
    }

    fn request_json<T>(
        &self,
        method: &'static str,
        path: String,
        body: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<T, ClientError>> + Send + '_>>
    where
        T: serde::de::DeserializeOwned + Send + 'static,
    {
        let socket = self.socket.clone();
        Box::pin(async move {
            let response = send(&socket, HttpRequest { method, path, body }).await?;
            if response.status == 404 {
                return Err(ClientError::NotFound {
                    what: path_of(&response.body),
                });
            }
            if !status_ok(response.status) {
                return Err(map_api_error(response.status, &response.body));
            }
            decode_json::<T>(&response.body)
        })
    }

    fn empty(
        &self,
        method: &'static str,
        path: String,
        body: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        let socket = self.socket.clone();
        Box::pin(async move {
            let response = send(&socket, HttpRequest { method, path, body }).await?;
            if response.status == 404 {
                return Err(ClientError::NotFound {
                    what: path_of(&response.body),
                });
            }
            if !status_ok(response.status) {
                return Err(map_api_error(response.status, &response.body));
            }
            Ok(())
        })
    }
}

impl LoopctlClient for DaemonClient {
    fn list_loops(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<LoopSummary>, ClientError>> + Send + '_>> {
        self.request_json("GET", "/v1/loops".to_string(), None)
    }

    fn get_loop(
        &self,
        loop_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LoopSummary, ClientError>> + Send + '_>> {
        self.request_json(
            "GET",
            format!("/v1/loops/{}", percent_encoded(loop_id)),
            None,
        )
    }

    fn start_run(
        &self,
        loop_id: &str,
        input: &str,
    ) -> Pin<Box<dyn Future<Output = Result<RunHandle, ClientError>> + Send + '_>> {
        self.request_json(
            "POST",
            format!("/v1/loops/{}/runs", percent_encoded(loop_id)),
            Some(serde_json::json!({ "input": input })),
        )
    }

    fn get_run(
        &self,
        run_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<RunState, ClientError>> + Send + '_>> {
        self.request_json("GET", format!("/v1/runs/{run_id}"), None)
    }

    fn stop_run(
        &self,
        run_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        self.empty("POST", format!("/v1/runs/{run_id}/stop"), None)
    }

    fn pending_gates(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<GatePending>, ClientError>> + Send + '_>> {
        self.request_json("GET", "/v1/gates".to_string(), None)
    }

    fn approve_gate(
        &self,
        gate_id: &str,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        self.empty(
            "POST",
            format!("/v1/gates/{}/approve", percent_encoded(gate_id)),
            Some(serde_json::json!({ "reason": reason })),
        )
    }

    fn deny_gate(
        &self,
        gate_id: &str,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        self.empty(
            "POST",
            format!("/v1/gates/{}/deny", percent_encoded(gate_id)),
            Some(serde_json::json!({ "reason": reason })),
        )
    }

    fn list_schedules(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScheduleSummary>, ClientError>> + Send + '_>> {
        self.request_json("GET", "/v1/schedules".to_string(), None)
    }

    fn run_schedule_now(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        self.empty(
            "POST",
            format!("/v1/schedules/{}/run-now", percent_encoded(name)),
            None,
        )
    }

    fn subscribe(
        &self,
        loop_id: &str,
        since: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, ClientError>> + Send + '_>> {
        let socket = self.socket.clone();
        let path = match since {
            Some(cursor) => {
                format!(
                    "/v1/loops/{}/events?since={cursor}",
                    percent_encoded(loop_id)
                )
            }
            None => format!("/v1/loops/{}/events", percent_encoded(loop_id)),
        };
        Box::pin(async move {
            let request = HttpRequest {
                method: "GET",
                path,
                body: None,
            };
            let (status, mut reader) = open(&socket, &request).await?;
            if status == 410 {
                return Err(ClientError::CursorExpired { since });
            }
            if !status_ok(status) {
                let body = reader.read_to_body_end().await?;
                return Err(map_api_error(status, &body));
            }
            Ok(EventStream::new(reader))
        })
    }
}

/// Whether a status code is a success for the verb layer.
///
/// Any 2xx; the protocol pins no finer granularity.
fn status_ok(status: u16) -> bool {
    (200..300).contains(&status)
}

/// Decode a JSON body into `T`.
///
/// # Errors
///
/// A body that is not the expected JSON shape is a malformed
/// response carrying the parse diagnosis.
fn decode_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ClientError> {
    serde_json::from_slice(body).map_err(|error| {
        ClientError::MalformedResponse(format!("the body was not the expected JSON: {error}"))
    })
}

/// Map a non-success status onto the typed error surface.
///
/// The protocol's error body is `{"error": {"code": ..., "message": ...}}`;
/// a body that misses the shape still surfaces the status with
/// whatever text arrived.
fn map_api_error(status: u16, body: &[u8]) -> ClientError {
    let parsed: Option<serde_json::Value> = serde_json::from_slice(body).ok();
    let error = parsed.as_ref().and_then(|value| value.get("error"));
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let message = match error
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        Some(message) => message.to_string(),
        None => String::from_utf8_lossy(body).trim().to_string(),
    };
    ClientError::Api {
        status,
        code,
        message,
    }
}

/// The resource name an error body names, when it carries one.
///
/// From the 404 envelope's `what` field; the fallback text
/// keeps the message honest when the daemon sent none.
fn path_of(body: &[u8]) -> String {
    let parsed: Option<serde_json::Value> = serde_json::from_slice(body).ok();
    match parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(|error| error.get("what"))
        .and_then(serde_json::Value::as_str)
    {
        Some(what) => what.to_string(),
        None => "the requested resource".to_string(),
    }
}

/// Percent-encode one path segment.
///
/// The resource names are host-chosen identifiers; encoding keeps the
/// request line well-formed for any byte sequence without inventing
/// query parsing on the client side.
fn percent_encoded(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            other => {
                write!(encoded, "%{other:02X}").ok();
            }
        }
    }
    encoded
}
