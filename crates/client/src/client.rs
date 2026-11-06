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
#[cfg(unix)]
use std::path::Path;
#[cfg(any(unix, feature = "tls"))]
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
use crate::http::status_ok;

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

/// The default bound on one request/response exchange.
///
/// Generous against a loaded daemon — the bound exists so a stalled
/// peer surfaces as a typed refusal instead of an eternal await, not
/// to pace a healthy one.
const DEFAULT_EXCHANGE_DEADLINE: Duration = Duration::from_secs(30);

/// Where a [`DaemonClient`] reaches the daemon.
///
/// The endpoint picks the transport and nothing else: the
/// handshake, every verb, and the event stream behave identically
/// across variants. Local access control for [`Endpoint::Socket`]
/// is the filesystem's; the remote variants carry the bearer token
/// a listening daemon expects.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Endpoint {
    /// The daemon's unix socket on the same machine.
    ///
    /// The transport the crate was built around: one peer, one
    /// user, one machine. A missing socket file is the typed
    /// [`SocketAbsent`](ClientError::SocketAbsent) the auto-spawn
    /// and in-process degrade paths key on.
    #[cfg(unix)]
    Socket(PathBuf),

    /// A daemon on another machine, over plain TCP.
    ///
    /// For trusted networks — a tailnet, a VPN, a loopback forward —
    /// where transport encryption is provided from outside. The
    /// bearer token, when one is set, crosses the wire in the
    /// clear; weigh what a remote gate approval authorizes against
    /// that before choosing this over the TLS endpoint.
    Tcp {
        /// The daemon's host, a name or IP literal.
        ///
        /// Resolved at each connection; a DNS name works exactly
        /// like an address.
        host: String,

        /// The daemon's TCP port.
        ///
        /// What the listening daemon was configured with.
        port: u16,

        /// The bearer token the daemon expects, when it wants one.
        ///
        /// Sent as the `Authorization: Bearer` header on every
        /// request, the handshake included; `None` for a daemon
        /// that authenticates by other means — or refuses politely
        /// with the typed [`Unauthorized`](ClientError::Unauthorized).
        /// A token carrying CR, LF, or NUL is refused at the wire
        /// rather than spliced into the request head.
        token: Option<String>,
    },

    /// A daemon on another machine, over TLS.
    ///
    /// Requires the `tls` feature. The connection is encrypted and
    /// the daemon verified against `ca_pem` — or the public `WebPKI`
    /// roots when no certificate-authority file is given — and the
    /// bearer token rides encrypted with it.
    #[cfg(feature = "tls")]
    Tls {
        /// The daemon's host, a name or IP literal.
        ///
        /// Also the TLS server name the handshake presents, so the
        /// daemon's certificate must cover this name.
        host: String,

        /// The daemon's TCP port.
        ///
        /// What the listening daemon was configured with.
        port: u16,

        /// The bearer token the daemon expects, when it wants one.
        ///
        /// Sent as the `Authorization: Bearer` header on every
        /// request, inside the encrypted channel. A token carrying
        /// CR, LF, or NUL is refused at the wire rather than spliced
        /// into the request head.
        token: Option<String>,

        /// A PEM file of certificate authorities to verify the daemon with.
        ///
        /// `None` verifies against the public `WebPKI` roots — for a
        /// daemon named in a publicly-trusted certificate; a
        /// self-hosted daemon's own CA belongs here.
        ca_pem: Option<PathBuf>,

        /// A PEM file of this client's certificate chain and private key.
        ///
        /// `None` presents no client certificate; a daemon that requires
        /// one refuses the handshake, surfaced as a typed transport
        /// failure. The file carries the chain followed by the key — the
        /// same single-file shape `ca_pem` takes.
        client_identity: Option<PathBuf>,
    },
}

impl Endpoint {
    /// The socket endpoint at one path.
    ///
    /// The local shape, as [`ConnectOptions::new`] builds it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use loopctl_client::Endpoint;
    ///
    /// let endpoint = Endpoint::socket("/run/user/1000/loopctl/loopctl.sock");
    /// ```
    #[cfg(unix)]
    #[must_use]
    pub fn socket(path: impl Into<PathBuf>) -> Self {
        Self::Socket(path.into())
    }

    /// The plain-TCP endpoint at one host and port.
    ///
    /// Trusted networks only — the variant's own docs say why. The
    /// token, when one is given, is validated at the wire: a token
    /// carrying CR, LF, or NUL refuses the request as a transport
    /// error rather than splicing it into the head.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use loopctl_client::Endpoint;
    ///
    /// let endpoint = Endpoint::tcp("daemon.tailnet.ts.net", 8787, None);
    /// ```
    #[must_use]
    pub fn tcp(host: impl Into<String>, port: u16, token: Option<String>) -> Self {
        Self::Tcp {
            host: host.into(),
            port,
            token,
        }
    }

    /// The TLS endpoint at one host and port.
    ///
    /// The encrypted remote shape; `ca_pem` names the certificate
    /// authority to verify the daemon with, or `None` for the public
    /// `WebPKI` roots. The token, when one is given, is validated at
    /// the wire: a token carrying CR, LF, or NUL refuses the request
    /// as a transport error rather than splicing it into the head.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "tls")]
    /// # async fn demo() {
    /// use loopctl_client::Endpoint;
    ///
    /// let endpoint = Endpoint::tls(
    ///     "daemon.tailnet.ts.net",
    ///     8787,
    ///     None,
    ///     Some("/etc/loopctl/ca.pem".into()),
    /// );
    /// # }
    /// ```
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(
        host: impl Into<String>,
        port: u16,
        token: Option<String>,
        ca_pem: Option<PathBuf>,
    ) -> Self {
        Self::Tls {
            host: host.into(),
            port,
            token,
            ca_pem,
            client_identity: None,
        }
    }

    /// The TLS endpoint presenting a client certificate.
    ///
    /// The mutual-TLS shape: the connection is encrypted and verified
    /// against `ca_pem` exactly as [`tls`](Self::tls) does, and the
    /// client additionally presents the certificate chain and key in
    /// `client_identity` — one PEM file, chain first — for a daemon
    /// that authenticates clients by certificate. An identity file
    /// that cannot be read, or holds no certificate or no key, refuses
    /// as a typed transport error before the connection opens.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "tls")]
    /// # async fn demo() {
    /// use loopctl_client::Endpoint;
    ///
    /// let endpoint = Endpoint::mtls(
    ///     "daemon.tailnet.ts.net",
    ///     8787,
    ///     None,
    ///     Some("/etc/loopctl/ca.pem".into()),
    ///     "/etc/loopctl/client.pem",
    /// );
    /// # }
    /// ```
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn mtls(
        host: impl Into<String>,
        port: u16,
        token: Option<String>,
        ca_pem: Option<PathBuf>,
        client_identity: impl Into<PathBuf>,
    ) -> Self {
        Self::Tls {
            host: host.into(),
            port,
            token,
            ca_pem,
            client_identity: Some(client_identity.into()),
        }
    }

    /// The bearer token this endpoint authenticates with, when set.
    ///
    /// What the transport writes into the `Authorization` header of
    /// every request; `None` for the socket endpoint and for remote
    /// endpoints that carry no token.
    pub(crate) fn bearer_token(&self) -> Option<&str> {
        match self {
            #[cfg(unix)]
            Self::Socket(_) => None,
            Self::Tcp { token, .. } => token.as_deref(),
            #[cfg(feature = "tls")]
            Self::Tls { token, .. } => token.as_deref(),
        }
    }

    /// The authority the request head's `Host` header carries.
    ///
    /// The remote endpoints name their `host:port` — an `IPv6`
    /// literal bracketed, as the authority form requires — what a
    /// proxy-terminated deployment routes on; the socket variant has
    /// no authority of its own and carries the fixed family name,
    /// which the local daemon ignores. A host carrying CR, LF, or
    /// NUL refuses outright, the same stance the bearer token takes:
    /// endpoint-chosen bytes never splice into the request head
    /// unvalidated.
    ///
    /// # Errors
    ///
    /// [`ClientError::Transport`] when a remote host carries CR, LF,
    /// or NUL.
    pub(crate) fn authority(&self) -> Result<String, ClientError> {
        match self {
            #[cfg(unix)]
            Self::Socket(_) => Ok(FAMILY.to_string()),
            Self::Tcp { host, port, .. } => remote_authority(host, *port),
            #[cfg(feature = "tls")]
            Self::Tls { host, port, .. } => remote_authority(host, *port),
        }
    }

    /// Whether the endpoint is the local socket.
    ///
    /// The only shape an auto-spawn action can serve — a daemon on
    /// another machine is not this process's to start.
    #[cfg(unix)]
    fn is_local_socket(&self) -> bool {
        matches!(self, Self::Socket(_))
    }

    /// Whether the endpoint is the local socket.
    ///
    /// Off unix no endpoint is local — there is no socket variant —
    /// so an attached spawn action never runs.
    #[cfg(not(unix))]
    fn is_local_socket(&self) -> bool {
        false
    }

    /// One human-readable line naming this endpoint.
    ///
    /// What error variants quote when the peer misbehaves — the
    /// socket path, or the `host:port` a remote endpoint addresses.
    fn describe(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::Socket(path) => path.display().to_string(),
            Self::Tcp { host, port, .. } => format!("{host}:{port}"),
            #[cfg(feature = "tls")]
            Self::Tls { host, port, .. } => format!("{host}:{port}"),
        }
    }
}

/// The `host:port` authority one remote endpoint addresses.
///
/// Joins and validates in one step: the value rides the request
/// head's `Host` header verbatim, so the join refuses a host
/// carrying CR, LF, or NUL rather than splicing it — the same
/// boundary the bearer token takes, at the same place. An `IPv6`
/// literal rides bracketed, the authority form RFC 9110 reserves
/// for it — `::1` addresses `[::1]:{port}`, never the ambiguous
/// unbracketed join.
///
/// # Errors
///
/// [`ClientError::Transport`] when the host carries CR, LF, or NUL.
fn remote_authority(host: &str, port: u16) -> Result<String, ClientError> {
    if host
        .bytes()
        .any(|byte| byte == b'\r' || byte == b'\n' || byte == 0)
    {
        return Err(ClientError::Transport(
            "the endpoint host carries CR, LF, or NUL — refused rather than spliced into the \
             request head"
                .into(),
        ));
    }
    if host
        .parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_ipv6())
    {
        return Ok(format!("[{host}]:{port}"));
    }
    Ok(format!("{host}:{port}"))
}

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
/// The endpoint, the spawn policy for an absent socket, and the
/// bounded window a spawn waits inside.
#[derive(Debug, Clone)]
pub struct ConnectOptions {
    /// Where the daemon is reached.
    ///
    /// The local socket by default ([`new`](Self::new)); the remote
    /// constructors ([`tcp`](Self::tcp), and `tls` behind the `tls`
    /// feature) attach
    /// their own transport.
    pub endpoint: Endpoint,

    /// Whether and how to spawn a daemon when the socket is absent.
    ///
    /// [`Never`](SpawnPolicy::Never) keeps the constructor pure;
    /// an action carries the host's spawn recipe. Socket endpoints
    /// only — a remote daemon cannot be spawned from here, and an
    /// absent remote endpoint surfaces
    /// [`SocketAbsent`](ClientError::SocketAbsent) without running
    /// the action.
    pub spawn: SpawnPolicy,

    /// The bounded retry window an auto-spawn waits inside.
    ///
    /// Applies only under [`SpawnPolicy::Spawn`].
    pub retry: RetryWindow,

    /// The bound on any single exchange.
    ///
    /// One deadline covers the handshake, each verb, and the
    /// establishment of an event subscription — a peer that accepts
    /// and stalls answers the typed transport refusal naming the
    /// bound, never an eternal await. The subscription's body is
    /// untouched: a quiet daemon may stream for as long as it keeps
    /// the connection. Defaults to thirty seconds;
    /// [`with_deadline`](Self::with_deadline) tunes it.
    pub deadline: Duration,
}

impl ConnectOptions {
    /// Options for one socket, no spawning.
    ///
    /// The pure embedder entry: an absent socket is the typed
    /// [`SocketAbsent`](ClientError::SocketAbsent) the host's
    /// in-process degrade keys on.
    #[cfg(unix)]
    #[must_use]
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: Endpoint::Socket(socket_path.into()),
            spawn: SpawnPolicy::Never,
            retry: RetryWindow::default(),
            deadline: DEFAULT_EXCHANGE_DEADLINE,
        }
    }

    /// Options for one remote daemon over plain TCP, no spawning.
    ///
    /// The pure remote entry: an absent daemon is the typed
    /// [`SocketAbsent`](ClientError::SocketAbsent), never a spawn
    /// attempt — see [`with_spawn`](Self::with_spawn) for the one
    /// endpoint it serves.
    #[must_use]
    pub fn tcp(host: impl Into<String>, port: u16, token: Option<String>) -> Self {
        Self {
            endpoint: Endpoint::tcp(host, port, token),
            spawn: SpawnPolicy::Never,
            retry: RetryWindow::default(),
            deadline: DEFAULT_EXCHANGE_DEADLINE,
        }
    }

    /// Options for one remote daemon over TLS, no spawning.
    ///
    /// The encrypted remote entry; `ca_pem` names the certificate
    /// authority to verify the daemon with, or `None` for the
    /// public `WebPKI` roots.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(
        host: impl Into<String>,
        port: u16,
        token: Option<String>,
        ca_pem: Option<PathBuf>,
    ) -> Self {
        Self {
            endpoint: Endpoint::tls(host, port, token, ca_pem),
            spawn: SpawnPolicy::Never,
            retry: RetryWindow::default(),
            deadline: DEFAULT_EXCHANGE_DEADLINE,
        }
    }

    /// Options for one remote daemon over mutual TLS, no spawning.
    ///
    /// The authenticated remote entry: `ca_pem` names the authority
    /// the daemon is verified against, and `client_identity` names the
    /// PEM file carrying this client's certificate chain and key —
    /// see [`Endpoint::mtls`] for the handshake contract.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn mtls(
        host: impl Into<String>,
        port: u16,
        token: Option<String>,
        ca_pem: Option<PathBuf>,
        client_identity: impl Into<PathBuf>,
    ) -> Self {
        Self {
            endpoint: Endpoint::mtls(host, port, token, ca_pem, client_identity),
            spawn: SpawnPolicy::Never,
            retry: RetryWindow::default(),
            deadline: DEFAULT_EXCHANGE_DEADLINE,
        }
    }

    /// Attach a spawn action and retry window, chaining.
    ///
    /// The action runs once per connect; the window bounds how long
    /// the handshake retries after it — and every retry is itself
    /// bounded by the exchange [`deadline`](Self::deadline), so a
    /// peer that accepts and never answers cannot outlast the window
    /// either.
    #[must_use]
    pub fn with_spawn(mut self, action: SpawnAction, retry: RetryWindow) -> Self {
        self.spawn = SpawnPolicy::Spawn(action);
        self.retry = retry;
        self
    }

    /// Bound every single exchange at `deadline`, chaining.
    ///
    /// One bound covers the handshake, each verb, and the
    /// establishment of an event subscription — a peer that accepts
    /// and stalls answers the typed transport refusal naming the
    /// bound, never an eternal await. The subscription's body is
    /// untouched by design: a quiet daemon streams for as long as it
    /// keeps the connection.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
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
    /// One action run, bounded retries, a typed timeout on failure —
    /// never an unbounded wait: transient handshake failures (a peer
    /// still warming, a partial early response) retry inside the
    /// window beside absence, and every attempt answers within the
    /// exchange deadline.
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
/// verb is one HTTP request over the endpoint's transport, and the
/// event stream is one long-lived SSE connection.
#[derive(Debug, Clone)]
pub struct DaemonClient {
    /// The endpoint every verb addresses.
    ///
    /// Kept from [`ConnectOptions`] at construction so each request
    /// reconnects — the connection-per-request shape.
    endpoint: Endpoint,

    /// The negotiated handshake snapshot.
    ///
    /// What the daemon answered at connect: its family, protocol
    /// version, and floor. Kept for [`handshake`](Self::handshake)
    /// and never re-negotiated within one client's lifetime.
    hello: Hello,

    /// The bound each single exchange answers within.
    ///
    /// Kept from [`ConnectOptions`] at construction; every verb and
    /// the establishment of each subscription run under it.
    deadline: Duration,
}

impl DaemonClient {
    /// Connect to the daemon under `options`, performing the
    /// handshake.
    ///
    /// With [`SpawnPolicy::Spawn`], an absent socket — or a transient
    /// handshake failure, the shape a warming daemon takes — runs the
    /// spawn action and retries inside the window; a daemon that
    /// never appears surfaces as the window's typed timeout naming
    /// what was tried. Refusals that retrying cannot heal (a foreign
    /// family, a floor violation, a credential answer) surface
    /// immediately. Every attempt answers within the exchange
    /// [`deadline`](ConnectOptions::deadline).
    /// # Errors
    ///
    /// [`ClientError::SocketAbsent`] when spawning is off and no
    /// daemon listens; [`ClientError::SpawnFailed`] and
    /// [`ClientError::ConnectTimeout`] for the spawn path; transport,
    /// malformed-response, and the three handshake refusals
    /// (`NotADaemon`, family mismatch, floor violation) otherwise.
    pub async fn connect(options: ConnectOptions) -> Result<Self, ClientError> {
        match Self::try_handshake(&options.endpoint, options.deadline).await {
            Ok(Some(client)) => return Ok(client),
            Ok(None) => {}
            Err(error) => {
                let SpawnPolicy::Spawn(_) = &options.spawn else {
                    return Err(error);
                };
                if !options.endpoint.is_local_socket() || !handshake_should_retry(&error) {
                    return Err(error);
                }
            }
        }
        let SpawnPolicy::Spawn(action) = &options.spawn else {
            return Err(ClientError::SocketAbsent);
        };
        if !options.endpoint.is_local_socket() {
            return Err(ClientError::SocketAbsent);
        }
        action().map_err(|source| ClientError::SpawnFailed {
            detail: source.to_string(),
        })?;
        for attempt in 0..options.retry.attempts {
            if attempt > 0 {
                tokio::time::sleep(options.retry.delay).await;
            }
            match Self::try_handshake(&options.endpoint, options.deadline).await {
                Ok(Some(client)) => return Ok(client),
                Ok(None) => {}
                Err(error) if handshake_should_retry(&error) => {}
                Err(error) => return Err(error),
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
    /// `LOOPCTL_NO_AUTOSPAWN` is set to `1` — the exact value `1`, any
    /// other setting leaves spawning on.
    ///
    /// # Errors
    ///
    /// [`ClientError::SocketPathUnavailable`] when the runtime dir is
    /// unset; otherwise the [`DaemonClient::connect`] set.
    #[cfg(unix)]
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
    async fn try_handshake(
        endpoint: &Endpoint,
        deadline: Duration,
    ) -> Result<Option<Self>, ClientError> {
        let request = HttpRequest {
            method: "GET",
            path: "/v1/hello".to_string(),
            body: None,
        };
        let response = match send(endpoint, deadline, request).await {
            Ok(response) => response,
            Err(ClientError::SocketAbsent) => return Ok(None),
            Err(error) => return Err(error),
        };
        if response.status == 404 {
            return Err(ClientError::NotADaemon {
                endpoint: endpoint.describe(),
            });
        }
        if !status_ok(response.status) {
            return Err(map_api_error(response.status, &response.body));
        }
        let hello: Hello = decode_json(&response.body)?;
        negotiate(&hello)?;
        Ok(Some(Self {
            endpoint: endpoint.clone(),
            hello,
            deadline,
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
        let endpoint = self.endpoint.clone();
        Box::pin(async move {
            let response =
                send(&endpoint, self.deadline, HttpRequest { method, path, body }).await?;
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
        let endpoint = self.endpoint.clone();
        Box::pin(async move {
            let response =
                send(&endpoint, self.deadline, HttpRequest { method, path, body }).await?;
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
        let endpoint = self.endpoint.clone();
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
            let (status, mut reader) = open(&endpoint, self.deadline, &request).await?;
            if status == 410 {
                return Err(ClientError::CursorExpired { since });
            }
            if !status_ok(status) {
                let body = reader.drain_error_body().await;
                return Err(map_api_error(status, &body));
            }
            let answered = reader
                .content_type()
                .map_or_else(|| "no content type".to_string(), ToOwned::to_owned);
            if !answered
                .to_ascii_lowercase()
                .starts_with("text/event-stream")
            {
                return Err(ClientError::MalformedResponse(format!(
                    "the event subscription answered {answered:?}, not an event stream"
                )));
            }
            Ok(EventStream::new(reader))
        })
    }
}

/// Whether one handshake failure may heal inside the retry window.
///
/// Absence — the `Ok(None)` the caller sees — plus a port another
/// service answers while the daemon warms, a malformed or partial
/// early response, a transport hiccup, and a 5xx from a daemon not
/// yet serving all change with time. A foreign family, a floor
/// violation, a credential refusal, and a typed 4xx do not: retrying
/// those only delays the honest error, so they surface immediately.
fn handshake_should_retry(error: &ClientError) -> bool {
    match error {
        ClientError::NotADaemon { .. }
        | ClientError::MalformedResponse(_)
        | ClientError::Transport(_) => true,
        ClientError::Api { status, .. } => *status >= 500,
        _ => false,
    }
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
/// whatever text arrived. A 401/403 refusal maps to the typed
/// [`Unauthorized`](ClientError::Unauthorized) — the credential
/// answer, not a generic API error.
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
    if status == 401 || status == 403 {
        return ClientError::Unauthorized { status, message };
    }
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

#[cfg(test)]
mod tests {
    use super::percent_encoded;
    use proptest::prelude::*;

    #[test]
    fn control_bytes_and_spaces_encode_fully() {
        assert_eq!(
            percent_encoded("\r\n x"),
            "%0D%0A%20x",
            "CR, LF, and space never ride the request line raw"
        );
    }

    proptest! {
        #[test]
        fn percent_encoding_round_trips_and_emits_no_raw_delimiters(
            segment in "\\PC{0,64}",
        ) {
            let encoded = percent_encoded(&segment);
            for byte in encoded.bytes() {
                let safe = byte.is_ascii_alphanumeric()
                    || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'%');
                prop_assert!(safe, "a raw byte escaped the encoder: {byte:?} in {encoded:?}");
            }
            let bytes = encoded.as_bytes();
            let mut decoded = Vec::new();
            let mut index = 0;
            while let Some(&first) = bytes.get(index) {
                if first == b'%'
                    && let (Some(&high), Some(&low)) =
                        (bytes.get(index.saturating_add(1)), bytes.get(index.saturating_add(2)))
                    && let Ok(value) =
                        u8::from_str_radix(&String::from_utf8_lossy(&[high, low]), 16)
                {
                    decoded.push(value);
                    index = index.saturating_add(3);
                } else {
                    decoded.push(first);
                    index = index.saturating_add(1);
                }
            }
            let decoded = String::from_utf8(decoded).expect("the round trip stays utf8");
            prop_assert_eq!(decoded, segment);
        }
    }
}
