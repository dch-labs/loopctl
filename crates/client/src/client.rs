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
/// a listening daemon expects — a token no `Debug` render ever
/// carries, so logging an endpoint, the options holding it, or a
/// connected client cannot leak the credential.
#[derive(Clone)]
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
        /// rather than spliced into the request head, and a token
        /// never renders through `Debug` — it formats as
        /// `<redacted>` there.
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
        /// into the request head, and a token never renders through
        /// `Debug` — it formats as `<redacted>` there.
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

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(unix)]
            Self::Socket(path) => formatter.debug_tuple("Socket").field(path).finish(),
            Self::Tcp { host, port, token } => formatter
                .debug_struct("Tcp")
                .field("host", host)
                .field("port", port)
                .field("token", &token.as_deref().map(|_| "<redacted>"))
                .finish(),
            #[cfg(feature = "tls")]
            Self::Tls {
                host,
                port,
                token,
                ca_pem,
                client_identity,
            } => formatter
                .debug_struct("Tls")
                .field("host", host)
                .field("port", port)
                .field("token", &token.as_deref().map(|_| "<redacted>"))
                .field("ca_pem", ca_pem)
                .field("client_identity", client_identity)
                .finish(),
        }
    }
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
/// unbracketed join — and a scoped literal (`fe80::1%eth0`) rides
/// its zone identifier stripped for the same reason: the authority
/// grammar carries no zone, while the connection still dials the
/// scoped host the endpoint named.
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
    if let Some((address, _zone)) = host.split_once('%')
        && address.parse::<std::net::Ipv6Addr>().is_ok()
    {
        return Ok(format!("[{address}]:{port}"));
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

    /// The bound on any single exchange — and on the whole spawn
    /// window.
    ///
    /// One deadline covers the handshake, each verb, and the
    /// establishment of an event subscription — and the post-spawn
    /// retry sequence spends from the same deadline as one aggregate
    /// budget, attempts and delays together, so a peer that stalls
    /// each handshake cannot stretch the window past it — a peer
    /// that accepts and stalls answers the typed transport refusal
    /// naming the bound, never an eternal await. The subscription's
    /// body is untouched: a quiet daemon may stream for as long as
    /// it keeps the connection. Defaults to thirty seconds;
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
    /// the handshake retries after it, and the whole sequence —
    /// every attempt and every delay together — spends from one
    /// aggregate budget, the exchange [`deadline`](Self::deadline),
    /// so a peer that accepts and never answers cannot outlast the
    /// window either.
    #[must_use]
    pub fn with_spawn(mut self, action: SpawnAction, retry: RetryWindow) -> Self {
        self.spawn = SpawnPolicy::Spawn(action);
        self.retry = retry;
        self
    }

    /// Bound every single exchange at `deadline`, chaining.
    ///
    /// One bound covers the handshake, each verb, the establishment
    /// of an event subscription, and — as one aggregate budget — the
    /// whole post-spawn retry sequence: a peer that accepts and
    /// stalls answers the typed transport refusal naming the bound,
    /// never an eternal await. The subscription's body is untouched
    /// by design: a quiet daemon streams for as long as it keeps the
    /// connection.
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
    /// window beside absence, and the whole sequence — every attempt
    /// and every delay — spends from one aggregate budget, the
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
/// by `attempts − 1` delays — and the whole sequence spends from one
/// aggregate budget, the exchange deadline, so a peer that stalls
/// each handshake cannot stretch the window past it; the default
/// matches the family's bounded-retry guidance (~5 s).
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
    /// the span actually waited. The whole sequence — every attempt
    /// and every delay — spends from one aggregate budget, the
    /// exchange [`deadline`](ConnectOptions::deadline), so a peer
    /// that stalls each handshake cannot stretch the window past it.
    /// Refusals that retrying cannot heal (a foreign family, a floor
    /// violation, a credential answer) surface immediately.
    /// # Errors
    ///
    /// [`ClientError::SocketAbsent`] when spawning is off and no
    /// daemon listens; [`ClientError::SpawnFailed`] and
    /// [`ClientError::ConnectTimeout`] for the spawn path; transport,
    /// malformed-response, and the three handshake refusals
    /// (`NotADaemon`, family mismatch, floor violation) otherwise.
    pub async fn connect(options: ConnectOptions) -> Result<Self, ClientError> {
        match Self::try_handshake(&options.endpoint, options.deadline, options.deadline).await {
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
        Self::retry_window(&options).await
    }

    /// Run the post-spawn retry window under one aggregate budget.
    ///
    /// Every attempt and every delay spends from the same aggregate
    /// budget — the exchange
    /// [`deadline`](ConnectOptions::deadline) — so a peer that
    /// stalls each handshake cannot stretch the window past one
    /// deadline however many attempts remain. The window ends with
    /// whichever runs out first, the attempts or the budget, and
    /// answers the typed timeout naming the span actually waited;
    /// [`handshake_should_retry`] still decides which failures
    /// retry and which surface immediately.
    ///
    /// # Errors
    ///
    /// [`ClientError::ConnectTimeout`] when the window closes
    /// without a handshake; otherwise the deciding attempt's own
    /// refusal.
    async fn retry_window(options: &ConnectOptions) -> Result<Self, ClientError> {
        let started = std::time::Instant::now();
        for attempt in 0..options.retry.attempts {
            if attempt > 0 {
                let remaining = remaining_budget(options.deadline, started);
                if remaining.is_zero() {
                    break;
                }
                tokio::time::sleep(options.retry.delay.min(remaining)).await;
            }
            let remaining = remaining_budget(options.deadline, started);
            if remaining.is_zero() {
                break;
            }
            match Self::try_handshake(&options.endpoint, remaining, options.deadline).await {
                Ok(Some(client)) => return Ok(client),
                Ok(None) => {}
                Err(error) if handshake_should_retry(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Err(ClientError::ConnectTimeout {
            window_ms: elapsed_ms(started),
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
    /// `budget` bounds the exchange this one attempt makes, while
    /// `deadline` is the bound the constructed client keeps for
    /// every later verb — the two differ only inside the spawn
    /// window, where each attempt spends from a shared aggregate
    /// budget rather than the full deadline.
    ///
    /// # Errors
    ///
    /// Propagates transport and malformed-response failures; refuses
    /// non-family or floor-violating peers through [`negotiate`].
    async fn try_handshake(
        endpoint: &Endpoint,
        budget: Duration,
        deadline: Duration,
    ) -> Result<Option<Self>, ClientError> {
        let request = HttpRequest {
            method: "GET",
            path: "/v1/hello".to_string(),
            body: None,
        };
        let response = match send(endpoint, budget, request).await {
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
            let started = std::time::Instant::now();
            let (status, mut reader) = open(&endpoint, self.deadline, &request).await?;
            let remaining = self.deadline.saturating_sub(started.elapsed());
            if status == 410 {
                return Err(ClientError::CursorExpired { since });
            }
            if !status_ok(status) {
                let body = match crate::http::exchange_timeout(remaining, reader.drain_error_body())
                {
                    Ok(bounded) => bounded.await.unwrap_or_default(),
                    Err(error) => return Err(error),
                };
                return Err(map_api_error(status, &body));
            }
            let answered = reader
                .content_type()
                .map_or_else(|| "no content type".to_string(), ToOwned::to_owned);
            if !answered
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
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

/// The unspent share of one aggregate budget, zero once exhausted.
///
/// The spawn window's clock: attempts and delays spend from the same
/// deadline, so the window cannot outlast it even when every
/// attempt stalls for the budget's whole span.
fn remaining_budget(deadline: Duration, started: std::time::Instant) -> Duration {
    deadline
        .checked_sub(started.elapsed())
        .unwrap_or(Duration::ZERO)
}

/// A wall-clock span in whole milliseconds, saturating.
///
/// The honest unit for [`ClientError::ConnectTimeout`]: the span the
/// window actually waited, never a formula about it.
fn elapsed_ms(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
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
    use super::ConnectOptions;
    use super::Endpoint;
    use super::percent_encoded;
    use super::remote_authority;
    use proptest::prelude::*;

    #[test]
    fn control_bytes_and_spaces_encode_fully() {
        assert_eq!(
            percent_encoded("\r\n x"),
            "%0D%0A%20x",
            "CR, LF, and space never ride the request line raw"
        );
    }

    #[test]
    fn authenticated_endpoints_debug_without_leaking_the_token() {
        let tcp = Endpoint::tcp("daemon.example", 8787, Some("sekrit".to_string()));
        let rendered = format!("{tcp:?}");
        assert!(
            !rendered.contains("sekrit"),
            "the TCP endpoint's Debug never carries the token: {rendered}"
        );
        assert!(
            rendered.contains("token: Some(\"<redacted>\")"),
            "the token's presence renders redacted, not absent: {rendered}"
        );
        let options = ConnectOptions::tcp("daemon.example", 8787, Some("sekrit".to_string()));
        let rendered = format!("{options:?}");
        assert!(
            !rendered.contains("sekrit"),
            "the options' Debug carries the endpoint's redaction with it: {rendered}"
        );
        let tokenless = Endpoint::tcp("daemon.example", 8787, None);
        let rendered = format!("{tokenless:?}");
        assert!(
            rendered.contains("token: None"),
            "a tokenless endpoint renders its honest absence: {rendered}"
        );
    }

    #[cfg(feature = "tls")]
    #[test]
    fn the_tls_endpoint_debugs_without_leaking_the_token() {
        let tls = Endpoint::mtls(
            "daemon.example",
            8787,
            Some("sekrit".to_string()),
            Some("/etc/loopctl/ca.pem".into()),
            "/etc/loopctl/client.pem",
        );
        let rendered = format!("{tls:?}");
        assert!(
            !rendered.contains("sekrit"),
            "the TLS endpoint's Debug never carries the token: {rendered}"
        );
        assert!(
            rendered.contains("token: Some(\"<redacted>\")"),
            "the token's presence renders redacted, not absent: {rendered}"
        );
    }

    #[test]
    fn scoped_ipv6_hosts_address_their_zone_stripped_authority() {
        assert_eq!(
            remote_authority("fe80::1%eth0", 8787).ok().as_deref(),
            Some("[fe80::1]:8787"),
            "a scoped IPv6 literal rides bracketed with its zone stripped — the authority \
             grammar carries no zone identifier"
        );
        assert_eq!(
            remote_authority("::1", 8787).ok().as_deref(),
            Some("[::1]:8787"),
            "an unscoped literal keeps its plain bracketed form"
        );
        assert_eq!(
            remote_authority("daemon.example", 8787).ok().as_deref(),
            Some("daemon.example:8787"),
            "a name never brackets"
        );
        assert_eq!(
            remote_authority("host%zone", 8787).ok().as_deref(),
            Some("host%zone:8787"),
            "a percent sign alone does not make a scoped address — the prefix must parse \
             as IPv6"
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
