//! The shared daemon client for the loopctl family.
//!
//! One crate speaks the daemon protocol so no family binary carries
//! protocol code of its own: the CLI's daemon-facing verbs, the TUI's
//! attach/stream/approval surfaces, and any future third-party
//! client all consume [`DaemonClient`] through the
//! [`LoopctlClient`] interface — in-process consumers implement the
//! same interface, so a binary chooses a mode without the call sites
//! changing shape.
//!
//! The protocol is HTTP/1.1 REST + SSE over a per-user unix socket
//! (the crate is unix-only by design — the transport *is* the socket)
//! with `/v1` versioned paths and JSON bodies: every verb is one
//! request/response, the observation surface is
//! `GET /v1/loops/{id}/events?since=<cursor>` streaming
//! trajectory-JSONL events whose SSE `id` is the resume cursor (a
//! stale cursor answers `410 Gone`, surfaced as the typed
//! [`CursorExpired`](ClientError::CursorExpired) that says: relist
//! and resubscribe). Construction performs the versioned handshake
//! and refuses a family member outside the negotiated floor in
//! either direction — an old server is never silently absorbed, and
//! neither is a too-new one.
//!
//! ```toml
//! [dependencies]
//! loopctl-client = "0.3"
//! ```
//!
//! ```rust,no_run
//! use loopctl_client::{ConnectOptions, DaemonClient, LoopctlClient as _};
//!
//! # async fn demo() -> Result<(), loopctl_client::ClientError> {
//! let client = DaemonClient::connect(ConnectOptions::new(
//!     "/run/user/1000/loopctl/loopctl.sock",
//! ))
//! .await?;
//! for pending in client.pending_gates().await? {
//!     println!("waiting approval: {}", pending.gate_id);
//! }
//! let mut stream = client.subscribe("watch-repos", None).await?;
//! use futures::StreamExt as _;
//! while let Some(event) = stream.next().await {
//!     let event = event?;
//!     println!("cursor {} — {:?}", event.cursor, event.event.kind);
//! }
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::missing_errors_doc,
        clippy::unnecessary_wraps,
        clippy::clone_on_ref_ptr,
        clippy::doc_markdown,
        clippy::field_reassign_with_default,
        clippy::used_underscore_items,
        clippy::wildcard_imports,
    )
)]

#[cfg(not(unix))]
compile_error!(
    "loopctl-client speaks the daemon's unix-socket protocol and has no \
     non-unix transport; this crate is unix-only by design"
);

mod client;
mod events;
mod http;

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use uuid::Uuid;

pub use client::ConnectOptions;
pub use client::DaemonClient;
pub use client::RetryWindow;
pub use client::SpawnAction;
pub use client::SpawnPolicy;
pub use events::EventStream;
pub use events::LedgerEvent;

/// The one interface both modes implement.
///
/// [`DaemonClient`] is the socket-side implementation; an in-process
/// consumer implements the same methods against the engine directly,
/// so call sites are mode-agnostic — the seam the family's two
/// binaries share. Every method is one protocol verb; the event
/// subscription is one long-lived stream.
///
/// The trait is object-safe: a host can hold `Box<dyn LoopctlClient>`
/// and swap modes at construction time.
pub trait LoopctlClient: Send + Sync {
    /// List the daemon's registered loops.
    ///
    /// # Errors
    ///
    /// Propagates transport, handshake-family, and daemon errors.
    fn list_loops(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<LoopSummary>, ClientError>> + Send + '_>>;

    /// Fetch one loop's summary.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the id matches no loop.
    fn get_loop(
        &self,
        loop_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LoopSummary, ClientError>> + Send + '_>>;

    /// Start one run of a loop with the given input.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the loop id matches no loop.
    fn start_run(
        &self,
        loop_id: &str,
        input: &str,
    ) -> Pin<Box<dyn Future<Output = Result<RunHandle, ClientError>> + Send + '_>>;

    /// Fetch one run's state.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the id matches no run.
    fn get_run(
        &self,
        run_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<RunState, ClientError>> + Send + '_>>;

    /// Request a run's cooperative stop.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the id matches no run.
    fn stop_run(
        &self,
        run_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;

    /// List the approvals waiting on a human.
    ///
    /// # Errors
    ///
    /// Propagates transport, handshake-family, and daemon errors.
    fn pending_gates(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<GatePending>, ClientError>> + Send + '_>>;

    /// Approve one pending gate, recording the reason.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the gate id matches no pending
    /// gate.
    fn approve_gate(
        &self,
        gate_id: &str,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;

    /// Deny one pending gate, recording the reason.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the gate id matches no pending
    /// gate.
    fn deny_gate(
        &self,
        gate_id: &str,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;

    /// List the registered schedules.
    ///
    /// # Errors
    ///
    /// Propagates transport, handshake-family, and daemon errors.
    fn list_schedules(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScheduleSummary>, ClientError>> + Send + '_>>;

    /// Trigger one schedule immediately, off its calendar.
    ///
    /// # Errors
    ///
    /// [`ClientError::NotFound`] when the name matches no schedule.
    fn run_schedule_now(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;

    /// Subscribe to a loop's event stream.
    ///
    /// `since` resumes exactly after that cursor — the value each
    /// delivered [`LedgerEvent`] carries. A cursor the daemon no
    /// longer honors answers the typed
    /// [`CursorExpired`](ClientError::CursorExpired): relist events
    /// and resubscribe from the newest cursor.
    ///
    /// # Errors
    ///
    /// [`ClientError::CursorExpired`] on a stale cursor;
    /// [`ClientError::NotFound`] when the loop id matches no loop.
    fn subscribe(
        &self,
        loop_id: &str,
        since: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, ClientError>> + Send + '_>>;
}

/// One registered loop, as the daemon summarizes it.
///
/// Fields are additive across versions — new columns appear here as
/// the registry grows.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoopSummary {
    /// The loop's identifier, stable across runs.
    ///
    /// Host-chosen at registration; it is the path segment the verbs
    /// and the event subscription address.
    pub id: String,

    /// The loop's current lifecycle status, e.g. `idle`, `running`.
    ///
    /// A free-form daemon vocabulary, additive across versions; new
    /// states appear without a protocol bump.
    pub status: String,

    /// The model the loop is configured for, when declared.
    ///
    /// `None` when the loop runs on its manifest's default rather
    /// than an explicit model choice.
    pub model: Option<String>,
}

/// The handle a run start returns.
///
/// The minimal acknowledgement of a detached start: the caller
/// polls [`RunState`] with the id to observe
/// progress.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunHandle {
    /// The started run's identifier.
    ///
    /// Minted by the daemon; every later run-scoped verb takes it.
    pub run_id: Uuid,
}

/// One run's observable state.
///
/// The run record as the daemon holds it — the polling answer for
/// detached runs and the resume surface after reconnects.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunState {
    /// The run's identifier.
    ///
    /// Echoes the handle the start returned.
    pub run_id: Uuid,

    /// The loop the run belongs to.
    ///
    /// The parent loop's identifier, for correlating a run with its
    /// loop's event stream.
    pub loop_id: String,

    /// The run's lifecycle status, e.g. `running`, `completed`,
    /// `failed`, `cancelled`.
    pub status: String,

    /// How the run ended, when it has.
    ///
    /// `None` while the run is in flight; the daemon's terminal
    /// vocabulary (`completed`, `failed`, `cancelled`, …) once it
    /// ends.
    pub stop_reason: Option<String>,
}

/// One approval waiting on a human.
///
/// The unattended-ask queue entry: a paused run's gated tool call,
/// surfaced so a CLI or the TUI can answer it from anywhere.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GatePending {
    /// The pending gate's identifier — the approve/deny target.
    ///
    /// Unique among pending gates; spent by either answer.
    pub gate_id: String,

    /// The loop the gated call belongs to.
    ///
    /// Names the paused loop so an approver can weigh context.
    pub loop_id: String,

    /// The tool the gated call invokes.
    ///
    /// The model-facing tool name, as the gate recorded it.
    pub tool: String,

    /// The question the gate puts to the human.
    ///
    /// The hook's own prompt text, verbatim — what an approval UI
    /// renders.
    pub prompt: String,
}

/// One registered schedule, as the daemon summarizes it.
///
/// The calendar surface only — the spec itself lives in the loop's
/// manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScheduleSummary {
    /// The schedule's name.
    ///
    /// Unique within the daemon; the run-now verb addresses it.
    pub name: String,

    /// Whether the schedule is suspended (no automatic triggers).
    ///
    /// A suspended schedule keeps its history but fires nothing
    /// until resumed.
    pub suspended: bool,
}

/// Everything that can go wrong on the client side of the seam.
///
/// One typed surface for both binaries: transport failures, the
/// handshake refusals, the relist contract, and the daemon's own
/// error envelopes — every variant carries its diagnosis in its
/// message, so a host can print it without translating.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// No daemon is listening on the socket.
    ///
    /// The in-process degrade signal: a host may fall back to
    /// embedding the engine instead of demanding a daemon.
    #[error("no daemon is listening on the socket — start one, spawn one, or run in-process")]
    SocketAbsent,

    /// The standard socket path could not be resolved.
    ///
    /// `$XDG_RUNTIME_DIR` is unset; pass an explicit
    /// [`ConnectOptions`] socket path.
    #[error(
        "the standard socket path is unavailable (XDG_RUNTIME_DIR is unset) — pass an explicit path"
    )]
    SocketPathUnavailable,

    /// Something answered, but it is not the family daemon.
    ///
    /// The socket path reached an HTTP peer whose `/v1/hello` is a
    /// miss — likely another service's socket.
    #[error("the peer at {path} is not a loopctl daemon (no /v1/hello)")]
    NotADaemon {
        /// The socket path that answered.
        ///
        /// Names the peer that failed the family check, so a
        /// misconfigured `XDG_RUNTIME_DIR` is diagnosable.
        path: PathBuf,
    },

    /// The spawn action failed.
    ///
    /// The host-supplied closure errored before any retry — the
    /// daemon was never asked for.
    #[error("spawning the daemon failed: {detail}")]
    SpawnFailed {
        /// The spawn action's own diagnosis.
        ///
        /// The closure's error text, verbatim.
        detail: String,
    },

    /// The daemon never appeared inside the retry window.
    ///
    /// The spawn ran and the bounded retries elapsed without a
    /// handshake — the window's span rides the message.
    #[error("the daemon did not appear within {window_ms} ms of spawning")]
    ConnectTimeout {
        /// The retry window's span, in milliseconds.
        ///
        /// The delays between attempts that elapsed before giving
        /// up — the immediate first attempt waits nothing, so the
        /// span is exactly the waited delays; what a host should
        /// compare its daemon's startup time against.
        window_ms: u64,
    },

    /// The peer speaks a different family.
    ///
    /// The handshake's `family` disagrees with this client's — a
    /// different product's daemon on the same socket.
    #[error("family mismatch: the daemon speaks {daemon:?} but this client is {client:?}")]
    FamilyMismatch {
        /// The family the daemon reported.
        ///
        /// Verbatim from the handshake document.
        daemon: String,

        /// This client's family.
        ///
        /// The constant this crate was built with.
        client: String,
    },

    /// The server is older than this client's floor.
    ///
    /// The negotiated protocol cannot serve this client; the fix is
    /// updating the daemon, stated in the message.
    #[error(
        "the daemon's protocol {daemon_protocol} is below this client's floor {client_floor} — update the daemon"
    )]
    ServerTooOld {
        /// The protocol version the daemon reported.
        ///
        /// Below this client's floor in this variant.
        daemon_protocol: u64,

        /// The oldest server this client talks to.
        ///
        /// The floor this crate pins; both numbers ride the
        /// message.
        client_floor: u64,
    },

    /// The server's floor is above this client's version.
    ///
    /// The mirror refusal: a daemon too new for this client — the
    /// fix is updating this client, stated in the message.
    #[error(
        "the daemon's floor {daemon_floor} is above this client's protocol {client_protocol} — update this client"
    )]
    ServerTooNew {
        /// The oldest client the daemon talks to.
        ///
        /// The floor the handshake carried.
        daemon_floor: u64,

        /// This client's protocol version.
        ///
        /// The constant this crate was built with.
        client_protocol: u64,
    },

    /// The requested event cursor is gone.
    ///
    /// The relist contract: the daemon no longer retains the history
    /// `since` names — list the current events and resubscribe from
    /// the newest cursor.
    #[error(
        "the event cursor {since:?} is no longer retained — relist events and resubscribe from the newest cursor"
    )]
    CursorExpired {
        /// The cursor the subscription passed as `since`.
        ///
        /// Echoed so the caller knows exactly which resume point
        /// expired.
        since: Option<u64>,
    },

    /// The named resource does not exist.
    ///
    /// The daemon's 404 mapping; the message names what was
    /// requested as the daemon saw it.
    #[error("not found: {what}")]
    NotFound {
        /// What was requested, as the daemon named it.
        ///
        /// From the error envelope's `what` field when present.
        what: String,
    },

    /// The daemon refused or failed the request.
    ///
    /// A 4xx/5xx with the protocol's error envelope — the status,
    /// the stable machine `code` when sent, and the human message
    /// ride verbatim.
    #[error("daemon error ({status}): {message}")]
    Api {
        /// The HTTP status the daemon answered.
        ///
        /// The raw three-digit code.
        status: u16,

        /// The daemon's stable machine code, when it sent one.
        ///
        /// Matchable across versions; `None` when the envelope
        /// carried none.
        code: Option<String>,

        /// The daemon's human diagnosis.
        ///
        /// Printable as-is, per the error-surface contract.
        message: String,
    },

    /// The peer's bytes did not parse as the dialect.
    ///
    /// A head, framing, or payload outside the pinned client
    /// contract — the daemon and this crate disagree on the wire.
    #[error("malformed response: {0}")]
    MalformedResponse(String),

    /// The socket transport failed.
    ///
    /// An IO failure beneath the protocol, carried with its
    /// diagnosis.
    #[error("transport failure: {0}")]
    Transport(String),
}
