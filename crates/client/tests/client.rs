//! The client contract suite: every wire behavior pinned against a
//! local mock daemon speaking the protocol dialect.
//!
//! The mock serves scripted raw responses on a tempdir unix socket —
//! one connection per scripted response, bytes exactly as the test
//! writes them, so the framings (Content-Length, chunked,
//! close-delimited) are pinned, not assumed — and records every
//! request it receives (method, path, body), so the verb layer's
//! paths and payloads are pinned too. The event-stream mock holds one
//! connection open and forwards frames the test pushes, so cursor
//! semantics and SSE tolerance are pinned against a realistic feed.

#![cfg(unix)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use loopctl::testing::EnvGuard;
use loopctl_client::ClientError;
use loopctl_client::ConnectOptions;
use loopctl_client::DaemonClient;
use loopctl_client::LoopctlClient;
use loopctl_client::RetryWindow;
use loopctl_client::SpawnAction;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::UnixListener;

struct InProcessFake;
impl LoopctlClient for InProcessFake {
    fn list_loops(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<loopctl_client::LoopSummary>, ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move { Ok(vec![loopctl_client::LoopSummary::new("in-process", "idle")]) })
    }
    fn get_loop(
        &self,
        loop_id: &str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<loopctl_client::LoopSummary, ClientError>>
                + Send
                + '_,
        >,
    > {
        let id = loop_id.to_string();
        Box::pin(async move { Ok(loopctl_client::LoopSummary::new(id, "idle")) })
    }
    fn start_run(
        &self,
        _loop_id: &str,
        _input: &str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<loopctl_client::RunHandle, ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move { Ok(loopctl_client::RunHandle::new(uuid::Uuid::new_v4())) })
    }
    fn get_run(
        &self,
        run_id: uuid::Uuid,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<loopctl_client::RunState, ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            Ok(loopctl_client::RunState::new(
                run_id,
                "in-process",
                "completed",
            ))
        })
    }
    fn stop_run(
        &self,
        _run_id: uuid::Uuid,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ClientError>> + Send + '_>>
    {
        Box::pin(async move { Ok(()) })
    }
    fn pending_gates(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<loopctl_client::GatePending>, ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn approve_gate(
        &self,
        _gate_id: &str,
        _reason: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ClientError>> + Send + '_>>
    {
        Box::pin(async move { Ok(()) })
    }
    fn deny_gate(
        &self,
        _gate_id: &str,
        _reason: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ClientError>> + Send + '_>>
    {
        Box::pin(async move { Ok(()) })
    }
    fn list_schedules(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Vec<loopctl_client::ScheduleSummary>, ClientError>,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn run_schedule_now(
        &self,
        _name: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ClientError>> + Send + '_>>
    {
        Box::pin(async move { Ok(()) })
    }
    fn subscribe(
        &self,
        _loop_id: &str,
        _since: Option<u64>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<loopctl_client::EventStream, ClientError>>
                + Send
                + '_,
        >,
    > {
        let event: loopctl::memory::trajectory::TrajectoryEvent =
            serde_json::from_value(run_started_event(1)).expect("the fixture event parses");
        let source =
            futures::stream::iter(vec![Ok(loopctl_client::LedgerEvent { cursor: 1, event })]);
        Box::pin(std::future::ready(Ok(
            loopctl_client::EventStream::from_pin_box(Box::pin(source)),
        )))
    }
}

/// One request the mock observed.
#[derive(Debug, Clone)]
struct RecordedRequest {
    /// The request method.
    method: String,
    /// The request path with query.
    path: String,
    /// The `Host` header value, empty when the head carried none.
    host: String,
    /// The `Authorization` header value, when the request carried one.
    authorization: Option<String>,
    /// The decoded body, empty when none.
    body: String,
}

/// A mock daemon serving scripted raw responses over a unix socket.
struct MockDaemon {
    socket: PathBuf,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl MockDaemon {
    /// The socket path the daemon listens on.
    fn socket(&self) -> PathBuf {
        self.socket.clone()
    }

    /// Every request observed so far.
    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Stop the listener loop.
    fn stop(self) {
        self.handle.abort();
    }
}

/// Serve `responses` sequentially, one per connection, then stop
/// accepting.
///
/// Bytes are written verbatim — the test owns the framing.
fn spawn_mock(dir: &tempfile::TempDir, name: &str, responses: Vec<Vec<u8>>) -> MockDaemon {
    let socket = dir.path().join(name);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::UnixListener::bind(&socket).expect("the mock socket binds");
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for response in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            stream.write_all(&response).await.expect("the mock writes");
        }
    });
    MockDaemon {
        socket,
        requests,
        handle,
    }
}

/// Serve the handshake forever: every connection gets `hello`.
fn spawn_hello_mock(
    dir: &tempfile::TempDir,
    family: &str,
    protocol: u64,
    floor: u64,
) -> MockDaemon {
    spawn_mock(
        dir,
        "hello.sock",
        vec![hello_response(family, protocol, floor); 100],
    )
}

/// Serve connections that accept, read one request, and never answer.
///
/// The stalled peer: a daemon-shaped socket that accepts the
/// exchange and stalls it forever — the liveness pins' subject.
fn spawn_stalled_mock(dir: &tempfile::TempDir, name: &str) -> MockDaemon {
    let socket = dir.path().join(name);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::UnixListener::bind(&socket).expect("the stalled socket binds");
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            std::future::pending::<()>().await;
        }
    });
    MockDaemon {
        socket,
        requests,
        handle,
    }
}

/// The handshake response bytes for one family/version/floor triple.
fn hello_response(family: &str, protocol: u64, floor: u64) -> Vec<u8> {
    let body = serde_json::json!({
        "family": family,
        "protocol": protocol,
        "floor": floor,
    })
    .to_string();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\
         Connection: close\r\n\r\n{body}",
        len = body.len()
    )
    .into_bytes()
}

/// A JSON response with a declared length.
fn json_response(status: &str, body: &serde_json::Value) -> Vec<u8> {
    let body = body.to_string();
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\
         Connection: close\r\n\r\n{body}",
        len = body.len()
    )
    .into_bytes()
}

/// A scripted event-stream answer whose content type and one frame the test controls.
fn sse_response(content_type: &str, frame: &str) -> Vec<u8> {
    format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n{frame}")
        .into_bytes()
}

/// Read one full request (head + Content-Length body) from `stream`.
async fn read_one_request<Stream>(stream: &mut Stream) -> RecordedRequest
where
    Stream: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        let read = stream.read(&mut chunk).await.expect("the request reads");
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(offset) = find_head_end(&buffer) {
            break offset;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let mut length = 0usize;
    let mut authorization = None;
    let mut host = String::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length")
            && let Ok(parsed) = value.trim().parse::<usize>()
        {
            length = parsed;
        }
        if name.trim().eq_ignore_ascii_case("authorization") {
            authorization = Some(value.trim().to_string());
        }
        if name.trim().eq_ignore_ascii_case("host") {
            host = value.trim().to_string();
        }
    }
    let body_start = head_end.saturating_add(4);
    let mut body = buffer[body_start..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk).await.expect("the body reads");
        body.extend_from_slice(&chunk[..read]);
    }
    RecordedRequest {
        method,
        path,
        host,
        authorization,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|quad| quad == b"\r\n\r\n")
}

/// A minimal, valid trajectory event as the wire carries it.
fn run_started_event(seq: u64) -> serde_json::Value {
    serde_json::json!({
        "seq": seq,
        "ts": "2026-10-07T00:00:00Z",
        "run_id": 1,
        "session_id": "0b6bb8ab-6c5a-4f78-9c86-bd294c1a9e12",
        "turn": null,
        "kind": "run.started",
        "data": {"input": "hello"},
    })
}

fn client_test_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("the tempdir creates")
}

/// A mock daemon serving scripted raw responses over TCP.
///
/// The remote-transport twin of [`MockDaemon`]: same scripted-bytes
/// contract, same request recording, at a loopback port instead of
/// a socket path.
struct TcpMockDaemon {
    /// The mock's loopback port.
    port: u16,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl TcpMockDaemon {
    /// Every request observed so far.
    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Stop the accept loop.
    fn stop(self) {
        self.handle.abort();
    }
}

/// Serve `responses` over TCP sequentially, one per connection.
async fn spawn_mock_tcp(responses: Vec<Vec<u8>>) -> TcpMockDaemon {
    spawn_mock_tcp_on("127.0.0.1", responses).await
}

/// Serve `responses` over TCP at one bind host, one per connection.
///
/// The addressable twin of [`spawn_mock_tcp`]: a pin that reaches the
/// mock through a host literal of its own choosing — the `IPv6`
/// loopback a bracketed-authority pin needs.
async fn spawn_mock_tcp_on(bind: &str, responses: Vec<Vec<u8>>) -> TcpMockDaemon {
    let listener = TcpListener::bind((bind, 0))
        .await
        .expect("the mock tcp listener binds");
    let port = listener
        .local_addr()
        .expect("the tcp address resolves")
        .port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for response in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            stream.write_all(&response).await.expect("the mock writes");
        }
    });
    TcpMockDaemon {
        port,
        requests,
        handle,
    }
}

/// A loopback port that reliably refuses connections.
///
/// Binds a listener, notes its port, then drops the listener — the
/// address answers nothing afterward, a refusal the test controls
/// rather than borrowing a well-known port any service on the
/// machine might hold and stall instead of refusing.
async fn a_refused_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the probe listener binds");
    listener
        .local_addr()
        .expect("the probe address resolves")
        .port()
}

#[tokio::test]
async fn connect_performs_the_versioned_handshake() {
    let dir = client_test_dir();
    let mock = spawn_hello_mock(&dir, "loopctl", 1, 1);
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the family member at the floor connects");
    assert_eq!(
        client.handshake(),
        ("loopctl".to_string(), 1, 1),
        "the negotiated snapshot reports what the daemon said"
    );
    assert_eq!(
        mock.requests()[0].path,
        "/v1/hello",
        "the handshake rides the versioned path"
    );
    assert_eq!(
        mock.requests()[0].host,
        "loopctl",
        "the socket endpoint carries the fixed family authority — a unix \
         socket has no authority of its own"
    );
    mock.stop();
}

#[tokio::test]
async fn handshake_floor_refuses_mismatched_family_member() {
    let dir = client_test_dir();
    let mock = spawn_hello_mock(&dir, "somebody-else", 1, 1);
    let rejection = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        matches!(
            rejection,
            Err(ClientError::FamilyMismatch { ref daemon, ref client }) if daemon == "somebody-else" && client == "loopctl"
        ),
        "a foreign family refuses naming both sides: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_older_server_refuses_within_the_floor() {
    let dir = client_test_dir();
    let mock = spawn_hello_mock(&dir, "loopctl", 0, 0);
    let rejection = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        matches!(
            rejection,
            Err(ClientError::ServerTooOld {
                daemon_protocol: 0,
                client_floor: 1
            })
        ),
        "a server below the floor refuses with both numbers: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_newer_server_refuses_by_its_own_floor() {
    let dir = client_test_dir();
    let mock = spawn_hello_mock(&dir, "loopctl", 2, 2);
    let rejection = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        matches!(
            rejection,
            Err(ClientError::ServerTooNew {
                daemon_floor: 2,
                client_protocol: 1
            })
        ),
        "a server whose floor excludes this client refuses: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_content_length_response_body_decodes() {
    let dir = client_test_dir();
    let body = serde_json::json!({"family": "loopctl", "protocol": 1, "floor": 1});
    let mock = spawn_mock(&dir, "len.sock", vec![json_response("200 OK", &body)]);
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        client.is_ok(),
        "a declared-length body decodes into the handshake document: {client:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_chunked_response_body_decodes() {
    let dir = client_test_dir();
    let body = r#"{"family":"loopctl","protocol":1,"floor":1}"#;
    let head = String::new();
    let chunked = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\
         Connection: close\r\n\r\n{head}"
    );
    let mut bytes = chunked.into_bytes();
    let split = body.len() / 2;
    let (first, second) = body.split_at(split);
    for part in [first, second] {
        bytes.extend_from_slice(format!("{:x}\r\n", part.len()).as_bytes());
        bytes.extend_from_slice(part.as_bytes());
        bytes.extend_from_slice(b"\r\n");
    }
    bytes.extend_from_slice(b"0\r\n\r\n");
    let mock = spawn_mock(&dir, "chunked.sock", vec![bytes]);
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        client.is_ok(),
        "a chunked body decodes through the transfer coding: {client:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_close_delimited_response_body_decodes() {
    let dir = client_test_dir();
    let body = r#"{"family":"loopctl","protocol":1,"floor":1}"#;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}"
    );
    let mock = spawn_mock(&dir, "close.sock", vec![response.into_bytes()]);
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        client.is_ok(),
        "an unframed body reads until the peer closes: {client:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn error_responses_surface_status_code_and_message() {
    let dir = client_test_dir();
    let error_body = serde_json::json!({
        "error": {"code": "loop_paused", "message": "the loop is paused"}
    });
    let mock = spawn_mock(
        &dir,
        "error.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            json_response("409 Conflict", &error_body),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the handshake succeeds");
    let rejection = client.start_run("watch-repos", "go").await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Api { status: 409, code, message })
                if code.as_deref() == Some("loop_paused") && message == "the loop is paused"
        ),
        "the daemon's status, code, and message surface verbatim: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn verbs_hit_their_versioned_paths_with_json_bodies() {
    let dir = client_test_dir();
    let loops = serde_json::json!([
        {"id": "watch-repos", "status": "running", "model": null}
    ]);
    let one_loop = serde_json::json!({"id": "watch-repos", "status": "running", "model": null});
    let run_id = "2f0d6a6e-0d3a-4d8e-9f2b-2b1e93bd78f1";
    let run_handle = serde_json::json!({ "run_id": run_id });
    let run_state = serde_json::json!({
        "run_id": run_id,
        "loop_id": "watch-repos",
        "status": "completed",
        "stop_reason": null,
    });
    let gates = serde_json::json!([
        {"gate_id": "g1", "loop_id": "watch-repos", "tool": "Bash", "prompt": "run rm?"}
    ]);
    let schedules = serde_json::json!([
        {"name": "nightly", "suspended": false}
    ]);
    let mock = spawn_mock(
        &dir,
        "verbs.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            json_response("200 OK", &loops),
            json_response("200 OK", &one_loop),
            json_response("200 OK", &run_handle),
            json_response("200 OK", &run_state),
            json_response("200 OK", &serde_json::json!({})),
            json_response("200 OK", &gates),
            json_response("200 OK", &serde_json::json!({})),
            json_response("200 OK", &serde_json::json!({})),
            json_response("200 OK", &schedules),
            json_response("200 OK", &serde_json::json!({})),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the handshake succeeds");
    client.list_loops().await.expect("list");
    client.get_loop("watch-repos").await.expect("get loop");
    client
        .start_run("watch-repos", "go")
        .await
        .expect("start run");
    client
        .get_run(uuid::Uuid::parse_str(run_id).unwrap())
        .await
        .expect("get run");
    client
        .stop_run(uuid::Uuid::parse_str(run_id).unwrap())
        .await
        .expect("stop run");
    client.pending_gates().await.expect("gates");
    client
        .approve_gate("g1", "reviewed")
        .await
        .expect("approve");
    client.deny_gate("g2", "not today").await.expect("deny");
    client.list_schedules().await.expect("schedules");
    client.run_schedule_now("nightly").await.expect("run now");

    let requests = mock.requests();
    let paths: Vec<&str> = requests.iter().map(|r| r.path.as_str()).collect();
    assert_eq!(
        paths,
        vec![
            "/v1/hello",
            "/v1/loops",
            "/v1/loops/watch-repos",
            "/v1/loops/watch-repos/runs",
            "/v1/runs/2f0d6a6e-0d3a-4d8e-9f2b-2b1e93bd78f1",
            "/v1/runs/2f0d6a6e-0d3a-4d8e-9f2b-2b1e93bd78f1/stop",
            "/v1/gates",
            "/v1/gates/g1/approve",
            "/v1/gates/g2/deny",
            "/v1/schedules",
            "/v1/schedules/nightly/run-now",
        ],
        "every verb rides its versioned path"
    );
    let methods: Vec<&str> = requests.iter().map(|r| r.method.as_str()).collect();
    assert_eq!(
        methods,
        vec![
            "GET", "GET", "GET", "POST", "GET", "POST", "GET", "POST", "POST", "GET", "POST"
        ],
        "every verb carries its method"
    );
    assert_eq!(
        requests[3].body, r#"{"input":"go"}"#,
        "the run start carries its input as JSON"
    );
    assert_eq!(
        requests[7].body, r#"{"reason":"reviewed"}"#,
        "the approval carries its reason as JSON"
    );
    assert_eq!(
        requests[8].body, r#"{"reason":"not today"}"#,
        "the denial carries its reason as JSON"
    );
    mock.stop();
}

/// One SSE mock connection the test pushes frames into.
struct SseMock {
    daemon: MockDaemon,
    frames: tokio::sync::mpsc::UnboundedSender<String>,
}

/// The TCP twin of [`SseMock`]: handshake first, then one held-open
/// event-stream connection the test pushes frames into.
struct SseTcpMock {
    daemon: TcpMockDaemon,
    frames: tokio::sync::mpsc::UnboundedSender<String>,
}

/// Serve the handshake, then one held-open event-stream connection
/// over TCP.
///
/// Path-aware like the socket twin: `/v1/hello` answers the
/// handshake document, anything else answers the SSE head and
/// forwards pushed frames until the sender drops.
async fn spawn_sse_mock_tcp() -> SseTcpMock {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the events tcp socket binds");
    let port = listener
        .local_addr()
        .expect("the tcp address resolves")
        .port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (frames_tx, mut frames_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request.clone());
            if request.path == "/v1/hello" {
                stream
                    .write_all(&hello_response("loopctl", 1, 1))
                    .await
                    .expect("the handshake");
                continue;
            }
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                        Connection: close\r\n\r\n";
            stream.write_all(head.as_bytes()).await.expect("the head");
            while let Some(frame) = frames_rx.recv().await {
                stream.write_all(frame.as_bytes()).await.expect("the frame");
            }
            return;
        }
    });
    SseTcpMock {
        daemon: TcpMockDaemon {
            port,
            requests,
            handle,
        },
        frames: frames_tx,
    }
}

/// Serve the handshake, then one held-open event-stream connection.
///
/// Path-aware: a `/v1/hello` request answers the handshake document,
/// anything else answers the SSE head and forwards pushed frames
/// until the sender drops.
fn spawn_sse_mock(dir: &tempfile::TempDir) -> SseMock {
    let socket = dir.path().join("events.sock");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (frames_tx, mut frames_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let listener = UnixListener::bind(&socket).expect("the events socket binds");
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request.clone());
            if request.path == "/v1/hello" {
                stream
                    .write_all(&hello_response("loopctl", 1, 1))
                    .await
                    .expect("the handshake");
                continue;
            }
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                        Connection: close\r\n\r\n";
            stream.write_all(head.as_bytes()).await.expect("the head");
            while let Some(frame) = frames_rx.recv().await {
                stream.write_all(frame.as_bytes()).await.expect("the frame");
            }
            return;
        }
    });
    SseMock {
        daemon: MockDaemon {
            socket,
            requests,
            handle,
        },
        frames: frames_tx,
    }
}

#[tokio::test]
async fn the_event_stream_yields_events_with_monotone_cursors() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");

    let first = run_started_event(1).to_string();
    let second_json = run_started_event(2).to_string();
    let (second_head, second_tail) = second_json.split_at(second_json.len() / 2);
    sse.frames
        .send(": keep-alive comment\r\n\r\n".to_string())
        .ok();
    sse.frames
        .send(format!("id: 7\ndata: {first}\r\n\r\nretry: 100\r\n\r\n"))
        .ok();
    sse.frames
        .send(format!(
            "id: 8\r\ndata: {second_head}\r\ndata: {second_tail}\r\n\r\n"
        ))
        .ok();

    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the first event arrives")
        .expect("the stream is open")
        .expect("the first frame parses");
    assert_eq!(
        event.cursor, 7,
        "the SSE id is the delivered cursor: {event:?}"
    );
    assert_eq!(
        event.event.seq, 1,
        "the payload is the typed trajectory event: {event:?}"
    );
    assert!(
        matches!(
            event.event.kind,
            loopctl::memory::trajectory::TrajectoryEventKind::RunStarted
        ),
        "the kind deserializes into the typed enum: {event:?}"
    );
    let second = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the second event arrives")
        .expect("the stream is open")
        .expect("the split frame joins and parses");
    assert!(
        second.cursor > event.cursor,
        "delivered cursors strictly increase across frames: {} then {}",
        event.cursor,
        second.cursor
    );
    assert_eq!(
        second.cursor, 8,
        "the multi-data frame's id is its cursor: {second:?}"
    );
    assert_eq!(
        second.event.seq, 2,
        "the joined payload is the second fixture event: {second:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_multi_data_frame_joins_into_one_event() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(3).to_string();
    let (head, tail) = payload.split_at(payload.len() / 2);
    sse.frames
        .send(format!("id: 11\r\ndata: {head}\r\ndata: {tail}\r\n\r\n"))
        .ok();
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the event arrives")
        .expect("the stream is open")
        .expect("the split payload joins and parses");
    assert_eq!(
        event.cursor, 11,
        "the joined frame delivers with its cursor: {event:?}"
    );
    assert_eq!(
        event.event.seq, 3,
        "the joined payload is the third fixture event: {event:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_malformed_event_payload_surfaces_as_a_typed_error() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    sse.frames
        .send("id: 12\r\ndata: not-json\r\n\r\n".to_string())
        .ok();
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(outcome, Err(ClientError::MalformedResponse(_))),
        "a payload outside the interchange shape is a typed error: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_in_process_subscription_delivers_through_the_shared_event_type() {
    let event_one: loopctl::memory::trajectory::TrajectoryEvent =
        serde_json::from_value(run_started_event(1)).expect("the fixture parses");
    let event_two: loopctl::memory::trajectory::TrajectoryEvent =
        serde_json::from_value(run_started_event(2)).expect("the fixture parses");
    let source = futures::stream::iter(vec![
        Ok(loopctl_client::LedgerEvent {
            cursor: 1,
            event: event_one,
        }),
        Ok(loopctl_client::LedgerEvent {
            cursor: 2,
            event: event_two,
        }),
    ]);
    let mut stream = loopctl_client::EventStream::from_pin_box(Box::pin(source));
    let first = stream
        .next()
        .await
        .expect("the first arrives")
        .expect("parses");
    assert_eq!(
        first.cursor, 1,
        "a host-authored source delivers through the shared event type: {first:?}"
    );
    let second = stream
        .next()
        .await
        .expect("the second arrives")
        .expect("parses");
    assert_eq!(second.cursor, 2);
    assert!(
        stream.next().await.is_none(),
        "the adapted source ends when the host source ends"
    );
}

/// Serve the handshake, then one chunked event-stream connection
/// fed piece by piece.
///
/// Each chunk-syntax piece (size lines, data, separators, the
/// terminator) is written as its own write with a sleep between, so
/// the client is polled to `Pending` mid-syntax between pieces — the
/// cancel-safety conditions the close-delimited mock can never
/// produce.
fn spawn_chunked_sse_mock(dir: &tempfile::TempDir, pieces: Vec<String>) -> PathBuf {
    let socket = dir.path().join("chunked-events.sock");
    let listener = UnixListener::bind(&socket).expect("the chunked socket binds");
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let request = read_one_request(&mut stream).await;
            if request.path == "/v1/hello" {
                stream
                    .write_all(&hello_response("loopctl", 1, 1))
                    .await
                    .expect("the handshake");
                continue;
            }
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                        Transfer-Encoding: chunked\r\n\r\n";
            stream.write_all(head.as_bytes()).await.expect("the head");
            for piece in pieces {
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                stream.write_all(piece.as_bytes()).await.expect("a piece");
            }
            return;
        }
    });
    socket
}

/// One SSE frame's bytes (`id:` + `data:` + the blank dispatch line).
fn sse_frame(cursor: u64, payload: &serde_json::Value) -> String {
    format!("id: {cursor}\r\ndata: {payload}\r\n\r\n")
}

/// The chunk-syntax pieces for one event, size line split in two.
fn chunked_frame_pieces(frame: &str) -> Vec<String> {
    let hex = format!("{:x}", frame.len());
    vec![
        hex.get(0..1).unwrap_or("0").to_string(),
        hex.get(1..).unwrap_or("").to_string() + "\r\n",
        frame.to_string(),
        "\r\n".to_string(),
    ]
}

#[tokio::test]
async fn a_chunked_sse_feed_with_split_writes_delivers_events() {
    let dir = client_test_dir();
    let first = sse_frame(7, &run_started_event(1));
    let second = sse_frame(8, &run_started_event(2));
    let mut pieces = chunked_frame_pieces(&first);
    pieces.extend(chunked_frame_pieces(&second));
    pieces.push("0\r\n\r\n".to_string());
    let socket = spawn_chunked_sse_mock(&dir, pieces);
    let client = DaemonClient::connect(ConnectOptions::new(socket))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");

    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the first event arrives")
        .expect("the stream is open")
        .expect("the first chunked frame parses");
    assert_eq!(
        event.cursor, 7,
        "a chunked event stream delivers with its cursor: {event:?}"
    );
    let next = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the second event arrives")
        .expect("the stream is open")
        .expect("the second chunked frame parses");
    assert_eq!(next.cursor, 8);
    let end = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next()).await;
    assert!(
        matches!(end, Ok(None)),
        "the zero-size terminator ends the stream cleanly: {end:?}"
    );
}

#[test]
fn construction_still_works_through_the_constructors() {
    let summary = loopctl_client::LoopSummary::new("watch-repos", "running");
    assert_eq!(summary.id, "watch-repos");
    assert_eq!(summary.model, None);
    let handle = loopctl_client::RunHandle::new(uuid::Uuid::nil());
    assert_eq!(handle.run_id, uuid::Uuid::nil());
    let state = loopctl_client::RunState::new(uuid::Uuid::nil(), "watch-repos", "completed");
    assert_eq!(state.status, "completed");
    assert_eq!(state.stop_reason, None);
    let pending = loopctl_client::GatePending::new("g1", "watch-repos", "Bash", "run rm?");
    assert_eq!(pending.tool, "Bash");
    let schedule = loopctl_client::ScheduleSummary::new("nightly", false);
    assert!(!schedule.suspended);
}

#[test]
fn optional_fields_deserialize_leniently_from_older_daemons() {
    let loop_json = r#"{"id":"watch-repos","status":"idle"}"#;
    let summary: loopctl_client::LoopSummary =
        serde_json::from_str(loop_json).expect("a loop envelope omitting model still parses");
    assert_eq!(
        summary.model, None,
        "the omitted optional field defaults rather than failing the envelope"
    );
    let run_json = r#"{"run_id":"2f0d6a6e-0d3a-4d8e-9f2b-2b1e93bd78f1","loop_id":"watch-repos","status":"running"}"#;
    let state: loopctl_client::RunState =
        serde_json::from_str(run_json).expect("a run envelope omitting stop_reason still parses");
    assert_eq!(state.stop_reason, None);
    let schedule_json = r#"{"name":"nightly"}"#;
    let schedule: loopctl_client::ScheduleSummary = serde_json::from_str(schedule_json)
        .expect("a schedule envelope omitting suspended still parses");
    assert!(
        !schedule.suspended,
        "the omitted boolean defaults to false — not suspended"
    );
}

#[tokio::test]
async fn a_stale_cursor_returns_typed_gone_for_relisting() {
    let dir = client_test_dir();
    let body = serde_json::json!({
        "error": {"code": "cursor_expired", "message": "relist"}
    });
    let socket = dir.path().join("gone.sock");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let listener = UnixListener::bind(&socket).expect("the gone socket binds");
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request.clone());
            let response = if request.path == "/v1/hello" {
                hello_response("loopctl", 1, 1)
            } else {
                json_response("410 Gone", &body)
            };
            stream.write_all(&response).await.expect("the write");
        }
    });
    let client = DaemonClient::connect(ConnectOptions::new(socket.clone()))
        .await
        .expect("the first request is the subscription");
    let rejection = client.subscribe("watch-repos", Some(9)).await;
    match rejection {
        Err(ClientError::CursorExpired { since: Some(9) }) => {}
        other => panic!(
            "a 410 answers the typed relist error naming the cursor: {other:?}",
            other = other.map(|_| ())
        ),
    }
    let paths: Vec<String> = requests
        .lock()
        .unwrap()
        .iter()
        .map(|request| request.path.clone())
        .collect();
    assert_eq!(
        paths,
        vec!["/v1/hello", "/v1/loops/watch-repos/events?since=9",],
        "the handshake precedes; the since cursor rides the query"
    );
    handle.abort();
    drop(client);
}

#[tokio::test]
async fn a_hello_miss_answers_not_a_daemon() {
    let dir = client_test_dir();
    let missing = json_response(
        "404 Not Found",
        &serde_json::json!({"error": {"code": None::<String>, "message": "no such route"}}),
    );
    let mock = spawn_mock(&dir, "notadaemon.sock", vec![missing; 3]);
    let rejection = DaemonClient::connect(ConnectOptions::new(mock.socket())).await;
    assert!(
        matches!(rejection, Err(ClientError::NotADaemon { .. })),
        "a peer whose /v1/hello is a miss refuses as not-a-daemon: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_verb_404_surfaces_not_found_with_the_envelope_s_what() {
    let dir = client_test_dir();
    let mock = spawn_mock(
        &dir,
        "notfound.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            json_response(
                "404 Not Found",
                &serde_json::json!({
                    "error": {"code": "no_loop", "message": "unknown loop", "what": "loop ghost"}
                }),
            ),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the handshake succeeds");
    let rejection = client.get_loop("ghost").await;
    match rejection {
        Err(ClientError::NotFound { what }) => assert_eq!(
            what, "loop ghost",
            "the envelope's what names the missing resource"
        ),
        other => panic!("a verb 404 maps to NotFound: {other:?}"),
    }
    mock.stop();
}

#[tokio::test]
async fn a_verb_404_without_the_envelope_names_the_resource_fallback() {
    let dir = client_test_dir();
    let mock = spawn_mock(
        &dir,
        "bare404.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            json_response("404 Not Found", &serde_json::Value::Null),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the handshake succeeds");
    let rejection = client.get_loop("ghost").await;
    match rejection {
        Err(ClientError::NotFound { what }) => assert_eq!(
            what, "the requested resource",
            "an envelope-less 404 falls back to the honest generic name"
        ),
        other => panic!("a bare 404 maps to NotFound: {other:?}"),
    }
    mock.stop();
}

#[tokio::test]
async fn a_failing_spawn_action_surfaces_spawn_failed() {
    let dir = client_test_dir();
    let action: SpawnAction = Arc::new(|| Err(std::io::Error::other("no loopctl binary on PATH")));
    let options = ConnectOptions::new(dir.path().join("never.sock"))
        .with_spawn(action, RetryWindow::default());
    let rejection = DaemonClient::connect(options).await;
    match rejection {
        Err(ClientError::SpawnFailed { detail }) => assert!(
            detail.contains("no loopctl binary"),
            "the action's own diagnosis rides the error: {detail}"
        ),
        other => panic!("a failing spawn action surfaces SpawnFailed: {other:?}"),
    }
}

#[tokio::test]
async fn an_unset_runtime_dir_is_a_typed_error() {
    let env = EnvGuard::acquire(&["XDG_RUNTIME_DIR"]);
    env.remove("XDG_RUNTIME_DIR");
    let rejection = DaemonClient::connect_default(None).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketPathUnavailable)),
        "an unset XDG_RUNTIME_DIR names the missing standard path: {rejection:?}"
    );
}

#[tokio::test]
async fn an_error_without_the_envelope_surfaces_the_lossy_body_text() {
    let dir = client_test_dir();
    let plain = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 22\r\nConnection: close\r\n\r\ninternal daemon error!";
    let mock = spawn_mock(
        &dir,
        "bare500.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            plain.as_bytes().to_vec(),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the handshake succeeds");
    let rejection = client.list_loops().await;
    match rejection {
        Err(ClientError::Api {
            status: 500,
            code: None,
            message,
        }) => assert_eq!(
            message, "internal daemon error!",
            "an envelope-less error surfaces the body text as the message"
        ),
        other => panic!("a bare 5xx maps to Api with no code: {other:?}"),
    }
    mock.stop();
}

#[tokio::test]
async fn a_data_frame_without_an_id_is_refused() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(1).to_string();
    sse.frames.send(format!("data: {payload}\r\n\r\n")).ok();
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(outcome, Err(ClientError::MalformedResponse(_))),
        "a data frame without an id cannot advance the cursor contract: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_non_numeric_id_cursor_is_refused() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(1).to_string();
    sse.frames
        .send(format!("id: soon\ndata: {payload}\r\n\r\n"))
        .ok();
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(outcome, Err(ClientError::MalformedResponse(_))),
        "a non-numeric id is a malformed cursor, never a silent zero: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_absent_socket_is_a_typed_error_not_a_panic() {
    let dir = client_test_dir();
    let rejection = DaemonClient::connect(ConnectOptions::new(dir.path().join("never.sock"))).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketAbsent)),
        "an absent socket is the typed absence the degrade path keys on: {rejection:?}"
    );
}

#[tokio::test]
async fn autospawn_spawns_then_connects_within_the_bounded_window() {
    let dir = client_test_dir();
    let socket = dir.path().join("late.sock");
    let late_socket = socket.clone();
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        let late_socket = late_socket.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            std::fs::remove_file(&late_socket).ok();
            let responses = vec![hello_response("loopctl", 1, 1); 10];
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            runtime.block_on(async move {
                let listener = UnixListener::bind(&late_socket)?;
                for response in responses {
                    let (mut stream, _) = listener.accept().await?;
                    let mut buffer = Vec::new();
                    let mut chunk = [0u8; 1024];
                    loop {
                        let read = stream.read(&mut chunk).await?;
                        buffer.extend_from_slice(&chunk[..read]);
                        if buffer.windows(4).any(|quad| quad == b"\r\n\r\n") {
                            break;
                        }
                    }
                    stream.write_all(&response).await?;
                }
                Ok::<(), std::io::Error>(())
            })
        });
        Ok(())
    });
    let options = ConnectOptions::new(socket).with_spawn(
        action,
        RetryWindow {
            attempts: 50,
            delay: std::time::Duration::from_millis(20),
        },
    );
    let client = DaemonClient::connect(options)
        .await
        .expect("the spawned daemon appears inside the window");
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the spawn action ran exactly once"
    );
    let (_, _, _) = client.handshake();
}

#[tokio::test]
async fn an_unspawnable_daemon_gives_up_after_the_bounded_window() {
    let dir = client_test_dir();
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(dir.path().join("never.sock")).with_spawn(
        action,
        RetryWindow {
            attempts: 3,
            delay: std::time::Duration::from_millis(10),
        },
    );
    let started = std::time::Instant::now();
    let rejection = DaemonClient::connect(options).await;
    let elapsed = started.elapsed();
    match rejection {
        Err(ClientError::ConnectTimeout { window_ms }) => assert!(
            (20..=1000).contains(&window_ms),
            "the reported span is what the window actually waited — two 10 ms delays \
             delimit it, never a formula about it: {window_ms}"
        ),
        other => panic!("a daemon that never appears surfaces the timeout: {other:?}"),
    }
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "the window is bounded — gave up in {elapsed:?}"
    );
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the spawn action ran exactly once across the window"
    );
}

#[tokio::test]
async fn no_autospawn_env_disables_spawning() {
    let env = EnvGuard::acquire(&["LOOPCTL_NO_AUTOSPAWN", "XDG_RUNTIME_DIR"]);
    env.set("LOOPCTL_NO_AUTOSPAWN", "1");
    let dir = client_test_dir();
    env.set("XDG_RUNTIME_DIR", dir.path().to_str().expect("utf8 path"));
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let rejection = DaemonClient::connect_default(Some(action)).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketAbsent)),
        "the opt-out downgrades spawning to the typed absence: {rejection:?}"
    );
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        0,
        "the spawn action never ran under the opt-out"
    );
}

#[tokio::test]
async fn default_socket_path_follows_xdg_runtime_dir() {
    let env = EnvGuard::acquire(&["LOOPCTL_NO_AUTOSPAWN", "XDG_RUNTIME_DIR"]);
    env.remove("LOOPCTL_NO_AUTOSPAWN");
    let dir = client_test_dir();
    let runtime_root = dir.path().join("run");
    std::fs::create_dir_all(&runtime_root).expect("the runtime dir creates");
    env.set("XDG_RUNTIME_DIR", runtime_root.to_str().expect("utf8 path"));
    let socket = runtime_root.join("loopctl").join("loopctl.sock");
    std::fs::create_dir_all(socket.parent().expect("parent")).expect("the socket dir");
    let listener = UnixListener::bind(&socket).expect("the standard socket binds");
    let handle = tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buffer = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        buffer.extend_from_slice(&chunk[..read]);
                        if buffer.windows(4).any(|quad| quad == b"\r\n\r\n") {
                            break;
                        }
                    }
                }
            }
            let response = hello_response("loopctl", 1, 1);
            stream.write_all(&response).await.ok();
        }
    });
    let client = DaemonClient::connect_default(None).await;
    assert!(
        client.is_ok(),
        "the standard path resolves and connects: {client:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn one_interface_serves_both_modes() {
    let dir = client_test_dir();
    let mock = spawn_mock(
        &dir,
        "modes.sock",
        vec![
            json_response(
                "200 OK",
                &serde_json::json!({
                    "family": "loopctl", "protocol": 1, "floor": 1
                }),
            ),
            json_response("200 OK", &serde_json::json!([])),
        ],
    );
    let daemon: Box<dyn LoopctlClient> = Box::new(
        DaemonClient::connect(ConnectOptions::new(mock.socket()))
            .await
            .unwrap(),
    );
    let in_process: Box<dyn LoopctlClient> = Box::new(InProcessFake);
    let daemon_loops = daemon.list_loops().await;
    let in_process_loops = in_process.list_loops().await;
    assert!(
        daemon_loops.is_ok() && in_process_loops.is_ok(),
        "both modes answer the same verb through one object-safe interface: \
         {daemon_loops:?} {in_process_loops:?}"
    );
    assert_eq!(
        in_process_loops.unwrap()[0].id,
        "in-process",
        "the in-process mode's answer flows through the shared surface"
    );
    let mut in_process_events = in_process
        .subscribe("watch-repos", None)
        .await
        .expect("the in-process mode's subscription opens through the shared event type");
    let delivered = in_process_events
        .next()
        .await
        .expect("the in-process event arrives")
        .expect("the in-process event parses");
    assert_eq!(
        delivered.cursor, 1,
        "subscribe is mode-unified in full — the stream surface, not only the verbs"
    );
    mock.stop();
}

#[tokio::test]
async fn a_remote_tcp_endpoint_performs_the_handshake_and_verbs() {
    let mock = spawn_mock_tcp(vec![
        hello_response("loopctl", 1, 1),
        json_response(
            "200 OK",
            &serde_json::json!([{"id": "watch-repos", "status": "idle", "model": null}]),
        ),
    ])
    .await;
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, Some("sekrit".to_string()));
    let client = DaemonClient::connect(options)
        .await
        .expect("the remote family member at the floor connects");
    let loops = client.list_loops().await.expect("the verb rides TCP");
    assert_eq!(
        loops[0].id, "watch-repos",
        "the remote answer decodes through the shared wire types"
    );
    let requests = mock.requests();
    assert_eq!(
        requests[0].path, "/v1/hello",
        "the remote handshake rides the versioned path"
    );
    assert_eq!(
        requests[0].host,
        format!("127.0.0.1:{}", mock.port),
        "the remote request head addresses the endpoint's authority, never a placeholder"
    );
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer sekrit"),
        "the bearer token rides every remote request, the handshake included"
    );
    assert_eq!(
        requests[1].authorization.as_deref(),
        Some("Bearer sekrit"),
        "the verb carries the same credentials as the handshake"
    );
    mock.stop();
}

#[tokio::test]
async fn an_ipv6_literal_endpoint_addresses_its_bracketed_authority() {
    let mock = spawn_mock_tcp_on(
        "::1",
        vec![
            hello_response("loopctl", 1, 1),
            json_response("200 OK", &serde_json::json!([])),
        ],
    )
    .await;
    let options = ConnectOptions::tcp("::1", mock.port, Some("sekrit".to_string()));
    let client = DaemonClient::connect(options)
        .await
        .expect("the IPv6 loopback endpoint connects");
    client
        .list_loops()
        .await
        .expect("the verb rides the IPv6 connection");
    let requests = mock.requests();
    assert_eq!(
        requests[0].host,
        format!("[::1]:{}", mock.port),
        "an IPv6 literal rides its authority bracketed, the form RFC 9110 \
         reserves — never the ambiguous unbracketed join"
    );
    mock.stop();
}

#[tokio::test]
async fn a_remote_tcp_endpoint_without_a_token_sends_no_authorization() {
    let mock = spawn_mock_tcp(vec![
        hello_response("loopctl", 1, 1),
        json_response("200 OK", &serde_json::json!([])),
    ])
    .await;
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, None);
    let client = DaemonClient::connect(options)
        .await
        .expect("a tokenless remote endpoint connects when the daemon allows it");
    client.list_loops().await.expect("the verb rides TCP");
    let requests = mock.requests();
    assert!(
        requests[0].authorization.is_none(),
        "no token means no Authorization header, never an empty one"
    );
    mock.stop();
}

#[tokio::test]
async fn an_unauthorized_answer_is_typed_over_tcp() {
    let refusal = json_response(
        "401 Unauthorized",
        &serde_json::json!({
            "error": {"code": "bad_token", "message": "the token was rejected"}
        }),
    );
    let mock = spawn_mock_tcp(vec![refusal]).await;
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, Some("wrong".to_string()));
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Unauthorized { status: 401, message })
                if message == "the token was rejected"
        ),
        "a refused credential answers the typed error, not a generic Api: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_forbidden_answer_is_typed_over_tcp() {
    let refusal = json_response(
        "403 Forbidden",
        &serde_json::json!({
            "error": {"code": "forbidden", "message": "the token may not do this"}
        }),
    );
    let mock = spawn_mock_tcp(vec![refusal]).await;
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, Some("limited".to_string()));
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Unauthorized { status: 403, message })
                if message == "the token may not do this"
        ),
        "a forbidden action answers the typed refusal with its own status, not a generic Api: \
         {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_bearer_token_with_control_characters_is_refused() {
    let mock = spawn_mock_tcp(vec![hello_response("loopctl", 1, 1)]).await;
    let injected = "t\r\nContent-Length: 0\r\n\r\nGET /v1/gates";
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, Some(injected.to_string()));
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Transport(detail)) if detail.contains("CR, LF, or NUL")
        ),
        "a token carrying CR/LF is a typed transport refusal, never a spliced header: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "the daemon received nothing — no byte of the injected token reached the wire"
    );
    mock.stop();
}

#[tokio::test]
async fn a_bearer_token_with_a_nul_byte_is_refused() {
    let mock = spawn_mock_tcp(vec![hello_response("loopctl", 1, 1)]).await;
    let injected = "sekrit\0Host: evil.example";
    let options = ConnectOptions::tcp("127.0.0.1", mock.port, Some(injected.to_string()));
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Transport(detail)) if detail.contains("CR, LF, or NUL")
        ),
        "a token carrying NUL is a typed transport refusal, never a spliced header: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "the daemon received nothing — no byte of the NUL-bearing token reached the wire"
    );
    mock.stop();
}

#[tokio::test]
async fn a_host_with_control_characters_is_refused() {
    let mock = spawn_mock_tcp(vec![hello_response("loopctl", 1, 1)]).await;
    let injected = "daemon\r\n.example:443";
    let options = ConnectOptions::tcp(injected, mock.port, None);
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Transport(detail)) if detail.contains("CR, LF, or NUL")
        ),
        "a host carrying CR/LF is a typed transport refusal, never a spliced Host header: \
         {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "the daemon received nothing — the injected host never reached the wire"
    );
    mock.stop();
}

#[tokio::test]
async fn the_event_stream_delivers_over_tcp_with_cursors() {
    let sse = spawn_sse_mock_tcp().await;
    let client = DaemonClient::connect(ConnectOptions::tcp("127.0.0.1", sse.daemon.port, None))
        .await
        .expect("the mock speaks the handshake over TCP");

    let first = run_started_event(1).to_string();
    let second = run_started_event(2).to_string();
    sse.frames
        .send(": keep-alive comment\r\n\r\n".to_string())
        .ok();
    sse.frames
        .send(format!("id: 7\ndata: {first}\r\n\r\n"))
        .ok();
    sse.frames
        .send(format!("id: 8\ndata: {second}\r\n\r\n"))
        .ok();

    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the remote stream opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the first remote event arrives")
        .expect("the remote stream is open")
        .expect("the first remote frame parses");
    assert_eq!(
        event.cursor, 7,
        "the SSE id is the cursor over TCP exactly as over the socket"
    );
    let second_event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the second remote event arrives")
        .expect("the remote stream is open")
        .expect("the second remote frame parses");
    assert_eq!(
        second_event.cursor, 8,
        "remote cursors advance frame by frame: {second_event:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_absent_remote_daemon_is_the_typed_absence() {
    let port = a_refused_port().await;
    let rejection = DaemonClient::connect(ConnectOptions::tcp("127.0.0.1", port, None)).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketAbsent)),
        "a refused remote connection is the same typed absence the degrade path keys on: {rejection:?}"
    );
}

#[tokio::test]
async fn auto_spawn_never_runs_for_remote_endpoints() {
    let port = a_refused_port().await;
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options =
        ConnectOptions::tcp("127.0.0.1", port, None).with_spawn(action, RetryWindow::default());
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketAbsent)),
        "an absent remote daemon surfaces the typed absence: {rejection:?}"
    );
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        0,
        "a daemon on another machine is not this process's to spawn — the action never ran"
    );
}

#[tokio::test]
async fn a_stalled_peer_answers_the_typed_deadline_rather_than_hanging() {
    let dir = client_test_dir();
    let mock = spawn_stalled_mock(&dir, "stalled.sock");
    let options =
        ConnectOptions::new(mock.socket()).with_deadline(std::time::Duration::from_millis(150));
    let rejection = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        DaemonClient::connect(options),
    )
    .await
    .expect("the stalled peer answers within the deadline instead of hanging");
    assert!(
        matches!(&rejection, Err(ClientError::Transport(detail))
            if detail.contains("did not answer within")),
        "a peer that accepts and stalls answers the typed deadline refusal: {rejection:?}"
    );
    assert_eq!(
        mock.requests().len(),
        1,
        "the stalled exchange was attempted exactly once"
    );
    mock.stop();
}

#[tokio::test]
async fn a_stalled_peer_cannot_outlast_the_spawn_window() {
    let dir = client_test_dir();
    let mock = spawn_stalled_mock(&dir, "stalled-window.sock");
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(mock.socket())
        .with_spawn(
            action,
            RetryWindow {
                attempts: 3,
                delay: std::time::Duration::from_millis(10),
            },
        )
        .with_deadline(std::time::Duration::from_millis(80));
    let rejection = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        DaemonClient::connect(options),
    )
    .await
    .expect("the window closes even against a stalled peer")
    .expect_err("no handshake ever completes against a stalled peer");
    match &rejection {
        ClientError::ConnectTimeout { window_ms } => assert!(
            (80..2000).contains(window_ms),
            "the configured 80 ms deadline bounded the whole window — the first stalled \
             attempt spent the aggregate budget, so the span names that deadline, not a \
             longer timeout: {window_ms}"
        ),
        other => panic!(
            "a stalled peer inside the window surfaces the window's typed timeout: {other:?}"
        ),
    }
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the action ran once — the window, not the stall, owns the wait"
    );
    mock.stop();
}

#[tokio::test]
async fn a_warming_daemons_transient_failures_retry_inside_the_window() {
    let dir = client_test_dir();
    let warming = json_response(
        "404 Not Found",
        &serde_json::json!({"error": {"code": "no_hello", "message": "warming up"}}),
    );
    let mock = spawn_mock(
        &dir,
        "warming.sock",
        vec![warming, hello_response("loopctl", 1, 1)],
    );
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(mock.socket()).with_spawn(
        action,
        RetryWindow {
            attempts: 4,
            delay: std::time::Duration::from_millis(10),
        },
    );
    let client = DaemonClient::connect(options)
        .await
        .expect("the window retries the warming daemon's transient refusal");
    assert_eq!(
        client.handshake().0,
        "loopctl",
        "the third connection's hello is the one negotiated"
    );
    assert_eq!(
        mock.requests().len(),
        2,
        "the refused first exchange and the successful second are both on the record"
    );
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the action ran once before the retrying began"
    );
    mock.stop();
}

#[tokio::test]
async fn a_non_event_stream_answer_refuses_rather_than_streaming_silently() {
    let dir = client_test_dir();
    let mock = spawn_mock(
        &dir,
        "not-sse.sock",
        vec![
            hello_response("loopctl", 1, 1),
            json_response("200 OK", &serde_json::json!([{"id": "watch-repos"}])),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let rejection = client.subscribe("watch-repos", None).await;
    assert!(
        matches!(&rejection, Err(ClientError::MalformedResponse(text))
            if text.contains("not an event stream")),
        "a 2xx that is not an event stream refuses typed, never a silent empty stream: \
         {:?}",
        rejection.err().map(|error| error.to_string())
    );
    mock.stop();
}

#[tokio::test]
async fn a_mixed_case_event_stream_media_type_is_accepted() {
    let dir = client_test_dir();
    let frame = format!("id: 5\ndata: {}\r\n\r\n", run_started_event(5));
    let mock = spawn_mock(
        &dir,
        "mixed-case-sse.sock",
        vec![
            hello_response("loopctl", 1, 1),
            sse_response("Text/Event-Stream", &frame),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the media type's case cannot refuse an event stream");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the event arrives")
        .expect("the stream is open")
        .expect("the frame parses");
    assert_eq!(
        event.cursor, 5,
        "the media type matches case-insensitively, the RFC's own rule: {event:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_event_stream_media_type_with_parameters_is_accepted() {
    let dir = client_test_dir();
    let frame = format!("id: 6\ndata: {}\r\n\r\n", run_started_event(6));
    let mock = spawn_mock(
        &dir,
        "parametered-sse.sock",
        vec![
            hello_response("loopctl", 1, 1),
            sse_response("text/event-stream; charset=utf-8", &frame),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("a parameter tail cannot refuse an event stream");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the event arrives")
        .expect("the stream is open")
        .expect("the frame parses");
    assert_eq!(
        event.cursor, 6,
        "everything after the media type is tolerated: {event:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_undelimited_frame_at_the_streams_end_refuses() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(9).to_string();
    sse.frames
        .send(format!("id: 9\r\ndata: {payload}\r\n"))
        .ok();
    drop(sse.frames);
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the stream's end arrives")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("ended mid-frame")),
        "a frame whose blank line never arrived refuses — its cursor never committed: \
         {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_cursor_expired_answer_without_a_cursor_names_the_beginning() {
    let dir = client_test_dir();
    let gone = json_response(
        "410 Gone",
        &serde_json::json!({"error": {"code": "cursor_gone", "message": "retention"}}),
    );
    let mock = spawn_mock(
        &dir,
        "gone.sock",
        vec![hello_response("loopctl", 1, 1), gone],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let Err(error) = client.subscribe("watch-repos", None).await else {
        panic!("a 410 answers the typed relist error");
    };
    assert!(
        matches!(&error, ClientError::CursorExpired { since: None }),
        "the no-cursor shape is carried, not invented: {error:?}"
    );
    let text = error.to_string();
    assert!(
        text.contains("from the beginning"),
        "a no-cursor expiry names the beginning, never a None cursor: {text}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_oversized_error_body_keeps_its_status() {
    let dir = client_test_dir();
    let mut flood = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\
         Connection: close\r\n\r\n"
        .as_bytes()
        .to_vec();
    flood.extend(std::iter::repeat_n(b'x', 11 * 1024 * 1024));
    let mock = spawn_mock(
        &dir,
        "flood.sock",
        vec![hello_response("loopctl", 1, 1), flood],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let rejection = client.list_loops().await;
    assert!(
        matches!(&rejection, Err(ClientError::Api { status: 500, .. })),
        "the status the daemon did send survives a flooded error body: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_cursor_expired_answer_is_typed_over_tcp() {
    let gone = json_response(
        "410 Gone",
        &serde_json::json!({"error": {"code": "cursor_gone", "message": "retention"}}),
    );
    let mock = spawn_mock_tcp(vec![hello_response("loopctl", 1, 1), gone]).await;
    let client = DaemonClient::connect(ConnectOptions::tcp("127.0.0.1", mock.port, None))
        .await
        .expect("the mock speaks the handshake over TCP");
    let rejection = client.subscribe("watch-repos", Some(9)).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::CursorExpired { since: Some(9) })
        ),
        "the relist contract answers typed over TCP exactly as over the socket: {:?}",
        rejection.err().map(|error| error.to_string())
    );
    mock.stop();
}

#[tokio::test]
async fn an_unauthorized_answer_is_typed_over_the_socket() {
    let dir = client_test_dir();
    let refusal = json_response(
        "401 Unauthorized",
        &serde_json::json!({
            "error": {"code": "bad_token", "message": "the token was rejected"}
        }),
    );
    let mock = spawn_mock(
        &dir,
        "unauth.sock",
        vec![hello_response("loopctl", 1, 1), refusal],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let rejection = client.list_loops().await;
    assert!(
        matches!(&rejection, Err(ClientError::Unauthorized { status: 401, message })
            if message == "the token was rejected"),
        "the credential refusal answers typed over the socket exactly as over TCP: \
         {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_cursorless_subscription_addresses_the_plain_path() {
    let dir = client_test_dir();
    let stream_head =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    let mock = spawn_mock(
        &dir,
        "plain-path.sock",
        vec![
            hello_response("loopctl", 1, 1),
            stream_head.as_bytes().to_vec(),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let ended = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the held connection settles when the mock ends");
    assert!(ended.is_none(), "the empty stream ends cleanly: {ended:?}");
    let requests = mock.requests();
    assert_eq!(
        requests[1].path, "/v1/loops/watch-repos/events",
        "a cursorless subscription addresses the plain path, never an empty ?since="
    );
    mock.stop();
}

#[tokio::test]
async fn a_zero_attempt_window_spawns_then_times_out_with_no_span() {
    let dir = client_test_dir();
    let absent = dir.path().join("absent-zero.sock");
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(absent).with_spawn(
        action,
        RetryWindow {
            attempts: 0,
            delay: std::time::Duration::from_millis(1),
        },
    );
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::ConnectTimeout { window_ms: 0 })
        ),
        "zero attempts: the spawn ran, no retry followed, and nothing was awaited inside \
         the window: {rejection:?}"
    );
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the action still ran — zero attempts bounds the retries, not the spawn"
    );
}

#[tokio::test]
async fn a_one_attempt_window_tries_once_and_spans_no_delay() {
    let dir = client_test_dir();
    let absent = dir.path().join("absent-one.sock");
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(absent).with_spawn(
        action,
        RetryWindow {
            attempts: 1,
            delay: std::time::Duration::from_millis(1),
        },
    );
    let rejection = DaemonClient::connect(options).await;
    match &rejection {
        Err(ClientError::ConnectTimeout { window_ms }) => assert!(
            *window_ms <= 100,
            "one attempt with no delay: the span is at most the one absent-socket try, \
             never a slept delay: {window_ms}"
        ),
        other => panic!("the one-attempt window surfaces the timeout: {other:?}"),
    }
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        1,
        "the action ran once for the one connect"
    );
}

#[tokio::test]
async fn a_one_attempt_window_completes_a_real_handshake() {
    let dir = client_test_dir();
    let socket = dir.path().join("one-shot.sock");
    let server_socket = socket.clone();
    let action: SpawnAction = Arc::new(move || {
        let listener = UnixListener::bind(&server_socket).expect("the spawned socket binds");
        tokio::spawn(async move {
            for _ in 0..2 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let request = read_one_request(&mut stream).await;
                let response = if request.path == "/v1/hello" {
                    hello_response("loopctl", 1, 1)
                } else {
                    json_response("200 OK", &serde_json::json!([]))
                };
                stream
                    .write_all(&response)
                    .await
                    .expect("the answer writes");
            }
        });
        Ok(())
    });
    let options = ConnectOptions::new(socket).with_spawn(
        action,
        RetryWindow {
            attempts: 1,
            delay: std::time::Duration::from_millis(1),
        },
    );
    let client = DaemonClient::connect(options)
        .await
        .expect("the window's single attempt completes the handshake");
    client
        .list_loops()
        .await
        .expect("the verb rides the one-attempt connection");
}

#[tokio::test]
async fn a_spawn_action_runs_once_per_connect() {
    let dir = client_test_dir();
    let absent = dir.path().join("absent-twice.sock");
    let spawns = Arc::new(AtomicUsize::new(0));
    let spawn_counter = Arc::clone(&spawns);
    let action: SpawnAction = Arc::new(move || {
        spawn_counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let options = ConnectOptions::new(absent).with_spawn(
        action,
        RetryWindow {
            attempts: 1,
            delay: std::time::Duration::from_millis(1),
        },
    );
    let first = DaemonClient::connect(options.clone()).await;
    let second = DaemonClient::connect(options).await;
    assert!(first.is_err() && second.is_err(), "no daemon ever appears");
    assert_eq!(
        spawns.load(Ordering::SeqCst),
        2,
        "the action runs once per connect — two connects, two runs"
    );
}

#[tokio::test]
async fn the_autospawn_optout_honors_only_the_exact_value_one() {
    for setting in ["0", "true"] {
        let env = EnvGuard::acquire(&["LOOPCTL_NO_AUTOSPAWN", "XDG_RUNTIME_DIR"]);
        env.set("LOOPCTL_NO_AUTOSPAWN", setting);
        let dir = client_test_dir();
        let runtime_root = dir.path().join("run");
        std::fs::create_dir_all(&runtime_root).expect("the runtime dir creates");
        env.set("XDG_RUNTIME_DIR", runtime_root.to_str().expect("utf8 path"));
        let socket = runtime_root.join("loopctl").join("loopctl.sock");
        let spawn_socket = socket.clone();
        let action: SpawnAction = Arc::new(move || {
            std::fs::create_dir_all(spawn_socket.parent().expect("the socket dir"))
                .expect("the socket dir creates");
            let listener = UnixListener::bind(&spawn_socket).expect("the spawned socket binds");
            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = Vec::new();
                    let mut chunk = [0u8; 1024];
                    loop {
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => break,
                            Ok(read) => {
                                buffer.extend_from_slice(&chunk[..read]);
                                if buffer.windows(4).any(|quad| quad == b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    let response = hello_response("loopctl", 1, 1);
                    stream.write_all(&response).await.ok();
                }
            });
            Ok(())
        });
        let client = DaemonClient::connect_default(Some(action)).await;
        assert!(
            client.is_ok(),
            "setting {setting:?} is not the opt-out — spawning proceeds and the daemon \
             answers: {client:?}"
        );
    }
}

#[tokio::test]
async fn a_repeated_id_field_delivers_the_last_value() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(1).to_string();
    sse.frames
        .send(format!("id: 5\r\nid: 9\r\ndata: {payload}\r\n\r\n"))
        .ok();
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open")
        .expect("the repeated id parses");
    assert_eq!(
        event.cursor, 9,
        "a repeated id field delivers the last value, the WHATWG-conform arm: {event:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_empty_id_field_refuses_the_frame() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(1).to_string();
    sse.frames
        .send(format!("id:\r\ndata: {payload}\r\n\r\n"))
        .ok();
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("was not numeric")),
        "an empty id cannot advance the cursor contract and refuses: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_empty_data_value_refuses_the_frame() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    sse.frames.send("id: 9\r\ndata:\r\n\r\n".to_string()).ok();
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("was not a trajectory event")),
        "a data field with no value is not a payload and refuses: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn an_unknown_field_is_tolerated() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let payload = run_started_event(1).to_string();
    sse.frames
        .send(format!("event: add\r\nid: 9\r\ndata: {payload}\r\n\r\n"))
        .ok();
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the frame arrives")
        .expect("the stream is open")
        .expect("the tolerated frame delivers");
    assert_eq!(
        event.cursor, 9,
        "an unknown field before id and data changes nothing: {event:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_length_cut_mid_frame_refuses() {
    let dir = client_test_dir();
    let frame = format!("id: 9\r\ndata: {}\r\n\r\n", run_started_event(9));
    let cut: String = frame.chars().take(20).collect();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{cut}",
        cut.len()
    );
    let mock = spawn_mock(
        &dir,
        "cut.sock",
        vec![hello_response("loopctl", 1, 1), response.into_bytes()],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens on the cut body");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the cut arrives")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("ended mid-line")),
        "a declared length that ends inside a frame's line refuses: {outcome:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_poisoned_stream_ends_and_a_fresh_connection_is_unaffected() {
    let dir = client_test_dir();
    let poisoned_feed = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
         id: 12\r\ndata: not-json\r\n\r\n";
    let clean_feed = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{}",
        sse_frame(21, &run_started_event(21))
    );
    let mock = spawn_mock(
        &dir,
        "poison.sock",
        vec![
            hello_response("loopctl", 1, 1),
            poisoned_feed.as_bytes().to_vec(),
            clean_feed.into_bytes(),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the stream opens");
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the poisoned frame arrives")
        .expect("the stream is open");
    assert!(
        matches!(first, Err(ClientError::MalformedResponse(_))),
        "the malformed payload answers its typed error: {first:?}"
    );
    let after = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the poisoned stream settles");
    assert!(
        after.is_none(),
        "a poisoned stream ends — the error never repeats: {after:?}"
    );
    let mut reopened = client
        .subscribe("watch-repos", None)
        .await
        .expect("the original client opens a fresh subscription after the poison");
    let redelivered = tokio::time::timeout(std::time::Duration::from_secs(5), reopened.next())
        .await
        .expect("the clean feed's event arrives")
        .expect("the reopened stream is open")
        .expect("the clean frame parses");
    assert_eq!(
        redelivered.cursor, 21,
        "the same client recovers a fresh stream after a poisoned one: {redelivered:?}"
    );
    let fresh = spawn_mock(&dir, "fresh.sock", vec![hello_response("loopctl", 1, 1)]);
    let client_two = DaemonClient::connect(ConnectOptions::new(fresh.socket()))
        .await
        .expect("a fresh connection is unaffected by the poisoned one");
    assert_eq!(
        client_two.handshake().0,
        "loopctl",
        "the client object reconnects cleanly after a poisoned stream"
    );
    fresh.stop();
    mock.stop();
}

#[tokio::test]
async fn a_connected_client_debugs_without_leaking_the_token() {
    let mock = spawn_mock_tcp(vec![hello_response("loopctl", 1, 1)]).await;
    let client = DaemonClient::connect(ConnectOptions::tcp(
        "127.0.0.1",
        mock.port,
        Some("sekrit".to_string()),
    ))
    .await
    .expect("the mock speaks the handshake over TCP");
    let rendered = format!("{client:?}");
    assert!(
        !rendered.contains("sekrit"),
        "the connected client's Debug carries its endpoint's redaction with it: {rendered}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_subtype_suffixed_media_type_refuses() {
    let dir = client_test_dir();
    let frame = format!("id: 5\r\ndata: {}\r\n\r\n", run_started_event(5));
    let mock = spawn_mock(
        &dir,
        "subtype-sse.sock",
        vec![
            hello_response("loopctl", 1, 1),
            sse_response("text/event-stream-json", &frame),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let rejection = client.subscribe("watch-repos", None).await;
    assert!(
        matches!(&rejection, Err(ClientError::MalformedResponse(text))
            if text.contains("not an event stream")),
        "text/event-stream-json is a different media type, not a prefix match — the \
         complete type is compared: {:?}",
        rejection.err().map(|error| error.to_string())
    );
    mock.stop();
}

#[tokio::test]
async fn an_undelimited_frame_flood_refuses_at_the_frame_bound() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    // 4 MiB of data lines with no blank-line delimiter: the frame's
    // retained size is the only bound that can see it — per-line
    // bounds never trip, because every line is small.
    let flood = format!("data: {}\r\n", "x".repeat(1024));
    for _ in 0..(4 * 1024 + 8) {
        sse.frames.send(flood.clone()).ok();
    }
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
        .await
        .expect("the flood refuses at the bound")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("exceeded the size bound")),
        "an undelimited drip of data lines refuses at the frame bound rather than \
         buffering without end: {outcome:?}"
    );
    sse.daemon.stop();
}

#[tokio::test]
async fn a_cursor_that_does_not_advance_refuses_the_feed() {
    let dir = client_test_dir();
    let sse = spawn_sse_mock(&dir);
    let client = DaemonClient::connect(ConnectOptions::new(sse.daemon.socket()))
        .await
        .expect("the mock speaks the handshake");
    let first = run_started_event(9).to_string();
    let lower = run_started_event(1).to_string();
    let repeat = run_started_event(9).to_string();
    sse.frames
        .send(format!("id: 9\r\ndata: {first}\r\n\r\n"))
        .ok();
    sse.frames
        .send(format!("id: 8\r\ndata: {lower}\r\n\r\n"))
        .ok();
    sse.frames
        .send(format!("id: 9\r\ndata: {repeat}\r\n\r\n"))
        .ok();
    let mut stream = client.subscribe("watch-repos", None).await.expect("opens");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the first event arrives")
        .expect("the stream is open")
        .expect("the first frame parses");
    assert_eq!(event.cursor, 9, "the advancing cursor delivers: {event:?}");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the regression arrives")
        .expect("the stream is open");
    assert!(
        matches!(&outcome, Err(ClientError::MalformedResponse(text))
            if text.contains("not strictly increasing")),
        "a cursor at or below the last delivered one refuses — a resume from the last \
         cursor would re-receive the event: {outcome:?}"
    );
    let after = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the refused stream settles");
    assert!(after.is_none(), "the refusal poisons the stream: {after:?}");
    sse.daemon.stop();
}

#[tokio::test]
async fn a_stalled_error_body_on_subscription_answers_within_the_deadline() {
    let dir = client_test_dir();
    let head = "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\
                Content-Length: 500\r\nConnection: close\r\n\r\n{\"started\"";
    let mock = spawn_mock(
        &dir,
        "stalled-error-subscription.sock",
        vec![hello_response("loopctl", 1, 1), head.as_bytes().to_vec()],
    );
    let options =
        ConnectOptions::new(mock.socket()).with_deadline(std::time::Duration::from_millis(150));
    let client = DaemonClient::connect(options)
        .await
        .expect("the mock speaks the handshake");
    let rejection = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.subscribe("watch-repos", None),
    )
    .await
    .expect("the stalled error body answers within the deadline, never an eternal await");
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Unauthorized { status: 401, .. })
        ),
        "the refused status surfaces even when its body stalls — the drain is bounded: {:?}",
        rejection.err().map(|error| error.to_string())
    );
    mock.stop();
}

#[tokio::test]
async fn a_stalled_error_body_on_a_verb_preserves_the_status() {
    let dir = client_test_dir();
    let head = "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n\
                Content-Length: 500\r\nConnection: close\r\n\r\n{\"started\"";
    let mock = spawn_mock(
        &dir,
        "stalled-error-verb.sock",
        vec![hello_response("loopctl", 1, 1), head.as_bytes().to_vec()],
    );
    let options =
        ConnectOptions::new(mock.socket()).with_deadline(std::time::Duration::from_millis(150));
    let client = DaemonClient::connect(options)
        .await
        .expect("the mock speaks the handshake");
    let rejection = tokio::time::timeout(std::time::Duration::from_secs(5), client.list_loops())
        .await
        .expect("the stalled error body answers within the deadline");
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Unauthorized { status: 403, .. })
        ),
        "a 403 head whose body stalls answers its typed refusal — the status survives the \
             stalled drain: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_204_verb_answer_completes_without_waiting_for_a_close() {
    let dir = client_test_dir();
    let no_content = "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n";
    let mock = spawn_mock(
        &dir,
        "no-content.sock",
        vec![
            hello_response("loopctl", 1, 1),
            no_content.as_bytes().to_vec(),
        ],
    );
    let client = DaemonClient::connect(ConnectOptions::new(mock.socket()))
        .await
        .expect("the mock speaks the handshake");
    let stopped = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.stop_run(uuid::Uuid::nil()),
    )
    .await
    .expect("a bodyless 204 completes without waiting for the peer's close");
    assert!(
        stopped.is_ok(),
        "the 204 answer is empty and successful: {stopped:?}"
    );
    mock.stop();
}
