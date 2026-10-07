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
        Box::pin(async move {
            Ok(vec![loopctl_client::LoopSummary {
                id: "in-process".to_string(),
                status: "idle".to_string(),
                model: None,
            }])
        })
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
        Box::pin(async move {
            Ok(loopctl_client::LoopSummary {
                id,
                status: "idle".to_string(),
                model: None,
            })
        })
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
        Box::pin(async move {
            Ok(loopctl_client::RunHandle {
                run_id: uuid::Uuid::new_v4(),
            })
        })
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
            Ok(loopctl_client::RunState {
                run_id,
                loop_id: "in-process".to_string(),
                status: "completed".to_string(),
                stop_reason: None,
            })
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

/// Read one full request (head + Content-Length body) from `stream`.
async fn read_one_request(stream: &mut tokio::net::UnixStream) -> RecordedRequest {
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
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
            && let Ok(parsed) = value.trim().parse::<usize>()
        {
            length = parsed;
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
        Err(ClientError::ConnectTimeout { window_ms }) => assert_eq!(
            window_ms, 20,
            "the window spans the delays between attempts — three tries, two \
             10 ms delays, never an over-report"
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
