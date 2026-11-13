//! The TLS transport contract: the remote encrypted endpoint pinned
//! against a local mock daemon serving scripted responses over TLS.
//!
//! The mock mints a self-signed certificate at test time, serves the
//! same scripted-raw-bytes dialect the socket suite uses, and the
//! client verifies against the generated certificate as its CA —
//! the exact shape a self-hosted daemon's PKI takes.
//!
//! *Runs only under the `tls` feature.*

#![cfg(feature = "tls")]
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

use futures::StreamExt;
use loopctl::testing::EnvGuard;
use loopctl_client::ClientError;
use loopctl_client::ConnectOptions;
use loopctl_client::DaemonClient;
use loopctl_client::LoopctlClient;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

/// One request the TLS mock observed.
#[derive(Debug, Clone)]
struct RecordedRequest {
    /// The request path with query.
    path: String,
    /// The `Host` header value, empty when the head carried none.
    host: String,
    /// The `Authorization` header value, when the request carried one.
    authorization: Option<String>,
}

/// A mock daemon over TLS: the CA file it answers for, its port,
/// and every request it observed.
struct TlsMock {
    /// The tempdir holding `ca.pem`; held, never read — the client's
    /// CA file must outlive every connect.
    _dir: tempfile::TempDir,
    /// The CA file path the client verifies against.
    ca_pem: PathBuf,
    /// The mock's loopback port.
    port: u16,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl TlsMock {
    /// Every request observed so far.
    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Stop the accept loop.
    fn stop(self) {
        self.handle.abort();
    }
}

/// Serve `responses` sequentially over TLS, one per connection.
///
/// The certificate is self-signed for `localhost` and also written
/// to the tempdir as the client's CA file; clients present no
/// certificate.
async fn spawn_tls_mock(responses: Vec<Vec<u8>>) -> TlsMock {
    spawn_tls_mock_with(responses, None).await
}

/// Serve `responses` over TLS, optionally requiring client
/// certificates verified against `client_roots_pem`.
///
/// The mutual-TLS twin of [`spawn_tls_mock`]: when `client_roots_pem`
/// is given the server refuses any client whose certificate does not
/// verify against it — the exact shape a daemon that authenticates
/// clients by certificate takes.
async fn spawn_tls_mock_with(responses: Vec<Vec<u8>>, client_roots_pem: Option<&[u8]>) -> TlsMock {
    let dir = tempfile::tempdir().expect("the tempdir creates");
    let certified =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("the cert mints");
    let cert_pem = certified.cert.pem();
    let key_pem = certified.signing_key.serialize_pem();
    let ca_pem = dir.path().join("ca.pem");
    std::fs::write(&ca_pem, &cert_pem).expect("the CA file writes");

    let mut cert_bytes = cert_pem.as_bytes();
    let certificates = rustls_pemfile::certs(&mut cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .expect("the certificate PEM parses");
    let mut key_bytes = key_pem.as_bytes();
    let key = rustls_pemfile::private_key(&mut key_bytes)
        .expect("the key PEM parses")
        .expect("the key is present");
    let builder = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("the versions negotiate");
    let config = match client_roots_pem {
        Some(roots_pem) => {
            let mut roots = rustls::RootCertStore::empty();
            let mut root_bytes = roots_pem;
            let root_certs = rustls_pemfile::certs(&mut root_bytes)
                .collect::<Result<Vec<_>, _>>()
                .expect("the client roots parse");
            roots.add_parsable_certificates(root_certs);
            let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
                Arc::new(roots),
                Arc::new(rustls::crypto::ring::default_provider()),
            )
            .build()
            .expect("the client verifier builds");
            builder
                .with_client_cert_verifier(verifier)
                .with_single_cert(certificates, key)
                .expect("the server config builds")
        }
        None => builder
            .with_no_client_auth()
            .with_single_cert(certificates, key)
            .expect("the server config builds"),
    };
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the mock binds");
    let port = listener.local_addr().expect("the address resolves").port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for response in responses {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut stream) = acceptor.accept(stream).await else {
                return;
            };
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            stream.write_all(&response).await.expect("the mock writes");
        }
    });
    TlsMock {
        _dir: dir,
        ca_pem,
        port,
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
    let path = request_line
        .split(' ')
        .nth(1)
        .unwrap_or_default()
        .to_string();
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
    drop(body);
    RecordedRequest {
        path,
        host,
        authorization,
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|quad| quad == b"\r\n\r\n")
}

fn env_for_test_runs() -> loopctl::testing::EnvGuard {
    EnvGuard::acquire(&["XDG_RUNTIME_DIR"])
}

/// Write `bytes` as `ca.pem` in a fresh tempdir.
///
/// The returned dir handle must outlive the connect it feeds — the
/// CA file vanishes with it on drop.
fn ca_file(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("the tempdir creates");
    let path = dir.path().join("ca.pem");
    std::fs::write(&path, bytes).expect("the CA file writes");
    (dir, path)
}

/// Write one client-identity PEM file — chain then key — in a fresh
/// tempdir.
///
/// The returned dir handle must outlive the connect it feeds, the
/// same lifetime rule [`ca_file`]'s carries.
fn identity_file(cert_pem: &str, key_pem: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("the tempdir creates");
    let path = dir.path().join("client.pem");
    std::fs::write(&path, format!("{cert_pem}\n{key_pem}")).expect("the identity file writes");
    (dir, path)
}

/// Attempt one TLS connect against `port` with `ca_pem` as the sole
/// trust root, returning the outcome unchanged.
///
/// # Errors
///
/// Whatever `DaemonClient::connect` answers — each refusal test
/// below asserts on exactly one arm of it.
async fn connect_with_ca(port: u16, ca_pem: PathBuf) -> Result<DaemonClient, ClientError> {
    DaemonClient::connect(ConnectOptions::tls(
        "localhost",
        port,
        Some("sekrit".to_string()),
        Some(ca_pem),
    ))
    .await
}

#[tokio::test]
async fn the_tls_endpoint_performs_the_handshake_and_verbs() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![
        hello_response("loopctl", 1, 1),
        json_response("200 OK", &serde_json::json!([])),
    ])
    .await;
    let options = ConnectOptions::tls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
    );
    let client = DaemonClient::connect(options)
        .await
        .expect("the TLS handshake completes against the mock's certificate");
    client
        .list_loops()
        .await
        .expect("the verb rides the encrypted connection");
    let requests = mock.requests();
    assert_eq!(
        requests[0].path, "/v1/hello",
        "the handshake rides the encrypted connection"
    );
    assert_eq!(
        requests[0].host,
        format!("localhost:{}", mock.port),
        "the encrypted request head addresses the endpoint's authority, never a placeholder"
    );
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer sekrit"),
        "the bearer token rides every TLS request, the handshake included"
    );
    assert_eq!(
        requests[1].authorization.as_deref(),
        Some("Bearer sekrit"),
        "the verb carries the same credentials"
    );
    mock.stop();
}

#[tokio::test]
async fn an_unauthorized_answer_is_typed_over_tls() {
    let _env = env_for_test_runs();
    let refusal = json_response(
        "401 Unauthorized",
        &serde_json::json!({
            "error": {"code": "bad_token", "message": "the token was rejected"}
        }),
    );
    let mock = spawn_tls_mock(vec![refusal]).await;
    let options = ConnectOptions::tls(
        "localhost",
        mock.port,
        Some("wrong".to_string()),
        Some(mock.ca_pem.clone()),
    );
    let rejection = DaemonClient::connect(options).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Unauthorized { status: 401, message })
                if message == "the token was rejected"
        ),
        "a refused credential answers the typed error even before the handshake: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_refused_tls_port_is_the_typed_absence() {
    let _env = env_for_test_runs();
    let rejection = DaemonClient::connect(ConnectOptions::tls("127.0.0.1", 1, None, None)).await;
    assert!(
        matches!(rejection, Err(ClientError::SocketAbsent)),
        "a refused TLS connection is the same typed absence the degrade path keys on: {rejection:?}"
    );
}

#[tokio::test]
async fn a_tls_host_with_control_characters_is_refused() {
    let _env = env_for_test_runs();
    let rejection =
        DaemonClient::connect(ConnectOptions::tls("daemon\r\n.example:443", 1, None, None)).await;
    assert!(
        matches!(
            &rejection,
            Err(ClientError::Transport(detail)) if detail.contains("CR, LF, or NUL")
        ),
        "a TLS host carrying CR/LF refuses before the connection opens, never a spliced \
         Host header: {rejection:?}"
    );
}

#[tokio::test]
async fn a_missing_ca_file_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let holder = tempfile::tempdir().expect("the tempdir creates");
    let missing = holder.path().join("absent-ca.pem");
    let rejection = connect_with_ca(mock.port, missing).await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the CA file could not be read")),
        "an unreadable CA file is a typed transport refusal, never a WebPKI fallback: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_empty_ca_file_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let (_holder, empty) = ca_file(b"");
    let rejection = connect_with_ca(mock.port, empty).await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the CA file carried no certificates")),
        "a CA file with zero certificates refuses rather than trusting an empty root set: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_non_pem_ca_file_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let (_holder, not_pem) =
        ca_file(b"-----BEGIN CERTIFICATE-----\n!!!not base64!!!\n-----END CERTIFICATE-----\n");
    let rejection = connect_with_ca(mock.port, not_pem).await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the CA file is not PEM")),
        "a CA file whose PEM body does not decode refuses as a typed transport failure: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_ca_file_with_no_parsable_certificate_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    // The body decodes cleanly — 16 bytes of 0xAA — so the PEM parses
    // while the certificate does not: the parsable-none arm.
    let (_holder, garbage) = ca_file(
        b"-----BEGIN CERTIFICATE-----\nqqqqqqqqqqqqqqqqqqqqqg==\n-----END CERTIFICATE-----\n",
    );
    let rejection = connect_with_ca(mock.port, garbage).await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the CA file held no parsable certificate")),
        "a CA file whose only certificate does not parse refuses rather than trusting nothing: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_wrong_ca_refuses_rather_than_connecting() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let stranger = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("the stranger CA mints");
    let (_holder, wrong_ca) = ca_file(stranger.cert.pem().as_bytes());
    let rejection = connect_with_ca(mock.port, wrong_ca).await;
    assert!(
        matches!(rejection, Err(ClientError::Transport(_))),
        "a certificate that does not verify refuses — never a degraded unverified connection: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "no request ever rides a connection the named CA refused to verify"
    );
    mock.stop();
}

#[tokio::test]
async fn the_mtls_endpoint_presents_its_identity_and_verbs() {
    let _env = env_for_test_runs();
    let minted = rcgen::generate_simple_self_signed(vec!["loopctl-client".to_string()])
        .expect("the client certificate mints");
    let mock = spawn_tls_mock_with(
        vec![
            hello_response("loopctl", 1, 1),
            json_response("200 OK", &serde_json::json!([])),
        ],
        Some(minted.cert.pem().as_bytes()),
    )
    .await;
    let (_holder, identity) =
        identity_file(&minted.cert.pem(), &minted.signing_key.serialize_pem());
    let options = ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        identity,
    );
    let client = DaemonClient::connect(options)
        .await
        .expect("the mutual handshake completes against the verifier");
    client
        .list_loops()
        .await
        .expect("the verb rides the mutual connection");
    let requests = mock.requests();
    assert_eq!(
        requests[0].path, "/v1/hello",
        "the mutual handshake rides the versioned path"
    );
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer sekrit"),
        "the bearer token rides the mutual connection like any other"
    );
    mock.stop();
}

#[tokio::test]
async fn a_missing_identity_file_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let holder = tempfile::tempdir().expect("the tempdir creates");
    let missing = holder.path().join("absent-client.pem");
    let rejection = DaemonClient::connect(ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        missing,
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the client identity could not be read")),
        "an unreadable identity file is a typed transport refusal: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "the identity loads before the connection opens — nothing reached the daemon"
    );
    mock.stop();
}

#[tokio::test]
async fn a_non_pem_identity_file_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let (_holder, identity) = identity_file(
        "-----BEGIN CERTIFICATE-----\n!!!not base64!!!\n-----END CERTIFICATE-----",
        "",
    );
    let rejection = DaemonClient::connect(ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        identity,
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the client identity is not PEM")),
        "an identity file whose PEM body does not decode refuses as a typed transport failure: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_identity_file_with_no_certificates_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let key_only = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("a key pair mints");
    let (_holder, identity) = identity_file("", &key_only.signing_key.serialize_pem());
    let rejection = DaemonClient::connect(ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        identity,
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the client identity carried no certificates")),
        "an identity file with only a key refuses — a chain must ride it: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_identity_file_with_no_private_key_is_a_typed_transport_refusal() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let certificate_only =
        std::fs::read_to_string(&mock.ca_pem).expect("the mock's certificate reads");
    let (_holder, identity) = identity_file(&certificate_only, "");
    let rejection = DaemonClient::connect(ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        identity,
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(refusal))
            if refusal.contains("the client identity carried no private key")),
        "an identity file with only a certificate refuses — a key must sign the handshake: {rejection:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn a_cert_requiring_daemon_refuses_an_identity_less_endpoint() {
    let _env = env_for_test_runs();
    let minted = rcgen::generate_simple_self_signed(vec!["loopctl-client".to_string()])
        .expect("the client certificate mints");
    let mock = spawn_tls_mock_with(
        vec![hello_response("loopctl", 1, 1)],
        Some(minted.cert.pem().as_bytes()),
    )
    .await;
    let rejection = DaemonClient::connect(ConnectOptions::tls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
    ))
    .await;
    assert!(
        matches!(rejection, Err(ClientError::Transport(_))),
        "a daemon that requires a client certificate refuses one that presents none: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "no request rides a handshake the daemon's verifier refused"
    );
    mock.stop();
}

#[tokio::test]
async fn a_wrong_client_identity_refuses_rather_than_connecting() {
    let _env = env_for_test_runs();
    let legitimate = rcgen::generate_simple_self_signed(vec!["loopctl-client".to_string()])
        .expect("the legitimate client certificate mints");
    let mock = spawn_tls_mock_with(
        vec![hello_response("loopctl", 1, 1)],
        Some(legitimate.cert.pem().as_bytes()),
    )
    .await;
    let stranger = rcgen::generate_simple_self_signed(vec!["loopctl-client".to_string()])
        .expect("the stranger client certificate mints");
    let (_holder, identity) =
        identity_file(&stranger.cert.pem(), &stranger.signing_key.serialize_pem());
    let rejection = DaemonClient::connect(ConnectOptions::mtls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
        identity,
    ))
    .await;
    assert!(
        matches!(rejection, Err(ClientError::Transport(_))),
        "a client certificate the daemon does not trust refuses the handshake: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "no request ever rides a connection the daemon's verifier refused"
    );
    mock.stop();
}

/// A minimal, valid trajectory event as the wire carries it.
fn run_started_event(seq: u64) -> serde_json::Value {
    serde_json::json!({
        "seq": seq,
        "ts": "2026-11-07T00:00:00Z",
        "run_id": 1,
        "session_id": "0b6bb8ab-6c5a-4f78-9c86-bd294c1a9e12",
        "turn": null,
        "kind": "run.started",
        "data": {"input": "hello"},
    })
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

/// Serve per-connection answer pieces over TLS, paced write by write.
///
/// The split-write twin of [`spawn_tls_mock`]: every piece lands as
/// its own write with a pause between, so the client is polled to
/// `Pending` mid-syntax — the conditions a single `write_all` can
/// never produce.
async fn spawn_tls_mock_paced(answers: Vec<Vec<Vec<u8>>>) -> TlsMock {
    let dir = tempfile::tempdir().expect("the tempdir creates");
    let certified =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("the cert mints");
    let cert_pem = certified.cert.pem();
    let key_pem = certified.signing_key.serialize_pem();
    let ca_pem = dir.path().join("ca.pem");
    std::fs::write(&ca_pem, &cert_pem).expect("the CA file writes");
    let mut cert_bytes = cert_pem.as_bytes();
    let certificates = rustls_pemfile::certs(&mut cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .expect("the certificate PEM parses");
    let mut key_bytes = key_pem.as_bytes();
    let key = rustls_pemfile::private_key(&mut key_bytes)
        .expect("the key PEM parses")
        .expect("the key is present");
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("the versions negotiate")
    .with_no_client_auth()
    .with_single_cert(certificates, key)
    .expect("the server config builds");
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the mock binds");
    let port = listener.local_addr().expect("the address resolves").port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for answer in answers {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut stream) = acceptor.accept(stream).await else {
                return;
            };
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            for piece in answer {
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                stream.write_all(&piece).await.expect("a piece writes");
            }
        }
    });
    TlsMock {
        _dir: dir,
        ca_pem,
        port,
        requests,
        handle,
    }
}

/// Serve scripted responses over TLS bound to the `IPv6` loopback.
///
/// The certificate's SAN is the `::1` address itself — the IP-shape
/// coverage the `localhost` fixture cannot produce, so the bracketed
/// authority verifies against the name actually dialed.
async fn spawn_tls_mock_ipv6(responses: Vec<Vec<u8>>) -> TlsMock {
    let dir = tempfile::tempdir().expect("the tempdir creates");
    let mut params = rcgen::CertificateParams::new(vec![]).expect("the params build");
    params.subject_alt_names = vec![rcgen::SanType::IpAddress(
        "::1".parse::<std::net::IpAddr>().expect("::1 parses"),
    )];
    let key = rcgen::KeyPair::generate().expect("the key mints");
    let certificate = params.self_signed(&key).expect("the certificate mints");
    let cert_pem = certificate.pem();
    let key_pem = key.serialize_pem();
    let ca_pem = dir.path().join("ca.pem");
    std::fs::write(&ca_pem, &cert_pem).expect("the CA file writes");
    let mut cert_bytes = cert_pem.as_bytes();
    let certificates = rustls_pemfile::certs(&mut cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .expect("the certificate PEM parses");
    let mut key_bytes = key_pem.as_bytes();
    let key = rustls_pemfile::private_key(&mut key_bytes)
        .expect("the key PEM parses")
        .expect("the key is present");
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("the versions negotiate")
    .with_no_client_auth()
    .with_single_cert(certificates, key)
    .expect("the server config builds");
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind(("::1", 0))
        .await
        .expect("the ipv6 mock binds");
    let port = listener.local_addr().expect("the address resolves").port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for response in responses {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut stream) = acceptor.accept(stream).await else {
                return;
            };
            let request = read_one_request(&mut stream).await;
            recorded.lock().unwrap().push(request);
            stream.write_all(&response).await.expect("the mock writes");
        }
    });
    TlsMock {
        _dir: dir,
        ca_pem,
        port,
        requests,
        handle,
    }
}

#[tokio::test]
async fn a_name_mismatched_certificate_refuses_the_handshake() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let rejection = DaemonClient::connect(ConnectOptions::tls(
        "127.0.0.1",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(_))),
        "a certificate that does not cover the dialed name refuses the handshake: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "no request rode a connection whose server name did not verify"
    );
    mock.stop();
}

#[tokio::test]
async fn a_self_signed_daemon_refuses_without_a_ca_file() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock(vec![hello_response("loopctl", 1, 1)]).await;
    let rejection = DaemonClient::connect(ConnectOptions::tls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        None,
    ))
    .await;
    assert!(
        matches!(&rejection, Err(ClientError::Transport(_))),
        "the public WebPKI roots never trust a self-signed daemon — refusal, never a \
         fallback: {rejection:?}"
    );
    assert!(
        mock.requests().is_empty(),
        "no request rode the untrusted connection"
    );
    mock.stop();
}

#[tokio::test]
async fn a_chunked_event_feed_delivers_over_tls() {
    let _env = env_for_test_runs();
    let first_frame = sse_frame(7, &run_started_event(1));
    let second_frame = sse_frame(8, &run_started_event(2));
    let mut pieces = chunked_frame_pieces(&first_frame);
    pieces.extend(chunked_frame_pieces(&second_frame));
    pieces.push("0\r\n\r\n".to_string());
    let mut answer = vec![
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            .to_string(),
    ];
    answer.extend(pieces);
    let mock = spawn_tls_mock_paced(vec![
        vec![hello_response("loopctl", 1, 1)],
        answer.into_iter().map(String::into_bytes).collect(),
    ])
    .await;
    let client = DaemonClient::connect(ConnectOptions::tls(
        "localhost",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
    ))
    .await
    .expect("the TLS handshake completes against the mock's certificate");
    let mut stream = client
        .subscribe("watch-repos", None)
        .await
        .expect("the chunked stream opens over TLS");
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the first paced chunk arrives")
        .expect("the stream is open")
        .expect("the first frame parses");
    assert_eq!(
        event.cursor, 7,
        "the chunk decoder delivers over TLS exactly as over the socket: {event:?}"
    );
    let second = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the second paced chunk arrives")
        .expect("the stream is open")
        .expect("the second frame parses");
    assert_eq!(
        second.cursor, 8,
        "the split-write chunk decoding holds under TLS: {second:?}"
    );
    mock.stop();
}

#[tokio::test]
async fn an_ipv6_literal_tls_endpoint_addresses_its_bracketed_authority() {
    let _env = env_for_test_runs();
    let mock = spawn_tls_mock_ipv6(vec![hello_response("loopctl", 1, 1)]).await;
    DaemonClient::connect(ConnectOptions::tls(
        "::1",
        mock.port,
        Some("sekrit".to_string()),
        Some(mock.ca_pem.clone()),
    ))
    .await
    .expect("the IPv6 loopback endpoint verifies against the address SAN");
    let requests = mock.requests();
    assert_eq!(
        requests[0].host,
        format!("[::1]:{}", mock.port),
        "an IPv6 literal rides its authority bracketed over TLS too, against a certificate \
         covering the address itself"
    );
    mock.stop();
}
