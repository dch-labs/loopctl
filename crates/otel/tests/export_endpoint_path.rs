//! The path an OTLP/HTTP export is actually posted to, asserted against a
//! listener rather than a mock transport.
//!
//! Every other endpoint test in this crate points a live exporter at a dead
//! listener and asserts that it *builds*. That is exactly the gap which let a
//! pathless default endpoint ship: `opentelemetry_otlp` uses a programmatically
//! supplied endpoint verbatim, so a bare `http://host:4318` posts to `/`, a
//! collector answers `404`, and the failure arrives at the closing flush of a
//! run that had looked healthy. Building is not delivering, so this test asserts
//! delivery — one configured endpoint, two signal kinds, the request lines the
//! collector actually received, and the service name riding inside the payload.
//!
//! [`install`] claims the global subscriber and the `OTel` globals, both
//! once-per-process, so this binary holds exactly one test.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::unwrap_used
)]

use std::collections::BTreeMap;
use std::io::Read as _;
use std::io::Write as _;
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use loopctl::metrics::MetricsRegistry;
use loopctl_otel::OtelConfig;
use loopctl_otel::OtelMetricsMode;
use opentelemetry::trace::Tracer as _;

/// What a collector answers a well-formed export with: an empty `Export*ServiceResponse`
/// is a valid protobuf body.
const OK: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-type: application/x-protobuf\r\n\
                    content-length: 0\r\nconnection: close\r\n\r\n";

/// A stalled connection must end the test with an assertion, not a hang.
const READ_TIMEOUT: Duration = Duration::from_secs(15);

/// The `service.name` under test: what the config sets and what the payload
/// must carry, so the two cannot drift apart.
const SERVICE_NAME: &str = "path-contract";

/// Answer export requests on an ephemeral port, reporting each delivered
/// request.
///
/// Serves until the process exits: a batch processor may post more than once per
/// signal, and the test reads as many requests as it needs. The base URL it
/// listens on carries no signal path, which is the shape under test.
fn spawn_collector() -> (String, mpsc::Receiver<Delivery>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("an ephemeral port binds");
    // The exporter rejects a schemeless endpoint, so the base is a URL: a
    // listener address and the `http` scheme OTLP/HTTP speaks.
    let addr = listener.local_addr().expect("a bound address");
    let base = format!("http://{addr}");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        for incoming in listener.incoming() {
            let Ok(stream) = incoming else { return };
            if serve(stream, &sender).is_none() {
                return;
            }
        }
    });
    (base, receiver)
}

/// One export the collector received: where it was posted and what it carried.
#[derive(Clone, Debug)]
struct Delivery {
    /// The request target, which is where the signal was delivered.
    path: String,
    /// The request body, which is the encoded payload as it went on the wire.
    body: Vec<u8>,
}

/// Read one request, answer it, and hand back what it delivered.
///
/// The declared body is drained before the reply goes out: answering while a
/// request body is still in flight makes the client see a reset connection and
/// report a send failure where there was a delivery.
fn serve(mut stream: TcpStream, sender: &mpsc::Sender<Delivery>) -> Option<Delivery> {
    stream
        .set_read_timeout(Some(READ_TIMEOUT))
        .expect("a read timeout is settable");
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 8192];
    let mut needed = usize::MAX;
    while buf.len() < needed {
        let Ok(read) = stream.read(&mut chunk) else {
            return None;
        };
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..read]);
        if needed == usize::MAX
            && let Some(head) = head_end(&buf)
        {
            needed = head + 4 + declared_body(&buf[..head]);
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let request_line = text.lines().next()?.to_string();
    let path = request_line.split_whitespace().nth(1)?.to_string();
    // Whatever followed the head is the encoded payload. OTLP/HTTP ships it
    // uncompressed by default, so a resource attribute reads out of it raw.
    let body = buf
        .get(head_end(&buf)? + 4..)
        .map_or_else(Vec::new, <[u8]>::to_vec);
    if stream.write_all(OK).is_err() || stream.flush().is_err() {
        return None;
    }
    let delivery = Delivery { path, body };
    if sender.send(delivery.clone()).is_err() {
        return None;
    }
    Some(delivery)
}

/// Where the request head ends, as the byte offset of its terminator.
fn head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

/// The `content-length` a head declares; zero when it declares none.
fn declared_body(head: &[u8]) -> usize {
    let text = String::from_utf8_lossy(head).into_owned();
    for line in text.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            return value.trim().parse().unwrap_or(0);
        }
    }
    0
}

/// One configured endpoint delivers each signal at its own path.
///
/// Both exporters are built from the same `OtelConfig::endpoint`, and OTLP/HTTP
/// names `/v1/traces` and `/v1/metrics`. A collector answering both request lines
/// here is the proof that one endpoint can serve two signals; a `/` in either
/// position is the bug this test exists to keep dead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_configured_endpoint_delivers_each_signal_at_its_own_path() {
    let (base, paths) = spawn_collector();
    let registry = Arc::new(MetricsRegistry::default());
    let guard = loopctl_otel::install(&OtelConfig {
        endpoint: base,
        service_name: SERVICE_NAME.to_string(),
        metrics: Some(OtelMetricsMode::Pipeline),
        registry: Some(Arc::clone(&registry)),
        ..OtelConfig::default()
    })
    .expect("install claims the globals in this dedicated test binary");

    // A span through the provider `install` set globally — the route a host's
    // own spans take, not a stand-in transport. The scope name deliberately
    // differs from the service name: a payload that mentions only the scope
    // proves nothing about the resource.
    opentelemetry::global::tracer("path-probe-scope").in_span("path-probe", |_| {});

    // A series through the registry into the meter pipeline, which is what
    // `Pipeline` mode exists to deliver.
    let descriptor =
        loopctl::metrics::registry::lookup("loopctl.tools.calls").expect("a declared catalog row");
    registry.counter_add(descriptor, BTreeMap::default(), 1);
    loopctl_otel::export_metrics(None, &registry).expect("a metrics pipeline is installed");

    guard
        .force_flush()
        .expect("both providers deliver to the listener");

    let mut seen: Vec<Delivery> = Vec::new();
    for _ in 0..2 {
        match paths.recv_timeout(READ_TIMEOUT) {
            Ok(delivery) => seen.push(delivery),
            Err(err) => panic!(
                "only {:?} reached the collector within {READ_TIMEOUT:?} of read timeout \
                 ({err})",
                paths_of(&seen)
            ),
        }
    }
    let delivered = paths_of(&seen);
    assert!(
        delivered.contains(&"/v1/traces"),
        "traces must reach the collector at /v1/traces, saw {delivered:?} — a `/` is \
         a pathless endpoint handed to the exporter verbatim"
    );
    assert!(
        delivered.contains(&"/v1/metrics"),
        "metrics must reach the collector at /v1/metrics, saw {delivered:?} — one \
         shared endpoint serves both signals only if each gains its own path"
    );

    // Arriving is not the whole contract. A wrapper that leaves `set_resource`
    // to the SDK's no-op default ships a batch that reaches the collector
    // carrying no `service.name`, and the backend invents a service to file it
    // under — invisible from the client, and unrecoverable in the UI.
    let traces = seen
        .iter()
        .find(|delivery| delivery.path == "/v1/traces")
        .expect("a traces delivery was received above");
    assert!(
        contains(&traces.body, SERVICE_NAME.as_bytes()),
        "the configured service.name must travel with the spans; a payload without it \
         arrives and is filed under a placeholder service"
    );

    drop(guard);
}

/// The paths of received deliveries, for an assertion that reads as a list.
fn paths_of(deliveries: &[Delivery]) -> Vec<&str> {
    deliveries
        .iter()
        .map(|delivery| delivery.path.as_str())
        .collect()
}

/// Whether `haystack` holds `needle`, on bytes a protobuf payload keeps raw.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
