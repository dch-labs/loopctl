//! Pins the advisory rate-limit low-water event end to end.
//!
//! The documented promise is "warns once per window (latched, re-armed
//! by recovery or silence)" — this binary proves it the only way a
//! `tracing` event can be proven: a process-global capture subscriber,
//! real clients driving recorded quota headers, and counted events. The
//! capture lives in its own test binary (the `embeddings_metrics.rs`
//! discipline) because a global subscriber cannot be shared across
//! concurrently running tests, and the unit pins in
//! `src/provider/forensics.rs` can only see the latch bit, not whether
//! the event is actually emitted.
//!
//! One provider client is enough: the emission lives in the shared
//! `post_json_checked` path every provider goes through, so counting it
//! through the OpenAI client pins the contract for all of them.
//!
//! Run: `cargo test --all-features --test low_water_events`

#![cfg(feature = "openai")]
#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::sync::Arc;
use std::sync::Mutex;

use loopctl::api::{ApiClient, StreamRequest};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The marker text of the event under test.
const LOW_WATER: &str = "provider rate-limit window below 10% remaining";

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the low-water event is asserted rather than trusted.
struct EventCapture {
    /// The joined `field=value` render of every captured event, in order.
    events: Mutex<Vec<String>>,
}

impl EventCapture {
    /// The captured events, oldest first.
    fn events(&self) -> Vec<String> {
        self.events.lock().expect("capture lock").clone()
    }

    /// The rows carrying the low-water marker, in order.
    fn low_water(&self) -> Vec<String> {
        self.events()
            .into_iter()
            .filter(|row| row.contains(LOW_WATER))
            .collect()
    }
}

impl tracing::Subscriber for EventCapture {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target().starts_with("loopctl::")
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::Id {
        tracing::Id::from_u64(1)
    }
    fn record(&self, _id: &tracing::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct FieldVisitor {
            fields: Vec<String>,
        }
        impl tracing::field::Visit for FieldVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.fields.push(format!("{}={:?}", field.name(), value));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.fields.push(format!("{}={}", field.name(), value));
            }
        }
        let mut visitor = FieldVisitor { fields: Vec::new() };
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture lock")
            .push(visitor.fields.join(" "));
    }
    fn enter(&self, _id: &tracing::Id) {}
    fn exit(&self, _id: &tracing::Id) {}
}

/// The OpenAI non-streaming 200 body the client's parser accepts.
const OPENAI_OK: &str = "{\"id\":\"cmpl-1\",\"object\":\"chat.completion\",\"created\":1,\
    \"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\
    \"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\
    \"completion_tokens\":1,\"total_tokens\":2}}";

/// Serve one HTTP 200 per accepted connection, `quota[i]` headers for the
/// i-th connection, closing each socket after answering.
async fn serve_quotas(quota: &[&[(&str, &str)]]) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let rendered: Vec<String> = quota
        .iter()
        .map(|headers| {
            let mut head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n", OPENAI_OK.len());
            for (name, value) in *headers {
                head.push_str(name);
                head.push_str(": ");
                head.push_str(value);
                head.push_str("\r\n");
            }
            head.push_str("Connection: close\r\n\r\n");
            format!("{head}{OPENAI_OK}")
        })
        .collect();
    let task = tokio::spawn(async move {
        for response in rendered {
            let (mut sock, _) = listener.accept().await.unwrap();
            read_full_request(&mut sock).await;
            sock.write_all(response.as_bytes()).await.unwrap();
            sock.flush().await.unwrap();
        }
    });
    (format!("http://{addr}"), task)
}

/// Read one HTTP/1.1 request (head plus body) off `sock`.
async fn read_full_request(sock: &mut tokio::net::TcpStream) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let head_end = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|at| at + 4);
        if let Some(head_end) = head_end {
            let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
            let clen = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= head_end + clen {
                return;
            }
        }
        let n = sock.read(&mut chunk).await.unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

#[tokio::test]
async fn low_water_event_fires_once_per_window_end_to_end() {
    let capture = Arc::new(EventCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    let window = |remaining: &'static str| {
        vec![
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-remaining", remaining),
        ]
    };
    // Crossing → latched → recovery rearms → fresh crossing → silence
    // rearms → below again after the re-arm.
    let phases: Vec<Vec<(&'static str, &'static str)>> = vec![
        window("5"),
        window("4"),
        window("80"),
        window("2"),
        Vec::new(),
        window("1"),
    ];
    let per_connection: Vec<&[(&str, &str)]> = phases.iter().map(Vec::as_slice).collect();
    let (base_url, server) = serve_quotas(&per_connection).await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key("test-key")
        .with_base_url(&base_url)
        .build()
        .unwrap();

    let mut seen_counts = Vec::new();
    for _ in 0..6 {
        client
            .create_message(&StreamRequest::new(vec![]))
            .await
            .expect("the mocked 200 must succeed");
        seen_counts.push(capture.low_water().len());
    }
    server.await.unwrap();

    assert_eq!(
        seen_counts,
        vec![1, 1, 1, 2, 2, 3],
        "one event per window crossing — latched while below, re-armed by \
         recovery or silence (the CHANGELOG promise)"
    );
    let events = capture.low_water();
    assert!(
        events[0].contains("remaining=5") && events[0].contains("limit=100"),
        "the first event carries the numbers that crossed: {:?}",
        events[0]
    );
    assert!(
        events[1].contains("remaining=2"),
        "the re-armed event carries the new crossing's numbers: {:?}",
        events[1]
    );
}
