//! `install_simple` smoke: the one-call install must actually export.
//!
//! The global-subscriber convenience path composes the layer and the
//! target filter itself, so its correctness rests on one knob — the
//! filter has to admit the `DEBUG` spans the crate turns into traces
//! and observations. This binary drives a real engine run through the
//! globally installed subscriber and holds the mock collector to its
//! word: the cloud would receive a trace. One test per binary because
//! `install_simple` claims the **global** subscriber; a second call
//! (or any other test here wanting a scoped subscriber) cannot
//! coexist with that claim.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines
)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::testing::MockApiClient;
use loopctl::tool::ToolRegistry;
use loopctl_langfuse::{LangfuseConfig, install_simple};
use serde_json::Value;

/// Captured ingestion events, exactly as the mock collector saw them.
static EVENTS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

fn spawn_mock(port: u16) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind mock collector");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            if handle_connection(stream).is_err() {
                break;
            }
        }
    });
}

fn handle_connection(mut stream: TcpStream) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut byte = [0_u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            return Ok(());
        }
        buf.push(byte[0]);
    }
    let header_text = String::from_utf8_lossy(&buf).to_string();
    let length: usize = header_text
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|rest| rest.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let mut body = vec![0_u8; length];
    stream.read_exact(&mut body)?;
    if let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(&body)
        && let Some(Value::Array(batch)) = payload.get("batch")
    {
        EVENTS.lock().unwrap().extend(batch.iter().cloned());
    }
    let response = "HTTP/1.1 207 Multi-Status\r\ncontent-type: application/json\r\n\
                    connection: close\r\ncontent-length: 24\r\n\r\n\
                    {\"successes\":[],\"errors\":[]}";
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

#[tokio::test(flavor = "current_thread")]
async fn install_simple_exports_a_trace_for_a_real_run() {
    let port = 47_899;
    spawn_mock(port);

    let guard = install_simple(LangfuseConfig {
        host: format!("http://127.0.0.1:{port}"),
        public_key: "pk-test".to_string(),
        secret_key: "sk-test".to_string(),
        batch_size: 100,
    })
    .expect("install_simple claims the global subscriber");

    let client = Arc::new(MockApiClient::new("test-model").with_responses(vec![
        loopctl::testing::MockResponse {
            text: "all done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]));
    let mut engine = BareLoop::new(client, ToolRegistry::new(), SessionConfig::default());
    let result = engine
        .run("install simple smoke", &RunConfig::default())
        .await;
    assert!(result.is_ok(), "the run must succeed: {result:?}");

    // The cooperative drain is both the teardown contract and the
    // observation window: it returns once the queue is empty with no
    // batch in flight — which only happens if the flusher actually
    // shipped whatever the run emitted.
    let drained = guard.flush(Duration::from_secs(5)).await;
    assert!(
        drained,
        "the queue must drain cooperatively on a current-thread runtime"
    );

    let events = EVENTS.lock().unwrap().clone();
    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(
        events.iter().any(|e| e["type"] == "trace-create"),
        "install_simple's filter must admit the DEBUG run span so a \
         trace-create reaches the collector; got {types:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e["type"] != "trace-create" && !e["type"].is_null()),
        "the turn span must ride the same batch as an observation; got {types:?}"
    );
}
