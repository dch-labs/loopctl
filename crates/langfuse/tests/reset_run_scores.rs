//! One loop, two runs, one shared score source — and which run each score
//! describes.
//!
//! Auto-scoring reads the observer when a run's trace closes, so the scores
//! on a trace claim to be that run's figures. This binary drives the shape
//! that claim is easiest to break: a second run on the same loop that the
//! manager reset refuses before its first turn, on an observer still holding
//! the completed run's rollup. The mock collector is the one the export
//! suite uses — the assertions run over the ingestion batches a Langfuse
//! deployment would receive.

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

use std::future::Future;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use loopctl::capabilities::Detectable as _;
use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::managers::LoopManagers;
use loopctl::metrics::MetricsObserver;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_langfuse::{LangfuseConfig, LangfuseLayer, install};
use serde_json::Value;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Captured ingestion events (one JSON value per envelope).
static EVENTS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

fn spawn_mock() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind on an ephemeral port");
    let port = listener.local_addr().expect("assigned port").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            if handle_connection(stream).is_err() {
                break;
            }
        }
    });
    port
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
        let mut events = EVENTS.lock().unwrap();
        events.extend(batch.iter().cloned());
    }
    let ack = "{\"successes\":[],\"errors\":[]}";
    let response = format!(
        "HTTP/1.1 207 Multi-Status\r\ncontent-type: application/json\r\n\
         connection: close\r\ncontent-length: {}\r\n\r\n{ack}",
        ack.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

/// A tool that echoes a fixed successful output.
struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Echoes a fixed output"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name().to_string(),
            self.description().to_string(),
            serde_json::json!({"type": "object"}),
        )
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Ok(ToolOutput::text("done")) })
    }
}

/// One scripted tool-call turn plus the terminal reply that closes the run.
fn tool_turn_then_terminal() -> Vec<MockResponse> {
    vec![
        MockResponse {
            text: "calling echo".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "finished".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]
}

/// Poison the convergence detector the loop holds: panic while the guard is
/// held, which is what poisons the lock.
fn poison_detector(engine: &BareLoop<MockApiClient>) {
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = engine
            .managers()
            .detection()
            .convergence_detector()
            .lock()
            .expect("lock before poisoning");
        panic!("poison the convergence detector");
    }));
    assert!(
        poisoned.is_err(),
        "the panic inside the guard is what poisons the lock"
    );
}

/// Read one score's body off the captured batches.
fn score(events: &[Value], trace_id: &str, name: &str) -> Value {
    events
        .iter()
        .find(|event| {
            event["type"] == "score-create"
                && event["body"]["traceId"] == trace_id
                && event["body"]["name"] == name
        })
        .unwrap_or_else(|| panic!("a {name} score on trace {trace_id}"))
        .clone()
}

#[tokio::test(flavor = "current_thread")]
async fn a_run_failing_at_manager_reset_scores_its_own_run() {
    let port = spawn_mock();
    let observer = Arc::new(MetricsObserver::new());

    let (layer, guard) = install(LangfuseConfig {
        host: format!("http://127.0.0.1:{port}"),
        public_key: "pk-test".to_string(),
        secret_key: "sk-test".to_string(),
        batch_size: 100,
    })
    .expect("install");
    let source_observer = Arc::<MetricsObserver>::clone(&observer);
    let layer = LangfuseLayer::with_metrics_source(
        layer,
        Arc::new(move || Some(source_observer.snapshot())),
    );
    let subscriber = Registry::default().with(layer);

    let client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut engine = BareLoop::new_with_managers(
        client,
        registry,
        SessionConfig::default(),
        LoopManagers::new(),
    )
    .with_observer(Arc::<MetricsObserver>::clone(&observer));

    {
        let _guard = tracing::subscriber::set_default(subscriber);

        let first = engine
            .run("the run that completes", &RunConfig::default())
            .await;
        assert!(first.is_ok(), "the first run completes: {first:?}");

        poison_detector(&engine);
        let second = engine
            .run(
                "the run that fails at the reset",
                &RunConfig::default().with_reset_managers(true),
            )
            .await;
        assert!(
            matches!(second, Err(loopctl::error::LoopError::LockPoisoned { .. })),
            "a poisoned manager reset fails the run with its own error: {second:?}"
        );
    }

    let drained = guard.flush(std::time::Duration::from_secs(5)).await;
    assert!(
        drained,
        "every queued envelope ships before the drain returns"
    );

    let events = EVENTS.lock().unwrap().clone();
    let traces: Vec<&Value> = events
        .iter()
        .filter(|event| event["type"] == "trace-create")
        .collect();
    assert_eq!(
        traces.len(),
        2,
        "one trace per run — the closed failure included"
    );
    let failed_traces: Vec<&&Value> = traces
        .iter()
        .filter(|trace| trace["body"]["level"] == "ERROR")
        .collect();
    assert_eq!(
        failed_traces.len(),
        1,
        "exactly the reset-refused run closes its trace at level ERROR"
    );
    let failed_id = failed_traces[0]["body"]["id"]
        .as_str()
        .expect("failed trace id");
    let completed_id = traces
        .iter()
        .find(|trace| trace["body"]["level"].is_null())
        .and_then(|trace| trace["body"]["id"].as_str())
        .expect("the completed run's trace carries no level");

    assert_eq!(
        score(&events, failed_id, "outcome")["body"]["stringValue"],
        "error",
        "the refused run's own outcome is what its trace scores"
    );
    assert_eq!(
        score(&events, failed_id, "total_turns")["body"]["value"],
        0.0,
        "a run that never reached a turn scores no turns — not its predecessor's"
    );
    assert_eq!(
        score(&events, failed_id, "input_tokens")["body"]["value"],
        0.0,
        "a run that never reached the provider scores no tokens"
    );

    assert_eq!(
        score(&events, completed_id, "outcome")["body"]["stringValue"],
        "success",
        "the completed run still scores its own success"
    );
    assert_eq!(
        score(&events, completed_id, "total_turns")["body"]["value"],
        2.0,
        "the completed run keeps both of its turns on its own trace"
    );

    drop(guard);
}
