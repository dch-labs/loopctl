//! End-to-end: a real engine run, the real layer, the real flusher — and
//! a hand-rolled localhost HTTP mock standing in for Langfuse. The mock
//! captures the exact ingestion batches the cloud would receive, and the
//! assertions are the Langfuse acceptance contract: envelope shape, event
//! types, trace/observation parenting, session grouping, scores.

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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::MetricsObserver;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_langfuse::{LangfuseConfig, LangfuseLayer, install};
use serde_json::Value;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Captured ingestion events (one JSON value per envelope).
static EVENTS: Mutex<Vec<Value>> = Mutex::new(Vec::new());
static AUTH_OK: AtomicBool = AtomicBool::new(true);
static STOP: AtomicBool = AtomicBool::new(false);

fn spawn_mock() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind on an ephemeral port");
    let port = listener.local_addr().expect("assigned port").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            let Ok(stream) = stream else { break };
            if handle_connection(stream).is_err() {
                AUTH_OK.store(false, Ordering::SeqCst);
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
    // Basic auth must be present and well-formed; the mock records it.
    if !header_text
        .to_ascii_lowercase()
        .contains("authorization: basic ")
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing basic auth",
        ));
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

#[tokio::test(flavor = "current_thread")]
async fn the_cloud_would_receive_traces_observations_and_scores() {
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
    let mut engine = BareLoop::new(client, registry, SessionConfig::default())
        .with_observer(Arc::<MetricsObserver>::clone(&observer));

    {
        let _guard = tracing::subscriber::set_default(subscriber);
        let result = engine.run("langfuse e2e", &RunConfig::default()).await;
        assert!(result.is_ok(), "run must succeed: {:?}", result.err());

        // A failed run rides the same pipeline: its trace envelope must
        // close at level ERROR — the `outcome`/`error.type` legs of the
        // mapping, unpinned by the clean run above.
        let failing_client = Arc::new(
            MockApiClient::new("test-model")
                .with_errors(vec![Some("provider exploded".to_string())]),
        );
        let failing_registry = ToolRegistry::new();
        failing_registry.register(EchoTool);
        let mut failing_engine =
            BareLoop::new(failing_client, failing_registry, SessionConfig::default())
                .with_observer(Arc::<MetricsObserver>::clone(&observer));
        let failed = failing_engine
            .run("langfuse failing run", &RunConfig::default())
            .await;
        assert!(
            failed.is_err(),
            "the scripted provider failure fails the run"
        );
    }

    // The cooperative drain is the shipped teardown contract: on a
    // current-thread runtime it awaits the flusher through to the last
    // batch — no racing a fixed sleep against the 250 ms tick.
    let drained = guard.flush(std::time::Duration::from_secs(5)).await;
    assert!(
        drained,
        "every queued envelope ships before the drain returns"
    );

    let events = EVENTS.lock().unwrap().clone();
    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    let traces: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "trace-create")
        .collect();
    let observations: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "span-create")
        .collect();
    let scores: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "score-create")
        .collect();

    assert_eq!(
        traces.len(),
        2,
        "one trace per run — the clean run and the failed one; got types {types:?}"
    );
    let clean_trace = traces
        .iter()
        .find(|t| t["body"]["level"].is_null())
        .expect("the clean run's trace carries no level");
    let errored: Vec<&&Value> = traces
        .iter()
        .filter(|t| t["body"]["level"] == "ERROR")
        .collect();
    assert_eq!(
        errored.len(),
        1,
        "exactly the failed run closes its trace at level ERROR"
    );
    assert_eq!(
        errored[0]["body"]["name"], "agent.run",
        "the ERROR trace is the failed run's"
    );
    assert!(
        observations
            .iter()
            .any(|o| o["body"]["name"] == "agent.turn"),
        "turn observation present"
    );
    let tool_obs = observations
        .iter()
        .find(|o| o["body"]["name"] == "agent.tool")
        .expect("tool observation");
    let turn_obs = observations
        .iter()
        .find(|o| o["body"]["name"] == "agent.turn")
        .expect("turn observation");

    // Envelope contract: id, type, rfc3339 timestamp, body.
    for event in &events {
        assert!(event["id"].is_string(), "envelope id");
        assert!(
            event["timestamp"]
                .as_str()
                .is_some_and(|t| t.ends_with('Z') && t.contains('T')),
            "timestamp {event}"
        );
        assert!(event["body"].is_object(), "body");
    }

    // Trace: name, session, tag.
    let trace = &clean_trace["body"];
    assert_eq!(trace["name"], "agent.run");
    assert!(trace["sessionId"].is_string(), "session grouping");
    assert_eq!(trace["tags"][0], "loopctl");

    // Parenting: tool observation nests under the turn; both share the
    // trace id.
    let trace_id = clean_trace["body"]["id"].as_str().expect("trace id");
    assert_eq!(turn_obs["body"]["traceId"], trace_id);
    assert_eq!(tool_obs["body"]["traceId"], trace_id);
    // The engine's span tree puts turns and tool executions side by side
    // under the run span, so both are direct children of the trace.
    assert!(
        turn_obs["body"]["parentObservationId"].is_null(),
        "turns attach to the trace root"
    );
    assert!(
        tool_obs["body"]["parentObservationId"].is_null(),
        "tool executions attach to the trace root (engine span tree)"
    );
    assert_eq!(
        tool_obs["body"]["metadata"]["gen_ai.tool.name"], "echo",
        "a permitted field rides the exported metadata — the query suppression \
         above must not become a metadata-wide embargo"
    );

    // The turn observation's metadata never carries the assembled
    // prompt: the engine records `query` on DEBUG turn spans for hosts
    // that opt into content capture, and the exporter must not upload
    // that text (or the PII inside it) as free-form metadata.
    assert!(
        turn_obs["body"]["metadata"].get("query").is_none(),
        "the turn's prompt text never rides the exported metadata: {}",
        turn_obs["body"]["metadata"]
    );

    // The failed run's trace carries its OWN outcome score — the
    // shared observer's read at that trace's close reflects the failed
    // run, never the successful run before it.
    let failed_trace_id = errored[0]["body"]["id"].as_str().expect("failed trace id");
    let failed_outcome = scores
        .iter()
        .find(|score| {
            score["body"]["traceId"] == failed_trace_id && score["body"]["name"] == "outcome"
        })
        .expect("the failed trace carries an outcome score");
    assert_eq!(
        failed_outcome["body"]["stringValue"], "error",
        "the failed run's outcome score reads its own run"
    );

    // Scores ride the trace and cover the contract set.
    assert!(!scores.is_empty(), "auto-scores present");
    let score_names: Vec<&str> = scores
        .iter()
        .filter_map(|s| s["body"]["name"].as_str())
        .collect();
    for want in ["outcome", "input_tokens", "token_total", "total_turns"] {
        assert!(
            score_names.contains(&want),
            "score {want} in {score_names:?}"
        );
    }
    // Scores ride their own run's trace — the source fires at every
    // trace close, reading the observer at that instant.
    let trace_ids: Vec<&str> = traces
        .iter()
        .filter_map(|t| t["body"]["id"].as_str())
        .collect();
    for score in &scores {
        let carrier = score["body"]["traceId"].as_str().expect("score trace id");
        assert!(
            trace_ids.contains(&carrier),
            "every score rides an emitted trace: {carrier} not in {trace_ids:?}"
        );
    }

    assert!(
        AUTH_OK.load(Ordering::SeqCst),
        "every ingest POST carried Basic auth"
    );
    STOP.store(true, Ordering::SeqCst);
    drop(guard);
}
