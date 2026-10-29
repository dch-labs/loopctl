//! The privacy claim in the crate docs, checked over the whole payload: no
//! exported body carries prompt text.
//!
//! The engine records the assembled prompt on the `DEBUG` turn span, and this
//! subscriber accepts `DEBUG`, so the text is available to the layer at every
//! turn. The assertion is not that one field is missing but that the prompt
//! appears nowhere in what a Langfuse deployment receives — envelope
//! wrappers, trace bodies, observation metadata, and scores all swept.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::panic
)]

use std::future::Future;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_langfuse::{LangfuseConfig, install};
use serde_json::Value;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Captured ingestion events (one JSON value per envelope).
static EVENTS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

/// The turn's input text, distinctive enough that no generated id, metric
/// name, or tool label could contain it by accident.
const PROMPT: &str = "prompt-text-never-ships-4f9b2e7c";

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

/// One scripted tool-call turn plus the terminal reply that closes the run,
/// so the run opens a tool span as well as a turn span.
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
async fn no_exported_body_carries_the_turn_prompt() {
    let port = spawn_mock();

    let (layer, guard) = install(LangfuseConfig {
        host: format!("http://127.0.0.1:{port}"),
        public_key: "pk-test".to_string(),
        secret_key: "sk-test".to_string(),
        batch_size: 100,
    })
    .expect("install");
    // No filter: every level the engine emits is of interest, which is the
    // host configuration the prompt text rides.
    let subscriber = Registry::default().with(layer);

    let client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut engine = BareLoop::new(client, registry, SessionConfig::default());

    {
        let _guard = tracing::subscriber::set_default(subscriber);
        let result = engine.run(PROMPT, &RunConfig::default()).await;
        assert!(result.is_ok(), "the run completes: {result:?}");
    }

    let drained = guard.flush(std::time::Duration::from_secs(5)).await;
    assert!(
        drained,
        "every queued envelope ships before the drain returns"
    );

    let events = EVENTS.lock().unwrap().clone();
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "span-create" && event["body"]["name"] == "agent.turn"),
        "the batch carries the turn observation whose span records the prompt, \
         so the sweep below is not scanning an empty payload"
    );

    for event in &events {
        let rendered = event.to_string();
        assert!(
            !rendered.contains(PROMPT),
            "the prompt text reached an exported envelope: {rendered}"
        );
    }

    drop(guard);
}
