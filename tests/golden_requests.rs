//! Golden request bytes: the refactoring gate for request construction.
//!
//! Each scenario drives a deterministic fixture — the engine against a
//! recording mock, or a provider client against a local server — and
//! compares the exact outbound request it builds against a committed
//! golden file under `tests/golden/`. A refactor that changes system
//! folding, tool schema emission, memory injection, option routing, or
//! any provider's wire body fails a golden diff instead of a review
//! round.
//!
//! Golden files are reviewed artifacts. When construction changes on
//! purpose, regenerate deliberately:
//!
//! ```text
//! UPDATE_GOLDENS=1 cargo test --all-features --test golden_requests
//! ```
//!
//! and review the diff like code. Without the variable set, any
//! difference — including a missing or corrupt golden — fails loudly,
//! naming the scenario.
//!
//! Requires the `streaming` and `testing` features; the provider wire
//! scenarios additionally need their provider features.

#![cfg(all(feature = "streaming", feature = "testing"))]
#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::{Stream, StreamExt};
use loopctl::api::error::ApiError;
use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::fallback::FallbackManager;
use loopctl::managers::LoopManagers;
use loopctl::memory::{InMemoryStore, LoopMemory, MemoryCategory, MemoryEntry};
use loopctl::message::{Message, MessagePart, Role};
use loopctl::stream::StreamEvent;
use loopctl::structured::{RequestOptions, ResponseFormat};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use serde_json::{Value, json};

/// An [`ApiClient`] that records every outbound call verbatim.
///
/// Wraps a [`MockApiClient`] for response scripting while capturing the
/// exact request-and-options pair of each call, on both transports —
/// the pair the engine hands to the provider is precisely what the
/// golden files pin. An optional script of permanent auth failures
/// lets a call error before a response is served, without the response
/// queue advancing; the failed call is still recorded, exactly like a
/// served one.
struct RecordingClient {
    /// Serves the scripted responses for calls that succeed.
    ///
    /// The wrapper delegates every non-failing call to this client, so
    /// response scripting — text replies, queued tool-call turns —
    /// stays exactly as the mock defines it while the wrapper adds
    /// only the recording surface.
    inner: Arc<MockApiClient>,

    /// Every recorded call, oldest first.
    ///
    /// Each entry is the outbound request paired with the resolved
    /// request options, cloned at call time — before the stream is
    /// polled or the future is awaited — so a scenario reads the exact
    /// pair the engine handed the provider for that call.
    calls: Mutex<Vec<(StreamRequest, RequestOptions)>>,

    /// One entry per call.
    ///
    /// `Some(message)` fails that call with a permanent auth error —
    /// the kind the stream handler never retries, so one failed call
    /// is exactly one golden entry — while `None` delegates to the
    /// scripted responses. An exhausted script serves every further
    /// call normally, mirroring the mock's own error-script semantics.
    auth_failures: Mutex<Vec<Option<String>>>,
}

impl RecordingClient {
    /// Wrap `inner` for recording, optionally scripting failures.
    ///
    /// The failure script is consumed one entry per call, oldest
    /// first; pass an empty vector to record a run in which every
    /// call is served by the scripted responses.
    fn new(inner: MockApiClient, auth_failures: Vec<Option<String>>) -> Self {
        Self {
            inner: Arc::new(inner),
            calls: Mutex::new(Vec::new()),
            auth_failures: Mutex::new(auth_failures),
        }
    }

    /// A snapshot of every recorded call, oldest first.
    ///
    /// The vector is cloned out under the lock, so calls made after
    /// the snapshot cannot mutate what a scenario is asserting on.
    fn calls(&self) -> Vec<(StreamRequest, RequestOptions)> {
        self.calls.lock().expect("recorded calls lock").clone()
    }

    /// Record one call before it is served or failed.
    ///
    /// Capture happens ahead of the failure script on purpose: a
    /// failed call is recorded exactly like a served one, so the
    /// golden shows every outbound construction, including the ones
    /// that errored on the wire.
    fn record(&self, request: &StreamRequest, options: &RequestOptions) {
        self.calls
            .lock()
            .expect("recorded calls lock")
            .push((request.clone(), options.clone()));
    }

    /// Pop the next scripted auth failure, if the script still has one.
    ///
    /// An exhausted script yields `None` from then on, after which
    /// every call delegates to the mock's responses.
    fn next_auth_failure(&self) -> Option<String> {
        let mut script = self.auth_failures.lock().expect("auth failure script lock");
        if script.is_empty() {
            None
        } else {
            script.remove(0)
        }
    }
}

impl ApiClient for RecordingClient {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn stream_messages(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.stream_messages_with_options(request, RequestOptions::default())
    }

    fn stream_messages_with_options(
        &self,
        request: &StreamRequest,
        options: RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.record(request, &options);
        if let Some(message) = self.next_auth_failure() {
            return Box::pin(futures::stream::iter(vec![Err(
                ApiError::auth_invalid_key(message),
            )]));
        }
        self.inner.stream_messages_with_options(request, options)
    }

    fn create_message(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        self.create_message_with_options(request, RequestOptions::default())
    }

    fn create_message_with_options(
        &self,
        request: &StreamRequest,
        options: RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        self.record(request, &options);
        self.inner.create_message_with_options(request, options)
    }
}

/// Canonical JSON for one recorded outbound call.
///
/// Messages, parts, and tool schemas serialize through their serde
/// impls; object keys land sorted (serde_json maps are sorted maps), so
/// the golden pins content and array order, never struct field order.
fn call_to_value(request: &StreamRequest, options: &RequestOptions) -> Value {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|message| serde_json::to_value(message).expect("messages serialize"))
        .collect();
    let tools: Option<Vec<Value>> = request.tools.as_ref().map(|schemas| {
        schemas
            .iter()
            .map(|schema| serde_json::to_value(schema).expect("tool schemas serialize"))
            .collect()
    });
    json!({
        "request": {
            "system": request.system,
            "messages": messages,
            "tools": tools,
        },
        "options": options_to_value(options),
    })
}

/// Canonical JSON for the per-turn request options.
///
/// Hand-mapped because [`RequestOptions`] is not `Serialize`. The
/// `tool_constraint` rendering is the variant's `Debug` name, stable
/// for a `#[non_exhaustive]` enum of unit variants.
fn options_to_value(options: &RequestOptions) -> Value {
    let response_format = options.response_format.as_ref().map(|format| {
        json!({
            "name": format.name,
            "schema": format.schema,
            "strict": format.strict,
        })
    });
    json!({
        "model": options.model,
        "response_format": response_format,
        "tool_constraint": format!("{:?}", options.tool_constraint),
    })
}

/// The committed golden directory, resolved against the crate root.
///
/// Anchored at `CARGO_MANIFEST_DIR` rather than the process working
/// directory, so the tests locate the goldens no matter where cargo
/// launches the test binary from.
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// The golden file path for one scenario.
///
/// Scenario names map one-to-one onto file names, so a failure naming
/// its scenario points straight at the file to inspect or regenerate.
fn golden_path(scenario: &str) -> PathBuf {
    golden_dir().join(format!("{scenario}.json"))
}

/// Whether a raw `UPDATE_GOLDENS` value asks for regeneration.
///
/// Only the exact value `1` counts; every other setting — including
/// `yes` or an empty string — stays in compare mode, so a stray
/// variable never silently rewrites goldens. Split out as a pure
/// predicate so the meta-pins can cover every direction without
/// mutating the process environment.
fn is_update_mode(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Whether the environment asks for golden regeneration.
///
/// The wiring seam between the `UPDATE_GOLDENS` variable and the
/// predicate that interprets it. Every golden assertion consults it at
/// call time, so the mode applies per run rather than per binary.
fn updating() -> bool {
    is_update_mode(std::env::var("UPDATE_GOLDENS").ok().as_deref())
}

/// The on-disk rendering: pretty JSON, one trailing newline.
///
/// Object keys arrive sorted by construction (serde_json maps are
/// sorted), so the rendering is canonical: two runs that build the
/// same value produce byte-identical files, which is what makes a
/// regenerated golden's diff reviewable.
fn render_canonical(value: &Value) -> String {
    let mut text = serde_json::to_string_pretty(value).expect("canonical JSON renders");
    text.push('\n');
    text
}

/// Compare `actual` against the golden at `path`.
///
/// Any difference — a drifted value, an unreadable file, or a golden
/// that is not valid JSON — panics with the scenario name and, for a
/// drift, both renderings, so the failure reads as a diff rather than
/// a bare inequality.
fn compare_golden(scenario: &str, path: &Path, actual: &Value) {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!("golden {scenario}: cannot read {}: {error}", path.display())
    });
    let expected: Value = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "golden {scenario}: {} is not valid JSON: {error}",
            path.display()
        )
    });
    if expected != *actual {
        panic!(
            "golden {scenario} drifted: request construction changed; if the change is \
             intentional, regenerate with UPDATE_GOLDENS=1 and review the diff\n\n\
             committed golden:\n{}\n\nbuilt this run:\n{}",
            render_canonical(&expected),
            render_canonical(actual),
        );
    }
}

/// Write `value` as the golden at `path`.
///
/// Creates the containing directory when missing and renders exactly
/// what the comparison reads back, so a regeneration round cannot
/// leave a file the next compare run would reject.
fn write_golden(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("the golden directory is creatable");
    }
    std::fs::write(path, render_canonical(value)).expect("the golden file is writable");
    println!("updated golden {}", path.display());
}

/// The mode-dispatching core of [`assert_golden`], split out so the
/// meta-pins can drive both directions without touching the process
/// environment.
fn assert_golden_mode(scenario: &str, path: &Path, value: &Value, update: bool) {
    if update {
        write_golden(path, value);
    } else {
        compare_golden(scenario, path, value);
    }
}

/// Compare `value` against the committed golden for `scenario`, or
/// write it when the environment asked for regeneration.
///
/// The scenario-facing entry point: every scenario funnels through
/// here, so the compare and write modes cannot diverge per call site,
/// and the committed file is the only thing compare mode ever reads.
fn assert_golden(scenario: &str, value: &Value) {
    assert_golden_mode(scenario, &golden_path(scenario), value, updating());
}

/// Golden every call a client recorded, oldest first.
///
/// The array order is the call order, so a multi-turn scenario pins
/// not just each request's construction but the sequence of
/// constructions the engine issued to produce the run.
fn golden_calls(scenario: &str, calls: &[(StreamRequest, RequestOptions)]) {
    let value = Value::Array(
        calls
            .iter()
            .map(|(request, options)| call_to_value(request, options))
            .collect(),
    );
    assert_golden(scenario, &value);
}

/// A deterministic registry tool: fixed schema, fixed reply, no side
/// effects.
///
/// The engine scenarios need tools whose advertised schemas and
/// executed outputs are byte-stable across runs — everything a tool
/// contributes to a later request must be reproducible for a golden
/// to pin it.
struct FixtureTool {
    /// The name the registry and the request schemas carry.
    ///
    /// Registration order across fixture tools is part of what the
    /// `system_tools` golden pins, so the names are chosen and
    /// registered deliberately, never sorted.
    name: &'static str,

    /// The one-line description the schema advertises.
    ///
    /// Flows verbatim into the advertised `ToolSchema`, so the golden
    /// pins the description text exactly as the model would see it.
    description: &'static str,

    /// The reply served for every invocation, regardless of input.
    ///
    /// A fixed reply keeps the tool-result content deterministic in
    /// later turns — the next request's golden depends on it.
    reply: &'static str,
}

impl Tool for FixtureTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name.to_string(),
            self.description.to_string(),
            json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }),
        )
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let reply = self.reply;
        Box::pin(async move { Ok(ToolOutput::text(reply)) })
    }
}

/// The one conversation every provider converter renders to its native
/// wire body.
///
/// Carries a system prompt, a tool request, the assistant's tool call,
/// and the tool's result, so each dialect has to exercise its full
/// mapping surface — and all three wire goldens answer to the same
/// fixture, which keeps cross-provider comparisons meaningful.
fn shared_conversation() -> StreamRequest {
    StreamRequest::new(vec![
        Message::user("Check the weather in Paris with the weather tool."),
        Message::new(
            Role::Assistant,
            vec![MessagePart::tool_call(
                "call_gold_weather",
                "get_weather",
                json!({"city": "Paris"}),
            )],
        ),
        Message::new(
            Role::User,
            vec![MessagePart::tool_result(
                "call_gold_weather",
                "get_weather",
                "18 degrees, clear sky",
                false,
            )],
        ),
    ])
    .with_system(Some("You are a concise weather agent.".to_string()))
    .with_tools(Some(vec![ToolSchema::new(
        "get_weather".to_string(),
        "Look up the current weather for a city".to_string(),
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )]))
}

/// Capture the exact body a provider client puts on the wire for the
/// shared conversation, canonicalize it, and golden it under
/// `scenario`.
///
/// The local server mocks no response, so the client's request meets a
/// 404 — the body is already on the wire by then, the recording holds
/// it, and the client's error is expected and discarded.
async fn golden_wire_body<C: ApiClient>(scenario: &str, build_client: impl FnOnce(&str) -> C) {
    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let client = build_client(&server.base_url());
    let request = shared_conversation();
    let mut stream = client.stream_messages(&request);
    // Drain to completion: the 404 arrives as an error event, and only
    // the request the client built matters here.
    while stream.next().await.is_some() {}
    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the provider client sent its request");
    let body = first_recorded_request_body(&String::from_utf8_lossy(&bytes));
    let value: Value = serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("golden {scenario}: the wire body is not JSON: {error}"));
    assert_golden(scenario, &value);
}

/// The first textual request body in a recording export.
///
/// The export is a multi-document YAML stream with one interaction per
/// request, and a wire scenario sends exactly one request, so the
/// first `when.body` is the body under test. Panics when no document
/// carries a textual body — a binary body would mean the provider
/// serialized something other than the JSON a golden can canon.
fn first_recorded_request_body(export: &str) -> String {
    for document in serde_yaml::Deserializer::from_str(export) {
        let value: serde_yaml::Value =
            serde::Deserialize::deserialize(document).expect("the recording export parses");
        let body = value
            .get("when")
            .and_then(|when| when.get("body"))
            .and_then(|body| body.as_str());
        if let Some(body) = body {
            return body.to_string();
        }
    }
    panic!("the recording holds no textual request body");
}

#[tokio::test]
async fn minimal_chat_request_is_golden() {
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("golden-model").with_text_response("ready"),
        Vec::new(),
    ));
    let mut agent = BareLoop::new(
        Arc::clone(&client),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    let run = agent
        .run("Reply with the single word: ready.", &RunConfig::default())
        .await;
    assert!(
        run.is_ok(),
        "golden minimal_chat: the scripted run must complete: {run:?}"
    );
    golden_calls("minimal_chat", &client.calls());
}

#[tokio::test]
async fn system_and_tool_schemas_are_golden() {
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("golden-model").with_text_response("probed"),
        Vec::new(),
    ));
    let mut registry = ToolRegistry::new();
    registry.register(FixtureTool {
        name: "read_note",
        description: "Read a saved note by title",
        reply: "note-body",
    });
    registry.register(FixtureTool {
        name: "search_notes",
        description: "Search saved notes for a term",
        reply: "note-hit",
    });
    registry.register(FixtureTool {
        name: "write_note",
        description: "Save a note with a title and body",
        reply: "written",
    });
    let mut agent = BareLoop::new(
        Arc::clone(&client),
        registry,
        SessionConfig::default().with_system_prompt("You are a deterministic fixture agent."),
    );
    let run = agent
        .run("Probe the registered note tools.", &RunConfig::default())
        .await;
    assert!(
        run.is_ok(),
        "golden system_tools: the scripted run must complete: {run:?}"
    );
    golden_calls("system_tools", &client.calls());
}

#[tokio::test]
async fn multi_turn_tool_results_are_golden() {
    let responses = vec![
        MockResponse {
            text: "calling the echo tool".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_gold_echo".to_string(),
                name: "echo".to_string(),
                input: json!({"text": "ping"}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "echoed".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ];
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("golden-model").with_responses(responses),
        Vec::new(),
    ));
    let mut registry = ToolRegistry::new();
    registry.register(FixtureTool {
        name: "echo",
        description: "Echo a word back",
        reply: "pong",
    });
    let mut agent = BareLoop::new(Arc::clone(&client), registry, SessionConfig::default());
    let run = agent
        .run(
            "Echo the word ping through the echo tool.",
            &RunConfig::default(),
        )
        .await;
    assert!(
        run.is_ok(),
        "golden multi_turn_tool_results: the scripted run must complete: {run:?}"
    );
    golden_calls("multi_turn_tool_results", &client.calls());
}

#[tokio::test]
async fn fallback_routing_is_golden() {
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager
        .set_fallback_model("fallback-model")
        .expect("the fallback model registers");
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("primary-model").with_text_response("served by the fallback"),
        vec![Some("scripted permanent auth failure".to_string())],
    ));
    let managers = LoopManagers::new().with_fallback(manager);
    let mut agent = BareLoop::new_with_managers(
        Arc::clone(&client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    let first = agent
        .run(
            "Route me to whichever model is healthy.",
            &RunConfig::default(),
        )
        .await;
    assert!(
        first.is_err(),
        "golden fallback_routed: the scripted failure must fail the first run"
    );
    let second = agent
        .run(
            "Route me to whichever model is healthy.",
            &RunConfig::default(),
        )
        .await;
    assert!(
        second.is_ok(),
        "golden fallback_routed: the fallback model serves the second run: {second:?}"
    );
    golden_calls("fallback_routed", &client.calls());
}

#[tokio::test]
async fn constrained_decode_is_golden() {
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("golden-model").with_text_response("{\"greeting\": \"hello\"}"),
        Vec::new(),
    ));
    let mut registry = ToolRegistry::new();
    registry.register(FixtureTool {
        name: "read_note",
        description: "Read a saved note by title",
        reply: "note-body",
    });
    let mut agent = BareLoop::new(Arc::clone(&client), registry, SessionConfig::default());
    agent.set_request_options(RequestOptions::new().with_response_format(ResponseFormat {
        name: "greeting".to_string(),
        schema: json!({
            "type": "object",
            "properties": {"greeting": {"type": "string"}},
            "required": ["greeting"],
            "additionalProperties": false
        }),
        strict: true,
    }));
    let run = agent
        .run("Return a greeting object.", &RunConfig::default())
        .await;
    assert!(
        run.is_ok(),
        "golden constrained_decode: the scripted run must complete: {run:?}"
    );
    golden_calls("constrained_decode", &client.calls());
}

#[tokio::test]
async fn memory_injection_is_golden() {
    let memory = Arc::new(InMemoryStore::new());
    memory
        .store(MemoryEntry::new(
            MemoryCategory::Insight,
            "catch request drift with committed golden bytes",
        ))
        .await
        .expect("the memory store accepts the fixture entry");
    memory
        .store(MemoryEntry::new(
            MemoryCategory::Strategy,
            "a refactor changes request construction only when its goldens change",
        ))
        .await
        .expect("the memory store accepts the fixture entry");
    let client = Arc::new(RecordingClient::new(
        MockApiClient::new("golden-model").with_text_response("caught"),
        Vec::new(),
    ));
    let managers = LoopManagers::new().with_memory(memory);
    let mut agent = BareLoop::new_with_managers(
        Arc::clone(&client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    let run = agent
        .run(
            "How should request drift be caught in a refactor of request construction?",
            &RunConfig::default(),
        )
        .await;
    assert!(
        run.is_ok(),
        "golden memory_injected: the scripted run must complete: {run:?}"
    );
    golden_calls("memory_injected", &client.calls());
}

#[tokio::test]
#[cfg(feature = "openai")]
async fn openai_wire_body_is_golden() {
    golden_wire_body("openai_wire", |base| {
        loopctl::provider::OpenAiClient::builder()
            .with_api_key("golden-key")
            .with_base_url(format!("{base}/v1"))
            .with_model("golden-model")
            .build()
            .expect("the openai client builds")
    })
    .await;
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn anthropic_wire_body_is_golden() {
    golden_wire_body("anthropic_wire", |base| {
        loopctl::provider::AnthropicClient::builder()
            .with_api_key("golden-key")
            .with_base_url(base.to_string())
            .with_model("golden-model")
            .build()
            .expect("the anthropic client builds")
    })
    .await;
}

#[tokio::test]
#[cfg(feature = "gemini")]
async fn gemini_wire_body_is_golden() {
    golden_wire_body("gemini_wire", |base| {
        loopctl::provider::GeminiClient::builder()
            .with_api_key("golden-key")
            .with_base_url(format!("{base}/v1beta"))
            .with_model("golden-model")
            .build()
            .expect("the gemini client builds")
    })
    .await;
}

/// A fresh scratch directory unique to this test process.
///
/// Uniqueness comes from the process id plus the tag, so parallel test
/// binaries never share a scratch path. The meta-pins own creation and
/// cleanup of their directory; scenario tests never touch scratch
/// space — their goldens live in the committed tree.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("loopctl-golden-pin-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the scratch directory is creatable");
    dir
}

/// The panic payload as text, for asserting on failure messages.
///
/// `catch_unwind` yields an opaque `Any` that carries either a `String`
/// or a `&str` depending on how the panic was formatted; this renders
/// either to owned text so a pin can assert that a failure named its
/// scenario without matching on the payload type first.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[test]
fn golden_diff_fails_loudly() {
    let dir = scratch_dir("diff");
    let path = dir.join("drifted.json");
    std::fs::write(&path, render_canonical(&json!({"answer": "frozen"})))
        .expect("the scratch golden writes");

    let drift = std::panic::catch_unwind(|| {
        compare_golden("drift_pin", &path, &json!({"answer": "rebuilt"}));
    })
    .expect_err("a drifted construction must fail the comparison");
    let message = panic_message(&drift);
    assert!(
        message.contains("drift_pin") && message.contains("drifted"),
        "the failure must name the scenario and the drift: {message}"
    );

    let missing = std::panic::catch_unwind(|| {
        compare_golden("absent_pin", &dir.join("absent.json"), &json!({}));
    })
    .expect_err("a missing golden must fail the comparison");
    let message = panic_message(&missing);
    assert!(
        message.contains("absent_pin"),
        "the missing-golden failure must name the scenario: {message}"
    );

    let corrupt = dir.join("corrupt.json");
    std::fs::write(&corrupt, "not json").expect("the corrupt golden writes");
    let corrupt_panic = std::panic::catch_unwind(|| {
        compare_golden("corrupt_pin", &corrupt, &json!({}));
    })
    .expect_err("a corrupt golden must fail the comparison");
    let message = panic_message(&corrupt_panic);
    assert!(
        message.contains("corrupt_pin"),
        "the corrupt-golden failure must name the scenario: {message}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn update_goldens_is_explicit() {
    let dir = scratch_dir("explicit");
    let path = dir.join("explicit.json");
    let frozen = render_canonical(&json!({"mode": "frozen"}));
    std::fs::write(&path, &frozen).expect("the scratch golden writes");

    assert!(!is_update_mode(None), "an unset variable is compare mode");
    assert!(
        !is_update_mode(Some("yes")),
        "only the exact value 1 is update mode"
    );
    assert!(is_update_mode(Some("1")));

    let drift = std::panic::catch_unwind(|| {
        assert_golden_mode(
            "explicit_pin",
            &path,
            &json!({"mode": "regenerated"}),
            false,
        );
    })
    .expect_err("compare mode must fail a drift instead of writing it");
    let message = panic_message(&drift);
    assert!(
        message.contains("explicit_pin"),
        "the failure must name the scenario: {message}"
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("the scratch golden still reads"),
        frozen,
        "compare mode must leave the golden untouched"
    );

    assert_golden_mode("explicit_pin", &path, &json!({"mode": "regenerated"}), true);
    assert_eq!(
        std::fs::read_to_string(&path).expect("the regenerated golden reads"),
        render_canonical(&json!({"mode": "regenerated"})),
        "update mode rewrites the golden to the built value"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The structured-exclusivity claim, pinned at the wire: a
/// `response_format` option removes the tool array from the body the
/// provider built — a regression emitting both fails here instead of
/// only against the live API.
async fn constrained_wire_omits_tools<C: ApiClient>(build_client: impl FnOnce(&str) -> C) -> Value {
    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let client = build_client(&server.base_url());
    let request = shared_conversation();
    let options = loopctl::structured::RequestOptions::new().with_response_format(
        loopctl::structured::ResponseFormat {
            name: "out".to_string(),
            schema: serde_json::json!({"type": "object"}),
            strict: false,
        },
    );
    let mut stream = client.stream_messages_with_options(&request, options);
    while stream.next().await.is_some() {}
    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the provider client sent its request");
    let body = first_recorded_request_body(&String::from_utf8_lossy(&bytes));
    let value: Value = serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("the constrained wire body is not JSON: {error}"));
    value
}

#[tokio::test]
#[cfg(feature = "openai")]
async fn openai_constrained_wire_omits_tools() {
    let value = constrained_wire_omits_tools(|base| {
        loopctl::provider::OpenAiClient::builder()
            .with_api_key("golden-key")
            .with_base_url(format!("{base}/v1"))
            .with_model("golden-model")
            .build()
            .expect("the openai client builds")
    })
    .await;
    assert!(
        value.get("tools").is_none(),
        "the openai-shaped body carries no tools array at all under a format"
    );
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn anthropic_constrained_wire_omits_tools() {
    let value = constrained_wire_omits_tools(|base| {
        loopctl::provider::AnthropicClient::builder()
            .with_api_key("golden-key")
            .with_base_url(base.to_string())
            .with_model("golden-model")
            .build()
            .expect("the anthropic client builds")
    })
    .await;
    // Anthropic expresses the format as a forced tool: the registry's
    // tools are gone and exactly the forced entry remains.
    let tools = value
        .get("tools")
        .and_then(Value::as_array)
        .expect("the forced tool rides the tools array");
    assert_eq!(
        tools.len(),
        1,
        "one forced tool, not the registry: {tools:?}"
    );
    assert_eq!(
        tools[0].get("name").and_then(Value::as_str),
        Some("out"),
        "the forced entry is the format's own tool"
    );
}

#[tokio::test]
#[cfg(feature = "gemini")]
async fn gemini_constrained_wire_omits_tools() {
    let value = constrained_wire_omits_tools(|base| {
        loopctl::provider::GeminiClient::builder()
            .with_api_key("golden-key")
            .with_base_url(base.to_string())
            .with_model("golden-model")
            .build()
            .expect("the gemini client builds")
    })
    .await;
    assert!(
        value.get("tools").is_none(),
        "the gemini body carries no tools array at all under a format"
    );
}
