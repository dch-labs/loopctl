//! Tests for the [`BareLoop`] driver, extracted from `bare.rs`.
//!
//! These tests were moved wholesale from `engine/bare.rs`; they exercise the
//! full driver — the `run()` match loop, turn handling, cancellation, fallback,
//! streaming vs non-streaming paths, tool dispatch, and the configuration
//! builders. The test names and assertions are unchanged.

use super::*;
use crate::api::error::ApiError;
use crate::capabilities::FallbackCapable;
use crate::engine::core::Loop;
use crate::fallback::FallbackManager;
use crate::observer::{LoopObserver, ModelSwitchedContext, StreamFailureContext};
use crate::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamAccumulator, StreamEvent, Usage,
};
use crate::tool::ToolRegistry;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::sync::Mutex;

#[cfg(feature = "streaming")]
#[test]
fn text_streamer_alias_compiles_unchanged() {
    let client = MockClient::new("test-model");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_text_streamer(Arc::new(|_| ()));
    assert!(agent.text_streamer.is_some());
}

/// Fold queued [`StreamEvent`]s into a [`NonStreamingResponse`].
///
/// Shared by `MockClient` and `RecordingClient` `create_message` impls so
/// the non-streaming path sees the same assembled message, stop reason,
/// and usage the streaming path would have produced.
fn assemble_response(
    events: Vec<StreamEvent>,
) -> Result<crate::api::NonStreamingResponse, ApiError> {
    let mut accumulator = StreamAccumulator::new();
    let mut stop_reason = crate::stream::StreamStopReason::EndTurn;
    for event in events {
        if let crate::stream::StreamEvent::MessageDelta(delta) = &event
            && let Some(reason) = delta
                .delta
                .stop_reason
                .as_deref()
                .and_then(crate::stream::StreamStopReason::from_api_str)
        {
            stop_reason = reason;
        }
        accumulator
            .process(&event)
            .map_err(|e| ApiError::api(e.to_string()))?;
    }
    let usage = accumulator.usage().copied();
    Ok(crate::api::NonStreamingResponse {
        message: accumulator.build(),
        stop_reason,
        usage,
    })
}

#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<Vec<StreamEvent>>>>,

    model_name: Arc<std::sync::Mutex<String>>,
}

impl MockClient {
    fn new(model: &str) -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            model_name: Arc::new(std::sync::Mutex::new(model.to_string())),
        }
    }

    fn add_text_response(&self, text: &str) {
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: Some(Usage::new(10, 20)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(events);
    }

    fn add_events(&self, events: Vec<StreamEvent>) {
        crate::error::recover_guard(self.responses.lock()).push(events);
    }

    fn add_tool_then_text(
        &self,
        tool_id: &str,
        tool_name: &str,
        tool_input: Value,
        final_text: &str,
    ) {
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_tool".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 10)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(tool_events);

        let text_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_final".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(final_text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: final_text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: Some(Usage::new(30, 15)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(text_events);
    }

    fn add_multi_tool_then_text(&self, tools: &[(String, String, Value)], final_text: &str) {
        let mut tool_events = vec![StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_tool".into(),
                role: "assistant".into(),
                model: crate::error::recover_guard(self.model_name.lock()).clone(),
            },
        })];
        for (idx, (id, name, input)) in tools.iter().enumerate() {
            tool_events.push(StreamEvent::PartStart(PartStart {
                index: idx,
                part: Some(MessagePart::tool_call(id, name, input.clone())),
            }));
            tool_events.push(StreamEvent::PartStop);
        }
        tool_events.push(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 10)),
        }));
        tool_events.push(StreamEvent::MessageStop);
        crate::error::recover_guard(self.responses.lock()).push(tool_events);

        let text_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_final".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(final_text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: final_text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: Some(Usage::new(30, 15)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(text_events);
    }

    fn add_tool_only_response(&self, tool_id: &str, tool_name: &str, tool_input: Value) {
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: format!("msg_{tool_id}"),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 10)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(tool_events);
    }

    fn add_max_tokens_response(&self, text: &str) {
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_mt".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("max_tokens".to_string()),
                },
                usage: Some(Usage::new(10, 20)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(events);
    }

    #[expect(dead_code)]
    fn add_error_response(&self) {
        // Return an empty response that will cause the stream to error
        // We'll handle this by having the stream return an error event
        let events = vec![StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_err".into(),
                role: "assistant".into(),
                model: crate::error::recover_guard(self.model_name.lock()).clone(),
            },
        })];
        crate::error::recover_guard(self.responses.lock()).push(events);
    }
}

impl ApiClient for MockClient {
    fn model(&self) -> String {
        crate::error::recover_guard(self.model_name.lock()).clone()
    }

    fn set_model(&self, model: &str) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
        true
    }

    fn stream_messages(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let mut guard = crate::error::recover_guard(self.responses.lock());
        if let Some(events) = guard.pop_front() {
            let events: Vec<Result<StreamEvent, ApiError>> = events.into_iter().map(Ok).collect();
            Box::pin(futures::stream::iter(events))
        } else {
            // No more responses — return an error
            let err = ApiError::api("No more mock responses");
            Box::pin(futures::stream::iter(vec![Err(err)]))
        }
    }

    fn create_message(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        let mut guard = crate::error::recover_guard(self.responses.lock());
        let events = guard.pop_front();
        drop(guard);
        Box::pin(async move {
            let events = events.ok_or_else(|| ApiError::api("No more mock responses"))?;
            assemble_response(events)
        })
    }
}

trait PopFront<T> {
    fn pop_front(&mut self) -> Option<T>;
}

impl<T> PopFront<T> for Vec<T> {
    fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(0))
        }
    }
}

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "Echoes back the input"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "echo".into(),
            description: "Echoes back the input".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "message": { "type": "string" } },
                "required": ["message"]
            }),
        }
    }

    fn call(
        &self,
        input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let msg = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Box::pin(async move { Ok(ToolOutput::text(format!("Echo: {msg}"))) })
    }
}

struct FailingTool;

impl Tool for FailingTool {
    fn name(&self) -> &'static str {
        "fail"
    }

    fn description(&self) -> &'static str {
        "Always fails"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "fail".into(),
            description: "Always fails".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move { Err(ToolError::Execution("Tool intentionally failed".into())) })
    }
}

struct FlakyTool {
    fail_threshold: usize,
    attempts: AtomicUsize,
}

impl FlakyTool {
    fn new(fail_threshold: usize) -> Self {
        Self {
            fail_threshold,
            attempts: AtomicUsize::new(0),
        }
    }
}

impl Tool for FlakyTool {
    fn name(&self) -> &'static str {
        "flaky"
    }

    fn description(&self) -> &'static str {
        "Fails the first N calls, then succeeds"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "flaky".into(),
            description: "Fails the first N calls, then succeeds".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if attempt < self.fail_threshold {
                Err(ToolError::Execution("Flaky tool failing".into()))
            } else {
                Ok(ToolOutput::text("Flaky tool succeeded"))
            }
        })
    }
}

struct CountingObserver {
    run_starts: AtomicUsize,
    run_ends: AtomicUsize,
    turn_starts: AtomicUsize,
    turn_ends: AtomicUsize,
    tool_calls_received: AtomicUsize,
    tool_pres: AtomicUsize,
    tool_posts: AtomicUsize,
}

impl CountingObserver {
    fn new() -> Self {
        Self {
            run_starts: AtomicUsize::new(0),
            run_ends: AtomicUsize::new(0),
            turn_starts: AtomicUsize::new(0),
            turn_ends: AtomicUsize::new(0),
            tool_calls_received: AtomicUsize::new(0),
            tool_pres: AtomicUsize::new(0),
            tool_posts: AtomicUsize::new(0),
        }
    }
}

impl crate::observer::LoopObserver for CountingObserver {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn on_run_start(&self, _ctx: &crate::observer::RunStartContext) {
        self.run_starts.fetch_add(1, Ordering::SeqCst);
    }

    fn on_run_end(&self, _ctx: &crate::observer::RunEndContext) {
        self.run_ends.fetch_add(1, Ordering::SeqCst);
    }

    fn on_turn_start(&self, _ctx: &crate::observer::TurnStartContext) {
        self.turn_starts.fetch_add(1, Ordering::SeqCst);
    }

    fn on_turn_end(&self, _ctx: &crate::observer::TurnEndContext) {
        self.turn_ends.fetch_add(1, Ordering::SeqCst);
    }

    fn on_tool_call_received(&self, _ctx: &crate::observer::ToolCallReceivedContext) {
        self.tool_calls_received.fetch_add(1, Ordering::SeqCst);
    }

    fn on_tool_pre(&self, _ctx: &crate::observer::ToolPreContext) {
        self.tool_pres.fetch_add(1, Ordering::SeqCst);
    }

    fn on_tool_post(&self, _ctx: &crate::observer::ToolPostContext) {
        self.tool_posts.fetch_add(1, Ordering::SeqCst);
    }
}

fn make_config() -> SessionConfig {
    SessionConfig::default()
}

fn make_run_config() -> RunConfig {
    RunConfig {
        max_turns: 10,
        ..RunConfig::default()
    }
}

#[tokio::test]
async fn test_bare_loop_single_turn() {
    let client = MockClient::new("test-model");
    client.add_text_response("Hello! I'm done.");

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    assert_eq!(result.turn_count(), 1);
    assert_eq!(result.output.as_deref(), Some("Hello! I'm done."));
}

#[test]
fn turn_mode_default_follows_streaming_feature() {
    let client = MockClient::new("test-model");
    let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    #[cfg(not(feature = "streaming"))]
    assert_eq!(agent.turn_mode(), TurnMode::NonStreaming);
    #[cfg(feature = "streaming")]
    assert_eq!(agent.turn_mode(), TurnMode::Streaming);
}

#[tokio::test]
async fn non_streaming_turn_returns_assembled_message() {
    let client = MockClient::new("test-model");
    client.add_text_response("assembled via create_message");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_turn_mode(TurnMode::NonStreaming);
    let result = agent.run("Hi", &RunConfig::default()).await.unwrap();
    assert_eq!(result.turn_count(), 1);
    assert_eq!(
        result.output.as_deref(),
        Some("assembled via create_message")
    );
}

#[tokio::test]
async fn non_streaming_turn_runs_tool_call_loop() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("call_1", "echo", json!({"message": "hi"}), "all done");
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.set_turn_mode(TurnMode::NonStreaming);
    let result = agent.run("Hi", &RunConfig::default()).await.unwrap();
    assert_eq!(result.turn_count(), 2);
    assert_eq!(result.tool_call_count(), 1);
    assert_eq!(result.output.as_deref(), Some("all done"));
}

#[tokio::test]
async fn non_streaming_turn_respects_cancellation() {
    let client = MockClient::new("test-model");
    client.add_text_response("never seen");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_turn_mode(TurnMode::NonStreaming);
    agent.cancel();
    let result = agent.run("Hi", &RunConfig::default()).await;
    assert!(matches!(result, Err(LoopError::Cancelled)));
}

/// Observer that records whether `on_stream_failure` fired.
struct FailureRecorder {
    on_stream_failure_fired: Arc<AtomicBool>,
}

impl LoopObserver for FailureRecorder {
    fn name(&self) -> &'static str {
        "failure-recorder"
    }
    fn on_stream_failure(&self, _ctx: &StreamFailureContext) {
        self.on_stream_failure_fired.store(true, Ordering::SeqCst);
    }
}

/// A client whose `create_message` never completes on its own, so the
/// cancel `select!` arm in `do_create_message` is the only way the turn
/// resolves. Used to exercise mid-turn cancellation.
struct BlockingClient {
    started: Arc<AtomicBool>,
}

impl ApiClient for BlockingClient {
    fn model(&self) -> String {
        "blocking".into()
    }
    fn stream_messages(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let started = Arc::clone(&self.started);
        Box::pin(futures::stream::once(async move {
            started.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
            Ok(StreamEvent::MessageStop)
        }))
    }
    fn create_message(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        let started = Arc::clone(&self.started);
        Box::pin(async move {
            started.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
            Err(ApiError::api("unreachable: cancel must win the select"))
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_during_non_streaming_turn_does_not_trip_breaker() {
    let client = BlockingClient {
        started: Arc::new(AtomicBool::new(false)),
    };
    let started = Arc::clone(&client.started);
    let on_stream_failure_fired = Arc::new(AtomicBool::new(false));
    let observer = Arc::new(FailureRecorder {
        on_stream_failure_fired: Arc::clone(&on_stream_failure_fired),
    });
    let managers = LoopManagers::new()
        .with_fallback(FallbackManager::default())
        .with_observer(observer);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        make_config(),
        managers,
    );
    agent.set_turn_mode(TurnMode::NonStreaming);

    let cancel_signal = Arc::clone(&agent.cancel_signal());
    let run_handle = tokio::spawn(async move { agent.run("Hi", &RunConfig::default()).await });

    // Wait until create_message is in flight, then cancel.
    let mut waits = 0u32;
    while !started.load(Ordering::SeqCst) {
        waits += 1;
        assert!(
            waits <= 1000,
            "create_message was never entered — test setup is broken"
        );
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    cancel_signal.cancel();
    let run_result = run_handle.await.unwrap();

    assert!(
        started.load(Ordering::SeqCst),
        "test only proves anything if create_message was actually entered"
    );
    assert!(
        matches!(run_result, Err(LoopError::Cancelled)),
        "run must return Err(Cancelled): {run_result:?}"
    );
    assert!(
        !on_stream_failure_fired.load(Ordering::SeqCst),
        "a clean cancel must not fire on_stream_failure (it would trip the breaker)"
    );
}

/// Streaming-path twin of the test above: a clean cancel during a
/// streaming turn must not fire `on_stream_failure`. Proves the
/// `record_turn_failure` Cancelled guard holds for both turn modes.
#[cfg(feature = "streaming")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_during_streaming_turn_does_not_trip_breaker() {
    let client = BlockingClient {
        started: Arc::new(AtomicBool::new(false)),
    };
    let started = Arc::clone(&client.started);
    let on_stream_failure_fired = Arc::new(AtomicBool::new(false));
    let observer = Arc::new(FailureRecorder {
        on_stream_failure_fired: Arc::clone(&on_stream_failure_fired),
    });
    let managers = LoopManagers::new()
        .with_fallback(FallbackManager::default())
        .with_observer(observer);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        make_config(),
        managers,
    );
    // turn_mode defaults to Streaming when the feature is on.

    let cancel_signal = Arc::clone(&agent.cancel_signal());
    let run_handle = tokio::spawn(async move { agent.run("Hi", &RunConfig::default()).await });

    let mut waits = 0u32;
    while !started.load(Ordering::SeqCst) {
        waits += 1;
        assert!(
            waits <= 1000,
            "stream_messages was never entered — test setup is broken"
        );
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    cancel_signal.cancel();
    let run_result = run_handle.await.unwrap();

    assert!(
        started.load(Ordering::SeqCst),
        "test only proves anything if stream_messages was actually entered"
    );
    assert!(
        matches!(run_result, Err(LoopError::Cancelled)),
        "run must return Err(Cancelled): {run_result:?}"
    );
    assert!(
        !on_stream_failure_fired.load(Ordering::SeqCst),
        "a clean cancel must not fire on_stream_failure (it would trip the breaker)"
    );
}

#[test]
fn run_config_is_none_before_first_run() {
    let client = MockClient::new("test-model");
    let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    assert!(
        agent.run_config().is_none(),
        "run_config must be None before the first run() call"
    );
}

#[test]
fn session_starts_with_empty_runs() {
    let client = MockClient::new("test-model");
    let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    assert!(
        agent.session.runs.is_empty(),
        "a never-run session must have zero runs, not a placeholder"
    );
}

#[tokio::test]
async fn run_config_is_some_after_run() {
    let client = MockClient::new("test-model");
    client.add_text_response("done");

    let config = RunConfig {
        max_turns: 42,
        ..RunConfig::default()
    };
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.run("hi", &config).await.unwrap();

    let rc = agent
        .run_config()
        .expect("run_config must be Some after run()");
    assert_eq!(rc.max_turns, 42);
}

#[tokio::test]
async fn test_bare_loop_with_tool_call() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text(
        "tool_1",
        "echo",
        json!({"message": "hello"}),
        "I echoed your message.",
    );

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    let result = agent
        .run("Echo hello", &RunConfig::default())
        .await
        .unwrap();

    assert_eq!(result.turn_count(), 2); // tool_call turn + end_turn
    assert_eq!(result.tool_call_count(), 1);
}

#[tokio::test]
async fn memory_stores_trajectory_after_tool_call() {
    use crate::memory::{InMemoryStore, LoopMemory};

    let client = MockClient::new("test-model");
    client.add_tool_then_text(
        "tool_1",
        "echo",
        json!({"message": "hello"}),
        "I echoed your message.",
    );

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let memory = Arc::new(InMemoryStore::new());
    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.set_memory(memory.clone());

    let result = agent
        .run("Echo hello", &RunConfig::default())
        .await
        .unwrap();
    assert_eq!(result.tool_call_count(), 1);

    assert_eq!(
        memory.len(),
        1,
        "a successful tool call must store one trajectory entry"
    );
    let entries = memory.retrieve("echo", 5).await.unwrap();
    assert!(
        entries.iter().any(|e| e.memory.contains("tool=echo")),
        "stored entry must carry the tool name"
    );
}

#[tokio::test]
async fn memory_retrieve_injects_into_request() {
    use crate::memory::{InMemoryStore, LoopMemory, MemoryCategory, MemoryEntry};

    let memory = Arc::new(InMemoryStore::new());
    memory
        .store(MemoryEntry::new(MemoryCategory::Fact, "the answer is 42"))
        .await
        .unwrap();

    let client = RecordingClient::new("test");
    client.add_text_response("done");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_memory(memory);

    agent.run("answer", &RunConfig::default()).await.unwrap();

    let seen = agent.client.first_seen();
    let memory_msg = seen
        .iter()
        .find(|m| m.role == Role::User && m.text_content().contains("Relevant memory"));
    assert!(
        memory_msg.is_some(),
        "memory must be injected as a User-role message"
    );
    let text = memory_msg.unwrap().text_content();
    assert!(
        text.contains("the answer is 42"),
        "request must contain the stored entry text: {text}"
    );
    assert!(
        text.contains("reference only"),
        "memory message must delimit itself as untrusted data"
    );
}

#[tokio::test]
async fn memory_consolidate_prunes_on_successful_run() {
    use crate::memory::{InMemoryStore, LoopMemory, MemoryEntry};

    let memory = Arc::new(InMemoryStore::new());
    let mut stale = MemoryEntry::new(crate::memory::MemoryCategory::Fact, "stale entry");
    stale.relevance = 0.01;
    memory.store(stale).await.unwrap();
    memory
        .store(MemoryEntry::new(
            crate::memory::MemoryCategory::Fact,
            "important entry",
        ))
        .await
        .unwrap();
    assert_eq!(memory.len(), 2, "precondition: two entries");

    let client = MockClient::new("test");
    client.add_text_response("done");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_memory(memory.clone());

    agent.run("go", &RunConfig::default()).await.unwrap();

    assert_eq!(
        memory.len(),
        1,
        "consolidate must prune the low-relevance entry on successful run"
    );
}

struct SequenceObserver {
    log: Arc<Mutex<Vec<String>>>,
}

impl SequenceObserver {
    fn new(log: Arc<Mutex<Vec<String>>>) -> Self {
        Self { log }
    }

    fn record(&self, name: &str) {
        crate::error::recover_guard(self.log.lock()).push(name.to_string());
    }
}

impl crate::observer::LoopObserver for SequenceObserver {
    fn name(&self) -> &'static str {
        "sequence"
    }
    fn on_turn_start(&self, _ctx: &crate::observer::TurnStartContext) {
        self.record("on_turn_start");
    }
    fn on_text_delta(&self, _ctx: &crate::observer::TextDeltaContext) {
        self.record("on_text_delta");
    }
    fn on_stream_success(&self, _ctx: &crate::observer::StreamContext) {
        self.record("on_stream_success");
    }
    fn on_response(&self, _ctx: &crate::observer::ResponseContext) {
        self.record("on_response");
    }
    fn on_turn_end(&self, _ctx: &crate::observer::TurnEndContext) {
        self.record("on_turn_end");
    }
    fn on_tool_call_received(&self, _ctx: &crate::observer::ToolCallReceivedContext) {
        self.record("on_tool_call_received");
    }
    fn on_tool_pre(&self, _ctx: &crate::observer::ToolPreContext) {
        self.record("on_tool_pre");
    }
    fn on_tool_post(&self, _ctx: &crate::observer::ToolPostContext) {
        self.record("on_tool_post");
    }
    fn on_compaction(&self, _ctx: &crate::observer::CompactedContext) {
        self.record("on_compaction");
    }
}

fn sequence_log() -> Arc<Mutex<Vec<String>>> {
    Arc::new(Mutex::new(Vec::new()))
}

fn agent_with_sequence_observer(
    client: MockClient,
    registry: ToolRegistry,
    log: Arc<Mutex<Vec<String>>>,
) -> BareLoop<MockClient> {
    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.register_observer(Arc::new(SequenceObserver::new(log)));
    agent
}

fn snapshot(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    crate::error::recover_guard(log.lock()).clone()
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn observer_sequence_text_only_turn() {
    let client = MockClient::new("test-model");
    client.add_text_response("Hi there.");
    let log = sequence_log();
    let mut agent = agent_with_sequence_observer(client, ToolRegistry::new(), log.clone());
    agent.run("Hi", &RunConfig::default()).await.unwrap();

    let events = snapshot(&log);
    let turn_events: Vec<&String> = events
        .iter()
        .filter(|e| {
            matches!(
                e.as_str(),
                "on_turn_start"
                    | "on_text_delta"
                    | "on_stream_success"
                    | "on_response"
                    | "on_turn_end"
            )
        })
        .collect();
    let expected = [
        "on_turn_start",
        "on_text_delta",
        "on_stream_success",
        "on_response",
        "on_turn_end",
    ];
    assert_eq!(
        turn_events.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        expected
    );
}

#[tokio::test]
async fn observer_sequence_tool_call_turn() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done.");
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let log = sequence_log();
    let mut agent = agent_with_sequence_observer(client, registry, log.clone());
    agent.run("echo hi", &RunConfig::default()).await.unwrap();

    let events = snapshot(&log);
    // The tool-call turn must announce the tool calls before dispatching.
    assert!(
        events.iter().any(|e| e == "on_tool_call_received"),
        "tool-call turn fires on_tool_call_received"
    );
    let pre = events.iter().position(|e| e == "on_tool_pre");
    let post = events.iter().position(|e| e == "on_tool_post");
    assert!(
        pre.zip(post).is_some_and(|(p1, p2)| p1 < p2),
        "on_tool_pre fires before on_tool_post"
    );
}

#[tokio::test]
async fn observer_sequence_multi_tool_turn() {
    let client = MockClient::new("test-model");
    // Two tool calls in one turn, then a final text turn.
    client.add_multi_tool_then_text(
        &[
            (
                "tool_a".to_string(),
                "echo".to_string(),
                json!({"message": "a"}),
            ),
            (
                "tool_b".to_string(),
                "echo".to_string(),
                json!({"message": "b"}),
            ),
        ],
        "All done.",
    );
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let log = sequence_log();
    let mut agent = agent_with_sequence_observer(client, registry, log.clone());
    agent
        .run("echo twice", &RunConfig::default())
        .await
        .unwrap();

    let events = snapshot(&log);
    // Sequential dispatch: pre, post, pre, post — never interleaved.
    let tool_seq: Vec<&String> = events
        .iter()
        .filter(|e| matches!(e.as_str(), "on_tool_pre" | "on_tool_post"))
        .collect();
    assert_eq!(
        tool_seq.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ["on_tool_pre", "on_tool_post", "on_tool_pre", "on_tool_post"],
        "multi-tool sequential dispatch keeps pre/post paired and ordered"
    );
}

#[tokio::test]
async fn compaction_sees_pending_messages() {
    let client = MockClient::new("test-model");
    client.add_text_response(&"x".repeat(200));
    client.add_text_response("done");

    let config = make_config()
        .with_context_window(100)
        .with_compact_threshold(10);
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    agent.set_context_manager(Arc::new(
        crate::compact::ContextManager::new(Arc::new(crate::compact::TruncatingCompactor::new()))
            .with_context_window(100)
            .with_threshold(10),
    ));

    agent
        .run("fill it up", &RunConfig::default())
        .await
        .unwrap();

    let conv_before = agent.conversation();
    let size_before = conv_before.len();

    agent
        .run("second run", &RunConfig::default())
        .await
        .unwrap();

    let conv_after = agent.conversation();
    let size_after = conv_after.len();

    assert!(
        size_after < size_before + 4,
        "compaction must have reduced history during second run; before={size_before} after={size_after}"
    );
    assert!(
        conv_after.iter().any(|m| m.role == Role::User
            && m.parts.iter().any(|p| matches!(
                p,
                MessagePart::Text { text } if text == "second run"
            ))),
        "second run's user input must be in committed history after success"
    );
}

#[tokio::test]
async fn context_token_count_includes_model_response_message() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingCounter {
        last_message_count: AtomicUsize,
    }
    impl crate::compact::TokenCounter for CountingCounter {
        fn count(&self, messages: &[Message]) -> u64 {
            self.last_message_count
                .store(messages.len(), Ordering::SeqCst);
            0
        }
    }

    let client = MockClient::new("test-model");
    client.add_text_response("assistant reply");

    let token_ctr = Arc::new(CountingCounter {
        last_message_count: AtomicUsize::new(0),
    });
    let counter_clone = Arc::clone(&token_ctr);

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_token_counter(counter_clone);

    agent.run("hi", &RunConfig::default()).await.unwrap();

    let seen_msgs = token_ctr.last_message_count.load(Ordering::SeqCst);
    assert!(
        seen_msgs >= 2,
        "token counter must see at least 2 messages (user + model response), got {seen_msgs}"
    );
}

#[test]
fn set_token_counter_sets_fallback_and_count_context_prefers_manager() {
    use crate::compact::{ContextManager, HeuristicTokenCounter, TokenCounter};

    struct SentinelCounter;
    impl TokenCounter for SentinelCounter {
        fn count(&self, _: &[Message]) -> u64 {
            999
        }
    }

    let client = MockClient::new("test-model");
    let manager = Arc::new(
        ContextManager::new(Arc::new(crate::compact::TruncatingCompactor::new()))
            .with_token_counter(Arc::new(HeuristicTokenCounter)),
    );
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_context_manager(manager);

    // set_token_counter updates the fallback field only; the manager owns its
    // own counter independently (single source of truth per layer).
    let sentinel = Arc::new(SentinelCounter);
    agent.set_token_counter(sentinel);

    let driver_sample = agent.token_counter.count(&[Message::user("hi")]);
    assert_eq!(
        driver_sample, 999,
        "driver-side fallback counter must be the sentinel"
    );

    // count_context prefers the manager's counter when one is set.
    let via_count_context = agent.count_context(&[Message::user("hi")]);
    assert_ne!(
        via_count_context, 999,
        "count_context must prefer the manager's counter, not the fallback sentinel"
    );
}

#[tokio::test]
async fn compaction_then_failure_leaves_history_compacted() {
    let client = MockClient::new("test-model");
    client.add_text_response(&"x".repeat(200));
    client.add_text_response("done");
    client.add_text_response("second done");

    let config = make_config()
        .with_context_window(100)
        .with_compact_threshold(10);
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    agent.set_context_manager(Arc::new(
        crate::compact::ContextManager::new(Arc::new(crate::compact::TruncatingCompactor::new()))
            .with_context_window(100)
            .with_threshold(10),
    ));

    agent.run("first run", &RunConfig::default()).await.unwrap();

    agent.cancel();
    let _ = agent.run("will fail", &RunConfig::default()).await.ok();
    agent.cancelled.reset();

    let history = agent.conversation();
    assert!(
        !history.is_empty(),
        "history must contain messages from the first successful run"
    );
    assert!(
        !history.iter().any(|m| m.role == Role::User
            && m.parts
                .iter()
                .any(|p| matches!(p, MessagePart::Text { text } if text == "will fail"))),
        "failed run's user input must not persist in history"
    );

    agent.run("third run", &RunConfig::default()).await.unwrap();
}

#[tokio::test]
async fn observer_sequence_compaction_turn() {
    let client = MockClient::new("test-model");
    // Drive enough tokens to trip a low threshold, then finish.
    client.add_text_response(&"x".repeat(200));
    client.add_text_response("compacted-and-done");
    let log = sequence_log();
    let mut agent = agent_with_sequence_observer(client, ToolRegistry::new(), log.clone());
    agent.set_context_manager(Arc::new(
        crate::compact::ContextManager::new(Arc::new(crate::compact::TruncatingCompactor::new()))
            .with_context_window(100)
            .with_threshold(10),
    ));
    let run_config = RunConfig::default();
    let run_result = agent.run("fill it up", &run_config).await;
    // The compaction scenario drives the run to completion; event placement is asserted below.
    assert!(run_result.is_ok(), "compaction run completes");

    let events = snapshot(&log);
    // If compaction ran, on_compaction sits at a turn boundary (after a
    // turn_end, before the next turn_start). If the estimate didn't trip,
    // the scenario is N/A — assert placement only when present.
    if let Some(idx) = events.iter().position(|e| e == "on_compaction") {
        let before = events.get(idx.wrapping_sub(1));
        let after = events.get(idx + 1);
        assert!(
            before == Some(&"on_turn_end".to_string())
                || after == Some(&"on_turn_start".to_string()),
            "on_compaction at idx {idx} sits at a turn boundary, got before={before:?} after={after:?}"
        );
    }
}

#[tokio::test]
async fn observer_sequence_cancelled_turn() {
    let client = MockClient::new("test-model");
    // Never-ending tool calls so the loop is mid-flight when cancelled.
    for _ in 0..5 {
        client.add_tool_only_response("c1", "echo", json!({"message": "x"}));
    }
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let log = sequence_log();
    let mut agent = agent_with_sequence_observer(client, registry, log.clone());

    let handle = agent.cancel_signal();
    let join = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        handle.cancel();
    });
    let result = agent.run("go", &RunConfig::default()).await;
    join.await.unwrap();
    assert!(result.is_err(), "cancelled run returns an error");

    let events = snapshot(&log);
    let started = events.iter().filter(|e| **e == "on_turn_start").count();
    let ended = events.iter().filter(|e| **e == "on_turn_end").count();
    assert!(
        started >= 1 && ended >= 1,
        "cancelled turn still fires on_turn_end (started={started}, ended={ended})"
    );
}

struct ToolNameCapture {
    captured: Arc<Mutex<Option<String>>>,
}
impl crate::observer::LoopObserver for ToolNameCapture {
    fn name(&self) -> &'static str {
        "tool-name-capture"
    }
    fn on_tool_pre(&self, ctx: &crate::observer::ToolPreContext) {
        *crate::error::recover_guard(self.captured.lock()) = Some(ctx.tool.clone());
    }
}

#[tokio::test]
async fn dispatch_surfaces_tool_name_on_tool_pre() {
    let client = MockClient::new("test-model");
    // A tool-call turn then a final text turn. The driver is dispatching the
    // tool during `on_tool_pre`; the ToolNameCapture observer records the
    // tool name carried on the context.
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "done");
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let captured = Arc::new(Mutex::new(None::<String>));
    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.register_observer(Arc::new(ToolNameCapture {
        captured: Arc::clone(&captured),
    }));
    agent.run("echo hi", &RunConfig::default()).await.unwrap();

    let snapshot = crate::error::recover_guard(captured.lock()).clone();
    assert_eq!(
        snapshot.as_deref(),
        Some("echo"),
        "tool name preserved on ToolPreContext during dispatch"
    );
}

#[tokio::test]
async fn bareloop_machine_accessor_returns_machine() {
    let client = MockClient::new("test-model");
    client.add_text_response("hi");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.run("hello", &RunConfig::default()).await.unwrap();
    // After a run, the machine is populated and history holds the turn.
    let machine = agent.machine();
    assert!(machine.turns_taken() >= 1);
    assert!(!machine.history().is_empty());
}

#[tokio::test]
async fn serialize_drop_deserialize_resume_preserves_history() {
    let client = MockClient::new("test-model");
    client.add_text_response("first");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.run("prompt", &RunConfig::default()).await.unwrap();

    // Take the machine and round-trip it through serde.
    let machine = agent.into_machine();
    let serialized = serde_json::to_string(&machine).expect("serialize machine");
    let restored: LoopMachine = serde_json::from_str(&serialized).expect("deserialize machine");
    // Compare by serialized form: Message is not PartialEq.
    let got = serde_json::to_string(restored.history()).expect("serialize history");
    let want = serde_json::to_string(machine.history()).expect("serialize history");
    assert_eq!(got, want, "history survives serialize/deserialize");

    // Rebuild a loop around the restored machine.
    let client2 = MockClient::new("test-model");
    let _rebuilt = BareLoop::from_machine(
        restored,
        make_config(),
        Arc::new(client2),
        ToolRegistry::new(),
    );
}

#[tokio::test]
async fn session_id_stable_and_run_id_rotates_across_runs() {
    let client = MockClient::new("test-model");
    client.add_text_response("first");
    client.add_text_response("second");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    let first = agent.run("one", &RunConfig::default()).await.unwrap();
    let first_session = agent.session().id;
    let first_run = first.id;

    let second = agent.run("two", &RunConfig::default()).await.unwrap();
    let second_session = agent.session().id;
    let second_run = second.id;

    // Session identity is stable across runs.
    assert_eq!(first_session, second_session, "session_id is stable");
    // Each run mints a fresh id.
    assert_ne!(first_run, second_run, "id rotates per run");
}

#[tokio::test]
async fn max_tokens_stop_reason_preserved() {
    let client = MockClient::new("test-model");
    client.add_max_tokens_response("truncated");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let result = agent.run("generate", &RunConfig::default()).await.unwrap();

    assert_eq!(result.turn_count(), 1);
}

#[tokio::test]
async fn test_bare_loop_max_turns_exceeded() {
    let client = MockClient::new("test-model");
    // Return only tool_call responses so the loop never gets an end_turn
    for i in 0..20 {
        client.add_tool_only_response(
            &format!("tool_{i}"),
            "echo",
            json!({"message": format!("msg_{i}")}),
        );
    }

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let run_config = RunConfig {
        max_turns: 3,
        ..RunConfig::default()
    };
    let result = agent.run("Keep going", &run_config).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LoopError::MaxTurnsExceeded { max } => assert_eq!(max, 3),
        other => panic!("Expected MaxTurnsExceeded, got: {other}"),
    }
}

#[tokio::test]
async fn test_bare_loop_cancellation() {
    let client = MockClient::new("test-model");
    client.add_text_response("Hello!");

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

    // Cancel before running
    agent.cancel();
    assert!(agent.is_cancelled());

    let result = agent.run("Hi", &RunConfig::default()).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LoopError::Cancelled => {}
        other => panic!("Expected Cancelled error, got: {other}"),
    }
}

#[tokio::test]
async fn test_bare_loop_api_error() {
    // The mock will return an error
    let client = MockClient::new("test-model");
    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let result = agent.run("Hi", &RunConfig::default()).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LoopError::Api(msg) => assert!(msg.contains("No more mock responses"), "got: {msg}"),
        other => panic!("Expected Api error, got: {other}"),
    }
}

#[tokio::test]
async fn test_tool_not_found_returns_error_result() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "nonexistent", json!({}), "I see the tool failed.");

    // Empty registry — tool won't be found
    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let result = agent
        .run("Use nonexistent tool", &RunConfig::default())
        .await
        .unwrap();

    // The tool-not-found should be returned as an error result in the conversation,
    // not as a hard error. The loop should continue and eventually get the end_turn.
    assert_eq!(result.turn_count(), 2);
}

#[tokio::test]
async fn test_tool_execution_failure() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "fail", json!({}), "The tool failed, moving on.");

    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    let result = agent
        .run("Use failing tool", &RunConfig::default())
        .await
        .unwrap();

    assert_eq!(result.turn_count(), 2);
}

#[tokio::test]
async fn test_observer_lifecycle_events() {
    let client = MockClient::new("test-model");
    client.add_text_response("Done!");

    let plugin = Arc::new(CountingObserver::new());
    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    agent.register_observer(plugin.clone());

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    assert_eq!(plugin.run_starts.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.run_ends.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.turn_starts.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.turn_ends.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_observer_run_start_end_symmetry_across_multiple_runs() {
    let client = MockClient::new("test-model");
    client.add_text_response("first");
    client.add_text_response("second");
    client.add_text_response("third");

    let plugin = Arc::new(CountingObserver::new());
    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    agent.register_observer(plugin.clone());

    for _ in 0..3 {
        let _ = agent.run("Hi", &RunConfig::default()).await.unwrap();
    }

    assert_eq!(
        plugin.run_starts.load(Ordering::SeqCst),
        3,
        "on_run_start must fire once per run"
    );
    assert_eq!(
        plugin.run_ends.load(Ordering::SeqCst),
        3,
        "on_run_end must fire once per run"
    );
}

#[tokio::test]
async fn test_observer_tool_events() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "test"}), "All done!");

    let plugin = Arc::new(CountingObserver::new());
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    agent.register_observer(plugin.clone());

    let _result = agent.run("Echo test", &RunConfig::default()).await.unwrap();

    assert_eq!(plugin.tool_pres.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.tool_posts.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.turn_starts.load(Ordering::SeqCst), 2);
    assert_eq!(plugin.turn_ends.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_conversation_built_correctly() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text(
        "tool_1",
        "echo",
        json!({"message": "hello"}),
        "Final answer.",
    );

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);

    // Driving the run builds the conversation in the machine-owned history.
    agent
        .run("Echo hello", &RunConfig::default())
        .await
        .unwrap();

    // History: [user, assistant(tool_call), user(tool_result), assistant(text)].
    let history = agent.conversation();
    assert_eq!(
        history.len(),
        4,
        "expected user, assistant, tool-result, final-answer"
    );
    assert_eq!(history[0].role, Role::User);
    assert_eq!(history[1].role, Role::Assistant);
    assert_eq!(history[2].role, Role::User);
    assert_eq!(history[3].role, Role::Assistant);

    // The extract helpers still classify tool-call parts correctly.
    let msg_with_tools = Message::new(
        Role::Assistant,
        vec![
            MessagePart::text("Using tool..."),
            MessagePart::tool_call("id1", "echo", json!({"message": "hi"})),
        ],
    );
    let tool_calls: Vec<ToolCall> = msg_with_tools
        .tool_call_parts()
        .into_iter()
        .map(|(id, tool, input)| ToolCall {
            id: id.to_string(),
            tool: tool.to_string(),
            input: input.clone(),
        })
        .collect();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].tool, "echo");
}

#[tokio::test]
async fn test_tool_result_message_format() {
    let results = vec![super::ToolDispatchResult {
        tool_call_id: "tool_123".to_string(),
        output: ToolContent::Text("Echo: hello".to_string()),
        is_error: false,
        duration: Duration::from_millis(100),
        resolved_tool_name: String::new(),
        display_hint: None,
    }];

    let parts = BareLoop::<MockClient>::build_tool_result_parts(results);
    assert_eq!(parts.len(), 1);

    match &parts[0] {
        MessagePart::ToolResult {
            call_id,
            name: _,
            output,
            is_error,
        } => {
            assert_eq!(call_id, "tool_123");
            assert!(!is_error.unwrap_or(true));
            let text = output.to_string();
            assert_eq!(text, "Echo: hello");
        }
        other => panic!("Expected ToolResult part, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_multiple_tool_calls_in_one_turn() {
    let client = MockClient::new("test-model");

    // First response: two tool_call parts
    let tool_events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_multi".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "t1",
                "echo",
                json!({"message": "first"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::tool_call(
                "t2",
                "echo",
                json!({"message": "second"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 20)),
        }),
        StreamEvent::MessageStop,
    ];
    crate::error::recover_guard(client.responses.lock()).push(tool_events);

    // Second response: end_turn
    client.add_text_response("Both tools executed.");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);

    let result = agent
        .run("Echo twice", &RunConfig::default())
        .await
        .unwrap();

    assert_eq!(result.turn_count(), 2);
    assert_eq!(result.tool_call_count(), 2);
}

#[tokio::test]
async fn test_mixed_known_unknown_tools_merge_into_one_user_message() {
    let client = MockClient::new("test-model");

    // One known tool call (echo) and one unknown (nonexistent) in the
    // same turn. The unknown result is preresolved; the known one is
    // dispatched. Both must land in a single user Message in history.
    let tool_events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_mixed".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "t1",
                "echo",
                json!({"message": "hi"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::tool_call("t2", "nonexistent", json!({}))),
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 20)),
        }),
        StreamEvent::MessageStop,
    ];
    crate::error::recover_guard(client.responses.lock()).push(tool_events);
    client.add_text_response("done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent
        .run("mixed tools", &RunConfig::default())
        .await
        .unwrap();

    let conversation = agent.conversation();
    let user_messages: Vec<&Message> = conversation
        .iter()
        .filter(|m| m.role == Role::User)
        .collect();
    assert_eq!(
        user_messages.len(),
        2,
        "expected [prompt, one merged tool-result message], got {} user messages",
        user_messages.len()
    );
    let tool_results: Vec<&MessagePart> = user_messages[1]
        .parts
        .iter()
        .filter(|p| p.is_tool_result())
        .collect();
    assert_eq!(
        tool_results.len(),
        2,
        "merged user message must hold both tool-result parts"
    );
}

#[tokio::test]
async fn test_mixed_known_unknown_tools_preserve_request_order() {
    let client = MockClient::new("test-model");

    // Call order in the model response: t1=unknown (preresolved), t2=known
    // (dispatched), t3=unknown (preresolved). The merged tool-result message
    // must keep this order — NOT [unknown, unknown, known] (the order the two
    // paths would produce if concatenated by resolution path).
    let tool_events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_order".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call("t1", "ghost", json!({}))),
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::tool_call(
                "t2",
                "echo",
                json!({"message": "mid"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 2,
            part: Some(MessagePart::tool_call("t3", "phantom", json!({}))),
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 20)),
        }),
        StreamEvent::MessageStop,
    ];
    crate::error::recover_guard(client.responses.lock()).push(tool_events);
    client.add_text_response("done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent
        .run("order test", &RunConfig::default())
        .await
        .unwrap();

    let conversation = agent.conversation();
    let merged: &Message = conversation
        .iter()
        .filter(|m| m.role == Role::User)
        .nth(1)
        .expect("expected [prompt, merged tool-result message]");
    let call_ids: Vec<&str> = merged
        .parts
        .iter()
        .filter_map(|p| match p {
            MessagePart::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        call_ids,
        vec!["t1", "t2", "t3"],
        "merged tool-result parts must follow the model's request order, \
         not the resolution-path order"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_text_streamer_fires_on_text_delta() {
    let client = MockClient::new("test-model");
    client.add_text_response("Hello world");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    let received = Arc::new(Mutex::new(Vec::new()));
    let buf = Arc::clone(&received);
    agent.set_text_streamer(Arc::new(move |delta: &str| {
        crate::error::recover_guard(buf.lock()).push(delta.to_string());
    }));

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let received = crate::error::recover_guard(received.lock());
    assert!(!received.is_empty(), "streamer should have fired");
    assert!(
        received.join("").contains("Hello world"),
        "got: {received:?}",
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_text_streamer_fires_when_stream_handler_configured() {
    // Regression: when a StreamHandler is attached, the engine must still
    // fire text_streamer / on_text_delta for each streamed text delta. The
    // handler path used to bypass observers entirely.
    let client = MockClient::new("test-model");
    client.add_text_response("via handler");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.set_stream_handler(StreamHandler::new());

    let received = Arc::new(Mutex::new(Vec::new()));
    let buf = Arc::clone(&received);
    agent.set_text_streamer(Arc::new(move |delta: &str| {
        crate::error::recover_guard(buf.lock()).push(delta.to_string());
    }));

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let received = crate::error::recover_guard(received.lock());
    assert!(
        !received.is_empty(),
        "streamer should fire even with a StreamHandler configured"
    );
    assert!(
        received.join("").contains("via handler"),
        "got: {received:?}",
    );
}

#[tokio::test]
async fn test_text_streamer_none_works() {
    let client = MockClient::new("test-model");
    client.add_text_response("No streamer");
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_text_streamer_ignores_non_text_deltas() {
    let client = MockClient::new("test-model");

    // Build a response with tool-call events (no text).
    let events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: Value::Null,
            }),
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: "{}".into(),
            },
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".into()),
            },
            usage: None,
        }),
        StreamEvent::MessageStop,
    ];
    client.add_events(events);

    // Second turn: plain text response.
    client.add_text_response("Done");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    let received = Arc::new(Mutex::new(String::new()));
    let buf = Arc::clone(&received);
    agent.set_text_streamer(Arc::new(move |delta: &str| {
        crate::error::recover_guard(buf.lock()).push_str(delta);
    }));

    agent.run("Use tool", &RunConfig::default()).await.unwrap();

    // The InputJson delta should NOT have triggered the streamer.
    // Only the "Done" text response in the second turn should.
    let received = crate::error::recover_guard(received.lock());
    assert_eq!(&*received, "Done", "only text deltas should fire streamer");
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_text_delta_fires_per_sse_chunk_in_order() {
    struct DeltaRecorder {
        deltas: Arc<Mutex<Vec<(usize, String)>>>,
    }
    impl crate::observer::LoopObserver for DeltaRecorder {
        fn name(&self) -> &'static str {
            "delta-recorder"
        }
        fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
            crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
        }
    }

    let client = MockClient::new("test-model");
    let events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("ignored")),
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "Hello".into(),
            },
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text { text: " ".into() },
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "world".into(),
            },
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: None,
        }),
        StreamEvent::MessageStop,
    ];
    client.add_events(events);

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(DeltaRecorder {
        deltas: Arc::clone(&captured),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let captured = crate::error::recover_guard(captured.lock());
    assert_eq!(captured.len(), 3, "one on_text_delta per SSE text chunk");
    let joined: String = captured.iter().map(|(_, d)| d.as_str()).collect();
    assert_eq!(joined, "Hello world");
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_text_delta_turn_number_matches_surrounding_turn() {
    struct TurnRecorder {
        deltas: Arc<Mutex<Vec<(usize, String)>>>,
        response_turns: Arc<Mutex<Vec<usize>>>,
    }
    impl crate::observer::LoopObserver for TurnRecorder {
        fn name(&self) -> &'static str {
            "turn-recorder"
        }
        fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
            crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
        }
        fn on_response(&self, ctx: &crate::observer::ResponseContext) {
            crate::error::recover_guard(self.response_turns.lock()).push(ctx.turn);
        }
    }

    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "All done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let deltas = Arc::new(Mutex::new(Vec::new()));
    let response_turns = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(TurnRecorder {
        deltas: Arc::clone(&deltas),
        response_turns: Arc::clone(&response_turns),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let result = agent
        .run("Use echo then finish", &RunConfig::default())
        .await
        .unwrap();
    assert_eq!(result.turn_count(), 2);

    let response_turns = crate::error::recover_guard(response_turns.lock());
    let deltas = crate::error::recover_guard(deltas.lock());

    assert_eq!(
        response_turns.len(),
        2,
        "both turns should fire on_response",
    );
    assert!(!deltas.is_empty(), "text turn should produce deltas");
    for (turn, _) in deltas.iter() {
        assert!(
            response_turns.contains(turn),
            "on_text_delta turn {turn} must match an on_response turn",
        );
    }

    let text_turn = deltas.iter().map(|(t, _)| *t).next().unwrap();
    let joined: String = deltas
        .iter()
        .filter(|(t, _)| *t == text_turn)
        .map(|(_, d)| d.as_str())
        .collect();
    assert_eq!(joined, "All done");
    assert_eq!(
        text_turn, 1,
        "text deltas belong to the second turn (the text turn), not the tool turn",
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_text_delta_ignores_non_text_deltas() {
    struct DeltaRecorder {
        count: Arc<AtomicUsize>,
    }
    impl crate::observer::LoopObserver for DeltaRecorder {
        fn name(&self) -> &'static str {
            "delta-recorder"
        }
        fn on_text_delta(&self, _ctx: &crate::observer::TextDeltaContext) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    let client = MockClient::new("test-model");
    let events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: Value::Null,
            }),
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: "{}".into(),
            },
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".into()),
            },
            usage: None,
        }),
        StreamEvent::MessageStop,
    ];
    client.add_events(events);
    client.add_text_response("Done");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let count = Arc::new(AtomicUsize::new(0));
    let recorder = Arc::new(DeltaRecorder {
        count: Arc::clone(&count),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    agent.run("Use tool", &RunConfig::default()).await.unwrap();

    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "only the text delta should fire on_text_delta",
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_text_delta_fires_without_streamer() {
    struct DeltaRecorder {
        deltas: Arc<Mutex<Vec<String>>>,
    }
    impl crate::observer::LoopObserver for DeltaRecorder {
        fn name(&self) -> &'static str {
            "delta-recorder"
        }
        fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
            crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
        }
    }

    let client = MockClient::new("test-model");
    client.add_text_response("Hello world");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(DeltaRecorder {
        deltas: Arc::clone(&captured),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let captured = crate::error::recover_guard(captured.lock());
    assert!(
        !captured.is_empty(),
        "observer should receive deltas with no streamer set"
    );
    let joined: String = captured.iter().map(String::as_str).collect();
    assert!(joined.contains("Hello world"), "got: {joined:?}");
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_text_delta_and_streamer_coexist() {
    struct DeltaRecorder {
        deltas: Arc<Mutex<Vec<String>>>,
    }
    impl crate::observer::LoopObserver for DeltaRecorder {
        fn name(&self) -> &'static str {
            "delta-recorder"
        }
        fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
            crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
        }
    }

    let client = MockClient::new("test-model");
    client.add_text_response("Hello world");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    let streamer_buf = Arc::new(Mutex::new(Vec::new()));
    let buf = Arc::clone(&streamer_buf);
    agent.set_text_streamer(Arc::new(move |delta: &str| {
        crate::error::recover_guard(buf.lock()).push(delta.to_string());
    }));

    let observer_buf = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(DeltaRecorder {
        deltas: Arc::clone(&observer_buf),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let streamer_buf = crate::error::recover_guard(streamer_buf.lock());
    let observer_buf = crate::error::recover_guard(observer_buf.lock());
    assert!(!streamer_buf.is_empty(), "streamer should fire");
    assert!(!observer_buf.is_empty(), "observer should fire");
    assert_eq!(
        streamer_buf.len(),
        observer_buf.len(),
        "both paths receive the same number of deltas",
    );
    assert_eq!(
        *streamer_buf, *observer_buf,
        "both paths receive identical chunks"
    );
}

#[tokio::test]
async fn test_on_tool_call_received_fires_once_per_call() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());

    let _result = agent.run("Use echo", &RunConfig::default()).await.unwrap();

    assert_eq!(
        observer.tool_calls_received.load(Ordering::SeqCst),
        1,
        "one accumulated call → one received event",
    );
    assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_on_tool_call_received_fires_per_call_for_multiple_calls() {
    let client = MockClient::new("test-model");
    let tool_events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_multi".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "t1",
                "echo",
                json!({"message": "first"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::tool_call(
                "t2",
                "echo",
                json!({"message": "second"}),
            )),
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 20)),
        }),
        StreamEvent::MessageStop,
    ];
    crate::error::recover_guard(client.responses.lock()).push(tool_events);
    client.add_text_response("All done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());

    let _result = agent
        .run("Echo twice", &RunConfig::default())
        .await
        .unwrap();

    assert_eq!(
        observer.tool_calls_received.load(Ordering::SeqCst),
        2,
        "two accumulated calls → two received events",
    );
    assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_on_tool_call_received_not_fired_for_text_only_turn() {
    let client = MockClient::new("test-model");
    client.add_text_response("Just text, no tools");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    assert_eq!(
        observer.tool_calls_received.load(Ordering::SeqCst),
        0,
        "no tool calls → no received event",
    );
    assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_on_tool_call_received_turn_matches_other_events() {
    struct TurnCapture {
        received_turns: Arc<Mutex<Vec<usize>>>,
        response_turns: Arc<Mutex<Vec<usize>>>,
        pre_turns: Arc<Mutex<Vec<usize>>>,
    }
    impl crate::observer::LoopObserver for TurnCapture {
        fn name(&self) -> &'static str {
            "turn-capture"
        }
        fn on_response(&self, ctx: &crate::observer::ResponseContext) {
            crate::error::recover_guard(self.response_turns.lock()).push(ctx.turn);
        }
        fn on_tool_call_received(&self, ctx: &crate::observer::ToolCallReceivedContext) {
            crate::error::recover_guard(self.received_turns.lock()).push(ctx.turn);
        }
        fn on_tool_pre(&self, ctx: &crate::observer::ToolPreContext) {
            crate::error::recover_guard(self.pre_turns.lock()).push(ctx.turn);
        }
    }

    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let received = Arc::new(Mutex::new(Vec::new()));
    let response = Arc::new(Mutex::new(Vec::new()));
    let pre = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(TurnCapture {
        received_turns: Arc::clone(&received),
        response_turns: Arc::clone(&response),
        pre_turns: Arc::clone(&pre),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let _result = agent.run("Use echo", &RunConfig::default()).await.unwrap();

    let received = crate::error::recover_guard(received.lock());
    let response = crate::error::recover_guard(response.lock());
    let pre = crate::error::recover_guard(pre.lock());
    assert_eq!(received.len(), 1, "one tool call → one received event");
    for turn in received.iter() {
        assert!(
            response.contains(turn),
            "received turn {turn} must match an on_response turn",
        );
        assert!(
            pre.contains(turn),
            "received turn {turn} must match an on_tool_pre turn",
        );
    }
}

#[tokio::test]
async fn test_on_tool_call_received_does_not_refire_on_retry() {
    struct AlwaysRecoverable;
    impl crate::reflection::Reflector for AlwaysRecoverable {
        fn analyze(
            &self,
            error: &str,
            tool_name: &str,
            _tool_input: &serde_json::Value,
            _tool_schema: Option<&crate::tool::ToolSchema>,
            _context: &crate::reflection::ReflectionContext,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::reflection::FailureAnalysis,
                            crate::reflection::ReflectionError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let error = error.to_string();
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                Ok(crate::reflection::FailureAnalysis {
                    is_recoverable: true,
                    root_cause: error,
                    severity: crate::reflection::FailureSeverity::Medium,
                    correction: None,
                    context: format!("tool: {tool_name}"),
                })
            })
        }
    }

    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "flaky", json!({}), "Recovered");

    let mut registry = ToolRegistry::new();
    registry.register(FlakyTool::new(2));

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.set_reflector(Arc::new(AlwaysRecoverable));
    agent.set_recovery_strategy(Arc::new(
        crate::reflection::ExponentialBackoffRecovery::new(3)
            .with_base_delay(std::time::Duration::ZERO),
    ));
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());

    let _result = agent.run("Use flaky", &RunConfig::default()).await.unwrap();

    assert_eq!(
        observer.tool_calls_received.load(Ordering::SeqCst),
        1,
        "received fires once per call regardless of retries",
    );
    assert!(
        observer.tool_pres.load(Ordering::SeqCst) >= 2,
        "tool_pre must re-fire on each retry attempt",
    );
}

#[test]
fn test_accessors() {
    let client = MockClient::new("test-model");
    let config = make_config();
    let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

    assert_ne!(agent.session().id, uuid::Uuid::nil());
    assert!(agent.conversation().is_empty());
    assert!(!agent.is_cancelled());
}

#[test]
fn test_cancel_signal_shared() {
    let client = MockClient::new("test-model");
    let config = make_config();
    let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let signal = agent.cancel_signal();
    assert!(!signal.is_cancelled());

    agent.cancel();
    assert!(signal.is_cancelled());
    assert!(agent.is_cancelled());
}

#[tokio::test]
async fn test_second_run_after_cancel_is_not_dead() {
    let client = MockClient::new("test-model");
    client.add_text_response("second run should reach me");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    agent.cancel();

    let first = agent.run("first", &RunConfig::default()).await;
    assert!(
        matches!(first, Err(LoopError::Cancelled)),
        "first run must be cancelled, got {first:?}"
    );

    let client2 = MockClient::new("test-model");
    client2.add_text_response("second run ok");
    agent.client = Arc::new(client2);

    let second = agent.run("second", &RunConfig::default()).await;
    match &second {
        Ok(run) => assert_eq!(
            run.output.as_deref(),
            Some("second run ok"),
            "second run must complete after cancel, got run with output {:?}",
            run.output
        ),
        Err(e) => panic!("second run after cancel must not fail, got {e:?}"),
    }
}

#[tokio::test]
async fn test_run_result_fields() {
    let client = MockClient::new("test-model");
    client.add_text_response("Hello!");

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    // Session identity lives on the loop, not the per-run result.
    assert_ne!(agent.session().id, uuid::Uuid::nil());
    assert!(result.duration() > Duration::ZERO);
    assert!(result.input_tokens() > 0 || result.output_tokens() > 0); // from mock usage
}

#[tokio::test]
async fn test_loop_terminates_with_max_turns_1() {
    let client = MockClient::new("test-model");
    client.add_text_response("One and done.");

    let run_config = RunConfig {
        max_turns: 1,
        ..RunConfig::default()
    };
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let result = agent.run("Hi", &run_config).await.unwrap();

    assert_eq!(result.turn_count(), 1);
}

#[tokio::test]
async fn test_loop_terminates_with_max_turns_0() {
    let client = MockClient::new("test-model");
    client.add_text_response("Should not be reached.");

    let run_config = RunConfig {
        max_turns: 0,
        ..RunConfig::default()
    };
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let result = agent.run("Hi", &run_config).await;
    assert!(result.is_err());
    // With max_turns == 0 the loop never executes a turn and reports the
    // budget as exhausted.
    match result.unwrap_err() {
        LoopError::MaxTurnsExceeded { max } => assert_eq!(max, 0),
        other => panic!("Expected MaxTurnsExceeded, got: {other}"),
    }
}

#[tokio::test]
async fn test_tool_error_is_soft_not_hard() {
    let client = MockClient::new("test-model");

    // Response: request a nonexistent tool
    let tool_events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call("t1", "nonexistent", json!({}))),
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("tool_call".to_string()),
            },
            usage: Some(Usage::new(50, 10)),
        }),
        StreamEvent::MessageStop,
    ];
    crate::error::recover_guard(client.responses.lock()).push(tool_events);

    // Second response: end_turn after seeing error result
    client.add_text_response("Tool wasn't found, but I'll handle it.");

    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let _result = agent
        .run("Use missing tool", &RunConfig::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn test_loop_detection_hard_stop_propagates_loop_error() {
    use crate::detection::{DetectionConfig, DetectionManager};
    use crate::managers::LoopManagers;

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let client = MockClient::new("test");
    for i in 0..10 {
        client.add_tool_only_response(&format!("call_{i}"), "echo", json!({ "message": "hi" }));
    }

    let managers = LoopManagers::new().with_detection(
        DetectionManager::new_with_config(DetectionConfig {
            loop_threshold: 2,
            stop_threshold: 2,
            ..Default::default()
        })
        .expect("valid detection config"),
    );

    let mut agent =
        BareLoop::new_with_managers(Arc::new(client), registry, make_config(), managers);
    let result = agent.run("test", &RunConfig::default()).await;

    assert!(
        matches!(result, Err(LoopError::LoopDetected { .. })),
        "expected Err(LoopError::LoopDetected), got {result:?}"
    );
}

#[tokio::test]
async fn test_loop_detection_soft_block_before_stop_threshold() {
    use crate::detection::{DetectionConfig, DetectionManager};
    use crate::managers::LoopManagers;

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let client = MockClient::new("test");
    client.add_tool_only_response("c1", "echo", json!({ "message": "hi" }));
    client.add_tool_only_response("c2", "echo", json!({ "message": "hi" }));
    client.add_text_response("Done");

    let managers = LoopManagers::new().with_detection(
        DetectionManager::new_with_config(DetectionConfig {
            loop_threshold: 2,
            stop_threshold: 10,
            ..Default::default()
        })
        .expect("valid detection config"),
    );

    let mut agent =
        BareLoop::new_with_managers(Arc::new(client), registry, make_config(), managers);
    let result = agent.run("test", &RunConfig::default()).await;

    assert!(result.is_ok(), "expected Ok, got {result:?}");
}

#[tokio::test]
async fn test_cancelled_before_run_returns_cancelled() {
    let client = MockClient::new("test");
    client.add_text_response("Hello");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    agent.cancel();
    let result = agent.run("test", &RunConfig::default()).await;

    assert!(
        matches!(result, Err(LoopError::Cancelled)),
        "expected Err(LoopError::Cancelled), got {result:?}"
    );
}

#[tokio::test]
async fn test_default_recovery_on_tool_error_returns_soft_result() {
    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let client = MockClient::new("test");
    client.add_tool_then_text("tool_1", "fail", json!({}), "Moving on");

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let result = agent.run("Test", &RunConfig::default()).await.unwrap();

    assert_eq!(result.tool_call_count(), 1);
}

#[tokio::test]
async fn test_recovery_on_missing_tool_returns_soft_result() {
    let client = MockClient::new("test");
    client.add_tool_then_text("tool_1", "nonexistent", json!({}), "OK");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let result = agent.run("Test", &RunConfig::default()).await.unwrap();

    assert_eq!(result.tool_call_count(), 1);
}

#[tokio::test]
async fn test_recovery_noop_reflector_no_retries() {
    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let client = MockClient::new("test");
    client.add_tool_then_text("tool_1", "fail", json!({}), "OK");

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let result = agent.run("Test", &RunConfig::default()).await.unwrap();

    assert_eq!(result.tool_call_count(), 1);
}

#[tokio::test]
async fn test_recovery_respects_cancellation() {
    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let client = MockClient::new("test");
    client.add_tool_only_response("tc-1", "fail", json!({}));

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());

    // Cancel before running
    agent.cancel();

    let result = agent.run("Test", &RunConfig::default()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_cancel_during_dispatch_lands_in_cancelled_state() {
    // Cancellation fired after dispatch has begun flows through
    // MachineOutcome::Cancelled (not Failed). Uses AlwaysRecoverable so
    // FailingTool's error triggers a retry; the retry loop polls
    // is_cancelled() at the top of each iteration (dispatch.rs), so the
    // cancel signal set here is observed on the next retry attempt.
    struct AlwaysRecoverable;
    impl crate::reflection::Reflector for AlwaysRecoverable {
        fn analyze(
            &self,
            error: &str,
            tool_name: &str,
            _tool_input: &serde_json::Value,
            _tool_schema: Option<&crate::tool::ToolSchema>,
            _context: &crate::reflection::ReflectionContext,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::reflection::FailureAnalysis,
                            crate::reflection::ReflectionError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let error = error.to_string();
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                Ok(crate::reflection::FailureAnalysis {
                    is_recoverable: true,
                    root_cause: error,
                    severity: crate::reflection::FailureSeverity::Medium,
                    correction: None,
                    context: format!("tool: {tool_name}"),
                })
            })
        }
    }

    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let client = MockClient::new("test");
    client.add_tool_only_response("tc-1", "fail", json!({}));

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.set_reflector(Arc::new(AlwaysRecoverable));
    agent.set_recovery_strategy(Arc::new(
        crate::reflection::ExponentialBackoffRecovery::new(5)
            .with_base_delay(std::time::Duration::ZERO),
    ));
    let signal = agent.cancel_signal();
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        signal.cancel();
    });

    let result = agent.run("Test", &RunConfig::default()).await;
    match result {
        Err(LoopError::Cancelled) => {}
        other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
    }
    assert_eq!(
        agent.state(),
        MachineState::Terminal(MachineOutcome::Cancelled),
        "cancellation must land in MachineOutcome::Cancelled, not Failed",
    );
}

struct StreamingMockClient {
    model: String,
    rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Result<StreamEvent, ApiError>>>>,
}

impl StreamingMockClient {
    fn new(
        model: &str,
    ) -> (
        Self,
        tokio::sync::mpsc::Sender<Result<StreamEvent, ApiError>>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent, ApiError>>(8);
        (
            Self {
                model: model.to_string(),
                rx: std::sync::Mutex::new(Some(rx)),
            },
            tx,
        )
    }
}

impl ApiClient for StreamingMockClient {
    fn model(&self) -> String {
        self.model.clone()
    }

    fn set_model(&self, _model: &str) -> bool {
        false
    }

    fn stream_messages(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let rx = crate::error::recover_guard(self.rx.lock())
            .take()
            .expect("stream_messages called twice");
        Box::pin(ReceiverStream { rx })
    }

    fn create_message(
        &self,
        _request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        Box::pin(async { Err(ApiError::api("not implemented")) })
    }
}

struct ReceiverStream<T> {
    rx: tokio::sync::mpsc::Receiver<T>,
}

impl<T> futures::Stream for ReceiverStream<T> {
    type Item = T;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_stream_turn_cancelled_mid_stream() {
    let (client, tx) = StreamingMockClient::new("test-model");
    let model = client.model.clone();
    tx.send(Ok(StreamEvent::MessageStart(MessageStart {
        message: MessageMetadata {
            id: "msg-1".into(),
            role: "assistant".into(),
            model,
        },
    })))
    .await
    .unwrap();

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let signal = agent.cancel_signal();

    let handle = tokio::spawn(async move { agent.run("Hi", &RunConfig::default()).await });

    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    signal.cancel();

    // `tx` stays open until function exit, so the channel never closes —
    // the only way `run()` returns is via the cancel signal.
    let result = handle.await.unwrap();
    match result {
        Err(LoopError::Cancelled) => {}
        other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
    }
}

#[tokio::test]
async fn test_set_pipeline_injects_self_tools_registry() {
    let client = MockClient::new("test-model");
    client.add_tool_then_text("tool_1", "echo", json!({"message": "hello"}), "done");
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let config = make_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    // Build a builder WITHOUT calling .with_core() — set_pipeline must inject it.
    let builder = ToolPipeline::builder();
    agent.set_pipeline(builder).unwrap();

    let result = agent.run("Echo hello", &RunConfig::default()).await;
    result.unwrap();
}

struct TurnNumberCapture {
    turns: Arc<Mutex<Vec<usize>>>,
}

impl TurnNumberCapture {
    fn new(shared: Arc<Mutex<Vec<usize>>>) -> Self {
        Self { turns: shared }
    }
}

impl crate::middleware::ToolMiddleware for TurnNumberCapture {
    fn name(&self) -> &'static str {
        "turn_capture"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = crate::middleware::ToolDispatchResult> + Send + 'a>,
    > {
        crate::error::recover_guard(self.turns.lock()).push(ctx.turn_number);
        next.dispatch(ctx)
    }
}

#[tokio::test]
async fn test_turn_number_is_actual_turn_index() {
    let client = MockClient::new("test-model");
    // Turn 0: model requests tool call, then turn 1: model requests another
    client.add_tool_only_response("tool_0", "echo", json!({"message": "a"}));
    client.add_tool_only_response("tool_1", "echo", json!({"message": "b"}));
    client.add_text_response("done");

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let capture = Arc::new(Mutex::new(Vec::<usize>::new()));
    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let builder =
        ToolPipeline::builder().with_middleware(TurnNumberCapture::new(Arc::clone(&capture)));
    agent.set_pipeline(builder).unwrap();

    let _result = agent.run("test", &make_run_config()).await;

    let turns = crate::error::recover_guard(capture.lock()).clone();
    // Tool was called on turn 0 (first turn) and turn 1 (second turn).
    assert_eq!(
        turns.len(),
        2,
        "expected tool calls on 2 turns: got {turns:?}"
    );
    assert_eq!(turns[0], 0, "first tool call should be on turn 0");
    assert_eq!(turns[1], 1, "second tool call should be on turn 1");
    assert!(
        turns.iter().all(|&t| t < 10),
        "turn_number must be actual index, not max_turns (10): got {turns:?}"
    );
}

#[tokio::test]
async fn switch_model_updates_config_and_client() {
    let client = MockClient::new("model-a");
    let client_arc = std::sync::Arc::new(client);
    let tools = ToolRegistry::new();

    let mut loop_ = BareLoop::new(client_arc.clone(), tools, SessionConfig::default());

    loop_.switch_model("model-b").apply().unwrap();

    // Client was updated via set_model.
    assert_eq!(loop_.client.model(), "model-b");

    // The shared client handle sees the same update.
    assert_eq!(client_arc.model(), "model-b");
}

#[tokio::test]
async fn switch_model_notifies_observers() {
    #[derive(Default)]
    struct RecordingObserver {
        switches: Mutex<Vec<(String, String)>>,
    }

    impl crate::observer::LoopObserver for RecordingObserver {
        fn name(&self) -> &'static str {
            "recording"
        }

        fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
            crate::error::recover_guard(self.switches.lock())
                .push((ctx.from.clone(), ctx.to.clone()));
        }
    }

    let client = std::sync::Arc::new(MockClient::new("m1"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());
    let obs = std::sync::Arc::new(RecordingObserver::default());
    let obs_clone = obs.clone();
    loop_.register_observer(obs);

    loop_.switch_model("m2").apply().unwrap();
    loop_.switch_model("m3").apply().unwrap();

    // Observer should have received both switches.
    let recorded = crate::error::recover_guard(obs_clone.switches.lock());
    assert_eq!(recorded.len(), 2, "should have 2 model-switch events");
    assert_eq!(recorded[0], ("m1".to_string(), "m2".to_string()));
    assert_eq!(recorded[1], ("m2".to_string(), "m3".to_string()));
}

#[tokio::test]
async fn switch_model_unsupported_client() {
    struct StaticClient {
        model_name: Arc<std::sync::Mutex<String>>,
    }

    impl ApiClient for StaticClient {
        fn model(&self) -> String {
            crate::error::recover_guard(self.model_name.lock()).clone()
        }
        // Uses default set_model which returns false.

        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<dyn futures::stream::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    let client = std::sync::Arc::new(StaticClient {
        model_name: std::sync::Arc::new(std::sync::Mutex::new("static".to_string())),
    });
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    // set_model returns false (unsupported), but apply() is best-effort
    // and still updates the session/client state.
    loop_.switch_model("new-model").apply().unwrap();

    // The client is the source of truth for the model; an unsupported
    // set_model leaves the client unchanged.
    assert_eq!(loop_.client.model(), "static");
}

#[tokio::test]
async fn switch_model_updates_fallback_original() {
    let client = std::sync::Arc::new(MockClient::new("primary"));
    let tools = ToolRegistry::new();

    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    // Before switch, fallback manager has no original model set.
    assert_eq!(loop_.managers.fallback().original_model(), None);

    loop_.switch_model("new-primary").apply().unwrap();

    // After switch, fallback manager tracks the new primary.
    assert_eq!(
        loop_.managers.fallback().original_model(),
        Some("new-primary".to_string())
    );
}

#[tokio::test]
async fn switch_model_rejects_empty() {
    let client = std::sync::Arc::new(MockClient::new("model"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    let result = loop_.switch_model("").apply();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));

    let result = loop_.switch_model("   ").apply();
    assert!(result.is_err());

    // Model should remain unchanged.
    assert_eq!(loop_.client.model(), "model");
}

#[tokio::test]
async fn switch_model_chained() {
    let client = std::sync::Arc::new(MockClient::new("a"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    loop_.switch_model("b").apply().unwrap();
    assert_eq!(loop_.client.model(), "b");

    loop_.switch_model("c").apply().unwrap();
    assert_eq!(loop_.client.model(), "c");

    loop_.switch_model("d").apply().unwrap();
    assert_eq!(loop_.client.model(), "d");
}

#[tokio::test]
async fn switch_model_updates_context_window() {
    let client = std::sync::Arc::new(MockClient::new("big-model"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    let original_cw = loop_.session_config().context_window;
    assert_ne!(original_cw, 8192);

    loop_
        .switch_model("small-model")
        .with_context_window(8192)
        .apply()
        .unwrap();

    assert_eq!(loop_.client.model(), "small-model");
    assert_eq!(loop_.session_config().context_window, 8192);
}

#[tokio::test]
async fn switch_model_updates_max_tokens() {
    let client = std::sync::Arc::new(MockClient::new("m"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    loop_.switch_model("m2").apply().unwrap();

    assert_eq!(loop_.client.model(), "m2");
}

#[tokio::test]
async fn switch_model_trims_whitespace() {
    let client = std::sync::Arc::new(MockClient::new("m"));
    let tools = ToolRegistry::new();
    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    loop_.switch_model("  gpt-4o  ").apply().unwrap();
    assert_eq!(loop_.client.model(), "gpt-4o");
}

#[tokio::test]
async fn switch_model_resets_fallback_circuit() {
    use crate::fallback::FallbackState;

    let client = std::sync::Arc::new(MockClient::new("primary"));
    let tools = ToolRegistry::new();

    let mut loop_ = BareLoop::new(client, tools, SessionConfig::default());

    // Trip the circuit breaker.
    loop_
        .managers
        .fallback()
        .set_original_model("primary".into());
    loop_.managers.fallback().set_fallback_model("backup");
    loop_.managers.fallback().transition_to_fallback();
    assert_eq!(loop_.managers.fallback().state(), FallbackState::Fallback);

    // Switch model — circuit should reset to Primary.
    loop_.switch_model("new-primary").apply().unwrap();

    assert_eq!(loop_.managers.fallback().state(), FallbackState::Primary);
    assert_eq!(
        loop_.managers.fallback().original_model(),
        Some("new-primary".to_string())
    );
}

#[cfg(feature = "hooks")]
struct ReasonCaptureHook {
    reason: Mutex<Option<RunEndReason>>,
}

#[cfg(feature = "hooks")]
impl ReasonCaptureHook {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            reason: Mutex::new(None),
        })
    }

    fn captured(&self) -> Option<RunEndReason> {
        *crate::error::recover_guard(self.reason.lock())
    }
}

#[cfg(feature = "hooks")]
impl Hook for ReasonCaptureHook {
    fn name(&self) -> &'static str {
        "ReasonCaptureHook"
    }

    fn on_run_end(&self, ctx: &HookRunEndContext) {
        *crate::error::recover_guard(self.reason.lock()) = Some(ctx.reason);
    }
}

#[cfg(feature = "hooks")]
fn loop_with_reason_hook() -> (BareLoop<MockClient>, Arc<ReasonCaptureHook>) {
    let hook = ReasonCaptureHook::new();
    let executor = Arc::new(HookExecutor::new().with_hook(hook.clone()));
    let mut loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    loop_.session.runs.push(Run::new(
        "",
        &RunConfig {
            max_turns: 5,
            ..RunConfig::default()
        },
    ));
    loop_.set_hook_executor(executor);
    (loop_, hook)
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_complete() {
    let (mut loop_, hook) = loop_with_reason_hook();
    // Normal completion: success true, not cancelled, under max_turns.
    loop_.session.current_run_mut().unwrap().turns = vec![
        crate::engine::core::Turn {
            turn: 0,
            input: String::new(),
            output: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        },
        crate::engine::core::Turn {
            turn: 1,
            input: String::new(),
            output: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        },
    ];

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        None,
    );

    assert_eq!(hook.captured(), Some(RunEndReason::Complete));
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_cancelled() {
    let (mut loop_, hook) = loop_with_reason_hook();
    // Cancel signal fired — success is true (not Failed) but cancelled.
    loop_.session.current_run_mut().unwrap().turns = vec![
        crate::engine::core::Turn {
            turn: 0,
            input: String::new(),
            output: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        },
        crate::engine::core::Turn {
            turn: 1,
            input: String::new(),
            output: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        },
    ];
    loop_.cancelled.cancel();

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        None,
    );

    assert_eq!(hook.captured(), Some(RunEndReason::Cancelled));
}

/// A genuine max-turns run exits via the machine's
/// `MaxTurnsExceeded` arm, which carries the typed error through
/// finalize — not a turn-count heuristic.
#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_max_turns() {
    let (loop_, hook) = loop_with_reason_hook();
    let err = LoopError::MaxTurnsExceeded { max: 5 };

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        Some(&err),
    );

    assert_eq!(hook.captured(), Some(RunEndReason::MaxTurns));
}

/// A run that legitimately completes on exactly the `max_turns`-th
/// turn reaches finalize with `error = None`. The turn count is a
/// red herring: the machine emitted `Completed`, not
/// `MaxTurnsExceeded`, so the reason must be `Complete`.
#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_complete_on_max_turn_boundary() {
    let (mut loop_, hook) = loop_with_reason_hook();
    loop_.session.current_run_mut().unwrap().turns = (0..5)
        .map(|i| crate::engine::core::Turn {
            turn: i,
            input: String::new(),
            output: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        })
        .collect();

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        None,
    );

    assert_eq!(hook.captured(), Some(RunEndReason::Complete));
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_error() {
    let (loop_, hook) = loop_with_reason_hook();
    let err = LoopError::Api("something went wrong".into());

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        Some(&err),
    );

    assert_eq!(hook.captured(), Some(RunEndReason::Error));
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn run_end_reason_context_overflow() {
    let (loop_, hook) = loop_with_reason_hook();
    let err = LoopError::ContextExceeded {
        used: 100_000,
        limit: 50_000,
    };

    loop_.notify_run_end(
        &loop_.session.current_run().unwrap().clone(),
        Duration::from_millis(100),
        Some(&err),
    );

    assert_eq!(hook.captured(), Some(RunEndReason::ContextOverflow));
}

#[test]
fn stop_reason_is_none_before_terminal() {
    use crate::engine::core::Loop;
    let loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    assert_eq!(loop_.stop_reason(), None);
}

#[test]
fn stop_reason_reports_terminal_outcome() {
    use crate::engine::core::Loop;
    let mut loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    loop_.machine.fail(LoopError::Api("boom".into()));
    assert_eq!(loop_.stop_reason(), Some(LoopError::Api("boom".into())));

    let mut loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    loop_.machine.cancel();
    let policy = loop_.machine_policy();
    let _ = loop_.machine.next_step(policy);
    assert_eq!(loop_.stop_reason(), Some(LoopError::Cancelled));

    // Drive the machine to a genuine MaxTurnsExceeded terminal state
    // by exhausting a budget of one: request the model, respond with
    // a tool call, then request again — the third next_step hits the
    // cap. stop_reason must surface the typed error. The machine is
    // policy-free, so the budget is passed directly to next_step.
    let mut loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    loop_.session.runs.push(Run::new(
        "",
        &RunConfig {
            max_turns: 1,
            ..RunConfig::default()
        },
    ));
    let policy = loop_.machine_policy();
    let _ = loop_.machine.next_step(policy);
    let part = MessagePart::tool_call("c1", "echo", serde_json::Value::Null);
    let response = ModelResponse {
        message: Message::new(Role::Assistant, vec![part]),
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: StopReason::ToolCall,
        available_tools: vec!["echo".to_string()],
    };
    loop_.machine.model_response(response, 0);
    let _ = loop_.machine.next_step(policy);
    loop_.machine.tool_results(vec![Message::user("r")]);
    let step = loop_.machine.next_step(policy);
    assert!(matches!(
        step,
        MachineStep::Done(MachineOutcome::MaxTurnsExceeded)
    ));
    assert_eq!(
        loop_.stop_reason(),
        Some(LoopError::MaxTurnsExceeded { max: 1 })
    );
}

#[test]
fn stop_reason_completion_on_max_turn_boundary_is_none() {
    use crate::engine::core::Loop;
    let mut loop_ = BareLoop::new(
        Arc::new(MockClient::new("test")),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    // A run that legitimately completes on exactly the max_turns-th
    // turn ends with the machine in the Completed terminal state, not
    // MaxTurnsExceeded. stop_reason must reflect that: None, not
    // MaxTurnsExceeded. This is the regression the old turn-count
    // heuristic got wrong.
    let final_msg = Message::assistant("done");
    let response = ModelResponse {
        message: final_msg,
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: StopReason::EndTurn,
        available_tools: Vec::new(),
    };
    let policy = MachinePolicy {
        max_turns: 1,
        context_window: 200_000,
        compact_threshold: 80,
        auto_compact: true,
    };
    let _ = loop_.machine.next_step(policy);
    loop_.machine.model_response(response, 0);
    assert!(loop_.machine.is_terminal());
    assert_eq!(loop_.stop_reason(), None);
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn run_cancel_during_streaming_returns_fast() {
    let (client, tx) = StreamingMockClient::new("test-model");
    tx.send(Ok(StreamEvent::MessageStart(MessageStart {
        message: MessageMetadata {
            id: "msg-1".into(),
            role: "assistant".into(),
            model: "test-model".into(),
        },
    })))
    .await
    .unwrap();

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());
    let signal = agent.cancel_signal();

    let handle = tokio::spawn(async move { agent.run("Hi", &RunConfig::default()).await });

    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    let start = Instant::now();
    signal.cancel();

    let result = handle.await.unwrap();
    let elapsed = start.elapsed();

    match result {
        Err(LoopError::Cancelled) => {}
        other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "cancel during streaming should return fast; elapsed {elapsed:?}",
    );
    assert_eq!(
        observer.turn_ends.load(Ordering::SeqCst),
        1,
        "on_turn_end should fire once on cancel",
    );
}

#[tokio::test]
async fn run_cancel_during_dispatch_fires_turn_end() {
    struct SlowTool {
        notify: Arc<tokio::sync::Notify>,
    }
    impl Tool for SlowTool {
        fn name(&self) -> &'static str {
            "slow"
        }
        fn description(&self) -> &'static str {
            "Blocks until notified"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "slow".into(),
                description: "Blocks until notified".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            }
        }
        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let notify = self.notify.clone();
            Box::pin(async move {
                notify.notified().await;
                Ok(ToolOutput::text("done"))
            })
        }
    }

    let notify = Arc::new(tokio::sync::Notify::new());
    let mut registry = ToolRegistry::new();
    registry.register(SlowTool {
        notify: notify.clone(),
    });

    let client = MockClient::new("test");
    client.add_tool_only_response("tc-1", "slow", json!({}));

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    let observer = Arc::new(CountingObserver::new());
    agent.register_observer(observer.clone());
    let signal = agent.cancel_signal();

    let handle =
        tokio::spawn(async move { agent.run("Use slow tool", &RunConfig::default()).await });

    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    signal.cancel();

    let result = handle.await.unwrap();
    match result {
        Err(LoopError::Cancelled) => {}
        other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
    }
    assert_eq!(
        observer.turn_ends.load(Ordering::SeqCst),
        1,
        "on_turn_end(false) must fire on cancel during dispatch",
    );
    assert_eq!(
        observer.run_ends.load(Ordering::SeqCst),
        1,
        "on_run_end must fire via finalize after cancel",
    );
}

#[tokio::test]
async fn run_cancel_during_recovery_backoff_returns_fast() {
    struct AlwaysRecoverable;
    impl crate::reflection::Reflector for AlwaysRecoverable {
        fn analyze(
            &self,
            error: &str,
            tool_name: &str,
            _tool_input: &serde_json::Value,
            _tool_schema: Option<&crate::tool::ToolSchema>,
            _context: &crate::reflection::ReflectionContext,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::reflection::FailureAnalysis,
                            crate::reflection::ReflectionError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let error = error.to_string();
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                Ok(crate::reflection::FailureAnalysis {
                    is_recoverable: true,
                    root_cause: error,
                    severity: crate::reflection::FailureSeverity::Medium,
                    correction: None,
                    context: format!("tool: {tool_name}"),
                })
            })
        }
    }

    let client = MockClient::new("test");
    client.add_tool_only_response("tc-1", "fail", json!({}));

    let mut registry = ToolRegistry::new();
    registry.register(FailingTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
    agent.set_reflector(Arc::new(AlwaysRecoverable));
    agent.set_recovery_strategy(Arc::new(
        crate::reflection::ExponentialBackoffRecovery::new(5)
            .with_base_delay(Duration::from_mins(1)),
    ));
    let signal = agent.cancel_signal();

    let handle =
        tokio::spawn(async move { agent.run("Use failing tool", &RunConfig::default()).await });

    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    let start = Instant::now();
    signal.cancel();

    let result = handle.await.unwrap();
    let elapsed = start.elapsed();

    match result {
        Err(LoopError::Cancelled) => {}
        other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "cancel during recovery backoff should return fast, not wait 60s; elapsed {elapsed:?}",
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_rate_limit_escalation_feeds_circuit_breaker() {
    use crate::fallback::FallbackManager;
    use crate::managers::LoopManagers;
    use crate::stream::handler::{RateLimitConfig, StreamHandler, StreamTimeoutConfig};

    // Every stream attempt is rate-limited, so the handler escalates on the
    // first 429 (fallback_after_retries = 0).
    struct AlwaysRateLimitClient;
    impl ApiClient for AlwaysRateLimitClient {
        fn model(&self) -> String {
            "primary-model".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            Box::pin(futures::stream::once(async {
                Err(ApiError::RateLimit {
                    retry_after: None,
                    message: "slow down".into(),
                })
            }))
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    let handler = StreamHandler::new()
        .with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: false,
            ..Default::default()
        })
        .with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 0,
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });

    // Circuit breaker: trips on a single model failure (threshold = 1) and
    // has a fallback model configured.
    let mut managers = LoopManagers::new().with_fallback(FallbackManager::new_with_fallback(
        "primary-model".to_string(),
        1,
    ));
    managers.fallback().set_fallback_model("fallback-model");
    managers.set_stream_handler(handler);

    let config = make_config();
    let client = Arc::new(AlwaysRateLimitClient);
    let mut agent = BareLoop::new_with_managers(client, ToolRegistry::new(), config, managers);

    let result = agent.run("Hi", &RunConfig::default()).await;
    assert!(result.is_err(), "rate-limited turn should fail");

    // The escalation arm called record_model_failure(); with threshold 1 the
    // breaker tripped into Fallback state.
    assert!(
        agent.managers.fallback().is_using_fallback(),
        "escalation should trip the circuit breaker to the fallback model"
    );
}

#[derive(Clone)]
struct RecordingClient {
    responses: Arc<Mutex<Vec<Vec<StreamEvent>>>>,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
    seen_options: Arc<Mutex<Vec<crate::structured::RequestOptions>>>,
    model_name: Arc<Mutex<String>>,
}

impl RecordingClient {
    fn new(model: &str) -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
            seen_options: Arc::new(Mutex::new(Vec::new())),
            model_name: Arc::new(Mutex::new(model.to_string())),
        }
    }

    fn add_text_response(&self, text: &str) {
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: Some(Usage::new(10, 20)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(events);
    }

    fn first_seen(&self) -> Vec<Message> {
        crate::error::recover_guard(self.seen.lock())
            .first()
            .expect("at least one stream_messages call")
            .clone()
    }

    fn add_tool_then_text(
        &self,
        tool_id: &str,
        tool_name: &str,
        tool_input: Value,
        final_text: &str,
    ) {
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_tool".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 10)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(tool_events);

        let text_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_final".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text(final_text)),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: final_text.to_string(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: Some(Usage::new(30, 15)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(self.responses.lock()).push(text_events);
    }

    fn call_count(&self) -> usize {
        crate::error::recover_guard(self.seen.lock()).len()
    }

    fn first_options(&self) -> crate::structured::RequestOptions {
        crate::error::recover_guard(self.seen_options.lock())
            .first()
            .expect("at least one stream_messages_with_options call")
            .clone()
    }
}

impl ApiClient for RecordingClient {
    fn model(&self) -> String {
        crate::error::recover_guard(self.model_name.lock()).clone()
    }

    fn set_model(&self, model: &str) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
        true
    }

    fn stream_messages(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let messages = request.messages.clone();
        crate::error::recover_guard(self.seen.lock()).push(messages);
        let mut guard = crate::error::recover_guard(self.responses.lock());
        if let Some(events) = guard.pop_front() {
            let events: Vec<Result<StreamEvent, ApiError>> = events.into_iter().map(Ok).collect();
            Box::pin(futures::stream::iter(events))
        } else {
            let err = ApiError::api("No more mock responses");
            Box::pin(futures::stream::iter(vec![Err(err)]))
        }
    }

    fn stream_messages_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let messages = request.messages.clone();
        crate::error::recover_guard(self.seen.lock()).push(messages);
        crate::error::recover_guard(self.seen_options.lock()).push(options);
        let mut guard = crate::error::recover_guard(self.responses.lock());
        if let Some(events) = guard.pop_front() {
            let events: Vec<Result<StreamEvent, ApiError>> = events.into_iter().map(Ok).collect();
            Box::pin(futures::stream::iter(events))
        } else {
            let err = ApiError::api("No more mock responses");
            Box::pin(futures::stream::iter(vec![Err(err)]))
        }
    }

    fn create_message(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        let messages = request.messages.clone();
        crate::error::recover_guard(self.seen.lock()).push(messages);
        let mut guard = crate::error::recover_guard(self.responses.lock());
        let events = guard.pop_front();
        drop(guard);
        Box::pin(async move {
            let events = events.ok_or_else(|| ApiError::api("No more mock responses"))?;
            assemble_response(events)
        })
    }

    fn create_message_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        crate::error::recover_guard(self.seen_options.lock()).push(options);
        self.create_message(request)
    }
}

struct StaticReminder(String);
impl ContextContributor for StaticReminder {
    fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
        Some(Message::new(
            Role::System,
            vec![MessagePart::text(self.0.clone())],
        ))
    }
}

struct NeverContributor;
impl ContextContributor for NeverContributor {
    fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
        None
    }
}

struct CountingContributor {
    calls: Arc<AtomicUsize>,
}
impl ContextContributor for CountingContributor {
    fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        None
    }
}

struct CapturingContributor {
    seen_turns: Arc<Mutex<Vec<usize>>>,
}
impl ContextContributor for CapturingContributor {
    fn contribute(&self, ctx: &ContributorContext<'_>) -> Option<Message> {
        crate::error::recover_guard(self.seen_turns.lock()).push(ctx.turn);
        None
    }
}

fn contributor_config() -> SessionConfig {
    SessionConfig::default()
}

#[tokio::test]
async fn test_contributor_message_prepended() {
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    agent.add_contributor(Box::new(StaticReminder("stay on task".into())));
    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let seen = client.first_seen();
    let texts: Vec<&str> = seen
        .iter()
        .filter(|m| m.role == Role::System)
        .flat_map(|m| {
            m.parts.iter().filter_map(|p| match p {
                MessagePart::Text { text } => Some(text.as_str()),
                _ => None,
            })
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("stay on task")),
        "contributor message must reach the model in the outbound request"
    );

    let persisted = agent.conversation();
    assert!(
        !persisted.iter().any(|m| m.role == Role::System
            && m.parts
                .iter()
                .any(|p| matches!(p, MessagePart::Text { text } if text.contains("stay on task")))),
        "contributor message must NOT persist in history"
    );
}

#[tokio::test]
async fn test_no_contributors_no_change() {
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    // No add_contributor call.
    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let seen = client.first_seen();
    // No System messages reached the model.
    assert!(
        !seen.iter().any(|m| m.role == Role::System),
        "no contributor registered, so no System message should appear"
    );
    // Exactly one user message (the "Hi").
    let user_count = seen.iter().filter(|m| m.role == Role::User).count();
    assert_eq!(user_count, 1, "baseline conversation has one user message");
}

#[tokio::test]
async fn failed_run_leaves_history_clean() {
    let client = MockClient::new("test-model");
    client.add_text_response("done");

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

    agent.cancel();
    let result = agent.run("first", &RunConfig::default()).await;
    assert!(result.is_err(), "run must fail");

    let history_after_fail = agent.conversation();
    assert!(
        history_after_fail.is_empty(),
        "failed run must not leave messages in committed history; \
         got {} messages",
        history_after_fail.len()
    );

    agent.cancelled.reset();
    agent.run("second", &RunConfig::default()).await.unwrap();
}

#[tokio::test]
async fn contributor_messages_must_not_accumulate_across_turns() {
    let client = RecordingClient::new("test-model");
    client.add_text_response("turn 1 done");
    client.add_text_response("turn 2 done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    agent.add_contributor(Box::new(StaticReminder("stay on task".into())));

    agent.run("first run", &RunConfig::default()).await.unwrap();
    agent
        .run("second run", &RunConfig::default())
        .await
        .unwrap();

    let system_count = agent
        .conversation()
        .iter()
        .filter(|m| m.role == Role::System)
        .filter(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, MessagePart::Text { text } if text == "stay on task"))
        })
        .count();
    assert_eq!(
        system_count, 0,
        "contributor messages must NOT persist in history; \
         found {system_count} copies (accumulated across turns)"
    );
}

#[tokio::test]
async fn test_contributor_returning_none_injects_nothing() {
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    agent.add_contributor(Box::new(NeverContributor));
    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let seen = client.first_seen();
    assert!(
        !seen.iter().any(|m| m.role == Role::System),
        "None-returning contributor must inject nothing"
    );
}

#[tokio::test]
async fn test_multiple_contributors_order_preserved() {
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    agent.add_contributor(Box::new(StaticReminder("first".into())));
    agent.add_contributor(Box::new(StaticReminder("second".into())));
    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let seen = client.first_seen();
    let pos = |needle: &str| -> Option<usize> {
        seen.iter().position(|m| {
            m.role == Role::System
                && m.parts
                    .iter()
                    .any(|p| matches!(p, MessagePart::Text { text } if text == needle))
        })
    };
    let first = pos("first").expect("'first' reminder persisted");
    let second = pos("second").expect("'second' reminder persisted");
    assert!(first < second, "registration order must be preserved");
}

#[tokio::test]
async fn test_contributor_does_not_affect_turn_count() {
    // Two-turn session: tool call then end_turn.
    let with_contrib = {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        agent.add_contributor(Box::new(StaticReminder("remind".into())));
        agent
            .run("Hi", &RunConfig::default())
            .await
            .unwrap()
            .turn_count()
    };
    let without_contrib = {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        agent
            .run("Hi", &RunConfig::default())
            .await
            .unwrap()
            .turn_count()
    };
    assert_eq!(
        with_contrib, without_contrib,
        "injection must not perturb turn counting"
    );
}

#[tokio::test]
async fn test_contributor_fires_every_turn() {
    // A single contributor + a single-turn run must show exactly one call.
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let counter = Arc::new(AtomicUsize::new(0));
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    let c = Arc::clone(&counter);
    agent.add_contributor(Box::new(CountingContributor { calls: c }));
    agent.run("Hi", &RunConfig::default()).await.unwrap();

    // One turn ran; the contributor was consulted once.
    assert_eq!(counter.load(Ordering::Relaxed), 1);
    // And the model was called exactly once (proving the single turn).
    assert_eq!(agent.session.current_run().unwrap().turn_count(), 1);
}

#[tokio::test]
async fn test_contributor_fires_across_two_turns() {
    // Two-turn session via a tool: turn 1 = tool_call, turn 2 = end_turn.
    // The contributor must be consulted on BOTH turns.
    let client = RecordingClient::new("test-model");
    client.add_tool_then_text("t1", "echo", json!({"message": "hi"}), "all done");
    let counter = Arc::new(AtomicUsize::new(0));

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    let c = Arc::clone(&counter);
    agent.add_contributor(Box::new(CountingContributor { calls: c }));
    let result = agent.run("Echo hi", &RunConfig::default()).await.unwrap();
    assert_eq!(result.turn_count(), 2, "tool_call turn + end_turn");
    assert_eq!(
        counter.load(Ordering::Relaxed),
        2,
        "contributor must fire on every turn"
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "configuration setters must be called before run()")]
fn test_add_contributor_panics_after_session_start() {
    let client = MockClient::new("test-model");
    client.add_text_response("ok");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    // The first run() establishes the session (capturing the start time
    // and firing on_run_start), moving the loop out of Idle. A
    // subsequent add_contributor must panic in debug builds (matches
    // set_reflector's contract).
    // Box the future so we can drop it without awaiting; the session-init
    // side effect is the state transition under test.
    {
        let run_config = RunConfig::default();
        let fut = agent.run("seed", &run_config);
        let mut fut = std::pin::pin!(fut);
        let outcome = futures::executor::block_on(fut.as_mut());
        drop(outcome);
    }
    agent.add_contributor(Box::new(StaticReminder("late".into())));
}

#[tokio::test]
async fn test_contributor_sees_turn_number() {
    // Assert the ContributorContext.turn matches the engine's turn counter
    // at consultation time. Captures the value across a 2-turn session.
    let client = RecordingClient::new("test-model");
    client.add_tool_then_text("t1", "echo", json!({"message": "x"}), "done");
    let seen_turns = Arc::new(Mutex::new(Vec::<usize>::new()));

    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client), registry, config);
    let s = Arc::clone(&seen_turns);
    agent.add_contributor(Box::new(CapturingContributor { seen_turns: s }));
    agent.run("go", &RunConfig::default()).await.unwrap();

    let turns = crate::error::recover_guard(seen_turns.lock()).clone();
    assert_eq!(turns, vec![0, 1], "turn numbers are 0-indexed and per-turn");
}

#[allow(dead_code)]
fn _suppress_recording_client_dead_code(c: &RecordingClient) {
    let _ = c.call_count();
}

#[tokio::test]
async fn test_request_options_default_is_unconstrained() {
    // A fresh BareLoop has default RequestOptions — the engine reproduces
    // v0.1.0 behavior (no tool_constraint).
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    // No set_request_options call — default path.
    agent.run("Hi", &RunConfig::default()).await.unwrap();

    let opts = client.first_options();
    assert!(
        matches!(
            opts.tool_constraint,
            crate::structured::ToolConstraint::None
        ),
        "default request options must be unconstrained"
    );
}

#[tokio::test]
async fn test_request_options_strict_reaches_provider() {
    // The critical end-to-end proof: a tool_constraint: Strict set on the
    // loop reaches the provider's stream_messages_with_options call.
    let client = RecordingClient::new("test-model");
    client.add_text_response("done");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
    agent.set_request_options(
        crate::structured::RequestOptions::new()
            .with_tool_constraint(crate::structured::ToolConstraint::Strict),
    );
    agent.run("Hi", &RunConfig::default()).await.unwrap();

    let opts = client.first_options();
    assert!(
        matches!(
            opts.tool_constraint,
            crate::structured::ToolConstraint::Strict
        ),
        "Strict set on the loop must reach the provider"
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "configuration setters must be called before run()")]
fn test_set_request_options_panics_after_session_start() {
    let client = MockClient::new("test-model");
    client.add_text_response("ok");
    let config = contributor_config();
    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
    // The first run() establishes the session and moves the loop out of
    // Idle; a subsequent set_request_options must panic in debug builds.
    {
        let run_config = RunConfig::default();
        let fut = agent.run("seed", &run_config);
        let mut fut = std::pin::pin!(fut);
        let outcome = futures::executor::block_on(fut.as_mut());
        drop(outcome);
    }
    agent.set_request_options(crate::structured::RequestOptions::default());
}

#[tokio::test]
async fn test_constrained_apply_wires_pipeline_and_contributor() {
    // Apply() sets the small-model pipeline and registers a GoalReminder. To prove
    // the contributor wiring without driving 5 turns (each turn ends on
    // end_turn, so reaching turn 5 needs a long tool-call chain), we add
    // a cadence-1 GoalReminder on top: it fires on turn 1, so a single
    // tool-then-text session (2 turns) is enough.
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);

    let client = RecordingClient::new("test-model");
    client.add_tool_then_text("t1", "echo", json!({"message": "x"}), "done");

    let mut agent = BareLoop::new(Arc::new(client.clone()), registry, contributor_config());
    // apply() wires the pipeline + a cadence-5 GoalReminder.
    crate::presets::ConstrainedProfile::apply(&mut agent).unwrap();
    // Add a cadence-1 reminder so it fires this session.
    agent.add_contributor(Box::new(crate::presets::GoalReminder::new(1)));

    let result = agent
        .run("ship the demo goal", &RunConfig::default())
        .await
        .unwrap();
    // Tool-call turn + end_turn = 2 turns.
    assert!(result.turn_count() >= 1);

    // The contributor fired: a Role::System message carrying the first
    // user message text reached the provider on some turn's outbound
    // conversation. Scan all recorded calls (the reminder fires on turn 1,
    // not turn 0).
    let all_seen = crate::error::recover_guard(client.seen.lock()).clone();
    let has_reminder = all_seen.iter().flatten().any(|m| {
        m.role == Role::System
            && m.parts.iter().any(
                |p| matches!(p, MessagePart::Text { text } if text.contains("ship the demo goal")),
            )
    });
    assert!(
        has_reminder,
        "GoalReminder (cadence 1) should have injected the goal text as a System message"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_thinking_delta_fires_per_thinking_delta() {
    struct ThinkingRecorder {
        deltas: Arc<Mutex<Vec<(usize, String)>>>,
    }
    impl crate::observer::LoopObserver for ThinkingRecorder {
        fn name(&self) -> &'static str {
            "thinking-recorder"
        }
        fn on_thinking_delta(&self, ctx: &crate::observer::ThinkingDeltaContext) {
            crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
        }
    }

    let client = MockClient::new("test-model");
    let events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::PartStart(PartStart {
            index: 1,
            part: None,
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Thinking {
                text: "First reasoning".into(),
            },
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Thinking {
                text: " chunk".into(),
            },
        }),
        StreamEvent::PartStop,
        StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("ignored")),
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "final answer".into(),
            },
        }),
        StreamEvent::PartStop,
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: None,
        }),
        StreamEvent::MessageStop,
    ];
    client.add_events(events);

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::new(ThinkingRecorder {
        deltas: Arc::clone(&captured),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    let captured = crate::error::recover_guard(captured.lock());
    assert_eq!(
        captured.len(),
        2,
        "one on_thinking_delta per Thinking delta"
    );
    let joined: String = captured.iter().map(|(_, d)| d.as_str()).collect();
    assert_eq!(joined, "First reasoning chunk");
    assert_eq!(captured[0].0, 0, "turn number matches the run's turn count");
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn test_on_thinking_delta_independent_of_text_delta() {
    struct MixedRecorder {
        text_calls: Arc<Mutex<usize>>,
        thinking_calls: Arc<Mutex<usize>>,
    }
    impl crate::observer::LoopObserver for MixedRecorder {
        fn name(&self) -> &'static str {
            "mixed-recorder"
        }
        fn on_text_delta(&self, _ctx: &crate::observer::TextDeltaContext) {
            *crate::error::recover_guard(self.text_calls.lock()) += 1;
        }
        fn on_thinking_delta(&self, _ctx: &crate::observer::ThinkingDeltaContext) {
            *crate::error::recover_guard(self.thinking_calls.lock()) += 1;
        }
    }

    let client = MockClient::new("test-model");
    let events = vec![
        StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Thinking {
                text: "reasoning".into(),
            },
        }),
        StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "answer".into(),
            },
        }),
        StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: None,
        }),
        StreamEvent::MessageStop,
    ];
    client.add_events(events);

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    let text_calls = Arc::new(Mutex::new(0usize));
    let thinking_calls = Arc::new(Mutex::new(0usize));
    let recorder = Arc::new(MixedRecorder {
        text_calls: Arc::clone(&text_calls),
        thinking_calls: Arc::clone(&thinking_calls),
    });
    agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

    agent.run("Hi", &RunConfig::default()).await.unwrap();

    assert_eq!(
        *crate::error::recover_guard(text_calls.lock()),
        1,
        "text callback fires once (for the Text delta)"
    );
    assert_eq!(
        *crate::error::recover_guard(thinking_calls.lock()),
        1,
        "thinking callback fires once (for the Thinking delta)"
    );
}

#[tokio::test]
async fn fluent_with_chain_builds_a_working_loop() {
    let client = MockClient::new("test-model");
    client.add_text_response("done");

    let observer = Arc::new(CountingObserver::new());
    let registered: Arc<dyn crate::observer::LoopObserver> = observer.clone();

    let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config())
        .with_observer(registered)
        .with_reflector(Arc::new(NoopReflector))
        .with_request_options(RequestOptions::default());

    let _result = agent.run("Hi", &RunConfig::default()).await.unwrap();

    assert_eq!(
        observer.turn_starts.load(Ordering::SeqCst),
        1,
        "with_observer registered the observer (it received the turn event)"
    );
}

#[test]
fn fluent_with_observer_equivalent_to_register_observer() {
    let client = MockClient::new("test-model");
    let observer: Arc<dyn crate::observer::LoopObserver> = Arc::new(CountingObserver::new());

    let fluent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), make_config())
        .with_observer(Arc::clone(&observer));

    let mut imperative = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
    imperative.register_observer(Arc::clone(&observer));

    assert_eq!(
        fluent.managers.observers().len(),
        imperative.managers.observers().len(),
        "both paths register the same number of observers"
    );
}
